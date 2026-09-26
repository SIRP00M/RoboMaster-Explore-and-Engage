#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP - DFS Maze Explorer
---------------------------------
Sensors:
  Sharp LEFT  : Sensor Adapter ID 2, Port 1
  Sharp RIGHT : Sensor Adapter ID 3, Port 1
  ToF         : CAN bus distance_info[0], mounted on gimbal

Behavior:
  - DFS on a discrete cell graph.
  - Gimbal ToF scans LEFT / FRONT / RIGHT at each new cell.
  - Gimbal returns to FRONT and pitches DOWN 5 deg before chassis motion.
  - Corridor motion uses single-authority Sharp centering.
  - Both Sharp sensors may monitor, but only LEFT or RIGHT owns y-control.
  - Authority transfer uses hysteresis/hold time so controllers never fight.
  - BOTH IR LOW stops the robot and lets the Gimbal ToF scan for a route.
  - Front ToF is always a collision stop while moving.
  - Chassis yaw hold reduces gradual Z-axis drift.
  - Persistent JSON + ASCII map is autosaved while exploring.
  - Saved maps can be replayed later without re-discovering topology.
  - Known-map mode can BFS to a requested goal cell.

IMPORTANT:
  Tune CELL_LENGTH_M and TOF_OPEN_THRESHOLD_MM for the real maze geometry.
  Known-map mode assumes the same physical start/root and initial orientation.
"""

from robomaster import robot
import argparse
import json
import math
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ============================================================
# CONNECTION / SENSOR WIRING
# ============================================================

CONN_TYPE = "ap"

SENSOR_PORT = 1

# Digital IR obstacle sensors (ACTIVE LOW)
#   0 = WALL / obstacle detected
#   1 = clear
IR_LEFT_ID = 1
IR_RIGHT_ID = 4

# Analog Sharp GP2Y0A41SK0F
SHARP_LEFT_ID = 2
SHARP_RIGHT_ID = 3

TOF_INDEX = 0
TOF_FREQ_HZ = 20

POSITION_FREQ_HZ = 20
ATTITUDE_FREQ_HZ = 50

# Extra chassis feedback:
# ESC gives wheel motor speed + motor angle (raw wheel encoder-level telemetry).
# STATUS gives the chassis slip flag and impact/status information.
ESC_FREQ_HZ = 10
STATUS_FREQ_HZ = 5
GIMBAL_ANGLE_FREQ_HZ = 20


# ============================================================
# SHARP GP2Y0A41SK0F CALIBRATION
# median ADC from user's 100-sample calibration
#
# Only the monotonic/reliable section is used for control.
# Beyond ~24 cm these sensors become noisy/non-monotonic in the
# measured dataset, so values below the last ADC are treated as
# "wall too far / unavailable for wall-follow".
# ============================================================

LEFT_CAL = [
    (4.0, 864.5),
    (6.0, 610.0),
    (8.0, 475.0),
    (10.0, 378.0),
    (12.0, 316.0),
    (14.0, 274.0),
    (16.0, 247.0),
    (18.0, 224.0),
    (20.0, 212.0),
    (22.0, 198.0),
    (24.0, 182.0),
]

RIGHT_CAL = [
    (4.0, 822.0),
    (6.0, 589.0),
    (8.0, 455.0),
    (10.0, 374.0),
    (12.0, 313.0),
    (14.0, 276.0),
    (16.0, 239.0),
    (18.0, 205.0),
    (20.0, 174.0),
    (22.0, 161.0),
    (24.0, 133.0),
]

SHARP_FILTER_SAMPLES = 5
SHARP_MIN_PLAUSIBLE_ADC = 20


# ============================================================
# IR HARD-SAFETY / CORNER-CLEARANCE RECOVERY
# ============================================================
#
# IR is treated as the last close-range safety layer.
# When one side goes LOW:
#   LEFT LOW  -> STOP -> slide RIGHT a little
#   RIGHT LOW -> STOP -> slide LEFT a little
#
# After each slide, the gimbal ToF checks the triggered side AND front again.
# This is especially useful after a 90/180 degree turn where the chassis may
# not have completely cleared the corner yet.
IR_FILTER_SAMPLES = 3

# ------------------------------------------------------------
# Sequential dual-side IR latch
# ------------------------------------------------------------
# LEFT and RIGHT do NOT have to become LOW at the exact same sample.
#
# Example:
#   t=0.00  LEFT becomes LOW
#   t=0.45  LEFT clears
#   t=0.70  RIGHT becomes LOW
#
# Because the opposite-side event occurred inside this window, the robot
# treats it as a dual-side/corner event:
#       STOP -> Gimbal LEFT/FRONT/RIGHT route scan
#
# This catches the common case where one side of the chassis reaches a corner
# slightly before the other side.
IR_DUAL_EVENT_WINDOW_SEC = 1.50

# Small recovery nudge.  0.025 m = 2.5 cm.
IR_RECOVERY_SLIDE_M = 0.025
IR_RECOVERY_SLIDE_SPEED_MPS = 0.08
IR_RECOVERY_SLIDE_TIMEOUT_SEC = 1.2
IR_RECOVERY_MAX_ATTEMPTS = 3
IR_RECOVERY_SETTLE_SEC = 0.10

# Destination-side Sharp interlock while sliding.
#
# Example:
#   LEFT IR LOW -> command is slide RIGHT
#   RIGHT Sharp is now the destination-side guard.
#
# If RIGHT Sharp reaches <= this distance, the slide is vetoed BEFORE the
# chassis is allowed to keep moving into the right wall.
IR_SLIDE_DEST_SHARP_STOP_CM = 10.0

# After sliding, ToF on the offending side should be at least this far away.
# 100 mm matches the user's requested ~10 cm close-wall safety idea.
IR_GIMBAL_SIDE_CLEAR_MM = 100.0


# ============================================================
# CORRIDOR CONTROL
# ============================================================

# Corridor center target.
# Calibration/logs show the corridor is roughly 25-27 cm wide, so keeping
# one wall at ~13 cm places the chassis close to the middle.
#
# IMPORTANT CONTROL-LAW RULE:
# Both Sharp sensors may be READ for supervision/arbitration, but ONLY ONE
# Sharp sensor owns lateral control authority at a time.
CENTER_TARGET_CM = 13.0
CENTER_DEADBAND_CM = 0.55
CENTER_KP = 0.022
MAX_CENTER_STRAFE_MPS = 0.085

# Authority arbitration.
# Once LEFT or RIGHT owns y-control, keep it for this long unless safety
# requires an immediate handover.
AUTHORITY_MIN_HOLD_SEC = 0.75
AUTHORITY_DANGER_CM = 10.0
AUTHORITY_HARD_CM = 7.0
AUTHORITY_FAR_RELEASE_CM = 21.0
AUTHORITY_SWITCH_MARGIN_CM = 1.0

# A hard-distance owner is allowed a slightly stronger centering correction,
# but the OTHER Sharp still does not output a competing y command.
AUTHORITY_HARD_STRAFE_MPS = 0.13

# BOTH-IR behavior.
# When both digital IR sensors are LOW, do NOT slide randomly.
# Stop and let the gimbal ToF scan LEFT / FRONT / RIGHT.
IR_BOTH_ROUTE_OPEN_MM = 400.0
IR_BOTH_FRONT_OVERRIDE_SEC = 1.00

# Forward motion.
FORWARD_SPEED_MPS = 0.16
SLOW_FORWARD_SPEED_MPS = 0.10

# Closed-loop approach to the requested cell length.
# Run at normal speed for most of the cell, then reduce x near the target
# instead of blasting at constant speed and relying only on the stop threshold.
CELL_APPROACH_SLOW_M = 0.10
CELL_APPROACH_MIN_MPS = 0.07

# If ToF is this close while driving, stop immediately.
FRONT_HARD_STOP_MM = 160

# Begin slowing when front obstacle gets closer than this.
FRONT_SLOW_MM = 330


# ============================================================
# CELL / DFS GEOMETRY
# ============================================================

# Center-to-center distance between maze nodes/cells.
# *** TUNE THIS FOR THE REAL MAZE ***
CELL_LENGTH_M = 0.55

# Reaching this fraction counts as arriving at the next cell if ToF
# sees the end wall slightly earlier than odometry expects.
CELL_SUCCESS_FRACTION = 0.82

# ToF at cell center:
# <= threshold -> WALL
# > threshold  -> OPEN
#
# *** TUNE THIS FOR THE REAL MAZE ***
TOF_OPEN_THRESHOLD_MM = 400

# Explicit dead-end safety override.
# After the gimbal has measured LEFT / FRONT / RIGHT at a cell:
# if ALL THREE are <= this distance, the cell is treated as a hard dead end.
# 100 mm = 10 cm.
#
# This is intentionally independent of TOF_OPEN_THRESHOLD_MM:
#   - OPEN threshold decides whether a route is navigable.
#   - DEAD_END threshold is an extra close-range "boxed in" override.
#
# If the real maze geometry keeps the ToF farther than 10 cm from a wall even
# at a dead end, increase this to e.g. 120, 150, or 200 mm.
DEAD_END_THRESHOLD_MM = 100.0

# Keep 7 samples for robust median filtering.
# At 20 Hz this is roughly 0.35 s of sensor evidence per measurement.
# Mechanical scan time is still kept low by using direct gimbal moveto()
# without repeated recenter operations.
TOF_SCAN_SAMPLES = 7
TOF_SCAN_INTERVAL_SEC = 0.050

# The gimbal action already waits for the commanded position.
# Only a short sensor/mechanical settle is needed afterwards.
GIMBAL_SETTLE_SEC = 0.040
GIMBAL_ACTION_TIMEOUT_SEC = 1.80
GIMBAL_ANGLE_TOL_DEG = 2.0

# Gimbal pitch:
# negative = down on RoboMaster convention.
GIMBAL_PITCH_DEG = -5.0

# Direct absolute gimbal moves are used now; no repeated recenter/45-degree
# stepping during every L/F/R measurement.
GIMBAL_PITCH_SPEED = 120
GIMBAL_YAW_SPEED = 180

# BOTH-IR supervisor optimization:
# If the gimbal is already forward and FRONT is clearly open, continue
# immediately.  Scan LEFT/RIGHT only when FRONT is blocked/uncertain.
IR_FAST_FRONT_FIRST = True

# Root is assumed to start at the maze entrance, with its back outside.
# Therefore root-back is not explored.
ROOT_BACK_IS_WALL = True


# ============================================================
# PERSISTENT MAP / KNOWN-MAP NAVIGATION
# ============================================================

MAP_SCHEMA = "robomaster_dfs_grid_map"
MAP_SCHEMA_VERSION = 1
MAP_DIR = Path("maps")
MAP_LATEST_JSON = MAP_DIR / "latest_map.json"
MAP_LATEST_ASCII = MAP_DIR / "latest_map.txt"

# Save latest_map.json/.txt after every meaningful topology change.
# This protects the learned map even if the run is interrupted later.
MAP_AUTOSAVE = True


# ============================================================
# YAW HOLD / DRIFT CORRECTION
# ============================================================

YAW_HOLD_ENABLED = True

# Moving yaw lock: keep chassis on the logical DFS heading.
YAW_HOLD_KP = 1.8
YAW_HOLD_MAX_DPS = 22.0
YAW_HOLD_DEADBAND_DEG = 0.35

# Stronger yaw lock while the robot is stationary and the gimbal is moving.
# This directly counters reaction torque from the gimbal so the chassis does
# not slowly walk to the left/right during LEFT/FRONT/RIGHT ToF scans.
STATIONARY_YAW_HOLD_KP = 2.8
STATIONARY_YAW_HOLD_MAX_DPS = 28.0
STATIONARY_YAW_HOLD_HZ = 30.0

# turn_closed_loop() already performs its own target settle; this residual
# settle only removes a small final error.  Keep it short for field speed.
STATIONARY_SETTLE_SEC = 0.12

# REAL ROBOT convention confirmed by the latest test:
#   positive chassis z / increasing attitude yaw = RIGHT
#   negative chassis z / decreasing attitude yaw = LEFT
#
# Therefore target-current yaw error can be used directly.
YAW_DRIVE_SIGN = 1.0


# ============================================================
# CLOSED-LOOP CHASSIS TURN
# ============================================================
#
# IMPORTANT:
# Do NOT use chassis.move(...).wait_for_completed() for DFS turns.
# On the real robot that action can remain waiting after a turn command.
# We rotate with drive_speed(z=...) and close the loop from chassis attitude.
#
# During the actual turn the robot is temporarily put in CHASSIS_LEAD mode:
# the gimbal follows the chassis instead of trying to hold an independent yaw.
# After the turn we switch back to FREE before using the gimbal ToF scanner.
TURN_KP = 1.10
TURN_MAX_DPS = 45.0
TURN_MIN_DPS = 8.0
TURN_TOLERANCE_DEG = 1.2
TURN_SETTLE_SEC = 0.18
TURN_CONTROL_HZ = 30.0
TURN_TIMEOUT_90_SEC = 5.0
TURN_TIMEOUT_180_SEC = 8.0
TURN_DEBUG_PERIOD_SEC = 0.20


# ============================================================
# TIMING / SAFETY
# ============================================================

CONTROL_DT = 0.05
DRIVE_COMMAND_TIMEOUT = 0.25
POSITION_WAIT_TIMEOUT = 3.0
MAX_CELL_TIME_SEC = max(5.0, (CELL_LENGTH_M / FORWARD_SPEED_MPS) * 2.5)

DEBUG_MOVE_PRINT_PERIOD_SEC = 0.25


# ============================================================
# DIRECTIONS
#
# 0=N, 1=E, 2=S, 3=W
# Coordinates are logical DFS coordinates only.
# ============================================================

DIR_NAMES = ("N", "E", "S", "W")
DIR_VEC = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}

REL_LEFT = -1
REL_FRONT = 0
REL_RIGHT = 1
REL_BACK = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    for d, (vx, vy) in DIR_VEC.items():
        if (dx, dy) == (vx, vy):
            return d

    raise ValueError(f"Cells are not adjacent: {a} -> {b}")


def neighbor(cell, direction):
    dx, dy = DIR_VEC[direction]
    return (cell[0] + dx, cell[1] + dy)


# ============================================================
# CALIBRATION / SENSOR HELPERS
# ============================================================

def adc_to_cm(adc, calibration):
    """
    Piecewise-linear interpolation using monotonic calibration points.

    Returns:
        float cm : within reliable calibrated region
        None     : ADC says wall is farther than reliable calibration region,
                   sensor value is invalid, or wall is effectively unavailable
                   for wall-follow.
    """
    if adc is None:
        return None

    try:
        adc = float(adc)
    except Exception:
        return None

    if not math.isfinite(adc) or adc < SHARP_MIN_PLAUSIBLE_ADC:
        return None

    # Calibration is near -> far, ADC high -> low.
    near_cm, near_adc = calibration[0]
    far_cm, far_adc = calibration[-1]

    # Closer than nearest calibration point.
    if adc >= near_adc:
        return near_cm

    # Farther than reliable calibrated range.
    if adc < far_adc:
        return None

    for i in range(len(calibration) - 1):
        d1, a1 = calibration[i]
        d2, a2 = calibration[i + 1]

        # a1 >= adc >= a2
        if a1 >= adc >= a2:
            if abs(a1 - a2) < 1e-9:
                return (d1 + d2) * 0.5

            t = (a1 - adc) / (a1 - a2)
            return d1 + t * (d2 - d1)

    return None


# ============================================================
# ROBOT STATE
# ============================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        self.tof_mm = None
        self.position = None          # (x, y, z)
        self.attitude = None          # (yaw, pitch, roll)

        # Gimbal feedback:
        # (pitch relative chassis, yaw relative chassis,
        #  pitch ground, yaw ground)
        self.gimbal_angle = None

        # Raw chassis ESC / motor encoder-level telemetry.
        self.esc_speed = None         # rpm[4]
        self.esc_angle = None         # motor-angle raw[4]
        self.esc_timestamp = None
        self.esc_state = None

        # Chassis status flags, including slip flag.
        self.chassis_status = None

    def set_tof(self, value):
        with self.lock:
            self.tof_mm = value

    def get_tof(self):
        with self.lock:
            return self.tof_mm

    def set_position(self, value):
        with self.lock:
            self.position = value

    def get_position(self):
        with self.lock:
            return self.position

    def set_attitude(self, value):
        with self.lock:
            self.attitude = value

    def get_attitude(self):
        with self.lock:
            return self.attitude

    def set_gimbal_angle(self, value):
        with self.lock:
            self.gimbal_angle = value

    def get_gimbal_angle(self):
        with self.lock:
            return self.gimbal_angle

    def set_esc(self, speed, angle, timestamp, state):
        with self.lock:
            self.esc_speed = tuple(speed) if speed is not None else None
            self.esc_angle = tuple(angle) if angle is not None else None
            self.esc_timestamp = timestamp
            self.esc_state = state

    def get_esc(self):
        with self.lock:
            return (
                self.esc_speed,
                self.esc_angle,
                self.esc_timestamp,
                self.esc_state,
            )

    def set_chassis_status(self, value):
        with self.lock:
            self.chassis_status = tuple(value)

    def get_chassis_status(self):
        with self.lock:
            return self.chassis_status


# ============================================================
# DFS EXPLORER
# ============================================================

class DFSMazeExplorer:
    def __init__(self):
        self.ep_robot = robot.Robot()

        self.chassis = None
        self.gimbal = None
        self.sensor_adapter = None
        self.distance_sensor = None

        self.state = SharedState()

        self.left_adc_hist = deque(maxlen=SHARP_FILTER_SAMPLES)
        self.right_adc_hist = deque(maxlen=SHARP_FILTER_SAMPLES)

        # Logical robot heading. Start direction = N.
        self.heading = 0

        # Absolute chassis yaw reference.
        # base_yaw_deg is captured ONCE at startup and is never replaced by
        # a yaw value that may have drifted because of gimbal motion.
        self.base_yaw_deg = None
        self.yaw_ref_deg = None

        # Prevent two yaw-hold loops from commanding the chassis at once.
        self.yaw_hold_lock = threading.Lock()

        # DFS state.
        self.root = (0, 0)
        self.current = self.root
        self.visited = set()
        self.parent = {}
        self.open_dirs = {}       # cell -> ordered list of absolute open dirs
        self.blocked_edges = set()

        # Cells where LEFT + FRONT + RIGHT are all inside the close-range
        # dead-end threshold.  DFS immediately reverses out of these cells.
        self.dead_end_cells = set()

        # Raw LEFT / FRONT / RIGHT ToF scan values for debugging/map logs.
        self.cell_scan_mm = {}

        # Persistent-map state.
        self.map_created_at = datetime.now().isoformat(timespec="seconds")
        self.map_complete = False
        self.loaded_map_path = None
        self.known_map_cells = set()

        # ----------------------------------------------------
        # Sharp control-authority state
        # ----------------------------------------------------
        # Only this side is allowed to generate lateral y commands.
        self.sharp_authority = None
        self.sharp_authority_since = 0.0

        # BOTH-IR / sequential dual-side supervisor state.
        self.ir_both_front_override_until = 0.0
        self.ir_replan_requested = False
        self.ir_route_hint = None
        self.ir_route_scan_mm = None

        # Edge/event memory for IR sensors.
        # This allows LEFT then RIGHT (or RIGHT then LEFT) to trigger the
        # gimbal supervisor even if they were never LOW simultaneously.
        self.ir_prev_left_low = False
        self.ir_prev_right_low = False
        self.ir_last_left_event_t = None
        self.ir_last_right_event_t = None
        self.ir_dual_sequence_pending = False
        self.ir_dual_sequence_reason = None

        self.running = True

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    def tof_callback(self, distance_info):
        try:
            if distance_info and len(distance_info) > TOF_INDEX:
                value = distance_info[TOF_INDEX]
                if value is not None:
                    self.state.set_tof(float(value))
        except Exception:
            pass

    def position_callback(self, position_info):
        try:
            if position_info and len(position_info) >= 3:
                x, y, z = position_info[:3]
                self.state.set_position((float(x), float(y), float(z)))
        except Exception:
            pass

    def attitude_callback(self, attitude_info):
        try:
            if attitude_info and len(attitude_info) >= 3:
                yaw, pitch, roll = attitude_info[:3]
                self.state.set_attitude(
                    (float(yaw), float(pitch), float(roll))
                )
        except Exception:
            pass

    def gimbal_angle_callback(self, angle_info):
        try:
            if angle_info and len(angle_info) >= 4:
                p, y, pg, yg = angle_info[:4]
                self.state.set_gimbal_angle(
                    (float(p), float(y), float(pg), float(yg))
                )
        except Exception:
            pass

    def esc_callback(self, esc_info):
        try:
            if esc_info and len(esc_info) >= 4:
                speed, angle, timestamp, state = esc_info[:4]
                self.state.set_esc(speed, angle, timestamp, state)
        except Exception:
            pass

    def chassis_status_callback(self, status_info):
        try:
            if status_info:
                self.state.set_chassis_status(status_info)
        except Exception:
            pass

    # --------------------------------------------------------
    # CONNECT / CLEANUP
    # --------------------------------------------------------

    def connect(self):
        print("============================================================")
        print(" RoboMaster DFS + Sharp wall-follow + Gimbal ToF")
        print("============================================================")
        print(f" IR LEFT    : Adapter ID {IR_LEFT_ID}, Port {SENSOR_PORT} (ACTIVE LOW)")
        print(f" Sharp LEFT : Adapter ID {SHARP_LEFT_ID}, Port {SENSOR_PORT}")
        print(f" Sharp RIGHT: Adapter ID {SHARP_RIGHT_ID}, Port {SENSOR_PORT}")
        print(f" IR RIGHT   : Adapter ID {IR_RIGHT_ID}, Port {SENSOR_PORT} (ACTIVE LOW)")
        print(f" ToF        : CAN distance_info[{TOF_INDEX}]")
        print(f" Gimbal     : front, pitch {GIMBAL_PITCH_DEG:+.1f} deg")
        print(
            f" IR latch   : opposite-side events within "
            f"{IR_DUAL_EVENT_WINDOW_SEC:.2f}s -> Gimbal route scan"
        )
        print(
            f" Slide guard: destination Sharp <= "
            f"{IR_SLIDE_DEST_SHARP_STOP_CM:.1f}cm -> STOP + Gimbal scan"
        )
        print(
            " Feedback   : position + attitude + ESC encoder telemetry + slip"
        )
        print(
            " Gimbal     : absolute fast scan; recenter only at startup/cleanup"
        )
        print("============================================================")

        print("[CONNECT] RoboMaster AP...")
        self.ep_robot.initialize(conn_type=CONN_TYPE)

        # Gimbal must be independent of chassis.
        self.ep_robot.set_robot_mode(mode=robot.FREE)

        self.chassis = self.ep_robot.chassis
        self.gimbal = self.ep_robot.gimbal
        self.sensor_adapter = self.ep_robot.sensor_adaptor
        self.distance_sensor = self.ep_robot.sensor

        print("[SUB] ToF...")
        print("      result =", self.distance_sensor.sub_distance(
            freq=TOF_FREQ_HZ,
            callback=self.tof_callback
        ))

        print("[SUB] chassis position...")
        print("      result =", self.chassis.sub_position(
            freq=POSITION_FREQ_HZ,
            callback=self.position_callback
        ))

        print("[SUB] chassis attitude...")
        print("      result =", self.chassis.sub_attitude(
            freq=ATTITUDE_FREQ_HZ,
            callback=self.attitude_callback
        ))

        print("[SUB] chassis ESC / wheel encoder telemetry...")
        print("      result =", self.chassis.sub_esc(
            freq=ESC_FREQ_HZ,
            callback=self.esc_callback
        ))

        print("[SUB] chassis status / slip...")
        print("      result =", self.chassis.sub_status(
            freq=STATUS_FREQ_HZ,
            callback=self.chassis_status_callback
        ))

        print("[SUB] gimbal relative angle...")
        print("      result =", self.gimbal.sub_angle(
            freq=GIMBAL_ANGLE_FREQ_HZ,
            callback=self.gimbal_angle_callback
        ))

        # Give telemetry a moment to arrive.
        time.sleep(0.5)

        self.stop_chassis()

        # IMPORTANT: Capture the chassis reference BEFORE moving the gimbal.
        # If the turret kicks the base a little, that disturbed yaw must NOT
        # become the new target.
        t0 = time.monotonic()
        while self.current_yaw() is None and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)

        self.base_yaw_deg = self.current_yaw()
        if self.base_yaw_deg is None:
            raise RuntimeError("No chassis yaw telemetry; cannot initialize yaw lock")

        self.yaw_ref_deg = self.desired_yaw_for_heading(self.heading)
        print(
            f"[YAW LOCK] base={self.base_yaw_deg:+.2f} deg "
            f"heading={DIR_NAMES[self.heading]} "
            f"target={self.yaw_ref_deg:+.2f} deg"
        )

        # Physical gimbal homing ONCE at startup.  Normal scans after this use
        # absolute moveto() and angle feedback instead of repeated recenter().
        action = self.gimbal.recenter(
            pitch_speed=GIMBAL_PITCH_SPEED,
            yaw_speed=GIMBAL_YAW_SPEED
        )
        self.run_gimbal_action_with_yaw_lock(
            action,
            timeout=GIMBAL_ACTION_TIMEOUT_SEC,
            residual_settle=GIMBAL_SETTLE_SEC
        )
        self.gimbal_front_down(force=True)

        print("[READY] Connected.")

    def cleanup(self):
        print("\n[CLEANUP] stopping robot...")

        # Best-effort partial-map save before shutting telemetry down.
        try:
            if MAP_AUTOSAVE and self.mapped_cells():
                self.save_map(final=False)
        except Exception as e:
            print(f"[MAP SAVE WARN] cleanup autosave failed: {e}")

        self.running = False

        try:
            self.stop_chassis()
        except Exception:
            pass

        try:
            self.gimbal.recenter(
                pitch_speed=GIMBAL_PITCH_SPEED,
                yaw_speed=GIMBAL_YAW_SPEED
            ).wait_for_completed()
        except Exception:
            pass

        try:
            self.distance_sensor.unsub_distance()
        except Exception:
            pass

        try:
            self.chassis.unsub_position()
        except Exception:
            pass

        try:
            self.chassis.unsub_attitude()
        except Exception:
            pass

        try:
            self.chassis.unsub_esc()
        except Exception:
            pass

        try:
            self.chassis.unsub_status()
        except Exception:
            pass

        try:
            self.gimbal.unsub_angle()
        except Exception:
            pass

        try:
            self.ep_robot.close()
        except Exception:
            pass

        print("[CLEANUP] done.")

    # --------------------------------------------------------
    # BASIC MOTION
    # --------------------------------------------------------

    def stop_chassis(self):
        if self.chassis is not None:
            self.chassis.drive_speed(
                x=0.0,
                y=0.0,
                z=0.0,
                timeout=DRIVE_COMMAND_TIMEOUT
            )

    def wait_for_position(self, timeout=POSITION_WAIT_TIMEOUT):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            pos = self.state.get_position()
            if pos is not None:
                return pos
            time.sleep(0.05)
        return None

    def current_yaw(self):
        att = self.state.get_attitude()
        if att is None:
            return None
        return att[0]

    def chassis_slip_detected(self):
        status = self.state.get_chassis_status()

        # SDK order:
        # static, uphill, downhill, on_slope, pickup, slip, impact_x,
        # impact_y, impact_z, roll_over, hill_static
        if status is None or len(status) < 6:
            return False

        return bool(status[5])

    def encoder_snapshot(self):
        speed, angle, timestamp, state = self.state.get_esc()
        return {
            "speed_rpm": speed,
            "angle_raw": angle,
            "timestamp": timestamp,
            "state": state,
        }

    def desired_yaw_for_heading(self, heading=None):
        """
        Return the fixed absolute yaw target for a logical DFS heading.

        Real-robot sign confirmed by test:
            +yaw / +z = rotate RIGHT
            -yaw / -z = rotate LEFT

        Therefore, from the startup N reference:
            N = base
            E = base + 90
            S = base + 180
            W = base - 90   (same as base + 270, wrapped)

        The target is always calculated from base_yaw_deg, never from the
        current yaw, so gimbal reaction torque cannot accumulate into the map.
        """
        if self.base_yaw_deg is None:
            return None

        if heading is None:
            heading = self.heading

        return wrap_deg(self.base_yaw_deg + 90.0 * (heading % 4))

    def yaw_error_deg(self, target_yaw=None):
        if target_yaw is None:
            target_yaw = self.yaw_ref_deg

        now = self.current_yaw()
        if target_yaw is None or now is None:
            return None

        return wrap_deg(target_yaw - now)

    def yaw_hold_command(self, target_yaw=None, stationary=False):
        """Return corrective chassis z command that drives yaw error to 0."""
        if not YAW_HOLD_ENABLED:
            return 0.0

        error = self.yaw_error_deg(target_yaw)
        if error is None:
            return 0.0

        if abs(error) <= YAW_HOLD_DEADBAND_DEG:
            return 0.0

        if stationary:
            kp = STATIONARY_YAW_HOLD_KP
            limit = STATIONARY_YAW_HOLD_MAX_DPS
        else:
            kp = YAW_HOLD_KP
            limit = YAW_HOLD_MAX_DPS

        z = YAW_DRIVE_SIGN * kp * error
        return clamp(z, -limit, limit)

    def hold_heading_stationary(self, duration=STATIONARY_SETTLE_SEC):
        """
        Keep x=y=0 and actively drive chassis yaw error toward zero.
        Used after turns and after gimbal movements.
        """
        if not YAW_HOLD_ENABLED or self.yaw_ref_deg is None:
            self.stop_chassis()
            return

        end_t = time.monotonic() + max(0.0, duration)
        dt = 1.0 / STATIONARY_YAW_HOLD_HZ

        with self.yaw_hold_lock:
            while self.running and time.monotonic() < end_t:
                z = self.yaw_hold_command(self.yaw_ref_deg, stationary=True)
                self.chassis.drive_speed(
                    x=0.0, y=0.0, z=z, timeout=DRIVE_COMMAND_TIMEOUT
                )
                time.sleep(dt)

            self.stop_chassis()

    def run_gimbal_action_with_yaw_lock(
        self,
        action,
        timeout=GIMBAL_ACTION_TIMEOUT_SEC,
        residual_settle=GIMBAL_SETTLE_SEC,
    ):
        """
        Wait for one gimbal action while actively holding chassis yaw.

        v7 difference:
          - every action has a timeout
          - no fixed 0.12 s penalty after every single gimbal action
          - residual settle is short and configurable
        """
        if not YAW_HOLD_ENABLED or self.yaw_ref_deg is None:
            action.wait_for_completed(timeout=timeout)
            return

        stop_event = threading.Event()

        def _holder():
            dt = 1.0 / STATIONARY_YAW_HOLD_HZ
            with self.yaw_hold_lock:
                while self.running and not stop_event.is_set():
                    z = self.yaw_hold_command(
                        self.yaw_ref_deg, stationary=True
                    )
                    self.chassis.drive_speed(
                        x=0.0, y=0.0, z=z, timeout=DRIVE_COMMAND_TIMEOUT
                    )
                    stop_event.wait(dt)

                self.stop_chassis()

        thread = threading.Thread(target=_holder, daemon=True)
        thread.start()

        try:
            action.wait_for_completed(timeout=timeout)
        finally:
            stop_event.set()
            thread.join(timeout=0.35)

        if residual_settle > 0:
            self.hold_heading_stationary(residual_settle)

    # --------------------------------------------------------
    # GIMBAL - FAST ABSOLUTE SCAN
    # --------------------------------------------------------

    def current_gimbal_relative(self):
        data = self.state.get_gimbal_angle()

        if data is None:
            return None, None

        pitch, yaw, _, _ = data
        return pitch, yaw

    def gimbal_at_target(self, pitch, yaw):
        p_now, y_now = self.current_gimbal_relative()

        if p_now is None or y_now is None:
            return False

        return (
            abs(p_now - pitch) <= GIMBAL_ANGLE_TOL_DEG
            and abs(y_now - yaw) <= GIMBAL_ANGLE_TOL_DEG
        )

    def gimbal_goto(self, yaw_deg, pitch_deg=GIMBAL_PITCH_DEG, force=False):
        """
        Move directly to one gimbal pose relative to chassis.

        SDK sub_angle() reports pitch_angle/yaw_angle relative to the chassis.
        After one startup recenter, using moveto() avoids the old pattern:
            recenter -> pitch -> +/-45 -> +/-45 -> settle
        on every single measurement.
        """
        yaw_deg = clamp(float(yaw_deg), -250.0, 250.0)
        pitch_deg = clamp(float(pitch_deg), -25.0, 30.0)

        if not force and self.gimbal_at_target(pitch_deg, yaw_deg):
            return

        action = self.gimbal.moveto(
            pitch=pitch_deg,
            yaw=yaw_deg,
            pitch_speed=GIMBAL_PITCH_SPEED,
            yaw_speed=GIMBAL_YAW_SPEED
        )

        self.run_gimbal_action_with_yaw_lock(
            action,
            timeout=GIMBAL_ACTION_TIMEOUT_SEC,
            residual_settle=GIMBAL_SETTLE_SEC
        )

    def gimbal_front_down(self, force=False):
        """
        Ensure ToF is forward and pitched down 5 degrees.

        No recenter here.  Recenter is reserved for startup/cleanup only.
        If angle feedback says the turret is already there, this returns
        immediately and costs essentially no field time.
        """
        self.gimbal_goto(
            yaw_deg=0.0,
            pitch_deg=GIMBAL_PITCH_DEG,
            force=force
        )

    def gimbal_point_relative_from_front(self, yaw_deg):
        """
        Direct absolute yaw relative to chassis.
        """
        self.gimbal_goto(
            yaw_deg=yaw_deg,
            pitch_deg=GIMBAL_PITCH_DEG
        )

    # --------------------------------------------------------
    # ToF
    # --------------------------------------------------------

    def sample_tof_median(self, samples=TOF_SCAN_SAMPLES):
        values = []

        for _ in range(samples):
            v = self.state.get_tof()

            if v is not None and math.isfinite(v) and v > 0:
                values.append(float(v))

            time.sleep(TOF_SCAN_INTERVAL_SEC)

        if not values:
            return None

        return float(statistics.median(values))

    def scan_tof_at_yaw(self, yaw_deg):
        self.gimbal_point_relative_from_front(yaw_deg)
        return self.sample_tof_median()

    # --------------------------------------------------------
    # SHARP
    # --------------------------------------------------------

    def read_sharp_adc(self):
        left = None
        right = None

        try:
            left = self.sensor_adapter.get_adc(
                id=SHARP_LEFT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        try:
            right = self.sensor_adapter.get_adc(
                id=SHARP_RIGHT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        if left is not None:
            try:
                self.left_adc_hist.append(float(left))
            except Exception:
                pass

        if right is not None:
            try:
                self.right_adc_hist.append(float(right))
            except Exception:
                pass

        left_med = (
            statistics.median(self.left_adc_hist)
            if self.left_adc_hist else None
        )
        right_med = (
            statistics.median(self.right_adc_hist)
            if self.right_adc_hist else None
        )

        return left_med, right_med

    def read_sharp_cm(self):
        left_adc, right_adc = self.read_sharp_adc()

        left_cm = adc_to_cm(left_adc, LEFT_CAL)
        right_cm = adc_to_cm(right_adc, RIGHT_CAL)

        return left_cm, right_cm, left_adc, right_adc

    # --------------------------------------------------------
    # IR HARD SAFETY
    # --------------------------------------------------------

    def read_ir_once(self):
        """
        Returns:
            left_low, right_low, left_raw, right_raw

        Sensors are ACTIVE LOW:
            raw == 0 -> obstacle / wall
            raw == 1 -> clear
        """
        left_raw = None
        right_raw = None

        try:
            left_raw = self.sensor_adapter.get_io(
                id=IR_LEFT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        try:
            right_raw = self.sensor_adapter.get_io(
                id=IR_RIGHT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        left_low = (left_raw == 0)
        right_low = (right_raw == 0)

        return left_low, right_low, left_raw, right_raw

    def _update_ir_event_latch(self, left_low, right_low):
        """
        Record LOW edges and detect a dual-side sequence.

        A scan is requested when:
          - LEFT and RIGHT become LOW together, OR
          - one side becomes LOW while the other side is already LOW, OR
          - LEFT and RIGHT LOW edges occur within IR_DUAL_EVENT_WINDOW_SEC,
            even if the first side has already cleared.

        Continuous LOW does not repeatedly create new events; a new LOW edge
        is required after the previous event has been consumed.
        """
        now = time.monotonic()

        left_edge = bool(left_low and not self.ir_prev_left_low)
        right_edge = bool(right_low and not self.ir_prev_right_low)

        if left_edge:
            self.ir_last_left_event_t = now

        if right_edge:
            self.ir_last_right_event_t = now

        dual = False
        reason = None

        # Simultaneous / overlapping event.
        if (left_edge and right_low) or (right_edge and left_low):
            dual = True
            reason = "IR sides overlapped"

        # Sequential event: first side may already have cleared.
        if (
            self.ir_last_left_event_t is not None
            and self.ir_last_right_event_t is not None
        ):
            dt = abs(
                self.ir_last_left_event_t - self.ir_last_right_event_t
            )

            if dt <= IR_DUAL_EVENT_WINDOW_SEC:
                dual = True

                if reason is None:
                    if self.ir_last_left_event_t < self.ir_last_right_event_t:
                        reason = f"LEFT then RIGHT within {dt:.2f}s"
                    elif self.ir_last_right_event_t < self.ir_last_left_event_t:
                        reason = f"RIGHT then LEFT within {dt:.2f}s"
                    else:
                        reason = "LEFT + RIGHT same-time event"

        if dual:
            self.ir_dual_sequence_pending = True
            self.ir_dual_sequence_reason = reason

        self.ir_prev_left_low = bool(left_low)
        self.ir_prev_right_low = bool(right_low)

    def consume_ir_dual_sequence(self):
        """
        Consume one latched dual-side event.

        The timestamps are cleared so the same pair of old edges cannot
        repeatedly force route scans.
        """
        if not self.ir_dual_sequence_pending:
            return False, None

        reason = self.ir_dual_sequence_reason

        self.ir_dual_sequence_pending = False
        self.ir_dual_sequence_reason = None
        self.ir_last_left_event_t = None
        self.ir_last_right_event_t = None

        return True, reason

    def read_ir_filtered(self, samples=IR_FILTER_SAMPLES):
        """
        Majority filter for digital IR.

        Besides returning current LOW states, this updates the edge/event latch
        used to detect LEFT->RIGHT or RIGHT->LEFT events that happen close
        together in time.
        """
        left_hits = 0
        right_hits = 0
        left_last = None
        right_last = None

        samples = max(1, int(samples))

        for _ in range(samples):
            l_low, r_low, l_raw, r_raw = self.read_ir_once()

            left_last = l_raw
            right_last = r_raw

            if l_low:
                left_hits += 1
            if r_low:
                right_hits += 1

            time.sleep(0.015)

        needed = samples // 2 + 1

        left_low = left_hits >= needed
        right_low = right_hits >= needed

        self._update_ir_event_latch(left_low, right_low)

        return (
            left_low,
            right_low,
            left_last,
            right_last,
        )

    def slide_lateral_distance(self, direction, distance_m=IR_RECOVERY_SLIDE_M):
        """
        Small odometry-controlled lateral nudge while preserving chassis yaw.

        Single command authority is preserved:
            the recovery controller owns the y command.

        The Sharp sensor on the DESTINATION side is only a VETO/interlock;
        it never generates a second competing y command.

        direction:
            "RIGHT" -> +y, RIGHT Sharp guards the destination wall
            "LEFT"  -> -y, LEFT Sharp guards the destination wall

        Returns:
            "DONE"               target slide distance reached
            "DUAL_IR"            sequential/simultaneous two-side IR event
            "DEST_IR_BLOCKED"    destination IR became LOW
            "DEST_SHARP_BLOCKED" destination Sharp <= safety threshold
            "TIMEOUT"            slide timed out
        """
        direction = direction.upper()

        if direction not in ("LEFT", "RIGHT"):
            raise ValueError(f"Invalid slide direction: {direction}")

        y_sign = +1.0 if direction == "RIGHT" else -1.0
        y_cmd = y_sign * IR_RECOVERY_SLIDE_SPEED_MPS

        start_pos = self.wait_for_position(timeout=0.5)
        target_yaw = self.yaw_ref_deg
        start_t = time.monotonic()

        print(
            f"[IR RECOVERY] slide {direction} "
            f"{distance_m * 100.0:.1f} cm "
            f"(dest Sharp stop <= {IR_SLIDE_DEST_SHARP_STOP_CM:.1f} cm)"
        )

        try:
            while self.running:
                elapsed = time.monotonic() - start_t

                if elapsed >= IR_RECOVERY_SLIDE_TIMEOUT_SEC:
                    print("[IR RECOVERY WARN] slide timeout")
                    return "TIMEOUT"

                # ------------------------------------------------
                # IR supervision + sequential dual-side latch.
                # ------------------------------------------------
                l_low, r_low, _, _ = self.read_ir_filtered(samples=1)

                dual_event, dual_reason = self.consume_ir_dual_sequence()

                if dual_event:
                    print(
                        f"[IR RECOVERY STOP] dual-side IR event during slide: "
                        f"{dual_reason}"
                    )
                    return "DUAL_IR"

                # Destination digital IR is a hard stop.
                if direction == "RIGHT" and r_low:
                    print(
                        "[IR RECOVERY STOP] RIGHT IR became LOW "
                        "while sliding RIGHT"
                    )
                    return "DEST_IR_BLOCKED"

                if direction == "LEFT" and l_low:
                    print(
                        "[IR RECOVERY STOP] LEFT IR became LOW "
                        "while sliding LEFT"
                    )
                    return "DEST_IR_BLOCKED"

                # ------------------------------------------------
                # Destination Sharp interlock.
                #
                # We read BOTH for diagnostics, but only the sensor on the
                # direction we are sliding toward can veto the slide.
                # ------------------------------------------------
                left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

                if direction == "RIGHT":
                    dest_cm = right_cm
                    dest_adc = right_adc
                    dest_name = "RIGHT"
                else:
                    dest_cm = left_cm
                    dest_adc = left_adc
                    dest_name = "LEFT"

                if (
                    dest_cm is not None
                    and dest_cm <= IR_SLIDE_DEST_SHARP_STOP_CM
                ):
                    print(
                        f"[IR RECOVERY STOP] {dest_name} Sharp destination "
                        f"too close: {dest_cm:.1f} cm "
                        f"(ADC={dest_adc}) <= "
                        f"{IR_SLIDE_DEST_SHARP_STOP_CM:.1f} cm"
                    )
                    return "DEST_SHARP_BLOCKED"

                # ------------------------------------------------
                # Odometry slide-distance completion.
                # ------------------------------------------------
                if start_pos is not None:
                    pos = self.state.get_position()

                    if pos is not None:
                        dx = pos[0] - start_pos[0]
                        dy = pos[1] - start_pos[1]
                        moved = math.hypot(dx, dy)

                        if moved >= distance_m:
                            return "DONE"

                else:
                    # Telemetry fallback: conservative timed nudge.
                    if elapsed >= distance_m / max(
                        IR_RECOVERY_SLIDE_SPEED_MPS, 1e-6
                    ):
                        return "DONE"

                z_cmd = self.yaw_hold_command(
                    target_yaw=target_yaw,
                    stationary=False
                )

                self.chassis.drive_speed(
                    x=0.0,
                    y=y_cmd,
                    z=z_cmd,
                    timeout=DRIVE_COMMAND_TIMEOUT
                )

                time.sleep(CONTROL_DT)

        finally:
            self.stop_chassis()
            self.hold_heading_stationary(IR_RECOVERY_SETTLE_SEC)

    def scan_route_for_both_ir(self, context=""):
        """
        IR dual-side supervisor.

        FIELD-FAST behavior:
          1) STOP
          2) ensure gimbal is FRONT/-5 deg
          3) read FRONT first
          4) if FRONT is open -> continue immediately (no needless head sweep)
          5) only if FRONT is blocked/uncertain, scan LEFT and RIGHT
          6) return gimbal to FRONT

        This removes the long delay that used to occur when both IR sensors
        saw the side walls of an otherwise-open straight corridor.
        """
        self.stop_chassis()

        prefix = f"[IR BOTH {context}]" if context else "[IR BOTH]"
        print(f"{prefix} STOP -> FAST GIMBAL ROUTE CHECK")

        scan = {
            "LEFT": None,
            "FRONT": None,
            "RIGHT": None,
        }

        # FRONT FIRST.  Usually the turret is already here, so this often
        # requires zero mechanical motion.
        self.gimbal_front_down()
        scan["FRONT"] = self.sample_tof_median()

        def is_open(v):
            return (
                v is not None
                and math.isfinite(v)
                and v >= IR_BOTH_ROUTE_OPEN_MM
            )

        if IR_FAST_FRONT_FIRST and is_open(scan["FRONT"]):
            action = "FRONT"
            self.ir_route_hint = action
            self.ir_route_scan_mm = dict(scan)

            print(
                f"{prefix} FRONT={scan['FRONT']} mm OPEN "
                "-> continue immediately (skip L/R sweep)"
            )
            return action, scan

        # FRONT is blocked/uncertain -> now side information matters.
        scan["LEFT"] = self.scan_tof_at_yaw(-90.0)
        scan["RIGHT"] = self.scan_tof_at_yaw(+90.0)

        # Return once, not before every measurement.
        self.gimbal_front_down()

        self.ir_route_scan_mm = dict(scan)

        side_candidates = []

        if is_open(scan["LEFT"]):
            side_candidates.append(("LEFT", scan["LEFT"]))

        if is_open(scan["RIGHT"]):
            side_candidates.append(("RIGHT", scan["RIGHT"]))

        if is_open(scan["FRONT"]):
            action = "FRONT"
        elif side_candidates:
            action = max(side_candidates, key=lambda item: item[1])[0]
        else:
            action = "BACK"

        self.ir_route_hint = action

        print(
            f"{prefix} L={scan['LEFT']} "
            f"F={scan['FRONT']} "
            f"R={scan['RIGHT']} mm "
            f"-> action={action}"
        )

        return action, scan

    def ir_clearance_recovery(self, context=""):
        """
        Digital IR supervisory safety.

        ONE IR LOW:
            Keep the previous opposite-slide recovery:
              LEFT LOW  -> slide RIGHT a little
              RIGHT LOW -> slide LEFT a little

        BOTH IR LOW:
            NEVER let LEFT and RIGHT recovery commands fight each other.
            Stop and ask the Gimbal ToF where the route is.

            FRONT open:
                allow forward motion to resume; Sharp authority keeps the
                chassis centered.

            LEFT/RIGHT open, FRONT blocked:
                request DFS replan/rescan instead of forcing a lateral slide.

            No route:
                request BACK/dead-end behavior.
        """
        self.stop_chassis()

        for attempt in range(1, IR_RECOVERY_MAX_ATTEMPTS + 1):
            left_low, right_low, left_raw, right_raw = self.read_ir_filtered()

            dual_event, dual_reason = self.consume_ir_dual_sequence()

            # LEFT then RIGHT (or RIGHT then LEFT) within the latch window
            # is treated exactly like BOTH LOW, even if they are not LOW at
            # the same instant.
            if dual_event:
                print(
                    f"[IR SEQUENCE {context}] {dual_reason} "
                    "-> STOP + GIMBAL ROUTE SCAN"
                )

                action, _ = self.scan_route_for_both_ir(
                    context=f"{context}/SEQUENCE"
                )

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False
                    return True

                self.ir_replan_requested = True
                return True

            if not left_low and not right_low:
                self.ir_route_hint = None
                return True

            prefix = f"[IR RECOVERY {context}]" if context else "[IR RECOVERY]"

            print(
                f"{prefix} attempt={attempt}/{IR_RECOVERY_MAX_ATTEMPTS} "
                f"IR_L={left_raw} IR_R={right_raw}"
            )

            # ------------------------------------------------
            # BOTH LOW -> supervisor scan. No random slide.
            # ------------------------------------------------
            if left_low and right_low:
                action, scan = self.scan_route_for_both_ir(context=context)

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False

                    print(
                        f"{prefix} FRONT is open -> "
                        "resume with Sharp single-authority centering"
                    )
                    return True

                # FRONT blocked, but another direction is available (or BACK).
                # Do not move blindly. The DFS loop will rescan/replan the cell.
                self.ir_replan_requested = True

                print(
                    f"{prefix} FRONT blocked -> DFS REPLAN hint={action}"
                )
                return True

            # ------------------------------------------------
            # ONE LOW -> opposite nudge remains unambiguous.
            # ------------------------------------------------
            if left_low:
                escape_dir = "RIGHT"
                offending_side = "LEFT"
            else:
                escape_dir = "LEFT"
                offending_side = "RIGHT"

            slide_result = self.slide_lateral_distance(escape_dir)

            # If the slide was vetoed because the destination side became
            # unsafe (Sharp/IR), or because the opposite IR fired shortly
            # after the first one, do NOT try another blind lateral move.
            # Ask the gimbal where the route actually is.
            if slide_result != "DONE":
                print(
                    f"{prefix} slide interrupted: {slide_result} "
                    "-> GIMBAL ROUTE SCAN"
                )

                action, _ = self.scan_route_for_both_ir(
                    context=f"{context}/SLIDE_ABORT"
                )

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False
                    return True

                self.ir_replan_requested = True
                return True

            side_yaw = -90.0 if offending_side == "LEFT" else +90.0

            side_mm = self.scan_tof_at_yaw(side_yaw)
            front_mm = self.scan_tof_at_yaw(0.0)
            self.gimbal_front_down()

            print(
                f"{prefix} after slide {escape_dir}: "
                f"{offending_side}_ToF={side_mm} mm "
                f"FRONT_ToF={front_mm} mm"
            )

            if front_mm is not None and front_mm <= FRONT_HARD_STOP_MM:
                print(
                    f"{prefix} FRONT still too close "
                    f"({front_mm:.0f} mm) -> STOP"
                )
                self.stop_chassis()
                return False

            left_low2, right_low2, left_raw2, right_raw2 = (
                self.read_ir_filtered()
            )

            dual_event2, dual_reason2 = self.consume_ir_dual_sequence()

            if dual_event2:
                print(
                    f"{prefix} opposite IR followed the first event: "
                    f"{dual_reason2} -> GIMBAL ROUTE SCAN"
                )

                action, _ = self.scan_route_for_both_ir(
                    context=f"{context}/SEQUENCE_AFTER_SLIDE"
                )

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False
                    return True

                self.ir_replan_requested = True
                return True

            # If the nudge caused BOTH sensors to become LOW, switch immediately
            # to the gimbal supervisor instead of issuing another opposite slide.
            if left_low2 and right_low2:
                action, _ = self.scan_route_for_both_ir(context=context)

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False
                    return True

                self.ir_replan_requested = True
                return True

            side_clear_by_ir = (
                (offending_side == "LEFT" and not left_low2)
                or (offending_side == "RIGHT" and not right_low2)
            )

            side_clear_by_tof = (
                side_mm is None
                or side_mm > IR_GIMBAL_SIDE_CLEAR_MM
            )

            if side_clear_by_ir and side_clear_by_tof:
                print(
                    f"{prefix} CLEARED "
                    f"IR_L={left_raw2} IR_R={right_raw2}"
                )
                return True

            print(
                f"{prefix} still close; retrying "
                f"IR_L={left_raw2} IR_R={right_raw2}"
            )

        self.stop_chassis()
        self.gimbal_front_down()
        print("[IR RECOVERY] max attempts reached -> STOP")
        return False

    # --------------------------------------------------------
    # CORRIDOR CONTROL
    # --------------------------------------------------------

    def _set_sharp_authority(self, side, reason=""):
        if side not in ("LEFT", "RIGHT", None):
            raise ValueError(f"Invalid Sharp authority: {side}")

        if side != self.sharp_authority:
            old = self.sharp_authority
            self.sharp_authority = side
            self.sharp_authority_since = time.monotonic()

            print(
                f"[AUTH] {old} -> {side}"
                + (f" reason={reason}" if reason else "")
            )

        return self.sharp_authority

    def choose_initial_authority(self, left_cm, right_cm):
        """
        Both sensors may be observed here, but they DO NOT both command y.

        Pick the sensor whose wall is in the more useful control region.
        Once selected, the authority manager keeps it sticky.
        """
        if left_cm is None and right_cm is None:
            return self._set_sharp_authority(None, "no valid Sharp")

        if left_cm is None:
            return self._set_sharp_authority("RIGHT", "LEFT unavailable")

        if right_cm is None:
            return self._set_sharp_authority("LEFT", "RIGHT unavailable")

        # Safety takes precedence at acquisition.
        if left_cm <= AUTHORITY_DANGER_CM or right_cm <= AUTHORITY_DANGER_CM:
            if left_cm <= right_cm:
                return self._set_sharp_authority(
                    "LEFT", "LEFT is nearest wall"
                )
            return self._set_sharp_authority(
                "RIGHT", "RIGHT is nearest wall"
            )

        lerr = abs(left_cm - CENTER_TARGET_CM)
        rerr = abs(right_cm - CENTER_TARGET_CM)

        if lerr <= rerr:
            return self._set_sharp_authority(
                "LEFT", "initial center authority"
            )

        return self._set_sharp_authority(
            "RIGHT", "initial center authority"
        )

    def update_sharp_authority(self, left_cm, right_cm):
        """
        Aircraft-style arbitration:
          - both sensors are monitors
          - exactly ONE sensor owns lateral y-control
          - no blending / no simultaneous left+right y commands
          - immediate transfer only for a safety reason
          - otherwise hold authority to avoid chatter
        """
        now = time.monotonic()
        current = self.sharp_authority

        # No owner yet.
        if current is None:
            return self.choose_initial_authority(left_cm, right_cm)

        current_dist = left_cm if current == "LEFT" else right_cm
        other = "RIGHT" if current == "LEFT" else "LEFT"
        other_dist = right_cm if other == "RIGHT" else left_cm

        # Current owner's measurement vanished -> hand over immediately if
        # the other monitor still has a valid wall.
        if current_dist is None:
            if other_dist is not None:
                return self._set_sharp_authority(
                    other, f"{current} unavailable"
                )
            return self._set_sharp_authority(None, "both unavailable")

        # The non-owner sees a dangerously close wall.
        # Transfer AUTHORITY to that sensor; do not combine commands.
        if (
            other_dist is not None
            and other_dist <= AUTHORITY_DANGER_CM
            and (
                current_dist > AUTHORITY_DANGER_CM
                or other_dist + AUTHORITY_SWITCH_MARGIN_CM < current_dist
            )
        ):
            return self._set_sharp_authority(
                other, f"{other} safety takeover"
            )

        held_for = now - self.sharp_authority_since

        if held_for < AUTHORITY_MIN_HOLD_SEC:
            return current

        # Owner is seeing a very distant wall while the other sensor has a
        # better usable reference: release and hand over.
        if (
            current_dist >= AUTHORITY_FAR_RELEASE_CM
            and other_dist is not None
            and other_dist < current_dist - AUTHORITY_SWITCH_MARGIN_CM
        ):
            return self._set_sharp_authority(
                other, f"{current} wall too far"
            )

        # Otherwise keep the same master.  This is the anti-fighting rule.
        return current

    def corridor_lateral_command(self, left_cm, right_cm, authority):
        """
        Generate y from ONE Sharp sensor only.

        y > 0 -> slide RIGHT
        y < 0 -> slide LEFT
        """
        if authority == "LEFT":
            dist = left_cm

            if dist is None:
                return 0.0, "AUTH_LEFT_NO_DATA"

            error = CENTER_TARGET_CM - dist

            if abs(error) <= CENTER_DEADBAND_CM:
                return 0.0, "AUTH_LEFT_CENTERED"

            y = CENTER_KP * error

            if dist <= AUTHORITY_HARD_CM:
                y = max(y, AUTHORITY_HARD_STRAFE_MPS)
                mode = "AUTH_LEFT_HARD"
            else:
                mode = "AUTH_LEFT_CENTER"

            return (
                clamp(y, -MAX_CENTER_STRAFE_MPS, AUTHORITY_HARD_STRAFE_MPS),
                mode
            )

        if authority == "RIGHT":
            dist = right_cm

            if dist is None:
                return 0.0, "AUTH_RIGHT_NO_DATA"

            # Right wall too close -> negative y (slide LEFT).
            error = dist - CENTER_TARGET_CM

            if abs(error) <= CENTER_DEADBAND_CM:
                return 0.0, "AUTH_RIGHT_CENTERED"

            y = CENTER_KP * error

            if dist <= AUTHORITY_HARD_CM:
                y = min(y, -AUTHORITY_HARD_STRAFE_MPS)
                mode = "AUTH_RIGHT_HARD"
            else:
                mode = "AUTH_RIGHT_CENTER"

            return (
                clamp(y, -AUTHORITY_HARD_STRAFE_MPS, MAX_CENTER_STRAFE_MPS),
                mode
            )

        return 0.0, "NO_SHARP_AUTHORITY"

    def move_one_cell(self):
        """
        Continuous corridor motion:
          - forward drive
          - Sharp wall-follow / nearest-wall priority
          - yaw hold
          - front ToF collision stop
          - stop after CELL_LENGTH_M odometry displacement
        """

        self.gimbal_front_down()

        start_pos = self.wait_for_position()
        if start_pos is None:
            print("[MOVE ERROR] no chassis position telemetry.")
            return False

        # NEVER capture a new target from current_yaw() here.
        # Current yaw may already have been disturbed by the gimbal.
        target_yaw = self.yaw_ref_deg

        # Prime Sharp filter before selecting a wall.
        for _ in range(SHARP_FILTER_SAMPLES):
            left_cm, right_cm, _, _ = self.read_sharp_cm()
            time.sleep(0.02)

        # Acquire one-and-only-one Sharp lateral-control master.
        authority = self.update_sharp_authority(left_cm, right_cm)

        print(
            f"[MOVE] start authority={authority} "
            f"center_target={CENTER_TARGET_CM:.1f}cm "
            f"target_yaw={target_yaw}"
        )

        t0 = time.monotonic()
        last_debug = 0.0

        while self.running:
            now_t = time.monotonic()

            pos = self.state.get_position()
            if pos is None:
                self.stop_chassis()
                print("[MOVE ERROR] position telemetry lost.")
                return False

            dx = pos[0] - start_pos[0]
            dy = pos[1] - start_pos[1]
            traveled = math.hypot(dx, dy)

            if traveled >= CELL_LENGTH_M:
                self.stop_chassis()
                print(f"[MOVE OK] reached cell: {traveled:.3f} m")
                return True

            if now_t - t0 > MAX_CELL_TIME_SEC:
                self.stop_chassis()

                if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                    print(
                        f"[MOVE WARN] timeout but close enough: "
                        f"{traveled:.3f} m -> accept"
                    )
                    return True

                print(
                    f"[MOVE FAIL] timeout: "
                    f"{traveled:.3f}/{CELL_LENGTH_M:.3f} m"
                )
                return False

            # ------------------------------------------------
            # Digital IR supervisor.
            # ------------------------------------------------
            ir_l_low, ir_r_low, ir_l_raw, ir_r_raw = self.read_ir_filtered(
                samples=1
            )

            dual_event, dual_reason = self.consume_ir_dual_sequence()

            both_override = (
                time.monotonic() < self.ir_both_front_override_until
            )

            # A fresh LEFT->RIGHT or RIGHT->LEFT sequence is a higher-priority
            # event than the short FRONT override.  Stop and rescan the route.
            if dual_event:
                self.stop_chassis()
                recovery_t0 = time.monotonic()

                print(
                    f"[IR SEQUENCE MOVE] {dual_reason} "
                    "-> STOP + GIMBAL ROUTE SCAN"
                )

                action, route_scan = self.scan_route_for_both_ir(
                    context="MOVE/SEQUENCE"
                )

                t0 += time.monotonic() - recovery_t0

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False

                    print(
                        "[IR SEQUENCE MOVE] FRONT clear -> continue; "
                        "Sharp authority keeps centering"
                    )
                    continue

                if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                    print(
                        f"[IR SEQUENCE MOVE] FRONT blocked, hint={action}, "
                        f"near node -> accept node early"
                    )
                    return True

                print(
                    f"[IR SEQUENCE MOVE] FRONT blocked, hint={action} "
                    "before node -> stop edge"
                )
                return False

            # BOTH LOW is NOT two competing recovery commands.
            if ir_l_low and ir_r_low and not both_override:
                self.stop_chassis()
                recovery_t0 = time.monotonic()

                action, route_scan = self.scan_route_for_both_ir(
                    context="MOVE"
                )

                t0 += time.monotonic() - recovery_t0

                if action == "FRONT":
                    self.ir_both_front_override_until = (
                        time.monotonic() + IR_BOTH_FRONT_OVERRIDE_SEC
                    )
                    self.ir_replan_requested = False

                    print(
                        "[IR BOTH MOVE] FRONT clear -> continue straight; "
                        "Sharp authority keeps centering"
                    )
                    continue

                # If this is already close enough to the next cell/node,
                # accept the arrival early. The normal cell scan will then
                # classify LEFT/FRONT/RIGHT and DFS can choose the branch.
                if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                    print(
                        f"[IR BOTH MOVE] FRONT blocked, hint={action}, "
                        f"but traveled={traveled:.3f}m -> accept node early"
                    )
                    return True

                print(
                    f"[IR BOTH MOVE] FRONT blocked before node "
                    f"(hint={action}, d={traveled:.3f}m) -> stop edge"
                )
                return False

            # ONE LOW keeps the unambiguous opposite-slide recovery.
            if (ir_l_low ^ ir_r_low) and not both_override:
                self.stop_chassis()

                recovery_t0 = time.monotonic()
                ok = self.ir_clearance_recovery(context="MOVE")
                t0 += time.monotonic() - recovery_t0

                if not ok:
                    raise RuntimeError(
                        "Single-IR clearance recovery failed during motion."
                    )

                # Recovery can become BOTH LOW and ask for a DFS replan.
                if self.ir_replan_requested:
                    if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                        print(
                            "[IR MOVE] recovery requested replan near node "
                            "-> accept node early"
                        )
                        return True

                    print(
                        "[IR MOVE] recovery requested replan before node "
                        "-> stop edge"
                    )
                    return False

                continue

            tof_mm = self.state.get_tof()

            if tof_mm is not None and tof_mm <= FRONT_HARD_STOP_MM:
                self.stop_chassis()

                if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                    print(
                        f"[MOVE WARN] front wall at {tof_mm:.0f} mm, "
                        f"but traveled {traveled:.3f} m -> accept cell"
                    )
                    return True

                print(
                    f"[MOVE BLOCKED] front ToF={tof_mm:.0f} mm at "
                    f"{traveled:.3f} m"
                )
                return False

            left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

            # Arbitration may transfer ownership, but never blends both
            # sensors into the y command.
            authority = self.update_sharp_authority(left_cm, right_cm)

            y_cmd, side_mode = self.corridor_lateral_command(
                left_cm,
                right_cm,
                authority
            )

            z_cmd = self.yaw_hold_command(target_yaw)

            # Closed-loop longitudinal approach from chassis odometry.
            # Position feedback now affects x-speed before the endpoint,
            # instead of only being used as a final stop threshold.
            remaining_m = max(0.0, CELL_LENGTH_M - traveled)

            if remaining_m <= CELL_APPROACH_SLOW_M:
                ratio = remaining_m / max(CELL_APPROACH_SLOW_M, 1e-6)
                x_cmd = CELL_APPROACH_MIN_MPS + (
                    FORWARD_SPEED_MPS - CELL_APPROACH_MIN_MPS
                ) * ratio
            else:
                x_cmd = FORWARD_SPEED_MPS

            if tof_mm is not None and tof_mm < FRONT_SLOW_MM:
                x_cmd = min(x_cmd, SLOW_FORWARD_SPEED_MPS)

            # Slow down while the active Sharp master is in hard-close mode.
            if side_mode.endswith("_HARD"):
                x_cmd = min(x_cmd, SLOW_FORWARD_SPEED_MPS)

            self.chassis.drive_speed(
                x=x_cmd,
                y=y_cmd,
                z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT
            )

            if now_t - last_debug >= DEBUG_MOVE_PRINT_PERIOD_SEC:
                ltxt = "far" if left_cm is None else f"{left_cm:4.1f}"
                rtxt = "far" if right_cm is None else f"{right_cm:4.1f}"
                ttxt = "None" if tof_mm is None else f"{tof_mm:4.0f}"

                yaw_now = self.current_yaw()
                yaw_err = self.yaw_error_deg(target_yaw)
                yaw_txt = "None" if yaw_now is None else f"{yaw_now:+.2f}"
                err_txt = "None" if yaw_err is None else f"{yaw_err:+.2f}"

                print(
                    f"[CTRL] d={traveled:5.3f}m "
                    f"L={ltxt}cm({left_adc}) "
                    f"R={rtxt}cm({right_adc}) "
                    f"ToF={ttxt}mm "
                    f"auth={authority or '-':<5} "
                    f"mode={side_mode:<20} "
                    f"yaw={yaw_txt} err={err_txt} "
                    f"slip={int(self.chassis_slip_detected())} "
                    f"x={x_cmd:+.2f} y={y_cmd:+.2f} z={z_cmd:+.1f}"
                )

                last_debug = now_t

            time.sleep(CONTROL_DT)

        self.stop_chassis()
        return False

    # --------------------------------------------------------
    # TURNING
    # --------------------------------------------------------

    def turn_closed_loop(self, target_yaw, timeout_sec):
        """
        Rotate chassis to an absolute yaw target using attitude feedback.

        This deliberately avoids:
            chassis.move(...).wait_for_completed()

        because a chassis position action can remain blocked even though the
        robot has physically attempted the turn.

        The loop can NEVER wait forever:
          - attitude feedback closes the yaw loop
          - target must remain inside TURN_TOLERANCE_DEG for TURN_SETTLE_SEC
          - timeout stops the chassis and returns False
        """
        target_yaw = wrap_deg(float(target_yaw))
        dt = 1.0 / TURN_CONTROL_HZ

        self.stop_chassis()

        # Make the turret follow the chassis while rotating.
        # We do not issue gimbal yaw commands in this mode.
        try:
            mode_ok = self.ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
            print(f"[TURN MODE] CHASSIS_LEAD result={mode_ok}")
        except Exception as e:
            print(f"[TURN MODE WARN] CHASSIS_LEAD failed: {e}")

        time.sleep(0.10)

        start_t = time.monotonic()
        in_tolerance_since = None
        last_debug = 0.0

        try:
            while self.running:
                now_t = time.monotonic()

                if now_t - start_t >= timeout_sec:
                    self.stop_chassis()

                    current = self.current_yaw()
                    err = (
                        wrap_deg(target_yaw - current)
                        if current is not None else None
                    )

                    print(
                        f"[TURN TIMEOUT] target={target_yaw:+.2f} "
                        f"actual={current} err={err}"
                    )
                    return False

                current = self.current_yaw()

                if current is None:
                    self.stop_chassis()
                    time.sleep(dt)
                    continue

                error = wrap_deg(target_yaw - current)

                # Target reached: require it to remain stable for a short time
                # so inertia does not immediately throw it out again.
                if abs(error) <= TURN_TOLERANCE_DEG:
                    self.stop_chassis()

                    if in_tolerance_since is None:
                        in_tolerance_since = now_t

                    if now_t - in_tolerance_since >= TURN_SETTLE_SEC:
                        print(
                            f"[TURN OK] target={target_yaw:+.2f} "
                            f"actual={current:+.2f} err={error:+.2f}"
                        )
                        return True

                else:
                    in_tolerance_since = None

                    z_cmd = YAW_DRIVE_SIGN * TURN_KP * error
                    z_cmd = clamp(z_cmd, -TURN_MAX_DPS, TURN_MAX_DPS)

                    # Enough command to overcome static friction near target.
                    if abs(z_cmd) < TURN_MIN_DPS:
                        z_cmd = math.copysign(TURN_MIN_DPS, z_cmd)

                    self.chassis.drive_speed(
                        x=0.0,
                        y=0.0,
                        z=z_cmd,
                        timeout=DRIVE_COMMAND_TIMEOUT
                    )

                if now_t - last_debug >= TURN_DEBUG_PERIOD_SEC:
                    print(
                        f"[TURN CTRL] target={target_yaw:+7.2f} "
                        f"yaw={current:+7.2f} "
                        f"err={error:+7.2f}"
                    )
                    last_debug = now_t

                time.sleep(dt)

        finally:
            self.stop_chassis()

            # FREE is required again because DFS needs independent gimbal scans.
            try:
                mode_ok = self.ep_robot.set_robot_mode(mode=robot.FREE)
                print(f"[TURN MODE] FREE result={mode_ok}")
            except Exception as e:
                print(f"[TURN MODE WARN] FREE failed: {e}")

            time.sleep(0.10)

    def turn_to_direction(self, target_dir):
        target_dir %= 4
        delta = (target_dir - self.heading) % 4

        target_yaw = self.desired_yaw_for_heading(target_dir)

        if target_yaw is None:
            raise RuntimeError("Yaw base reference is not initialized.")

        if delta == 0:
            self.yaw_ref_deg = target_yaw
            self.hold_heading_stationary(0.15)
            self.gimbal_front_down()
            return

        self.stop_chassis()

        if delta == 1:
            label = "RIGHT 90"
            timeout_sec = TURN_TIMEOUT_90_SEC

        elif delta == 3:
            label = "LEFT 90"
            timeout_sec = TURN_TIMEOUT_90_SEC

        else:
            label = "180"
            timeout_sec = TURN_TIMEOUT_180_SEC

        print(
            f"[TURN] {DIR_NAMES[self.heading]} -> "
            f"{DIR_NAMES[target_dir]} : {label} "
            f"target_yaw={target_yaw:+.2f}"
        )

        ok = self.turn_closed_loop(
            target_yaw=target_yaw,
            timeout_sec=timeout_sec
        )

        if not ok:
            # Never continue DFS with an unknown heading.
            raise RuntimeError(
                f"Closed-loop turn failed: "
                f"{DIR_NAMES[self.heading]} -> {DIR_NAMES[target_dir]}. "
                f"Robot stopped safely instead of hanging."
            )

        # Only update the logical DFS orientation AFTER the physical turn
        # has actually reached its attitude target.
        self.heading = target_dir
        self.yaw_ref_deg = target_yaw

        # New corridor geometry after a turn: release the previous Sharp
        # master and reacquire exactly one authority on the next translation.
        self._set_sharp_authority(None, "heading changed")

        print(
            f"[YAW LOCK] logical={DIR_NAMES[self.heading]} "
            f"target={self.yaw_ref_deg:+.2f} "
            f"actual={self.current_yaw()}"
        )

        # Remove the small residual error, then physically re-center the ToF
        # turret to the NEW chassis front and restore pitch -5 deg.
        self.hold_heading_stationary(STATIONARY_SETTLE_SEC)
        self.gimbal_front_down()

        # A turn can leave the chassis between two close walls.
        # ONE LOW -> small opposite nudge.
        # BOTH LOW -> STOP + Gimbal route scan.  If FRONT is blocked the
        # recovery sets ir_replan_requested so DFS can rescan instead of
        # blindly translating.
        if not self.ir_clearance_recovery(context="AFTER_TURN"):
            raise RuntimeError(
                "IR remained unsafe after turn/corner-clearance recovery."
            )

    # --------------------------------------------------------
    # CELL SCAN
    # --------------------------------------------------------

    def scan_cell(self, cell):
        """
        Scan LEFT / FRONT / RIGHT with the gimbal ToF.

        Normal maze classification:
            distance > TOF_OPEN_THRESHOLD_MM -> OPEN
            otherwise                         -> WALL

        Additional close-range dead-end override:
            LEFT <= DEAD_END_THRESHOLD_MM
            AND FRONT <= DEAD_END_THRESHOLD_MM
            AND RIGHT <= DEAD_END_THRESHOLD_MM

        When that close-range condition is true, DFS does not attempt any
        forward/side branch.  It immediately reverses toward the parent cell.
        """
        print(
            f"\n[SCAN] cell={cell} heading={DIR_NAMES[self.heading]}"
        )

        # IR supervisor before the normal cell scan.
        # BOTH LOW must NOT produce two opposite slide commands.  We simply
        # stop and let this cell's gimbal scan decide the available routes.
        pre_l_low, pre_r_low, pre_l_raw, pre_r_raw = self.read_ir_filtered()
        pre_dual_event, pre_dual_reason = self.consume_ir_dual_sequence()

        if pre_dual_event:
            self.stop_chassis()
            print(
                f"[IR SEQUENCE SCAN {cell}] {pre_dual_reason} "
                "-> no slide; Gimbal will classify routes"
            )
        elif pre_l_low and pre_r_low:
            self.stop_chassis()
            print(
                f"[IR BOTH SCAN {cell}] IR_L={pre_l_raw} IR_R={pre_r_raw} "
                "-> no slide; Gimbal will classify routes"
            )
        elif pre_l_low or pre_r_low:
            if not self.ir_clearance_recovery(context=f"SCAN {cell}"):
                raise RuntimeError(
                    f"IR clearance recovery failed before scanning cell {cell}."
                )

        # A cell scan itself is a fresh replan, so consume any old hint.
        self.ir_replan_requested = False
        self.ir_route_hint = None

        relative_scans = [
            ("LEFT",  -90.0, REL_LEFT),
            ("FRONT",   0.0, REL_FRONT),
            ("RIGHT", +90.0, REL_RIGHT),
        ]

        ordered_open_dirs = []
        scan_mm = {}

        for label, yaw_deg, rel_dir in relative_scans:
            before_yaw = self.current_yaw()
            mm = self.scan_tof_at_yaw(yaw_deg)
            after_yaw = self.current_yaw()
            yaw_err = self.yaw_error_deg()

            scan_mm[label] = mm

            print(
                f"  [YAW] before={before_yaw} after={after_yaw} "
                f"target={self.yaw_ref_deg} err={yaw_err}"
            )

            if mm is None:
                is_open = False
                print(f"  {label:<5}: NO DATA -> CLOSED for safety")
            else:
                is_open = mm > TOF_OPEN_THRESHOLD_MM
                state = "OPEN" if is_open else "WALL"

                print(
                    f"  {label:<5}: {mm:7.1f} mm -> {state}"
                )

            abs_dir = (self.heading + rel_dir) % 4

            if is_open:
                ordered_open_dirs.append(abs_dir)

        # Save actual measurements for later debugging.
        self.cell_scan_mm[cell] = dict(scan_mm)

        # ----------------------------------------------------
        # HARD DEAD-END DETECTION
        # ----------------------------------------------------
        all_valid = all(
            scan_mm.get(k) is not None
            for k in ("LEFT", "FRONT", "RIGHT")
        )

        hard_dead_end = (
            all_valid
            and scan_mm["LEFT"] <= DEAD_END_THRESHOLD_MM
            and scan_mm["FRONT"] <= DEAD_END_THRESHOLD_MM
            and scan_mm["RIGHT"] <= DEAD_END_THRESHOLD_MM
        )

        if hard_dead_end:
            self.dead_end_cells.add(cell)

            print(
                "  [DEAD END] CLOSE WALLS ON ALL 3 SIDES "
                f"(threshold={DEAD_END_THRESHOLD_MM:.0f} mm)"
            )
            print(
                f"             L={scan_mm['LEFT']:.0f} "
                f"F={scan_mm['FRONT']:.0f} "
                f"R={scan_mm['RIGHT']:.0f} mm"
            )

            # Any apparent side/front OPEN caused by a bad ToF sample must not
            # be trusted once the explicit dead-end condition has fired.
            ordered_open_dirs = []

        else:
            self.dead_end_cells.discard(cell)

        # Parent direction is a guaranteed known connection because the robot
        # physically came through that edge.
        p = self.parent.get(cell)

        if p is not None:
            back_dir = direction_between(cell, p)

            if back_dir not in ordered_open_dirs:
                ordered_open_dirs.append(back_dir)

        elif not ROOT_BACK_IS_WALL:
            # Optional root back scan if the entrance must also be explored.
            mm = self.scan_tof_at_yaw(180.0)

            if mm is not None and mm > TOF_OPEN_THRESHOLD_MM:
                ordered_open_dirs.append(
                    (self.heading + REL_BACK) % 4
                )

        # Always put the turret physically back on the new chassis front and
        # restore pitch -5 degrees before any chassis motion.
        self.gimbal_front_down()

        print(
            "  open absolute dirs:",
            [DIR_NAMES[d] for d in ordered_open_dirs]
        )

        return ordered_open_dirs

    # --------------------------------------------------------
    # MAP HELPERS / PERSISTENT MAP
    # --------------------------------------------------------

    @staticmethod
    def cell_key(cell):
        return f"{int(cell[0])},{int(cell[1])}"

    @staticmethod
    def parse_cell_key(value):
        x_str, y_str = str(value).split(",", 1)
        return (int(x_str), int(y_str))

    def edge_key(self, a, b):
        return tuple(sorted((a, b)))

    def mark_blocked(self, a, b):
        self.blocked_edges.add(self.edge_key(a, b))

        if MAP_AUTOSAVE:
            self.save_map(final=False)

    def is_blocked(self, a, b):
        return self.edge_key(a, b) in self.blocked_edges

    def mapped_cells(self):
        cells = set(self.visited)
        cells.update(self.open_dirs.keys())
        cells.update(self.cell_scan_mm.keys())
        cells.update(self.dead_end_cells)

        for edge in self.blocked_edges:
            cells.update(edge)

        return cells

    def build_map_payload(self):
        """
        JSON map representation designed to be reusable on future runs.

        Coordinate convention:
            +Y = North
            +X = East

        A known-map run assumes the robot is physically placed at `root`
        and initially faces `start_heading`.
        """
        cells = self.mapped_cells()

        cell_records = {}

        for cell in sorted(cells, key=lambda p: (p[1], p[0])):
            dirs = list(self.open_dirs.get(cell, []))

            cell_records[self.cell_key(cell)] = {
                "x": int(cell[0]),
                "y": int(cell[1]),
                "visited": cell in self.visited,
                "dead_end": cell in self.dead_end_cells,
                "open_dirs": [DIR_NAMES[d] for d in dirs],
                "open_dir_indices": [int(d) for d in dirs],
                "open_neighbors": [
                    [int(v) for v in neighbor(cell, d)]
                    for d in dirs
                    if not self.is_blocked(cell, neighbor(cell, d))
                ],
                "tof_scan_mm": self.cell_scan_mm.get(cell),
            }

        blocked = []

        for a, b in sorted(self.blocked_edges):
            blocked.append([
                [int(a[0]), int(a[1])],
                [int(b[0]), int(b[1])],
            ])

        payload = {
            "schema": MAP_SCHEMA,
            "schema_version": MAP_SCHEMA_VERSION,
            "created_at": self.map_created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "complete": bool(self.map_complete),
            "root": [int(self.root[0]), int(self.root[1])],
            "start_heading": "N",
            "start_heading_index": 0,
            "coordinate_system": {
                "N": [0, 1],
                "E": [1, 0],
                "S": [0, -1],
                "W": [-1, 0],
            },
            "geometry": {
                "cell_length_m": CELL_LENGTH_M,
                "cell_success_fraction": CELL_SUCCESS_FRACTION,
            },
            "sensor_policy": {
                "tof_open_threshold_mm": TOF_OPEN_THRESHOLD_MM,
                "dead_end_threshold_mm": DEAD_END_THRESHOLD_MM,
                "tof_scan_samples": TOF_SCAN_SAMPLES,
                "gimbal_pitch_deg": GIMBAL_PITCH_DEG,
                "sharp_center_target_cm": CENTER_TARGET_CM,
            },
            "visited_cells": [
                [int(c[0]), int(c[1])]
                for c in sorted(self.visited, key=lambda p: (p[1], p[0]))
            ],
            "dead_end_cells": [
                [int(c[0]), int(c[1])]
                for c in sorted(self.dead_end_cells)
            ],
            "blocked_edges": blocked,
            "cells": cell_records,
            "usage_note": (
                "Known-map mode assumes the robot starts at the same physical "
                "root position and same North-facing orientation used during "
                "mapping. Safety sensors remain active during replay."
            ),
        }

        return payload

    def render_ascii_map(self):
        """
        Human-readable topological map.

        Legend:
            S = root/start
            D = hard dead end
            o = mapped cell
            ? = mapped record not physically visited
        """
        cells = self.mapped_cells()

        if not cells:
            return "(map empty)\n"

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        width = (max_x - min_x) * 4 + 3
        height = (max_y - min_y) * 2 + 1

        canvas = [[" " for _ in range(width)] for _ in range(height)]

        def xy_to_rc(cell):
            x, y = cell
            col = (x - min_x) * 4 + 1
            row = (max_y - y) * 2
            return row, col

        for cell in cells:
            row, col = xy_to_rc(cell)

            if cell == self.root:
                ch = "S"
            elif cell in self.dead_end_cells:
                ch = "D"
            elif cell in self.visited:
                ch = "o"
            else:
                ch = "?"

            canvas[row][col] = ch

        # Draw only trusted non-blocked links between mapped cells.
        for cell in cells:
            for d in self.open_dirs.get(cell, []):
                nb = neighbor(cell, d)

                if nb not in cells or self.is_blocked(cell, nb):
                    continue

                r1, c1 = xy_to_rc(cell)
                r2, c2 = xy_to_rc(nb)

                if r1 == r2:
                    lo, hi = sorted((c1, c2))
                    for c in range(lo + 1, hi):
                        canvas[r1][c] = "-"
                elif c1 == c2:
                    lo, hi = sorted((r1, r2))
                    for r in range(lo + 1, hi):
                        canvas[r][c1] = "|"

        lines = [
            "RoboMaster DFS Persistent Map",
            "N = up, E = right",
            "Legend: S=start, o=mapped, D=dead-end, ?=known/unvisited",
            "",
        ]
        lines.extend("".join(row).rstrip() for row in canvas)
        lines.append("")

        return "\n".join(lines)

    def save_map(self, final=False):
        """
        Save reusable JSON + human-readable ASCII map.

        latest_map.* is overwritten intentionally.
        A timestamped snapshot is also created when final=True.
        """
        if not self.mapped_cells():
            return None

        MAP_DIR.mkdir(parents=True, exist_ok=True)

        payload = self.build_map_payload()

        # Atomic-ish replacement so a power/program interruption is less
        # likely to leave a half-written latest_map.json.
        tmp_json = MAP_LATEST_JSON.with_suffix(".json.tmp")

        with tmp_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        os.replace(tmp_json, MAP_LATEST_JSON)

        ascii_text = self.render_ascii_map()

        tmp_txt = MAP_LATEST_ASCII.with_suffix(".txt.tmp")
        tmp_txt.write_text(ascii_text, encoding="utf-8")
        os.replace(tmp_txt, MAP_LATEST_ASCII)

        snapshot = None

        if final:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = MAP_DIR / f"maze_{stamp}.json"
            snapshot.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            ascii_snapshot = MAP_DIR / f"maze_{stamp}.txt"
            ascii_snapshot.write_text(ascii_text, encoding="utf-8")

        print(
            f"[MAP SAVE] cells={len(payload['cells'])} "
            f"complete={payload['complete']} -> {MAP_LATEST_JSON}"
        )

        if snapshot is not None:
            print(f"[MAP SNAPSHOT] {snapshot}")

        return MAP_LATEST_JSON

    def load_map(self, map_path):
        """
        Load a previously learned grid topology.

        This restores topology only.  Absolute chassis yaw is intentionally
        NOT restored because the robot gets a fresh startup yaw reference on
        every physical run.
        """
        path = Path(map_path)

        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("schema") != MAP_SCHEMA:
            raise ValueError(
                f"Unsupported map schema: {data.get('schema')!r}"
            )

        if int(data.get("schema_version", -1)) != MAP_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported map version: {data.get('schema_version')}"
            )

        root = data.get("root", [0, 0])
        self.root = (int(root[0]), int(root[1]))
        self.current = self.root

        self.open_dirs = {}
        self.cell_scan_mm = {}
        self.dead_end_cells = set()
        self.blocked_edges = set()
        self.known_map_cells = set()

        cells_data = data.get("cells", {})

        for key, rec in cells_data.items():
            cell = (int(rec["x"]), int(rec["y"]))
            self.known_map_cells.add(cell)

            dirs = rec.get("open_dir_indices")

            if dirs is None:
                dirs = [
                    DIR_NAMES.index(name)
                    for name in rec.get("open_dirs", [])
                ]

            self.open_dirs[cell] = [int(d) % 4 for d in dirs]

            scan = rec.get("tof_scan_mm")
            if scan is not None:
                self.cell_scan_mm[cell] = scan

            if rec.get("dead_end", False):
                self.dead_end_cells.add(cell)

        for edge in data.get("blocked_edges", []):
            if len(edge) != 2:
                continue

            a = (int(edge[0][0]), int(edge[0][1]))
            b = (int(edge[1][0]), int(edge[1][1]))
            self.blocked_edges.add(self.edge_key(a, b))

        self.map_created_at = data.get(
            "created_at",
            datetime.now().isoformat(timespec="seconds")
        )
        self.map_complete = bool(data.get("complete", False))
        self.loaded_map_path = path

        # Keep loaded map's cell length warning visible, but do not silently
        # mutate the runtime constant.
        saved_cell_length = (
            data.get("geometry", {}).get("cell_length_m")
        )

        print("\n================ MAP LOADED ================")
        print(f"File        : {path}")
        print(f"Cells       : {len(self.known_map_cells)}")
        print(f"Complete    : {self.map_complete}")
        print(f"Root        : {self.root}")
        print(f"Start facing: {data.get('start_heading', 'N')}")
        print(f"Saved cell  : {saved_cell_length} m")
        print(f"Runtime cell: {CELL_LENGTH_M} m")
        print("============================================")

        return data

    def known_neighbors(self, cell):
        """
        Trusted neighbors from the saved graph.

        An edge is usable only if:
          - the direction was saved as open
          - the destination is a known mapped cell
          - the edge is not marked blocked
        """
        result = []

        for d in self.open_dirs.get(cell, []):
            nb = neighbor(cell, d)

            if nb not in self.known_map_cells:
                continue

            if self.is_blocked(cell, nb):
                continue

            result.append((d, nb))

        return result

    def shortest_known_path(self, start, goal):
        """
        BFS shortest path over the known unweighted grid graph.
        Returns a list of cells including start and goal.
        """
        if start not in self.known_map_cells:
            raise ValueError(f"Start cell not in map: {start}")

        if goal not in self.known_map_cells:
            raise ValueError(f"Goal cell not in map: {goal}")

        q = deque([start])
        came_from = {start: None}

        while q:
            cell = q.popleft()

            if cell == goal:
                break

            for _, nb in self.known_neighbors(cell):
                if nb not in came_from:
                    came_from[nb] = cell
                    q.append(nb)

        if goal not in came_from:
            raise RuntimeError(
                f"No known path from {start} to {goal}"
            )

        path = []
        cur = goal

        while cur is not None:
            path.append(cur)
            cur = came_from[cur]

        path.reverse()
        return path

    def execute_known_path(self, path):
        """
        Follow a saved path while retaining all real-time safety layers:
        yaw lock, Sharp corridor authority, IR interlocks and front ToF.
        """
        if not path:
            return

        self.current = path[0]

        for target in path[1:]:
            current = self.current
            d = direction_between(current, target)

            print(
                f"\n[KNOWN] {current} -> {target} "
                f"dir={DIR_NAMES[d]}"
            )

            self.turn_to_direction(d)

            if self.ir_replan_requested:
                self.ir_replan_requested = False
                self.stop_chassis()
                raise RuntimeError(
                    "Saved map route disagrees with current IR/Gimbal "
                    f"observation before {current}->{target}. "
                    "Stopping instead of trusting stale topology."
                )

            ok = self.move_one_cell()

            if not ok:
                self.stop_chassis()
                raise RuntimeError(
                    f"Known-map motion failed at edge {current}->{target}. "
                    "The environment may have changed."
                )

            self.current = target

        print(f"\n[KNOWN] reached {self.current}")

    def run_known_map(self, goal=None):
        """
        Reuse an already learned map.

        If goal is provided:
            compute BFS shortest path from root to goal and run it.

        If goal is None:
            perform a full coverage replay of all reachable known cells
            WITHOUT re-scanning topology with the gimbal at every node.
        """
        if not self.known_map_cells:
            raise RuntimeError("No map loaded.")

        self.heading = 0
        self.current = self.root

        if goal is not None:
            path = self.shortest_known_path(self.root, goal)

            print("\n================ KNOWN MAP ROUTE ================")
            print(f"Start: {self.root}")
            print(f"Goal : {goal}")
            print(f"Cells: {len(path)}")
            print("Path :", path)
            print("=================================================")

            self.execute_known_path(path)
            return

        print("\n[KNOWN] FULL MAP REPLAY/COVERAGE")
        print("[KNOWN] topology scans are skipped; safety sensors remain active")

        seen = {self.root}
        stack = [(self.root, 0)]

        while self.running and stack:
            cell, next_index = stack[-1]
            neighbors = self.known_neighbors(cell)

            # Find next unvisited known neighbor.
            chosen = None

            while next_index < len(neighbors):
                d, nb = neighbors[next_index]
                next_index += 1
                stack[-1] = (cell, next_index)

                if nb not in seen:
                    chosen = (d, nb)
                    break

            if chosen is not None:
                d, nb = chosen

                print(
                    f"\n[KNOWN] VISIT {cell} -> {nb} "
                    f"dir={DIR_NAMES[d]}"
                )

                self.turn_to_direction(d)

                if self.ir_replan_requested:
                    self.ir_replan_requested = False
                    raise RuntimeError(
                        f"Current sensors disagree with saved edge {cell}->{nb}"
                    )

                if not self.move_one_cell():
                    raise RuntimeError(
                        f"Could not traverse saved edge {cell}->{nb}"
                    )

                self.current = nb
                seen.add(nb)
                stack.append((nb, 0))
                continue

            # Finished this node: go back to DFS parent in replay stack.
            if len(stack) == 1:
                break

            child = stack.pop()[0]
            parent = stack[-1][0]
            d = direction_between(child, parent)

            print(
                f"\n[KNOWN] BACKTRACK {child} -> {parent} "
                f"dir={DIR_NAMES[d]}"
            )

            self.turn_to_direction(d)

            if self.ir_replan_requested:
                self.ir_replan_requested = False
                raise RuntimeError(
                    f"Current sensors disagree with saved edge {child}->{parent}"
                )

            if not self.move_one_cell():
                raise RuntimeError(
                    f"Could not backtrack saved edge {child}->{parent}"
                )

            self.current = parent

        print(
            f"\n[KNOWN] replay complete: "
            f"{len(seen)}/{len(self.known_map_cells)} cells reached"
        )

    def print_map_summary(self):
        print("\n================ DFS MAP SUMMARY ================")
        print(f"Visited cells: {len(self.visited)}")
        print("Cells:", sorted(self.visited, key=lambda p: (p[1], p[0])))

        for cell in sorted(self.open_dirs, key=lambda p: (p[1], p[0])):
            dirs = [DIR_NAMES[d] for d in self.open_dirs[cell]]
            print(f"  {cell}: open={dirs}")

        if self.dead_end_cells:
            print("Hard dead-end cells:")
            for cell in sorted(self.dead_end_cells):
                scan = self.cell_scan_mm.get(cell, {})
                print(
                    f"  {cell}: "
                    f"L={scan.get('LEFT')} "
                    f"F={scan.get('FRONT')} "
                    f"R={scan.get('RIGHT')} mm"
                )

        if self.blocked_edges:
            print("Blocked / failed edges:")
            for edge in sorted(self.blocked_edges):
                print(" ", edge)

        print("=================================================")
        print(self.render_ascii_map())

    # --------------------------------------------------------
    # DFS
    # --------------------------------------------------------

    def run_dfs(self):
        self.visited = {self.root}
        self.parent = {self.root: None}
        self.current = self.root

        stack = [self.root]

        print("\n[DFS] START")
        print(f"[DFS] root={self.root}, heading={DIR_NAMES[self.heading]}")

        while self.running and stack:
            cell = stack[-1]
            self.current = cell

            if cell not in self.open_dirs:
                self.open_dirs[cell] = self.scan_cell(cell)

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

            # ------------------------------------------------
            # HARD DEAD-END OVERRIDE
            # ------------------------------------------------
            # LEFT + FRONT + RIGHT are all <= DEAD_END_THRESHOLD_MM.
            # Do not spend another DFS decision cycle here: reverse 180 deg
            # and immediately go back through the edge we entered from.
            if cell in self.dead_end_cells:
                parent = self.parent.get(cell)

                if parent is None:
                    # At the root there is no mapped parent cell.  Still obey
                    # the requested dead-end behavior by turning around, then
                    # stop the exploration safely at the entrance.
                    reverse_dir = (self.heading + 2) % 4

                    print(
                        f"\n[DEAD END] root {cell}: "
                        f"LEFT/FRONT/RIGHT <= {DEAD_END_THRESHOLD_MM:.0f} mm"
                    )
                    print(
                        f"[DEAD END] TURN 180 "
                        f"{DIR_NAMES[self.heading]} -> {DIR_NAMES[reverse_dir]}"
                    )

                    self.turn_to_direction(reverse_dir)
                    print("[DFS] root is boxed in; exploration complete.")
                    break

                back_dir = direction_between(cell, parent)

                print(
                    f"\n[DEAD END] {cell}: "
                    f"LEFT/FRONT/RIGHT <= {DEAD_END_THRESHOLD_MM:.0f} mm"
                )
                print(
                    f"[DEAD END] TURN 180 + BACKTRACK "
                    f"{cell} -> {parent} dir={DIR_NAMES[back_dir]}"
                )

                # Because the robot entered this cell facing away from parent,
                # back_dir should normally be a 180-degree logical turn.
                self.turn_to_direction(back_dir)

                ok = self.move_one_cell()

                if not ok:
                    self.stop_chassis()
                    raise RuntimeError(
                        f"Dead-end backtrack failed: {cell} -> {parent}."
                    )

                stack.pop()
                self.current = parent

                print(f"[DEAD END] escaped; back at {parent}")

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                continue

            # Classic DFS:
            # choose the first open neighbor that has not been visited.
            next_dir = None
            next_cell = None

            for d in self.open_dirs[cell]:
                nb = neighbor(cell, d)

                if self.is_blocked(cell, nb):
                    continue

                if nb not in self.visited:
                    next_dir = d
                    next_cell = nb
                    break

            if next_cell is not None:
                print(
                    f"\n[DFS] EXPLORE {cell} -> {next_cell} "
                    f"dir={DIR_NAMES[next_dir]}"
                )

                self.turn_to_direction(next_dir)

                # BOTH-IR after the turn may reveal that the intended front
                # direction is actually blocked while another branch is open.
                # Stay on the SAME logical cell and rescan/replan.
                if self.ir_replan_requested:
                    print(
                        f"[DFS] IR/Gimbal requests REPLAN at {cell} "
                        f"hint={self.ir_route_hint}"
                    )
                    self.ir_replan_requested = False
                    self.open_dirs.pop(cell, None)
                    continue

                ok = self.move_one_cell()

                if not ok:
                    print(
                        f"[DFS] edge {cell}->{next_cell} failed; "
                        f"mark blocked and continue."
                    )
                    self.mark_blocked(cell, next_cell)

                    # Remove this false-positive opening from the cell map.
                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if neighbor(cell, d) != next_cell
                    ]

                    if MAP_AUTOSAVE:
                        self.save_map(final=False)

                    continue

                self.parent[next_cell] = cell
                self.visited.add(next_cell)
                stack.append(next_cell)

                print(
                    f"[DFS] ARRIVED {next_cell}; "
                    f"visited={len(self.visited)}"
                )

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                continue

            # No unvisited open neighbor -> backtrack.
            parent = self.parent.get(cell)

            if parent is None:
                print("\n[DFS] Root has no unvisited neighbors.")
                print("[DFS] COMPLETE.")
                break

            back_dir = direction_between(cell, parent)

            print(
                f"\n[DFS] BACKTRACK {cell} -> {parent} "
                f"dir={DIR_NAMES[back_dir]}"
            )

            self.turn_to_direction(back_dir)

            if self.ir_replan_requested:
                self.ir_replan_requested = False
                self.stop_chassis()
                raise RuntimeError(
                    "IR/Gimbal says the DFS parent/backtrack direction is "
                    f"blocked at {cell}; topology cannot be trusted."
                )

            ok = self.move_one_cell()

            if not ok:
                self.stop_chassis()
                raise RuntimeError(
                    f"Backtrack failed: {cell} -> {parent}. "
                    f"Stopping because DFS topology is no longer reliable."
                )

            stack.pop()
            self.current = parent

            print(f"[DFS] back at {parent}")

            if MAP_AUTOSAVE:
                self.save_map(final=False)

        # Reaching here without an exception means DFS finished normally.
        self.map_complete = True

        self.stop_chassis()
        self.gimbal_front_down()
        self.print_map_summary()
        self.save_map(final=True)


# ============================================================
# MAIN / CLI
# ============================================================

def parse_goal(text):
    if text is None:
        return None

    parts = str(text).split(",")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "goal must be x,y, for example --goal 2,4"
        )

    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "goal coordinates must be integers"
        ) from e


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "RoboMaster DFS explorer with persistent reusable grid maps."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("explore", "known", "auto"),
        default="explore",
        help=(
            "explore: learn/save a map; "
            "known: load saved map; "
            "auto: use saved map if present, otherwise explore"
        ),
    )

    parser.add_argument(
        "--map",
        dest="map_path",
        default=str(MAP_LATEST_JSON),
        help="map JSON used by known/auto mode",
    )

    parser.add_argument(
        "--goal",
        type=parse_goal,
        default=None,
        help=(
            "known-map target cell x,y. "
            "If omitted, replay/cover the full saved map."
        ),
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    explorer = DFSMazeExplorer()

    try:
        mode = args.mode

        if mode == "auto":
            if Path(args.map_path).exists():
                mode = "known"
                print(
                    f"[MODE AUTO] found {args.map_path} -> KNOWN MAP"
                )
            else:
                mode = "explore"
                print(
                    f"[MODE AUTO] no map at {args.map_path} -> EXPLORE"
                )

        # Load topology before connecting; physical yaw reference is still
        # captured fresh during connect().
        if mode == "known":
            explorer.load_map(args.map_path)

        explorer.connect()

        if mode == "explore":
            explorer.run_dfs()
        else:
            explorer.run_known_map(goal=args.goal)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
