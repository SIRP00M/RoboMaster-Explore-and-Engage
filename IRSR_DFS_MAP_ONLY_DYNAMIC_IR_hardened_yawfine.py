#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboMaster EP - DFS MAP-ONLY / PID RUNTIME-ZERO
================================================
Purpose
-------
Keep ONLY the machinery required to explore an unknown grid maze with DFS and
build a persistent map.  Target vision, firing, radar sweeps, side-radar anchor
moves, attack excursions, known-map replay, and operator-exit workflows are
intentionally removed.

Runtime policy
--------------
* RUN pose is logical cell (0, 0); chassis FRONT at RUN is logical North.
* Gimbal ToF scans LEFT / FRONT / RIGHT at each newly visited cell.
* Root BACK is checked once by a closed-loop 180-degree chassis scan, then the
  chassis is restored to runtime North.
* Translation uses front ToF, Sharp side-wall centering, digital IR emergency
  supervision, chassis odometry, and runtime-zero PID yaw hold.
* A close front wall is NOT treated as an immediate motion failure.  The chassis
  brakes progressively, stops, settles, and performs a stationary Gimbal-ToF
  LEFT/FRONT/RIGHT topology probe.  This lets DFS recognize a dead-end/corner
  node before deciding whether to accept the node or return to the source cell.
* A recoverable motion/sensor failure never raises RuntimeError.  The robot
  stops, retries/reacquires, retreats to the source cell when possible, and
  defers the affected edge.
* If the physical pose cannot be proven, autonomous motion stops and a partial
  map is saved.  This is deliberate: continuing from an unknown pose corrupts
  the DFS map and is less safe than a controlled stop.
* Map files are autosaved atomically after topology changes.

Outputs
-------
  maps/latest_map.json
  maps/latest_map.txt
  maps/latest_map.svg

Important tuning
----------------
CELL_LENGTH_M and TOF_OPEN_THRESHOLD_MM must match the real field.
"""

from robomaster import robot

import json
import math
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

IR_LEFT_ID = 1       # digital, ACTIVE LOW
SHARP_LEFT_ID = 2    # GP2Y0A41SK0F ADC
SHARP_RIGHT_ID = 3   # GP2Y0A41SK0F ADC
IR_RIGHT_ID = 4      # digital, ACTIVE LOW

TOF_INDEX = 0
TOF_FREQ_HZ = 20
POSITION_FREQ_HZ = 20
ATTITUDE_FREQ_HZ = 50
GIMBAL_ANGLE_FREQ_HZ = 20
STATUS_FREQ_HZ = 5


# ============================================================
# FIELD / DFS GEOMETRY
# ============================================================
GRID_TILE_M = 0.60
CELL_LENGTH_M = 0.60
CELL_SUCCESS_FRACTION = 0.82
TOF_OPEN_THRESHOLD_MM = 600.0

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

# DFS tries directions in the order measured from the arrival heading.
SCAN_RELATIVE_ORDER = (
    ("LEFT", -90.0, REL_LEFT),
    ("FRONT", 0.0, REL_FRONT),
    ("RIGHT", +90.0, REL_RIGHT),
)


# ============================================================
# SHARP CALIBRATION
# ============================================================
LEFT_CAL = [
    (4.0, 864.5), (6.0, 610.0), (8.0, 475.0), (10.0, 378.0),
    (12.0, 316.0), (14.0, 274.0), (16.0, 247.0), (18.0, 224.0),
    (20.0, 212.0), (22.0, 198.0), (24.0, 182.0),
]
RIGHT_CAL = [
    (4.0, 822.0), (6.0, 589.0), (8.0, 455.0), (10.0, 374.0),
    (12.0, 313.0), (14.0, 276.0), (16.0, 239.0), (18.0, 205.0),
    (20.0, 174.0), (22.0, 161.0), (24.0, 133.0),
]
SHARP_FILTER_SAMPLES = 5
SHARP_MIN_PLAUSIBLE_ADC = 20.0
SHARP_SENSOR_FLOOR_CM = 4.0
SHARP_EMERGENCY_CM = 4.5

# One-side-at-a-time centering.  Never let both Sharp sensors fight over y.
CENTER_TARGET_CM = 13.0
SHARP_FOLLOW_NEAR_CM = 11.5
SHARP_FOLLOW_FAR_CM = 15.0
SHARP_FOLLOW_KP = 0.030
SHARP_FOLLOW_MIN_STRAFE_MPS = 0.025
SHARP_MAX_STRAFE_MPS = 0.055
# While translating, a hard side correction must never dominate forward speed.
# Stronger close-wall recovery is performed as a pure sideways correction (x=0)
# rather than combining forward + strafe into a diagonal vector.
SHARP_HARD_STRAFE_MPS = 0.080
SHARP_AUTHORITY_DANGER_CM = 10.0
SHARP_AUTHORITY_RELEASE_CM = 21.0
SHARP_AUTHORITY_HOLD_SEC = 0.75

# Forward/strafe mixer.  The old code could still request y=0.07..0.11 m/s
# while x was slowing to 0.07..0.10 m/s near a node; that makes the chassis
# visibly approach the stop diagonally.  Fade wall-follow out before the node
# and cap y as a fraction of forward velocity.
LATERAL_FADE_START_M = 0.24
# Final 10 cm is a straight-only braking lane: normal Sharp centering is zero.
# Emergency side clearance still has its own pure-strafe override below.
LATERAL_ZERO_REMAIN_M = 0.10
LATERAL_MAX_RATIO_CRUISE = 0.30
LATERAL_MAX_RATIO_APPROACH = 0.15
LATERAL_SLEW_MPS2 = 0.30

# If a Sharp sensor is genuinely too close, do not continue diagonally.
# Stop forward motion and translate purely away from the wall until the sensor
# leaves the danger zone.  Forward progress is projection-based, so this
# sideways escape is not counted as progress through the 60 cm cell.
SHARP_SIDE_ESCAPE_TRIGGER_CM = 7.0
SHARP_SIDE_ESCAPE_CLEAR_CM = 9.0
SHARP_SIDE_ESCAPE_SPEED_MPS = 0.055
SHARP_SIDE_ESCAPE_MAX_SEC = 0.80


# ============================================================
# MOTION / SAFETY
# ============================================================
FORWARD_SPEED_MPS = 0.16
SLOW_FORWARD_SPEED_MPS = 0.10
CELL_APPROACH_SLOW_M = 0.10
CELL_APPROACH_MIN_MPS = 0.07

# Front-ToF approach/brake policy.  Do not jump directly from forward motion to
# reverse when a wall appears.  Slow progressively, stop, then let the gimbal
# inspect LEFT/FRONT/RIGHT because the wall can be the end of a valid DFS cell.
FRONT_BRAKE_START_MM = 430.0
FRONT_CRAWL_START_MM = 250.0
FRONT_STOP_SCAN_MM = 165.0
FRONT_SLOW_MM = FRONT_BRAKE_START_MM   # compatibility name used in logs/tuning
FRONT_HARD_STOP_MM = FRONT_STOP_SCAN_MM
FRONT_MIN_BRAKE_SPEED_MPS = 0.045
FRONT_NODE_SCAN_SETTLE_SEC = 0.18
FRONT_NODE_CAPTURE_MIN_PROGRESS_M = 0.34
FRONT_NODE_CAPTURE_MIN_VALID_RAYS = 2
# If the stop happens earlier than this, the edge is treated as a mid-edge
# obstruction unless a later retry proves otherwise.
FRONT_MIDEDGE_MAX_PROGRESS_M = 0.30
FRONT_TOF_STALE_SEC = 0.70

CONTROL_DT = 0.05
DRIVE_COMMAND_TIMEOUT = 0.25
POSITION_STALE_SEC = 0.60
ATTITUDE_STALE_SEC = 0.40
MAX_CELL_TIME_SEC = max(6.0, (CELL_LENGTH_M / FORWARD_SPEED_MPS) * 2.8)

# If translation really must abort, return to the source smoothly.  Retreat is
# projection-based (forward + lateral), so a 5-8 cm Mecanum side drift no longer
# makes a pure-x reverse stall forever.
RETREAT_SPEED_MPS = 0.10
RETREAT_MIN_SPEED_MPS = 0.035
RETREAT_RAMP_SEC = 0.40
RETREAT_TIMEOUT_SEC = 8.0
RETREAT_FORWARD_TOL_M = 0.050
RETREAT_LATERAL_TOL_M = 0.045
RETREAT_SOFT_FORWARD_TOL_M = 0.080
RETREAT_SOFT_LATERAL_TOL_M = 0.070
RETREAT_LATERAL_KP = 1.25
RETREAT_LATERAL_MAX_MPS = 0.060
RETREAT_NEAR_SOURCE_M = 0.16
# Legacy aliases retained for any diagnostics/helpers that still reference them.
RETREAT_TOL_M = RETREAT_FORWARD_TOL_M
RETREAT_SOFT_TOL_M = max(RETREAT_SOFT_FORWARD_TOL_M, RETREAT_SOFT_LATERAL_TOL_M)

# Digital IR emergency handling.  IR is now a dynamic clearance supervisor,
# not a fixed-time micro-nudge.  A one-sample LOW may stop the chassis quickly,
# but motion recovery starts only after a 2-of-3 confirmation.  While recovering,
# the gimbal ToF checks the source side, front and destination side, and odometry
# limits the total lateral displacement.
IR_FILTER_SAMPLES = 3
IR_DYNAMIC_MAX_LATERAL_M = 0.10
IR_DYNAMIC_STEP_M = 0.018
IR_DYNAMIC_MAX_SEC = 3.0
IR_DYNAMIC_MIN_SPEED_MPS = 0.025
IR_DYNAMIC_MAX_SPEED_MPS = 0.065
IR_DYNAMIC_SOURCE_CLEAR_MM = 155.0
IR_DYNAMIC_DEST_MIN_MM = 145.0
IR_DYNAMIC_FRONT_MIN_MM = 175.0
IR_DYNAMIC_CLEAR_CONFIRM = 2
IR_DYNAMIC_RESCAN_SETTLE_SEC = 0.05
IR_DEST_SHARP_STOP_CM = 9.0
IR_DEST_SHARP_CAUTION_CM = 11.0

# Edge failures are deferred, not allowed to kill DFS.
EDGE_MAX_FAILURES = 2
TURN_EDGE_MAX_FAILURES = 2

# Runtime hardening. RoboMaster Wi-Fi/SDK callbacks can momentarily stall even
# though the physical robot is still healthy. A single transport exception or
# stale callback must therefore not cancel the entire mapping mission.
SDK_COMMAND_RETRIES = 3
SDK_RETRY_DELAY_SEC = 0.08
TELEMETRY_RECOVERY_ATTEMPTS = 3
TELEMETRY_RECOVERY_WAIT_SEC = 1.20
TELEMETRY_RECOVERY_SETTLE_SEC = 0.10
SCAN_EXCEPTION_RETRIES = 2
DFS_EXCEPTION_RESTARTS = 3
MOVE_EXCEPTION_SOURCE_FWD_TOL_M = 0.10
MOVE_EXCEPTION_SOURCE_LAT_TOL_M = 0.08
MOVE_EXCEPTION_DEST_FWD_TOL_M = 0.11
MOVE_EXCEPTION_DEST_LAT_TOL_M = 0.08

# A wall-clock watchdog is useful diagnostically, but ending a valid competition
# run only because it took longer than expected is worse than saving a checkpoint
# and continuing. Set True only when a hard mission time limit is desired.
MISSION_WATCHDOG_HARD_STOP = False


# ============================================================
# YAW PID / CLOSED-LOOP TURN
# ============================================================
# IMPORTANT FRAME POLICY:
# The RoboMaster IMU yaw value itself may have any value when the robot powers on.
# We DO NOT use power-on yaw as the maze frame.  After this Python program has
# connected, stopped the chassis, and collected a stable yaw window, that heading
# becomes logical 0 deg / North for THIS RUN only.
YAW_DRIVE_SIGN = 1.0  # confirmed robot convention: +z / +yaw = turn RIGHT

RUNTIME_YAW_ZERO_SAMPLES = 15
RUNTIME_YAW_ZERO_INTERVAL_SEC = 0.025
RUNTIME_YAW_ZERO_MAX_SPREAD_DEG = 2.5
RUNTIME_YAW_ZERO_RETRIES = 3

# PID while translating.  A small I term cancels persistent drivetrain bias; D
# damps oscillation after turns.  Integral is active only near the target heading.
STRAIGHT_YAW_KP = 2.15
STRAIGHT_YAW_KI = 0.10
STRAIGHT_YAW_KD = 0.16
STRAIGHT_YAW_MAX_DPS = 22.0
STRAIGHT_YAW_DEADBAND_DEG = 0.22
STRAIGHT_YAW_I_LIMIT = 7.0
STRAIGHT_YAW_I_ZONE_DEG = 8.0
STRAIGHT_YAW_D_ALPHA = 0.68

# PID used while stationary after a turn and before translating.
STATIONARY_YAW_KP = 3.00
STATIONARY_YAW_KI = 0.12
STATIONARY_YAW_KD = 0.20
STATIONARY_YAW_MAX_DPS = 28.0
STATIONARY_YAW_MIN_DPS = 2.8
# Near the target, use a gentler minimum turn rate so tiny residual errors do not
# overshoot back and forth.  Keep the normal 2.8 dps floor outside this fine zone.
STATIONARY_YAW_FINE_ZONE_DEG = 0.60
STATIONARY_YAW_FINE_MIN_DPS = 1.20
STATIONARY_YAW_DEADBAND_DEG = 0.18
STATIONARY_YAW_I_LIMIT = 5.0
STATIONARY_YAW_I_ZONE_DEG = 5.0
STATIONARY_YAW_D_ALPHA = 0.72
STATIONARY_HOLD_HZ = 30.0
STATIONARY_SETTLE_SEC = 0.24
STATIONARY_SETTLE_TOL_DEG = 0.70
STATIONARY_ALIGN_TIMEOUT_SEC = 1.25
PRE_MOVE_ALIGN_TOL_DEG = 0.90
PRE_MOVE_ALIGN_TIMEOUT_SEC = 1.10
POST_MOVE_ALIGN_TIMEOUT_SEC = 1.00
# Precision target used only after a normal full-cell arrival.  Pre-move/retreat
# recovery tolerances remain unchanged so resilience is not sacrificed.
POST_MOVE_ALIGN_TOL_DEG = 0.20

# Turn PID.  Integral is deliberately weak and only active in the final approach;
# most of the turn is P+D so inertia does not wind the controller up.
TURN_KP = 1.55
TURN_KI = 0.035
TURN_KD = 0.22
TURN_MAX_DPS = 45.0
TURN_FINE_MAX_DPS = 22.0
TURN_FINE_ZONE_DEG = 14.0
TURN_MIN_DPS = 5.5
TURN_FINE_MIN_DPS = 3.0
TURN_TOLERANCE_DEG = 0.75
TURN_SETTLE_SEC = 0.28
TURN_CONTROL_HZ = 40.0
TURN_I_LIMIT = 7.0
TURN_I_ZONE_DEG = 12.0
TURN_D_ALPHA = 0.72
TURN_TIMEOUT_90_SEC = 5.5
TURN_TIMEOUT_180_SEC = 8.5
TURN_RECOVERY_ATTEMPTS = 2
TURN_RECOVERY_TIMEOUT_SCALE = 1.35
CARDINAL_SNAP_TOL_DEG = 6.0

# If yaw becomes wildly inconsistent with the requested cardinal while translating,
# stop instead of letting DFS drive diagonally into the next logical cell.
MOVE_YAW_ABORT_ERROR_DEG = 14.0
MOVE_YAW_ABORT_CONFIRM_SEC = 0.30


# ============================================================
# GIMBAL / TOF SCAN
# ============================================================
GIMBAL_PITCH_DEG = -5.0
GIMBAL_YAW_SPEED = 180
GIMBAL_PITCH_SPEED = 120
GIMBAL_ACTION_TIMEOUT_SEC = 2.2
# Yaw accuracy is topology-critical; pitch accuracy is not.  The real gimbal
# in field logs repeatedly settled around -8.2 deg when commanded to -5 deg,
# while yaw was already within ~0.1 deg.  Rejecting those poses turned valid
# routes into UNKNOWN and could stop DFS.  Keep yaw strict, but accept a safe
# downward pitch envelope for ToF mapping/collision sensing.
GIMBAL_ANGLE_TOL_DEG = 3.0              # legacy alias / strict yaw tolerance
GIMBAL_YAW_TOL_DEG = 3.0
GIMBAL_PITCH_EXACT_TOL_DEG = 3.0
GIMBAL_TOF_SAFE_PITCH_MIN_DEG = -12.0
GIMBAL_TOF_SAFE_PITCH_MAX_DEG = +2.0
GIMBAL_MOTION_FRONT_YAW_TOL_DEG = 4.5
GIMBAL_SETTLE_SEC = 0.06
GIMBAL_SCAN_RETRIES = 2
UNKNOWN_EDGE_RESCAN_PASSES = 2
UNKNOWN_EDGE_RESCAN_SETTLE_SEC = 0.12
GIMBAL_SOFT_YAW_MIN_DEG = -135.0
GIMBAL_SOFT_YAW_MAX_DEG = +135.0
GIMBAL_SOFT_PITCH_MIN_DEG = -15.0
GIMBAL_SOFT_PITCH_MAX_DEG = +20.0

TOF_SCAN_SAMPLES = 7
TOF_SCAN_INTERVAL_SEC = 0.050
TOF_SCAN_TIMEOUT_SEC = 1.20
TOF_VALID_MIN_MM = 20.0
TOF_VALID_MAX_MM = 10000.0

STARTUP_CONNECT_RETRIES = 3
STARTUP_TELEMETRY_WAIT_SEC = 5.0
STARTUP_GIMBAL_WAIT_SEC = 2.5
ROOT_BACK_SCAN_ENABLED = True
ROOT_BACK_RESTORE_RETRIES = 2


# ============================================================
# MAP / LIMIT GUARDS
# ============================================================
MAP_DIR = Path("maps")
MAP_LATEST_JSON = MAP_DIR / "latest_map.json"
MAP_LATEST_ASCII = MAP_DIR / "latest_map.txt"
MAP_LATEST_SVG = MAP_DIR / "latest_map.svg"
MAP_AUTOSAVE = True

# Last-resort runaway guard.  It does not define maze geometry; it only prevents
# a bad sensor from producing an unbounded graph forever.
MAX_VISITED_CELLS = 80
MAX_MISSION_SEC = 15 * 60


# ============================================================
# HELPERS
# ============================================================
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_deg(angle):
    angle = float(angle)
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def angle_diff_deg(a, b):
    return abs(wrap_deg(float(a) - float(b)))


def neighbor(cell, direction):
    dx, dy = DIR_VEC[int(direction) % 4]
    return (cell[0] + dx, cell[1] + dy)


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    for d, (vx, vy) in DIR_VEC.items():
        if (dx, dy) == (vx, vy):
            return d
    return None


def adc_to_cm(adc, calibration):
    if adc is None:
        return None
    try:
        adc = float(adc)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(adc) or adc < SHARP_MIN_PLAUSIBLE_ADC:
        return None

    near_cm, near_adc = calibration[0]
    _, far_adc = calibration[-1]
    if adc >= near_adc:
        return near_cm
    if adc < far_adc:
        return None

    for i in range(len(calibration) - 1):
        d1, a1 = calibration[i]
        d2, a2 = calibration[i + 1]
        if a1 >= adc >= a2:
            if abs(a1 - a2) < 1e-9:
                return (d1 + d2) * 0.5
            t = (a1 - adc) / (a1 - a2)
            return d1 + t * (d2 - d1)
    return None


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def circular_mean_deg(values):
    """Circular mean that remains correct across -180/+180 wrap."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    sx = sum(math.cos(math.radians(v)) for v in vals)
    sy = sum(math.sin(math.radians(v)) for v in vals)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return wrap_deg(vals[-1])
    return wrap_deg(math.degrees(math.atan2(sy, sx)))


def angular_spread_deg(values, center=None):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("inf")
    if center is None:
        center = circular_mean_deg(vals)
    return max(abs(wrap_deg(v - center)) for v in vals)


def fmt_deg(value):
    """Never let optional telemetry crash diagnostic logging."""
    try:
        value = float(value)
        return f"{value:+.2f}" if math.isfinite(value) else "nan"
    except (TypeError, ValueError):
        return "NA"


class YawPID:
    """Small robust PID specialized for wrapped yaw error in degrees.

    Output units are deg/s because RoboMaster chassis drive_speed(z=...) uses
    angular velocity.  Integral anti-windup and a filtered derivative keep the
    controller predictable on noisy attitude telemetry.
    """
    def __init__(self, kp, ki, kd, out_limit, i_limit, i_zone_deg, d_alpha, deadband_deg=0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.out_limit = abs(float(out_limit))
        self.i_limit = abs(float(i_limit))
        self.i_zone_deg = abs(float(i_zone_deg))
        self.d_alpha = clamp(float(d_alpha), 0.0, 0.98)
        self.deadband_deg = abs(float(deadband_deg))
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_t = None
        self.d_filtered = 0.0

    def step(self, error_deg, now=None, output_limit=None):
        error = wrap_deg(float(error_deg))
        now = time.monotonic() if now is None else float(now)
        if self.prev_t is None:
            dt = 0.0
        else:
            dt = clamp(now - self.prev_t, 0.001, 0.20)

        # Integral is useful only near the requested heading.  Reset it during
        # large maneuvers so a 90/180-degree turn cannot wind it up.
        if abs(error) <= self.i_zone_deg and dt > 0.0:
            self.integral += error * dt
            self.integral = clamp(self.integral, -self.i_limit, self.i_limit)
        elif abs(error) > self.i_zone_deg:
            self.integral = 0.0

        derivative = 0.0
        if self.prev_error is not None and dt > 0.0:
            derivative = wrap_deg(error - self.prev_error) / dt
        self.d_filtered = self.d_alpha * self.d_filtered + (1.0 - self.d_alpha) * derivative

        self.prev_error = error
        self.prev_t = now

        if abs(error) <= self.deadband_deg:
            # Avoid slowly integrating while already straight.
            self.integral *= 0.85
            return 0.0

        limit = self.out_limit if output_limit is None else min(self.out_limit, abs(float(output_limit)))
        u = self.kp * error + self.ki * self.integral + self.kd * self.d_filtered
        return clamp(u, -limit, limit)


# ============================================================
# THREAD-SAFE TELEMETRY WITH FRESHNESS
# ============================================================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.tof = None
        self.position = None
        self.attitude = None
        self.gimbal = None
        self.status = None
        self.seq = {"tof": 0, "position": 0, "attitude": 0, "gimbal": 0, "status": 0}

    def _set(self, name, value):
        with self.lock:
            self.seq[name] += 1
            setattr(self, name, (value, time.monotonic(), self.seq[name]))

    def _get(self, name):
        with self.lock:
            item = getattr(self, name)
            return None if item is None else item

    def set_tof(self, value): self._set("tof", value)
    def get_tof(self): return self._get("tof")
    def set_position(self, value): self._set("position", value)
    def get_position(self): return self._get("position")
    def set_attitude(self, value): self._set("attitude", value)
    def get_attitude(self): return self._get("attitude")
    def set_gimbal(self, value): self._set("gimbal", value)
    def get_gimbal(self): return self._get("gimbal")
    def set_status(self, value): self._set("status", value)
    def get_status(self): return self._get("status")


# ============================================================
# DFS MAP-ONLY EXPLORER
# ============================================================
class DFSMapOnlyExplorer:
    MOVE_ARRIVED = "ARRIVED"
    MOVE_BLOCKED_RETURNED = "BLOCKED_RETURNED"
    MOVE_POSE_UNCERTAIN = "POSE_UNCERTAIN"
    MOVE_STOPPED = "STOPPED"

    def __init__(self):
        self.ep_robot = robot.Robot()
        self.chassis = None
        self.gimbal = None
        self.sensor_adapter = None
        self.distance_sensor = None

        self.state = SharedState()
        self.running = True
        self.connected = False
        self.pose_trusted = True
        self.safe_pause_reason = None

        self.position_origin_raw = None
        self.position_raw_latest = None
        # Preserve logical odometry continuity if the SDK position callback has
        # to be unsubscribed/re-subscribed during a Wi-Fi recovery.
        self.position_rebase_pending = False
        self.position_rebase_logical = None
        self.base_yaw_deg = None
        self.yaw_ref_deg = None
        self.gimbal_zero_pitch_raw = None
        self.gimbal_zero_yaw_raw = None

        self.heading = 0
        self.root = (0, 0)
        self.current = self.root
        self.visited = set()
        self.parent = {}
        self.open_dirs = {}
        self.edge_state = {}      # (cell, dir) -> OPEN/WALL/UNKNOWN/BLOCKED/DEFERRED
        self.cell_scan_mm = {}
        # Learned physical distance for a traversed logical edge.  Normally this
        # is ~0.60 m, but a sensor-confirmed node may be reached a little early.
        # Remembering that distance makes the reverse/backtrack traversal use the
        # same physical anchor instead of blindly overshooting by 60 cm.
        self.edge_travel_m = {}   # ((cell), dir) -> metres
        self.last_move_distance_m = None
        self.last_stop_probe = None
        self.edge_failures = {}
        self.turn_failures = {}
        self.deferred_edges = set()
        self.unknown_rescan_counts = {}
        self.map_complete = False
        self.map_created_at = datetime.now().isoformat(timespec="seconds")

        self.left_adc_hist = deque(maxlen=SHARP_FILTER_SAMPLES)
        self.right_adc_hist = deque(maxlen=SHARP_FILTER_SAMPLES)
        self.sharp_authority = None
        self.sharp_authority_since = 0.0

        self.yaw_hold_lock = threading.Lock()
        self.pid_straight = YawPID(
            STRAIGHT_YAW_KP, STRAIGHT_YAW_KI, STRAIGHT_YAW_KD,
            STRAIGHT_YAW_MAX_DPS, STRAIGHT_YAW_I_LIMIT,
            STRAIGHT_YAW_I_ZONE_DEG, STRAIGHT_YAW_D_ALPHA,
            STRAIGHT_YAW_DEADBAND_DEG,
        )
        self.pid_stationary = YawPID(
            STATIONARY_YAW_KP, STATIONARY_YAW_KI, STATIONARY_YAW_KD,
            STATIONARY_YAW_MAX_DPS, STATIONARY_YAW_I_LIMIT,
            STATIONARY_YAW_I_ZONE_DEG, STATIONARY_YAW_D_ALPHA,
            STATIONARY_YAW_DEADBAND_DEG,
        )
        self.pid_turn = YawPID(
            TURN_KP, TURN_KI, TURN_KD, TURN_MAX_DPS, TURN_I_LIMIT,
            TURN_I_ZONE_DEG, TURN_D_ALPHA, 0.0,
        )
        self.fault_log = deque(maxlen=240)
        self.mission_start_t = None
        self._mission_watchdog_warned = False

    # --------------------------------------------------------
    # LOGGING / SAFE CALLS
    # --------------------------------------------------------
    def fault(self, subsystem, detail, action="continue safely"):
        msg = f"[{subsystem}] {detail} -> {action}"
        self.fault_log.append({
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "subsystem": str(subsystem),
            "detail": str(detail),
            "action": str(action),
        })
        print(f"[SOFT FAULT] {msg}")

    def safe_stop(self):
        if self.chassis is None:
            return
        try:
            self.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=DRIVE_COMMAND_TIMEOUT)
        except Exception as exc:
            self.fault("STOP", f"drive_speed(0) failed: {type(exc).__name__}: {exc}", "no further command")

    def drive_speed_resilient(self, x=0.0, y=0.0, z=0.0, timeout=DRIVE_COMMAND_TIMEOUT, label="DRIVE CMD"):
        """Send a chassis velocity command with bounded retries.

        A transient SDK/Wi-Fi exception is treated as a transport fault, not as
        proof that the robot pose is lost. The caller decides what to do only
        after all retries fail.
        """
        if self.chassis is None:
            self.fault(label, "chassis unavailable", "command not sent")
            return False
        for attempt in range(1, SDK_COMMAND_RETRIES + 1):
            try:
                self.chassis.drive_speed(
                    x=float(x), y=float(y), z=float(z), timeout=float(timeout)
                )
                return True
            except Exception as exc:
                self.fault(
                    label,
                    f"attempt {attempt}/{SDK_COMMAND_RETRIES}: {type(exc).__name__}: {exc}",
                    "retry command" if attempt < SDK_COMMAND_RETRIES else "caller recovery",
                )
                self.safe_stop()
                if attempt < SDK_COMMAND_RETRIES:
                    time.sleep(SDK_RETRY_DELAY_SEC * attempt)
        return False

    def _resubscribe_stream(self, name):
        """Best-effort repair of one callback stream; never raises."""
        try:
            if name == "position" and self.chassis is not None:
                # Remember the last logical pose. Some SDK/firmware combinations
                # can restart a position stream with a new raw origin after re-sub.
                # position_callback() rebases the new raw sample onto this logical
                # value so DFS distances do not jump discontinuously.
                self.position_rebase_logical = self.current_position(fresh=False)
                self.position_rebase_pending = True
                try:
                    self.chassis.unsub_position()
                except Exception:
                    pass
                self.chassis.sub_position(freq=POSITION_FREQ_HZ, callback=self.position_callback)
            elif name == "attitude" and self.chassis is not None:
                try:
                    self.chassis.unsub_attitude()
                except Exception:
                    pass
                self.chassis.sub_attitude(freq=ATTITUDE_FREQ_HZ, callback=self.attitude_callback)
            elif name == "tof" and self.distance_sensor is not None:
                try:
                    self.distance_sensor.unsub_distance()
                except Exception:
                    pass
                self.distance_sensor.sub_distance(freq=TOF_FREQ_HZ, callback=self.tof_callback)
            elif name == "gimbal" and self.gimbal is not None:
                try:
                    self.gimbal.unsub_angle()
                except Exception:
                    pass
                self.gimbal.sub_angle(freq=GIMBAL_ANGLE_FREQ_HZ, callback=self.gimbal_callback)
            elif name == "status" and self.chassis is not None:
                try:
                    self.chassis.unsub_status()
                except Exception:
                    pass
                self.chassis.sub_status(freq=STATUS_FREQ_HZ, callback=self.status_callback)
            else:
                return False
            self.fault("TELEMETRY RECOVER", f"re-subscribed {name}", "wait for fresh callback")
            return True
        except Exception as exc:
            self.fault(
                "TELEMETRY RECOVER",
                f"{name}: {type(exc).__name__}: {exc}",
                "retry bounded recovery",
            )
            return False

    def recover_telemetry(self, require_position=True, require_attitude=True,
                          require_tof=False, require_gimbal=False, reason="runtime recovery"):
        """Try to reacquire required streams while the chassis is stopped."""
        self.safe_stop()

        def ready():
            if require_position and self.current_position() is None:
                return False
            if require_attitude and self.current_yaw() is None:
                return False
            if require_tof and self.latest_tof() is None:
                return False
            if require_gimbal:
                p, y = self.current_gimbal_raw()
                if p is None or y is None:
                    return False
            return True

        if ready():
            return True

        for attempt in range(1, TELEMETRY_RECOVERY_ATTEMPTS + 1):
            missing = []
            if require_position and self.current_position() is None:
                missing.append("position")
            if require_attitude and self.current_yaw() is None:
                missing.append("attitude")
            if require_tof and self.latest_tof() is None:
                missing.append("tof")
            if require_gimbal:
                p, y = self.current_gimbal_raw()
                if p is None or y is None:
                    missing.append("gimbal")

            for name in missing:
                self._resubscribe_stream(name)

            deadline = time.monotonic() + TELEMETRY_RECOVERY_WAIT_SEC
            while self.running and time.monotonic() < deadline:
                if ready():
                    time.sleep(TELEMETRY_RECOVERY_SETTLE_SEC)
                    self.fault(
                        "TELEMETRY RECOVER",
                        f"{reason}: streams restored on attempt {attempt}",
                        "resume",
                    )
                    return True
                time.sleep(0.04)

        self.fault(
            "TELEMETRY RECOVER",
            f"{reason}: required streams still unavailable",
            "caller decides safe fallback",
        )
        return False

    def recover_gimbal_front(self):
        """Recover a transient gimbal action/feedback failure while stationary."""
        if self.gimbal_front_safe_for_motion():
            return True
        self.recover_telemetry(
            require_position=False, require_attitude=False, require_gimbal=True,
            reason="gimbal feedback recovery",
        )
        for attempt in range(1, GIMBAL_SCAN_RETRIES + 2):
            if self.gimbal_front_down(force=True) and self.gimbal_front_safe_for_motion():
                return True
            self.safe_stop()
            time.sleep(0.08 * attempt)
        # Last attempt: physical recenter is safe here because callers invoke this
        # only while stopped on a node / before translation.
        if self.recenter_gimbal_and_zero():
            return self.gimbal_front_down(force=True) and self.gimbal_front_safe_for_motion()
        return False

    def enter_safe_pause(self, reason):
        self.pose_trusted = False
        self.safe_pause_reason = str(reason)
        self.safe_stop()
        print(f"\n[SAFE PAUSE] {reason}")
        print("[SAFE PAUSE] autonomous motion stopped; partial map will be saved.")
        self.save_map(final=False)

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------
    def tof_callback(self, distance_info):
        try:
            if distance_info and len(distance_info) > TOF_INDEX:
                v = float(distance_info[TOF_INDEX])
                if math.isfinite(v) and TOF_VALID_MIN_MM <= v <= TOF_VALID_MAX_MM:
                    self.state.set_tof(v)
        except Exception:
            pass

    def position_callback(self, position_info):
        try:
            if not position_info or len(position_info) < 3:
                return
            raw = tuple(float(v) for v in position_info[:3])
            self.position_raw_latest = raw
            if self.position_rebase_pending:
                logical = self.position_rebase_logical or (0.0, 0.0, 0.0)
                self.position_origin_raw = (
                    raw[0] - float(logical[0]),
                    raw[1] - float(logical[1]),
                    raw[2] - float(logical[2]),
                )
                self.position_rebase_pending = False
                self.position_rebase_logical = None
            if self.position_origin_raw is None:
                self.position_origin_raw = raw
            ox, oy, oz = self.position_origin_raw
            self.state.set_position((raw[0] - ox, raw[1] - oy, raw[2] - oz))
        except Exception:
            pass

    def attitude_callback(self, attitude_info):
        try:
            if attitude_info and len(attitude_info) >= 3:
                yaw, pitch, roll = attitude_info[:3]
                self.state.set_attitude((float(yaw), float(pitch), float(roll)))
        except Exception:
            pass

    def gimbal_callback(self, angle_info):
        try:
            if angle_info and len(angle_info) >= 4:
                self.state.set_gimbal(tuple(float(v) for v in angle_info[:4]))
        except Exception:
            pass

    def status_callback(self, status_info):
        try:
            if status_info:
                self.state.set_status(tuple(status_info))
        except Exception:
            pass

    # --------------------------------------------------------
    # TELEMETRY ACCESS
    # --------------------------------------------------------
    @staticmethod
    def _fresh(item, max_age):
        if item is None:
            return None
        value, stamp, seq = item
        if time.monotonic() - stamp > max_age:
            return None
        return value

    def current_position(self, fresh=True):
        item = self.state.get_position()
        return self._fresh(item, POSITION_STALE_SEC) if fresh else (None if item is None else item[0])

    def current_yaw(self, fresh=True):
        item = self.state.get_attitude()
        value = self._fresh(item, ATTITUDE_STALE_SEC) if fresh else (None if item is None else item[0])
        return None if value is None else float(value[0])

    def current_gimbal_raw(self, fresh=True):
        item = self.state.get_gimbal()
        value = self._fresh(item, 0.50) if fresh else (None if item is None else item[0])
        if value is None:
            return None, None
        return float(value[0]), float(value[1])

    def latest_tof(self, fresh=True):
        item = self.state.get_tof()
        value = self._fresh(item, FRONT_TOF_STALE_SEC) if fresh else (None if item is None else item[0])
        return None if value is None else float(value)

    def chassis_slip_detected(self):
        item = self.state.get_status()
        value = self._fresh(item, 1.0)
        return bool(value is not None and len(value) >= 6 and value[5])

    def wait_for_telemetry(self, name, getter, timeout):
        deadline = time.monotonic() + float(timeout)
        while self.running and time.monotonic() < deadline:
            value = getter()
            if value is not None:
                return value
            time.sleep(0.03)
        self.fault("TELEMETRY", f"{name} unavailable for {timeout:.1f}s", "retry or safe stop")
        return None

    # --------------------------------------------------------
    # CONNECT / CLEANUP
    # --------------------------------------------------------
    def connect(self):
        print("============================================================")
        print(" RoboMaster DFS MAP-ONLY / RESILIENT RESET")
        print("============================================================")
        print(f" Grid       : {GRID_TILE_M:.2f} m")
        print(f" ToF OPEN   : > {TOF_OPEN_THRESHOLD_MM:.0f} mm")
        print(f" Runtime    : RUN pose=(0,0), FRONT=N")
        print(" Features   : DFS + map + ToF node-probe + resilient gimbal + Sharp + IR + runtime-zero yaw PID")
        print(" Removed    : Vision / Fire / Radar / Target / Attack / Known-map")
        print("============================================================")

        init_ok = False
        for attempt in range(1, STARTUP_CONNECT_RETRIES + 1):
            try:
                print(f"[CONNECT] AP attempt {attempt}/{STARTUP_CONNECT_RETRIES}")
                self.ep_robot.initialize(conn_type=CONN_TYPE)
                init_ok = True
                break
            except Exception as exc:
                self.fault("CONNECT", f"attempt {attempt}: {type(exc).__name__}: {exc}", "retry")
                time.sleep(0.6)

        if not init_ok:
            self.enter_safe_pause("RoboMaster connection could not be established")
            return False

        try:
            self.ep_robot.set_robot_mode(mode=robot.FREE)
        except Exception as exc:
            self.fault("ROBOT MODE", f"FREE failed: {type(exc).__name__}: {exc}", "continue; scans may retry")

        try:
            self.chassis = self.ep_robot.chassis
            self.gimbal = self.ep_robot.gimbal
            self.sensor_adapter = self.ep_robot.sensor_adaptor
            self.distance_sensor = self.ep_robot.sensor
        except Exception as exc:
            self.enter_safe_pause(f"required RoboMaster module unavailable: {type(exc).__name__}: {exc}")
            return False

        subscriptions = [
            ("ToF", lambda: self.distance_sensor.sub_distance(freq=TOF_FREQ_HZ, callback=self.tof_callback)),
            ("position", lambda: self.chassis.sub_position(freq=POSITION_FREQ_HZ, callback=self.position_callback)),
            ("attitude", lambda: self.chassis.sub_attitude(freq=ATTITUDE_FREQ_HZ, callback=self.attitude_callback)),
            ("gimbal", lambda: self.gimbal.sub_angle(freq=GIMBAL_ANGLE_FREQ_HZ, callback=self.gimbal_callback)),
            ("status", lambda: self.chassis.sub_status(freq=STATUS_FREQ_HZ, callback=self.status_callback)),
        ]
        for label, fn in subscriptions:
            try:
                result = fn()
                print(f"[SUB] {label}: {result}")
            except Exception as exc:
                self.fault("SUBSCRIBE", f"{label}: {type(exc).__name__}: {exc}", "telemetry wait will verify")

        self.safe_stop()

        yaw = self.wait_for_telemetry("chassis yaw", self.current_yaw, STARTUP_TELEMETRY_WAIT_SEC)
        pos = self.wait_for_telemetry("chassis position", self.current_position, STARTUP_TELEMETRY_WAIT_SEC)
        if yaw is None or pos is None:
            self.enter_safe_pause("critical chassis telemetry missing at startup")
            return False

        # Create the maze yaw frame NOW, not when the robot was powered on.
        # The first telemetry value above only proves that attitude data exists;
        # capture_runtime_yaw_zero() takes a stable multi-sample snapshot after RUN.
        runtime_zero = self.capture_runtime_yaw_zero()
        if runtime_zero is None:
            self.enter_safe_pause("could not establish a stable runtime yaw zero")
            return False
        self.base_yaw_deg = runtime_zero
        self.heading = 0
        self.yaw_ref_deg = self.base_yaw_deg
        self.reset_yaw_pid()

        # Re-zero mission position after telemetry has settled.
        if self.position_raw_latest is not None:
            self.position_origin_raw = tuple(self.position_raw_latest)
            self.state.set_position((0.0, 0.0, 0.0))

        # Establish a program-local gimbal zero.  Failure is retried; if angle
        # feedback remains unavailable mapping cannot classify directions safely.
        if not self.recenter_gimbal_and_zero():
            self.enter_safe_pause("gimbal could not establish a reliable runtime zero")
            return False

        if not self.gimbal_front_down(force=True):
            self.fault("GIMBAL", "front/down startup pose not exact", "continue; each scan will retry")

        self.connected = True
        self.mission_start_t = time.monotonic()
        print(f"[READY] runtime yaw-zero raw={self.base_yaw_deg:+.2f} deg -> logical yaw=0.00 deg; root=(0,0); heading=N")
        return True

    def cleanup(self):
        print("\n[CLEANUP] stopping robot and saving map...")
        self.running = False
        self.safe_stop()
        try:
            self.save_map(final=self.map_complete)
        except Exception as exc:
            print(f"[MAP SAVE WARN] cleanup save failed: {exc}")

        for fn in (
            lambda: self.distance_sensor.unsub_distance() if self.distance_sensor else None,
            lambda: self.chassis.unsub_position() if self.chassis else None,
            lambda: self.chassis.unsub_attitude() if self.chassis else None,
            lambda: self.chassis.unsub_status() if self.chassis else None,
            lambda: self.gimbal.unsub_angle() if self.gimbal else None,
        ):
            try:
                fn()
            except Exception:
                pass

        try:
            self.ep_robot.close()
        except Exception:
            pass
        print("[CLEANUP] done.")

    # --------------------------------------------------------
    # YAW / TURNING
    # --------------------------------------------------------
    def reset_yaw_pid(self):
        self.pid_straight.reset()
        self.pid_stationary.reset()
        self.pid_turn.reset()

    def capture_runtime_yaw_zero(self):
        """Define logical North from the chassis pose at program RUN time.

        This intentionally ignores whatever yaw the robot had at power-on.
        Multiple samples are circular-averaged so +/-180 wrap cannot corrupt zero.
        """
        self.safe_stop()
        for attempt in range(1, RUNTIME_YAW_ZERO_RETRIES + 1):
            samples = []
            deadline = time.monotonic() + max(1.0, RUNTIME_YAW_ZERO_SAMPLES * RUNTIME_YAW_ZERO_INTERVAL_SEC * 3.0)
            while self.running and len(samples) < RUNTIME_YAW_ZERO_SAMPLES and time.monotonic() < deadline:
                yaw = self.current_yaw()
                if yaw is not None and math.isfinite(yaw):
                    samples.append(float(yaw))
                time.sleep(RUNTIME_YAW_ZERO_INTERVAL_SEC)

            center = circular_mean_deg(samples)
            spread = angular_spread_deg(samples, center) if center is not None else float("inf")
            if center is not None and len(samples) >= max(5, RUNTIME_YAW_ZERO_SAMPLES // 2) and spread <= RUNTIME_YAW_ZERO_MAX_SPREAD_DEG:
                print(f"[YAW ZERO] RUN-time raw={center:+.3f} deg spread={spread:.3f} deg n={len(samples)}")
                return center

            self.fault(
                "YAW ZERO",
                f"unstable capture attempt {attempt}: n={len(samples)} spread={spread:.2f}deg",
                "stop and retry capture",
            )
            self.safe_stop()
            time.sleep(0.15)
        return None

    def desired_yaw_for_heading(self, heading):
        if self.base_yaw_deg is None:
            return None
        return wrap_deg(self.base_yaw_deg + 90.0 * (int(heading) % 4))

    def logical_yaw_deg(self, raw_yaw=None):
        if self.base_yaw_deg is None:
            return None
        if raw_yaw is None:
            raw_yaw = self.current_yaw()
        if raw_yaw is None:
            return None
        return wrap_deg(float(raw_yaw) - self.base_yaw_deg)

    def yaw_error_deg(self, target_yaw=None):
        if target_yaw is None:
            target_yaw = self.yaw_ref_deg
        yaw = self.current_yaw()
        if target_yaw is None or yaw is None:
            return None
        return wrap_deg(float(target_yaw) - yaw)

    def yaw_hold_command(self, target_yaw=None, stationary=False):
        err = self.yaw_error_deg(target_yaw)
        if err is None:
            return 0.0
        pid = self.pid_stationary if stationary else self.pid_straight
        return YAW_DRIVE_SIGN * pid.step(err)

    def align_heading_stationary(self, target_yaw=None, timeout_sec=STATIONARY_ALIGN_TIMEOUT_SEC,
                                 tolerance_deg=STATIONARY_SETTLE_TOL_DEG, settle_sec=STATIONARY_SETTLE_SEC):
        """PID-align chassis yaw and require it to remain in tolerance."""
        if self.chassis is None:
            return False
        if target_yaw is None:
            target_yaw = self.yaw_ref_deg
        if target_yaw is None:
            self.safe_stop()
            return False

        self.pid_stationary.reset()
        deadline = time.monotonic() + max(0.05, float(timeout_sec))
        settled_since = None
        dt = 1.0 / STATIONARY_HOLD_HZ

        with self.yaw_hold_lock:
            while self.running and time.monotonic() < deadline:
                yaw = self.current_yaw()
                if yaw is None:
                    self.safe_stop()
                    settled_since = None
                    time.sleep(dt)
                    continue

                err = wrap_deg(float(target_yaw) - yaw)
                if abs(err) <= float(tolerance_deg):
                    self.safe_stop()
                    if settled_since is None:
                        settled_since = time.monotonic()
                    if time.monotonic() - settled_since >= float(settle_sec):
                        self.safe_stop()
                        return True
                else:
                    settled_since = None
                    z = YAW_DRIVE_SIGN * self.pid_stationary.step(err)
                    min_dps = (
                        STATIONARY_YAW_FINE_MIN_DPS
                        if abs(err) <= STATIONARY_YAW_FINE_ZONE_DEG
                        else STATIONARY_YAW_MIN_DPS
                    )
                    if abs(z) < min_dps:
                        z = math.copysign(min_dps, z if abs(z) > 1e-9 else err)
                    if not self.drive_speed_resilient(
                        x=0.0, y=0.0, z=z, timeout=DRIVE_COMMAND_TIMEOUT, label="YAW ALIGN CMD"
                    ):
                        self.safe_stop()
                        return False
                time.sleep(dt)

        self.safe_stop()
        err = self.yaw_error_deg(target_yaw)
        if err is not None:
            self.fault("YAW ALIGN", f"timeout residual={err:+.2f}deg", "caller may retry/defer")
        return False

    def hold_heading_stationary(self, duration=STATIONARY_SETTLE_SEC):
        # Compatibility wrapper: duration remains a minimum settle request, but
        # the controller now verifies actual yaw instead of merely waiting.
        timeout = max(STATIONARY_ALIGN_TIMEOUT_SEC, float(duration) + 0.45)
        return self.align_heading_stationary(
            self.yaw_ref_deg, timeout_sec=timeout,
            tolerance_deg=STATIONARY_SETTLE_TOL_DEG,
            settle_sec=max(0.08, float(duration)),
        )

    def turn_closed_loop(self, target_yaw, timeout_sec):
        if self.chassis is None:
            return False
        target_yaw = wrap_deg(target_yaw)
        self.safe_stop()
        self.pid_turn.reset()
        try:
            self.ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
        except Exception as exc:
            self.fault("TURN MODE", f"CHASSIS_LEAD: {type(exc).__name__}: {exc}", "continue closed-loop")

        dt = 1.0 / TURN_CONTROL_HZ
        start = time.monotonic()
        settled_since = None
        ok = False
        last_yaw_t = time.monotonic()
        try:
            while self.running and time.monotonic() - start < float(timeout_sec):
                yaw = self.current_yaw()
                if yaw is None:
                    self.safe_stop()
                    if time.monotonic() - last_yaw_t > ATTITUDE_STALE_SEC * 2.0:
                        self.fault("TURN", "attitude telemetry stale", "stop turn and retry")
                        break
                    time.sleep(dt)
                    continue
                last_yaw_t = time.monotonic()

                err = wrap_deg(target_yaw - yaw)
                if abs(err) <= TURN_TOLERANCE_DEG:
                    self.safe_stop()
                    if settled_since is None:
                        settled_since = time.monotonic()
                    if time.monotonic() - settled_since >= TURN_SETTLE_SEC:
                        ok = True
                        break
                else:
                    settled_since = None
                    limit = TURN_FINE_MAX_DPS if abs(err) <= TURN_FINE_ZONE_DEG else TURN_MAX_DPS
                    z = YAW_DRIVE_SIGN * self.pid_turn.step(err, output_limit=limit)
                    min_dps = TURN_FINE_MIN_DPS if abs(err) <= 4.0 else TURN_MIN_DPS
                    if abs(z) < min_dps:
                        z = math.copysign(min_dps, z if abs(z) > 1e-9 else err)
                    if not self.drive_speed_resilient(
                        x=0.0, y=0.0, z=z, timeout=DRIVE_COMMAND_TIMEOUT, label="TURN CMD"
                    ):
                        break
                time.sleep(dt)
        finally:
            self.safe_stop()
            try:
                self.ep_robot.set_robot_mode(mode=robot.FREE)
            except Exception as exc:
                self.fault("TURN MODE", f"FREE: {type(exc).__name__}: {exc}", "continue")
            time.sleep(0.06)

        if ok:
            # A second low-energy PID settle in FREE mode removes residual chassis
            # reaction/inertia before DFS scans or translates from the new heading.
            self.pid_stationary.reset()
            if not self.align_heading_stationary(
                target_yaw, timeout_sec=STATIONARY_ALIGN_TIMEOUT_SEC,
                tolerance_deg=STATIONARY_SETTLE_TOL_DEG, settle_sec=STATIONARY_SETTLE_SEC
            ):
                self.fault("POST TURN", "coarse turn succeeded but final PID settle did not", "report turn failure for retry")
                return False
        return ok

    def recover_to_nearest_cardinal(self):
        yaw = self.current_yaw()
        if yaw is None or self.base_yaw_deg is None:
            return None
        ranked = []
        for d in range(4):
            target = self.desired_yaw_for_heading(d)
            ranked.append((angle_diff_deg(yaw, target), d, target))
        err, d, target = min(ranked)
        if err <= CARDINAL_SNAP_TOL_DEG:
            self.yaw_ref_deg = target
            if self.align_heading_stationary(
                target, timeout_sec=STATIONARY_ALIGN_TIMEOUT_SEC,
                tolerance_deg=STATIONARY_SETTLE_TOL_DEG, settle_sec=STATIONARY_SETTLE_SEC
            ):
                self.heading = d
                final_err = self.yaw_error_deg(target)
                self.fault("POSE RECOVER", f"aligned to {DIR_NAMES[d]} residual={fmt_deg(final_err)}deg", "resume")
                return d
            self.fault("POSE RECOVER", f"nearest cardinal {DIR_NAMES[d]} found but PID alignment failed", "do not snap pose")
        return None

    def turn_to_direction(self, target_dir):
        """Exception-contained cardinal turn. Safe to fail without ending DFS."""
        target_dir = int(target_dir) % 4
        try:
            return self._turn_to_direction_impl(target_dir)
        except Exception as exc:
            self.safe_stop()
            self.fault(
                "TURN UNEXPECTED",
                f"{type(exc).__name__}: {exc}",
                "reacquire attitude and recover nearest cardinal",
            )
            try:
                if self.recover_telemetry(
                    require_position=False, require_attitude=True,
                    reason="unexpected turn exception",
                ):
                    snapped = self.recover_to_nearest_cardinal()
                    if snapped == target_dir:
                        self.yaw_ref_deg = self.desired_yaw_for_heading(target_dir)
                        return True
            except Exception as recover_exc:
                self.fault(
                    "TURN RECOVER",
                    f"{type(recover_exc).__name__}: {recover_exc}",
                    "defer this edge",
                )
            return False

    def _turn_to_direction_impl(self, target_dir):
        target_dir = int(target_dir) % 4
        target_yaw = self.desired_yaw_for_heading(target_dir)
        if target_yaw is None:
            self.fault("TURN", "runtime yaw zero is not initialized", "defer turn")
            return False

        if target_dir == self.heading:
            self.yaw_ref_deg = target_yaw
            if not self.align_heading_stationary(
                target_yaw, timeout_sec=PRE_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.12
            ):
                self.fault("TURN", f"already facing {DIR_NAMES[target_dir]} but yaw could not re-align", "retry/defer")
                return False
            if self.gimbal_front_down():
                return True
            if self.recover_gimbal_front():
                self.fault("TURN", "gimbal front recovered after transient failure", "resume")
                return True
            return False

        delta = (target_dir - self.heading) % 4
        timeout = TURN_TIMEOUT_180_SEC if delta == 2 else TURN_TIMEOUT_90_SEC

        for attempt in range(TURN_RECOVERY_ATTEMPTS + 1):
            scale = 1.0 if attempt == 0 else TURN_RECOVERY_TIMEOUT_SCALE
            print(f"[TURN] {DIR_NAMES[self.heading]} -> {DIR_NAMES[target_dir]} attempt={attempt+1}")
            if self.turn_closed_loop(target_yaw, timeout * scale):
                # Update discrete pose ONLY after the final PID settle succeeds.
                self.heading = target_dir
                self.yaw_ref_deg = target_yaw
                self.sharp_authority = None
                self.pid_straight.reset()
                self.gimbal_front_down(force=True)
                final_err = self.yaw_error_deg(target_yaw)
                logical = self.logical_yaw_deg()
                print(f"[TURN OK] {DIR_NAMES[target_dir]} logical_yaw={fmt_deg(logical)}deg residual={fmt_deg(final_err)}deg")
                return True
            self.fault("TURN", f"failed to settle on {DIR_NAMES[target_dir]} attempt {attempt+1}", "retry")
            time.sleep(0.12)

        snapped = self.recover_to_nearest_cardinal()
        if snapped == target_dir:
            self.pid_straight.reset()
            return True
        self.safe_stop()
        return False

    # --------------------------------------------------------
    # GIMBAL / TOF
    # --------------------------------------------------------
    def _wait_action(self, action, timeout, label):
        try:
            result = action.wait_for_completed(timeout=float(timeout))
            if result is False:
                self.fault(label, "SDK action returned False", "verify feedback")
                return False
            return True
        except Exception as exc:
            self.fault(label, f"{type(exc).__name__}: {exc}", "verify feedback / retry")
            return False

    def recenter_gimbal_and_zero(self):
        if self.gimbal is None:
            return False
        for attempt in range(1, GIMBAL_SCAN_RETRIES + 2):
            action_ok = False
            try:
                action = self.gimbal.recenter(pitch_speed=GIMBAL_PITCH_SPEED, yaw_speed=GIMBAL_YAW_SPEED)
                action_ok = self._wait_action(action, GIMBAL_ACTION_TIMEOUT_SEC, "GIMBAL RECENTER")
            except Exception as exc:
                self.fault("GIMBAL RECENTER", f"attempt {attempt}: {type(exc).__name__}: {exc}", "retry")
            deadline = time.monotonic() + STARTUP_GIMBAL_WAIT_SEC
            while time.monotonic() < deadline:
                p, y = self.current_gimbal_raw()
                # sub_angle() reports gimbal angles relative to the chassis.
                # A valid recenter should therefore be physically near zero.
                # Do not redefine an arbitrary stuck turret angle as software
                # FRONT merely because telemetry exists.
                if (
                    p is not None and y is not None
                    and abs(p) <= 6.0 and abs(y) <= 6.0
                ):
                    self.gimbal_zero_pitch_raw = p
                    self.gimbal_zero_yaw_raw = y
                    print(
                        f"[GIMBAL ZERO] raw p={p:+.2f} y={y:+.2f} -> local zero "
                        f"(action_ok={action_ok})"
                    )
                    return True
                time.sleep(0.03)
            self.fault(
                "GIMBAL RECENTER",
                f"attempt {attempt}: feedback did not settle near physical zero",
                "retry instead of accepting a false FRONT",
            )
        return False

    def current_gimbal_relative(self):
        p, y = self.current_gimbal_raw()
        if p is None or y is None:
            return None, None
        if self.gimbal_zero_pitch_raw is None or self.gimbal_zero_yaw_raw is None:
            return p, y
        return p - self.gimbal_zero_pitch_raw, y - self.gimbal_zero_yaw_raw

    def gimbal_at_target(self, yaw_deg, pitch_deg):
        """Return True when the ToF is pointed at the requested bearing.

        For maze topology, yaw is the critical coordinate.  Pitch may settle a
        few degrees below the requested -5 deg on the real EP; that is still a
        perfectly usable ToF ray as long as it remains inside the safe downward
        envelope.  This deliberately prevents a stable -8 deg pitch offset from
        turning a valid +/-90 deg scan into UNKNOWN.
        """
        p, y = self.current_gimbal_relative()
        if p is None or y is None:
            return False
        yaw_ok = abs(wrap_deg(y - yaw_deg)) <= GIMBAL_YAW_TOL_DEG
        pitch_exact = abs(p - pitch_deg) <= GIMBAL_PITCH_EXACT_TOL_DEG
        pitch_safe = (
            GIMBAL_TOF_SAFE_PITCH_MIN_DEG
            <= p
            <= GIMBAL_TOF_SAFE_PITCH_MAX_DEG
        )
        return bool(yaw_ok and (pitch_exact or pitch_safe))

    def gimbal_front_safe_for_motion(self):
        """Front ToF only needs forward yaw + a safe pitch, not exact -5 deg."""
        p, y = self.current_gimbal_relative()
        if p is None or y is None:
            return False
        return bool(
            abs(wrap_deg(y)) <= GIMBAL_MOTION_FRONT_YAW_TOL_DEG
            and GIMBAL_TOF_SAFE_PITCH_MIN_DEG <= p <= GIMBAL_TOF_SAFE_PITCH_MAX_DEG
        )

    def gimbal_goto(self, yaw_deg, pitch_deg=GIMBAL_PITCH_DEG, force=False):
        if self.gimbal is None:
            return False
        yaw_deg = clamp(float(yaw_deg), GIMBAL_SOFT_YAW_MIN_DEG, GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg = clamp(float(pitch_deg), GIMBAL_SOFT_PITCH_MIN_DEG, GIMBAL_SOFT_PITCH_MAX_DEG)
        if not force and self.gimbal_at_target(yaw_deg, pitch_deg):
            return True

        raw_p = pitch_deg + (self.gimbal_zero_pitch_raw or 0.0)
        raw_y = yaw_deg + (self.gimbal_zero_yaw_raw or 0.0)

        for attempt in range(1, GIMBAL_SCAN_RETRIES + 2):
            try:
                action = self.gimbal.moveto(
                    pitch=raw_p, yaw=raw_y,
                    pitch_speed=GIMBAL_PITCH_SPEED,
                    yaw_speed=GIMBAL_YAW_SPEED,
                )
                self._wait_action(action, GIMBAL_ACTION_TIMEOUT_SEC + 0.6, "GIMBAL MOVETO")
            except Exception as exc:
                self.fault("GIMBAL MOVETO", f"attempt {attempt}: {type(exc).__name__}: {exc}", "retry")

            time.sleep(GIMBAL_SETTLE_SEC)
            if self.gimbal_at_target(yaw_deg, pitch_deg):
                p, y = self.current_gimbal_relative()
                if p is not None and abs(p - pitch_deg) > GIMBAL_PITCH_EXACT_TOL_DEG:
                    print(
                        f"[GIMBAL SOFT ACCEPT] yaw target={yaw_deg:+.1f} actual={y:+.1f}; "
                        f"pitch target={pitch_deg:+.1f} actual={p:+.1f} (safe ToF envelope)"
                    )
                return True
            p, y = self.current_gimbal_relative()
            self.fault("GIMBAL VERIFY", f"target p={pitch_deg:+.1f} y={yaw_deg:+.1f}; actual p={p} y={y}", "retry")

        return False

    def gimbal_front_down(self, force=False):
        return self.gimbal_goto(0.0, GIMBAL_PITCH_DEG, force=force)

    def sample_fresh_tof(self, samples=TOF_SCAN_SAMPLES, timeout=TOF_SCAN_TIMEOUT_SEC):
        values = []
        start_item = self.state.get_tof()
        last_seq = 0 if start_item is None else start_item[2]
        deadline = time.monotonic() + float(timeout)

        while self.running and time.monotonic() < deadline and len(values) < int(samples):
            item = self.state.get_tof()
            if item is not None:
                value, stamp, seq = item
                if seq != last_seq:
                    last_seq = seq
                    if (
                        time.monotonic() - stamp <= FRONT_TOF_STALE_SEC
                        and value is not None
                        and math.isfinite(value)
                        and TOF_VALID_MIN_MM <= value <= TOF_VALID_MAX_MM
                    ):
                        values.append(float(value))
            time.sleep(TOF_SCAN_INTERVAL_SEC)

        if not values:
            return None
        return float(statistics.median(values))

    def scan_tof_at_yaw(self, yaw_deg):
        for attempt in range(1, GIMBAL_SCAN_RETRIES + 2):
            if not self.gimbal_goto(yaw_deg, GIMBAL_PITCH_DEG, force=(attempt > 1)):
                self.fault("SCAN", f"gimbal could not reach yaw={yaw_deg:+.1f}", "retry")
                continue
            mm = self.sample_fresh_tof()
            if mm is not None:
                return mm
            self.fault("SCAN", f"no fresh ToF at yaw={yaw_deg:+.1f} attempt {attempt}", "retry")
        return None

    # --------------------------------------------------------
    # SHARP / IR
    # --------------------------------------------------------
    def read_sharp_adc(self):
        left = right = None
        try:
            left = self.sensor_adapter.get_adc(id=SHARP_LEFT_ID, port=SENSOR_PORT)
        except Exception:
            pass
        try:
            right = self.sensor_adapter.get_adc(id=SHARP_RIGHT_ID, port=SENSOR_PORT)
        except Exception:
            pass

        try:
            if left is not None:
                self.left_adc_hist.append(float(left))
        except Exception:
            pass
        try:
            if right is not None:
                self.right_adc_hist.append(float(right))
        except Exception:
            pass

        l = statistics.median(self.left_adc_hist) if self.left_adc_hist else None
        r = statistics.median(self.right_adc_hist) if self.right_adc_hist else None
        return l, r

    def read_sharp_cm(self):
        la, ra = self.read_sharp_adc()
        return adc_to_cm(la, LEFT_CAL), adc_to_cm(ra, RIGHT_CAL), la, ra

    def read_ir_once(self):
        l_raw = r_raw = None
        try:
            l_raw = self.sensor_adapter.get_io(id=IR_LEFT_ID, port=SENSOR_PORT)
        except Exception:
            pass
        try:
            r_raw = self.sensor_adapter.get_io(id=IR_RIGHT_ID, port=SENSOR_PORT)
        except Exception:
            pass
        return l_raw == 0, r_raw == 0, l_raw, r_raw

    def read_ir_filtered(self, samples=IR_FILTER_SAMPLES):
        samples = max(1, int(samples))
        lh = rh = 0
        ll = rr = None
        for _ in range(samples):
            l, r, ll, rr = self.read_ir_once()
            lh += int(l)
            rh += int(r)
            time.sleep(0.012)
        needed = samples // 2 + 1
        return lh >= needed, rh >= needed, ll, rr

    def choose_sharp_authority(self, left_cm, right_cm):
        now = time.monotonic()
        current = self.sharp_authority

        if current is not None and now - self.sharp_authority_since < SHARP_AUTHORITY_HOLD_SEC:
            current_dist = left_cm if current == "LEFT" else right_cm
            if current_dist is not None:
                return current

        if left_cm is None and right_cm is None:
            new = None
        elif left_cm is None:
            new = "RIGHT"
        elif right_cm is None:
            new = "LEFT"
        elif left_cm <= SHARP_AUTHORITY_DANGER_CM or right_cm <= SHARP_AUTHORITY_DANGER_CM:
            new = "LEFT" if left_cm <= right_cm else "RIGHT"
        else:
            lerr = abs(left_cm - CENTER_TARGET_CM)
            rerr = abs(right_cm - CENTER_TARGET_CM)
            new = "LEFT" if lerr <= rerr else "RIGHT"

        if new != self.sharp_authority:
            self.sharp_authority = new
            self.sharp_authority_since = now
        return new

    def sharp_lateral_command(self, left_cm, right_cm):
        auth = self.choose_sharp_authority(left_cm, right_cm)
        if auth == "LEFT":
            d = left_cm
            if d is None:
                return 0.0, auth
            if d < SHARP_FOLLOW_NEAR_CM:
                y = max(SHARP_FOLLOW_MIN_STRAFE_MPS, SHARP_FOLLOW_KP * (SHARP_FOLLOW_NEAR_CM - d))
                if d <= SHARP_EMERGENCY_CM:
                    y = max(y, SHARP_HARD_STRAFE_MPS)
                return clamp(y, 0.0, SHARP_HARD_STRAFE_MPS), auth
            if d > SHARP_FOLLOW_FAR_CM and d < SHARP_AUTHORITY_RELEASE_CM:
                y = -max(SHARP_FOLLOW_MIN_STRAFE_MPS, SHARP_FOLLOW_KP * (d - SHARP_FOLLOW_FAR_CM))
                return clamp(y, -SHARP_MAX_STRAFE_MPS, 0.0), auth
            return 0.0, auth

        if auth == "RIGHT":
            d = right_cm
            if d is None:
                return 0.0, auth
            if d < SHARP_FOLLOW_NEAR_CM:
                y = -max(SHARP_FOLLOW_MIN_STRAFE_MPS, SHARP_FOLLOW_KP * (SHARP_FOLLOW_NEAR_CM - d))
                if d <= SHARP_EMERGENCY_CM:
                    y = min(y, -SHARP_HARD_STRAFE_MPS)
                return clamp(y, -SHARP_HARD_STRAFE_MPS, 0.0), auth
            if d > SHARP_FOLLOW_FAR_CM and d < SHARP_AUTHORITY_RELEASE_CM:
                y = max(SHARP_FOLLOW_MIN_STRAFE_MPS, SHARP_FOLLOW_KP * (d - SHARP_FOLLOW_FAR_CM))
                return clamp(y, 0.0, SHARP_MAX_STRAFE_MPS), auth
            return 0.0, auth

        return 0.0, None

    def ir_dynamic_clearance_recovery(self, left_low, right_low, start_pos, target_yaw, traveled):
        """Dynamic IR recovery assisted by Gimbal ToF + Sharp + odometry.

        Status values:
            CLEARED : side event was noise or enough clearance was created; resume edge.
            NODE    : front geometry now looks like a legitimate node; accept arrival.
            BLOCKED : no safe lateral escape was proven; caller may retreat/rescan.

        Important: this routine never commits maze topology by itself.  If it
        returns NODE, last_stop_probe is populated and the normal DFS arrival
        path commits the stationary probe to the destination cell.
        """
        self.safe_stop()
        self.pid_straight.reset()

        # A single electrical LOW is enough to stop quickly, but not enough to
        # move sideways. Confirm with the normal 2-of-3 filter first.
        l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=IR_FILTER_SAMPLES)
        if not l_low and not r_low:
            print('[IR DYNAMIC] transient LOW cleared after stop -> resume, no strafe')
            self.gimbal_front_down(force=False)
            return 'CLEARED'

        self.align_heading_stationary(
            target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
            tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.06
        )

        def probe_three(tag):
            rays = {}
            self.safe_stop()
            time.sleep(IR_DYNAMIC_RESCAN_SETTLE_SEC)
            for label, yaw_deg, _rel in SCAN_RELATIVE_ORDER:
                mm = self.scan_tof_at_yaw(yaw_deg)
                rays[label] = mm
            print(
                f"[IR {tag}] L={rays.get('LEFT')} F={rays.get('FRONT')} "
                f"R={rays.get('RIGHT')} IR=({int(l_low)},{int(r_low)})"
            )
            return rays

        rays = probe_three('PROBE')
        front = rays.get('FRONT')
        valid = sum(v is not None for v in rays.values())

        # IR can happen at a real dead-end/corner.  If front has become close
        # after meaningful progress, do not shove sideways blindly: promote the
        # stationary L/F/R scan to the same node-candidate path as Front-ToF.
        if (
            front is not None and front <= FRONT_STOP_SCAN_MM
            and traveled >= FRONT_NODE_CAPTURE_MIN_PROGRESS_M
            and valid >= FRONT_NODE_CAPTURE_MIN_VALID_RAYS
        ):
            self.last_stop_probe = dict(rays)
            self.gimbal_front_down(force=True)
            print(f'[IR DYNAMIC] close-front geometry at {traveled:.3f}m -> NODE candidate')
            return 'NODE'

        # Re-check IR after the gimbal scan.  Mechanical vibration/noise often
        # disappears while stationary; in that case no lateral move is needed.
        l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=IR_FILTER_SAMPLES)
        if not l_low and not r_low:
            self.gimbal_front_down(force=True)
            print('[IR DYNAMIC] IR cleared during gimbal verification -> resume')
            return 'CLEARED'

        left_mm = rays.get('LEFT')
        right_mm = rays.get('RIGHT')

        def choose_escape_direction(ir_left, ir_right, side_rays):
            """Return (sign, source_label, dest_label) or None.

            The decision is recomputed whenever IR state changes, so recovery can
            reverse its lateral plan instead of blindly finishing an old nudge.
            """
            lmm = side_rays.get('LEFT')
            rmm = side_rays.get('RIGHT')
            if ir_left and not ir_right:
                return +1.0, 'LEFT', 'RIGHT'
            if ir_right and not ir_left:
                return -1.0, 'RIGHT', 'LEFT'
            if not ir_left and not ir_right:
                return 0.0, None, None
            candidates = []
            if rmm is not None:
                candidates.append((rmm, +1.0, 'LEFT', 'RIGHT'))
            if lmm is not None:
                candidates.append((lmm, -1.0, 'RIGHT', 'LEFT'))
            if not candidates:
                return None
            _best, sgn, src, dst = max(candidates, key=lambda x: x[0])
            return sgn, src, dst

        choice = choose_escape_direction(l_low, r_low, rays)
        if choice is None:
            self.gimbal_front_down(force=True)
            self.fault('IR DYNAMIC', 'IR active but side ToF unavailable', 'no blind strafe')
            return 'BLOCKED'
        sign, source_label, dest_label = choice
        if sign == 0.0:
            self.gimbal_front_down(force=True)
            return 'CLEARED'

        dest_mm = rays.get(dest_label)
        lcm, rcm, _, _ = self.read_sharp_cm()
        dest_cm = rcm if sign > 0.0 else lcm
        if dest_mm is None and dest_cm is None:
            self.gimbal_front_down(force=True)
            self.fault(
                'IR DYNAMIC',
                f'destination {dest_label} has neither usable ToF nor Sharp',
                'no blind lateral motion',
            )
            return 'BLOCKED'
        if (dest_cm is not None and dest_cm <= IR_DEST_SHARP_STOP_CM) or (
            dest_mm is not None and dest_mm <= IR_DYNAMIC_DEST_MIN_MM
        ):
            self.gimbal_front_down(force=True)
            self.fault(
                'IR DYNAMIC',
                f'destination {dest_label} not safe Sharp={dest_cm}cm ToF={dest_mm}mm',
                'do not strafe into opposite wall',
            )
            return 'BLOCKED'

        recovery_anchor = self.current_position()
        if recovery_anchor is None:
            self.gimbal_front_down(force=True)
            self.fault('IR DYNAMIC', 'position unavailable at recovery start', 'no blind strafe')
            return 'BLOCKED'

        started = time.monotonic()
        clear_count = 0
        total_lat = 0.0
        lateral_used = 0.0
        iteration = 0

        while self.running and time.monotonic() - started < IR_DYNAMIC_MAX_SEC:
            iteration += 1
            # Point the gimbal toward the side we are escaping FROM.  During the
            # short pure-strafe segment, live ToF should increase as clearance
            # improves. Front is re-checked after every measured step.
            source_yaw = -90.0 if source_label == 'LEFT' else +90.0
            source_mm = self.scan_tof_at_yaw(source_yaw)

            lcm, rcm, _, _ = self.read_sharp_cm()
            source_cm = lcm if source_label == 'LEFT' else rcm
            dest_cm = rcm if sign > 0.0 else lcm
            if dest_cm is not None and dest_cm <= IR_DEST_SHARP_STOP_CM:
                self.safe_stop()
                self.gimbal_front_down(force=True)
                self.fault('IR DYNAMIC', f'destination Sharp={dest_cm:.1f}cm', 'stop lateral recovery')
                return 'BLOCKED'

            # Dynamic speed: stronger only when the source is genuinely close;
            # taper as IR/ToF/Sharp indicate increasing clearance.
            severity = 0.35
            if source_cm is not None:
                severity = max(severity, clamp((IR_DEST_SHARP_CAUTION_CM - source_cm) / 6.0, 0.0, 1.0))
            if source_mm is not None:
                severity = max(severity, clamp((IR_DYNAMIC_SOURCE_CLEAR_MM - source_mm) / 100.0, 0.0, 1.0))
            speed = IR_DYNAMIC_MIN_SPEED_MPS + severity * (IR_DYNAMIC_MAX_SPEED_MPS - IR_DYNAMIC_MIN_SPEED_MPS)

            step_anchor = self.current_position()
            if step_anchor is None:
                self.safe_stop()
                self.gimbal_front_down(force=True)
                return 'BLOCKED'
            step_deadline = time.monotonic() + 0.75
            while self.running and time.monotonic() < step_deadline:
                pos = self.current_position()
                if pos is None:
                    break
                step_lat = abs(self.cell_lateral_offset(step_anchor, pos, target_yaw))
                total_lat = abs(self.cell_lateral_offset(recovery_anchor, pos, target_yaw))
                if step_lat >= IR_DYNAMIC_STEP_M or total_lat >= IR_DYNAMIC_MAX_LATERAL_M:
                    break

                # Destination Sharp remains the hard interlock while moving.
                lcm2, rcm2, _, _ = self.read_sharp_cm()
                dest2 = rcm2 if sign > 0.0 else lcm2
                if dest2 is not None and dest2 <= IR_DEST_SHARP_STOP_CM:
                    break

                if not self.drive_speed_resilient(
                    x=0.0, y=sign * speed,
                    z=self.yaw_hold_command(target_yaw),
                    timeout=DRIVE_COMMAND_TIMEOUT, label="IR DYNAMIC CMD",
                ):
                    self.safe_stop()
                    self.gimbal_front_down(force=True)
                    return 'BLOCKED'
                time.sleep(CONTROL_DT)

            self.safe_stop()

            # Account real odometry displacement. The safety budget is cumulative,
            # so a left-right oscillation cannot evade the maximum recovery travel.
            pos_step_end = self.current_position(fresh=False)
            if pos_step_end is not None:
                step_done = abs(self.cell_lateral_offset(step_anchor, pos_step_end, target_yaw))
                lateral_used += step_done

            # Verify IR after each odometry-measured step, not after a fixed time.
            l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=IR_FILTER_SAMPLES)
            if not l_low and not r_low:
                clear_count += 1
            else:
                clear_count = 0

            # Re-scan FRONT and destination before another sideways step.
            front = self.scan_tof_at_yaw(0.0)
            dest_yaw = +90.0 if dest_label == 'RIGHT' else -90.0
            dest_mm = self.scan_tof_at_yaw(dest_yaw)
            pos_now = self.current_position(fresh=False)
            if pos_now is not None:
                total_lat = abs(self.cell_lateral_offset(recovery_anchor, pos_now, target_yaw))

            print(
                f'[IR DYNAMIC CTRL] step={iteration} dir={dest_label} speed={speed:.3f} '
                f'lat={total_lat:.3f}m used={lateral_used:.3f}m srcToF={source_mm} '
                f'front={front} destToF={dest_mm} IR=({l_raw},{r_raw}) '
                f'clear={clear_count}/{IR_DYNAMIC_CLEAR_CONFIRM}'
            )

            # If the destination-side IR became LOW, the geometry changed while
            # moving. Stop and re-plan from a fresh L/F/R gimbal scan instead of
            # continuing the old direction.
            destination_low = r_low if sign > 0.0 else l_low
            if destination_low:
                rays = probe_three('REPLAN')
                choice = choose_escape_direction(l_low, r_low, rays)
                if choice is None:
                    self.gimbal_front_down(force=True)
                    return 'BLOCKED'
                new_sign, new_source, new_dest = choice
                if new_sign == 0.0:
                    clear_count = IR_DYNAMIC_CLEAR_CONFIRM
                else:
                    sign, source_label, dest_label = new_sign, new_source, new_dest
                    repl_front = rays.get('FRONT')
                    repl_dest = rays.get(dest_label)
                    if (
                        repl_front is not None and repl_front <= FRONT_STOP_SCAN_MM
                        and traveled >= FRONT_NODE_CAPTURE_MIN_PROGRESS_M
                    ):
                        self.last_stop_probe = dict(rays)
                        self.gimbal_front_down(force=True)
                        return 'NODE'
                    lcm_r, rcm_r, _, _ = self.read_sharp_cm()
                    repl_dest_cm = rcm_r if sign > 0.0 else lcm_r
                    if (
                        (repl_dest is not None and repl_dest <= IR_DYNAMIC_DEST_MIN_MM)
                        or (repl_dest_cm is not None and repl_dest_cm <= IR_DEST_SHARP_STOP_CM)
                        or (repl_dest is None and repl_dest_cm is None)
                    ):
                        self.gimbal_front_down(force=True)
                        self.fault(
                            'IR DYNAMIC',
                            f're-plan destination {dest_label} unsafe ToF={repl_dest} Sharp={repl_dest_cm}',
                            'no blind direction change',
                        )
                        return 'BLOCKED'
                    print(f'[IR DYNAMIC] re-plan -> strafe {dest_label}')
                    continue

            if (
                front is not None and front <= FRONT_STOP_SCAN_MM
                and traveled >= FRONT_NODE_CAPTURE_MIN_PROGRESS_M
            ):
                rays = probe_three('NODE CHECK')
                valid = sum(v is not None for v in rays.values())
                if valid >= FRONT_NODE_CAPTURE_MIN_VALID_RAYS:
                    self.last_stop_probe = dict(rays)
                    self.gimbal_front_down(force=True)
                    return 'NODE'

            if dest_mm is not None and dest_mm <= IR_DYNAMIC_DEST_MIN_MM:
                self.gimbal_front_down(force=True)
                self.fault('IR DYNAMIC', f'destination ToF={dest_mm:.0f}mm became tight', 'stop recovery')
                return 'BLOCKED'

            if clear_count >= IR_DYNAMIC_CLEAR_CONFIRM:
                self.gimbal_front_down(force=True)
                front_check = self.sample_fresh_tof(samples=3, timeout=0.7)
                if front_check is None or front_check >= IR_DYNAMIC_FRONT_MIN_MM:
                    self.pid_straight.reset()
                    print(f'[IR DYNAMIC] cleared after {total_lat:.3f}m lateral adjustment -> resume edge')
                    return 'CLEARED'
                if traveled >= FRONT_NODE_CAPTURE_MIN_PROGRESS_M:
                    rays = probe_three('FINAL NODE CHECK')
                    self.last_stop_probe = dict(rays)
                    self.gimbal_front_down(force=True)
                    return 'NODE'
                return 'BLOCKED'

            if lateral_used >= IR_DYNAMIC_MAX_LATERAL_M:
                break

        self.safe_stop()
        self.gimbal_front_down(force=True)
        self.fault(
            'IR DYNAMIC',
            f'could not clear IR within netLat={total_lat:.3f}m used={lateral_used:.3f}m / {IR_DYNAMIC_MAX_SEC:.1f}s',
            'caller may retreat/rescan',
        )
        return 'BLOCKED'

    # --------------------------------------------------------
    # TRANSLATION / RETREAT
    # --------------------------------------------------------
    @staticmethod
    def xy_distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def cell_forward_progress(start_pos, pos, target_raw_yaw_deg):
        """Signed progress along the requested RUN-time heading.

        RoboMaster position x/y remain expressed in the SDK's original planar
        frame.  The chassis may have been rotated before this Python program was
        started, so RUN-time North is not assumed to be SDK +x.  Instead use the
        raw yaw captured/derived from the RUN-time yaw-zero frame to project the
        displacement onto the requested heading.  This makes lateral strafe and
        pre-RUN chassis orientation irrelevant to 60 cm forward progress.
        """
        dx = float(pos[0]) - float(start_pos[0])
        dy = float(pos[1]) - float(start_pos[1])
        th = math.radians(float(target_raw_yaw_deg))
        return dx * math.cos(th) + dy * math.sin(th)

    @staticmethod
    def cell_lateral_offset(start_pos, pos, target_raw_yaw_deg):
        """Signed rightward offset from the requested RUN-time centreline."""
        dx = float(pos[0]) - float(start_pos[0])
        dy = float(pos[1]) - float(start_pos[1])
        th = math.radians(float(target_raw_yaw_deg))
        # right-unit vector for +x forward, +y right and clockwise-positive yaw
        return -dx * math.sin(th) + dy * math.cos(th)

    @staticmethod
    def _slew(current, target, max_delta):
        return current + clamp(target - current, -abs(max_delta), abs(max_delta))

    def remembered_edge_distance(self, cell, direction):
        key = (tuple(cell), int(direction) % 4)
        value = self.edge_travel_m.get(key)
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if not math.isfinite(value) or value < 0.20 or value > 0.90:
            return None
        return value

    def remember_edge_distance(self, cell, direction, distance_m):
        try:
            d = float(distance_m)
        except Exception:
            return
        if not math.isfinite(d) or d < 0.20 or d > 0.90:
            return
        cell = tuple(cell)
        direction = int(direction) % 4
        nb = neighbor(cell, direction)
        self.edge_travel_m[(cell, direction)] = d
        self.edge_travel_m[(nb, (direction + 2) % 4)] = d

    def shape_lateral_command(self, raw_y, x_cmd, remaining, previous_y):
        """Blend wall-follow out near a node and prevent diagonal stop-in."""
        raw_y = float(raw_y or 0.0)
        x_abs = abs(float(x_cmd))
        remaining = max(0.0, float(remaining))

        if remaining <= LATERAL_ZERO_REMAIN_M or x_abs < 1e-6:
            target = 0.0
        else:
            if remaining < LATERAL_FADE_START_M:
                span = max(1e-6, LATERAL_FADE_START_M - LATERAL_ZERO_REMAIN_M)
                fade = clamp((remaining - LATERAL_ZERO_REMAIN_M) / span, 0.0, 1.0)
                ratio = LATERAL_MAX_RATIO_APPROACH
            else:
                fade = 1.0
                ratio = LATERAL_MAX_RATIO_CRUISE
            target = raw_y * fade
            cap = x_abs * ratio
            target = clamp(target, -cap, +cap)

        max_delta = LATERAL_SLEW_MPS2 * CONTROL_DT
        y = self._slew(float(previous_y), target, max_delta)
        # Do not let slew memory keep lateral motion alive inside the final
        # straight-stop zone.  Force it to zero there.
        if remaining <= LATERAL_ZERO_REMAIN_M:
            y = 0.0
        return y

    def retreat_to_source(self, start_pos, reason, target_yaw=None):
        """Smoothly return to the translation anchor in RUN-time coordinates.

        The old retreat used Euclidean distance and x<0 only.  If Mecanum wall
        following had accumulated lateral drift, reverse-x could no longer reduce
        that residual and the watchdog declared "no progress".  This controller
        closes forward and lateral errors separately while yaw PID holds heading.
        """
        self.safe_stop()
        print(f"[RETREAT] {reason}")
        if target_yaw is None:
            target_yaw = self.yaw_ref_deg
        if target_yaw is None:
            self.fault("RETREAT", "target yaw unavailable", "cannot prove source pose")
            return False

        self.pid_straight.reset()
        self.align_heading_stationary(
            target_yaw, timeout_sec=PRE_MOVE_ALIGN_TIMEOUT_SEC,
            tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.10
        )
        self.pid_straight.reset()

        deadline = time.monotonic() + RETREAT_TIMEOUT_SEC
        ramp_start = time.monotonic()
        last_metric = None
        last_progress_t = time.monotonic()
        last_debug = 0.0

        while self.running and time.monotonic() < deadline:
            pos = self.current_position()
            if pos is None:
                self.safe_stop()
                time.sleep(CONTROL_DT)
                continue

            fwd = self.cell_forward_progress(start_pos, pos, target_yaw)
            lat = self.cell_lateral_offset(start_pos, pos, target_yaw)
            metric = math.hypot(fwd, lat)

            if abs(fwd) <= RETREAT_FORWARD_TOL_M and abs(lat) <= RETREAT_LATERAL_TOL_M:
                self.safe_stop()
                self.align_heading_stationary(
                    target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.10
                )
                print(f"[RETREAT OK] source fwd={fwd:+.3f}m lat={lat:+.3f}m")
                return True

            if last_metric is None or metric < last_metric - 0.008:
                last_progress_t = time.monotonic()
                last_metric = metric
            elif time.monotonic() - last_progress_t > 1.6:
                # Do not immediately abort: if forward is already nearly home,
                # spend the remaining time correcting only lateral displacement.
                if abs(fwd) > RETREAT_SOFT_FORWARD_TOL_M or abs(lat) > RETREAT_SOFT_LATERAL_TOL_M:
                    self.fault(
                        "RETREAT",
                        f"progress slow fwd={fwd:+.3f}m lat={lat:+.3f}m",
                        "continue low-speed pose closure",
                    )
                last_progress_t = time.monotonic()
                last_metric = metric

            # Forward return command.  Ramp in gently from zero so a front-stop
            # never turns into an instantaneous full-speed reverse command.
            if abs(fwd) <= RETREAT_FORWARD_TOL_M:
                x_cmd = 0.0
            else:
                desired = -math.copysign(RETREAT_SPEED_MPS, fwd)
                if abs(fwd) < RETREAT_NEAR_SOURCE_M:
                    desired = -math.copysign(
                        max(RETREAT_MIN_SPEED_MPS, RETREAT_SPEED_MPS * abs(fwd) / RETREAT_NEAR_SOURCE_M),
                        fwd,
                    )
                ramp = clamp((time.monotonic() - ramp_start) / max(1e-6, RETREAT_RAMP_SEC), 0.0, 1.0)
                x_cmd = desired * ramp

            # Close lateral drift independently. Positive lat means robot ended to
            # the RIGHT of the intended line, therefore command negative y.
            if abs(lat) <= RETREAT_LATERAL_TOL_M:
                y_cmd = 0.0
            else:
                y_cmd = clamp(-RETREAT_LATERAL_KP * lat, -RETREAT_LATERAL_MAX_MPS, +RETREAT_LATERAL_MAX_MPS)
                # Near the source give lateral closure priority over x.
                if abs(fwd) <= RETREAT_NEAR_SOURCE_M:
                    x_cmd *= 0.55

            z_cmd = self.yaw_hold_command(target_yaw)
            now = time.monotonic()
            if now - last_debug >= 0.35:
                print(
                    f"[RETREAT CTRL] fwd={fwd:+.3f}m lat={lat:+.3f}m "
                    f"x={x_cmd:+.2f} y={y_cmd:+.2f} z={z_cmd:+.1f}"
                )
                last_debug = now
            if not self.drive_speed_resilient(
                x=x_cmd, y=y_cmd, z=z_cmd, timeout=DRIVE_COMMAND_TIMEOUT, label="RETREAT CMD"
            ):
                break
            time.sleep(CONTROL_DT)

        self.safe_stop()
        pos = self.current_position(fresh=False)
        if pos is not None:
            fwd = self.cell_forward_progress(start_pos, pos, target_yaw)
            lat = self.cell_lateral_offset(start_pos, pos, target_yaw)
            if abs(fwd) <= RETREAT_SOFT_FORWARD_TOL_M and abs(lat) <= RETREAT_SOFT_LATERAL_TOL_M:
                self.align_heading_stationary(
                    target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08
                )
                self.fault(
                    "RETREAT",
                    f"soft-accepted source fwd={fwd:+.3f}m lat={lat:+.3f}m",
                    "resume at source",
                )
                return True
        return False

    def stopped_front_topology_probe(self, traveled, tof_front, target_yaw):
        """Stop completely and inspect L/F/R before deciding to reverse.

        This is intentionally a *probe*, not a map commit.  move_one_cell() uses
        it to decide whether the close wall is consistent with a real node.  Once
        MOVE_ARRIVED is returned, normal scan_cell() performs the authoritative
        map scan at that node.
        """
        self.safe_stop()
        time.sleep(FRONT_NODE_SCAN_SETTLE_SEC)
        self.align_heading_stationary(
            target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
            tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08
        )
        print(
            f"[NODE PROBE] stopped at fwd={traveled:.3f}m front={tof_front:.0f}mm; "
            "Gimbal L/F/R scan"
        )
        result = {}
        valid = 0
        for label, yaw_deg, _rel in SCAN_RELATIVE_ORDER:
            mm = self.scan_tof_at_yaw(yaw_deg)
            result[label] = mm
            if mm is None:
                state = "UNKNOWN"
            else:
                valid += 1
                state = "OPEN" if mm > TOF_OPEN_THRESHOLD_MM else "WALL"
            print(f"  [NODE PROBE] {label:<5} yaw={yaw_deg:+5.0f} ToF={mm} -> {state}")
        self.gimbal_front_down(force=True)
        self.last_stop_probe = dict(result)

        front = result.get("FRONT")
        front_wall = front is not None and front <= TOF_OPEN_THRESHOLD_MM
        states = {
            k: ("UNKNOWN" if v is None else "OPEN" if v > TOF_OPEN_THRESHOLD_MM else "WALL")
            for k, v in result.items()
        }
        dead_end = all(states.get(k) == "WALL" for k in ("LEFT", "FRONT", "RIGHT"))
        side_open = any(states.get(k) == "OPEN" for k in ("LEFT", "RIGHT"))
        node_like = (
            valid >= FRONT_NODE_CAPTURE_MIN_VALID_RAYS
            and front_wall
            and traveled >= FRONT_NODE_CAPTURE_MIN_PROGRESS_M
        )
        print(
            f"[NODE PROBE RESULT] valid={valid}/3 front_wall={front_wall} "
            f"dead_end={dead_end} side_open={side_open} node_like={node_like}"
        )
        return {
            "rays": result, "states": states, "valid": valid,
            "front_wall": front_wall, "dead_end": dead_end,
            "side_open": side_open, "node_like": node_like,
        }

    def _recover_after_translation_exception(self, start_pos, target_yaw, target_distance, detail):
        """Classify physical pose after an unexpected mid-translation exception.

        We first restore telemetry. If odometry proves that the chassis is still
        near the source anchor, return BLOCKED_RETURNED. If it is already near the
        destination anchor, return ARRIVED. Otherwise attempt a controlled retreat.
        Only an unprovable mid-edge pose becomes POSE_UNCERTAIN.
        """
        self.safe_stop()
        self.recover_telemetry(
            require_position=True, require_attitude=True,
            reason=f"translation exception: {detail}",
        )
        pos = self.current_position(fresh=False)
        if start_pos is None or pos is None or target_yaw is None:
            return self.MOVE_POSE_UNCERTAIN

        fwd = self.cell_forward_progress(start_pos, pos, target_yaw)
        lat = self.cell_lateral_offset(start_pos, pos, target_yaw)
        if (
            abs(fwd) <= MOVE_EXCEPTION_SOURCE_FWD_TOL_M
            and abs(lat) <= MOVE_EXCEPTION_SOURCE_LAT_TOL_M
        ):
            self.align_heading_stationary(
                target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08,
            )
            self.fault(
                "MOVE EXCEPTION RECOVER",
                f"proved source anchor fwd={fwd:+.3f}m lat={lat:+.3f}m",
                "retry/defer edge without ending mission",
            )
            return self.MOVE_BLOCKED_RETURNED

        if (
            abs(fwd - target_distance) <= MOVE_EXCEPTION_DEST_FWD_TOL_M
            and abs(lat) <= MOVE_EXCEPTION_DEST_LAT_TOL_M
        ):
            self.align_heading_stationary(
                target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08,
            )
            self.last_move_distance_m = max(0.20, fwd)
            self.fault(
                "MOVE EXCEPTION RECOVER",
                f"proved destination anchor fwd={fwd:+.3f}m lat={lat:+.3f}m",
                "accept arrival",
            )
            return self.MOVE_ARRIVED

        if self.retreat_to_source(
            start_pos, f"unexpected translation exception: {detail}", target_yaw=target_yaw
        ):
            return self.MOVE_BLOCKED_RETURNED
        return self.MOVE_POSE_UNCERTAIN

    def move_one_cell(self, source_cell, abs_dir):
        """Public exception boundary for one physical edge traversal."""
        abs_dir = int(abs_dir) % 4
        start_pos = self.current_position(fresh=False)
        target_yaw = self.desired_yaw_for_heading(abs_dir)
        learned_distance = self.remembered_edge_distance(source_cell, abs_dir)
        target_distance = learned_distance if learned_distance is not None else CELL_LENGTH_M
        try:
            return self._move_one_cell_impl(source_cell, abs_dir)
        except Exception as exc:
            self.safe_stop()
            detail = f"{type(exc).__name__}: {exc}"
            self.fault(
                "MOVE UNEXPECTED", detail,
                "prove source/destination pose, otherwise retreat",
            )
            try:
                return self._recover_after_translation_exception(
                    start_pos, target_yaw, target_distance, detail
                )
            except Exception as recover_exc:
                self.safe_stop()
                self.fault(
                    "MOVE RECOVER",
                    f"{type(recover_exc).__name__}: {recover_exc}",
                    "pose cannot be proven",
                )
                return self.MOVE_POSE_UNCERTAIN

    def _move_one_cell_impl(self, source_cell, abs_dir):
        if not self.pose_trusted or not self.running:
            return self.MOVE_STOPPED

        abs_dir = int(abs_dir) % 4
        self.last_move_distance_m = None
        self.last_stop_probe = None
        learned_distance = self.remembered_edge_distance(source_cell, abs_dir)
        target_distance = learned_distance if learned_distance is not None else CELL_LENGTH_M
        expected_yaw = self.desired_yaw_for_heading(abs_dir)
        if expected_yaw is None:
            self.fault("MOVE PREP", "runtime yaw zero unavailable", "do not translate")
            return self.MOVE_BLOCKED_RETURNED

        # Never translate merely because the discrete heading variable says so.
        # Physically PID-align the chassis to the requested runtime cardinal first.
        self.yaw_ref_deg = expected_yaw
        if not self.align_heading_stationary(
            expected_yaw, timeout_sec=PRE_MOVE_ALIGN_TIMEOUT_SEC,
            tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.12
        ):
            self.fault("MOVE PREP", f"yaw not aligned for {DIR_NAMES[abs_dir]}", "do not translate")
            return self.MOVE_BLOCKED_RETURNED

        if not self.gimbal_front_down(force=False):
            # Do not block a known DFS edge just because pitch settles a few
            # degrees low.  For translation safety, front yaw is the critical
            # condition; a safe downward pitch still keeps ToF looking ahead.
            if self.gimbal_front_safe_for_motion():
                p, y = self.current_gimbal_relative()
                self.fault(
                    "MOVE PREP",
                    f"exact front/down not reached but ToF bearing is safe (p={p:+.1f}, y={y:+.1f})",
                    "continue with forward ToF guard",
                )
            else:
                if self.recover_gimbal_front():
                    self.fault("MOVE PREP", "gimbal front recovered", "continue")
                else:
                    self.fault("MOVE PREP", "gimbal front yaw/ToF pose unsafe", "do not translate")
                    return self.MOVE_BLOCKED_RETURNED

        start_pos = self.current_position()
        if start_pos is None:
            self.fault("MOVE", "no fresh position at cell start", "do not translate")
            return self.MOVE_BLOCKED_RETURNED

        target_yaw = expected_yaw
        self.pid_straight.reset()
        self.sharp_authority = None
        for _ in range(SHARP_FILTER_SAMPLES):
            self.read_sharp_cm()
            time.sleep(0.015)

        start_t = time.monotonic()
        stale_tof_since = None
        yaw_bad_since = None
        last_debug = 0.0
        last_y_cmd = 0.0
        side_escape_since = None
        side_escape_sign = 0.0

        while self.running:
            now = time.monotonic()
            pos = self.current_position()
            if pos is None:
                self.safe_stop()
                if self.recover_telemetry(
                    require_position=True, require_attitude=True,
                    reason="position telemetry stale during translation",
                ):
                    pos = self.current_position()
                    if pos is not None:
                        self.pid_straight.reset()
                        start_t += TELEMETRY_RECOVERY_WAIT_SEC
                        continue
                if self.retreat_to_source(start_pos, "position telemetry became stale"):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN

            traveled = max(0.0, self.cell_forward_progress(start_pos, pos, target_yaw))
            lateral_offset = self.cell_lateral_offset(start_pos, pos, target_yaw)

            # Independent yaw watchdog: PID normally keeps this below ~1 deg.
            # If attitude diverges badly for multiple control cycles, stop before
            # the physical path can cross into the wrong logical grid cell.
            yaw_err = self.yaw_error_deg(target_yaw)
            if yaw_err is None:
                self.safe_stop()
                if self.recover_telemetry(
                    require_position=True, require_attitude=True,
                    reason="attitude telemetry stale during translation",
                ):
                    self.pid_straight.reset()
                    start_t += TELEMETRY_RECOVERY_WAIT_SEC
                    continue
                if self.retreat_to_source(start_pos, "attitude telemetry unavailable during translation"):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN
            if abs(yaw_err) >= MOVE_YAW_ABORT_ERROR_DEG:
                if yaw_bad_since is None:
                    yaw_bad_since = now
                elif now - yaw_bad_since >= MOVE_YAW_ABORT_CONFIRM_SEC:
                    self.safe_stop()
                    self.fault("MOVE YAW", f"diverged by {yaw_err:+.1f}deg at d={traveled:.3f}m", "retreat and rescan")
                    if self.retreat_to_source(start_pos, "yaw divergence during translation"):
                        return self.MOVE_BLOCKED_RETURNED
                    return self.MOVE_POSE_UNCERTAIN
            else:
                yaw_bad_since = None

            if traveled >= target_distance:
                self.safe_stop()
                # Remove the small yaw residual created by wheel inertia before the
                # next node scan.  Failure is logged but arrival position remains valid.
                aligned = self.align_heading_stationary(
                    target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=POST_MOVE_ALIGN_TOL_DEG, settle_sec=0.10
                )
                if not aligned:
                    self.fault("MOVE POST", "arrived but yaw final-settle timed out", "next turn/scan will realign")
                self.last_move_distance_m = traveled
                print(
                    f"[MOVE OK] {source_cell}->{neighbor(source_cell, abs_dir)} "
                    f"d={traveled:.3f}m target={target_distance:.3f}m "
                    f"yaw_err={self.yaw_error_deg(target_yaw)}"
                )
                return self.MOVE_ARRIVED

            if now - start_t >= MAX_CELL_TIME_SEC:
                self.safe_stop()
                if traveled >= target_distance * CELL_SUCCESS_FRACTION:
                    self.align_heading_stationary(target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC, tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08)
                    self.last_move_distance_m = traveled
                    self.fault("MOVE TIMEOUT", f"near node at {traveled:.3f}m", "accept node")
                    return self.MOVE_ARRIVED
                if self.retreat_to_source(start_pos, f"cell timeout at {traveled:.3f}m"):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN

            tof = self.latest_tof()
            if tof is None:
                if stale_tof_since is None:
                    stale_tof_since = now
                self.safe_stop()
                if now - stale_tof_since >= FRONT_TOF_STALE_SEC:
                    fresh = self.sample_fresh_tof(samples=3, timeout=0.8)
                    if fresh is None:
                        if self.recover_telemetry(
                            require_position=True, require_attitude=True, require_tof=True,
                            reason="front ToF stale during translation",
                        ):
                            fresh = self.sample_fresh_tof(samples=3, timeout=0.8)
                        if fresh is None:
                            if self.retreat_to_source(start_pos, "front ToF unavailable during motion"):
                                return self.MOVE_BLOCKED_RETURNED
                            return self.MOVE_POSE_UNCERTAIN
                    tof = fresh
                    stale_tof_since = None
                else:
                    time.sleep(CONTROL_DT)
                    continue
            else:
                stale_tof_since = None

            if tof <= FRONT_STOP_SCAN_MM:
                # Crucial policy: never jump directly from forward motion to
                # reverse merely because the front ToF became close.  Stop, let
                # chassis inertia settle, then rotate the Gimbal ToF to determine
                # whether this is the wall of a legitimate dead-end/corner node.
                probe = self.stopped_front_topology_probe(traveled, tof, target_yaw)

                if traveled >= target_distance * CELL_SUCCESS_FRACTION:
                    self.last_move_distance_m = traveled
                    self.fault(
                        "FRONT NODE",
                        f"ToF={tof:.0f}mm near nominal node at {traveled:.3f}m",
                        "accept after stationary L/F/R probe",
                    )
                    return self.MOVE_ARRIVED

                if probe.get("node_like"):
                    # Sensor-confirmed node before nominal odometry distance.
                    # Remember this physical edge length so DFS backtracking uses
                    # the same anchor instead of blindly reversing 0.60 m.
                    self.last_move_distance_m = max(0.20, traveled)
                    kind = "dead-end" if probe.get("dead_end") else "corner/junction"
                    self.fault(
                        "FRONT NODE",
                        f"sensor-confirmed {kind} at {traveled:.3f}m (front={tof:.0f}mm)",
                        "accept node; authoritative map scan follows",
                    )
                    return self.MOVE_ARRIVED

                # Too early / inconclusive: this is a true mid-edge obstruction.
                # Only now is a return to source justified, and retreat ramps in
                # smoothly after the gimbal has already checked the geometry.
                if self.retreat_to_source(
                    start_pos,
                    f"mid-edge front obstruction ToF={tof:.0f}mm at {traveled:.3f}m",
                    target_yaw=target_yaw,
                ):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN

            # IR is the close-range side emergency layer. A single LOW stops
            # immediately; the dynamic recovery then confirms 2-of-3, uses the
            # Gimbal ToF to inspect L/F/R, and moves only as far sideways as
            # sensor evidence requires. No fixed-duration nudge is used.
            l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=1)
            if l_low or r_low:
                self.safe_stop()
                ir_recovery_t0 = time.monotonic()
                ir_status = self.ir_dynamic_clearance_recovery(
                    l_low, r_low, start_pos, target_yaw, traveled
                )
                # Time spent stationary scanning/clearing IR must not consume the
                # cell-forward watchdog budget.
                start_t += max(0.0, time.monotonic() - ir_recovery_t0)
                if ir_status == 'CLEARED':
                    self.pid_straight.reset()
                    continue
                if ir_status == 'NODE':
                    self.align_heading_stationary(
                        target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                        tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08
                    )
                    self.last_move_distance_m = max(0.20, traveled)
                    self.fault(
                        'IR NODE',
                        f'stationary IR+Gimbal geometry accepted at d={traveled:.3f}m',
                        'accept node and map with stop probe',
                    )
                    return self.MOVE_ARRIVED
                if self.retreat_to_source(
                    start_pos, f'IR dynamic recovery blocked L={l_raw} R={r_raw}',
                    target_yaw=target_yaw,
                ):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN

            left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()
            if (
                left_cm is not None and right_cm is not None
                and left_cm <= SHARP_EMERGENCY_CM and right_cm <= SHARP_EMERGENCY_CM
            ):
                self.safe_stop()
                if traveled >= target_distance * CELL_SUCCESS_FRACTION:
                    self.align_heading_stationary(target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC, tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08)
                    self.last_move_distance_m = traveled
                    return self.MOVE_ARRIVED
                if self.retreat_to_source(start_pos, "both Sharp sensors at emergency floor"):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN

            raw_y_cmd, authority = self.sharp_lateral_command(left_cm, right_cm)
            remaining = max(0.0, target_distance - traveled)
            if remaining <= CELL_APPROACH_SLOW_M:
                ratio = remaining / max(1e-6, CELL_APPROACH_SLOW_M)
                x_cmd = max(CELL_APPROACH_MIN_MPS, FORWARD_SPEED_MPS * ratio)
            else:
                x_cmd = FORWARD_SPEED_MPS

            # Front-distance brake envelope.  The command now decreases smoothly
            # as the wall approaches instead of staying at 0.10 m/s until the
            # hard-stop threshold.  The final stop is handled above by the
            # stationary topology probe.
            if tof < FRONT_BRAKE_START_MM:
                if tof <= FRONT_CRAWL_START_MM:
                    tof_cmd = FRONT_MIN_BRAKE_SPEED_MPS
                else:
                    span = max(1.0, FRONT_BRAKE_START_MM - FRONT_CRAWL_START_MM)
                    alpha = clamp((tof - FRONT_CRAWL_START_MM) / span, 0.0, 1.0)
                    tof_cmd = FRONT_MIN_BRAKE_SPEED_MPS + alpha * (FORWARD_SPEED_MPS - FRONT_MIN_BRAKE_SPEED_MPS)
                x_cmd = min(x_cmd, tof_cmd)

            # A genuinely close side wall is handled as a pure lateral escape.
            # Never mix x forward with a large y correction; that was the main
            # source of the visible diagonal motion immediately before stopping.
            # Hysteretic side-escape state: once armed at <= trigger, remain in
            # pure-strafe mode until that SAME side reaches the clear threshold.
            escape_dist = None
            if side_escape_sign == 0.0:
                if left_cm is not None and left_cm <= SHARP_SIDE_ESCAPE_TRIGGER_CM:
                    side_escape_sign, escape_dist = +1.0, left_cm
                if right_cm is not None and right_cm <= SHARP_SIDE_ESCAPE_TRIGGER_CM:
                    if escape_dist is None or right_cm < escape_dist:
                        side_escape_sign, escape_dist = -1.0, right_cm
                if side_escape_sign != 0.0:
                    side_escape_since = now
                    self.fault(
                        "SIDE ESCAPE",
                        f"Sharp={escape_dist:.1f}cm at forward={traveled:.3f}m lat={lateral_offset:+.3f}m",
                        "pause forward and strafe away",
                    )
            else:
                watched = left_cm if side_escape_sign > 0.0 else right_cm
                if watched is None or watched >= SHARP_SIDE_ESCAPE_CLEAR_CM:
                    side_escape_sign = 0.0
                    side_escape_since = None
                    last_y_cmd = 0.0

            if side_escape_sign != 0.0:
                # If the destination side is also cramped, do not squeeze the
                # chassis sideways between two close walls. Retreat and rescan.
                destination = right_cm if side_escape_sign > 0.0 else left_cm
                if destination is not None and destination <= SHARP_SIDE_ESCAPE_TRIGGER_CM:
                    self.safe_stop()
                    if self.retreat_to_source(start_pos, "both sides too tight for safe lateral escape"):
                        return self.MOVE_BLOCKED_RETURNED
                    return self.MOVE_POSE_UNCERTAIN
                if side_escape_since is not None and now - side_escape_since > SHARP_SIDE_ESCAPE_MAX_SEC:
                    self.safe_stop()
                    if self.retreat_to_source(start_pos, "side-wall escape timed out"):
                        return self.MOVE_BLOCKED_RETURNED
                    return self.MOVE_POSE_UNCERTAIN
                x_cmd = 0.0
                y_cmd = side_escape_sign * SHARP_SIDE_ESCAPE_SPEED_MPS
                last_y_cmd = y_cmd
            else:
                y_cmd = self.shape_lateral_command(raw_y_cmd, x_cmd, remaining, last_y_cmd)
                last_y_cmd = y_cmd

            z_cmd = self.yaw_hold_command(target_yaw)
            if not self.drive_speed_resilient(
                x=x_cmd, y=y_cmd, z=z_cmd, timeout=DRIVE_COMMAND_TIMEOUT, label="MOVE CMD"
            ):
                self.safe_stop()
                if self.recover_telemetry(
                    require_position=True, require_attitude=True,
                    reason="drive command transport failure",
                ):
                    self.pid_straight.reset()
                    continue
                if self.retreat_to_source(start_pos, "drive command exception"):
                    return self.MOVE_BLOCKED_RETURNED
                return self.MOVE_POSE_UNCERTAIN

            if now - last_debug >= 0.35:
                print(
                    f"[CTRL] fwd={traveled:.3f}m lat={lateral_offset:+.3f}m ToF={tof:.0f}mm "
                    f"L={left_cm} R={right_cm} auth={authority or '-'} "
                    f"x={x_cmd:+.2f} y={y_cmd:+.2f} z={z_cmd:+.1f} "
                    f"yawErr={fmt_deg(yaw_err)} logicalYaw={fmt_deg(self.logical_yaw_deg())} "
                    f"slip={int(self.chassis_slip_detected())}"
                )
                last_debug = now
            time.sleep(CONTROL_DT)

        self.safe_stop()
        return self.MOVE_STOPPED

    # --------------------------------------------------------
    # CELL SCAN
    # --------------------------------------------------------
    def set_edge_state(self, cell, direction, state):
        """Store one sensed edge state.

        A mere ToF observation is not allowed to overwrite a contradictory
        reciprocal WALL/OPEN already measured from the other cell.  Physical
        traversal uses mark_traversed_open(), which is stronger evidence and
        forces both directions OPEN.
        """
        direction = int(direction) % 4
        cell = tuple(cell)
        state = str(state)
        self.edge_state[(cell, direction)] = state
        nb = neighbor(cell, direction)
        opposite = (direction + 2) % 4

        existing = self.edge_state.get((nb, opposite))
        if state == "OPEN":
            if existing not in ("WALL", "BLOCKED"):
                self.edge_state[(nb, opposite)] = "OPEN"
        elif state in ("WALL", "BLOCKED"):
            if existing != "OPEN":
                self.edge_state[(nb, opposite)] = state

    def mark_traversed_open(self, cell, direction):
        """Physical traversal is definitive evidence that an edge is OPEN."""
        direction = int(direction) % 4
        cell = tuple(cell)
        nb = neighbor(cell, direction)
        self.edge_state[(cell, direction)] = "OPEN"
        self.edge_state[(nb, (direction + 2) % 4)] = "OPEN"

    def root_back_scan(self):
        if not ROOT_BACK_SCAN_ENABLED or self.base_yaw_deg is None:
            return None
        original_heading = self.heading
        original_target = self.desired_yaw_for_heading(original_heading)
        back_target = wrap_deg(original_target + 180.0)
        print("[ROOT BACK] temporary 180-degree chassis scan")

        turned = self.turn_closed_loop(back_target, TURN_TIMEOUT_180_SEC)
        if not turned:
            self.fault("ROOT BACK", "could not reach back heading", "restore runtime North")
        mm = None
        if turned:
            self.gimbal_front_down(force=True)
            mm = self.sample_fresh_tof()

        restored = False
        for attempt in range(ROOT_BACK_RESTORE_RETRIES + 1):
            if self.turn_closed_loop(
                original_target,
                TURN_TIMEOUT_180_SEC * (1.0 if attempt == 0 else TURN_RECOVERY_TIMEOUT_SCALE),
            ):
                restored = True
                break
            self.fault("ROOT BACK", f"restore attempt {attempt+1} failed", "retry")

        self.heading = original_heading
        self.yaw_ref_deg = original_target
        if restored:
            # turn_closed_loop already performs a fine settle, but verify the
            # mission cardinal once more before the first DFS translation.
            restored = self.align_heading_stationary(
                original_target, timeout_sec=STATIONARY_ALIGN_TIMEOUT_SEC,
                tolerance_deg=STATIONARY_SETTLE_TOL_DEG, settle_sec=0.12,
            )

        if not restored:
            snapped = self.recover_to_nearest_cardinal()
            if snapped != original_heading:
                self.enter_safe_pause("root BACK scan could not restore a trusted runtime heading")
                return mm

        self.pid_straight.reset()
        self.gimbal_front_down(force=True)
        return mm

    def commit_stop_probe_to_cell(self, cell, parent_cell, probe_rays):
        """Commit an already-performed stopped L/F/R probe as this cell's map.

        The chassis has already been accepted at `cell`, so repeating the same
        mechanical scan is unnecessary.  BACK is definitive because it is the
        edge physically traversed from parent_cell.
        """
        if not isinstance(probe_rays, dict):
            return False
        cell = tuple(cell)
        parent_cell = tuple(parent_cell) if parent_cell is not None else None
        ordered_open = []
        scan = {}
        valid = 0

        print(f"[NODE MAP] commit stopped Gimbal probe -> cell={cell} heading={DIR_NAMES[self.heading]}")
        for label, _yaw_deg, rel in SCAN_RELATIVE_ORDER:
            mm = probe_rays.get(label)
            scan[label] = mm
            abs_dir = (self.heading + rel) % 4
            if mm is None:
                state = "UNKNOWN"
            else:
                valid += 1
                if mm > TOF_OPEN_THRESHOLD_MM:
                    state = "OPEN"
                    ordered_open.append(abs_dir)
                else:
                    state = "WALL"
            self.set_edge_state(cell, abs_dir, state)
            print(f"  [NODE MAP] {label:<5} ToF={mm} -> {state} ({DIR_NAMES[abs_dir]})")

        if parent_cell is not None:
            back_dir = direction_between(cell, parent_cell)
            if back_dir is not None:
                self.mark_traversed_open(cell, back_dir)
                scan["BACK"] = "TRAVERSED"
                if back_dir not in ordered_open:
                    ordered_open.append(back_dir)

        seen = set()
        ordered_open = [d for d in ordered_open if not (d in seen or seen.add(d))]
        self.cell_scan_mm[cell] = scan
        self.open_dirs[cell] = ordered_open
        print("  [NODE MAP] OPEN:", [DIR_NAMES[d] for d in ordered_open])
        return valid >= FRONT_NODE_CAPTURE_MIN_VALID_RAYS

    def scan_cell(self, cell):
        """Exception-contained stationary topology scan."""
        last_exc = None
        for attempt in range(1, SCAN_EXCEPTION_RETRIES + 2):
            try:
                return self._scan_cell_impl(cell)
            except Exception as exc:
                last_exc = exc
                self.safe_stop()
                self.fault(
                    "SCAN UNEXPECTED",
                    f"cell={cell} attempt {attempt}: {type(exc).__name__}: {exc}",
                    "recover telemetry/gimbal and retry stationary scan",
                )
                self.recover_telemetry(
                    require_position=True, require_attitude=True, require_tof=True, require_gimbal=True,
                    reason=f"scan_cell {cell}",
                )
                self.recover_gimbal_front()
                time.sleep(0.10 * attempt)

        # A scan failure at a known node must not crash DFS. Preserve the parent
        # edge (if any) so the robot can still backtrack; leave all other bearings
        # UNKNOWN for later recovery.
        ordered_open = []
        cell = tuple(cell)
        p = self.parent.get(cell)
        if p is not None:
            back_dir = direction_between(cell, p)
            if back_dir is not None:
                self.mark_traversed_open(cell, back_dir)
                ordered_open.append(back_dir)
        for d in range(4):
            if (cell, d) not in self.edge_state:
                self.set_edge_state(cell, d, "UNKNOWN")
        self.open_dirs[cell] = ordered_open
        self.cell_scan_mm[cell] = {"ERROR": None}
        self.fault(
            "SCAN FALLBACK",
            f"cell={cell}: {type(last_exc).__name__ if last_exc else 'unknown'}",
            "continue DFS with UNKNOWN edges / parent backtrack",
        )
        return ordered_open

    def _scan_cell_impl(self, cell):
        print(f"\n[SCAN] cell={cell} heading={DIR_NAMES[self.heading]}")
        ordered_open = []
        scan = {}

        # IR at a node is informational only.  Never slide the chassis during a
        # topology scan; that would move the node anchor.
        l_low, r_low, l_raw, r_raw = self.read_ir_filtered()
        if l_low or r_low:
            print(f"[IR NODE] L={l_raw} R={r_raw} -> HOLD POSITION, scan only")
            self.safe_stop()

        for label, yaw_deg, rel in SCAN_RELATIVE_ORDER:
            mm = self.scan_tof_at_yaw(yaw_deg)
            abs_dir = (self.heading + rel) % 4
            scan[label] = mm
            if mm is None:
                state = "UNKNOWN"
            elif mm > TOF_OPEN_THRESHOLD_MM:
                state = "OPEN"
                ordered_open.append(abs_dir)
            else:
                state = "WALL"
            self.set_edge_state(cell, abs_dir, state)
            print(f"  {label:<5} yaw={yaw_deg:+5.0f} ToF={mm} -> {state} ({DIR_NAMES[abs_dir]})")

        # A transient gimbal/ToF miss must not make DFS immediately backtrack
        # from a cell that may still have an unexplored route.  Retry ONLY the
        # UNKNOWN bearings while the chassis stays parked on the node anchor.
        for recover_pass in range(1, UNKNOWN_EDGE_RESCAN_PASSES + 1):
            missing = [item for item in SCAN_RELATIVE_ORDER if scan.get(item[0]) is None]
            if not missing:
                break
            print(
                f"[SCAN RECOVER] pass {recover_pass}/{UNKNOWN_EDGE_RESCAN_PASSES} "
                f"retry UNKNOWN={[m[0] for m in missing]}"
            )
            self.safe_stop()
            time.sleep(UNKNOWN_EDGE_RESCAN_SETTLE_SEC)
            for label, yaw_deg, rel in missing:
                mm = self.scan_tof_at_yaw(yaw_deg)
                if mm is None:
                    continue
                scan[label] = mm
                abs_dir = (self.heading + rel) % 4
                state = "OPEN" if mm > TOF_OPEN_THRESHOLD_MM else "WALL"
                self.set_edge_state(cell, abs_dir, state)
                if state == "OPEN" and abs_dir not in ordered_open:
                    ordered_open.append(abs_dir)
                print(
                    f"  [SCAN RECOVER] {label:<5} yaw={yaw_deg:+5.0f} "
                    f"ToF={mm} -> {state} ({DIR_NAMES[abs_dir]})"
                )

        p = self.parent.get(cell)
        if p is not None:
            back_dir = direction_between(cell, p)
            if back_dir is not None:
                self.mark_traversed_open(cell, back_dir)
                if back_dir not in ordered_open:
                    ordered_open.append(back_dir)
        elif tuple(cell) == self.root and ROOT_BACK_SCAN_ENABLED and self.pose_trusted:
            mm = self.root_back_scan()
            scan["BACK"] = mm
            back_dir = (self.heading + REL_BACK) % 4
            if mm is None:
                self.set_edge_state(cell, back_dir, "UNKNOWN")
            elif mm > TOF_OPEN_THRESHOLD_MM:
                self.mark_traversed_open(cell, back_dir)
                if back_dir not in ordered_open:
                    ordered_open.append(back_dir)
            else:
                self.set_edge_state(cell, back_dir, "WALL")
            print(f"  BACK  chassis-180 ToF={mm} -> {self.edge_state.get((tuple(cell), back_dir))} ({DIR_NAMES[back_dir]})")

        self.gimbal_front_down(force=True)
        self.cell_scan_mm[tuple(cell)] = scan
        # Deduplicate while preserving scan/parent order.
        seen = set()
        ordered_open = [d for d in ordered_open if not (d in seen or seen.add(d))]
        self.open_dirs[tuple(cell)] = ordered_open
        print("  OPEN:", [DIR_NAMES[d] for d in ordered_open])
        return ordered_open

    # --------------------------------------------------------
    # MAP RENDER / SAVE
    # --------------------------------------------------------
    def mapped_cells(self):
        cells = set(self.visited)
        cells.update(self.open_dirs.keys())
        for (cell, d), state in self.edge_state.items():
            cells.add(cell)
            if state == "OPEN" and neighbor(cell, d) in self.visited:
                cells.add(neighbor(cell, d))
        return cells

    def build_map_payload(self, final=False):
        edge_rows = []
        for (cell, d), state in sorted(self.edge_state.items(), key=lambda x: (x[0][0][1], x[0][0][0], x[0][1])):
            edge_rows.append({
                "cell": [cell[0], cell[1]],
                "dir": DIR_NAMES[d],
                "dir_index": d,
                "state": state,
            })
        return {
            "schema": "robomaster_dfs_map_only",
            "version": 1,
            "created_at": self.map_created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "complete": bool(final and self.map_complete),
            "safe_pause_reason": self.safe_pause_reason,
            "root": [0, 0],
            "current": [self.current[0], self.current[1]],
            "heading": DIR_NAMES[self.heading],
            "grid_tile_m": GRID_TILE_M,
            "cell_length_m": CELL_LENGTH_M,
            "tof_open_threshold_mm": TOF_OPEN_THRESHOLD_MM,
            "visited": [[x, y] for x, y in sorted(self.visited)],
            "parent": {
                f"{c[0]},{c[1]}": (None if p is None else [p[0], p[1]])
                for c, p in self.parent.items()
            },
            "open_dirs": {
                f"{c[0]},{c[1]}": [DIR_NAMES[d] for d in dirs]
                for c, dirs in self.open_dirs.items()
            },
            "edges": edge_rows,
            "cell_scan_mm": {
                f"{c[0]},{c[1]}": vals for c, vals in self.cell_scan_mm.items()
            },
            "edge_travel_m": [
                {
                    "cell": [c[0], c[1]],
                    "dir": DIR_NAMES[d],
                    "distance_m": round(float(dist), 4),
                }
                for (c, d), dist in sorted(
                    self.edge_travel_m.items(),
                    key=lambda item: (item[0][0][1], item[0][0][0], item[0][1]),
                )
            ],
            "deferred_edges": [
                {"cell": [c[0], c[1]], "dir": DIR_NAMES[d]}
                for c, d in sorted(self.deferred_edges)
            ],
            "faults": list(self.fault_log),
        }

    def edge_symbol_state(self, cell, d):
        return self.edge_state.get((tuple(cell), d), "UNKNOWN")

    def render_ascii_map(self):
        cells = self.mapped_cells()
        if not cells:
            return "<empty map>\n"
        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        out = []
        for y in range(max_y, min_y - 1, -1):
            # North walls
            top = []
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                state = self.edge_symbol_state(cell, 0)
                top.append("+" + ("   " if state == "OPEN" else "---" if state in ("WALL", "BLOCKED") else " ? "))
            out.append("".join(top) + "+")

            mid = []
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                west = self.edge_symbol_state(cell, 3)
                mid.append(" " if west == "OPEN" else "|" if west in ("WALL", "BLOCKED") else "?")
                if cell == self.current:
                    glyph = "^>v<"[self.heading]
                    mid.append(f" {glyph} ")
                elif cell == self.root:
                    mid.append(" S ")
                elif cell in self.visited:
                    mid.append(" . ")
                else:
                    mid.append("   ")
            east = self.edge_symbol_state((max_x, y), 1)
            mid.append(" " if east == "OPEN" else "|" if east in ("WALL", "BLOCKED") else "?")
            out.append("".join(mid))

        bottom = []
        y = min_y
        for x in range(min_x, max_x + 1):
            state = self.edge_symbol_state((x, y), 2)
            bottom.append("+" + ("   " if state == "OPEN" else "---" if state in ("WALL", "BLOCKED") else " ? "))
        out.append("".join(bottom) + "+")
        return "\n".join(out) + "\n"

    def render_svg_map(self):
        cells = self.mapped_cells()
        if not cells:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="120"><text x="20" y="60">empty map</text></svg>\n'

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)
        cell_px = 70
        pad = 30
        width = (max_x - min_x + 1) * cell_px + pad * 2
        height = (max_y - min_y + 1) * cell_px + pad * 2

        def xy(cell):
            x, y = cell
            sx = pad + (x - min_x) * cell_px
            sy = pad + (max_y - y) * cell_px
            return sx, sy

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g stroke-linecap="round" font-family="monospace">',
        ]

        for cell in cells:
            sx, sy = xy(cell)
            if cell in self.visited:
                lines.append(f'<rect x="{sx+3}" y="{sy+3}" width="{cell_px-6}" height="{cell_px-6}" fill="#f4f4f4" stroke="none"/>')
            for d in range(4):
                state = self.edge_symbol_state(cell, d)
                if state == "OPEN":
                    continue
                if d == 0:
                    x1, y1, x2, y2 = sx, sy, sx + cell_px, sy
                elif d == 1:
                    x1, y1, x2, y2 = sx + cell_px, sy, sx + cell_px, sy + cell_px
                elif d == 2:
                    x1, y1, x2, y2 = sx, sy + cell_px, sx + cell_px, sy + cell_px
                else:
                    x1, y1, x2, y2 = sx, sy, sx, sy + cell_px
                if state in ("WALL", "BLOCKED"):
                    dash = ""
                    stroke = "black"
                    sw = 4
                else:
                    dash = ' stroke-dasharray="5,5"'
                    stroke = "#999"
                    sw = 2
                lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{dash}/>' )

            label = "S" if cell == self.root else f"{cell[0]},{cell[1]}"
            lines.append(f'<text x="{sx+cell_px/2}" y="{sy+cell_px/2+5}" text-anchor="middle" font-size="12">{label}</text>')

        sx, sy = xy(self.current)
        cx, cy = sx + cell_px/2, sy + cell_px/2
        arrow = {0: (0, -18), 1: (18, 0), 2: (0, 18), 3: (-18, 0)}[self.heading]
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="none" stroke="black" stroke-width="2"/>')
        lines.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+arrow[0]}" y2="{cy+arrow[1]}" stroke="black" stroke-width="3"/>')
        lines.append('</g></svg>')
        return "\n".join(lines) + "\n"

    def save_map(self, final=False):
        try:
            MAP_DIR.mkdir(parents=True, exist_ok=True)
            payload = self.build_map_payload(final=final)
            atomic_write_text(MAP_LATEST_JSON, json.dumps(payload, ensure_ascii=False, indent=2))
            atomic_write_text(MAP_LATEST_ASCII, self.render_ascii_map())
            atomic_write_text(MAP_LATEST_SVG, self.render_svg_map())
            print(f"[MAP SAVE] visited={len(self.visited)} complete={bool(final and self.map_complete)} -> {MAP_LATEST_JSON}")
            return True
        except Exception as exc:
            self.fault("MAP SAVE", f"{type(exc).__name__}: {exc}", "keep mission state in memory")
            return False

    # --------------------------------------------------------
    # DFS CORE
    # --------------------------------------------------------
    def edge_key(self, cell, direction):
        return (tuple(cell), int(direction) % 4)

    def edge_is_deferred(self, cell, direction):
        return self.edge_key(cell, direction) in self.deferred_edges

    def defer_edge(self, cell, direction, reason, blocked=False):
        key = self.edge_key(cell, direction)
        self.deferred_edges.add(key)
        self.set_edge_state(cell, direction, "BLOCKED" if blocked else "DEFERRED")
        self.fault("DFS EDGE", f"{cell}->{DIR_NAMES[direction]}: {reason}", "skip edge and continue DFS")
        if direction in self.open_dirs.get(tuple(cell), []):
            self.open_dirs[tuple(cell)] = [d for d in self.open_dirs[tuple(cell)] if d != direction]
        if MAP_AUTOSAVE:
            self.save_map(final=False)

    def increment_edge_failure(self, cell, direction, reason):
        key = self.edge_key(cell, direction)
        n = self.edge_failures.get(key, 0) + 1
        self.edge_failures[key] = n
        self.fault("EDGE RETRY", f"{cell}->{DIR_NAMES[direction]} failure {n}/{EDGE_MAX_FAILURES}: {reason}", "rescan/retry")
        if n >= EDGE_MAX_FAILURES:
            self.defer_edge(cell, direction, f"repeated motion failure: {reason}", blocked=True)
            return False
        # Force a fresh topology scan before retrying this cell.
        self.open_dirs.pop(tuple(cell), None)
        return True

    def global_limits_ok(self):
        if len(self.visited) > MAX_VISITED_CELLS:
            self.enter_safe_pause(f"runaway guard: visited cells exceeded {MAX_VISITED_CELLS}")
            return False
        if self.mission_start_t is not None and time.monotonic() - self.mission_start_t > MAX_MISSION_SEC:
            if MISSION_WATCHDOG_HARD_STOP:
                self.enter_safe_pause(f"mission watchdog exceeded {MAX_MISSION_SEC}s")
                return False
            if not self._mission_watchdog_warned:
                self._mission_watchdog_warned = True
                self.fault(
                    "MISSION WATCHDOG",
                    f"elapsed time exceeded {MAX_MISSION_SEC}s",
                    "autosave checkpoint and continue (hard stop disabled)",
                )
                self.save_map(final=False)
        return True

    def try_explore_edge(self, cell, direction, next_cell):
        key = self.edge_key(cell, direction)
        if self.edge_is_deferred(cell, direction):
            return False

        if not self.turn_to_direction(direction):
            n = self.turn_failures.get(key, 0) + 1
            self.turn_failures[key] = n
            self.fault("DFS TURN", f"{cell}->{DIR_NAMES[direction]} failed {n}/{TURN_EDGE_MAX_FAILURES}", "defer if repeated")
            if n >= TURN_EDGE_MAX_FAILURES:
                self.defer_edge(cell, direction, "repeated closed-loop turn failure", blocked=False)
            return False

        result = self.move_one_cell(cell, direction)
        if result == self.MOVE_ARRIVED:
            self.mark_traversed_open(cell, direction)
            if self.last_move_distance_m is not None:
                self.remember_edge_distance(cell, direction, self.last_move_distance_m)
            self.edge_failures.pop(key, None)
            self.turn_failures.pop(key, None)
            self.current = next_cell
            self.visited.add(next_cell)
            self.parent.setdefault(next_cell, cell)
            if self.last_stop_probe is not None:
                # The stop-probe is already a stationary L/F/R scan at the
                # accepted node.  Commit it now so a dead-end is mapped before
                # DFS immediately chooses to backtrack.
                if not self.commit_stop_probe_to_cell(next_cell, cell, self.last_stop_probe):
                    self.fault(
                        "NODE MAP",
                        f"probe at {next_cell} had too little valid ToF data",
                        "normal scan_cell will retry",
                    )
                    self.open_dirs.pop(tuple(next_cell), None)
            return True

        if result == self.MOVE_BLOCKED_RETURNED:
            self.current = cell
            self.increment_edge_failure(cell, direction, "translation aborted but source pose recovered")
            return False

        if result == self.MOVE_POSE_UNCERTAIN:
            self.enter_safe_pause(f"pose uncertain while traversing {cell}->{next_cell}")
            return False

        return False

    def find_visited_route(self, start, goal, excluded_edges=None):
        """BFS over already-visited OPEN edges, used only as backtrack fallback."""
        start, goal = tuple(start), tuple(goal)
        excluded = set(excluded_edges or ())
        if start == goal:
            return [start]
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            for d in range(4):
                nb = neighbor(cur, d)
                key = (cur, d)
                rev = (nb, (d + 2) % 4)
                if key in excluded or rev in excluded:
                    continue
                if nb not in self.visited:
                    continue
                if self.edge_state.get(key) != "OPEN":
                    continue
                if nb in prev:
                    continue
                prev[nb] = cur
                if nb == goal:
                    q.clear()
                    break
                q.append(nb)
        if goal not in prev:
            return None
        route = []
        cur = goal
        while cur is not None:
            route.append(cur)
            cur = prev[cur]
        return list(reversed(route))

    def navigate_known_route(self, route):
        """Traverse a visited-cell route without changing DFS parent links."""
        if not route or tuple(route[0]) != tuple(self.current):
            return False
        for src, dst in zip(route, route[1:]):
            d = direction_between(src, dst)
            if d is None:
                return False
            if not self.turn_to_direction(d):
                return False
            result = self.move_one_cell(src, d)
            if result != self.MOVE_ARRIVED:
                if result == self.MOVE_BLOCKED_RETURNED:
                    self.current = src
                return False
            self.current = dst
            self.mark_traversed_open(src, d)
            if self.last_move_distance_m is not None:
                self.remember_edge_distance(src, d, self.last_move_distance_m)
        return True

    def backtrack_to_parent(self, cell, parent):
        d = direction_between(cell, parent)
        if d is None:
            self.enter_safe_pause(f"invalid DFS parent relation {cell}->{parent}")
            return False

        for attempt in range(1, EDGE_MAX_FAILURES + 2):
            if not self.turn_to_direction(d):
                self.fault("BACKTRACK TURN", f"{cell}->{parent} attempt {attempt}", "retry")
                continue
            result = self.move_one_cell(cell, d)
            if result == self.MOVE_ARRIVED:
                self.current = parent
                self.mark_traversed_open(cell, d)
                if self.last_move_distance_m is not None:
                    # Keep the shorter of two close measurements only if they are
                    # reasonably consistent; otherwise retain the original edge
                    # anchor learned on exploration.
                    old_d = self.remembered_edge_distance(cell, d)
                    new_d = self.last_move_distance_m
                    if old_d is None or abs(old_d - new_d) <= 0.10:
                        self.remember_edge_distance(cell, d, (new_d if old_d is None else 0.5 * (old_d + new_d)))
                return True
            if result == self.MOVE_BLOCKED_RETURNED:
                self.current = cell
                self.fault("BACKTRACK MOVE", f"{cell}->{parent} attempt {attempt} blocked", "retry known parent edge")
                time.sleep(0.20)
                continue
            if result == self.MOVE_POSE_UNCERTAIN:
                break

        failed_dir = direction_between(cell, parent)
        excluded = set()
        if failed_dir is not None:
            excluded.add((tuple(cell), failed_dir))
            excluded.add((tuple(parent), (failed_dir + 2) % 4))
        route = self.find_visited_route(cell, parent, excluded_edges=excluded)
        if route and len(route) > 2:
            self.fault(
                "BACKTRACK REROUTE",
                f"direct parent edge unavailable; alternate visited route={route}",
                "navigate alternate OPEN path",
            )
            if self.navigate_known_route(route):
                self.current = parent
                return True

        self.enter_safe_pause(f"known DFS parent edge could not be traversed safely: {cell}->{parent}")
        return False

    def reconstruct_dfs_stack(self):
        """Rebuild the root->current DFS ancestry after a contained Python fault."""
        cur = tuple(self.current)
        chain = []
        seen = set()
        while cur is not None:
            if cur in seen:
                return None
            seen.add(cur)
            chain.append(cur)
            cur = self.parent.get(cur)
        chain.reverse()
        if not chain or chain[0] != self.root:
            return None
        return chain

    def run_dfs(self):
        """Mission-level exception containment with bounded in-place resume."""
        resume = False
        for attempt in range(DFS_EXCEPTION_RESTARTS + 1):
            try:
                return self._run_dfs_impl(resume=resume)
            except Exception as exc:
                self.safe_stop()
                self.fault(
                    "DFS UNEXPECTED",
                    f"attempt {attempt + 1}/{DFS_EXCEPTION_RESTARTS + 1}: {type(exc).__name__}: {exc}",
                    "recover node telemetry/cardinal and resume DFS state",
                )
                if attempt >= DFS_EXCEPTION_RESTARTS:
                    break
                try:
                    telemetry_ok = self.recover_telemetry(
                        require_position=True, require_attitude=True,
                        reason="DFS top-level exception",
                    )
                    cardinal = self.recover_to_nearest_cardinal() if telemetry_ok else None
                    if not telemetry_ok or cardinal is None:
                        break
                    stack = self.reconstruct_dfs_stack()
                    if stack is None:
                        break
                    self.save_map(final=False)
                    resume = True
                    self.fault(
                        "DFS RESUME",
                        f"recovered at logical cell={self.current} heading={DIR_NAMES[self.heading]}",
                        "resume existing stack/map",
                    )
                except Exception as recover_exc:
                    self.fault(
                        "DFS RESUME",
                        f"{type(recover_exc).__name__}: {recover_exc}",
                        "cannot prove resumable node state",
                    )
                    break

        self.enter_safe_pause("DFS encountered repeated unexpected exceptions and could not resume safely")
        return False

    def _run_dfs_impl(self, resume=False):
        if not resume:
            self.visited = {self.root}
            self.parent = {self.root: None}
            self.current = self.root
            self.map_complete = False
            stack = [self.root]
            print("\n[DFS] START MAP-ONLY")
        else:
            stack = self.reconstruct_dfs_stack()
            if not stack:
                raise ValueError("cannot reconstruct DFS ancestry for resume")
            print(f"\n[DFS] RESUME cell={self.current} stack={stack}")

        while self.running and self.pose_trusted and stack:
            if not self.global_limits_ok():
                break

            cell = stack[-1]
            self.current = cell

            if cell not in self.open_dirs:
                self.scan_cell(cell)
                if not self.pose_trusted:
                    break
                if MAP_AUTOSAVE:
                    self.save_map(final=False)

            next_dir = None
            next_cell = None
            for d in self.open_dirs.get(cell, []):
                if self.edge_is_deferred(cell, d):
                    continue
                nb = neighbor(cell, d)
                if nb not in self.visited:
                    next_dir = d
                    next_cell = nb
                    break

            if next_cell is not None:
                print(f"\n[DFS] EXPLORE {cell} -> {next_cell} dir={DIR_NAMES[next_dir]}")
                if self.try_explore_edge(cell, next_dir, next_cell):
                    stack.append(next_cell)
                    if MAP_AUTOSAVE:
                        self.save_map(final=False)
                # On a recoverable failure we remain at `cell`; the next loop
                # rescans or selects another non-deferred edge.
                continue

            # Before backtracking, give unresolved UNKNOWN edges one bounded
            # stationary re-scan.  This catches a temporarily unhappy gimbal/ToF
            # instead of silently abandoning a physically open branch.
            unknown_here = [
                d for d in range(4)
                if self.edge_state.get((tuple(cell), d)) == "UNKNOWN"
            ]
            retry_count = self.unknown_rescan_counts.get(tuple(cell), 0)
            if unknown_here and retry_count < 1:
                self.unknown_rescan_counts[tuple(cell)] = retry_count + 1
                print(
                    f"[DFS UNKNOWN RECOVER] cell={cell} "
                    f"dirs={[DIR_NAMES[d] for d in unknown_here]} -> rescan before backtrack"
                )
                self.open_dirs.pop(tuple(cell), None)
                self.scan_cell(cell)
                if MAP_AUTOSAVE:
                    self.save_map(final=False)
                continue

            # No unvisited usable neighbor remains: classic DFS backtrack.
            parent = self.parent.get(cell)
            if parent is None:
                stack.pop()
                break

            print(f"\n[DFS] BACKTRACK {cell} -> {parent}")
            if not self.backtrack_to_parent(cell, parent):
                break
            stack.pop()
            if MAP_AUTOSAVE:
                self.save_map(final=False)

        self.safe_stop()

        if self.pose_trusted and not stack and self.current == self.root:
            unknown_count = sum(1 for s in self.edge_state.values() if s == "UNKNOWN")
            deferred_count = len(self.deferred_edges)
            self.map_complete = unknown_count == 0 and deferred_count == 0
            print("\n[DFS] FINISHED AT ROOT")
            print(f"[DFS] visited={len(self.visited)} unknown_edges={unknown_count} deferred_edges={deferred_count}")
            if not self.map_complete:
                print("[DFS] topology pass ended safely, but map is PARTIAL because some edges were unknown/deferred.")
        else:
            self.map_complete = False
            print("\n[DFS] stopped with a PARTIAL map.")

        self.save_map(final=self.map_complete)
        print(self.render_ascii_map())
        return self.map_complete


# ============================================================
# MAIN
# ============================================================
def main():
    explorer = DFSMapOnlyExplorer()
    try:
        if explorer.connect():
            explorer.run_dfs()
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        explorer.running = False
        explorer.safe_stop()
        explorer.save_map(final=False)
    except Exception as exc:
        # Final containment.  Mission code above intentionally returns status
        # instead of raising RuntimeError; any unexpected SDK/Python exception
        # is converted into a safe stop + partial-map save here.
        explorer.fault("UNEXPECTED", f"{type(exc).__name__}: {exc}", "SAFE STOP + save partial map")
        explorer.enter_safe_pause(f"unexpected exception contained: {type(exc).__name__}: {exc}")
    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
