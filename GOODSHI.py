#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboMaster EP - V20.7 9-MIN ATTEMPT / LOOSE-SIDE GOOD-ANGLE FIRE + FULL-35 + FAST RETURN
================================================
Purpose
-------
Explore an unknown grid maze with the hardened DFS/runtime-zero stack while
also scanning, verifying, aiming at and firing on colored shape targets. Target
vision is isolated from topology control: image/aim/fire faults are soft faults
and never invalidate the DFS pose. Chassis target-attack excursions remain
removed so a target service cannot corrupt the node anchor.

Runtime policy
--------------
* RUN pose is logical cell (0, 0); chassis FRONT at RUN is logical North.
* Gimbal ToF scans LEFT / FRONT / RIGHT at each newly visited cell.
* At a confirmed dead-end only, FRONT target service temporarily backs ~35 cm
  along the traversed corridor, scans/fires, then returns to the node anchor.
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

from robomaster import robot, blaster

import argparse
import json
import math
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None
    ttk = None


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

# Coordinate-frame geometry (measured on the real robot).
#
# IMPORTANT: the ToF is mounted on the rotating Gimbal.  Its optical origin is
# 8 cm forward of the robot/Gimbal yaw centre along the current Gimbal ray.
# Therefore topology/target POSITION estimates that are expressed from the
# robot centre must add this offset.  Collision/braking thresholds intentionally
# continue to use RAW ToF, because they protect the physical sensor/front end.
TOF_FORWARD_FROM_CENTER_M = 0.080

# TOF_OPEN_THRESHOLD_MM is defined in the ROBOT-CENTRE frame, not at the sensor.
# With an 8 cm forward ToF origin and near-horizontal scan, this corresponds to
# roughly 520 mm raw ToF.
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
SHARP_SIDE_ESCAPE_CLEAR_CM = 8.5
# V20.3: the old 0.055 m/s / 0.80 s escape could move only ~4.4 cm before
# timing out, then force a costly stationary L/F/R node probe.  A faster pure
# strafe clears the side wall first while the dedicated lateral yaw hold keeps
# the chassis cardinal.
SHARP_SIDE_ESCAPE_SPEED_MPS = 0.090
SHARP_SIDE_ESCAPE_MAX_SEC = 0.65
SHARP_SIDE_ESCAPE_FAST_RELEASE_CM = 7.5


# ============================================================
# MOTION / SAFETY
# ============================================================
# V20 10-MIN TURBO profile: preserves V19 full target-search geometry but removes
# avoidable waiting and roughly doubles the expensive mechanical motions.
# Unknown edges remain slower than already-proven OPEN edges for safety.
# because the robot is still discovering topology; already-proven OPEN edges
# can be traversed much faster during DFS backtrack / known-route travel.
DFS_EXPLORE_SPEED_MPS = 0.34
DFS_KNOWN_SPEED_MPS = 0.55
DFS_EXPLORE_APPROACH_MIN_MPS = 0.12
DFS_KNOWN_APPROACH_MIN_MPS = 0.16
DFS_EXPLORE_APPROACH_SLOW_M = 0.11
DFS_KNOWN_APPROACH_SLOW_M = 0.13

# Compatibility aliases.  Any legacy helper that still reads these constants
# gets the conservative exploration profile rather than the 0.35 m/s fast path.
FORWARD_SPEED_MPS = DFS_EXPLORE_SPEED_MPS
SLOW_FORWARD_SPEED_MPS = 0.12
CELL_APPROACH_SLOW_M = DFS_EXPLORE_APPROACH_SLOW_M
CELL_APPROACH_MIN_MPS = DFS_EXPLORE_APPROACH_MIN_MPS

# Front-ToF approach/brake policy.  Do not jump directly from forward motion to
# reverse when a wall appears.  Slow progressively, stop, then let the gimbal
# inspect LEFT/FRONT/RIGHT because the wall can be the end of a valid DFS cell.
FRONT_BRAKE_START_MM = 470.0
FRONT_CRAWL_START_MM = 240.0
FRONT_STOP_SCAN_MM = 165.0
# Faster known-edge travel starts braking sooner.  STOP_SCAN stays unchanged: it
# is a physical clearance/sensing limit and must not become looser with speed.
FRONT_BRAKE_START_FAST_MM = 520.0
FRONT_CRAWL_START_FAST_MM = 255.0
FRONT_SLOW_MM = FRONT_BRAKE_START_MM   # compatibility name used in logs/tuning
FRONT_HARD_STOP_MM = FRONT_STOP_SCAN_MM
FRONT_MIN_BRAKE_SPEED_MPS = 0.065
FRONT_NODE_SCAN_SETTLE_SEC = 0.10
FRONT_NODE_CAPTURE_MIN_PROGRESS_M = 0.34
FRONT_NODE_CAPTURE_MIN_VALID_RAYS = 2
# If the stop happens earlier than this, the edge is treated as a mid-edge
# obstruction unless a later retry proves otherwise.
FRONT_MIDEDGE_MAX_PROGRESS_M = 0.30
FRONT_TOF_STALE_SEC = 0.70

CONTROL_DT = 0.04
DRIVE_COMMAND_TIMEOUT = 0.25
POSITION_STALE_SEC = 0.60
ATTITUDE_STALE_SEC = 0.40
MAX_CELL_TIME_SEC = max(6.0, (CELL_LENGTH_M / DFS_EXPLORE_SPEED_MPS) * 2.8)

# If translation really must abort, return to the source smoothly.  Retreat is
# projection-based (forward + lateral), so a 5-8 cm Mecanum side drift no longer
# makes a pure-x reverse stall forever.
RETREAT_SPEED_MPS = 0.16
RETREAT_MIN_SPEED_MPS = 0.055
RETREAT_RAMP_SEC = 0.25
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

# Digital IR policy V9 -- intentionally SIMPLE.
#
# The two ACTIVE-LOW IR sensors are angled about +/-10 deg from FRONT and are
# used only as close corner/edge feelers:
#   LEFT LOW only  -> STOP forward motion -> strafe RIGHT until LEFT clears.
#   RIGHT LOW only -> STOP forward motion -> strafe LEFT  until RIGHT clears.
#   BOTH LOW       -> no directional guess.  Do not reverse/replan from IR;
#                     ToF + Sharp remain responsible for forward safety.
#
# IR never classifies topology, never requests a route scan, and never commands
# reverse motion.  A destination-side Sharp <= 6 cm is the only lateral veto.
IR_FILTER_SAMPLES = 3
IR_SENSOR_MOUNT_ANGLE_DEG = 10.0

# V20.3 merged IR+Sharp corner recovery.
# IR decides WHICH way to escape; Sharp helps decide HOW FAST and WHEN enough
# clearance has been recovered.  This avoids two back-to-back IR nudges followed
# by a second Sharp-only escape as seen in the field log.
IR_SIMPLE_STRAFE_SPEED_MPS = 0.085
IR_SIMPLE_STRAFE_FAST_MPS = 0.110
IR_SIMPLE_STRAFE_SLOW_MPS = 0.060
IR_SIMPLE_MAX_LATERAL_M = 0.140
IR_SIMPLE_TIMEOUT_SEC = 1.35
IR_SIMPLE_CLEAR_CONFIRM = 2
IR_SIMPLE_DEST_SHARP_STOP_CM = 6.0
IR_SIMPLE_DEST_SHARP_CAUTION_CM = 9.0
IR_SIMPLE_SOURCE_SHARP_CLEAR_CM = 9.0
IR_SIMPLE_SOURCE_SHARP_CLEAR_MIN_MOVE_M = 0.035
IR_SIMPLE_RETRIGGER_COOLDOWN_SEC = 0.35

# Lateral mecanum motion disturbs chassis yaw more than straight driving.
# Use a dedicated stronger P hold during IR / Sharp pure-strafe recovery without
# making the normal forward-drive PID overly aggressive.
LATERAL_YAW_HOLD_KP = 3.40
LATERAL_YAW_HOLD_MAX_DPS = 18.0
LATERAL_YAW_HOLD_MIN_DPS = 1.8
LATERAL_YAW_HOLD_DEADBAND_DEG = 0.12
LATERAL_RECOVERY_REALIGN_TRIGGER_DEG = 0.90


# Close-Sharp recovery also follows STOP/PROBE instead of full retreat.
SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC = 0.45
SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M = 0.20

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
STATIONARY_YAW_MAX_DPS = 42.0
STATIONARY_YAW_MIN_DPS = 4.0
# Near the target, use a gentler minimum turn rate so tiny residual errors do not
# overshoot back and forth.  Keep the normal 2.8 dps floor outside this fine zone.
STATIONARY_YAW_FINE_ZONE_DEG = 0.60
STATIONARY_YAW_FINE_MIN_DPS = 1.8
STATIONARY_YAW_DEADBAND_DEG = 0.18
STATIONARY_YAW_I_LIMIT = 5.0
STATIONARY_YAW_I_ZONE_DEG = 5.0
STATIONARY_YAW_D_ALPHA = 0.72
STATIONARY_HOLD_HZ = 30.0
STATIONARY_SETTLE_SEC = 0.12
STATIONARY_SETTLE_TOL_DEG = 0.85
STATIONARY_ALIGN_TIMEOUT_SEC = 0.85
PRE_MOVE_ALIGN_TOL_DEG = 1.10
PRE_MOVE_ALIGN_TIMEOUT_SEC = 0.75
POST_MOVE_ALIGN_TIMEOUT_SEC = 0.65
# Precision target used only after a normal full-cell arrival.  Pre-move/retreat
# recovery tolerances remain unchanged so resilience is not sacrificed.
POST_MOVE_ALIGN_TOL_DEG = 0.45

# Turn PID.  Integral is deliberately weak and only active in the final approach;
# most of the turn is P+D so inertia does not wind the controller up.
TURN_KP = 1.55
TURN_KI = 0.035
TURN_KD = 0.22
TURN_MAX_DPS = 90.0
TURN_FINE_MAX_DPS = 38.0
TURN_FINE_ZONE_DEG = 14.0
TURN_MIN_DPS = 8.0
TURN_FINE_MIN_DPS = 4.0
TURN_TOLERANCE_DEG = 0.90
TURN_SETTLE_SEC = 0.12
TURN_CONTROL_HZ = 40.0
TURN_I_LIMIT = 7.0
TURN_I_ZONE_DEG = 12.0
TURN_D_ALPHA = 0.72
TURN_TIMEOUT_90_SEC = 3.4
TURN_TIMEOUT_180_SEC = 5.0
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
GIMBAL_YAW_SPEED = 220
GIMBAL_PITCH_SPEED = 140
GIMBAL_ACTION_TIMEOUT_SEC = 1.10
# Yaw accuracy is topology-critical; pitch accuracy is not.  The real gimbal
# in field logs repeatedly settled around -8.2 deg when commanded to -5 deg,
# while yaw was already within ~0.1 deg.  Rejecting those poses turned valid
# routes into UNKNOWN and could stop DFS.  Keep yaw strict, but accept a safe
# downward pitch envelope for ToF mapping/collision sensing.
GIMBAL_ANGLE_TOL_DEG = 3.0              # legacy alias / strict yaw tolerance
GIMBAL_YAW_TOL_DEG = 3.0
GIMBAL_PITCH_EXACT_TOL_DEG = 3.0
# Topology scan can tolerate a wider pitch envelope than motion.  Real logs
# show the EP occasionally coupling pitch with large +/-90deg yaw and settling
# around -15deg / +5deg even though yaw itself is correct.  Do not turn a good
# cardinal ray into UNKNOWN solely because of that pitch offset.
GIMBAL_TOF_SAFE_PITCH_MIN_DEG = -16.5
GIMBAL_TOF_SAFE_PITCH_MAX_DEG = +6.0
# Motion remains stricter: before translation the front ToF is actively trimmed
# back toward -5deg and must lie in this narrower envelope.
GIMBAL_MOTION_SAFE_PITCH_MIN_DEG = -12.0
GIMBAL_MOTION_SAFE_PITCH_MAX_DEG = +2.0
GIMBAL_MOTION_FRONT_YAW_TOL_DEG = 4.5

# Velocity-mode recovery is the fallback when SDK moveto()/recenter acknowledges
# but the feedback remains far from the requested pose.
GIMBAL_RECOVERY_YAW_MAX_DPS = 95.0
GIMBAL_RECOVERY_YAW_MIN_DPS = 12.0
GIMBAL_RECOVERY_PITCH_MAX_DPS = 32.0
GIMBAL_RECOVERY_PITCH_MIN_DPS = 4.0
GIMBAL_RECOVERY_YAW_TOL_DEG = 3.0
GIMBAL_RECOVERY_PITCH_TOL_DEG = 2.0
GIMBAL_RECOVERY_SETTLE_SAMPLES = 2
GIMBAL_SETTLE_SEC = 0.035
GIMBAL_SCAN_RETRIES = 0
UNKNOWN_EDGE_RESCAN_PASSES = 1
UNKNOWN_EDGE_RESCAN_SETTLE_SEC = 0.04
GIMBAL_SOFT_YAW_MIN_DEG = -135.0
GIMBAL_SOFT_YAW_MAX_DEG = +135.0
GIMBAL_SOFT_PITCH_MIN_DEG = -15.0
GIMBAL_SOFT_PITCH_MAX_DEG = +20.0

TOF_SCAN_SAMPLES = 3
TOF_SCAN_INTERVAL_SEC = 0.025
TOF_SCAN_TIMEOUT_SEC = 0.45
TOF_VALID_MIN_MM = 20.0
TOF_VALID_MAX_MM = 10000.0

STARTUP_CONNECT_RETRIES = 3
STARTUP_TELEMETRY_WAIT_SEC = 5.0
STARTUP_GIMBAL_WAIT_SEC = 2.5
# Startup recenter is special: scan retries stay at 0 for competition speed,
# but initialization must not abort merely because the SDK action reports False.
STARTUP_GIMBAL_RECENTER_ATTEMPTS = 2
STARTUP_GIMBAL_RECENTER_VERIFY_SEC = 0.65
STARTUP_GIMBAL_FALLBACK_TIMEOUT_SEC = 2.20
STARTUP_GIMBAL_ACCEPT_YAW_DEG = 5.0
STARTUP_GIMBAL_ACCEPT_PITCH_MIN_DEG = -12.0
STARTUP_GIMBAL_ACCEPT_PITCH_MAX_DEG = +5.0
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
MAX_MISSION_SEC = 10 * 60


# ============================================================
# TARGET VISION / FOAM-BOARD GATE / AIM + FIRE
# ============================================================
# The detector combines the useful parts of the provided target/radar version
# with the friend's aggressive-field-test image policy:
#   Lab-L CLAHE -> broad HSV candidate ranges -> shape/color confidence ->
#   temporal verification -> stopped AIM lock -> fresh ToF range gate -> fire.
# Vision faults are NON-FATAL to DFS.  Chassis target-attack excursions are not
# used in this hardened mapper because a failed return would corrupt the map.
TARGET_VISION_ENABLED = True
TARGET_PREVIEW_ENABLED = True
TARGET_WINDOW_NAME = "RoboMaster Target Vision - Hardened"
TARGET_CAMERA_RESOLUTION = "360p"
TARGET_LATEST_JSON = MAP_DIR / "latest_targets.json"

# Search ROI.  During target lock this shrinks around the selected candidate.
TARGET_SEARCH_ROI = (0.12, 0.14, 0.88, 0.92)
TARGET_DYNAMIC_ROI_MIN_W = 0.34
TARGET_DYNAMIC_ROI_MIN_H = 0.44
TARGET_DYNAMIC_ROI_MAX_W = 0.58
TARGET_DYNAMIC_ROI_MAX_H = 0.72
TARGET_DYNAMIC_ROI_BBOX_SCALE = 3.0

# Continuous three-sector radar search.  Keep the side sectors narrower than
# the older +/-135 deg sweep so each pass spends more time near useful bearings:
#   LEFT  : -110 -> -45, after a temporary chassis slide RIGHT
#   FRONT :  -45 -> +45, from the logical node anchor
#   RIGHT :  +45 -> +110, after a temporary chassis slide LEFT
# Search pitch stays at -10 deg.  Side viewpoint shifts are temporary only; the
# robot returns to the same odometric node anchor before the next sector / DFS.
TARGET_SEARCH_YAW_LIMIT_DEG = 110.0
TARGET_SEARCH_PITCH_DEG = -10.0
TARGET_SEARCH_QUICK_GATE_FRAMES = 2
TARGET_SEARCH_MIN_FRESH_FRAMES = 2
TARGET_SCAN_SETTLE_SEC = 0.015
TARGET_SCAN_COOLDOWN_SEC = 0.18

TARGET_SWEEP_SPEED_DPS = 125.0
TARGET_SWEEP_MIN_DPS = 45.0
TARGET_SWEEP_FINE_ZONE_DEG = 4.0
TARGET_SWEEP_CONTROL_DT = 0.025
TARGET_SWEEP_TIMEOUT_SEC = 2.15
# V20.1: target scanning does not need topology-grade exact endpoint poses.
# A sector may start a few degrees early/late and still sweep through the useful cone.
TARGET_SEARCH_POSE_YAW_TOL_DEG = 4.0
TARGET_SEARCH_POSE_SOFT_YAW_TOL_DEG = 12.0
TARGET_SEARCH_POSE_PITCH_TOL_DEG = 5.0
TARGET_GIMBAL_STALL_WINDOW_SEC = 0.28
TARGET_GIMBAL_STALL_MIN_PROGRESS_DEG = 0.8
TARGET_SWEEP_MAX_INTERRUPTS_PER_SECTOR = 6
TARGET_SWEEP_RESUME_ADVANCE_DEG = 8.0
# Successful locks suppress the same color/shape around the original SWEEP
# bearing, not the post-Aim bearing.  Failed Aim gets a smaller cone so a
# genuinely separate nearby target can still be acquired.
TARGET_SWEEP_REPEAT_SUPPRESS_DEG = 30.0
TARGET_SWEEP_FAIL_SUPPRESS_DEG = 14.0

# V15 SOFT ACQUISITION.  The original +/-10deg cones remain the PREFERRED
# detection bearings, but a target may now earn SHOOT-INTENT for an additional
# +/-10deg outside them.  This catches near-cardinal detections such as +75.4deg
# or +100.7deg without weakening the later crosshair / aim-envelope / ToF gates.
TARGET_PREFERRED_ACQUISITION_CONES = {
    "LEFT":  (-100.0, -80.0),
    "FRONT": ( -10.0, +10.0),
    "RIGHT": ( +80.0, +100.0),
}

# V20.7 GOOD-ANGLE FIRE GATE.
# Detection / tracking stays wide so we never throw away a visible target, but
# WATER fire is authorized only when the FINAL crosshair lock is close to the
# sector's cardinal shooting direction.  This prevents a -50/-60deg side shot
# from being counted as fired before the robot later gets a much better -90deg
# view.  Keep these separate from acquisition cones so search coverage remains
# unchanged.
TARGET_GOOD_FIRE_CONES = {
    # Side shots are intentionally looser than the previous strict +/-10deg gate.
    # Field behavior was acceptable around the old side acquisition cone, while
    # the dangerous misses happened mostly near the FRONT seam (~45-60deg).
    "LEFT":  (-110.0, -70.0),
    "FRONT": ( -15.0, +15.0),
    "RIGHT": ( +70.0, +110.0),
}
# A bad-angle observation is only suppressed very locally.  The same physical
# target may be reacquired a few degrees later in the SAME sweep if the geometry
# improves; it is NOT treated like a successful shot.
TARGET_BAD_ANGLE_RETRY_SUPPRESS_DEG = 4.0

TARGET_SOFT_ACQUISITION_EXTRA_DEG = 10.0

# name, sweep_start, sweep_end, temporary chassis slide, SOFT acquisition low/high.
# V20.3: acquisition now covers the physical sweep instead of leaving wide
# memory-only gaps.  The old FRONT -20..+20 rule caused a real target at -38.3deg
# to be remembered, then AIM hit the +/-30deg envelope and timed out.
TARGET_RADAR_SECTORS = (
    ("LEFT",  -110.0, -45.0, "RIGHT", -110.0, -45.0),
    ("FRONT",  -45.0, +45.0, None,     -45.0, +45.0),
    ("RIGHT",  +45.0, +110.0, "LEFT", +45.0, +110.0),
)

# Two-stage bearing policy:
#   1) preferred cone = original +/-10deg around -90/0/+90.
#      soft acquisition cone = preferred cone expanded by +/-10deg.
#   2) once shoot-intent is established, the crosshair servo may rotate up to
#      +/-30deg from that sector's cardinal axis to put the target BBOX CENTRE
#      on the camera crosshair. Final fire still requires aim envelope + ToF gate.
TARGET_FIRE_ACQUISITION_HALF_CONE_DEG = 20.0
TARGET_CROSSHAIR_AIM_MAX_FROM_AXIS_DEG = 30.0
# Sector-specific V20.3 envelopes.  Front can use almost the full +/-45deg sweep;
# side sectors may center toward the adjacent FRONT boundary instead of timing out
# a few degrees before a perfectly visible target.
# V20.6 overlap the crosshair servo envelopes across sector seams.  Detection
# still starts inside each sector, but a verified target near +/-45 may need a few
# extra degrees of servo travel to place its bbox centre on the crosshair.
TARGET_FRONT_CROSSHAIR_AIM_HALF_DEG = 55.0
TARGET_SIDE_CROSSHAIR_AIM_INNER_DEG = 35.0
TARGET_SIDE_CROSSHAIR_AIM_OUTER_DEG = 120.0

# Temporary SIDE viewpoint shift -- V20.7 FULL-35 ODOMETRY POLICY.
#
# IMPORTANT: the 35 cm requirement is CAMERA VIEWPOINT displacement, not distance
# from a wall.  LEFT target search always attempts to move the chassis RIGHT by
# ~35 cm; RIGHT target search mirrors it LEFT.  This creates enough camera distance
# for close side targets to fit the detector/shape gate even when there is no wall.
#
# Odometry owns normal stopping/speed.  Destination Sharp is read fresh every
# control cycle and has exactly one job: emergency collision STOP.
TARGET_SIDE_SHIFT_ENABLED = True
TARGET_SIDE_SHIFT_DISTANCE_M = 0.350
TARGET_SIDE_SHIFT_STOP_TOL_M = 0.008       # >=342 mm counts as full viewpoint
TARGET_SIDE_SHIFT_MAX_M = 0.375            # runaway/overshoot guard only
TARGET_SIDE_SHIFT_SPEED_MPS = 0.220        # remaining >65 mm
TARGET_SIDE_SHIFT_MED_SPEED_MPS = 0.145    # remaining <=65 mm
TARGET_SIDE_SHIFT_SLOW_MPS = 0.080         # remaining <=32 mm
TARGET_SIDE_SHIFT_CRAWL_MPS = 0.040        # remaining <=14 mm
TARGET_SIDE_SHIFT_MED_REMAIN_M = 0.065
TARGET_SIDE_SHIFT_SLOW_REMAIN_M = 0.032
TARGET_SIDE_SHIFT_CRAWL_REMAIN_M = 0.014
TARGET_SIDE_SHIFT_CONTROL_DT = 0.020       # ~50 Hz Sharp guard
TARGET_SIDE_SHIFT_TIMEOUT_SEC = 3.2

# Sharp: direct STOP only.  Field log showed 6.8 cm could arrive one callback too
# late at 0.22 m/s, so use 9 cm as the software brake line; no Sharp speed tiers.
TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM = 9.0
TARGET_SIDE_SHIFT_HARD_STOP_CONFIRM_CYCLES = 1
TARGET_SIDE_SHIFT_FAST_SHARP_SAMPLES = 1
TARGET_SIDE_SHIFT_FAST_SHARP_INTERVAL_SEC = 0.0
TARGET_SIDE_SHIFT_SHARP_MISSING_MAX_CYCLES = 2

# Return runs over the path just proven moments earlier, so it can be much faster.
TARGET_SIDE_SHIFT_RETURN_SPEED_MPS = 0.38
TARGET_SIDE_SHIFT_RETURN_MED_SPEED_MPS = 0.22
TARGET_SIDE_SHIFT_RETURN_SLOW_MPS = 0.10
TARGET_SIDE_SHIFT_RETURN_MED_ZONE_M = 0.080
TARGET_SIDE_SHIFT_RETURN_SLOW_ZONE_M = 0.035
TARGET_SIDE_SHIFT_RETURN_TIMEOUT_SEC = 2.2
TARGET_SIDE_SHIFT_RETURN_LAT_TOL_M = 0.018
TARGET_SIDE_SHIFT_RETURN_FWD_TOL_M = 0.030
TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M = 0.040

# Adaptive short AIM deadline: normal lock gets 2.4 s; only a genuinely near-center
# target receives one small grace extension.
TARGET_AIM_NEAR_CENTER_GRACE_SEC = 0.45
TARGET_AIM_NEAR_CENTER_GRACE_ERR = 0.055

# Fast fired-memory suppression is shape-independent because the same physical
# square can look rectangular from an oblique side viewpoint.
TARGET_FAST_FIRED_GRID_DIST = 0.70
TARGET_FAST_FIRED_SAME_CELL_BEARING_DEG = 20.0
TARGET_CHASSIS_REALIGN_TRIGGER_DEG = 2.0
TARGET_CHASSIS_REALIGN_TIMEOUT_SEC = 0.35

# Temporary FRONT viewpoint shift at a CONFIRMED dead-end only.
# A dead-end must have LEFT + FRONT + RIGHT all explicitly classified WALL by
# the stationary topology scan.  Before the FRONT target sector, back the chassis
# along the already-traversed corridor by ~35 cm, scan/aim/fire from there, then
# return to the exact logical node anchor before continuing to RIGHT / DFS.
# This motion never changes the logical cell or the map.
TARGET_DEADEND_FRONT_BACKSHIFT_ENABLED = True
TARGET_DEADEND_FRONT_BACKSHIFT_M = 0.350
TARGET_DEADEND_FRONT_BACKSHIFT_SPEED_MPS = 0.26
TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_MPS = 0.12
TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_ZONE_M = 0.060
TARGET_DEADEND_FRONT_BACKSHIFT_TIMEOUT_SEC = 3.6
TARGET_DEADEND_FRONT_RETURN_SPEED_MPS = 0.32
TARGET_DEADEND_FRONT_RETURN_SLOW_MPS = 0.14
TARGET_DEADEND_FRONT_RETURN_TIMEOUT_SEC = 3.6
TARGET_DEADEND_FRONT_RETURN_FWD_TOL_M = 0.025
TARGET_DEADEND_FRONT_RETURN_LAT_TOL_M = 0.030
TARGET_DEADEND_FRONT_RETURN_SOFT_TOL_M = 0.050
TARGET_DEADEND_FRONT_LATERAL_KP = 1.20
TARGET_DEADEND_FRONT_LATERAL_MAX_MPS = 0.070

# Firing range policy. <=1200 mm remains the absolute competition hard limit,
# but <=350 mm is now the preferred high-confidence firing zone.  The robot does
# not launch a new forward attack excursion only to reach 350 mm; if the locked
# target is already between 350 and 1200 mm it may still fire as a legal fallback.
TARGET_FRONT_SWEET_SPOT_MM = 350.0

# Lighting normalization + broad field-tested HSV candidate ranges from the
# linked aggressive-field-test branch.  Broad masks are made safe by confidence,
# geometry and temporal verification rather than by over-tight HSV thresholds.
TARGET_CLAHE_CLIP_LIMIT = 2.0
TARGET_CLAHE_GRID = 8
TARGET_HSV_RANGES = {
    "RED": [
        ((0, 75, 45), (12, 255, 255)),
        ((168, 75, 45), (179, 255, 255)),
    ],
    "YELLOW": [((18, 65, 55), (39, 255, 255))],
    "GREEN": [((35, 55, 35), (90, 255, 255))],
    "BLUE": [((88, 65, 35), (138, 255, 255))],
}
TARGET_HUE_CENTERS = {
    "RED": 0.0,
    "GREEN": 62.0,
    "BLUE": 112.0,
    "YELLOW": 29.0,
}
TARGET_MORPH_OPEN_KERNEL = 3
TARGET_MORPH_CLOSE_KERNEL = 5
TARGET_MIN_AREA_FRAC_ROI = 0.0010
TARGET_MAX_AREA_FRAC_ROI = 0.35
TARGET_MIN_AREA_PX = 260.0
TARGET_BORDER_MARGIN_PX = 5

# Shape gates.  Keep separate horizontal/vertical rectangles because the field
# assignment can contain both.
TARGET_MIN_SOLIDITY = 0.84
TARGET_RECT_MIN_FILL = 0.62
TARGET_RECT_MAX_CORNER_COS = 0.46
TARGET_CIRCLE_MIN_CIRCULARITY = 0.69
TARGET_CIRCLE_ASPECT_MIN = 0.68
TARGET_CIRCLE_ASPECT_MAX = 1.46
TARGET_POLY_EPS_FRAC = 0.030
TARGET_SQUARE_ASPECT_MIN = 0.80
TARGET_SQUARE_ASPECT_MAX = 1.24
TARGET_RECT_ASPECT_MIN = 1.26

# Candidate + temporal gates.  A single colored reflection can never fire.
TARGET_MIN_CONFIDENCE = 0.50
TARGET_SAVE_CONFIDENCE = 0.60
TARGET_VERIFY_FRAMES = 2
TARGET_VERIFY_WINDOW_SEC = 1.15
TARGET_VERIFY_MAX_CENTER_STD = 0.045
TARGET_VERIFY_MAX_AREA_CV = 0.38
TARGET_TRACK_MAX_JUMP_NORM = 0.24
TARGET_HISTORY_SEC = 1.25

# AIM lock.  The friend's branch uses a 1.5% reticle tolerance; use the same
# order of precision while retaining a little extra vertical tolerance for the
# current gimbal/camera mount.
TARGET_AIM_TIMEOUT_SEC = 2.40
TARGET_AIM_POLL_SEC = 0.020
TARGET_AIM_LOST_GRACE_SEC = 0.35
# V15 center-only fire lock: tighter vertical tolerance, smaller servo deadband,
# and 3 consecutive centred frames before the ToF/fire pipeline.  center_norm is
# the bounding-box centre produced by _detect_targets().
TARGET_AIM_CENTER_TOL_X = 0.020
TARGET_AIM_CENTER_TOL_Y = 0.020
TARGET_AIM_CENTER_HOLD_FRAMES = 2
TARGET_AIM_YAW_GAIN_DPS = 150.0
TARGET_AIM_PITCH_GAIN_DPS = 110.0
TARGET_AIM_YAW_MAX_DPS = 52.0
TARGET_AIM_PITCH_MAX_DPS = 34.0
TARGET_AIM_DEADBAND_X = 0.006
TARGET_AIM_DEADBAND_Y = 0.008
TARGET_AIM_YAW_LIMIT_DEG = 132.0
# Soft Aim bound is deliberately above the requested -14deg hard floor because
# the physical gimbal can coast another ~1deg after drive_speed() is zeroed.
# Search/sweep still stays around -10 deg.  These legacy bounds are used by
# strict pose helpers when returning to a search pose; do NOT make topology
# scans follow the deeper camera-centering pitch.
TARGET_AIM_PITCH_MIN_DEG = -13.0
TARGET_AIM_PITCH_HARD_MIN_DEG = -14.0
TARGET_AIM_PITCH_MAX_DEG = -3.0
TARGET_AIM_PITCH_LIMIT_BRAKE_ZONE_DEG = 1.5

# V7 FLEX CENTER PITCH
# Once a target has earned SHOOT-INTENT from the +/-10deg acquisition cone,
# allow the camera/gimbal to pitch substantially farther down ONLY while
# centering the crosshair.  Field log showed RED stuck at pitch -14.6/-15.0
# with vertical image error +0.086..+0.092, so the old -14deg floor prevented
# the reticle from ever reaching the target center.
TARGET_CENTER_PITCH_SOFT_MIN_DEG = -20.5
TARGET_CENTER_PITCH_HARD_MIN_DEG = -22.5
TARGET_CENTER_PITCH_MAX_DEG = +4.0
TARGET_CENTER_PITCH_BRAKE_ZONE_DEG = 2.0
TARGET_CENTER_PITCH_MAX_DPS = 38.0
TARGET_CENTER_PITCH_GAIN_DPS = 125.0
TARGET_AIM_OFFSET_X = 0.0
TARGET_AIM_OFFSET_Y = 0.0

# Foam-board spatial gate.  Estimate the visible top edge of the white/grey
# board and reject targets whose centre is above it or whose box is mostly cut
# off above the board.  The gate fails open when the board cannot be estimated.
TARGET_FOAM_GATE_ENABLED = True
TARGET_FOAM_FAIL_CLOSED = False
TARGET_FOAM_HSV_LOW = (0, 0, 115)
TARGET_FOAM_HSV_HIGH = (180, 48, 255)
TARGET_FOAM_OPEN_KERNEL = 3
TARGET_FOAM_CLOSE_KERNEL = 9
TARGET_FOAM_MIN_COMPONENT_AREA_PX = 1800
TARGET_FOAM_MIN_COMPONENT_WIDTH_PX = 35
TARGET_FOAM_MIN_COMPONENT_HEIGHT_PX = 35
TARGET_FOAM_COMPONENT_MIN_BOTTOM_FRAC = 0.42
TARGET_FOAM_PROFILE_MAX_INTERP_GAP_PX = 64
TARGET_FOAM_PROFILE_MEDIAN_RADIUS_PX = 7
TARGET_FOAM_PROFILE_SAMPLE_RADIUS_PX = 12
TARGET_FOAM_PROFILE_TTL_SEC = 0.55
TARGET_FOAM_MIN_ROI_COVERAGE = 0.12
TARGET_FOAM_CENTER_MARGIN_PX = 5
TARGET_FOAM_MIN_BBOX_BELOW_FRAC = 0.55

# Fire policy.  Competition mode auto-fires every verified color/shape target,
# once per deduplicated target, only after a fresh stopped ToF range check.
TARGET_REAL_FIRE_ENABLED = True
TARGET_AUTO_FIRE_ENABLED = True
TARGET_FIRE_ONCE_PER_TARGET = True
TARGET_INFRARED_SHOTS = 1
# No application-level minimum range: a very close target may fire if the ToF
# still produces a valid fresh reading.  <=350 mm is the preferred / most certain
# zone; 350-1200 mm remains legal fallback so a good locked target is not wasted.
TARGET_FIRE_PREFERRED_RANGE_MM = 350.0
TARGET_FIRE_MAX_RANGE_MM = 1200.0
TARGET_FIRE_RANGE_SAMPLES = 3
TARGET_FIRE_RANGE_RECHECK_SEC = 0.03
TARGET_FIRE_SETTLE_SEC = 0.035
TARGET_DEDUPE_GRID_DIST = 0.65
TARGET_DEDUPE_BEARING_DEG = 20.0

# V10 selectable physical firing mode. GUI can change this live.
TARGET_FIRE_MODE_DEFAULT = "INFRARED"
TARGET_WATER_SHOTS = 1

# Camera / blaster / ToF geometry.
# Vertical: positive means ABOVE the blaster muzzle centre.
#   camera centre : +4.0 cm from muzzle
#   ToF centre    : +7.5 cm from muzzle
# Horizontal/forward from robot/Gimbal yaw centre (measured by user):
#   ToF optical origin : +8.0 cm
#   muzzle             : +15.0 cm
# Therefore the muzzle is 7 cm CLOSER to a forward target than the ToF origin.
# Example at pitch~=0: raw ToF=600 mm -> centre~=680 mm -> muzzle~=530 mm.
FIRE_CAMERA_ABOVE_MUZZLE_M = +0.040
FIRE_TOF_ABOVE_MUZZLE_M = +0.075
FIRE_TOF_ABOVE_CAMERA_M = FIRE_TOF_ABOVE_MUZZLE_M - FIRE_CAMERA_ABOVE_MUZZLE_M
FIRE_TOF_FORWARD_FROM_CENTER_M = TOF_FORWARD_FROM_CENTER_M
FIRE_MUZZLE_FORWARD_FROM_CENTER_M = +0.150
# Camera is on the same Gimbal head; until a separate camera-X measurement is
# supplied, use the ToF longitudinal plane.  Change this one constant later if
# the camera optical centre is measured at a different forward offset.
FIRE_CAMERA_FORWARD_FROM_CENTER_M = FIRE_TOF_FORWARD_FROM_CENTER_M
FIRE_MUZZLE_AHEAD_OF_TOF_M = (
    FIRE_MUZZLE_FORWARD_FROM_CENTER_M - FIRE_TOF_FORWARD_FROM_CENTER_M
)

# After the CAMERA crosshair is genuinely centered, use fresh ToF plus the full
# forward/vertical geometry to solve the LOS from the physical muzzle.
FIRE_AIM_GEOMETRY_ENABLED = True
FIRE_PARALLAX_PITCH_SIGN = +1.0
FIRE_COMP_MAX_ABS_DEG = 18.0
FIRE_COMP_PITCH_MIN_DEG = -22.5
FIRE_COMP_PITCH_MAX_DEG = +12.0
FIRE_COMP_TIMEOUT_SEC = 1.00
FIRE_COMP_PITCH_TOL_DEG = 0.75
FIRE_COMP_YAW_TOL_DEG = 1.50
FIRE_COMP_SETTLE_SAMPLES = 2
FIRE_COMP_SETTLE_SEC = 0.035

# Water trajectory needs empirical calibration from the real blaster. Keep the
# ballistic term zero until shot-vs-range measurements are collected; parallax
# compensation still works for both WATER and INFRARED now.
WATER_EXTRA_PITCH_DEG = 0.0

# V11 two-stage visual fire lock, adapted from the proven CENTER -> UPPER
# strategy in robo2(2).py.  First the target CENTER is genuinely locked by
# _aim_and_lock(); only then do we keep the SAME target and move the optical
# crosshair slightly upward on its face.  0.32 means 32% down from the target
# bounding-box top: deliberately less aggressive than the older 20% setting,
# so it biases the hit upward without hugging the top edge.
TARGET_UPPER_AIM_ENABLED = False
TARGET_UPPER_HIT_Y_RATIO = 0.32
TARGET_UPPER_AIM_TIMEOUT_SEC = 3.0
TARGET_UPPER_AIM_LOST_GRACE_SEC = 0.70
TARGET_UPPER_AIM_TOL_X = 0.028
TARGET_UPPER_AIM_TOL_Y = 0.030
TARGET_UPPER_AIM_HOLD_FRAMES = 3

# Selectable 1..6-shot burst. Each shot is an individual SDK fire command so
# cadence stays explicit and predictable. The GUI uses a readonly drop-down and
# can change the count live before the next locked target fires.
TARGET_FIRE_BURST_DEFAULT = 1
TARGET_FIRE_BURST_OPTIONS = (1, 2, 3, 4, 5, 6)
TARGET_FIRE_BURST_INTERVAL_SEC = 0.30

# Mission-control GUI. --no-gui keeps the old immediate headless workflow.
CONTROL_GUI_ENABLED = True
CONTROL_GUI_REFRESH_MS = 200
CONTROL_GUI_CANVAS_W = 620
CONTROL_GUI_CANVAS_H = 620

# ============================================================
# ROUND-2 ATTACK MEMORY / TARGET FILTER
# ============================================================
# 4 colors x 4 shapes = 16 independently selectable target classes in the GUI.
TARGET_FILTER_COLORS = ("RED", "YELLOW", "GREEN", "BLUE")
TARGET_FILTER_SHAPES = ("SQUARE", "CIRCLE", "RECT_HORIZONTAL", "RECT_VERTICAL")
TARGET_FILTER_ALL = tuple(
    (color, shape)
    for color in TARGET_FILTER_COLORS
    for shape in TARGET_FILTER_SHAPES
)

# Round 1 writes a stable snapshot here.  Round 2 NEVER depends on
# latest_targets.json because that file is allowed to change during round 2.
ROUND1_ATTACK_MEMORY_JSON = MAP_DIR / "round1_attack_memory.json"
ROUND2_RESULT_JSON = MAP_DIR / "round2_attack_result.json"

# Competition deadline.  Keep a few seconds of operational margin instead of
# starting another scan at 299.9 s and overrunning the 5 minute window.
ROUND2_HARD_LIMIT_SEC = 295.0
ROUND2_NARROW_SWEEP_HALF_DEG = 14.0
ROUND2_FALLBACK_FULL_SECTOR = True
ROUND2_MAX_HINT_ATTEMPTS = 2
ROUND2_EXACT_ORDER_MAX_ANCHORS = 11


class TargetVisionSubsystem:
    """Non-fatal camera target service attached to DFSMapOnlyExplorer."""

    def __init__(self, owner):
        self.owner = owner
        self.running = False
        self.available = False
        self.stream_started = False
        self.preview_enabled = bool(TARGET_PREVIEW_ENABLED)
        self.thread = None
        self.lock = threading.Lock()
        self.roi_lock = threading.Lock()
        self.dynamic_roi = None
        self.latest_frame = None
        self.latest_frame_t = 0.0
        self.latest_detections = []
        self.latest_color_blobs = []
        self.history = deque(maxlen=160)
        self.frame_seq = 0
        self.foam_profile = None
        self.foam_profile_t = 0.0
        self.foam_coverage = 0.0
        self.foam_components = []
        self.scanned_cells = set()
        self.last_scan_t = {}
        self.targets = []
        self.target_seq = 0
        self.fired_target_ids = set()
        self.status = "VISION OFF"
        self.last_fire_event = "NONE"
        self.last_aim_solution = {}
        self._clahe = None
        if cv2 is not None:
            try:
                self._clahe = cv2.createCLAHE(
                    clipLimit=float(TARGET_CLAHE_CLIP_LIMIT),
                    tileGridSize=(int(TARGET_CLAHE_GRID), int(TARGET_CLAHE_GRID)),
                )
            except Exception:
                self._clahe = None

    # ---------------- camera / thread ----------------
    def start(self):
        if not TARGET_VISION_ENABLED:
            self.status = "VISION DISABLED"
            return False
        if cv2 is None or np is None:
            self.owner.fault("VISION", "opencv-python or numpy unavailable", "continue DFS without targets")
            return False
        if self.owner.camera is None:
            self.owner.fault("VISION", "RoboMaster camera unavailable", "continue DFS without targets")
            return False
        try:
            try:
                from robomaster import camera as rm_camera
                resolution = getattr(rm_camera, "STREAM_360P", TARGET_CAMERA_RESOLUTION)
            except Exception:
                resolution = TARGET_CAMERA_RESOLUTION
            self.owner.camera.start_video_stream(display=False, resolution=resolution)
            self.stream_started = True
        except Exception as exc:
            self.owner.fault("VISION STREAM", f"{type(exc).__name__}: {exc}", "continue DFS without targets")
            return False

        self.running = True
        self.available = True
        self.status = "VISION READY"
        self.thread = threading.Thread(target=self._vision_loop, name="TargetVision", daemon=True)
        self.thread.start()
        print("[VISION] Lab-CLAHE + HSV/LAB target detector ready")
        print(
            f"[VISION] Foam-board top gate={'ON' if TARGET_FOAM_GATE_ENABLED else 'OFF'}; "
            f"real fire={'ON' if TARGET_REAL_FIRE_ENABLED else 'DRY'}; range<={TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
        )
        return True

    def stop(self):
        self.running = False
        self.available = False
        self.stream_started = False
        self._stop_gimbal_velocity()
        if self.owner.camera is not None:
            try:
                self.owner.camera.stop_video_stream()
            except Exception:
                pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            if cv2 is not None:
                cv2.destroyWindow(TARGET_WINDOW_NAME)
        except Exception:
            pass

    def _vision_loop(self):
        print("[VISION] frame thread started")
        while self.owner.running and self.running and self.stream_started:
            try:
                frame = self.owner.camera.read_cv2_image(strategy="newest", timeout=0.6)
            except Exception as exc:
                if self.running:
                    self.owner.fault("VISION FRAME", f"{type(exc).__name__}: {exc}", "retry frame")
                time.sleep(0.08)
                continue
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            try:
                foam_profile, coverage, components = self._compute_foam_wall_profile(frame)
                detections, blobs = self._detect_targets(frame, foam_profile, coverage)
            except Exception as exc:
                self.owner.fault("VISION DETECT", f"{type(exc).__name__}: {exc}", "drop frame / keep DFS alive")
                time.sleep(0.02)
                continue

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_frame_t = now
                self.latest_detections = [dict(d) for d in detections]
                self.latest_color_blobs = [dict(d) for d in blobs]
                self.foam_profile = None if foam_profile is None else foam_profile.copy()
                self.foam_profile_t = now
                self.foam_coverage = float(coverage)
                self.foam_components = [dict(c) for c in components]
                self.frame_seq += 1
                self.history.append((now, self.frame_seq, [dict(d) for d in detections]))
                cutoff = now - TARGET_HISTORY_SEC
                while self.history and self.history[0][0] < cutoff:
                    self.history.popleft()

            if self.preview_enabled:
                try:
                    annotated = self._draw_overlay(frame, detections, foam_profile, coverage)
                    cv2.imshow(TARGET_WINDOW_NAME, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self.preview_enabled = False
                except Exception as exc:
                    self.owner.fault("VISION PREVIEW", f"{type(exc).__name__}: {exc}", "disable preview only")
                    self.preview_enabled = False

        print("[VISION] frame thread stopped")

    # ---------------- image normalization / color ----------------
    def _normalize_lighting(self, frame):
        if self._clahe is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_norm = self._clahe.apply(l_chan)
        return cv2.cvtColor(cv2.merge((l_norm, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _hue_distance(h1, h2):
        raw = abs(float(h1) - float(h2))
        return min(raw, 180.0 - raw)

    def _color_confidence(self, color_name, median_hsv, median_lab):
        h, s, v = median_hsv
        hue_score = max(0.0, 1.0 - self._hue_distance(h, TARGET_HUE_CENTERS[color_name]) / 30.0)
        saturation_score = clamp((s - 45.0) / 100.0, 0.0, 1.0)
        brightness_score = clamp((v - 25.0) / 90.0, 0.0, 1.0)
        _l, a, b = median_lab
        if color_name == "GREEN":
            lab_score = clamp((128.0 - a) / 45.0 + 0.25, 0.0, 1.0)
        elif color_name == "RED":
            lab_score = clamp((a - 128.0) / 45.0 + 0.25, 0.0, 1.0)
        elif color_name == "YELLOW":
            lab_score = clamp((b - 128.0) / 55.0 + 0.20, 0.0, 1.0)
        elif color_name == "BLUE":
            lab_score = clamp((128.0 - b) / 55.0 + 0.20, 0.0, 1.0)
        else:
            lab_score = 0.5
        return float(0.50*hue_score + 0.22*saturation_score + 0.08*brightness_score + 0.20*lab_score)

    @staticmethod
    def _inside_statistics(contour, hsv, lab):
        region = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, thickness=-1)
        ys, xs = np.where(region > 0)
        if len(xs) == 0:
            return (0.0, 0.0, 0.0), (0.0, 128.0, 128.0)
        hsv_pixels = hsv[ys, xs]
        lab_pixels = lab[ys, xs]
        return (
            tuple(float(v) for v in np.median(hsv_pixels, axis=0)),
            tuple(float(v) for v in np.median(lab_pixels, axis=0)),
        )

    # ---------------- ROI / foam-board gate ----------------
    def _active_roi(self):
        with self.roi_lock:
            if self.dynamic_roi is not None:
                return tuple(self.dynamic_roi)
        return tuple(TARGET_SEARCH_ROI)

    def _set_focus_roi(self, candidate):
        if not isinstance(candidate, dict):
            return
        try:
            cx, cy = [float(v) for v in candidate.get("center_norm", (0.5, 0.5))]
            _x, _y, bw, bh = [float(v) for v in candidate.get("bbox_norm", (0, 0, 0, 0))]
        except Exception:
            return
        base_x1, base_y1, base_x2, base_y2 = TARGET_SEARCH_ROI
        rw = clamp(max(TARGET_DYNAMIC_ROI_MIN_W, bw*TARGET_DYNAMIC_ROI_BBOX_SCALE), TARGET_DYNAMIC_ROI_MIN_W, TARGET_DYNAMIC_ROI_MAX_W)
        rh = clamp(max(TARGET_DYNAMIC_ROI_MIN_H, bh*TARGET_DYNAMIC_ROI_BBOX_SCALE), TARGET_DYNAMIC_ROI_MIN_H, TARGET_DYNAMIC_ROI_MAX_H)
        rw = min(rw, base_x2-base_x1)
        rh = min(rh, base_y2-base_y1)
        x1 = clamp(cx-rw*0.5, base_x1, base_x2-rw)
        y1 = clamp(cy-rh*0.5, base_y1, base_y2-rh)
        with self.roi_lock:
            self.dynamic_roi = (x1, y1, x1+rw, y1+rh)

    def _clear_focus_roi(self):
        with self.roi_lock:
            self.dynamic_roi = None

    def _roi_px(self, frame):
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self._active_roi()
        x1 = max(0, min(w-1, int(round(rx1*w))))
        x2 = max(x1+1, min(w, int(round(rx2*w))))
        y1 = max(0, min(h-1, int(round(ry1*h))))
        y2 = max(y1+1, min(h, int(round(ry2*h))))
        return x1, y1, x2, y2

    def _compute_foam_wall_profile(self, frame):
        if not TARGET_FOAM_GATE_ENABLED or frame is None:
            return None, 0.0, []
        frame_h, frame_w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.asarray(TARGET_FOAM_HSV_LOW, dtype=np.uint8),
            np.asarray(TARGET_FOAM_HSV_HIGH, dtype=np.uint8),
        )
        ok = max(1, int(TARGET_FOAM_OPEN_KERNEL)); ck = max(1, int(TARGET_FOAM_CLOSE_KERNEL))
        if ok % 2 == 0: ok += 1
        if ck % 2 == 0: ck += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok,ok), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ck,ck), np.uint8))
        found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found)==2 else found[1]
        profile = np.full(frame_w, np.nan, dtype=np.float32)
        components = []
        min_bottom = frame_h * TARGET_FOAM_COMPONENT_MIN_BOTTOM_FRAC
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x,y,bw,bh = cv2.boundingRect(contour)
            if area < TARGET_FOAM_MIN_COMPONENT_AREA_PX or bw < TARGET_FOAM_MIN_COMPONENT_WIDTH_PX or bh < TARGET_FOAM_MIN_COMPONENT_HEIGHT_PX:
                continue
            if y+bh < min_bottom:
                continue
            components.append({"bbox_px":[int(x),int(y),int(bw),int(bh)], "area_px":area})
            local = np.zeros((bh,bw), dtype=np.uint8)
            shifted = contour.copy(); shifted[:,:,0] -= x; shifted[:,:,1] -= y
            cv2.drawContours(local, [shifted], -1, 255, thickness=-1)
            for lx in range(bw):
                ys = np.flatnonzero(local[:,lx])
                if ys.size:
                    px = x+lx; top_y = float(y+int(ys[0]))
                    if not math.isfinite(float(profile[px])) or top_y < float(profile[px]):
                        profile[px] = top_y
        valid = np.flatnonzero(np.isfinite(profile))
        max_gap = int(TARGET_FOAM_PROFILE_MAX_INTERP_GAP_PX)
        if valid.size >= 2 and max_gap > 0:
            for li,ri in zip(valid[:-1], valid[1:]):
                gap = int(ri-li-1)
                if 0 < gap <= max_gap:
                    profile[li:ri+1] = np.linspace(float(profile[li]), float(profile[ri]), int(ri-li+1), dtype=np.float32)
        radius = int(TARGET_FOAM_PROFILE_MEDIAN_RADIUS_PX)
        if radius > 0 and np.isfinite(profile).any():
            src = profile.copy(); smooth = profile.copy()
            for px in np.flatnonzero(np.isfinite(src)):
                vals = src[max(0,px-radius):min(frame_w,px+radius+1)]
                vals = vals[np.isfinite(vals)]
                if vals.size: smooth[px] = float(np.median(vals))
            profile = smooth
        rx1 = max(0,min(frame_w-1,int(TARGET_SEARCH_ROI[0]*frame_w)))
        rx2 = max(rx1+1,min(frame_w,int(TARGET_SEARCH_ROI[2]*frame_w)))
        rs = profile[rx1:rx2]
        coverage = float(np.count_nonzero(np.isfinite(rs))) / float(max(1,rs.size))
        return profile, coverage, components

    @staticmethod
    def _foam_wall_y(profile, x_px):
        if profile is None:
            return None
        x = int(round(float(x_px)))
        if x < 0 or x >= len(profile):
            return None
        r = int(TARGET_FOAM_PROFILE_SAMPLE_RADIUS_PX)
        vals = np.asarray(profile[max(0,x-r):min(len(profile),x+r+1)], dtype=float)
        vals = vals[np.isfinite(vals)]
        return None if vals.size == 0 else float(np.median(vals))

    def _foam_gate(self, frame_w, frame_h, cx, cy, bbox_px, profile, profile_t=None, coverage=None):
        if not TARGET_FOAM_GATE_ENABLED:
            return True, None, "disabled", 1.0
        if profile is None:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "profile-missing", 0.0
            return True, None, "profile-missing-fail-open", 1.0
        if coverage is not None and coverage < TARGET_FOAM_MIN_ROI_COVERAGE:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "profile-low-coverage", 0.0
            return True, None, "profile-low-coverage-fail-open", 1.0
        wall_y = self._foam_wall_y(profile, cx)
        if wall_y is None:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "local-profile-missing", 0.0
            return True, None, "local-profile-missing-fail-open", 1.0
        x,y,bw,bh = [float(v) for v in bbox_px]
        bh = max(1.0,bh)
        cutoff = wall_y + TARGET_FOAM_CENTER_MARGIN_PX
        bbox_bottom = y+bh
        below_h = max(0.0, bbox_bottom-max(y,cutoff))
        below_frac = clamp(below_h/bh,0.0,1.0)
        if float(cy) < cutoff:
            return False, wall_y, "centre-above-foam", below_frac
        if below_frac < TARGET_FOAM_MIN_BBOX_BELOW_FRAC:
            return False, wall_y, "bbox-cut-by-foam-edge", below_frac
        return True, wall_y, "foam-pass", below_frac

    # ---------------- geometry / detector ----------------
    def _classify_shape(self, contour):
        area = float(cv2.contourArea(contour)); perimeter = float(cv2.arcLength(contour, True))
        if area <= 1.0 or perimeter <= 1.0:
            return None
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area/hull_area if hull_area > 1e-6 else 0.0
        if solidity < TARGET_MIN_SOLIDITY:
            return None
        approx = cv2.approxPolyDP(contour, TARGET_POLY_EPS_FRAC*perimeter, True)
        x,y,w,h = cv2.boundingRect(contour)
        fill = area/float(max(1,w*h))
        circularity = 4.0*math.pi*area/max(1e-9, perimeter*perimeter)
        shape = None; quality = 0.0
        if len(approx)==4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1,2).astype(float); cosines=[]
            for i in range(4):
                v1=pts[(i-1)%4]-pts[i]; v2=pts[(i+1)%4]-pts[i]
                den=float(np.linalg.norm(v1)*np.linalg.norm(v2))
                cosines.append(1.0 if den<=1e-9 else abs(float(np.dot(v1,v2))/den))
            max_cos=max(cosines) if cosines else 1.0
            rect=cv2.minAreaRect(contour); rw,rh=rect[1]
            if rw>1.0 and rh>1.0 and fill>=TARGET_RECT_MIN_FILL and max_cos<=TARGET_RECT_MAX_CORNER_COS:
                aspect_rot=max(rw,rh)/max(1e-6,min(rw,rh))
                if TARGET_SQUARE_ASPECT_MIN <= aspect_rot <= TARGET_SQUARE_ASPECT_MAX:
                    shape="SQUARE"
                else:
                    axis=float(w)/max(1.0,float(h))
                    if axis>=TARGET_RECT_ASPECT_MIN: shape="RECT_HORIZONTAL"
                    elif (1.0/max(axis,1e-6))>=TARGET_RECT_ASPECT_MIN: shape="RECT_VERTICAL"
                quality=0.45*solidity+0.35*min(1.0,fill)+0.20*(1.0-min(1.0,max_cos))
        elif len(approx)>=5 and circularity>=TARGET_CIRCLE_MIN_CIRCULARITY:
            axis=float(w)/max(1.0,float(h))
            if TARGET_CIRCLE_ASPECT_MIN <= axis <= TARGET_CIRCLE_ASPECT_MAX:
                shape="CIRCLE"; quality=0.55*solidity+0.45*min(1.0,circularity)
        if shape is None:
            return None
        return {"shape":shape,"quality":float(quality),"solidity":solidity,"fill":fill,"circularity":circularity,"bbox_local":[x,y,w,h],"area":area}

    def _detect_targets(self, frame, foam_profile=None, foam_coverage=None):
        normalized = self._normalize_lighting(frame)
        frame_h,frame_w=frame.shape[:2]
        x1,y1,x2,y2=self._roi_px(frame)
        roi=normalized[y1:y2,x1:x2]
        if roi.size==0:
            return [],[]
        roi_h,roi_w=roi.shape[:2]; roi_area=float(max(1,roi_h*roi_w))
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV); lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)
        ok=max(1,int(TARGET_MORPH_OPEN_KERNEL)); ck=max(1,int(TARGET_MORPH_CLOSE_KERNEL))
        if ok%2==0: ok+=1
        if ck%2==0: ck+=1
        k_open=np.ones((ok,ok),np.uint8); k_close=np.ones((ck,ck),np.uint8)
        coverage = None if foam_coverage is None else float(foam_coverage)
        detections=[]; blobs=[]
        focus_active=self.dynamic_roi is not None
        for color_name,ranges in TARGET_HSV_RANGES.items():
            mask=np.zeros((roi_h,roi_w),dtype=np.uint8)
            for lo,hi in ranges:
                mask=cv2.bitwise_or(mask,cv2.inRange(hsv,np.asarray(lo,np.uint8),np.asarray(hi,np.uint8)))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k_open)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k_close)
            found=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours=found[0] if len(found)==2 else found[1]
            for contour in contours:
                area=float(cv2.contourArea(contour)); frac=area/roi_area
                min_frac=TARGET_MIN_AREA_FRAC_ROI*(0.55 if focus_active else 1.0)
                max_frac=0.62 if focus_active else TARGET_MAX_AREA_FRAC_ROI
                if area<TARGET_MIN_AREA_PX or frac<min_frac or frac>max_frac:
                    continue
                bx,by,bw,bh=cv2.boundingRect(contour)
                margin=2 if focus_active else TARGET_BORDER_MARGIN_PX
                # Explicit cut-border rejection from the friend's detector: a
                # contour clipped by the current ROI has untrustworthy shape.
                if bx<=margin or by<=margin or bx+bw>=roi_w-margin or by+bh>=roi_h-margin:
                    continue
                gx,gy=x1+bx,y1+by; cx=gx+bw*0.5; cy=gy+bh*0.5
                foam_ok,foam_y,foam_reason,below_frac=self._foam_gate(
                    frame_w,frame_h,cx,cy,[gx,gy,bw,bh],foam_profile,profile_t=time.monotonic(),coverage=coverage
                )
                if not foam_ok:
                    continue
                median_hsv,median_lab=self._inside_statistics(contour,hsv,lab)
                color_conf=self._color_confidence(color_name,median_hsv,median_lab)
                blob={
                    "kind":"COLOR_BLOB","color":color_name,
                    "center_norm":[cx/frame_w,cy/frame_h],
                    "bbox_norm":[gx/frame_w,gy/frame_h,bw/frame_w,bh/frame_h],
                    "bbox_px":[int(gx),int(gy),int(bw),int(bh)],
                    "area_frac_roi":frac,"color_confidence":color_conf,
                    "median_hsv":list(median_hsv),"median_lab":list(median_lab),
                    "foam_gate":foam_reason,"foam_wall_y_px":foam_y,"foam_bbox_below_frac":below_frac,
                }
                blobs.append(blob)
                classified=self._classify_shape(contour)
                if classified is None:
                    continue
                area_score=min(1.0,frac/0.025)
                score=clamp(0.58*color_conf+0.34*float(classified["quality"])+0.08*area_score,0.0,1.0)
                if score<TARGET_MIN_CONFIDENCE:
                    continue
                item=dict(blob)
                item.update({
                    "kind":"COLOR_SHAPE","shape":classified["shape"],"score":score,
                    "shape_confidence":float(classified["quality"]),"solidity":classified["solidity"],
                    "fill":classified["fill"],"circularity":classified["circularity"],
                })
                detections.append(item)
        detections.sort(key=lambda d:float(d.get("score",0.0)),reverse=True)
        blobs.sort(key=lambda d:float(d.get("color_confidence",0.0)),reverse=True)
        return detections,blobs

    # ---------------- search / tracking / verification ----------------
    def _fresh_frame_count(self, since_t):
        with self.lock:
            return sum(1 for ts,_seq,_d in self.history if ts>=float(since_t))

    def _snapshot_latest(self):
        with self.lock:
            return self.latest_frame_t, [dict(d) for d in self.latest_detections], [dict(b) for b in self.latest_color_blobs]

    @staticmethod
    def _center_distance(a,b):
        try:
            return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
        except Exception:
            return float("inf")

    def _best_candidate_since(self, since_t):
        with self.lock:
            rows=[(ts,[dict(d) for d in dets]) for ts,_seq,dets in self.history if ts>=float(since_t)]
        candidates=[]
        for _ts,dets in rows:
            candidates.extend(d for d in dets if float(d.get("score",0.0))>=TARGET_MIN_CONFIDENCE)
        if not candidates:
            return None
        return dict(max(candidates,key=lambda d:float(d.get("score",0.0))))

    def _quick_gate_candidate_since(self, since_t, min_frames=TARGET_SEARCH_QUICK_GATE_FRAMES):
        """Require the same color/shape candidate on distinct recent frames.

        This mirrors the linked branch's cheap multi-frame candidate gate so a
        one-frame colored reflection does not stop the gimbal and enter AIM.
        """
        with self.lock:
            rows=[(ts,[dict(d) for d in dets]) for ts,_seq,dets in self.history if ts>=float(since_t)]
        tracks=[]
        needed=max(1,int(min_frames))
        for _ts,dets in rows:
            next_tracks=[]; used=set()
            for d in dets:
                if float(d.get("score",0.0))<TARGET_MIN_CONFIDENCE:
                    continue
                best_i=None; best_dist=None
                for i,tr in enumerate(tracks):
                    if i in used: continue
                    if tr["color"]!=d.get("color") or tr["shape"]!=d.get("shape"):
                        continue
                    dist=self._center_distance(tr["center"],d.get("center_norm"))
                    if dist>TARGET_TRACK_MAX_JUMP_NORM:
                        continue
                    if best_dist is None or dist<best_dist:
                        best_i=i; best_dist=dist
                if best_i is None:
                    next_tracks.append({"color":d.get("color"),"shape":d.get("shape"),"count":1,"center":d.get("center_norm"),"best":dict(d)})
                else:
                    used.add(best_i); tr=tracks[best_i]
                    best=dict(d) if float(d.get("score",0.0))>=float(tr["best"].get("score",0.0)) else dict(tr["best"])
                    next_tracks.append({"color":d.get("color"),"shape":d.get("shape"),"count":int(tr["count"])+1,"center":d.get("center_norm"),"best":best})
            tracks=next_tracks
            qualified=[tr for tr in tracks if int(tr["count"])>=needed]
            if qualified:
                winner=max(qualified,key=lambda tr:(int(tr["count"]),float(tr["best"].get("score",0.0))))
                out=dict(winner["best"]); out["quick_gate_frames"]=int(winner["count"]); return out
        return None

    @staticmethod
    def _shape_compatible(expected_shape, observed_shape):
        """Allow perspective jitter among square/rectangle labels during AIM only.

        A planar quadrilateral can flip SQUARE <-> RECT_VERTICAL/HORIZONTAL as
        the gimbal moves.  Circle remains exact.  Color + spatial continuity
        still have to match, and the final fire gate still needs a stable target.
        """
        a=str(expected_shape or "")
        b=str(observed_shape or "")
        if a == b:
            return True
        quad={"SQUARE","RECT_VERTICAL","RECT_HORIZONTAL"}
        return a in quad and b in quad

    def _latest_match(self, expected, expected_center, allow_blob_fallback=True):
        _ts,dets,blobs=self._snapshot_latest()
        exact=[d for d in dets if d.get("color")==expected.get("color") and self._shape_compatible(expected.get("shape"), d.get("shape"))]
        if exact:
            best=min(exact,key=lambda d:self._center_distance(d.get("center_norm"),expected_center))
            if self._center_distance(best.get("center_norm"),expected_center)<=TARGET_TRACK_MAX_JUMP_NORM:
                return dict(best)
        if allow_blob_fallback:
            same=[b for b in blobs if b.get("color")==expected.get("color")]
            if same:
                best=min(same,key=lambda b:self._center_distance(b.get("center_norm"),expected_center))
                if self._center_distance(best.get("center_norm"),expected_center)<=TARGET_TRACK_MAX_JUMP_NORM:
                    pseudo=dict(expected)
                    pseudo.update(best)
                    pseudo["kind"]="COLOR_SHAPE"
                    pseudo["shape"]=expected.get("shape")
                    pseudo["score"]=max(float(expected.get("score",0.5)),float(best.get("color_confidence",0.5)))
                    pseudo["shape_fallback_track"]=True
                    return pseudo
        return None

    def _verify_exact(self, expected, since_t=None):
        start=time.monotonic() if since_t is None else float(since_t)
        deadline=time.monotonic()+TARGET_VERIFY_WINDOW_SEC
        last_seq=-1; hits=[]
        while self.owner.running and self.running and time.monotonic()<deadline and len(hits)<TARGET_VERIFY_FRAMES:
            with self.lock:
                seq=self.frame_seq; dets=[dict(d) for d in self.latest_detections]; frame_t=self.latest_frame_t
            if seq==last_seq or frame_t<start:
                time.sleep(TARGET_AIM_POLL_SEC); continue
            last_seq=seq
            matches=[d for d in dets if d.get("color")==expected.get("color") and self._shape_compatible(expected.get("shape"), d.get("shape")) and float(d.get("score",0.0))>=TARGET_MIN_CONFIDENCE]
            if not matches:
                hits=[]
                time.sleep(TARGET_AIM_POLL_SEC); continue
            ref=expected.get("center_norm",(0.5,0.5))
            best=min(matches,key=lambda d:self._center_distance(d.get("center_norm"),ref))
            if self._center_distance(best.get("center_norm"),ref)>TARGET_TRACK_MAX_JUMP_NORM:
                hits=[]; time.sleep(TARGET_AIM_POLL_SEC); continue
            hits.append(best); expected=dict(best)
            time.sleep(TARGET_AIM_POLL_SEC)
        if len(hits)<TARGET_VERIFY_FRAMES:
            return None
        scores=[float(d.get("score",0.0)) for d in hits]
        xs=[float(d["center_norm"][0]) for d in hits]; ys=[float(d["center_norm"][1]) for d in hits]
        areas=[float(d.get("area_frac_roi",0.0)) for d in hits]
        center_std=math.hypot(statistics.pstdev(xs) if len(xs)>1 else 0.0, statistics.pstdev(ys) if len(ys)>1 else 0.0)
        area_mean=statistics.fmean(areas) if areas else 0.0
        area_cv=(statistics.pstdev(areas)/max(area_mean,1e-6)) if len(areas)>1 else 0.0
        mean_score=statistics.fmean(scores)
        if mean_score<TARGET_SAVE_CONFIDENCE or center_std>TARGET_VERIFY_MAX_CENTER_STD or area_cv>TARGET_VERIFY_MAX_AREA_CV:
            return None
        best=dict(max(hits,key=lambda d:float(d.get("score",0.0))))
        best["confirm_frames"]=len(hits); best["temporal_score"]=mean_score; best["center_std"]=center_std; best["area_cv"]=area_cv
        return best

    def _stop_gimbal_velocity(self):
        if self.owner.gimbal is None:
            return
        try:
            self.owner.gimbal.drive_speed(pitch_speed=0.0,yaw_speed=0.0)
        except Exception:
            pass

    def _goto_target_pose_strict(self, yaw_deg, pitch_deg, timeout_sec=1.35, allow_soft=False):
        """Fast feedback-driven target-search pose.

        V20 used SDK moveto() here and the generic gimbal_goto() could retry the
        same failed action several times before falling back to velocity control.
        Field log 2026-10-01 showed the turret sitting around -53.6 deg while
        FRONT wanted -45 deg, wasting seconds and causing whole sectors to skip.

        Target SEARCH does not require topology-grade exact pitch.  Drive the
        turret directly from sub_angle feedback, detect a real stall, and permit
        a bounded soft yaw hand-off because the following continuous sweep will
        still cross the acquisition cone.
        """
        yaw_deg = clamp(float(yaw_deg), GIMBAL_SOFT_YAW_MIN_DEG, GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg = clamp(float(pitch_deg), TARGET_CENTER_PITCH_HARD_MIN_DEG, TARGET_CENTER_PITCH_MAX_DEG)

        p0, y0 = self.owner.current_gimbal_relative()
        yaw_distance = 90.0 if y0 is None else abs(wrap_deg(yaw_deg - float(y0)))
        pitch_distance = 10.0 if p0 is None else abs(float(pitch_deg) - float(p0))
        dynamic_timeout = max(
            float(timeout_sec),
            yaw_distance / max(45.0, GIMBAL_RECOVERY_YAW_MAX_DPS * 0.82) + 0.40,
            pitch_distance / max(14.0, GIMBAL_RECOVERY_PITCH_MAX_DPS * 0.75) + 0.35,
        )
        deadline = time.monotonic() + max(0.35, dynamic_timeout)
        stable = 0
        last_progress_t = time.monotonic()
        last_yaw = y0
        stall_pulses = 0

        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                p_now, y_now = self.owner.current_gimbal_relative()
                if p_now is None or y_now is None:
                    self._stop_gimbal_velocity()
                    time.sleep(0.04)
                    continue

                ey = wrap_deg(yaw_deg - float(y_now))
                ep = float(pitch_deg) - float(p_now)
                yaw_ok = abs(ey) <= TARGET_SEARCH_POSE_YAW_TOL_DEG
                pitch_ok = abs(ep) <= TARGET_SEARCH_POSE_PITCH_TOL_DEG
                if yaw_ok and pitch_ok:
                    stable += 1
                    self._stop_gimbal_velocity()
                    if stable >= 2:
                        return True
                    time.sleep(0.04)
                    continue
                stable = 0

                # Progress watchdog: repeated low-speed commands around a sticky
                # point were seen to leave the EP gimbal sitting at -53.x deg.
                if last_yaw is None or angle_diff_deg(float(y_now), float(last_yaw)) >= TARGET_GIMBAL_STALL_MIN_PROGRESS_DEG:
                    last_yaw = float(y_now)
                    last_progress_t = time.monotonic()
                    stall_pulses = 0
                elif abs(ey) > TARGET_SEARCH_POSE_YAW_TOL_DEG and (time.monotonic() - last_progress_t) >= TARGET_GIMBAL_STALL_WINDOW_SEC:
                    self._stop_gimbal_velocity()
                    time.sleep(0.045)
                    stall_pulses += 1
                    last_progress_t = time.monotonic()

                ys = 0.0 if yaw_ok else clamp(3.0 * ey, -GIMBAL_RECOVERY_YAW_MAX_DPS, +GIMBAL_RECOVERY_YAW_MAX_DPS)
                ps = 0.0 if pitch_ok else clamp(2.2 * ep, -GIMBAL_RECOVERY_PITCH_MAX_DPS, +GIMBAL_RECOVERY_PITCH_MAX_DPS)

                # Strong enough to overcome gimbal stiction, but only while more
                # than a few degrees away.  Near target we let proportional speed
                # settle instead of hammering it at 120-300 dps.
                yaw_floor = 22.0 if abs(ey) > 6.0 else 10.0
                if 0.0 < abs(ys) < yaw_floor:
                    ys = math.copysign(yaw_floor, ys)
                if 0.0 < abs(ps) < 3.0:
                    ps = math.copysign(3.0, ps)
                if stall_pulses > 0 and abs(ey) > 6.0:
                    ys = math.copysign(min(GIMBAL_RECOVERY_YAW_MAX_DPS, 85.0), ey)

                self.owner.gimbal.drive_speed(pitch_speed=float(ps), yaw_speed=float(ys))
                time.sleep(0.040)
        except Exception as exc:
            self.owner.fault(
                "TARGET POSE",
                f"yaw={yaw_deg:+.1f} pitch={pitch_deg:+.1f}: {type(exc).__name__}: {exc}",
                "soft-check pose / continue mission",
            )
        finally:
            self._stop_gimbal_velocity()

        p_now, y_now = self.owner.current_gimbal_relative()
        if p_now is None or y_now is None:
            return False
        yaw_err = abs(wrap_deg(float(y_now) - yaw_deg))
        pitch_err = abs(float(p_now) - pitch_deg)
        if yaw_err <= TARGET_SEARCH_POSE_YAW_TOL_DEG and pitch_err <= TARGET_SEARCH_POSE_PITCH_TOL_DEG:
            return True
        if allow_soft and yaw_err <= TARGET_SEARCH_POSE_SOFT_YAW_TOL_DEG:
            print(
                f"[TARGET GIMBAL SOFT HANDOFF] targetYaw={yaw_deg:+.1f} actual={float(y_now):+.1f} "
                f"yawErr={yaw_err:.1f} pitch={float(p_now):+.1f} -> sweep continues"
            )
            return True
        return False

    def _aim_velocity(self, center_norm, yaw_min=None, yaw_max=None):
        try:
            cx,cy=[float(v) for v in center_norm]
        except Exception:
            self._stop_gimbal_velocity(); return False
        desired_x=0.5+TARGET_AIM_OFFSET_X; desired_y=0.5+TARGET_AIM_OFFSET_Y
        ex=cx-desired_x; ey=cy-desired_y
        yaw_speed=0.0 if abs(ex)<=TARGET_AIM_DEADBAND_X else clamp(ex*TARGET_AIM_YAW_GAIN_DPS,-TARGET_AIM_YAW_MAX_DPS,TARGET_AIM_YAW_MAX_DPS)
        # V7: horizontal servo keeps the existing limits, while vertical
        # centering gets its own deeper/flexible pitch authority.  This is only
        # called after SHOOT-INTENT, never during DFS topology scans.
        pitch_speed=0.0 if abs(ey)<=TARGET_AIM_DEADBAND_Y else clamp(
            -ey*TARGET_CENTER_PITCH_GAIN_DPS,
            -TARGET_CENTER_PITCH_MAX_DPS,
            +TARGET_CENTER_PITCH_MAX_DPS,
        )
        p_now,y_now=self.owner.current_gimbal_relative()
        if y_now is not None:
            # Global mechanical/software limit intersected with the sector's
            # +/-30deg crosshair envelope.  The narrow +/-10deg acquisition
            # cone is intentionally NOT used here: once shoot-intent is earned,
            # the crosshair may move farther to put the real target centre on aim.
            lo=-float(TARGET_AIM_YAW_LIMIT_DEG)
            hi=+float(TARGET_AIM_YAW_LIMIT_DEG)
            if yaw_min is not None:
                lo=max(lo,float(yaw_min))
            if yaw_max is not None:
                hi=min(hi,float(yaw_max))
            if y_now<=lo and yaw_speed<0: yaw_speed=0.0
            if y_now>=hi and yaw_speed>0: yaw_speed=0.0
        if p_now is not None:
            # V7 FLEX CENTER PITCH: the old -14deg limit was correct for search
            # but too shallow for close targets low in the camera.  During AIM
            # only, brake near -20.5deg and permit a hard floor down to -22.5deg.
            # Search resumes at -10deg immediately after target service.
            if p_now<=TARGET_CENTER_PITCH_HARD_MIN_DEG and pitch_speed<0:
                pitch_speed=0.0
            elif pitch_speed<0 and p_now < (TARGET_CENTER_PITCH_SOFT_MIN_DEG + TARGET_CENTER_PITCH_BRAKE_ZONE_DEG):
                headroom=max(0.0,float(p_now)-TARGET_CENTER_PITCH_HARD_MIN_DEG)
                pitch_speed=max(float(pitch_speed),-max(2.0,6.0*headroom))
            if p_now>=TARGET_CENTER_PITCH_MAX_DEG and pitch_speed>0:
                pitch_speed=0.0
        try:
            self.owner.gimbal.drive_speed(pitch_speed=float(pitch_speed),yaw_speed=float(yaw_speed))
            return True
        except Exception as exc:
            self.owner.fault("TARGET AIM",f"drive_speed failed: {type(exc).__name__}: {exc}","abort target lock only")
            return False

    def _aim_and_lock(self,candidate, aim_yaw_min=None, aim_yaw_max=None):
        current=dict(candidate); expected=list(current.get("center_norm",(0.5,0.5)))
        self.owner.safe_stop(); self._set_focus_roi(current)
        # Keep the chassis on its DFS cardinal while the turret servos.  The log
        # showed repeated target locks could otherwise accumulate ~12deg base yaw.
        chassis_hold_yaw=self.owner.desired_yaw_for_heading(self.owner.heading)
        self.owner.pid_straight.reset()
        deadline=time.monotonic()+TARGET_AIM_TIMEOUT_SEC
        grace_used=False
        last_seq=-1; lost_since=None; centered_frames=0
        last_err=(None,None); verify_failures=0
        print(f"[TARGET ACQUIRE] {current.get('color')} {current.get('shape')} score={current.get('score',0):.2f}")
        if aim_yaw_min is not None and aim_yaw_max is not None:
            print(
                "[TARGET CROSSHAIR ENVELOPE] yaw={:+.1f}..{:+.1f}deg ".format(
                    float(aim_yaw_min), float(aim_yaw_max),
                )
                + "pitchCenter={:+.1f}..{:+.1f}deg hardDown={:+.1f}deg".format(
                    float(TARGET_CENTER_PITCH_SOFT_MIN_DEG),
                    float(TARGET_CENTER_PITCH_MAX_DEG),
                    float(TARGET_CENTER_PITCH_HARD_MIN_DEG),
                )
            )
        try:
            while self.owner.running and self.running and time.monotonic()<deadline:
                # Counter gimbal reaction torque with zero-translation chassis
                # yaw hold.  Failure is non-fatal to target service; the sector
                # hand-off performs a final stationary alignment as well.
                if chassis_hold_yaw is not None:
                    try:
                        z_hold=self.owner.yaw_hold_command(chassis_hold_yaw,stationary=False)
                        self.owner.drive_speed_resilient(
                            x=0.0,y=0.0,z=z_hold,timeout=DRIVE_COMMAND_TIMEOUT,
                            label="TARGET AIM YAW HOLD",
                        )
                    except Exception:
                        pass
                with self.lock: seq=self.frame_seq
                if seq==last_seq:
                    time.sleep(TARGET_AIM_POLL_SEC); continue
                last_seq=seq
                latest=self._latest_match(current,expected,allow_blob_fallback=True)
                if latest is None:
                    self._stop_gimbal_velocity(); centered_frames=0
                    if lost_since is None: lost_since=time.monotonic()
                    if time.monotonic()-lost_since>TARGET_AIM_LOST_GRACE_SEC:
                        print("[TARGET LOST] lock grace expired")
                        return None
                    time.sleep(TARGET_AIM_POLL_SEC); continue
                lost_since=None; current.update(latest); expected=list(current.get("center_norm",expected)); self._set_focus_roi(current)
                cx,cy=[float(v) for v in expected]
                ex=cx-(0.5+TARGET_AIM_OFFSET_X); ey=cy-(0.5+TARGET_AIM_OFFSET_Y)
                last_err=(ex,ey)
                if abs(ex)<=TARGET_AIM_CENTER_TOL_X and abs(ey)<=TARGET_AIM_CENTER_TOL_Y:
                    centered_frames+=1; self._stop_gimbal_velocity()
                else:
                    centered_frames=0
                    if not self._aim_velocity(expected, yaw_min=aim_yaw_min, yaw_max=aim_yaw_max): return None
                self.status=f"AIM {current.get('color')} {current.get('shape')} err=({ex:+.3f},{ey:+.3f}) hold={centered_frames}/{TARGET_AIM_CENTER_HOLD_FRAMES}"
                if centered_frames>=TARGET_AIM_CENTER_HOLD_FRAMES:
                    # V15: the quick-gate already established a multi-frame real target,
                    # and AIM itself has now held the BBOX centre on the crosshair for
                    # 3 consecutive fresh frames.  Do not add another verification wait
                    # here: proceed directly to fresh ToF -> camera/muzzle compensation.
                    self._stop_gimbal_velocity()
                    locked = dict(current)
                    locked["crosshair_centered"] = True
                    locked["crosshair_error_norm"] = [float(ex), float(ey)]
                    locked["center_hold_frames"] = int(centered_frames)
                    print(
                        "[TARGET CENTER LOCK] {} {} bbox-center err=({:+.3f},{:+.3f}) "
                        "hold={}/{} -> ToF/fire pipeline".format(
                            locked.get("color"), locked.get("shape"), ex, ey,
                            centered_frames, TARGET_AIM_CENTER_HOLD_FRAMES,
                        )
                    )
                    self._set_focus_roi(locked)
                    return locked
                time.sleep(TARGET_AIM_POLL_SEC)
            # One short grace only when the target is genuinely almost centered.
            ex,ey=last_err
            if (
                not grace_used and ex is not None and ey is not None
                and abs(ex) <= TARGET_AIM_NEAR_CENTER_GRACE_ERR
                and abs(ey) <= TARGET_AIM_NEAR_CENTER_GRACE_ERR
                and self.owner.running and self.running
            ):
                grace_used=True
                deadline=time.monotonic()+TARGET_AIM_NEAR_CENTER_GRACE_SEC
                print(
                    f"[TARGET AIM GRACE] near center err=({ex:+.3f},{ey:+.3f}) "
                    f"+{TARGET_AIM_NEAR_CENTER_GRACE_SEC:.2f}s"
                )
                while self.owner.running and self.running and time.monotonic()<deadline:
                    with self.lock: seq=self.frame_seq
                    if seq==last_seq:
                        time.sleep(TARGET_AIM_POLL_SEC); continue
                    last_seq=seq
                    latest=self._latest_match(current,expected,allow_blob_fallback=True)
                    if latest is None:
                        time.sleep(TARGET_AIM_POLL_SEC); continue
                    current.update(latest); expected=list(current.get("center_norm",expected)); self._set_focus_roi(current)
                    cx,cy=[float(v) for v in expected]
                    ex=cx-(0.5+TARGET_AIM_OFFSET_X); ey=cy-(0.5+TARGET_AIM_OFFSET_Y)
                    last_err=(ex,ey)
                    if abs(ex)<=TARGET_AIM_CENTER_TOL_X and abs(ey)<=TARGET_AIM_CENTER_TOL_Y:
                        centered_frames+=1; self._stop_gimbal_velocity()
                    else:
                        centered_frames=0
                        if not self._aim_velocity(expected, yaw_min=aim_yaw_min, yaw_max=aim_yaw_max): break
                    if centered_frames>=TARGET_AIM_CENTER_HOLD_FRAMES:
                        locked=dict(current)
                        locked["crosshair_centered"]=True
                        locked["crosshair_error_norm"]=[float(ex),float(ey)]
                        locked["center_hold_frames"]=int(centered_frames)
                        print(
                            f"[TARGET CENTER LOCK] {locked.get('color')} {locked.get('shape')} "
                            f"bbox-center err=({ex:+.3f},{ey:+.3f}) hold={centered_frames}/{TARGET_AIM_CENTER_HOLD_FRAMES} -> ToF/fire pipeline"
                        )
                        self._set_focus_roi(locked)
                        return locked
                    time.sleep(TARGET_AIM_POLL_SEC)

            p_end,y_end=self.owner.current_gimbal_relative()
            ex,ey=last_err
            print(
                "[TARGET AIM TIMEOUT] yaw={} pitch={} err=({}, {}) verifyFail={}".format(
                    fmt_deg(y_end), fmt_deg(p_end),
                    "NA" if ex is None else "{:+.3f}".format(ex),
                    "NA" if ey is None else "{:+.3f}".format(ey),
                    verify_failures,
                )
            )
            return None
        finally:
            self._stop_gimbal_velocity(); self._clear_focus_roi()
            self.owner.safe_stop()
            if chassis_hold_yaw is not None:
                try:
                    aim_exit_err = self.owner.yaw_error_deg(chassis_hold_yaw)
                    if aim_exit_err is None or abs(aim_exit_err) > TARGET_CHASSIS_REALIGN_TRIGGER_DEG:
                        self.owner.align_heading_stationary(
                            chassis_hold_yaw,timeout_sec=TARGET_CHASSIS_REALIGN_TIMEOUT_SEC,
                            tolerance_deg=max(0.90,PRE_MOVE_ALIGN_TOL_DEG),settle_sec=0.035,
                        )
                except Exception:
                    pass

    @staticmethod
    def _sector_fire_cone(sector_name):
        """Return the V15 SOFT acquisition cone used to authorize SHOOT-INTENT."""
        name = str(sector_name or "").upper()
        for sector, _start, _end, _slide, lo, hi in TARGET_RADAR_SECTORS:
            if sector == name:
                return float(lo), float(hi)
        return None

    @staticmethod
    def _sector_preferred_fire_cone(sector_name):
        """Return the original narrow preferred acquisition cone for telemetry/UI."""
        name = str(sector_name or "").upper()
        cone = TARGET_PREFERRED_ACQUISITION_CONES.get(name)
        if cone is None:
            return None
        return float(cone[0]), float(cone[1])

    @staticmethod
    def _sector_good_fire_cone(sector_name):
        """Final physical-fire cone; search/track may remain much wider."""
        name = str(sector_name or "").upper()
        cone = TARGET_GOOD_FIRE_CONES.get(name)
        if cone is None:
            return None
        return float(cone[0]), float(cone[1])

    @staticmethod
    def _sector_aim_envelope(sector_name):
        name = str(sector_name or "").upper()
        if name == "FRONT":
            half = float(TARGET_FRONT_CROSSHAIR_AIM_HALF_DEG)
            return (
                max(-float(TARGET_AIM_YAW_LIMIT_DEG), -half),
                min(+float(TARGET_AIM_YAW_LIMIT_DEG), +half),
            )
        if name == "LEFT":
            return (
                max(-float(TARGET_AIM_YAW_LIMIT_DEG), -float(TARGET_SIDE_CROSSHAIR_AIM_OUTER_DEG)),
                -float(TARGET_SIDE_CROSSHAIR_AIM_INNER_DEG),
            )
        if name == "RIGHT":
            return (
                +float(TARGET_SIDE_CROSSHAIR_AIM_INNER_DEG),
                min(+float(TARGET_AIM_YAW_LIMIT_DEG), +float(TARGET_SIDE_CROSSHAIR_AIM_OUTER_DEG)),
            )
        cone = TargetVisionSubsystem._sector_fire_cone(name)
        if cone is None:
            return None
        axis = 0.5 * (float(cone[0]) + float(cone[1]))
        half = float(TARGET_CROSSHAIR_AIM_MAX_FROM_AXIS_DEG)
        return (
            max(-float(TARGET_AIM_YAW_LIMIT_DEG), axis - half),
            min(+float(TARGET_AIM_YAW_LIMIT_DEG), axis + half),
        )

    @staticmethod
    def _yaw_in_cone(yaw_deg, cone):
        if yaw_deg is None or cone is None:
            return False
        lo, hi = cone
        y = float(yaw_deg)
        return float(lo) <= y <= float(hi)

    def _temporary_side_shift(self, direction, anchor_pos=None, preposition_yaw=None):
        """Move the chassis laterally ~35 cm to create the side-camera viewpoint.

        V20.7 fixes the old semantic mistake: 35 cm is the *chassis displacement*
        from the logical node anchor, not a 350 mm wall-ToF target.  Therefore a
        side shift is attempted even when the side is completely OPEN.

        Speed depends only on remaining odometric shift.  Fresh destination Sharp
        is sampled at ~50 Hz and can only emergency-STOP the excursion.
        """
        token = {
            "direction": direction,
            "anchor_pos": anchor_pos,
            "target_yaw": self.owner.desired_yaw_for_heading(self.owner.heading),
            "shift_m": 0.0,
            "result": "NO_SHIFT",
            "dest_sharp_cm": None,
        }
        if not TARGET_SIDE_SHIFT_ENABLED or direction not in ("LEFT", "RIGHT"):
            return token

        self.owner.safe_stop()
        if token["target_yaw"] is None:
            token["result"] = "NO_YAW_REF"
            return token
        if anchor_pos is None:
            anchor_pos = self.owner.current_position()
            token["anchor_pos"] = anchor_pos
        if anchor_pos is None:
            token["result"] = "NO_POSITION"
            return token

        # Only pay for a stationary realign if chassis yaw is visibly off.
        shift_yaw_err = self.owner.yaw_error_deg(token["target_yaw"])
        if shift_yaw_err is None or abs(shift_yaw_err) > TARGET_CHASSIS_REALIGN_TRIGGER_DEG:
            self.owner.align_heading_stationary(
                token["target_yaw"], timeout_sec=0.45,
                tolerance_deg=max(0.90, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
            )
        self.owner.pid_straight.reset()

        sign = +1.0 if direction == "RIGHT" else -1.0
        dest_side = "RIGHT" if sign > 0 else "LEFT"

        # Hide gimbal reposition time underneath the 35 cm chassis slide.  The
        # later sector sweep still verifies feedback before it starts, so failure
        # here only loses the overlap optimization, never target coverage.
        if preposition_yaw is not None:
            try:
                self.owner.gimbal.moveto(
                    pitch=float(TARGET_SEARCH_PITCH_DEG), yaw=float(preposition_yaw),
                    pitch_speed=float(GIMBAL_PITCH_SPEED), yaw_speed=float(GIMBAL_YAW_SPEED),
                )
            except Exception:
                pass

        start_t = time.monotonic()
        last_log = 0.0
        missing_sharp_cycles = 0
        stop_reason = "FULL_35CM"

        while self.owner.running and self.running:
            now = time.monotonic()
            pos = self.owner.current_position()
            if pos is None:
                stop_reason = "POSITION_STALE"
                break

            lateral = self.owner.cell_lateral_offset(anchor_pos, pos, token["target_yaw"])
            shifted = max(0.0, sign * float(lateral))
            token["shift_m"] = shifted
            remaining = max(0.0, TARGET_SIDE_SHIFT_DISTANCE_M - shifted)

            if shifted >= TARGET_SIDE_SHIFT_DISTANCE_M - TARGET_SIDE_SHIFT_STOP_TOL_M:
                stop_reason = "FULL_35CM"
                break
            if shifted >= TARGET_SIDE_SHIFT_MAX_M:
                stop_reason = "ODOM_MAX_GUARD"
                break
            if now - start_t >= TARGET_SIDE_SHIFT_TIMEOUT_SEC:
                stop_reason = "SHIFT_TIMEOUT"
                break

            # Fresh, unfiltered destination Sharp.  FAR>CAL means safely beyond
            # the calibrated close-range sensor span, not a missing sensor.
            left_cm, right_cm, la_raw, ra_raw = self.owner.read_sharp_cm_fast(
                samples=TARGET_SIDE_SHIFT_FAST_SHARP_SAMPLES,
                interval_sec=TARGET_SIDE_SHIFT_FAST_SHARP_INTERVAL_SEC,
            )
            dest_cm = right_cm if sign > 0 else left_cm
            dest_raw = ra_raw if sign > 0 else la_raw
            dest_cal = RIGHT_CAL if sign > 0 else LEFT_CAL
            dest_far = (dest_cm is None and sharp_raw_means_far(dest_raw, dest_cal))
            token["dest_sharp_cm"] = dest_cm

            if dest_far:
                missing_sharp_cycles = 0
            elif dest_cm is None:
                missing_sharp_cycles += 1
                if missing_sharp_cycles >= TARGET_SIDE_SHIFT_SHARP_MISSING_MAX_CYCLES:
                    stop_reason = f"{dest_side}_SHARP_UNAVAILABLE"
                    self.owner.safe_stop()
                    break
            else:
                missing_sharp_cycles = 0
                if dest_cm <= TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM:
                    stop_reason = f"{dest_side}_SHARP_EMERGENCY_STOP"
                    self.owner.safe_stop()
                    break

            # Exact user-requested four-stage profile, based on *remaining chassis
            # displacement* rather than wall-ToF distance.
            if remaining <= TARGET_SIDE_SHIFT_CRAWL_REMAIN_M:
                speed = TARGET_SIDE_SHIFT_CRAWL_MPS
            elif remaining <= TARGET_SIDE_SHIFT_SLOW_REMAIN_M:
                speed = TARGET_SIDE_SHIFT_SLOW_MPS
            elif remaining <= TARGET_SIDE_SHIFT_MED_REMAIN_M:
                speed = TARGET_SIDE_SHIFT_MED_SPEED_MPS
            else:
                speed = TARGET_SIDE_SHIFT_SPEED_MPS

            z_cmd = self.owner.lateral_yaw_hold_command(token["target_yaw"])
            if not self.owner.drive_speed_resilient(
                x=0.0, y=sign * speed, z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT, label="TARGET FULL35 SHIFT",
            ):
                stop_reason = "DRIVE_COMMAND_FAILED"
                break

            if now - last_log >= 0.18:
                sharp_text = (
                    "FAR>CAL" if dest_far else
                    ("NA" if dest_cm is None else f"{dest_cm:.1f}cm")
                )
                print(
                    f"[TARGET FULL35 {dest_side}] d={shifted:.3f}m "
                    f"remain={remaining*1000:.0f}mm {dest_side}SharpFAST={sharp_text} "
                    f"v={speed:.3f}m/s"
                )
                last_log = now
            time.sleep(TARGET_SIDE_SHIFT_CONTROL_DT)

        self.owner.safe_stop()
        # No extra ToF sample/settle here: the following sweep already validates
        # the gimbal pose and the camera thread is continuously live.
        token["result"] = stop_reason
        print(
            f"[TARGET FULL35 DONE] dir={direction} shift={token['shift_m']:.3f}m "
            f"destSharp={token.get('dest_sharp_cm')} result={stop_reason}"
        )
        return token

    def _is_confirmed_dead_end_cell(self, cell):
        """True only when stationary topology confirmed L/F/R are all walls.

        The BACK edge is the already-traversed way out of the cell.  UNKNOWN is
        deliberately not accepted as a wall: a failed ToF ray must never trigger
        a 35 cm target-viewpoint excursion.
        """
        cell = tuple(cell)
        # A temporary reverse is allowed only when BACK is an already-traversed
        # open edge.  This deliberately excludes the root/staging cell and any
        # boxed/uncertain pose where reversing 35 cm has not been proven safe.
        parent = self.owner.parent.get(cell)
        if parent is None:
            return False
        back_dir = direction_between(cell, parent)
        if back_dir is None or self.owner.edge_state.get((cell, back_dir)) != "OPEN":
            return False

        scan = self.owner.cell_scan_mm.get(cell, {})
        for label in ("LEFT", "FRONT", "RIGHT"):
            mm = scan.get(label)
            if mm is None:
                return False
            try:
                mm = float(mm)
            except Exception:
                return False
            if (not math.isfinite(mm)) or tof_is_open_from_center(mm):
                return False
        return True

    def _temporary_deadend_front_backshift(self, cell, anchor_pos=None):
        """Back ~35 cm only for the FRONT target scan of a confirmed dead-end.

        This is a target-viewpoint excursion, not DFS motion.  Odometry remains
        referenced to the logical node anchor and the already-traversed BACK path
        is used.  The robot is returned to the same anchor after the FRONT sector.
        """
        token = {
            "kind": "DEADEND_FRONT_BACKSHIFT",
            "cell": tuple(cell),
            "anchor_pos": anchor_pos,
            "target_yaw": self.owner.desired_yaw_for_heading(self.owner.heading),
            "shift_m": 0.0,
            "result": "NO_SHIFT",
            "anchor_front_tof_mm": self.owner.cell_scan_mm.get(tuple(cell), {}).get("FRONT"),
        }
        if not TARGET_DEADEND_FRONT_BACKSHIFT_ENABLED:
            token["result"] = "DISABLED"
            return token
        if not self._is_confirmed_dead_end_cell(cell):
            token["result"] = "NOT_CONFIRMED_DEADEND"
            return token

        self.owner.safe_stop()
        target_yaw = token.get("target_yaw")
        if target_yaw is None:
            token["result"] = "NO_YAW_REF"
            return token
        if anchor_pos is None:
            anchor_pos = self.owner.current_position()
            token["anchor_pos"] = anchor_pos
        if anchor_pos is None:
            token["result"] = "NO_POSITION"
            return token

        self.owner.align_heading_stationary(
            target_yaw, timeout_sec=0.9,
            tolerance_deg=max(0.55, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.08,
        )
        self.owner.pid_straight.reset()

        start_t = time.monotonic()
        last_log = 0.0
        stop_reason = "TARGET_DISTANCE"

        while self.owner.running and self.running:
            now = time.monotonic()
            pos = self.owner.current_position()
            if pos is None:
                stop_reason = "POSITION_STALE"
                break

            fwd = float(self.owner.cell_forward_progress(anchor_pos, pos, target_yaw))
            lat = float(self.owner.cell_lateral_offset(anchor_pos, pos, target_yaw))
            backed = max(0.0, -fwd)
            token["shift_m"] = backed

            if backed >= TARGET_DEADEND_FRONT_BACKSHIFT_M:
                stop_reason = "BACKSHIFT_35CM"
                break
            if now - start_t >= TARGET_DEADEND_FRONT_BACKSHIFT_TIMEOUT_SEC:
                stop_reason = "TIMEOUT_BEFORE_35CM"
                break

            remaining = max(0.0, TARGET_DEADEND_FRONT_BACKSHIFT_M - backed)
            x_mag = TARGET_DEADEND_FRONT_BACKSHIFT_SPEED_MPS
            if remaining <= TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_ZONE_M:
                x_mag = TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_MPS

            # Keep the temporary reverse on the same centreline.  This uses only
            # small odometry lateral correction; IR is intentionally not involved.
            y_cmd = clamp(
                -TARGET_DEADEND_FRONT_LATERAL_KP * lat,
                -TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
                +TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
            )
            if abs(lat) <= 0.010:
                y_cmd = 0.0
            z_cmd = self.owner.yaw_hold_command(target_yaw, stationary=False)

            if not self.owner.drive_speed_resilient(
                x=-x_mag, y=y_cmd, z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT, label="TARGET DEADEND BACKSHIFT",
            ):
                stop_reason = "DRIVE_CMD_FAIL"
                break

            if now - last_log >= 0.30:
                print(
                    f"[TARGET DEADEND BACKSHIFT] cell={tuple(cell)} "
                    f"back={backed:.3f}/{TARGET_DEADEND_FRONT_BACKSHIFT_M:.3f}m "
                    f"lat={lat:+.3f}m x={-x_mag:+.3f} y={y_cmd:+.3f}"
                )
                last_log = now
            time.sleep(TARGET_SWEEP_CONTROL_DT)

        self.owner.safe_stop()
        time.sleep(0.06)
        token["result"] = stop_reason
        print(
            f"[TARGET DEADEND BACKSHIFT DONE] cell={tuple(cell)} "
            f"back={token['shift_m']:.3f}m result={stop_reason}"
        )
        return token

    def _return_from_deadend_front_backshift(self, token):
        """Drive forward from the temporary dead-end viewpoint to its anchor."""
        if not isinstance(token, dict):
            return True
        anchor = token.get("anchor_pos")
        target_yaw = token.get("target_yaw")
        if anchor is None or target_yaw is None:
            return True
        if float(token.get("shift_m", 0.0) or 0.0) <= 0.005:
            return True

        self.owner.safe_stop()
        self.owner.pid_straight.reset()

        # Return toward the dead-end wall with ToF facing FRONT.  Odometry is
        # authoritative for the anchor; ToF is only an overshoot/collision veto
        # referenced to the stationary front range measured at that anchor.
        self._goto_target_pose_strict(0.0, GIMBAL_PITCH_DEG, timeout_sec=1.0)
        anchor_front_tof = token.get("anchor_front_tof_mm")
        try:
            anchor_front_tof = float(anchor_front_tof) if anchor_front_tof is not None else None
        except Exception:
            anchor_front_tof = None

        deadline = time.monotonic() + TARGET_DEADEND_FRONT_RETURN_TIMEOUT_SEC
        last_log = 0.0

        while self.owner.running and self.running and time.monotonic() < deadline:
            pos = self.owner.current_position()
            if pos is None:
                break
            fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
            lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))

            if (
                abs(fwd) <= TARGET_DEADEND_FRONT_RETURN_FWD_TOL_M
                and abs(lat) <= TARGET_DEADEND_FRONT_RETURN_LAT_TOL_M
            ):
                self.owner.safe_stop()
                self.owner.align_heading_stationary(
                    target_yaw, timeout_sec=0.8,
                    tolerance_deg=max(0.55, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.07,
                )
                print(
                    f"[TARGET DEADEND ANCHOR RETURN] fwd={fwd:+.3f}m "
                    f"lat={lat:+.3f}m -> OK"
                )
                return True

            remaining = max(0.0, -fwd)
            x_cmd = TARGET_DEADEND_FRONT_RETURN_SPEED_MPS
            if remaining <= TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_ZONE_M:
                x_cmd = TARGET_DEADEND_FRONT_RETURN_SLOW_MPS

            # If odometry says we have already passed the anchor, stop rather
            # than oscillating toward the dead-end wall.
            if fwd > TARGET_DEADEND_FRONT_RETURN_FWD_TOL_M:
                break

            # Secondary wall guard.  Do not drive materially closer to the wall
            # than the original anchor even if odometry has drifted.
            front_now = self.owner.latest_tof(fresh=True)
            if (
                anchor_front_tof is not None
                and front_now is not None
                and front_now <= max(55.0, anchor_front_tof - 25.0)
            ):
                print(
                    f"[TARGET DEADEND ANCHOR RETURN GUARD] front={front_now:.0f}mm "
                    f"anchorFront={anchor_front_tof:.0f}mm -> STOP"
                )
                break

            y_cmd = clamp(
                -TARGET_DEADEND_FRONT_LATERAL_KP * lat,
                -TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
                +TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
            )
            if abs(lat) <= 0.010:
                y_cmd = 0.0
            z_cmd = self.owner.yaw_hold_command(target_yaw, stationary=False)

            if not self.owner.drive_speed_resilient(
                x=+x_cmd, y=y_cmd, z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT, label="TARGET DEADEND ANCHOR RETURN",
            ):
                break

            now = time.monotonic()
            if now - last_log >= 0.30:
                print(
                    f"[TARGET DEADEND ANCHOR RETURN] fwd={fwd:+.3f}m "
                    f"lat={lat:+.3f}m x={x_cmd:+.3f} y={y_cmd:+.3f}"
                )
                last_log = now
            time.sleep(TARGET_SWEEP_CONTROL_DT)

        self.owner.safe_stop()
        pos = self.owner.current_position(fresh=False)
        if pos is None:
            self.owner.fault(
                "TARGET DEADEND ANCHOR",
                "position unavailable after FRONT backshift return",
                "remain stopped / DFS will re-align",
            )
            return False
        fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
        lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))
        soft = (
            abs(fwd) <= TARGET_DEADEND_FRONT_RETURN_SOFT_TOL_M
            and abs(lat) <= TARGET_DEADEND_FRONT_RETURN_SOFT_TOL_M
        )
        self.owner.align_heading_stationary(
            target_yaw, timeout_sec=0.9,
            tolerance_deg=max(0.70, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.07,
        )
        if soft:
            self.owner.fault(
                "TARGET DEADEND ANCHOR",
                f"soft return fwd={fwd:+.3f}m lat={lat:+.3f}m",
                "accept anchor tolerance and continue",
            )
            return True
        self.owner.fault(
            "TARGET DEADEND ANCHOR",
            f"return residual fwd={fwd:+.3f}m lat={lat:+.3f}m",
            "remain stopped; DFS will re-align before translation",
        )
        return False

    def _return_to_sector_anchor(self, token):
        """Fast return over the just-proven side-shift path."""
        if not isinstance(token, dict):
            return True
        anchor = token.get("anchor_pos")
        target_yaw = token.get("target_yaw")
        if anchor is None or target_yaw is None:
            return True

        self.owner.safe_stop()
        self.owner.pid_straight.reset()
        deadline = time.monotonic() + TARGET_SIDE_SHIFT_RETURN_TIMEOUT_SEC
        last_log = 0.0
        missing = 0

        while self.owner.running and self.running and time.monotonic() < deadline:
            pos = self.owner.current_position()
            if pos is None:
                break
            lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))
            fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
            remaining = abs(lat)

            if remaining <= TARGET_SIDE_SHIFT_RETURN_LAT_TOL_M:
                self.owner.safe_stop()
                yaw_err = self.owner.yaw_error_deg(target_yaw)
                # Do not burn ~0.5-0.8 s settling a yaw that is already good; the
                # next sector/move has its own cardinal verification.
                if yaw_err is None or abs(yaw_err) > 1.35:
                    self.owner.align_heading_stationary(
                        target_yaw, timeout_sec=0.40,
                        tolerance_deg=max(0.85, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
                    )
                ok = abs(fwd) <= TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M
                print(
                    f"[TARGET ANCHOR FAST RETURN] lat={lat:+.3f}m fwd={fwd:+.3f}m "
                    f"-> {'OK' if ok else 'LATERAL_OK/FWD_DRIFT'}"
                )
                return True

            if remaining <= TARGET_SIDE_SHIFT_RETURN_SLOW_ZONE_M:
                speed = TARGET_SIDE_SHIFT_RETURN_SLOW_MPS
            elif remaining <= TARGET_SIDE_SHIFT_RETURN_MED_ZONE_M:
                speed = TARGET_SIDE_SHIFT_RETURN_MED_SPEED_MPS
            else:
                speed = TARGET_SIDE_SHIFT_RETURN_SPEED_MPS
            y_cmd = math.copysign(speed, -lat)

            left_cm, right_cm, la_raw, ra_raw = self.owner.read_sharp_cm_fast(
                samples=TARGET_SIDE_SHIFT_FAST_SHARP_SAMPLES,
                interval_sec=TARGET_SIDE_SHIFT_FAST_SHARP_INTERVAL_SEC,
            )
            dest_cm = right_cm if y_cmd > 0 else left_cm
            dest_raw = ra_raw if y_cmd > 0 else la_raw
            dest_cal = RIGHT_CAL if y_cmd > 0 else LEFT_CAL
            dest_far = (dest_cm is None and sharp_raw_means_far(dest_raw, dest_cal))
            if dest_far:
                missing = 0
            elif dest_cm is None:
                missing += 1
                # Returning to anchor is required; tolerate one missing callback,
                # then reduce rather than blindly keeping 0.38 m/s.
                if missing >= 2:
                    y_cmd = math.copysign(min(abs(y_cmd), 0.10), y_cmd)
            else:
                missing = 0
                if dest_cm <= TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM:
                    self.owner.safe_stop()
                    print(
                        f"[TARGET ANCHOR RETURN BLOCK] y={y_cmd:+.3f} "
                        f"destSharpFAST={dest_cm:.1f}cm"
                    )
                    break

            z_cmd = self.owner.lateral_yaw_hold_command(target_yaw)
            if not self.owner.drive_speed_resilient(
                x=0.0, y=y_cmd, z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT, label="TARGET FAST ANCHOR RETURN",
            ):
                break

            now = time.monotonic()
            if now - last_log >= 0.20:
                print(
                    f"[TARGET ANCHOR FAST RETURN] lat={lat:+.3f}m "
                    f"remain={remaining:.3f}m v={abs(y_cmd):.2f}m/s"
                )
                last_log = now
            time.sleep(TARGET_SIDE_SHIFT_CONTROL_DT)

        self.owner.safe_stop()
        pos = self.owner.current_position(fresh=False)
        if pos is None:
            self.owner.fault("TARGET ANCHOR", "return ended without position", "continue cautiously")
            return False
        lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))
        fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
        soft = (
            abs(lat) <= TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M
            and abs(fwd) <= TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M
        )
        yaw_err = self.owner.yaw_error_deg(target_yaw)
        if yaw_err is None or abs(yaw_err) > 1.35:
            self.owner.align_heading_stationary(
                target_yaw, timeout_sec=0.40,
                tolerance_deg=max(0.85, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
            )
        if soft:
            self.owner.fault(
                "TARGET ANCHOR",
                f"fast return soft tolerance lat={lat:+.3f}m fwd={fwd:+.3f}m",
                "accept node and continue",
            )
            return True
        self.owner.fault(
            "TARGET ANCHOR",
            f"return residual lat={lat:+.3f}m fwd={fwd:+.3f}m",
            "remain stopped; DFS will re-align before translation",
        )
        return False

    def _candidate_probably_already_fired(self, candidate, detect_yaw):
        """Cheap pre-AIM check for a previously fired physical target.

        Prefer estimated world position using the currently-fresh gimbal ToF.
        Fall back to same-cell absolute bearing only.  Shape is intentionally
        ignored because oblique views can change SQUARE <-> RECT classification.
        """
        color = str(candidate.get("color") or "").upper()
        if not color:
            return None
        current_cell = tuple(self.owner.current)
        abs_bearing = wrap_deg(float(self.owner.heading) * 90.0 + float(detect_yaw))

        # Use the latest ToF without waiting for another multi-sample range read.
        est_xy = None
        tof_now = self.owner.latest_tof(fresh=True)
        if tof_now is not None:
            try:
                est_xy, _ = self._estimate_position(float(tof_now))
            except Exception:
                est_xy = None

        for old in self.targets:
            if str(old.get("color") or "").upper() != color:
                continue
            cand_shape = str(candidate.get("shape") or "").upper()
            old_shape = str(old.get("shape") or "").upper()
            rect_family = {"SQUARE", "RECT_HORIZONTAL", "RECT_VERTICAL"}
            if cand_shape != old_shape and not (cand_shape in rect_family and old_shape in rect_family):
                continue
            if not (
                str(old.get("fire_status") or "").startswith("FIRED_")
                or str(old.get("id") or "") in self.fired_target_ids
            ):
                continue

            old_xy = old.get("estimated_grid_xy")
            if est_xy is not None and old_xy is not None:
                try:
                    if math.hypot(
                        float(est_xy[0]) - float(old_xy[0]),
                        float(est_xy[1]) - float(old_xy[1]),
                    ) <= TARGET_FAST_FIRED_GRID_DIST:
                        return old
                except Exception:
                    pass

            old_cell = tuple(old.get("source_cell", ()))
            old_b = old.get("detected_bearing_deg_from_north")
            if old_b is None:
                old_b = old.get("bearing_deg_from_north")
            if old_cell == current_cell and old_b is not None:
                if angle_diff_deg(float(old_b), abs_bearing) <= TARGET_FAST_FIRED_SAME_CELL_BEARING_DEG:
                    return old
        return None

    def _candidate_suppressed_in_sector(self, candidate, yaw_now, handled):
        for item in handled:
            if isinstance(item, dict):
                color=item.get("color"); shape=item.get("shape")
                sweep_yaw=item.get("sweep_yaw"); radius=float(item.get("radius",TARGET_SWEEP_REPEAT_SUPPRESS_DEG))
            else:
                try:
                    color,shape,sweep_yaw=item[:3]; radius=TARGET_SWEEP_REPEAT_SUPPRESS_DEG
                except Exception:
                    continue
            if color != candidate.get("color"):
                continue
            if isinstance(item, dict) and item.get("ignore_shape"):
                pass
            elif shape != candidate.get("shape"):
                continue
            if sweep_yaw is not None and yaw_now is not None:
                if angle_diff_deg(float(sweep_yaw), float(yaw_now)) <= float(radius):
                    return True
        return False

    def _continuous_sector_sweep(self, sector_name, start_yaw, end_yaw, fire_lo, fire_hi):
        """Sweep one sector continuously, interrupting only for a stable target."""
        found = []
        handled = []
        interrupts = 0
        direction = +1.0 if end_yaw >= start_yaw else -1.0

        if not self._goto_target_pose_strict(start_yaw, TARGET_SEARCH_PITCH_DEG, timeout_sec=0.80, allow_soft=True):
            self.owner.fault(
                "TARGET SWEEP",
                f"{sector_name}: cannot reach start yaw={start_yaw:+.1f}",
                "skip sector",
            )
            return found

        time.sleep(TARGET_SCAN_SETTLE_SEC)
        gate_since = time.monotonic()
        deadline = time.monotonic() + TARGET_SWEEP_TIMEOUT_SEC
        last_debug = 0.0
        preferred_cone = self._sector_preferred_fire_cone(sector_name)
        pref_text = "NA" if preferred_cone is None else "{:+.0f}..{:+.0f}".format(*preferred_cone)
        print(
            f"[TARGET SWEEP {sector_name}] {start_yaw:+.0f}->{end_yaw:+.0f}deg "
            f"pitch={TARGET_SEARCH_PITCH_DEG:+.0f} preferred={pref_text} "
            f"softAcq={fire_lo:+.0f}..{fire_hi:+.0f} "
            f"aimEnvelope={self._sector_aim_envelope(sector_name)}"
        )

        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                p_now, y_now = self.owner.current_gimbal_relative()
                if p_now is None or y_now is None:
                    self._stop_gimbal_velocity()
                    time.sleep(0.03)
                    continue

                remaining = direction * (float(end_yaw) - float(y_now))
                if remaining <= 1.5:
                    break

                candidate = self._quick_gate_candidate_since(
                    gate_since, min_frames=TARGET_SEARCH_QUICK_GATE_FRAMES
                )
                if candidate is not None:
                    if self._candidate_suppressed_in_sector(candidate, y_now, handled):
                        gate_since = time.monotonic()
                    else:
                        candidate = dict(candidate)
                        detect_yaw = float(y_now)
                        old_fired = self._candidate_probably_already_fired(candidate, detect_yaw)
                        if old_fired is not None:
                            print(
                                f"[TARGET FAST SKIP FIRED] {old_fired.get('id')} "
                                f"{candidate.get('color')} detectYaw={detect_yaw:+.1f} "
                                f"shapeNow={candidate.get('shape')} -> keep sweeping"
                            )
                            handled.append({
                                "color": candidate.get("color"),
                                "shape": candidate.get("shape"),
                                "sweep_yaw": detect_yaw,
                                "radius": max(32.0, TARGET_SWEEP_REPEAT_SUPPRESS_DEG),
                                "ignore_shape": True,
                            })
                            gate_since = time.monotonic()
                            time.sleep(TARGET_SWEEP_CONTROL_DT)
                            continue

                        self._stop_gimbal_velocity()
                        service_started = time.monotonic()
                        acquisition_cone = (float(fire_lo), float(fire_hi))
                        preferred_cone = self._sector_preferred_fire_cone(sector_name)
                        aim_envelope = self._sector_aim_envelope(sector_name)
                        shoot_intent = self._yaw_in_cone(detect_yaw, acquisition_cone)
                        in_preferred = self._yaw_in_cone(detect_yaw, preferred_cone)
                        candidate["detected_sweep_yaw_deg"] = detect_yaw
                        candidate["detected_sector"] = sector_name
                        candidate["shoot_intent"] = bool(shoot_intent)
                        candidate["preferred_acquisition_cone_deg"] = (
                            None if preferred_cone is None else [float(preferred_cone[0]), float(preferred_cone[1])]
                        )
                        candidate["detected_in_preferred_cone"] = bool(in_preferred)
                        candidate["acquisition_cone_deg"] = [float(fire_lo), float(fire_hi)]
                        candidate["aim_envelope_deg"] = (
                            None if aim_envelope is None
                            else [float(aim_envelope[0]), float(aim_envelope[1])]
                        )

                        locked = None
                        fire_record = None
                        fire_success = False
                        if not shoot_intent:
                            # Outside the narrow +/-10deg acquisition cone: keep
                            # the target in memory, but do not spend time dragging
                            # the crosshair toward it and never authorize fire.
                            print(
                                f"[TARGET MEMORY {sector_name}] yaw={detect_yaw:+.1f} "
                                f"{candidate.get('color')} {candidate.get('shape')} "
                                f"outside softAcq={fire_lo:+.0f}..{fire_hi:+.0f} -> remember only"
                            )
                            tof_mm = self.owner.sample_fresh_tof(samples=TARGET_FIRE_RANGE_SAMPLES)
                            record = self._record_target(
                                candidate,
                                tof_mm,
                                sector_name=sector_name,
                                detection_yaw_deg=detect_yaw,
                                fire_cone=acquisition_cone,
                                shoot_intent=False,
                                aim_envelope=aim_envelope,
                            )
                            if record is not None and not any(
                                r.get("id") == record.get("id")
                                for r in found if isinstance(r, dict)
                            ):
                                found.append(record)
                        else:
                            acq_kind = "PREFERRED" if in_preferred else "SOFT"
                            print(
                                f"[TARGET HIT {sector_name}] yaw={detect_yaw:+.1f} "
                                f"{candidate.get('color')} {candidate.get('shape')} "
                                f"inside {acq_kind} acquisition -> SHOOT-INTENT LOCK"
                            )
                            aim_lo = None if aim_envelope is None else aim_envelope[0]
                            aim_hi = None if aim_envelope is None else aim_envelope[1]
                            locked = self._aim_and_lock(
                                candidate, aim_yaw_min=aim_lo, aim_yaw_max=aim_hi
                            )
                            if locked is not None:
                                tof_mm = self.owner.sample_fresh_tof(samples=TARGET_FIRE_RANGE_SAMPLES)
                                record = self._record_target(
                                    locked,
                                    tof_mm,
                                    sector_name=sector_name,
                                    detection_yaw_deg=detect_yaw,
                                    fire_cone=acquisition_cone,
                                    shoot_intent=True,
                                    aim_envelope=aim_envelope,
                                )
                                if record is not None:
                                    if not any(r.get("id") == record.get("id") for r in found if isinstance(r, dict)):
                                        found.append(record)
                                    fire_record = record
                                    fire_success = bool(self._maybe_fire(
                                        record,
                                        sector_name=sector_name,
                                        fire_cone=acquisition_cone,
                                    ))

                        interrupts += 1
                        # Suppress around the ORIGINAL sweep bearing.  Auto-Aim
                        # may rotate tens of degrees, so its final lock yaw must
                        # never become the sweep-progress coordinate.
                        fire_status = (
                            str(fire_record.get("fire_status") or "")
                            if isinstance(fire_record, dict) else ""
                        )
                        if fire_success or fire_status.startswith("FIRED_"):
                            suppress_radius = TARGET_SWEEP_REPEAT_SUPPRESS_DEG
                        elif fire_status == "DEFER_BAD_ANGLE":
                            # Critical: do NOT burn the good-angle opportunity.
                            # Skip only the immediate duplicate frames around the
                            # same bearing, then allow reacquisition farther along.
                            suppress_radius = TARGET_BAD_ANGLE_RETRY_SUPPRESS_DEG
                        elif locked is not None:
                            suppress_radius = TARGET_SWEEP_FAIL_SUPPRESS_DEG
                        elif not shoot_intent:
                            suppress_radius = TARGET_SWEEP_REPEAT_SUPPRESS_DEG
                        else:
                            suppress_radius = TARGET_SWEEP_FAIL_SUPPRESS_DEG

                        handled.append({
                            "color": candidate.get("color"),
                            "shape": candidate.get("shape"),
                            "sweep_yaw": detect_yaw,
                            "radius": suppress_radius,
                            # Once it has been locked, perspective may flip
                            # SQUARE <-> RECT while the sweep continues.
                            "ignore_shape": bool(fire_status == "DEFER_BAD_ANGLE"),
                        })

                        if interrupts >= TARGET_SWEEP_MAX_INTERRUPTS_PER_SECTOR:
                            print(f"[TARGET SWEEP {sector_name}] interrupt cap reached -> continue mission")
                            break

                        # Resume from the DETECTION bearing, not the post-Aim
                        # yaw.  This guarantees monotonic sector progress.
                        resume_yaw = detect_yaw + direction * TARGET_SWEEP_RESUME_ADVANCE_DEG
                        resume_yaw = clamp(resume_yaw, min(start_yaw, end_yaw), max(start_yaw, end_yaw))
                        if direction * (float(end_yaw) - resume_yaw) <= 1.5:
                            break
                        self._goto_target_pose_strict(
                            resume_yaw, TARGET_SEARCH_PITCH_DEG, timeout_sec=0.55, allow_soft=True
                        )
                        # Target lock / verification time must not consume the
                        # sector's continuous-sweep watchdog budget.
                        deadline += max(0.0, time.monotonic() - service_started)
                        gate_since = time.monotonic()
                        continue

                # Continuous velocity sweep + closed-loop pitch hold at -10 deg.
                remaining_abs = abs(float(end_yaw) - float(y_now))
                yaw_speed = TARGET_SWEEP_SPEED_DPS
                if remaining_abs < TARGET_SWEEP_FINE_ZONE_DEG:
                    yaw_speed = max(
                        TARGET_SWEEP_MIN_DPS,
                        min(TARGET_SWEEP_SPEED_DPS, 2.5 * remaining_abs),
                    )
                yaw_speed *= direction

                pitch_err = float(TARGET_SEARCH_PITCH_DEG) - float(p_now)
                pitch_speed = clamp(2.4 * pitch_err, -20.0, +20.0)
                if abs(pitch_err) <= 0.55:
                    pitch_speed = 0.0
                elif 0.0 < abs(pitch_speed) < 2.5:
                    pitch_speed = math.copysign(2.5, pitch_speed)

                self.owner.gimbal.drive_speed(
                    pitch_speed=float(pitch_speed), yaw_speed=float(yaw_speed)
                )

                now = time.monotonic()
                if now - last_debug >= 0.50:
                    print(
                        f"[TARGET SWEEP {sector_name}] yaw={y_now:+.1f} "
                        f"pitch={p_now:+.1f} speed={yaw_speed:+.1f}dps"
                    )
                    last_debug = now
                time.sleep(TARGET_SWEEP_CONTROL_DT)
        except Exception as exc:
            self.owner.fault(
                "TARGET SWEEP",
                f"{sector_name}: {type(exc).__name__}: {exc}",
                "stop gimbal / keep target memory / continue DFS",
            )
        finally:
            self._stop_gimbal_velocity()

        # V20.1: do NOT spend another strict moveto/retry at sector end.  The
        # next sector performs its own feedback-driven start positioning.  This
        # avoids the -53.6 -> -45 stall seen in the field log and removes a large
        # chunk of dead time per cell.
        p_end, y_end = self.owner.current_gimbal_relative()
        if y_end is not None and abs(wrap_deg(float(end_yaw) - float(y_end))) > TARGET_SEARCH_POSE_SOFT_YAW_TOL_DEG:
            self.owner.fault(
                "TARGET SWEEP END",
                f"{sector_name}: ended yaw={float(y_end):+.1f}, expected {float(end_yaw):+.1f}",
                "next sector will reacquire from feedback",
            )
        return found

    def scan_cell(self, cell, force=False):
        if not self.available or not self.owner.running or not self.owner.pose_trusted:
            return []
        cell = tuple(cell)
        if not force and cell in self.scanned_cells:
            return []

        self.owner.safe_stop()
        target_scan_started = time.monotonic()
        self.last_scan_t[cell] = target_scan_started
        found = []
        cell_anchor = self.owner.current_position()
        heading_yaw = self.owner.desired_yaw_for_heading(self.owner.heading)
        if heading_yaw is not None:
            initial_err = self.owner.yaw_error_deg(heading_yaw)
            if initial_err is None or abs(initial_err) > PRE_MOVE_ALIGN_TOL_DEG:
                self.owner.align_heading_stationary(
                    heading_yaw, timeout_sec=0.65,
                    tolerance_deg=max(0.70, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.05,
                )

        dead_end_front_shift = self._is_confirmed_dead_end_cell(cell)
        dead_end_note = " DEADEND-FRONT-BACKSHIFT=35cm" if dead_end_front_shift else ""
        print(
            f"\n[TARGET 3-SECTOR FAST35] cell={cell} "
            f"pitch={TARGET_SEARCH_PITCH_DEG:+.0f}deg "
            f"LEFT(-110..-45) FRONT(-45..+45) RIGHT(+45..+110)"
            f"{dead_end_note}"
        )

        # V20.7: no redundant FRONT pose before LEFT.  Side shift overlaps the
        # gimbal move toward that sector's start angle, saving one mechanical
        # positioning wait per cell.

        try:
            for sector_name, start_yaw, end_yaw, slide_dir, fire_lo, fire_hi in TARGET_RADAR_SECTORS:
                if not self.owner.running or not self.owner.pose_trusted:
                    break

                # Always start a sector with the chassis cardinal corrected.
                if heading_yaw is not None:
                    sector_err = self.owner.yaw_error_deg(heading_yaw)
                    if sector_err is None or abs(sector_err) > PRE_MOVE_ALIGN_TOL_DEG:
                        self.owner.align_heading_stationary(
                            heading_yaw, timeout_sec=0.60,
                            tolerance_deg=max(0.75, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.045,
                        )

                token = {
                    "direction": None,
                    "anchor_pos": cell_anchor,
                    "target_yaw": heading_yaw,
                    "shift_m": 0.0,
                    "result": "ANCHOR",
                }
                front_backshift_token = None
                if slide_dir is not None:
                    token = self._temporary_side_shift(slide_dir, anchor_pos=cell_anchor, preposition_yaw=start_yaw)
                elif sector_name == "FRONT" and dead_end_front_shift:
                    front_backshift_token = self._temporary_deadend_front_backshift(
                        cell, anchor_pos=cell_anchor
                    )

                try:
                    sector_found = self._continuous_sector_sweep(
                        sector_name, start_yaw, end_yaw, fire_lo, fire_hi
                    )
                    for rec in sector_found:
                        if not any(r.get("id") == rec.get("id") for r in found if isinstance(r, dict)):
                            found.append(rec)
                finally:
                    # LEFT/RIGHT and dead-end FRONT are viewpoint excursions
                    # only.  Always restore the logical node anchor before the
                    # next sector and before any DFS translation.
                    if slide_dir is not None:
                        self._return_to_sector_anchor(token)
                    elif front_backshift_token is not None:
                        self._return_from_deadend_front_backshift(front_backshift_token)

                # Re-check chassis yaw after large gimbal movement.  This
                # counters the small base reaction observed when the turret sweeps.
                if heading_yaw is not None:
                    yaw_before = self.owner.yaw_error_deg(heading_yaw)
                    # V20: turret reaction is usually tiny.  Only stop for a chassis
                    # re-align when it actually exceeds the competition tolerance.
                    if yaw_before is None or abs(yaw_before) > TARGET_CHASSIS_REALIGN_TRIGGER_DEG:
                        self.owner.align_heading_stationary(
                            heading_yaw, timeout_sec=TARGET_CHASSIS_REALIGN_TIMEOUT_SEC,
                            tolerance_deg=max(0.90, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
                        )
                    yaw_after = self.owner.yaw_error_deg(heading_yaw)
                    print(
                        f"[TARGET CHASSIS YAW] after {sector_name}: "
                        f"err before={fmt_deg(yaw_before)} after={fmt_deg(yaw_after)}"
                    )

            self.scanned_cells.add(cell)
            target_scan_sec = time.monotonic() - target_scan_started
            self.status = f"3-SECTOR 9MIN cell={cell} found={len(found)} t={target_scan_sec:.1f}s"
            print(f"[PERF TARGET SCAN] cell={cell} sec={target_scan_sec:.2f} found={len(found)}")
            self.save_targets()
            return found
        except Exception as exc:
            self.owner.fault(
                "TARGET SERVICE",
                f"cell={cell}: {type(exc).__name__}: {exc}",
                "stop target service / continue DFS",
            )
            return found
        finally:
            self._stop_gimbal_velocity()
            self._clear_focus_roi()
            try:
                if not self.owner.gimbal_front_safe_for_motion():
                    self.owner.recover_gimbal_front()
            except Exception:
                pass

    def _round2_matching_fired(self, records, hint):
        """Return True when this scan fired the same color/shape expected by a hint."""
        hc = str(hint.get("color") or "").upper()
        hs = str(hint.get("shape") or "").upper()
        for rec in records or []:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("color") or "").upper() != hc:
                continue
            if str(rec.get("shape") or "").upper() != hs:
                continue
            if str(rec.get("fire_status") or "").startswith("FIRED_"):
                return True
        return False

    def scan_round2_hint(self, hint):
        """Replay one proven Round-1 firing direction from the current logical cell.

        Fast path: sweep only +/- ROUND2_NARROW_SWEEP_HALF_DEG around the saved
        detection bearing.  Robust fallback: scan the original full LEFT/FRONT/RIGHT
        sector once if the narrow replay does not reacquire/fire the expected class.
        """
        if not self.available or not self.owner.running or not self.owner.pose_trusted:
            return False
        if not isinstance(hint, dict):
            return False

        sector = str(hint.get("sector") or hint.get("scan_sector") or "FRONT").upper()
        cfg = None
        for row in TARGET_RADAR_SECTORS:
            if str(row[0]).upper() == sector:
                cfg = row
                break
        if cfg is None:
            self.owner.fault("ROUND2 HINT", f"unknown sector={sector}", "skip hint")
            return False

        sector_name, full_start, full_end, slide_dir, fire_lo, fire_hi = cfg
        cell = tuple(self.owner.current)
        self.owner.safe_stop()
        cell_anchor = self.owner.current_position()
        heading_yaw = self.owner.desired_yaw_for_heading(self.owner.heading)
        if heading_yaw is not None:
            self.owner.align_heading_stationary(
                heading_yaw, timeout_sec=0.8,
                tolerance_deg=max(0.60, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.06,
            )

        saved_yaw = hint.get("detected_sweep_yaw_deg")
        if saved_yaw is None:
            saved_yaw = hint.get("lock_yaw_deg")
        if saved_yaw is None:
            saved_yaw = {"LEFT": -90.0, "FRONT": 0.0, "RIGHT": 90.0}.get(sector_name, 0.0)
        try:
            saved_yaw = float(saved_yaw)
        except Exception:
            saved_yaw = 0.0
        lo_bound, hi_bound = min(float(full_start), float(full_end)), max(float(full_start), float(full_end))
        narrow_lo = clamp(saved_yaw - ROUND2_NARROW_SWEEP_HALF_DEG, lo_bound, hi_bound)
        narrow_hi = clamp(saved_yaw + ROUND2_NARROW_SWEEP_HALF_DEG, lo_bound, hi_bound)
        if float(full_end) >= float(full_start):
            narrow_start, narrow_end = narrow_lo, narrow_hi
        else:
            narrow_start, narrow_end = narrow_hi, narrow_lo

        print(
            "\n[ROUND2 HINT] cell={} {} {} sector={} savedYaw={:+.1f} narrow={:+.1f}..{:+.1f}".format(
                cell, hint.get("color"), hint.get("shape"), sector_name,
                saved_yaw, narrow_start, narrow_end,
            )
        )

        self._goto_target_pose_strict(0.0, TARGET_SEARCH_PITCH_DEG, timeout_sec=0.9)
        token = {
            "direction": None, "anchor_pos": cell_anchor, "target_yaw": heading_yaw,
            "shift_m": 0.0, "result": "ANCHOR",
        }
        front_backshift_token = None
        all_found = []
        try:
            if slide_dir is not None:
                token = self._temporary_side_shift(slide_dir, anchor_pos=cell_anchor)
            elif sector_name == "FRONT" and self._is_confirmed_dead_end_cell(cell):
                front_backshift_token = self._temporary_deadend_front_backshift(
                    cell, anchor_pos=cell_anchor
                )

            found = self._continuous_sector_sweep(
                sector_name, narrow_start, narrow_end, fire_lo, fire_hi
            )
            all_found.extend(found or [])
            success = self._round2_matching_fired(all_found, hint)

            if (
                not success and ROUND2_FALLBACK_FULL_SECTOR
                and self.owner.running and self.owner.pose_trusted
            ):
                print(
                    "[ROUND2 FALLBACK] {} {} not fired in narrow window -> full {} sector".format(
                        hint.get("color"), hint.get("shape"), sector_name
                    )
                )
                found2 = self._continuous_sector_sweep(
                    sector_name, full_start, full_end, fire_lo, fire_hi
                )
                all_found.extend(found2 or [])
                success = self._round2_matching_fired(all_found, hint)
            return bool(success)
        except Exception as exc:
            self.owner.fault(
                "ROUND2 TARGET",
                "cell={} {} {}: {}: {}".format(
                    cell, hint.get("color"), hint.get("shape"), type(exc).__name__, exc
                ),
                "restore anchor / continue next hint",
            )
            return False
        finally:
            self._stop_gimbal_velocity()
            self._clear_focus_roi()
            try:
                if slide_dir is not None:
                    self._return_to_sector_anchor(token)
                elif front_backshift_token is not None:
                    self._return_from_deadend_front_backshift(front_backshift_token)
            except Exception as exc:
                self.owner.fault(
                    "ROUND2 ANCHOR", f"{type(exc).__name__}: {exc}",
                    "stop target replay / preserve logical pose",
                )
            try:
                if heading_yaw is not None:
                    self.owner.align_heading_stationary(
                        heading_yaw, timeout_sec=0.8,
                        tolerance_deg=max(0.60, PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.05,
                    )
                self.owner.gimbal_front_down(force=True)
            except Exception:
                pass

    # ---------------- target memory / fire ----------------
    def _estimate_position(self,tof_mm):
        p,y=self.owner.current_gimbal_relative()
        if tof_mm is None or y is None:
            return None,None
        try:
            pitch=float(p or 0.0)
            planar_mm = tof_center_planar_mm(tof_mm, pitch)
            if planar_mm is None:
                return None,None
            planar = planar_mm / 1000.0
        except Exception:
            return None,None
        bearing=wrap_deg(float(self.owner.heading)*90.0+float(y))
        rad=math.radians(bearing); dx=math.sin(rad); dy=math.cos(rad)
        grid_range=planar/max(GRID_TILE_M,1e-6)
        cell=tuple(self.owner.current)
        return [float(cell[0])+dx*grid_range,float(cell[1])+dy*grid_range],bearing

    @staticmethod
    def _same_identity(a,b):
        # V20.7: perspective can turn the same SQUARE into a horizontal/vertical
        # RECT, but a CIRCLE should never merge with that rectangular family.
        if a.get("color") != b.get("color"):
            return False
        sa = str(a.get("shape") or "").upper()
        sb = str(b.get("shape") or "").upper()
        if sa == sb:
            return True
        rect_family = {"SQUARE", "RECT_HORIZONTAL", "RECT_VERTICAL"}
        return sa in rect_family and sb in rect_family

    def _find_duplicate(self,record):
        for old in self.targets:
            if not self._same_identity(old,record): continue
            a=old.get("estimated_grid_xy"); b=record.get("estimated_grid_xy")
            if a is not None and b is not None:
                if math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))<=TARGET_DEDUPE_GRID_DIST:
                    return old
                continue
            if tuple(old.get("source_cell",()))!=tuple(record.get("source_cell",())):
                continue
            ba=old.get("bearing_deg_from_north"); bb=record.get("bearing_deg_from_north")
            if ba is not None and bb is not None and angle_diff_deg(ba,bb)<=TARGET_DEDUPE_BEARING_DEG:
                return old
        return None

    def _record_target(
        self, candidate, tof_mm, sector_name=None, detection_yaw_deg=None,
        fire_cone=None, shoot_intent=None, aim_envelope=None
    ):
        grid_xy, bearing = self._estimate_position(tof_mm)
        cell = tuple(self.owner.current)
        p, y = self.owner.current_gimbal_relative()
        cone = fire_cone if fire_cone is not None else self._sector_fire_cone(sector_name)
        preferred_cone = self._sector_preferred_fire_cone(sector_name)
        if aim_envelope is None:
            aim_envelope = self._sector_aim_envelope(sector_name)

        detect_yaw = y if detection_yaw_deg is None else float(detection_yaw_deg)
        detected_in_acq = self._yaw_in_cone(detect_yaw, cone)
        detected_in_preferred = self._yaw_in_cone(detect_yaw, preferred_cone)
        if shoot_intent is not None:
            detected_in_acq = bool(shoot_intent) and bool(detected_in_acq)
        in_aim_envelope = self._yaw_in_cone(y, aim_envelope)

        center = candidate.get("center_norm")
        crosshair_centered = bool(candidate.get("crosshair_centered", False))
        crosshair_error = candidate.get("crosshair_error_norm")
        if crosshair_error is None and center is not None:
            try:
                cx, cy = [float(v) for v in center]
                ex = cx - (0.5 + TARGET_AIM_OFFSET_X)
                ey = cy - (0.5 + TARGET_AIM_OFFSET_Y)
                crosshair_error = [float(ex), float(ey)]
                if detected_in_acq:
                    crosshair_centered = bool(
                        abs(ex) <= TARGET_AIM_CENTER_TOL_X
                        and abs(ey) <= TARGET_AIM_CENTER_TOL_Y
                    )
            except Exception:
                pass

        if detected_in_acq and in_aim_envelope and crosshair_centered:
            fire_status = "PENDING"
        elif not detected_in_acq:
            fire_status = "MEMORY_OUTSIDE_ACQ_CONE"
        elif not in_aim_envelope:
            fire_status = "MEMORY_OUTSIDE_AIM_ENVELOPE"
        else:
            fire_status = "MEMORY_CROSSHAIR_NOT_CENTERED"

        rec = {
            "id": None,
            "kind": "COLOR_SHAPE",
            "color": candidate.get("color"),
            "shape": candidate.get("shape"),
            "source_cell": [int(cell[0]), int(cell[1])],
            "heading": DIR_NAMES[int(self.owner.heading) % 4],
            "scan_sector": sector_name,
            "detected_sweep_yaw_deg": detect_yaw,
            "detected_bearing_deg_from_north": (
                None if detect_yaw is None
                else wrap_deg(float(self.owner.heading) * 90.0 + float(detect_yaw))
            ),
            "gimbal_yaw_deg": y,
            "gimbal_pitch_deg": p,
            "lock_yaw_deg": y,
            "bearing_deg_from_north": bearing,
            # Legacy field names are retained for compatibility, but in V5
            # fire_cone means the narrow ACQUISITION cone at detection time.
            "fire_cone_deg": None if cone is None else [float(cone[0]), float(cone[1])],
            "in_fire_cone": bool(detected_in_acq),
            "acquisition_cone_deg": None if cone is None else [float(cone[0]), float(cone[1])],
            "preferred_acquisition_cone_deg": (
                None if preferred_cone is None else [float(preferred_cone[0]), float(preferred_cone[1])]
            ),
            "detected_in_preferred_cone": bool(detected_in_preferred),
            "soft_acquisition": bool(detected_in_acq and not detected_in_preferred),
            "detected_in_acquisition_cone": bool(detected_in_acq),
            "shoot_intent": bool(detected_in_acq),
            "aim_envelope_deg": (
                None if aim_envelope is None
                else [float(aim_envelope[0]), float(aim_envelope[1])]
            ),
            "in_aim_envelope": bool(in_aim_envelope),
            "crosshair_centered": bool(crosshair_centered),
            "crosshair_error_norm": crosshair_error,
            "tof_mm": None if tof_mm is None else float(tof_mm),
            "estimated_grid_xy": grid_xy,
            "center_norm": center,
            "bbox_norm": candidate.get("bbox_norm"),
            "score": float(candidate.get("temporal_score", candidate.get("score", 0.0))),
            "color_confidence": float(candidate.get("color_confidence", 0.0)),
            "shape_confidence": float(candidate.get("shape_confidence", 0.0)),
            "median_hsv": candidate.get("median_hsv"),
            "median_lab": candidate.get("median_lab"),
            "confirm_frames": int(candidate.get("confirm_frames", 0)),
            "first_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "last_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "observations": 1,
            "fire_status": fire_status,
        }
        old = self._find_duplicate(rec)
        if old is not None:
            old["last_seen_at"] = rec["last_seen_at"]
            old["observations"] = int(old.get("observations", 1)) + 1
            for k in (
                "tof_mm", "estimated_grid_xy", "bearing_deg_from_north",
                "detected_bearing_deg_from_north",
                "gimbal_yaw_deg", "gimbal_pitch_deg", "lock_yaw_deg",
                "center_norm", "bbox_norm", "score", "color_confidence",
                "shape_confidence", "median_hsv", "median_lab",
                "scan_sector", "detected_sweep_yaw_deg", "fire_cone_deg",
                "in_fire_cone", "acquisition_cone_deg",
                "preferred_acquisition_cone_deg", "detected_in_preferred_cone",
                "soft_acquisition", "detected_in_acquisition_cone", "shoot_intent",
                "aim_envelope_deg", "in_aim_envelope",
                "good_fire_cone_deg", "good_fire_angle",
                "deferred_lock_yaw_deg", "deferred_fire_pose_yaw_deg",
                "crosshair_centered", "crosshair_error_norm",
            ):
                if rec.get(k) is not None:
                    old[k] = rec[k]
            if not str(old.get("fire_status") or "").startswith("FIRED_"):
                old["fire_status"] = fire_status
            return old

        self.target_seq += 1
        rec["id"] = "T{}".format(self.target_seq)
        self.targets.append(rec)
        if detected_in_acq:
            print(
                "[TARGET LOCKED] {} {} {} sector={} detectYaw={} lockYaw={} "
                "acqCone={} aimEnvelope={} crosshair={} ToF={}".format(
                    rec["id"], rec["color"], rec["shape"], sector_name,
                    fmt_deg(detect_yaw), fmt_deg(y), rec["acquisition_cone_deg"],
                    rec["aim_envelope_deg"], rec["crosshair_centered"], rec["tof_mm"],
                )
            )
        else:
            print(
                "[TARGET REMEMBERED] {} {} {} sector={} detectYaw={} outside acqCone={} ToF={}".format(
                    rec["id"], rec["color"], rec["shape"], sector_name,
                    fmt_deg(detect_yaw), rec["acquisition_cone_deg"], rec["tof_mm"],
                )
            )
        return rec

    def _target_upper_point_norm(self, candidate):
        """Return the SAME target's upper-biased aim point in normalized image coordinates."""
        if not isinstance(candidate, dict):
            return None
        bbox = candidate.get("bbox_norm")
        try:
            x, y, w, h = [float(v) for v in bbox]
            if w > 0.0 and h > 0.0:
                return (
                    x + 0.5 * w,
                    y + clamp(float(TARGET_UPPER_HIT_Y_RATIO), 0.05, 0.50) * h,
                )
        except Exception:
            pass
        # A missing bbox should not invent a shot point.  The center lock remains
        # valid memory, but physical fire waits for a real target box.
        return None

    def _aim_point_velocity(self, point_norm, yaw_min=None, yaw_max=None):
        """Servo an arbitrary visual point onto the camera crosshair during target service."""
        try:
            px, py = [float(v) for v in point_norm]
        except Exception:
            self._stop_gimbal_velocity()
            return False

        desired_x = 0.5 + TARGET_AIM_OFFSET_X
        desired_y = 0.5 + TARGET_AIM_OFFSET_Y
        ex = px - desired_x
        ey = py - desired_y

        yaw_speed = 0.0 if abs(ex) <= TARGET_AIM_DEADBAND_X else clamp(
            ex * TARGET_AIM_YAW_GAIN_DPS,
            -TARGET_AIM_YAW_MAX_DPS,
            +TARGET_AIM_YAW_MAX_DPS,
        )
        pitch_speed = 0.0 if abs(ey) <= TARGET_AIM_DEADBAND_Y else clamp(
            -ey * TARGET_CENTER_PITCH_GAIN_DPS,
            -TARGET_CENTER_PITCH_MAX_DPS,
            +TARGET_CENTER_PITCH_MAX_DPS,
        )

        p_now, y_now = self.owner.current_gimbal_relative()
        if y_now is not None:
            lo = -float(TARGET_AIM_YAW_LIMIT_DEG)
            hi = +float(TARGET_AIM_YAW_LIMIT_DEG)
            if yaw_min is not None:
                lo = max(lo, float(yaw_min))
            if yaw_max is not None:
                hi = min(hi, float(yaw_max))
            if y_now <= lo and yaw_speed < 0:
                yaw_speed = 0.0
            if y_now >= hi and yaw_speed > 0:
                yaw_speed = 0.0

        if p_now is not None:
            if p_now <= TARGET_CENTER_PITCH_HARD_MIN_DEG and pitch_speed < 0:
                pitch_speed = 0.0
            elif pitch_speed < 0 and p_now < (
                TARGET_CENTER_PITCH_SOFT_MIN_DEG + TARGET_CENTER_PITCH_BRAKE_ZONE_DEG
            ):
                headroom = max(0.0, float(p_now) - TARGET_CENTER_PITCH_HARD_MIN_DEG)
                pitch_speed = max(float(pitch_speed), -max(2.0, 6.0 * headroom))
            if p_now >= TARGET_CENTER_PITCH_MAX_DEG and pitch_speed > 0:
                pitch_speed = 0.0

        try:
            self.owner.gimbal.drive_speed(
                pitch_speed=float(pitch_speed),
                yaw_speed=float(yaw_speed),
            )
            return True
        except Exception as exc:
            self.owner.fault(
                "TARGET UPPER AIM",
                "drive_speed failed: {}: {}".format(type(exc).__name__, exc),
                "block this shot / keep target memory",
            )
            return False

    def _aim_upper_same_target(self, record, aim_envelope=None):
        """CENTER is already locked; now move the SAME target's upper point onto the crosshair."""
        if not TARGET_UPPER_AIM_ENABLED:
            record["upper_hit_verified"] = False
            record["upper_hit_disabled"] = True
            return True
        if not isinstance(record, dict):
            return False

        current = dict(record)
        expected = list(current.get("center_norm") or (0.5, 0.5))
        yaw_min = yaw_max = None
        if isinstance(aim_envelope, (tuple, list)) and len(aim_envelope) >= 2:
            yaw_min, yaw_max = float(aim_envelope[0]), float(aim_envelope[1])

        point = self._target_upper_point_norm(current)
        if point is None:
            record["upper_hit_verified"] = False
            record["upper_hit_error"] = "NO_BBOX"
            return False

        chassis_hold_yaw = self.owner.desired_yaw_for_heading(self.owner.heading)
        self.owner.pid_straight.reset()
        deadline = time.monotonic() + float(TARGET_UPPER_AIM_TIMEOUT_SEC)
        last_seq = -1
        lost_since = None
        stable_frames = 0
        last_err = (None, None)
        self._set_focus_roi(current)

        print(
            "[TARGET UPPER AIM] {} same-target point={:.0f}% from top; "
            "burst={} cadence={:.2f}s".format(
                record.get("id", "?"),
                100.0 * float(TARGET_UPPER_HIT_Y_RATIO),
                self.owner.get_fire_burst_count(),
                float(TARGET_FIRE_BURST_INTERVAL_SEC),
            )
        )

        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                if chassis_hold_yaw is not None:
                    try:
                        z_hold = self.owner.yaw_hold_command(chassis_hold_yaw, stationary=False)
                        self.owner.drive_speed_resilient(
                            x=0.0, y=0.0, z=z_hold,
                            timeout=DRIVE_COMMAND_TIMEOUT,
                            label="TARGET UPPER YAW HOLD",
                        )
                    except Exception:
                        pass

                with self.lock:
                    seq = self.frame_seq
                if seq == last_seq:
                    time.sleep(TARGET_AIM_POLL_SEC)
                    continue
                last_seq = seq

                latest = self._latest_match(current, expected, allow_blob_fallback=False)
                if latest is None:
                    self._stop_gimbal_velocity()
                    stable_frames = 0
                    if lost_since is None:
                        lost_since = time.monotonic()
                    if time.monotonic() - lost_since > TARGET_UPPER_AIM_LOST_GRACE_SEC:
                        record["upper_hit_verified"] = False
                        record["upper_hit_error"] = "TARGET_LOST"
                        print("[TARGET UPPER LOST] same target absent too long")
                        return False
                    time.sleep(TARGET_AIM_POLL_SEC)
                    continue

                lost_since = None
                current.update(latest)
                expected = list(current.get("center_norm", expected))
                self._set_focus_roi(current)
                point = self._target_upper_point_norm(current)
                if point is None:
                    record["upper_hit_verified"] = False
                    record["upper_hit_error"] = "NO_BBOX_DURING_TRACK"
                    return False

                px, py = point
                ex = float(px) - (0.5 + TARGET_AIM_OFFSET_X)
                ey = float(py) - (0.5 + TARGET_AIM_OFFSET_Y)
                last_err = (ex, ey)

                if abs(ex) <= TARGET_UPPER_AIM_TOL_X and abs(ey) <= TARGET_UPPER_AIM_TOL_Y:
                    stable_frames += 1
                    self._stop_gimbal_velocity()
                else:
                    stable_frames = 0
                    if not self._aim_point_velocity(point, yaw_min=yaw_min, yaw_max=yaw_max):
                        record["upper_hit_verified"] = False
                        record["upper_hit_error"] = "SERVO_FAILED"
                        return False

                self.status = (
                    "UPPER {} {} err=({:+.3f},{:+.3f}) hold={}/{}".format(
                        current.get("color"), current.get("shape"), ex, ey,
                        stable_frames, TARGET_UPPER_AIM_HOLD_FRAMES,
                    )
                )

                if stable_frames >= TARGET_UPPER_AIM_HOLD_FRAMES:
                    self._stop_gimbal_velocity()
                    verified = self._verify_exact(current, since_t=time.monotonic())
                    if verified is None:
                        stable_frames = 0
                        continue
                    vpoint = self._target_upper_point_norm(verified)
                    if vpoint is None:
                        stable_frames = 0
                        continue
                    vex = float(vpoint[0]) - (0.5 + TARGET_AIM_OFFSET_X)
                    vey = float(vpoint[1]) - (0.5 + TARGET_AIM_OFFSET_Y)
                    if abs(vex) > TARGET_UPPER_AIM_TOL_X or abs(vey) > TARGET_UPPER_AIM_TOL_Y:
                        current.update(verified)
                        expected = list(verified.get("center_norm", expected))
                        stable_frames = 0
                        continue

                    current.update(verified)
                    record["upper_hit_verified"] = True
                    record["upper_hit_y_ratio"] = float(TARGET_UPPER_HIT_Y_RATIO)
                    record["upper_hit_error_norm"] = [float(vex), float(vey)]
                    record["upper_lock_center_norm"] = current.get("center_norm")
                    record["upper_lock_bbox_norm"] = current.get("bbox_norm")
                    p_up, y_up = self.owner.current_gimbal_relative()
                    record["upper_lock_pitch_deg"] = p_up
                    record["upper_lock_yaw_deg"] = y_up
                    print(
                        "[TARGET UPPER LOCK] {} yaw={} pitch={} err=({:+.3f},{:+.3f})".format(
                            record.get("id", "?"), fmt_deg(y_up), fmt_deg(p_up), vex, vey
                        )
                    )
                    return True

                time.sleep(TARGET_AIM_POLL_SEC)

            ex, ey = last_err
            p_end, y_end = self.owner.current_gimbal_relative()
            record["upper_hit_verified"] = False
            record["upper_hit_error"] = "TIMEOUT"
            print(
                "[TARGET UPPER TIMEOUT] yaw={} pitch={} err=({}, {})".format(
                    fmt_deg(y_end), fmt_deg(p_end),
                    "NA" if ex is None else "{:+.3f}".format(ex),
                    "NA" if ey is None else "{:+.3f}".format(ey),
                )
            )
            return False
        finally:
            self._stop_gimbal_velocity()
            self._clear_focus_roi()
            self.owner.safe_stop()
            if chassis_hold_yaw is not None:
                try:
                    self.owner.align_heading_stationary(
                        chassis_hold_yaw,
                        timeout_sec=0.85,
                        tolerance_deg=max(0.65, PRE_MOVE_ALIGN_TOL_DEG),
                        settle_sec=0.06,
                    )
                except Exception:
                    pass

    def _compute_fire_solution(self, range_mm, fire_mode, yaw_now, pitch_now):
        """Compute physical muzzle LOS from a CAMERA-centred target.

        ``range_mm`` is the RAW slant range measured at the ToF origin.  The
        sensor is +8 cm forward of robot centre while the muzzle is +15 cm, so
        the muzzle is 7 cm farther forward than the ToF/camera longitudinal
        plane.  We reconstruct the target point from the camera-centred Gimbal
        pitch, then solve the pitch from the muzzle to that same point.

        At pitch~=0 this gives the requested geometry:
            ToF 600 mm -> robot-centre target ~= 680 mm
                       -> muzzle target      ~= 530 mm
        """
        try:
            raw_slant_m = max(0.02, float(range_mm) / 1000.0)
            yaw_now = float(yaw_now)
            pitch_now = float(pitch_now)
        except Exception:
            return None

        pitch_rad = math.radians(pitch_now)
        raw_planar_m = raw_slant_m * abs(math.cos(pitch_rad))

        # Target planar position measured from the chassis/Gimbal yaw centre.
        center_to_target_planar_m = (
            float(FIRE_TOF_FORWARD_FROM_CENTER_M) + raw_planar_m
        )

        # Camera is assumed to share the ToF forward plane until separately
        # measured.  Because the camera crosshair is already centred, its pitch
        # defines target height relative to the camera optical centre.
        camera_to_target_planar_m = max(0.005,
            center_to_target_planar_m - float(FIRE_CAMERA_FORWARD_FROM_CENTER_M)
        )
        target_z_from_camera_m = camera_to_target_planar_m * math.tan(pitch_rad)

        # Convert the same target point into the physical muzzle frame.
        muzzle_to_target_planar_m = (
            center_to_target_planar_m - float(FIRE_MUZZLE_FORWARD_FROM_CENTER_M)
        )
        # A target behind/effectively at the muzzle plane is not a valid ballistic
        # solution even if the ToF itself still returns a number.
        if muzzle_to_target_planar_m <= 0.015:
            return None

        target_z_from_muzzle_m = (
            float(FIRE_CAMERA_ABOVE_MUZZLE_M) + target_z_from_camera_m
        )
        geometric_fire_pitch = math.degrees(math.atan2(
            target_z_from_muzzle_m, muzzle_to_target_planar_m
        ))

        # Keep the old 'parallax' telemetry meaning: how much pitch changes from
        # the camera-centred pose to the physical muzzle LOS.
        parallax_deg = wrap_deg(geometric_fire_pitch - pitch_now)
        parallax_deg = clamp(parallax_deg, -FIRE_COMP_MAX_ABS_DEG, +FIRE_COMP_MAX_ABS_DEG)

        # ToF-vs-camera vertical separation remains diagnostic only.
        tof_to_camera_m = float(FIRE_TOF_ABOVE_CAMERA_M)
        tof_camera_parallax_deg = math.degrees(math.atan2(
            abs(tof_to_camera_m), max(0.005, camera_to_target_planar_m)
        ))
        tof_camera_parallax_deg *= (
            math.copysign(1.0, tof_to_camera_m) if tof_to_camera_m != 0 else 0.0
        )

        water_extra = float(WATER_EXTRA_PITCH_DEG) if str(fire_mode).upper() == "WATER" else 0.0
        target_pitch = clamp(
            pitch_now + parallax_deg + water_extra,
            FIRE_COMP_PITCH_MIN_DEG,
            FIRE_COMP_PITCH_MAX_DEG,
        )

        muzzle_slant_m = math.hypot(muzzle_to_target_planar_m, target_z_from_muzzle_m)
        center_slant_m = math.hypot(center_to_target_planar_m,
                                   target_z_from_camera_m + float(FIRE_CAMERA_ABOVE_MUZZLE_M))
        solution = {
            "fire_mode": str(fire_mode).upper(),
            "tof_range_mm": float(range_mm),
            "tof_raw_slant_mm": float(range_mm),
            "tof_raw_planar_mm": float(raw_planar_m * 1000.0),
            "robot_center_to_target_planar_mm": float(center_to_target_planar_m * 1000.0),
            "muzzle_to_target_planar_mm": float(muzzle_to_target_planar_m * 1000.0),
            "camera_to_target_planar_mm": float(camera_to_target_planar_m * 1000.0),
            "camera_lock_yaw_deg": yaw_now,
            "camera_lock_pitch_deg": pitch_now,
            "camera_forward_from_center_m": float(FIRE_CAMERA_FORWARD_FROM_CENTER_M),
            "tof_forward_from_center_m": float(FIRE_TOF_FORWARD_FROM_CENTER_M),
            "muzzle_forward_from_center_m": float(FIRE_MUZZLE_FORWARD_FROM_CENTER_M),
            "muzzle_ahead_of_tof_m": float(FIRE_MUZZLE_AHEAD_OF_TOF_M),
            "camera_above_muzzle_m": float(FIRE_CAMERA_ABOVE_MUZZLE_M),
            "tof_above_muzzle_m": float(FIRE_TOF_ABOVE_MUZZLE_M),
            "tof_above_camera_m": tof_to_camera_m,
            "camera_muzzle_parallax_pitch_deg": float(parallax_deg),
            "tof_camera_parallax_deg": float(tof_camera_parallax_deg),
            "water_extra_pitch_deg": float(water_extra),
            "fire_yaw_deg": yaw_now,
            "fire_pitch_deg": float(target_pitch),
            "muzzle_to_target_slant_m": float(muzzle_slant_m),
            "robot_center_to_target_slant_m": float(center_slant_m),
        }
        return solution

    def _move_to_fire_solution(self, solution):
        if not FIRE_AIM_GEOMETRY_ENABLED or not isinstance(solution, dict):
            return True
        yaw_target = solution.get("fire_yaw_deg")
        pitch_target = solution.get("fire_pitch_deg")
        if yaw_target is None or pitch_target is None:
            return False

        p0, y0 = self.owner.current_gimbal_relative()
        if p0 is None or y0 is None:
            return False
        if (
            abs(wrap_deg(float(y0) - float(yaw_target))) <= FIRE_COMP_YAW_TOL_DEG
            and abs(float(p0) - float(pitch_target)) <= FIRE_COMP_PITCH_TOL_DEG
        ):
            return True

        deadline = time.monotonic() + float(FIRE_COMP_TIMEOUT_SEC)
        stable = 0
        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                p_now, y_now = self.owner.current_gimbal_relative()
                if p_now is None or y_now is None:
                    time.sleep(0.025)
                    continue
                ey = wrap_deg(float(yaw_target) - float(y_now))
                ep = float(pitch_target) - float(p_now)
                if abs(ey) <= FIRE_COMP_YAW_TOL_DEG and abs(ep) <= FIRE_COMP_PITCH_TOL_DEG:
                    self._stop_gimbal_velocity()
                    stable += 1
                    if stable >= FIRE_COMP_SETTLE_SAMPLES:
                        return True
                    time.sleep(FIRE_COMP_SETTLE_SEC)
                    continue
                stable = 0
                ys = clamp(2.8 * ey, -30.0, +30.0)
                ps = clamp(3.0 * ep, -28.0, +28.0)
                if 0.0 < abs(ys) < 3.0:
                    ys = math.copysign(3.0, ys)
                if 0.0 < abs(ps) < 2.5:
                    ps = math.copysign(2.5, ps)
                # Never drive beyond the dedicated firing-pose pitch envelope.
                if float(p_now) <= FIRE_COMP_PITCH_MIN_DEG and ps < 0:
                    ps = 0.0
                if float(p_now) >= FIRE_COMP_PITCH_MAX_DEG and ps > 0:
                    ps = 0.0
                self.owner.gimbal.drive_speed(pitch_speed=float(ps), yaw_speed=float(ys))
                time.sleep(0.030)
        except Exception as exc:
            self.owner.fault(
                "FIRE AIM",
                "{}: {}".format(type(exc).__name__, exc),
                "do not fire / resume target sweep",
            )
        finally:
            self._stop_gimbal_velocity()

        p_now, y_now = self.owner.current_gimbal_relative()
        return bool(
            p_now is not None and y_now is not None
            and abs(wrap_deg(float(y_now) - float(yaw_target))) <= 2.5
            and abs(float(p_now) - float(pitch_target)) <= 1.5
        )

    def _maybe_fire(self, record, sector_name=None, fire_cone=None):
        if not TARGET_AUTO_FIRE_ENABLED or not isinstance(record, dict):
            return False
        tid = str(record.get("id") or "?")

        # GUI target rule: detection/memory still runs for every target, but the
        # blaster is permitted only for one of the 16 selected color/shape pairs.
        if not self.owner.target_allowed(record.get("color"), record.get("shape")):
            record["fire_status"] = "SKIPPED_TARGET_FILTER"
            record["selected_for_fire"] = False
            self.last_fire_event = "{} SKIP: {} {} disabled".format(
                tid, record.get("color"), record.get("shape")
            )
            self.save_targets()
            return False
        record["selected_for_fire"] = True

        if TARGET_FIRE_ONCE_PER_TARGET and (
            tid in self.fired_target_ids or str(record.get("fire_status") or "").startswith("FIRED_")
        ):
            return False

        sector = str(sector_name or record.get("scan_sector") or "").upper()
        cone = fire_cone if fire_cone is not None else self._sector_fire_cone(sector)
        aim_envelope = self._sector_aim_envelope(sector)
        detect_yaw = record.get("detected_sweep_yaw_deg")
        _p_now, yaw_now = self.owner.current_gimbal_relative()

        # V15 bearing policy:
        #   soft +/-20deg cone is checked at DETECTION time to establish
        #   shoot-intent (the inner +/-10deg remains PREFERRED); after that the crosshair may servo as far as +/-30deg
        #   from the sector axis to put the target centre on the reticle.
        preferred_cone = self._sector_preferred_fire_cone(sector)
        detected_in_acq = self._yaw_in_cone(detect_yaw, cone)
        detected_in_preferred = self._yaw_in_cone(detect_yaw, preferred_cone)
        in_aim_envelope = self._yaw_in_cone(yaw_now, aim_envelope)
        crosshair_centered = bool(record.get("crosshair_centered", False))

        record["scan_sector"] = sector
        record["lock_yaw_deg"] = yaw_now
        record["fire_cone_deg"] = None if cone is None else [float(cone[0]), float(cone[1])]
        record["in_fire_cone"] = bool(detected_in_acq)
        record["acquisition_cone_deg"] = record["fire_cone_deg"]
        record["preferred_acquisition_cone_deg"] = (
            None if preferred_cone is None else [float(preferred_cone[0]), float(preferred_cone[1])]
        )
        record["detected_in_preferred_cone"] = bool(detected_in_preferred)
        record["soft_acquisition"] = bool(detected_in_acq and not detected_in_preferred)
        record["detected_in_acquisition_cone"] = bool(detected_in_acq)
        record["shoot_intent"] = bool(detected_in_acq)
        record["aim_envelope_deg"] = (
            None if aim_envelope is None
            else [float(aim_envelope[0]), float(aim_envelope[1])]
        )
        record["in_aim_envelope"] = bool(in_aim_envelope)

        if not detected_in_acq:
            record["fire_status"] = "MEMORY_OUTSIDE_ACQ_CONE"
            print(
                "[FIRE MEMORY] {}: detectYaw={} outside {} acquisition cone={} -> remember only".format(
                    tid, fmt_deg(detect_yaw), sector, record["acquisition_cone_deg"]
                )
            )
            self.save_targets()
            return False

        if not in_aim_envelope:
            record["fire_status"] = "MEMORY_OUTSIDE_AIM_ENVELOPE"
            print(
                "[FIRE MEMORY] {}: Crosshair needed lockYaw={} outside {} aimEnvelope={} -> remember only".format(
                    tid, fmt_deg(yaw_now), sector, record["aim_envelope_deg"]
                )
            )
            self.save_targets()
            return False

        if not crosshair_centered:
            record["fire_status"] = "MEMORY_CROSSHAIR_NOT_CENTERED"
            print(
                "[FIRE MEMORY] {}: shoot-intent OK but crosshair is not verified at target centre -> remember only".format(tid)
            )
            self.save_targets()
            return False

        # V20.7 GOOD-ANGLE FIRE: wide detection/aim is useful for remembering
        # targets, but a water shot from a steep oblique angle is not.  Gate on
        # the FINAL lock yaw (not detectYaw) so Auto-Aim cannot drag a candidate
        # from a good detection bearing into a bad physical firing bearing.
        good_fire_cone = self._sector_good_fire_cone(sector)
        good_fire_angle = self._yaw_in_cone(yaw_now, good_fire_cone)
        record["good_fire_cone_deg"] = (
            None if good_fire_cone is None
            else [float(good_fire_cone[0]), float(good_fire_cone[1])]
        )
        record["good_fire_angle"] = bool(good_fire_angle)
        if not good_fire_angle:
            record["fire_status"] = "DEFER_BAD_ANGLE"
            record["deferred_lock_yaw_deg"] = yaw_now
            self.last_fire_event = "{} DEFER: BAD ANGLE {}".format(tid, fmt_deg(yaw_now))
            print(
                "[FIRE DEFER ANGLE] {}: {} lockYaw={} outside GOOD={} -> NO SHOT; keep target alive for a better view".format(
                    tid, sector, fmt_deg(yaw_now), record["good_fire_cone_deg"]
                )
            )
            self.save_targets()
            return False

        # V15 CENTER-ONLY: BBOX centre must sit on the camera crosshair for
        # 3 consecutive frames. Upper-target bias is disabled; range/parallax
        # compensation begins immediately after the centre lock.
        if TARGET_UPPER_AIM_ENABLED:
            if not self._aim_upper_same_target(record, aim_envelope=aim_envelope):
                record["fire_status"] = "BLOCKED_UPPER_AIM_NOT_LOCKED"
                self.last_fire_event = "{} BLOCKED: UPPER AIM".format(tid)
                self.save_targets()
                return False

        self.owner.safe_stop()
        time.sleep(TARGET_FIRE_RANGE_RECHECK_SEC)
        live = self.owner.sample_fresh_tof(samples=TARGET_FIRE_RANGE_SAMPLES)
        # RAW ToF remains the competition/safety range gate for backwards
        # compatibility.  Physical centre/muzzle ranges are added after the
        # fire solution is computed and are used for aiming geometry.
        record["fire_range_mm"] = None if live is None else float(live)
        record["fire_range_raw_tof_mm"] = None if live is None else float(live)
        record["fire_range_preferred_mm"] = float(TARGET_FIRE_PREFERRED_RANGE_MM)
        record["fire_range_limit_mm"] = float(TARGET_FIRE_MAX_RANGE_MM)

        if live is None:
            record["fire_status"] = "BLOCKED_RANGE_UNKNOWN"
            print("[FIRE BLOCK] {}: no fresh ToF".format(tid))
            return False
        if live > TARGET_FIRE_MAX_RANGE_MM:
            record["fire_status"] = "WAITING_TOO_FAR"
            print(
                "[FIRE HOLD] {}: {:.0f} mm > {:.0f} mm (remember target, do not fire)".format(
                    tid, live, TARGET_FIRE_MAX_RANGE_MM
                )
            )
            return False

        # No software minimum range: any valid fresh ToF <=1200 mm is eligible.
        # <=350 mm is the preferred high-confidence zone; 350-1200 mm is a legal
        # fallback and is explicitly labelled as such in telemetry/target memory.
        if sector == "FRONT":
            record["fire_range_zone"] = (
                "FRONT_PREFERRED_LE_350MM"
                if live <= TARGET_FRONT_SWEET_SPOT_MM
                else "FRONT_FALLBACK_350_TO_1200MM"
            )
        else:
            record["fire_range_zone"] = (
                "SIDE_PREFERRED_LE_350MM" if live <= TARGET_FIRE_PREFERRED_RANGE_MM
                else "SIDE_FALLBACK_350_TO_1200MM"
            )

        if live <= TARGET_FIRE_PREFERRED_RANGE_MM:
            print(
                "[FIRE RANGE] {}: {:.0f} mm -> PREFERRED <= {:.0f} mm".format(
                    tid, live, TARGET_FIRE_PREFERRED_RANGE_MM
                )
            )
        else:
            print(
                "[FIRE RANGE] {}: {:.0f} mm -> FALLBACK (preferred <= {:.0f}, hard <= {:.0f} mm)".format(
                    tid, live, TARGET_FIRE_PREFERRED_RANGE_MM, TARGET_FIRE_MAX_RANGE_MM
                )
            )

        # Camera crosshair is centered at this point. Convert that camera LOS
        # into the physical muzzle LOS using fresh ToF range + measured offsets.
        fire_mode = self.owner.get_fire_mode()
        p_lock, y_lock = self.owner.current_gimbal_relative()
        if p_lock is None or y_lock is None:
            record["fire_status"] = "BLOCKED_NO_GIMBAL_POSE"
            return False
        solution = self._compute_fire_solution(live, fire_mode, y_lock, p_lock)
        record["fire_mode"] = fire_mode
        record["fire_aim_solution"] = solution
        self.last_aim_solution = {} if solution is None else dict(solution)
        if solution is not None:
            record["fire_range_center_mm"] = solution.get("robot_center_to_target_planar_mm")
            record["fire_range_muzzle_mm"] = solution.get("muzzle_to_target_planar_mm")
        if solution is None:
            record["fire_status"] = "BLOCKED_NO_FIRE_SOLUTION"
            print("[FIRE BLOCK] {}: could not compute fire geometry".format(tid))
            return False
        if FIRE_AIM_GEOMETRY_ENABLED:
            print(
                "[FIRE SOLUTION] {} mode={} ToF(raw)={:.0f}mm center={:.0f}mm muzzle={:.0f}mm "
                "camPitch={:+.2f} parallax={:+.2f}deg -> firePitch={:+.2f}deg ToF-Cam={:+.2f}deg".format(
                    tid, fire_mode, live,
                    float(solution["robot_center_to_target_planar_mm"]),
                    float(solution["muzzle_to_target_planar_mm"]),
                    float(solution["camera_lock_pitch_deg"]),
                    float(solution["camera_muzzle_parallax_pitch_deg"]),
                    float(solution["fire_pitch_deg"]),
                    float(solution["tof_camera_parallax_deg"]),
                )
            )
            if not self._move_to_fire_solution(solution):
                record["fire_status"] = "BLOCKED_FIRE_POSE_NOT_REACHED"
                self.last_fire_event = "{} BLOCKED: FIRE POSE".format(tid)
                print("[FIRE BLOCK] {}: compensated fire pose not reached".format(tid))
                return False

        if not TARGET_REAL_FIRE_ENABLED:
            record["fire_status"] = "READY_DRY_RUN"
            print(
                "[FIRE DRY] {}: detectYaw={} lockYaw={} range={:.0f} mm zone={}".format(
                    tid, fmt_deg(detect_yaw), fmt_deg(yaw_now), live,
                    record["fire_range_zone"],
                )
            )
            return True
        if self.owner.blaster is None:
            record["fire_status"] = "BLOCKED_NO_BLASTER"
            print("[FIRE BLOCK] {}: blaster unavailable".format(tid))
            return False

        fire_mode = self.owner.get_fire_mode()
        if fire_mode == "WATER":
            fire_type = getattr(blaster, "WATER_FIRE", None)
        else:
            fire_mode = "INFRARED"
            fire_type = getattr(blaster, "INFRARED_FIRE", None)
        fire_times = self.owner.get_fire_burst_count()

        if fire_type is None:
            record["fire_status"] = "BLOCKED_FIRE_TYPE_UNAVAILABLE"
            self.last_fire_event = "{} BLOCKED: {} API unavailable".format(tid, fire_mode)
            print("[FIRE BLOCK] {}: {} constant unavailable in RoboMaster SDK".format(tid, fire_mode))
            return False

        # Re-check the ACTUAL yaw after fire-pose compensation.  Pitch
        # compensation should not change yaw, but feedback drift/coupling can.
        # Never let such drift turn a previously good lock into an oblique shot.
        _p_fire_now, y_fire_now = self.owner.current_gimbal_relative()
        good_fire_cone = self._sector_good_fire_cone(sector)
        if not self._yaw_in_cone(y_fire_now, good_fire_cone):
            record["fire_status"] = "DEFER_BAD_ANGLE"
            record["deferred_fire_pose_yaw_deg"] = y_fire_now
            record["good_fire_cone_deg"] = (
                None if good_fire_cone is None
                else [float(good_fire_cone[0]), float(good_fire_cone[1])]
            )
            self.last_fire_event = "{} DEFER: FIRE POSE ANGLE {}".format(tid, fmt_deg(y_fire_now))
            print(
                "[FIRE DEFER ANGLE] {}: finalFireYaw={} outside GOOD={} after compensation -> NO SHOT".format(
                    tid, fmt_deg(y_fire_now), record["good_fire_cone_deg"]
                )
            )
            self.save_targets()
            return False

        actual_shots = 0
        try:
            self._stop_gimbal_velocity()
            for shot_idx in range(int(fire_times)):
                ok = self.owner.blaster.fire(
                    fire_type=fire_type,
                    times=1,
                )
                if ok is False:
                    raise RuntimeError("blaster.fire returned False at burst shot {}".format(shot_idx + 1))
                actual_shots += 1
                print(
                    "[FIRE BURST] {} {} shot {}/{} cadence={:.2f}s".format(
                        tid, fire_mode, shot_idx + 1, fire_times,
                        float(TARGET_FIRE_BURST_INTERVAL_SEC),
                    )
                )
                if shot_idx + 1 < int(fire_times):
                    time.sleep(float(TARGET_FIRE_BURST_INTERVAL_SEC))
        except Exception as exc:
            record["fire_status"] = "FIRE_FAILED"
            record["fire_error"] = str(exc)
            record["fire_times_requested"] = int(fire_times)
            record["fire_times_actual"] = int(actual_shots)
            self.last_fire_event = "{} FAILED: {}".format(tid, exc)
            self.owner.fault(
                "FIRE",
                "{}: {}: {}".format(tid, type(exc).__name__, exc),
                "continue DFS / target remains remembered",
            )
            return False

        record["fire_status"] = "FIRED_{}".format(fire_mode)
        record["fired_at"] = datetime.now().isoformat(timespec="milliseconds")
        record["fire_times"] = int(actual_shots)
        record["fire_times_requested"] = int(fire_times)
        record["fire_burst_interval_sec"] = float(TARGET_FIRE_BURST_INTERVAL_SEC)
        record["fire_mode"] = fire_mode

        # Round-2 hint: save the LOGICAL node anchor plus the exact direction
        # from which a shot was proven to work.  Side/dead-end viewpoint shifts
        # are intentionally not baked into coordinates; round 2 replays the same
        # sector setup so temporary mecanum shifts remain safety-guarded by ToF/Sharp.
        fire_p, fire_y = self.owner.current_gimbal_relative()
        source_cell = tuple(self.owner.current)
        heading_idx = int(self.owner.heading) % 4
        record["fire_anchor"] = {
            "cell": [int(source_cell[0]), int(source_cell[1])],
            "heading_index": heading_idx,
            "heading": DIR_NAMES[heading_idx],
            "sector": str(record.get("scan_sector") or sector or "FRONT").upper(),
            "detected_sweep_yaw_deg": record.get("detected_sweep_yaw_deg"),
            "lock_yaw_deg": record.get("lock_yaw_deg"),
            "actual_fire_gimbal_yaw_deg": fire_y,
            "actual_fire_gimbal_pitch_deg": fire_p,
            "absolute_bearing_deg_from_north": (
                None if fire_y is None
                else wrap_deg(float(heading_idx) * 90.0 + float(fire_y))
            ),
            "range_mm": float(live),
            "raw_tof_range_mm": float(live),
            "robot_center_range_mm": (
                None if solution is None else solution.get("robot_center_to_target_planar_mm")
            ),
            "muzzle_range_mm": (
                None if solution is None else solution.get("muzzle_to_target_planar_mm")
            ),
            "target_estimated_grid_xy": record.get("estimated_grid_xy"),
            "color": record.get("color"),
            "shape": record.get("shape"),
            "source_target_id": tid,
            "fired_at": record.get("fired_at"),
        }
        self.fired_target_ids.add(tid)
        muzzle_mm = None if solution is None else solution.get("muzzle_to_target_planar_mm")
        self.last_fire_event = "{} FIRED {} x{} ToF={:.0f}mm Muzzle={}".format(
            tid, fire_mode, actual_shots, live,
            "NA" if muzzle_mm is None else "{:.0f}mm".format(float(muzzle_mm)),
        )
        print(
            "[FIRE {}] {} {} {} x{} sector={} detectYaw={} lockYaw={} ToF(raw)={:.0f}mm "
            "center={} muzzle={} zone={}".format(
                fire_mode, tid, record.get("color"), record.get("shape"), actual_shots,
                record.get("scan_sector"), fmt_deg(detect_yaw), fmt_deg(yaw_now), live,
                "NA" if solution is None else "{:.0f}mm".format(float(solution["robot_center_to_target_planar_mm"])),
                "NA" if muzzle_mm is None else "{:.0f}mm".format(float(muzzle_mm)),
                record["fire_range_zone"],
            )
        )
        time.sleep(TARGET_FIRE_SETTLE_SEC)
        self.save_targets()
        return True

    def save_targets(self):
        try:
            payload={
                "schema":"robomaster_hardened_target_memory","schema_version":2,
                "updated_at":datetime.now().isoformat(timespec="seconds"),
                "selected_fire_mode": self.owner.get_fire_mode(),
                "selected_fire_burst_count": self.owner.get_fire_burst_count(),
                "selected_target_classes": [
                    {"color": c, "shape": sh}
                    for c, sh in self.owner.get_target_selection()
                ],
                "fire_burst_interval_sec": float(TARGET_FIRE_BURST_INTERVAL_SEC),
                "upper_aim": {
                    "enabled": bool(TARGET_UPPER_AIM_ENABLED),
                    "hit_y_ratio_from_top": float(TARGET_UPPER_HIT_Y_RATIO),
                    "hold_frames": int(TARGET_UPPER_AIM_HOLD_FRAMES),
                },
                "aim_geometry": {
                    "camera_above_muzzle_m": FIRE_CAMERA_ABOVE_MUZZLE_M,
                    "tof_above_muzzle_m": FIRE_TOF_ABOVE_MUZZLE_M,
                    "tof_above_camera_m": FIRE_TOF_ABOVE_CAMERA_M,
                    "tof_forward_from_center_m": FIRE_TOF_FORWARD_FROM_CENTER_M,
                    "camera_forward_from_center_m": FIRE_CAMERA_FORWARD_FROM_CENTER_M,
                    "muzzle_forward_from_center_m": FIRE_MUZZLE_FORWARD_FROM_CENTER_M,
                    "muzzle_ahead_of_tof_m": FIRE_MUZZLE_AHEAD_OF_TOF_M,
                    "water_extra_pitch_deg": WATER_EXTRA_PITCH_DEG,
                },
                "detector":"Lab-CLAHE + broad HSV + LAB color confidence + shape + temporal + foam board gate",
                "real_fire_enabled":bool(TARGET_REAL_FIRE_ENABLED),
                "fire_range_preferred_mm":float(TARGET_FIRE_PREFERRED_RANGE_MM),
                "fire_range_limit_mm":float(TARGET_FIRE_MAX_RANGE_MM),
                "scan_pitch_deg":float(TARGET_SEARCH_PITCH_DEG),
                "radar_sectors":[
                    {
                        "name":name, "sweep_deg":[start,end],
                        "pre_slide":slide,
                        "preferred_acquisition_cone_deg":list(self._sector_preferred_fire_cone(name) or ()),
                        "acquisition_cone_deg":[lo,hi],
                        "aim_envelope_deg":list(self._sector_aim_envelope(name) or ()),
                    }
                    for name,start,end,slide,lo,hi in TARGET_RADAR_SECTORS
                ],
                "side_shift":{
                    "enabled":bool(TARGET_SIDE_SHIFT_ENABLED),
                    "viewpoint_shift_m":float(TARGET_SIDE_SHIFT_DISTANCE_M),
                    "destination_sharp_hard_stop_cm":float(TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM),
                    "max_shift_m":float(TARGET_SIDE_SHIFT_MAX_M),
                    "policy":"odometry ~35cm viewpoint shift; fresh Sharp emergency stop",
                },
                "foam_gate":{
                    "enabled":bool(TARGET_FOAM_GATE_ENABLED),"fail_closed":bool(TARGET_FOAM_FAIL_CLOSED),
                    "hsv_low":list(TARGET_FOAM_HSV_LOW),"hsv_high":list(TARGET_FOAM_HSV_HIGH),
                    "center_margin_px":TARGET_FOAM_CENTER_MARGIN_PX,"min_bbox_below_frac":TARGET_FOAM_MIN_BBOX_BELOW_FRAC,
                },
                "count":len(self.targets),"targets":self.targets,
            }
            atomic_write_text(TARGET_LATEST_JSON,json.dumps(payload,ensure_ascii=False,indent=2))
        except Exception as exc:
            self.owner.fault("TARGET SAVE",f"{type(exc).__name__}: {exc}","continue")

    # ---------------- preview ----------------
    def _draw_overlay(self,frame,detections,profile,coverage):
        out=frame.copy(); h,w=out.shape[:2]; x1,y1,x2,y2=self._roi_px(out)
        if profile is not None:
            pts=[]
            for px in range(max(0,x1),min(x2,len(profile))):
                wy=float(profile[px])
                if math.isfinite(wy):
                    pts.append((px,int(round(wy))))
                elif len(pts)>=2:
                    cv2.polylines(out,[np.asarray(pts,np.int32)],False,(255,255,0),2,cv2.LINE_AA); pts=[]
            if len(pts)>=2:
                cv2.polylines(out,[np.asarray(pts,np.int32)],False,(255,255,0),2,cv2.LINE_AA)
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),1)
        colors={"RED":(0,0,255),"YELLOW":(0,255,255),"GREEN":(0,220,0),"BLUE":(255,100,0)}
        for d in detections:
            x,y,bw,bh=d.get("bbox_px",(0,0,0,0)); c=colors.get(d.get("color"),(255,255,255))
            cv2.rectangle(out,(x,y),(x+bw,y+bh),c,2)
            cx=int((x+bw*0.5)); cy=int((y+bh*0.5)); cv2.circle(out,(cx,cy),3,c,-1)
            txt=f"{d.get('color')} {d.get('shape')} {float(d.get('score',0)):.2f}"
            cv2.putText(out,txt,(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1,cv2.LINE_AA)
        aim_x=int(round((0.5+TARGET_AIM_OFFSET_X)*w)); aim_y=int(round((0.5+TARGET_AIM_OFFSET_Y)*h))
        cv2.drawMarker(out,(aim_x,aim_y),(0,255,0),cv2.MARKER_CROSS,24,1)
        cv2.putText(out,f"FOAM {'OK' if coverage>=TARGET_FOAM_MIN_ROI_COVERAGE else 'FAIL-OPEN'} cov={coverage*100:.0f}% | {self.status[:70]}",(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
        return out


# ============================================================
# HELPERS
# ============================================================
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def tof_center_planar_mm(raw_tof_mm, pitch_deg=0.0):
    """Convert RAW ToF slant range to planar target range from robot centre.

    The sensor origin is TOF_FORWARD_FROM_CENTER_M in front of the yaw centre.
    This helper is for map/geometry calculations only.  Safety/braking must keep
    using raw ToF values.
    """
    if raw_tof_mm is None:
        return None
    try:
        raw_m = float(raw_tof_mm) / 1000.0
        if not math.isfinite(raw_m):
            return None
        planar_m = (raw_m * abs(math.cos(math.radians(float(pitch_deg))))) + float(TOF_FORWARD_FROM_CENTER_M)
        return max(0.0, planar_m * 1000.0)
    except Exception:
        return None


def tof_topology_center_mm(raw_tof_mm):
    """Robot-centre planar range used only for OPEN/WALL topology."""
    return tof_center_planar_mm(raw_tof_mm, GIMBAL_PITCH_DEG)


def tof_is_open_from_center(raw_tof_mm):
    centre_mm = tof_topology_center_mm(raw_tof_mm)
    return bool(centre_mm is not None and centre_mm > TOF_OPEN_THRESHOLD_MM)


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


def sharp_raw_means_far(raw_adc, calibration):
    """True when Sharp ADC is valid but below the calibrated far-end value.

    GP2Y0A41 calibration intentionally returns None beyond the last calibrated
    distance (~24 cm).  For side-shift collision guarding that is *safe/far*, not
    a sensor failure.  Distinguish it from a missing/invalid ADC value.
    """
    if raw_adc is None:
        return False
    try:
        raw = float(raw_adc)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(raw) or raw < SHARP_MIN_PLAUSIBLE_ADC:
        return False
    try:
        far_adc = float(calibration[-1][1])
    except Exception:
        return False
    return raw < far_adc


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
        self.camera = None
        self.vision = None
        self.blaster = None

        self.state = SharedState()
        self.target_system = TargetVisionSubsystem(self)
        self.fire_mode_lock = threading.Lock()
        self.fire_mode = str(TARGET_FIRE_MODE_DEFAULT).upper()
        self.fire_burst_lock = threading.Lock()
        self.fire_burst_count = int(TARGET_FIRE_BURST_DEFAULT)
        self.target_filter_lock = threading.Lock()
        self.target_selection = set(TARGET_FILTER_ALL)
        self.mission_mode = "ROUND1"
        self.round1_memory = None
        self.round2_hints = []
        self.round2_result = {}
        self.cleanup_done = False
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
        # Set only after an actual 90/180 chassis turn.  The next edge may use
        # the angled IR whiskers to trim itself through a doorway/corner.
        self.ir_corner_trim_pending = False

    def set_fire_mode(self, mode):
        mode = str(mode or "INFRARED").upper().strip()
        if mode not in ("INFRARED", "WATER"):
            mode = "INFRARED"
        with self.fire_mode_lock:
            self.fire_mode = mode
        print("[FIRE MODE] {}".format(mode))
        return mode

    def get_fire_mode(self):
        with self.fire_mode_lock:
            return str(self.fire_mode)

    def set_fire_burst_count(self, count):
        try:
            count = int(count)
        except Exception:
            count = int(TARGET_FIRE_BURST_DEFAULT)
        if count not in TARGET_FIRE_BURST_OPTIONS:
            count = int(TARGET_FIRE_BURST_DEFAULT)
        with self.fire_burst_lock:
            self.fire_burst_count = count
        print("[FIRE BURST MODE] {} shot(s), interval={:.2f}s".format(
            count, float(TARGET_FIRE_BURST_INTERVAL_SEC)
        ))
        return count

    def get_fire_burst_count(self):
        with self.fire_burst_lock:
            return int(self.fire_burst_count)

    def set_mission_mode(self, mode):
        mode = str(mode or "ROUND1").upper().replace(" ", "")
        if mode in ("1", "ROUND1", "R1"):
            mode = "ROUND1"
        elif mode in ("2", "ROUND2", "R2"):
            mode = "ROUND2"
        else:
            mode = "ROUND1"
        self.mission_mode = mode
        print("[MISSION MODE] {}".format(mode))
        return mode

    def set_target_selection(self, pairs):
        cleaned = set()
        for item in pairs or ():
            try:
                color, shape = item
            except Exception:
                continue
            key = (str(color).upper(), str(shape).upper())
            if key in TARGET_FILTER_ALL:
                cleaned.add(key)
        with self.target_filter_lock:
            self.target_selection = cleaned
        print("[TARGET FILTER] enabled {}/16: {}".format(
            len(cleaned), ", ".join("{}/{}".format(c, sh) for c, sh in sorted(cleaned)) or "NONE"
        ))
        return set(cleaned)

    def get_target_selection(self):
        with self.target_filter_lock:
            return sorted(self.target_selection)

    def target_allowed(self, color, shape):
        key = (str(color or "").upper(), str(shape or "").upper())
        with self.target_filter_lock:
            return key in self.target_selection

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
            # moveto may acknowledge while pitch remains around -15deg.  Use
            # feedback velocity control before resorting to physical recenter.
            if self.gimbal_velocity_recover(
                0.0,pitch_deg=GIMBAL_PITCH_DEG,timeout_sec=1.6,require_motion_safe=True
            ) and self.gimbal_front_safe_for_motion():
                self.fault("GIMBAL FRONT",f"velocity-recovered on attempt {attempt}","resume motion")
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
        raw_open_equiv = max(0.0, TOF_OPEN_THRESHOLD_MM - TOF_FORWARD_FROM_CENTER_M*1000.0)
        print(f" ToF OPEN   : center > {TOF_OPEN_THRESHOLD_MM:.0f} mm (raw approx > {raw_open_equiv:.0f} mm)")
        print(f" ToF origin : +{TOF_FORWARD_FROM_CENTER_M*100.0:.1f} cm from robot centre; safety uses RAW ToF")
        print(f" Muzzle     : +{FIRE_MUZZLE_FORWARD_FROM_CENTER_M*100.0:.1f} cm; +{FIRE_MUZZLE_AHEAD_OF_TOF_M*100.0:.1f} cm ahead of ToF")
        print(f" Runtime    : RUN pose=(0,0), FRONT=N")
        print(" Features   : DFS + map + ToF node-probe + resilient gimbal + Sharp + IR + runtime-zero yaw PID")
        print(" DFS speed  : explore={:.2f} m/s | known/backtrack={:.2f} m/s".format(DFS_EXPLORE_SPEED_MPS, DFS_KNOWN_SPEED_MPS))
        print(" Targets    : Lab-CLAHE + HSV/LAB + shape/temporal + Foam-board gate + CENTER -> ToF geometry -> selectable fire")
        print(" Fire mode  : {}".format(self.get_fire_mode()))
        print(" Pose rule  : side radar uses temporary lateral viewpoint shifts; always returns node anchor; no forward target attack")
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
            self.camera = getattr(self.ep_robot, "camera", None)
            self.vision = getattr(self.ep_robot, "vision", None)
            self.blaster = getattr(self.ep_robot, "blaster", None)
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

        # Establish physical/chassis-relative Gimbal FRONT.  SDK action=False is
        # non-fatal; recenter_gimbal_and_zero() verifies feedback and falls back
        # to closed-loop velocity control before declaring a real hardware fault.
        if not self.recenter_gimbal_and_zero():
            self.enter_safe_pause("gimbal could not establish a reliable runtime zero")
            return False

        if not self.gimbal_front_down(force=True):
            self.fault("GIMBAL", "front/down startup pose not exact", "continue; each scan will retry")

        # Vision is best-effort and deliberately non-fatal to maze exploration.
        try:
            self.target_system.start()
        except Exception as exc:
            self.fault("VISION START", f"{type(exc).__name__}: {exc}", "continue DFS without targets")

        self.connected = True
        self.mission_start_t = time.monotonic()
        print(f"[READY] runtime yaw-zero raw={self.base_yaw_deg:+.2f} deg -> logical yaw=0.00 deg; root=(0,0); heading=N")
        return True

    def cleanup(self):
        if self.cleanup_done:
            return
        self.cleanup_done = True
        print("\n[CLEANUP] stopping robot and saving map...")
        self.running = False
        self.safe_stop()
        try:
            self.target_system.stop()
        except Exception:
            pass
        try:
            self.target_system.save_targets()
        except Exception:
            pass
        try:
            self.save_map(final=self.map_complete)
        except Exception as exc:
            print(f"[MAP SAVE WARN] cleanup save failed: {exc}")
        if self.mission_mode == "ROUND1":
            try:
                self.save_round1_attack_memory()
            except Exception as exc:
                print(f"[ROUND1 MEMORY WARN] cleanup snapshot failed: {exc}")

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

    def lateral_yaw_hold_command(self, target_yaw=None):
        """Stronger yaw hold for pure mecanum lateral recovery.

        IR/Sharp recovery can create much more reaction torque than straight
        motion.  Keep this controller independent from the normal straight PID
        so normal forward travel is not made twitchy.
        """
        err = self.yaw_error_deg(target_yaw)
        if err is None or abs(err) <= LATERAL_YAW_HOLD_DEADBAND_DEG:
            return 0.0
        z = YAW_DRIVE_SIGN * LATERAL_YAW_HOLD_KP * float(err)
        z = clamp(z, -LATERAL_YAW_HOLD_MAX_DPS, +LATERAL_YAW_HOLD_MAX_DPS)
        if 0.0 < abs(z) < LATERAL_YAW_HOLD_MIN_DPS:
            z = math.copysign(LATERAL_YAW_HOLD_MIN_DPS, z)
        return z

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
        # If STOP was pressed, the loop exits because running=False.  That is a
        # user stop, not a yaw-controller timeout; do not emit a misleading fault.
        if not self.running:
            return False
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
            # V20: do not pay a full stationary settle when the chassis is already
            # inside the pre-move tolerance.  Yaw PID remains active during motion.
            same_heading_err = self.yaw_error_deg(target_yaw)
            if same_heading_err is None or abs(same_heading_err) > PRE_MOVE_ALIGN_TOL_DEG:
                if not self.align_heading_stationary(
                    target_yaw, timeout_sec=PRE_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.07
                ):
                    self.fault("TURN", f"already facing {DIR_NAMES[target_dir]} but yaw could not re-align", "retry/defer")
                    return False
            self.gimbal_front_down()
            if self.gimbal_front_safe_for_motion():
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
                self.ir_corner_trim_pending = True
                self.sharp_authority = None
                self.pid_straight.reset()
                self.gimbal_front_down(force=True)
                if not self.gimbal_front_safe_for_motion():
                    self.recover_gimbal_front()
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
        """Establish a reliable FRONT reference without trusting SDK action status.

        RoboMaster ``sub_angle`` already reports the gimbal axis angles relative
        to the chassis, so logical FRONT is raw yaw ~= 0 deg.  Older code could
        accidentally redefine a partially-recentered angle as a new software
        zero, which then added that error to every later moveto command.

        Competition policy:
          1) Try physical recenter briefly.
          2) Judge success from live angle feedback, NOT wait_for_completed().
          3) If recenter stalls/fails, immediately use closed-loop drive_speed()
             to bring the gimbal to yaw=0 / pitch=-5.
          4) Abort only when live feedback proves the gimbal cannot be placed in
             a safe forward pose.  SDK ``False`` alone is never fatal.
        """
        if self.gimbal is None:
            return False

        # sub_angle() is chassis-relative already.  Keep zero offsets at zero so
        # a raw -5deg pitch does not become "local zero" and later turn a -5deg
        # request into a -10deg physical command.
        self.gimbal_zero_pitch_raw = 0.0
        self.gimbal_zero_yaw_raw = 0.0

        def feedback_safe():
            p, y = self.current_gimbal_raw()
            if p is None or y is None:
                return False, p, y
            ok = (
                abs(wrap_deg(float(y))) <= STARTUP_GIMBAL_ACCEPT_YAW_DEG
                and STARTUP_GIMBAL_ACCEPT_PITCH_MIN_DEG
                    <= float(p)
                    <= STARTUP_GIMBAL_ACCEPT_PITCH_MAX_DEG
            )
            return bool(ok), float(p), float(y)

        for attempt in range(1, int(STARTUP_GIMBAL_RECENTER_ATTEMPTS) + 1):
            action_ok = False
            try:
                action = self.gimbal.recenter(
                    pitch_speed=min(float(GIMBAL_PITCH_SPEED), 110.0),
                    yaw_speed=min(float(GIMBAL_YAW_SPEED), 160.0),
                )
                action_ok = self._wait_action(
                    action, GIMBAL_ACTION_TIMEOUT_SEC, "GIMBAL RECENTER"
                )
            except Exception as exc:
                self.fault(
                    "GIMBAL RECENTER",
                    f"attempt {attempt}: {type(exc).__name__}: {exc}",
                    "verify feedback / velocity fallback",
                )

            # The action object can report False even though the mechanism moved.
            # Give feedback a short independent settle window.
            deadline = time.monotonic() + STARTUP_GIMBAL_RECENTER_VERIFY_SEC
            while self.running and time.monotonic() < deadline:
                ok, p, y = feedback_safe()
                if ok:
                    print(
                        f"[GIMBAL STARTUP OK] raw p={p:+.2f} y={y:+.2f} "
                        f"(recenter action_ok={action_ok})"
                    )
                    return True
                time.sleep(0.03)

            ok, p, y = feedback_safe()
            print(
                f"[GIMBAL STARTUP RECENTER MISS] attempt={attempt}/"
                f"{STARTUP_GIMBAL_RECENTER_ATTEMPTS} action_ok={action_ok} "
                f"raw_p={p} raw_y={y} -> closed-loop fallback"
            )

            # Do not repeat a failed SDK action for seconds.  Use the telemetry
            # controller that was already proven to move the real gimbal.
            if self.gimbal_velocity_recover(
                0.0,
                pitch_deg=GIMBAL_PITCH_DEG,
                timeout_sec=STARTUP_GIMBAL_FALLBACK_TIMEOUT_SEC,
                require_motion_safe=True,
            ):
                ok, p, y = feedback_safe()
                if ok or (
                    p is not None and y is not None
                    and abs(wrap_deg(float(y))) <= GIMBAL_MOTION_FRONT_YAW_TOL_DEG
                    and GIMBAL_MOTION_SAFE_PITCH_MIN_DEG
                        <= float(p)
                        <= GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
                ):
                    print(
                        f"[GIMBAL STARTUP FALLBACK OK] raw p={float(p):+.2f} "
                        f"y={float(y):+.2f} -> FRONT established"
                    )
                    return True

            # Stop velocity before another attempt.
            try:
                self.gimbal.drive_speed(pitch_speed=0.0, yaw_speed=0.0)
            except Exception:
                pass
            time.sleep(0.06)

        # One final feedback-only acceptance: if another command/settling event
        # brought the gimbal safely forward, do not throw the mission away.
        ok, p, y = feedback_safe()
        if ok:
            self.fault(
                "GIMBAL STARTUP",
                f"late feedback accept p={p:+.2f} y={y:+.2f}",
                "continue mission",
            )
            return True

        self.fault(
            "GIMBAL STARTUP",
            f"cannot establish FRONT after recenter + velocity fallback; p={p} y={y}",
            "mapping requires movable ToF gimbal",
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
            and GIMBAL_MOTION_SAFE_PITCH_MIN_DEG <= p <= GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
        )

    def gimbal_velocity_recover(self, yaw_deg, pitch_deg=GIMBAL_PITCH_DEG, timeout_sec=None, require_motion_safe=False):
        """Closed-loop velocity fallback for a stuck/lying SDK gimbal action.

        This uses sub_angle feedback directly.  It is intentionally available to
        topology/motion code, not only the camera radar, because the field log
        showed moveto() could leave +/-90deg scans at the correct yaw but a large
        pitch offset, or occasionally leave yaw tens of degrees short.
        """
        if self.gimbal is None:
            return False
        yaw_deg=clamp(float(yaw_deg),GIMBAL_SOFT_YAW_MIN_DEG,GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg=clamp(float(pitch_deg),GIMBAL_SOFT_PITCH_MIN_DEG,GIMBAL_SOFT_PITCH_MAX_DEG)
        p0,y0=self.current_gimbal_relative()
        yaw_dist=90.0 if y0 is None else abs(wrap_deg(yaw_deg-float(y0)))
        pitch_dist=10.0 if p0 is None else abs(float(pitch_deg)-float(p0))
        if timeout_sec is None:
            timeout_sec=max(
                0.9,
                yaw_dist/max(30.0,GIMBAL_RECOVERY_YAW_MAX_DPS*0.75)+0.65,
                pitch_dist/max(10.0,GIMBAL_RECOVERY_PITCH_MAX_DPS*0.65)+0.55,
            )
        deadline=time.monotonic()+float(timeout_sec)
        stable=0
        try:
            while self.running and time.monotonic()<deadline:
                p,y=self.current_gimbal_relative()
                if p is None or y is None:
                    time.sleep(0.03); continue
                ey=wrap_deg(yaw_deg-float(y)); ep=float(pitch_deg)-float(p)
                yaw_ok=abs(ey)<=GIMBAL_RECOVERY_YAW_TOL_DEG
                pitch_ok=abs(ep)<=GIMBAL_RECOVERY_PITCH_TOL_DEG
                if require_motion_safe:
                    pitch_ok = pitch_ok or (
                        GIMBAL_MOTION_SAFE_PITCH_MIN_DEG <= float(p) <= GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
                    )
                if yaw_ok and pitch_ok:
                    stable+=1
                    try: self.gimbal.drive_speed(pitch_speed=0.0,yaw_speed=0.0)
                    except Exception: pass
                    if stable>=GIMBAL_RECOVERY_SETTLE_SAMPLES:
                        return True
                    time.sleep(0.035); continue
                stable=0
                ys=0.0 if yaw_ok else clamp(2.7*ey,-GIMBAL_RECOVERY_YAW_MAX_DPS,+GIMBAL_RECOVERY_YAW_MAX_DPS)
                ps=0.0 if pitch_ok else clamp(2.4*ep,-GIMBAL_RECOVERY_PITCH_MAX_DPS,+GIMBAL_RECOVERY_PITCH_MAX_DPS)
                if 0.0<abs(ys)<GIMBAL_RECOVERY_YAW_MIN_DPS:
                    ys=math.copysign(GIMBAL_RECOVERY_YAW_MIN_DPS,ys)
                if 0.0<abs(ps)<GIMBAL_RECOVERY_PITCH_MIN_DPS:
                    ps=math.copysign(GIMBAL_RECOVERY_PITCH_MIN_DPS,ps)
                self.gimbal.drive_speed(pitch_speed=float(ps),yaw_speed=float(ys))
                time.sleep(0.03)
        except Exception as exc:
            self.fault("GIMBAL VELOCITY RECOVER",f"{type(exc).__name__}: {exc}","stop gimbal / caller fallback")
        finally:
            try: self.gimbal.drive_speed(pitch_speed=0.0,yaw_speed=0.0)
            except Exception: pass
        p,y=self.current_gimbal_relative()
        if p is None or y is None:
            return False
        yaw_ok=abs(wrap_deg(float(y)-yaw_deg))<=GIMBAL_RECOVERY_YAW_TOL_DEG
        if require_motion_safe:
            pitch_ok=GIMBAL_MOTION_SAFE_PITCH_MIN_DEG <= float(p) <= GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
        else:
            pitch_ok=(abs(float(p)-pitch_deg)<=GIMBAL_RECOVERY_PITCH_TOL_DEG or
                      GIMBAL_TOF_SAFE_PITCH_MIN_DEG <= float(p) <= GIMBAL_TOF_SAFE_PITCH_MAX_DEG)
        return bool(yaw_ok and pitch_ok)

    def gimbal_goto(self, yaw_deg, pitch_deg=GIMBAL_PITCH_DEG, force=False):
        """Topology/motion gimbal positioning with one-shot SDK + fast fallback.

        Repeating an SDK moveto() that has already failed costs several seconds
        and did not improve the 2026-10-01 field behavior.  Try it once, verify
        live feedback, then switch immediately to closed-loop velocity recovery.
        """
        if self.gimbal is None:
            return False
        yaw_deg = clamp(float(yaw_deg), GIMBAL_SOFT_YAW_MIN_DEG, GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg = clamp(float(pitch_deg), GIMBAL_SOFT_PITCH_MIN_DEG, GIMBAL_SOFT_PITCH_MAX_DEG)
        if not force and self.gimbal_at_target(yaw_deg, pitch_deg):
            return True

        raw_p = pitch_deg + (self.gimbal_zero_pitch_raw or 0.0)
        raw_y = yaw_deg + (self.gimbal_zero_yaw_raw or 0.0)

        try:
            action = self.gimbal.moveto(
                pitch=raw_p, yaw=raw_y,
                pitch_speed=GIMBAL_PITCH_SPEED,
                yaw_speed=GIMBAL_YAW_SPEED,
            )
            self._wait_action(action, GIMBAL_ACTION_TIMEOUT_SEC, "GIMBAL MOVETO")
        except Exception as exc:
            self.fault("GIMBAL MOVETO", f"{type(exc).__name__}: {exc}", "velocity fallback")

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
        self.fault(
            "GIMBAL VERIFY",
            f"target p={pitch_deg:+.1f} y={yaw_deg:+.1f}; actual p={p} y={y}",
            "immediate velocity fallback",
        )

        yaw_dist = 90.0 if y is None else abs(wrap_deg(yaw_deg - float(y)))
        recover_timeout = max(0.75, yaw_dist / 70.0 + 0.45)
        if self.gimbal_velocity_recover(yaw_deg, pitch_deg=pitch_deg, timeout_sec=recover_timeout):
            p, y = self.current_gimbal_relative()
            self.fault(
                "GIMBAL RECOVER",
                f"velocity fallback reached yaw={fmt_deg(y)} pitch={fmt_deg(p)}",
                "resume",
            )
            return True
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

    def read_sharp_cm_fast(self, samples=2, interval_sec=0.004):
        """Low-latency conservative Sharp read for fast lateral motion.

        The normal ``read_sharp_cm`` intentionally returns a median of the last
        SHARP_FILTER_SAMPLES ADC values.  That is excellent for wall following,
        but it adds dangerous phase lag while the chassis is strafing quickly
        toward a wall.  This helper bypasses that history and reads fresh ADC
        samples directly.  The *minimum distance* of the fresh samples is used,
        so a close reading wins immediately instead of waiting for a 5-sample
        median to catch up.

        It does not modify the normal history buffers, so forward wall-follow
        filtering remains unchanged.
        """
        n = max(1, int(samples))
        l_cm_vals = []
        r_cm_vals = []
        l_adc_last = None
        r_adc_last = None
        for i in range(n):
            try:
                raw = self.sensor_adapter.get_adc(id=SHARP_LEFT_ID, port=SENSOR_PORT)
                if raw is not None:
                    l_adc_last = float(raw)
                    cm = adc_to_cm(l_adc_last, LEFT_CAL)
                    if cm is not None and math.isfinite(float(cm)):
                        l_cm_vals.append(float(cm))
            except Exception:
                pass
            try:
                raw = self.sensor_adapter.get_adc(id=SHARP_RIGHT_ID, port=SENSOR_PORT)
                if raw is not None:
                    r_adc_last = float(raw)
                    cm = adc_to_cm(r_adc_last, RIGHT_CAL)
                    if cm is not None and math.isfinite(float(cm)):
                        r_cm_vals.append(float(cm))
            except Exception:
                pass
            if i + 1 < n and interval_sec > 0:
                time.sleep(float(interval_sec))

        # Conservative for collision avoidance: the closest fresh sample wins.
        l_cm = min(l_cm_vals) if l_cm_vals else None
        r_cm = min(r_cm_vals) if r_cm_vals else None
        return l_cm, r_cm, l_adc_last, r_adc_last

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

    def ir_simple_side_clear(self, target_yaw, context="MOVE"):
        """Fast merged IR + Sharp side-clear policy.

        ACTIVE LOW:
          LEFT only  -> strafe RIGHT.
          RIGHT only -> strafe LEFT.

        IR chooses the escape direction.  The Sharp sensor on the SAME/source
        side is allowed to confirm adequate clearance after a small real move,
        while destination Sharp remains the collision veto.  This avoids the
        old IR-clear -> immediate Sharp-side-escape double recovery.
        """
        self.safe_stop()
        l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=IR_FILTER_SAMPLES)

        if not l_low and not r_low:
            return "CLEAR"

        if l_low and r_low:
            print(
                "[IR SIMPLE {}] BOTH LOW IR=({},{}) -> STOP only; "
                "no lateral guess, ToF/Sharp own safety".format(
                    context, l_raw, r_raw
                )
            )
            return "BOTH"

        sign = +1.0 if l_low else -1.0
        source_name = "LEFT" if l_low else "RIGHT"
        dest_name = "RIGHT" if l_low else "LEFT"
        direction_name = "RIGHT" if sign > 0.0 else "LEFT"

        anchor = self.current_position()
        if anchor is None:
            print(
                "[IR SIMPLE {}] {} LOW but no odometry -> STOP only".format(
                    context, source_name
                )
            )
            return "HOLD"

        deadline = time.monotonic() + float(IR_SIMPLE_TIMEOUT_SEC)
        clear_count = 0
        sharp_clear_count = 0
        moved = 0.0
        stop_reason = "TIMEOUT"
        last_source_cm = None
        last_dest_cm = None

        print(
            "[IR+SHARP {}] {} LOW -> strafe {} until IR clears or {} Sharp >= {:.1f}cm".format(
                context, source_name, direction_name,
                source_name, IR_SIMPLE_SOURCE_SHARP_CLEAR_CM,
            )
        )

        while self.running and time.monotonic() < deadline:
            pos = self.current_position()
            if pos is None:
                stop_reason = "NO_ODOM"
                break

            moved = abs(self.cell_lateral_offset(anchor, pos, target_yaw))
            if moved >= IR_SIMPLE_MAX_LATERAL_M:
                stop_reason = "MAX_LATERAL"
                break

            # Keep IR responsive, but do not let repeated 3-sample filtering be
            # the only condition that ends the maneuver.
            l2, r2, lr2, rr2 = self.read_ir_filtered(samples=2)
            source_low = l2 if sign > 0.0 else r2

            lcm, rcm, _la, _ra = self.read_sharp_cm()
            source_cm = lcm if sign > 0.0 else rcm
            dest_cm = rcm if sign > 0.0 else lcm
            last_source_cm, last_dest_cm = source_cm, dest_cm

            if not source_low:
                clear_count += 1
            else:
                clear_count = 0

            # Sharp may terminate a noisy/sticky IR recovery only after the robot
            # has physically moved a few cm away from the triggering corner.
            sharp_safe = bool(
                source_cm is not None
                and source_cm >= IR_SIMPLE_SOURCE_SHARP_CLEAR_CM
                and moved >= IR_SIMPLE_SOURCE_SHARP_CLEAR_MIN_MOVE_M
            )
            sharp_clear_count = sharp_clear_count + 1 if sharp_safe else 0

            if clear_count >= IR_SIMPLE_CLEAR_CONFIRM:
                stop_reason = "IR_CLEAR"
                break
            if sharp_clear_count >= IR_SIMPLE_CLEAR_CONFIRM:
                stop_reason = "SOURCE_SHARP_CLEAR"
                break

            if dest_cm is not None and dest_cm <= IR_SIMPLE_DEST_SHARP_STOP_CM:
                stop_reason = "{}_SHARP_{:.1f}CM".format(dest_name, dest_cm)
                break

            # Adaptive escape speed: run fast while destination clearance is
            # healthy, slow down as the opposite wall approaches.
            speed = IR_SIMPLE_STRAFE_SPEED_MPS
            if dest_cm is None or dest_cm > IR_SIMPLE_DEST_SHARP_CAUTION_CM + 2.0:
                speed = IR_SIMPLE_STRAFE_FAST_MPS
            elif dest_cm <= IR_SIMPLE_DEST_SHARP_CAUTION_CM:
                speed = IR_SIMPLE_STRAFE_SLOW_MPS

            if not self.drive_speed_resilient(
                x=0.0,
                y=sign * speed,
                z=self.lateral_yaw_hold_command(target_yaw),
                timeout=DRIVE_COMMAND_TIMEOUT,
                label="IR+SHARP STRAFE",
            ):
                stop_reason = "DRIVE_FAIL"
                break

            time.sleep(CONTROL_DT)

        self.safe_stop()

        # Do not pay a stationary PID settle on every tiny corner nudge.  Only
        # re-align when the lateral maneuver actually displaced yaw visibly.
        post_err = self.yaw_error_deg(target_yaw)
        if post_err is None or abs(post_err) > LATERAL_RECOVERY_REALIGN_TRIGGER_DEG:
            self.align_heading_stationary(
                target_yaw,
                timeout_sec=min(0.45, POST_MOVE_ALIGN_TIMEOUT_SEC),
                tolerance_deg=max(0.75, PRE_MOVE_ALIGN_TOL_DEG),
                settle_sec=0.04,
            )

        l3, r3, lr3, rr3 = self.read_ir_filtered(samples=2)
        source_still_low = l3 if sign > 0.0 else r3

        # Sharp-confirmed clearance is accepted even if the angled digital IR
        # remains LOW for a moment; front ToF + Sharp continue guarding motion.
        source_sharp_clear = bool(
            last_source_cm is not None
            and last_source_cm >= IR_SIMPLE_SOURCE_SHARP_CLEAR_CM
            and moved >= IR_SIMPLE_SOURCE_SHARP_CLEAR_MIN_MOVE_M
        )
        if not source_still_low:
            stop_reason = "IR_CLEAR"
            result = "CLEAR"
        elif source_sharp_clear:
            stop_reason = "SOURCE_SHARP_CLEAR"
            result = "CLEAR"
        else:
            result = "HOLD"

        print(
            "[IR+SHARP {} DONE] moved={:.3f}m dir={} IR=({},{}) "
            "srcSharp={} dstSharp={} result={} reason={}".format(
                context, moved, direction_name, lr3, rr3,
                "NA" if last_source_cm is None else "{:.1f}".format(last_source_cm),
                "NA" if last_dest_cm is None else "{:.1f}".format(last_dest_cm),
                result, stop_reason,
            )
        )
        return result

    def ir_corner_entry_trim(self, target_yaw):
        """After a real turn, apply the same single simple IR rule once."""
        if not self.ir_corner_trim_pending:
            return True
        self.ir_corner_trim_pending = False
        status = self.ir_simple_side_clear(target_yaw, context="CORNER")
        return status in ("CLEAR", "BOTH", "HOLD")

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
                state = "OPEN" if tof_is_open_from_center(mm) else "WALL"
            center_mm = tof_topology_center_mm(mm)
            print(f"  [NODE PROBE] {label:<5} yaw={yaw_deg:+5.0f} ToF(raw)={mm} center={center_mm} -> {state}")
        self.gimbal_front_down(force=True)
        self.last_stop_probe = dict(result)

        front = result.get("FRONT")
        front_wall = front is not None and not tof_is_open_from_center(front)
        states = {
            k: ("UNKNOWN" if v is None else "OPEN" if tof_is_open_from_center(v) else "WALL")
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

        self.fault(
            "MOVE EXCEPTION HOLD",
            "mid-edge pose not source/destination: {}".format(detail),
            "STOP in place; automatic reverse disabled",
        )
        return self.MOVE_POSE_UNCERTAIN

    def _motion_profile(self, profile):
        """Return speed/braking parameters for one cell traversal.

        EXPLORE is used only when entering a previously unvisited cell.
        KNOWN_FAST/BACKTRACK_FAST are used on OPEN edges already proven by DFS.
        The fast profile keeps the same hard-stop threshold but starts braking
        earlier to preserve stopping margin at 0.35 m/s.
        """
        name = str(profile or "EXPLORE").upper()
        fast = name in ("KNOWN_FAST", "BACKTRACK_FAST", "ROUND2_FAST")
        cruise = DFS_KNOWN_SPEED_MPS if fast else DFS_EXPLORE_SPEED_MPS
        approach_min = DFS_KNOWN_APPROACH_MIN_MPS if fast else DFS_EXPLORE_APPROACH_MIN_MPS
        approach_zone = DFS_KNOWN_APPROACH_SLOW_M if fast else DFS_EXPLORE_APPROACH_SLOW_M
        brake_start = FRONT_BRAKE_START_FAST_MM if fast else FRONT_BRAKE_START_MM
        crawl_start = FRONT_CRAWL_START_FAST_MM if fast else FRONT_CRAWL_START_MM
        timeout = max(4.5, (CELL_LENGTH_M / max(0.05, cruise)) * 3.2)
        return {
            "name": name, "fast": fast, "cruise": float(cruise),
            "approach_min": float(approach_min), "approach_zone": float(approach_zone),
            "brake_start_mm": float(brake_start), "crawl_start_mm": float(crawl_start),
            "timeout_sec": float(timeout),
        }

    def move_one_cell(self, source_cell, abs_dir, motion_profile="EXPLORE"):
        """Public exception boundary for one physical edge traversal."""
        abs_dir = int(abs_dir) % 4
        profile = self._motion_profile(motion_profile)
        start_pos = self.current_position(fresh=False)
        target_yaw = self.desired_yaw_for_heading(abs_dir)
        learned_distance = self.remembered_edge_distance(source_cell, abs_dir)
        target_distance = learned_distance if learned_distance is not None else CELL_LENGTH_M
        try:
            return self._move_one_cell_impl(source_cell, abs_dir, profile)
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

    def _move_one_cell_impl(self, source_cell, abs_dir, profile=None):
        if not self.pose_trusted or not self.running:
            return self.MOVE_STOPPED

        profile = dict(profile or self._motion_profile("EXPLORE"))
        cruise_speed = float(profile.get("cruise", DFS_EXPLORE_SPEED_MPS))
        approach_min_speed = float(profile.get("approach_min", DFS_EXPLORE_APPROACH_MIN_MPS))
        approach_slow_m = float(profile.get("approach_zone", DFS_EXPLORE_APPROACH_SLOW_M))
        brake_start_mm = float(profile.get("brake_start_mm", FRONT_BRAKE_START_MM))
        crawl_start_mm = float(profile.get("crawl_start_mm", FRONT_CRAWL_START_MM))
        move_timeout_sec = float(profile.get("timeout_sec", MAX_CELL_TIME_SEC))
        print("[MOVE PROFILE] {} cruise={:.2f}m/s approach={:.2f}m brake={:.0f}/{:.0f}mm".format(
            profile.get("name", "EXPLORE"), cruise_speed, approach_slow_m,
            brake_start_mm, crawl_start_mm,
        ))

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
        prep_err = self.yaw_error_deg(expected_yaw)
        if prep_err is None or abs(prep_err) > PRE_MOVE_ALIGN_TOL_DEG:
            if not self.align_heading_stationary(
                expected_yaw, timeout_sec=PRE_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.07
            ):
                self.fault("MOVE PREP", f"yaw not aligned for {DIR_NAMES[abs_dir]}", "do not translate")
                return self.MOVE_BLOCKED_RETURNED

        goto_ok = self.gimbal_front_down(force=False)
        # gimbal_goto() may soft-accept a topology-safe pitch (e.g. -15deg).
        # Translation is stricter, so ALWAYS check the motion envelope even if
        # the goto call returned True.
        if not self.gimbal_front_safe_for_motion():
            if self.recover_gimbal_front():
                self.fault("MOVE PREP", "gimbal front velocity-recovered", "continue")
            else:
                p,y=self.current_gimbal_relative()
                self.fault(
                    "MOVE PREP",
                    f"gimbal front unsafe after goto={goto_ok} p={p} y={y}",
                    "do not translate",
                )
                return self.MOVE_BLOCKED_RETURNED
        elif not goto_ok:
            p,y=self.current_gimbal_relative()
            self.fault(
                "MOVE PREP",
                f"goto action unhappy but live motion pose is safe (p={p:+.1f}, y={y:+.1f})",
                "continue with forward ToF guard",
            )

        start_pos = self.current_position()
        if start_pos is None:
            self.fault("MOVE", "no fresh position at cell start", "do not translate")
            return self.MOVE_BLOCKED_RETURNED

        target_yaw = expected_yaw
        if self.ir_corner_trim_pending:
            self.ir_corner_entry_trim(target_yaw)
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
        side_escape_ignore_until = 0.0
        ir_retrigger_ignore_until = 0.0

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
                self.fault(
                    "MOVE HOLD", "position telemetry remained stale",
                    "STOP in place; automatic reverse disabled",
                )
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
                self.fault(
                    "MOVE HOLD", "attitude telemetry unavailable during translation",
                    "STOP in place; automatic reverse disabled",
                )
                return self.MOVE_POSE_UNCERTAIN
            if abs(yaw_err) >= MOVE_YAW_ABORT_ERROR_DEG:
                if yaw_bad_since is None:
                    yaw_bad_since = now
                elif now - yaw_bad_since >= MOVE_YAW_ABORT_CONFIRM_SEC:
                    self.safe_stop()
                    self.fault(
                        "MOVE YAW",
                        "diverged by {:+.1f}deg at d={:.3f}m".format(yaw_err, traveled),
                        "STOP/re-align; automatic reverse disabled",
                    )
                    if self.align_heading_stationary(
                        target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                        tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08,
                    ):
                        self.pid_straight.reset()
                        yaw_bad_since = None
                        continue
                    return self.MOVE_POSE_UNCERTAIN
            else:
                yaw_bad_since = None

            if traveled >= target_distance:
                self.safe_stop()
                # Remove the small yaw residual created by wheel inertia before the
                # next node scan.  Failure is logged but arrival position remains valid.
                post_err = self.yaw_error_deg(target_yaw)
                aligned = True
                if post_err is None or abs(post_err) > POST_MOVE_ALIGN_TOL_DEG:
                    aligned = self.align_heading_stationary(
                        target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC,
                        tolerance_deg=POST_MOVE_ALIGN_TOL_DEG, settle_sec=0.05
                    )
                if not aligned:
                    self.fault("MOVE POST", "arrived but yaw final-settle timed out", "next turn/scan will realign")
                self.last_move_distance_m = traveled
                print(
                    f"[MOVE OK] {source_cell}->{neighbor(source_cell, abs_dir)} "
                    f"d={traveled:.3f}m target={target_distance:.3f}m "
                    f"yaw_err={self.yaw_error_deg(target_yaw)} sec={time.monotonic()-start_t:.2f}"
                )
                return self.MOVE_ARRIVED

            if now - start_t >= move_timeout_sec:
                self.safe_stop()
                if traveled >= target_distance * CELL_SUCCESS_FRACTION:
                    self.align_heading_stationary(target_yaw, timeout_sec=POST_MOVE_ALIGN_TIMEOUT_SEC, tolerance_deg=PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08)
                    self.last_move_distance_m = traveled
                    self.fault("MOVE TIMEOUT", f"near node at {traveled:.3f}m", "accept node")
                    return self.MOVE_ARRIVED
                tof_timeout = self.latest_tof(fresh=True)
                if tof_timeout is not None and traveled >= SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M:
                    probe = self.stopped_front_topology_probe(traveled, tof_timeout, target_yaw)
                    if probe.get("valid", 0) >= 2:
                        self.last_move_distance_m = max(SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
                        self.last_stop_probe = dict(probe.get("rays", {}))
                        self.fault(
                            "MOVE TIMEOUT HOLD",
                            "timeout at {:.3f}m".format(traveled),
                            "accept stationary physical node; no retreat",
                        )
                        return self.MOVE_ARRIVED
                self.fault(
                    "MOVE TIMEOUT HOLD", "timeout at {:.3f}m".format(traveled),
                    "STOP in place; automatic reverse disabled",
                )
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
                            self.fault(
                                "MOVE HOLD", "front ToF unavailable during motion",
                                "STOP in place; automatic reverse disabled",
                            )
                            return self.MOVE_POSE_UNCERTAIN
                    tof = fresh
                    stale_tof_since = None
                else:
                    time.sleep(CONTROL_DT)
                    continue
            else:
                stale_tof_since = None

            if tof <= FRONT_STOP_SCAN_MM:
                # On a previously proven OPEN edge, reaching the expected node is
                # already supported by odometry + stored topology.  Do not spend
                # competition time rescanning L/F/R merely because a wall is close
                # beyond that known node.  An EARLY close obstacle still falls
                # through to the full stationary probe below.
                if bool(profile.get("fast")) and traveled >= target_distance * CELL_SUCCESS_FRACTION:
                    self.safe_stop()
                    self.last_move_distance_m = traveled
                    self.align_heading_stationary(
                        target_yaw, timeout_sec=0.70,
                        tolerance_deg=max(0.45, POST_MOVE_ALIGN_TOL_DEG), settle_sec=0.05,
                    )
                    print(
                        "[FAST NODE ACCEPT] {} d={:.3f}/{:.3f}m ToF={:.0f}mm -> no repeat topology scan".format(
                            profile.get("name", "KNOWN_FAST"), traveled, target_distance, tof
                        )
                    )
                    return self.MOVE_ARRIVED

                # Unknown/exploration edge or an unexpectedly EARLY obstruction:
                # stop and rotate the Gimbal ToF to determine whether this is a
                # legitimate dead-end/corner node or a mid-edge obstacle.
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

                # V9: front-wall handling never consults IR and never reverses.
                # Stop/probe in place; ToF topology owns the decision.
                if traveled >= SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M and probe.get("valid", 0) >= 2:
                    self.last_move_distance_m = max(SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
                    self.last_stop_probe = dict(probe.get("rays", {}))
                    self.fault(
                        "FRONT HOLD NODE",
                        "ToF={:.0f}mm at {:.3f}m; reverse gate not met".format(tof, traveled),
                        "accept stopped physical node; no retreat",
                    )
                    return self.MOVE_ARRIVED
                self.fault(
                    "FRONT HOLD",
                    "close wall too early at {:.3f}m and reverse gate not met".format(traveled),
                    "STOP in place; no automatic retreat",
                )
                return self.MOVE_POSE_UNCERTAIN

            # Digital IR policy V9: one rule only.
            #
            # LEFT LOW only  -> pure strafe RIGHT until LEFT clears.
            # RIGHT LOW only -> pure strafe LEFT  until RIGHT clears.
            # BOTH LOW       -> do not guess a direction; no reverse/replan.
            #
            # No Gimbal route scan, no topology classification, no source-cell
            # retreat is allowed from an IR event.
            l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=1)
            if (l_low ^ r_low) and now >= ir_retrigger_ignore_until:
                self.safe_stop()
                ir_t0 = time.monotonic()
                ir_status = self.ir_simple_side_clear(target_yaw, context="MOVE")
                start_t += max(0.0, time.monotonic() - ir_t0)
                self.pid_straight.reset()
                self.sharp_authority = None
                if ir_status == "CLEAR":
                    # Prevent a sticky angled IR from immediately causing a second
                    # stop a few centimetres later.  Sharp/ToF remain live during
                    # this short cooldown.
                    ir_retrigger_ignore_until = time.monotonic() + IR_SIMPLE_RETRIGGER_COOLDOWN_SEC
                    continue
                print(
                    "[IR SIMPLE MOVE] source IR still LOW -> no second policy; "
                    "resume cautiously under Sharp/ToF"
                )
            elif l_low and r_low:
                print(
                    "[IR SIMPLE MOVE] BOTH LOW IR=({},{}) -> no lateral guess, "
                    "no reverse; Sharp/ToF own motion".format(l_raw, r_raw)
                )

            left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()
            if (
                left_cm is not None and right_cm is not None
                and left_cm <= SHARP_EMERGENCY_CM and right_cm <= SHARP_EMERGENCY_CM
            ):
                self.safe_stop()
                probe = self.stopped_front_topology_probe(traveled, tof, target_yaw)
                if traveled >= target_distance * CELL_SUCCESS_FRACTION or probe.get("node_like"):
                    self.last_move_distance_m = max(SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
                    return self.MOVE_ARRIVED
                self.fault(
                    "SHARP HOLD",
                    "both Sharp at emergency floor at d={:.3f}m".format(traveled),
                    "STOP/PROBE; no full retreat",
                )
                side_escape_ignore_until = time.monotonic() + SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC
                continue

            raw_y_cmd, authority = self.sharp_lateral_command(left_cm, right_cm)
            remaining = max(0.0, target_distance - traveled)
            if remaining <= approach_slow_m:
                ratio = remaining / max(1e-6, approach_slow_m)
                x_cmd = max(approach_min_speed, cruise_speed * ratio)
            else:
                x_cmd = cruise_speed

            # Front-distance brake envelope.  The command now decreases smoothly
            # as the wall approaches instead of staying at 0.10 m/s until the
            # hard-stop threshold.  The final stop is handled above by the
            # stationary topology probe.
            if tof < brake_start_mm:
                if tof <= crawl_start_mm:
                    tof_cmd = FRONT_MIN_BRAKE_SPEED_MPS
                else:
                    span = max(1.0, brake_start_mm - crawl_start_mm)
                    alpha = clamp((tof - crawl_start_mm) / span, 0.0, 1.0)
                    tof_cmd = FRONT_MIN_BRAKE_SPEED_MPS + alpha * (cruise_speed - FRONT_MIN_BRAKE_SPEED_MPS)
                x_cmd = min(x_cmd, tof_cmd)

            # A genuinely close side wall is handled as a pure lateral escape.
            # Never mix x forward with a large y correction; that was the main
            # source of the visible diagonal motion immediately before stopping.
            # Hysteretic side-escape state: once armed at <= trigger, remain in
            # pure-strafe mode until that SAME side reaches the clear threshold.
            escape_dist = None
            if side_escape_sign == 0.0 and now >= side_escape_ignore_until:
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
                destination = right_cm if side_escape_sign > 0.0 else left_cm
                destination_tight = bool(
                    destination is not None
                    and destination <= SHARP_SIDE_ESCAPE_TRIGGER_CM
                )
                escape_timeout = bool(
                    side_escape_since is not None
                    and now - side_escape_since > SHARP_SIDE_ESCAPE_MAX_SEC
                )
                if destination_tight or escape_timeout:
                    self.safe_stop()
                    reason = "destination side tight" if destination_tight else "side-wall escape timed out"
                    source_now = left_cm if side_escape_sign > 0.0 else right_cm

                    # V20.3 fast release: if the timeout side has already improved
                    # out of the severe-danger range and FRONT is clearly open,
                    # do not spend several seconds on a full stationary Gimbal
                    # L/F/R probe.  Resume under normal Sharp+ToF control.
                    fast_release = bool(
                        escape_timeout
                        and not destination_tight
                        and source_now is not None
                        and source_now >= SHARP_SIDE_ESCAPE_FAST_RELEASE_CM
                        and tof is not None
                        and tof > max(FRONT_BRAKE_START_MM, TOF_OPEN_THRESHOLD_MM)
                    )
                    if fast_release:
                        side_escape_sign = 0.0
                        side_escape_since = None
                        side_escape_ignore_until = time.monotonic() + SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC
                        last_y_cmd = 0.0
                        yaw_after_escape = self.yaw_error_deg(target_yaw)
                        if (
                            yaw_after_escape is None
                            or abs(yaw_after_escape) > LATERAL_RECOVERY_REALIGN_TRIGGER_DEG
                        ):
                            self.align_heading_stationary(
                                target_yaw, timeout_sec=0.40,
                                tolerance_deg=max(0.75, PRE_MOVE_ALIGN_TOL_DEG),
                                settle_sec=0.04,
                            )
                        self.fault(
                            "SIDE FAST RELEASE",
                            "{} at d={:.3f}m Sharp L={} R={} front={:.0f}mm".format(
                                reason, traveled, left_cm, right_cm, tof
                            ),
                            "side clearance improved; skip node probe / resume",
                        )
                        continue

                    probe = self.stopped_front_topology_probe(traveled, tof, target_yaw)
                    if traveled >= target_distance * CELL_SUCCESS_FRACTION or probe.get("node_like"):
                        self.last_move_distance_m = max(SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
                        self.fault(
                            "SIDE HOLD NODE",
                            "{} at d={:.3f}m Sharp L={} R={}".format(
                                reason, traveled, left_cm, right_cm
                            ),
                            "accept stopped node; no retreat",
                        )
                        return self.MOVE_ARRIVED
                    side_escape_sign = 0.0
                    side_escape_since = None
                    side_escape_ignore_until = time.monotonic() + SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC
                    last_y_cmd = 0.0
                    self.gimbal_front_down(force=True)
                    self.fault(
                        "SIDE HOLD",
                        "{} at d={:.3f}m Sharp L={} R={}".format(
                            reason, traveled, left_cm, right_cm
                        ),
                        "STOP then continue cautiously; no retreat",
                    )
                    continue
                x_cmd = 0.0
                y_cmd = side_escape_sign * SHARP_SIDE_ESCAPE_SPEED_MPS
                last_y_cmd = y_cmd
            else:
                y_cmd = self.shape_lateral_command(raw_y_cmd, x_cmd, remaining, last_y_cmd)
                last_y_cmd = y_cmd

            z_cmd = (
                self.lateral_yaw_hold_command(target_yaw)
                if side_escape_sign != 0.0
                else self.yaw_hold_command(target_yaw)
            )
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
                self.fault(
                    "MOVE HOLD", "drive command exception after telemetry recovery failed",
                    "STOP in place; automatic reverse disabled",
                )
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
                if tof_is_open_from_center(mm):
                    state = "OPEN"
                    ordered_open.append(abs_dir)
                else:
                    state = "WALL"
            self.set_edge_state(cell, abs_dir, state)
            center_mm = tof_topology_center_mm(mm)
            print(f"  [NODE MAP] {label:<5} ToF(raw)={mm} center={center_mm} -> {state} ({DIR_NAMES[abs_dir]})")

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
            elif tof_is_open_from_center(mm):
                state = "OPEN"
                ordered_open.append(abs_dir)
            else:
                state = "WALL"
            self.set_edge_state(cell, abs_dir, state)
            center_mm = tof_topology_center_mm(mm)
            print(f"  {label:<5} yaw={yaw_deg:+5.0f} ToF(raw)={mm} center={center_mm} -> {state} ({DIR_NAMES[abs_dir]})")

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
                state = "OPEN" if tof_is_open_from_center(mm) else "WALL"
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
            elif tof_is_open_from_center(mm):
                self.mark_traversed_open(cell, back_dir)
                if back_dir not in ordered_open:
                    ordered_open.append(back_dir)
            else:
                self.set_edge_state(cell, back_dir, "WALL")
            print(f"  BACK  chassis-180 ToF={mm} -> {self.edge_state.get((tuple(cell), back_dir))} ({DIR_NAMES[back_dir]})")

        self.cell_scan_mm[tuple(cell)] = scan
        # Deduplicate while preserving scan/parent order.
        seen = set()
        ordered_open = [d for d in ordered_open if not (d in seen or seen.add(d))]
        self.open_dirs[tuple(cell)] = ordered_open
        print("  OPEN:", [DIR_NAMES[d] for d in ordered_open])

        # TARGET SERVICE ISOLATION:
        # mapping/topology has already been committed for this stationary node.
        # Any camera, detector, aim or fire failure is contained and cannot make
        # DFS forget the node or move the chassis off its anchor.
        try:
            self.target_system.scan_cell(cell)
        except Exception as exc:
            self.fault(
                "TARGET SCAN",
                f"cell={cell}: {type(exc).__name__}: {exc}",
                "skip targets at this node / continue DFS",
            )

        self.gimbal_front_down(force=True)
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
            "tof_open_threshold_reference": "robot_center_planar",
            "tof_forward_from_center_m": TOF_FORWARD_FROM_CENTER_M,
            "tof_raw_open_equivalent_mm_approx": max(
                0.0, TOF_OPEN_THRESHOLD_MM - TOF_FORWARD_FROM_CENTER_M*1000.0
            ),
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
            # V20: during the live run JSON is the crash-safe checkpoint.  ASCII/SVG
            # rendering is deferred until final save so DFS does not repeatedly spend
            # competition time regenerating presentation files after every move.
            if final:
                atomic_write_text(MAP_LATEST_ASCII, self.render_ascii_map())
                atomic_write_text(MAP_LATEST_SVG, self.render_svg_map())
            print(f"[MAP SAVE] visited={len(self.visited)} complete={bool(final and self.map_complete)} -> {MAP_LATEST_JSON}")
            return True
        except Exception as exc:
            self.fault("MAP SAVE", f"{type(exc).__name__}: {exc}", "keep mission state in memory")
            return False

    # --------------------------------------------------------
    # ROUND-1 SNAPSHOT / ROUND-2 SHORTEST ATTACK
    # --------------------------------------------------------
    def _build_round1_fire_hints(self):
        hints = []
        for target in self.target_system.targets:
            if not isinstance(target, dict):
                continue
            if not str(target.get("fire_status") or "").startswith("FIRED_"):
                continue
            anchor = target.get("fire_anchor")
            if not isinstance(anchor, dict):
                anchor = {
                    "cell": target.get("source_cell"),
                    "heading_index": DIR_NAMES.index(target.get("heading")) if target.get("heading") in DIR_NAMES else 0,
                    "heading": target.get("heading"),
                    "sector": target.get("scan_sector"),
                    "detected_sweep_yaw_deg": target.get("detected_sweep_yaw_deg"),
                    "lock_yaw_deg": target.get("lock_yaw_deg"),
                    "range_mm": target.get("fire_range_mm"),
                    "target_estimated_grid_xy": target.get("estimated_grid_xy"),
                    "color": target.get("color"),
                    "shape": target.get("shape"),
                    "source_target_id": target.get("id"),
                    "fired_at": target.get("fired_at"),
                }
            h = dict(anchor)
            h["color"] = str(h.get("color") or target.get("color") or "").upper()
            h["shape"] = str(h.get("shape") or target.get("shape") or "").upper()
            h["fire_status"] = target.get("fire_status")
            h["fire_mode"] = target.get("fire_mode")
            h["fire_times"] = target.get("fire_times")
            if isinstance(h.get("cell"), (list, tuple)) and len(h.get("cell")) >= 2:
                hints.append(h)
        return hints

    def save_round1_attack_memory(self):
        """Atomically freeze map + only PROVEN fired positions for Round 2."""
        try:
            hints = self._build_round1_fire_hints()
            payload = {
                "schema": "robomaster_round1_attack_memory",
                "version": 1,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "map_complete": bool(self.map_complete),
                "start_pose_rule": {"cell": [0, 0], "heading": "N"},
                "map": self.build_map_payload(final=self.map_complete),
                "selected_target_classes": [
                    {"color": c, "shape": sh} for c, sh in self.get_target_selection()
                ],
                "fire_hint_count": len(hints),
                "fire_hints": hints,
            }
            atomic_write_text(
                ROUND1_ATTACK_MEMORY_JSON,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            print(
                "[ROUND1 MEMORY] map_cells={} fired_hints={} complete={} -> {}".format(
                    len(self.visited), len(hints), self.map_complete, ROUND1_ATTACK_MEMORY_JSON
                )
            )
            return True
        except Exception as exc:
            self.fault(
                "ROUND1 MEMORY", f"{type(exc).__name__}: {exc}",
                "keep latest_map/latest_targets as fallback",
            )
            return False

    def load_round1_attack_memory(self):
        try:
            payload = json.loads(ROUND1_ATTACK_MEMORY_JSON.read_text(encoding="utf-8"))
            if payload.get("schema") != "robomaster_round1_attack_memory":
                raise ValueError("wrong round1 memory schema")
            mp = payload.get("map") or {}
            visited = {tuple(int(v) for v in c[:2]) for c in mp.get("visited", []) if len(c) >= 2}
            if not visited:
                raise ValueError("round1 memory has no visited cells")

            edges = {}
            for row in mp.get("edges", []):
                cell = row.get("cell")
                if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                    continue
                d = row.get("dir_index")
                if d is None and row.get("dir") in DIR_NAMES:
                    d = DIR_NAMES.index(row.get("dir"))
                if d is None:
                    continue
                edges[((int(cell[0]), int(cell[1])), int(d) % 4)] = str(row.get("state") or "UNKNOWN")

            open_dirs = {}
            for key, dirs in (mp.get("open_dirs") or {}).items():
                try:
                    xs, ys = key.split(",", 1)
                    cell = (int(xs), int(ys))
                except Exception:
                    continue
                out = []
                for d in dirs or []:
                    if d in DIR_NAMES:
                        out.append(DIR_NAMES.index(d))
                open_dirs[cell] = out

            edge_travel = {}
            for row in mp.get("edge_travel_m", []):
                cell = row.get("cell")
                dname = row.get("dir")
                if not isinstance(cell, (list, tuple)) or len(cell) < 2 or dname not in DIR_NAMES:
                    continue
                try:
                    edge_travel[((int(cell[0]), int(cell[1])), DIR_NAMES.index(dname))] = float(row.get("distance_m"))
                except Exception:
                    pass

            self.visited = visited
            self.edge_state = edges
            self.open_dirs = open_dirs
            self.edge_travel_m = edge_travel
            self.cell_scan_mm = {}
            for key, value in (mp.get("cell_scan_mm") or {}).items():
                try:
                    xs, ys = key.split(",", 1)
                    self.cell_scan_mm[(int(xs), int(ys))] = value
                except Exception:
                    pass
            self.root = (0, 0)
            self.current = self.root
            self.heading = 0
            self.map_complete = bool(payload.get("map_complete", mp.get("complete", False)))
            self.round1_memory = payload
            self.round2_hints = [dict(h) for h in payload.get("fire_hints", []) if isinstance(h, dict)]
            print(
                "[ROUND2 LOAD] cells={} hints={} map_complete={} <- {}".format(
                    len(self.visited), len(self.round2_hints), self.map_complete, ROUND1_ATTACK_MEMORY_JSON
                )
            )
            return True
        except FileNotFoundError:
            self.fault(
                "ROUND2 LOAD", f"missing {ROUND1_ATTACK_MEMORY_JSON}",
                "run ROUND1 first",
            )
        except Exception as exc:
            self.fault(
                "ROUND2 LOAD", f"{type(exc).__name__}: {exc}",
                "do not move on an unverified map",
            )
        return False

    def _round2_hint_selected(self, hint):
        return self.target_allowed(hint.get("color"), hint.get("shape"))

    def _round2_route_distance(self, a, b):
        route = self.find_visited_route(tuple(a), tuple(b))
        return (float("inf"), None) if not route else (float(len(route) - 1), route)

    def _round2_anchor_order(self, start, anchors):
        """Order unique firing cells by known-path distance.

        Uses exact Held-Karp open-path optimization for up to 11 unique firing
        cells; above that, a nearest-neighbour pass avoids exponential startup
        time.  Distances come from the learned OPEN-edge graph, never Euclidean
        shortcuts through walls.
        """
        start = tuple(start)
        anchors = list(dict.fromkeys(tuple(a) for a in anchors if tuple(a) != start))
        if not anchors:
            return []

        points = [start] + anchors
        dist = {}
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                if i == j:
                    dist[(i, j)] = 0.0
                elif (j, i) in dist:
                    dist[(i, j)] = dist[(j, i)]
                else:
                    d, _ = self._round2_route_distance(a, b)
                    dist[(i, j)] = d

        reachable = [i for i in range(1, len(points)) if math.isfinite(dist[(0, i)])]
        if len(reachable) != len(anchors):
            bad = [points[i] for i in range(1, len(points)) if i not in reachable]
            self.fault("ROUND2 PLAN", f"unreachable fire anchors={bad}", "skip unreachable anchors")
            anchors = [points[i] for i in reachable]
            points = [start] + anchors
            if not anchors:
                return []
            # Recompute compact distance table after dropping unreachable anchors.
            dist = {}
            for i, a in enumerate(points):
                for j, b in enumerate(points):
                    if i == j:
                        dist[(i, j)] = 0.0
                    elif (j, i) in dist:
                        dist[(i, j)] = dist[(j, i)]
                    else:
                        d, _ = self._round2_route_distance(a, b)
                        dist[(i, j)] = d

        n = len(anchors)
        if n <= ROUND2_EXACT_ORDER_MAX_ANCHORS:
            # dp[(mask,last)] = (distance, previous_last) ; last is 0..n-1.
            dp = {}
            for j in range(n):
                d = dist[(0, j + 1)]
                if math.isfinite(d):
                    dp[(1 << j, j)] = (d, None)
            for mask in range(1, 1 << n):
                for last in range(n):
                    state = dp.get((mask, last))
                    if state is None:
                        continue
                    base = state[0]
                    for nxt in range(n):
                        bit = 1 << nxt
                        if mask & bit:
                            continue
                        step = dist[(last + 1, nxt + 1)]
                        if not math.isfinite(step):
                            continue
                        nm = mask | bit
                        nd = base + step
                        old = dp.get((nm, nxt))
                        if old is None or nd < old[0]:
                            dp[(nm, nxt)] = (nd, last)
            full = (1 << n) - 1
            ends = [(v[0], last) for (mask, last), v in dp.items() if mask == full]
            if ends:
                _, last = min(ends)
                order_idx = []
                mask = full
                while last is not None:
                    order_idx.append(last)
                    prev = dp[(mask, last)][1]
                    mask &= ~(1 << last)
                    last = prev
                order = [anchors[i] for i in reversed(order_idx)]
                print("[ROUND2 PLAN] exact shortest anchor order={}".format(order))
                return order

        # Deterministic nearest-neighbour fallback for unusually many anchors.
        remaining = set(anchors)
        cur = start
        order = []
        while remaining:
            candidates = []
            for a in remaining:
                d, _ = self._round2_route_distance(cur, a)
                if math.isfinite(d):
                    candidates.append((d, a))
            if not candidates:
                break
            _, nxt = min(candidates, key=lambda x: (x[0], x[1][1], x[1][0]))
            order.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        print("[ROUND2 PLAN] nearest-neighbour anchor order={}".format(order))
        return order

    def _save_round2_result(self, started_at, attempts, successes, failed_hints, reason):
        try:
            payload = {
                "schema": "robomaster_round2_attack_result",
                "version": 1,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "reason": str(reason),
                "attempts": int(attempts),
                "successes": int(successes),
                "failed_hints": failed_hints,
                "current_cell": list(self.current),
                "selected_target_classes": [
                    {"color": c, "shape": sh} for c, sh in self.get_target_selection()
                ],
            }
            self.round2_result = payload
            atomic_write_text(ROUND2_RESULT_JSON, json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            self.fault("ROUND2 SAVE", f"{type(exc).__name__}: {exc}", "continue cleanup")

    def run_round2_attack(self):
        started_iso = datetime.now().isoformat(timespec="seconds")
        t0 = time.monotonic()
        if not self.load_round1_attack_memory():
            self.safe_stop()
            self._save_round2_result(started_iso, 0, 0, [], "round1 memory unavailable")
            return False

        selected = [h for h in self.round2_hints if self._round2_hint_selected(h)]
        if not selected:
            print("[ROUND2] no proven Round-1 fired hints match the current 16-target selection")
            self.safe_stop()
            self._save_round2_result(started_iso, 0, 0, [], "no selected fired hints")
            return True

        # Group hints by proven logical firing cell.  One route visit can service
        # several targets at the same node without redundant driving.
        hints_by_cell = {}
        for h in selected:
            cell = h.get("cell")
            if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                continue
            c = (int(cell[0]), int(cell[1]))
            if c not in self.visited:
                self.fault("ROUND2 HINT", f"hint cell {c} absent from map", "skip hint")
                continue
            hints_by_cell.setdefault(c, []).append(h)

        order = self._round2_anchor_order(self.current, list(hints_by_cell.keys()))
        # The root may itself contain targets; service it first without driving.
        if self.current in hints_by_cell:
            order = [self.current] + order

        attempts = 0
        successes = 0
        failed = []
        print(
            "\n[ROUND2] START shortest attack: hints={} anchor_cells={} hard_limit={:.0f}s".format(
                len(selected), len(hints_by_cell), ROUND2_HARD_LIMIT_SEC
            )
        )

        for anchor in order:
            if not self.running or not self.pose_trusted:
                break
            if time.monotonic() - t0 >= ROUND2_HARD_LIMIT_SEC:
                print("[ROUND2 DEADLINE] stopping before 5-minute limit")
                break

            if tuple(self.current) != tuple(anchor):
                route = self.find_visited_route(self.current, anchor)
                print("[ROUND2 ROUTE] {} -> {} route={}".format(self.current, anchor, route))
                if not route or not self.navigate_known_route(route):
                    self.fault("ROUND2 ROUTE", f"cannot reach anchor {anchor}", "skip this anchor")
                    for h in hints_by_cell.get(anchor, []):
                        failed.append(dict(h, failure="route_unavailable"))
                    continue

            # Service the proven shots at this anchor.  Sort by heading then sector
            # to reduce unnecessary chassis turns.
            local_hints = sorted(
                hints_by_cell.get(anchor, []),
                key=lambda h: (int(h.get("heading_index", 0)) % 4, str(h.get("sector") or "")),
            )
            for hint in local_hints:
                if time.monotonic() - t0 >= ROUND2_HARD_LIMIT_SEC:
                    break
                heading_idx = int(hint.get("heading_index", 0)) % 4
                if self.heading != heading_idx:
                    if not self.turn_to_direction(heading_idx):
                        failed.append(dict(hint, failure="turn_failed"))
                        continue

                ok = False
                for hint_attempt in range(1, ROUND2_MAX_HINT_ATTEMPTS + 1):
                    if time.monotonic() - t0 >= ROUND2_HARD_LIMIT_SEC:
                        break
                    attempts += 1
                    print(
                        "[ROUND2 ATTACK] attempt {}/{} {} {} @ cell={} heading={}".format(
                            hint_attempt, ROUND2_MAX_HINT_ATTEMPTS,
                            hint.get("color"), hint.get("shape"), anchor, DIR_NAMES[self.heading]
                        )
                    )
                    if self.target_system.scan_round2_hint(hint):
                        ok = True
                        successes += 1
                        break
                if not ok:
                    failed.append(dict(hint, failure="target_not_reacquired_or_not_fired"))

        self.safe_stop()
        elapsed = time.monotonic() - t0
        if elapsed >= ROUND2_HARD_LIMIT_SEC:
            reason = "deadline_guard"
        elif not self.running:
            reason = "stopped"
        elif failed:
            reason = "completed_with_failed_hints"
        else:
            reason = "completed"
        print(
            "[ROUND2 DONE] success={}/{} attempts={} elapsed={:.1f}s failed={}".format(
                successes, len(selected), attempts, elapsed, len(failed)
            )
        )
        self._save_round2_result(started_iso, attempts, successes, failed, reason)
        return not failed and successes >= len(selected)

    def run_selected_mission(self):
        if self.mission_mode == "ROUND2":
            return self.run_round2_attack()
        result = self.run_dfs()
        self.save_round1_attack_memory()
        return result

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

        result = self.move_one_cell(cell, direction, motion_profile="EXPLORE")
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
            result = self.move_one_cell(src, d, motion_profile="KNOWN_FAST")
            if result != self.MOVE_ARRIVED:
                if result == self.MOVE_BLOCKED_RETURNED:
                    self.current = src
                return False
            self.current = dst
            self.mark_traversed_open(src, d)
            if self.last_move_distance_m is not None:
                self.remember_edge_distance(src, d, self.last_move_distance_m)
        return True

    def find_fastest_known_route(self, start, goal):
        """Dijkstra over confirmed visited OPEN edges, including turn cost."""
        import heapq
        start = tuple(start); goal = tuple(goal)
        if start == goal:
            return [start]
        start_h = int(self.heading) % 4
        pq = [(0.0, start, start_h)]
        best = {(start, start_h): 0.0}
        prev = {}
        goal_state = None
        while pq:
            cost, cell, h = heapq.heappop(pq)
            state = (cell, h)
            if cost > best.get(state, float("inf")) + 1e-9:
                continue
            if cell == goal:
                goal_state = state
                break
            for d in range(4):
                if self.edge_state.get((cell, d)) != "OPEN":
                    continue
                nb = neighbor(cell, d)
                if nb not in self.visited:
                    continue
                edge_m = self.remembered_edge_distance(cell, d)
                if edge_m is None:
                    edge_m = CELL_LENGTH_M
                q = abs((d - h) % 4); q = min(q, 4-q)
                step = float(edge_m) / max(0.10, DFS_KNOWN_SPEED_MPS) + 0.55*float(q)
                ns = (nb, d)
                nc = cost + step
                if nc + 1e-9 < best.get(ns, float("inf")):
                    best[ns] = nc
                    prev[ns] = state
                    heapq.heappush(pq, (nc, nb, d))
        if goal_state is None:
            return None
        states=[]; cur=goal_state
        while True:
            states.append(cur)
            if cur == (start, start_h): break
            cur=prev.get(cur)
            if cur is None: return None
        states.reverse()
        route=[]
        for cell,_h in states:
            if not route or route[-1] != cell:
                route.append(cell)
        return route

    def _cell_has_unvisited_open_neighbor(self, cell):
        """Return True when a scanned visited cell still borders unexplored OPEN space."""
        c = tuple(cell)
        for d in self.open_dirs.get(c, []):
            if self.edge_is_deferred(c, d):
                continue
            if self.edge_state.get((c, d)) != "OPEN":
                continue
            if neighbor(c, d) not in self.visited:
                return True
        return False

    def _route_time_score(self, route):
        """Cheap travel-time estimate for a known route, including turn cost."""
        if not route or len(route) < 2:
            return 0.0
        h = int(self.heading) % 4
        score = 0.0
        for a, b in zip(route, route[1:]):
            d = direction_between(a, b)
            if d is None:
                return float("inf")
            q = abs((int(d) - h) % 4)
            q = min(q, 4 - q)
            # A known 60 cm edge is ~1.1-1.5 s in the field.  A 90 deg turn has
            # meaningful overhead, so equal-hop routes prefer fewer turns.
            edge_m = self.remembered_edge_distance(a, d)
            if edge_m is None or not math.isfinite(float(edge_m)):
                edge_m = CELL_LENGTH_M
            score += float(edge_m) / max(0.10, DFS_KNOWN_SPEED_MPS)
            score += 0.55 * float(q)
            h = int(d)
        return score

    def find_best_frontier_route(self):
        """Fastest known OPEN route from current cell to any remaining frontier.

        Discovery still uses the existing DFS edge policy.  Only the *return trip*
        changes: instead of blindly following DFS parent links, jump through the
        already-proven graph to the nearest useful visited cell.
        """
        start = tuple(self.current)
        best = None
        for c in list(self.visited):
            c = tuple(c)
            if c == start or not self._cell_has_unvisited_open_neighbor(c):
                continue
            route = self.find_fastest_known_route(start, c)
            if not route or len(route) < 2:
                continue
            score = self._route_time_score(route)
            key = (score, len(route), c)
            if best is None or key < best[0]:
                best = (key, route)
        return None if best is None else best[1]

    def shortcut_to_frontier(self):
        route = self.find_best_frontier_route()
        if not route:
            return False
        target = tuple(route[-1])
        print(
            f"\n[FRONTIER SHORTCUT] {tuple(self.current)} -> {target} "
            f"known_route={route} score={self._route_time_score(route):.2f}"
        )
        if not self.navigate_known_route(route):
            self.fault(
                "FRONTIER SHORTCUT",
                f"known route failed before frontier {target}",
                "fall back to classic DFS parent backtrack",
            )
            return False
        self.current = target
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
            result = self.move_one_cell(cell, d, motion_profile="BACKTRACK_FAST")
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

            # V20.6: no local branch remains.  Do NOT automatically walk the
            # entire DFS parent chain.  The discovered maze is already a graph, so
            # route through confirmed OPEN visited cells to the nearest remaining
            # frontier (e.g. 22->32->33->34 instead of 22->21->20->30->31->32...).
            # This changes only relocation; new-edge discovery remains DFS-safe.
            if self.shortcut_to_frontier():
                rebuilt = self.reconstruct_dfs_stack()
                if rebuilt:
                    stack = rebuilt
                else:
                    # Parent links remain first-visit ancestry, so this should be
                    # rare.  Keep current logical node usable even if ancestry was
                    # damaged by a prior recovery.
                    stack = [tuple(self.current)]
                if MAP_AUTOSAVE:
                    self.save_map(final=False)
                continue

            # V20.7: no useful frontier remains anywhere.  The exploration work is
            # done; do NOT unwind the DFS parent chain edge-by-edge.  Go home over
            # the fastest confirmed OPEN route in the map.
            if tuple(cell) != tuple(self.root):
                home_route = self.find_fastest_known_route(cell, self.root)
                if home_route and len(home_route) >= 2:
                    print(
                        f"\n[FAST HOME] exploration frontier exhausted: {cell} -> {self.root} "
                        f"route={home_route}"
                    )
                    if self.navigate_known_route(home_route):
                        self.current = self.root
                        stack = []
                        break
                    self.fault(
                        "FAST HOME", "shortest known route failed",
                        "fall back to classic DFS parent backtrack",
                    )

            # Fallback only if shortest-home cannot be executed safely.
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
        if self.mission_mode == "ROUND1":
            self.save_round1_attack_memory()
        print(self.render_ascii_map())
        return self.map_complete


# ============================================================
# MISSION CONTROL GUI
# ============================================================
class MissionControlGUI:
    def __init__(self, explorer, initial_fire_mode="INFRARED", initial_burst=1, initial_round="ROUND1"):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is unavailable")
        self.explorer = explorer
        self.explorer.set_fire_mode(initial_fire_mode)
        self.explorer.set_fire_burst_count(initial_burst)
        self.explorer.set_mission_mode(initial_round)
        self.root = tk.Tk()
        self.root.title("RoboMaster Mission Control - Round 1 Map / Round 2 Shortest Attack")
        self.root.geometry("1080x760")
        self.root.minsize(940, 650)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.mission_thread = None
        self.mission_started = False
        self.closing = False

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.mission_page = ttk.Frame(self.notebook)
        self.target_page = ttk.Frame(self.notebook)
        self.notebook.add(self.mission_page, text="1. Mission / Live Map")
        self.notebook.add(self.target_page, text="2. Target Selection (16)")

        # ---------------- Page 1: original mission page ----------------
        outer = ttk.Frame(self.mission_page, padding=4)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        map_frame = ttk.LabelFrame(outer, text="Live Map / Round-2 Hints")
        map_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        map_frame.rowconfigure(0, weight=1)
        map_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            map_frame,
            width=CONTROL_GUI_CANVAS_W,
            height=CONTROL_GUI_CANVAS_H,
            background="white",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(outer)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        mode_box = ttk.LabelFrame(side, text="Mission Round", padding=8)
        mode_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.round_var = tk.StringVar(value=self.explorer.mission_mode)
        mode_row = ttk.Frame(mode_box)
        mode_row.pack(fill="x")
        ttk.Radiobutton(
            mode_row, text="Round 1: Explore + Map + Fire",
            variable=self.round_var, value="ROUND1", command=self._round_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_row, text="Round 2: Known Map + Shortest Attack",
            variable=self.round_var, value="ROUND2", command=self._round_changed,
        ).pack(anchor="w")
        ttk.Label(
            mode_box,
            text="Round 2 loads maps/round1_attack_memory.json and visits only proven firing cells.",
            wraplength=340,
        ).pack(anchor="w", pady=(5, 0))

        fire_box = ttk.LabelFrame(side, text="Fire Type / Burst", padding=8)
        fire_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.fire_var = tk.StringVar(value=self.explorer.get_fire_mode())
        ttk.Radiobutton(
            fire_box, text="Infrared", variable=self.fire_var,
            value="INFRARED", command=self._fire_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            fire_box, text="Water Blaster", variable=self.fire_var,
            value="WATER", command=self._fire_changed,
        ).pack(anchor="w")
        ttk.Separator(fire_box, orient="horizontal").pack(fill="x", pady=4)
        self.burst_var = tk.IntVar(value=self.explorer.get_fire_burst_count())
        burst_row = ttk.Frame(fire_box)
        burst_row.pack(anchor="w")
        ttk.Label(burst_row, text="Shots:").pack(side="left", padx=(0, 6))
        self.burst_combo = ttk.Combobox(
            burst_row, textvariable=self.burst_var,
            values=TARGET_FIRE_BURST_OPTIONS, state="readonly", width=5,
        )
        self.burst_combo.pack(side="left")
        self.burst_combo.bind("<<ComboboxSelected>>", self._burst_changed)
        ttk.Label(
            fire_box,
            text="Range gate <= 1.2 m; {:.2f}s between burst shots.".format(
                TARGET_FIRE_BURST_INTERVAL_SEC
            ), wraplength=340,
        ).pack(anchor="w", pady=(4, 0))

        geom_box = ttk.LabelFrame(side, text="Aim Geometry", padding=8)
        geom_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.geometry_text = tk.StringVar()
        ttk.Label(geom_box, textvariable=self.geometry_text, justify="left").pack(anchor="w")

        status_box = ttk.LabelFrame(side, text="Robot / Target Status", padding=8)
        status_box.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        self.status_text = tk.StringVar(value="READY TO START")
        ttk.Label(
            status_box, textvariable=self.status_text, justify="left",
            wraplength=340,
        ).pack(anchor="w", fill="x")

        aim_box = ttk.LabelFrame(side, text="Last Fire Solution", padding=8)
        aim_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.aim_text = tk.StringVar(value="No fire solution yet")
        ttk.Label(aim_box, textvariable=self.aim_text, justify="left", wraplength=340).pack(anchor="w")

        controls = ttk.Frame(side)
        controls.grid(row=5, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.start_btn = ttk.Button(controls, text="START MISSION", command=self._start_mission)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_btn = ttk.Button(controls, text="STOP", command=self._stop_mission)
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # ---------------- Page 2: 16 color x shape rules ----------------
        target_outer = ttk.Frame(self.target_page, padding=14)
        target_outer.pack(fill="both", expand=True)
        ttk.Label(
            target_outer,
            text=(
                "Tick ONLY the color/shape combinations that are legal targets. "
                "Unselected combinations are still detected and remembered, but will never fire. "
                "Round 2 also ignores unselected Round-1 firing hints."
            ),
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 12))

        self.target_vars = {}
        pretty_shape = {
            "SQUARE": "Square",
            "CIRCLE": "Circle",
            "RECT_HORIZONTAL": "Rectangle H",
            "RECT_VERTICAL": "Rectangle V",
        }
        for col_idx, shape in enumerate(TARGET_FILTER_SHAPES, start=1):
            ttk.Label(target_outer, text=pretty_shape[shape]).grid(
                row=1, column=col_idx, padx=10, pady=4, sticky="w"
            )
        for row_idx, color in enumerate(TARGET_FILTER_COLORS, start=2):
            ttk.Label(target_outer, text=color.title()).grid(
                row=row_idx, column=0, padx=(0, 12), pady=7, sticky="w"
            )
            for col_idx, shape in enumerate(TARGET_FILTER_SHAPES, start=1):
                key = (color, shape)
                var = tk.BooleanVar(value=True)
                self.target_vars[key] = var
                ttk.Checkbutton(
                    target_outer,
                    variable=var,
                    command=self._target_filter_changed,
                ).grid(row=row_idx, column=col_idx, padx=10, pady=7, sticky="w")

        target_buttons = ttk.Frame(target_outer)
        target_buttons.grid(row=7, column=0, columnspan=5, sticky="w", pady=(16, 8))
        ttk.Button(target_buttons, text="SELECT ALL 16", command=self._select_all_targets).pack(side="left", padx=(0, 8))
        ttk.Button(target_buttons, text="CLEAR ALL", command=self._clear_all_targets).pack(side="left")
        self.target_filter_text = tk.StringVar(value="16 / 16 target classes enabled")
        ttk.Label(target_outer, textvariable=self.target_filter_text).grid(
            row=8, column=0, columnspan=5, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            target_outer,
            text=(
                "Examples: Red + Square ON and Blue + Square OFF means a red square can fire, "
                "while a blue square is detection-only. Changes are applied live before the next shot."
            ),
            wraplength=900,
            justify="left",
        ).grid(row=9, column=0, columnspan=5, sticky="w", pady=(10, 0))

        self._target_filter_changed()
        self._refresh()

    def _round_changed(self):
        if not self.mission_started:
            self.explorer.set_mission_mode(self.round_var.get())

    def _fire_changed(self):
        self.explorer.set_fire_mode(self.fire_var.get())

    def _burst_changed(self, event=None):
        self.explorer.set_fire_burst_count(self.burst_var.get())

    def _target_filter_changed(self):
        selected = [key for key, var in self.target_vars.items() if bool(var.get())]
        self.explorer.set_target_selection(selected)
        self.target_filter_text.set("{} / 16 target classes enabled".format(len(selected)))

    def _select_all_targets(self):
        for var in self.target_vars.values():
            var.set(True)
        self._target_filter_changed()

    def _clear_all_targets(self):
        for var in self.target_vars.values():
            var.set(False)
        self._target_filter_changed()

    def _start_mission(self):
        if self.mission_started:
            return
        self.mission_started = True
        self.explorer.running = True
        self.explorer.set_mission_mode(self.round_var.get())
        self.explorer.set_fire_mode(self.fire_var.get())
        self.explorer.set_fire_burst_count(self.burst_var.get())
        self._target_filter_changed()
        self.start_btn.configure(state="disabled")
        self.mission_thread = threading.Thread(
            target=self._mission_worker, name="RoboMasterMission", daemon=False
        )
        self.mission_thread.start()

    def _mission_worker(self):
        try:
            if self.explorer.connect():
                self.explorer.run_selected_mission()
        except Exception as exc:
            self.explorer.fault(
                "GUI MISSION",
                "{}: {}".format(type(exc).__name__, exc),
                "safe stop + save current state",
            )
            self.explorer.enter_safe_pause("GUI mission exception")
        finally:
            self.explorer.cleanup()

    def _stop_mission(self):
        self.explorer.running = False
        try:
            self.explorer.safe_stop()
        except Exception:
            pass
        self.status_text.set("STOP requested - robot stopping safely")

    def _on_close(self):
        self.closing = True
        self._stop_mission()
        self.root.after(250, self.root.destroy)

    @staticmethod
    def _heading_arrow(heading):
        return {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}.get(int(heading) % 4, (0, -1))

    def _snapshot_map(self):
        try:
            visited = set(self.explorer.visited)
            edge_state = dict(self.explorer.edge_state)
            current = tuple(self.explorer.current)
            root = tuple(self.explorer.root)
            heading = int(self.explorer.heading)
            targets = [dict(t) for t in self.explorer.target_system.targets]
            hints = [dict(h) for h in self.explorer.round2_hints]
            return visited, edge_state, current, root, heading, targets, hints
        except Exception:
            return set(), {}, (0, 0), (0, 0), 0, [], []

    def _draw_map(self):
        self.canvas.delete("all")
        visited, edge_state, current, root, heading, targets, hints = self._snapshot_map()
        cells = set(visited) | {current, root}
        for (cell, d), state in edge_state.items():
            c = tuple(cell)
            cells.add(c)
            if state == "OPEN":
                dx, dy = DIR_VEC[int(d) % 4]
                cells.add((c[0] + dx, c[1] + dy))
        for h in hints:
            c = h.get("cell")
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                cells.add((int(c[0]), int(c[1])))
        if not cells:
            return

        min_x = min(c[0] for c in cells); max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells); max_y = max(c[1] for c in cells)
        cw = max(320, self.canvas.winfo_width())
        ch = max(320, self.canvas.winfo_height())
        pad = 36.0
        nx = max(1, max_x - min_x + 1); ny = max(1, max_y - min_y + 1)
        cell_px = max(28.0, min((cw - 2*pad) / nx, (ch - 2*pad) / ny))

        def origin(cell):
            x, y = cell
            return (
                pad + (x - min_x) * cell_px,
                pad + (max_y - y) * cell_px,
            )

        for c in cells:
            x0, y0 = origin(c)
            if c in visited:
                self.canvas.create_rectangle(
                    x0+2, y0+2, x0+cell_px-2, y0+cell_px-2,
                    fill="#f2f2f2", outline="",
                )
            label = "S" if c == root else "{},{}".format(c[0], c[1])
            self.canvas.create_text(x0+cell_px/2, y0+cell_px/2, text=label, fill="#777777")

        for (cell, d), state in edge_state.items():
            c = tuple(cell); d = int(d) % 4
            if state == "OPEN":
                continue
            x0, y0 = origin(c)
            x1, y1, x2, y2 = x0, y0, x0, y0
            if d == 0: x1,y1,x2,y2 = x0,y0,x0+cell_px,y0
            elif d == 1: x1,y1,x2,y2 = x0+cell_px,y0,x0+cell_px,y0+cell_px
            elif d == 2: x1,y1,x2,y2 = x0,y0+cell_px,x0+cell_px,y0+cell_px
            else: x1,y1,x2,y2 = x0,y0,x0,y0+cell_px
            if state in ("WALL", "BLOCKED"):
                self.canvas.create_line(x1,y1,x2,y2, fill="black", width=4)
            else:
                self.canvas.create_line(x1,y1,x2,y2, fill="#999999", width=2, dash=(5,4))

        target_colors = {"RED":"#d62728", "GREEN":"#2ca02c", "BLUE":"#1f77b4", "YELLOW":"#c7a600"}
        for t in targets:
            pos = t.get("estimated_grid_xy")
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            try:
                gx, gy = float(pos[0]), float(pos[1])
            except Exception:
                continue
            px = pad + (gx - min_x + 0.5) * cell_px
            py = pad + (max_y - gy + 0.5) * cell_px
            color = target_colors.get(str(t.get("color") or "").upper(), "#555555")
            self.canvas.create_oval(px-6, py-6, px+6, py+6, fill=color, outline="black")
            self.canvas.create_text(px+10, py-9, text=str(t.get("id") or "T"), anchor="w", fill=color)

        # Proven Round-1 firing anchors are shown as H markers during Round 2.
        for h in hints:
            c = h.get("cell")
            if not isinstance(c, (list, tuple)) or len(c) < 2:
                continue
            cell = (int(c[0]), int(c[1]))
            x0, y0 = origin(cell)
            px, py = x0 + cell_px*0.78, y0 + cell_px*0.22
            color = target_colors.get(str(h.get("color") or "").upper(), "#7a3db8")
            self.canvas.create_rectangle(px-5, py-5, px+5, py+5, fill=color, outline="black")
            self.canvas.create_text(px-7, py, text="H", anchor="e", fill=color)

        x0, y0 = origin(current)
        cx, cy = x0 + cell_px/2, y0 + cell_px/2
        dx, dy = self._heading_arrow(heading)
        self.canvas.create_oval(cx-9, cy-9, cx+9, cy+9, outline="#0057b7", width=3)
        self.canvas.create_line(cx,cy,cx+dx*cell_px*0.32,cy+dy*cell_px*0.32, fill="#0057b7", width=4, arrow=tk.LAST)

    def _refresh(self):
        if self.closing:
            return
        try:
            self._draw_map()
            tof = self.explorer.latest_tof(fresh=False)
            gp, gy = self.explorer.current_gimbal_relative()
            current = tuple(self.explorer.current)
            heading = DIR_NAMES[int(self.explorer.heading) % 4]
            status = self.explorer.target_system.status
            fire_mode = self.explorer.get_fire_mode()
            burst_count = self.explorer.get_fire_burst_count()
            fire_event = self.explorer.target_system.last_fire_event
            selected_count = len(self.explorer.get_target_selection())
            self.status_text.set(
                "Mode={}  Cell={}  Heading={}\nVisited={}  PoseTrusted={}\n"
                "ToF={} mm  Gimbal P/Y={}/{}\n"
                "Target={}\nFire={} x{}  Selected={}/16\nLast={}".format(
                    self.explorer.mission_mode, current, heading,
                    len(self.explorer.visited), self.explorer.pose_trusted,
                    "NA" if tof is None else "{:.0f}".format(tof),
                    "NA" if gp is None else "{:+.1f}".format(gp),
                    "NA" if gy is None else "{:+.1f}".format(gy),
                    status, fire_mode, burst_count, selected_count, fire_event,
                )
            )
            self.geometry_text.set(
                "Center -> ToF forward      : {:.1f} cm\n"
                "Center -> muzzle forward   : {:.1f} cm\n"
                "ToF -> muzzle forward      : {:.1f} cm\n"
                "Camera -> muzzle vertical  : {:+.1f} cm\n"
                "ToF -> muzzle vertical     : {:+.1f} cm\n"
                "Aim policy                  : CENTER -> physical muzzle LOS\n"
                "DFS explore speed           : {:.2f} m/s\n"
                "DFS known/backtrack speed   : {:.2f} m/s\n"
                "Round1 target time          : {:.0f} s\n"
                "Fire RAW ToF gate           : <= {:.0f} mm\n"
                "Round2 narrow replay        : +/- {:.0f} deg\n"
                "Round2 deadline guard       : {:.0f} s".format(
                    FIRE_TOF_FORWARD_FROM_CENTER_M*100.0,
                    FIRE_MUZZLE_FORWARD_FROM_CENTER_M*100.0,
                    FIRE_MUZZLE_AHEAD_OF_TOF_M*100.0,
                    FIRE_CAMERA_ABOVE_MUZZLE_M*100.0,
                    FIRE_TOF_ABOVE_MUZZLE_M*100.0,
                    DFS_EXPLORE_SPEED_MPS,
                    DFS_KNOWN_SPEED_MPS,
                    MAX_MISSION_SEC,
                    TARGET_FIRE_MAX_RANGE_MM,
                    ROUND2_NARROW_SWEEP_HALF_DEG,
                    ROUND2_HARD_LIMIT_SEC,
                )
            )
            s = dict(self.explorer.target_system.last_aim_solution)
            if s:
                self.aim_text.set(
                    "Mode={}  ToF(raw)={:.0f} mm\n"
                    "Center range={:.0f} mm  Muzzle range={:.0f} mm\n"
                    "Camera P={:+.2f} deg\n"
                    "Muzzle correction={:+.2f} deg\n"
                    "Fire P={:+.2f} deg\n"
                    "ToF-Camera beam offset={:+.2f} deg".format(
                        s.get("fire_mode", "?"), float(s.get("tof_range_mm", 0.0)),
                        float(s.get("robot_center_to_target_planar_mm", 0.0)),
                        float(s.get("muzzle_to_target_planar_mm", 0.0)),
                        float(s.get("camera_lock_pitch_deg", 0.0)),
                        float(s.get("camera_muzzle_parallax_pitch_deg", 0.0)),
                        float(s.get("fire_pitch_deg", 0.0)),
                        float(s.get("tof_camera_parallax_deg", 0.0)),
                    )
                )
        except Exception:
            pass
        self.root.after(CONTROL_GUI_REFRESH_MS, self._refresh)

    def run(self):
        self.root.mainloop()
        if self.mission_thread is not None and self.mission_thread.is_alive():
            self.explorer.running = False
            try:
                self.explorer.safe_stop()
            except Exception:
                pass
            self.mission_thread.join(timeout=3.0)
        if not self.explorer.cleanup_done:
            self.explorer.cleanup()


# ============================================================
# MAIN
# ============================================================
def _run_headless(explorer):
    try:
        if explorer.connect():
            explorer.run_selected_mission()
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        explorer.running = False
        explorer.safe_stop()
        explorer.save_map(final=False)
    except Exception as exc:
        explorer.fault("UNEXPECTED", "{}: {}".format(type(exc).__name__, exc), "SAFE STOP + save partial map")
        explorer.enter_safe_pause("unexpected exception contained: {}: {}".format(type(exc).__name__, exc))
    finally:
        explorer.cleanup()


def main():
    parser = argparse.ArgumentParser(description="RoboMaster DFS + Target Mission Control")
    parser.add_argument("--no-gui", action="store_true", help="run immediately without Tk GUI")
    parser.add_argument(
        "--fire", choices=("INFRARED", "WATER"), default=TARGET_FIRE_MODE_DEFAULT,
        help="initial fire mode (GUI can change it live)",
    )
    parser.add_argument(
        "--shots", type=int, choices=TARGET_FIRE_BURST_OPTIONS,
        default=TARGET_FIRE_BURST_DEFAULT,
        help="shots per locked target; GUI can change 1-6 live",
    )
    parser.add_argument(
        "--round", dest="mission_round", choices=(1, 2), type=int, default=1,
        help="1=explore/map/fire and save attack memory, 2=load map + shortest attack",
    )
    args = parser.parse_args()

    explorer = DFSMapOnlyExplorer()
    explorer.set_fire_mode(args.fire)
    explorer.set_fire_burst_count(args.shots)
    explorer.set_mission_mode("ROUND{}".format(args.mission_round))
    use_gui = bool(CONTROL_GUI_ENABLED and not args.no_gui and tk is not None and ttk is not None)
    if use_gui:
        try:
            MissionControlGUI(
                explorer, initial_fire_mode=args.fire, initial_burst=args.shots,
                initial_round="ROUND{}".format(args.mission_round),
            ).run()
            return
        except Exception as exc:
            print("[GUI WARN] {}: {} -> headless mode".format(type(exc).__name__, exc))
    _run_headless(explorer)


if __name__ == "__main__":
    main()
