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
  - A wall-map SVG image is also autosaved so the maze can be seen directly.
  - Fake-exit guard defers broad/open boundary directions as EXIT_CANDIDATE.
  - Boundary containment is motion-based: a straight corridor is monitored
    almost to the cell target; one-wall loss slows the robot, both-wall loss
    triggers fan verification and retreat.
  - OPEN-AREA TRAP fallback prevents DFS from choosing another direction if a
    boundary was missed and the robot reaches an outside pseudo-cell.
  - Root (0,0) is an adaptive START/STAGING ANCHOR: FRONT is always the
    known maze ingress, while BACK is the known return/entrance side.
    This supports both corridor starts and fully-open staging-area starts.
  - Saved maps can be replayed later without re-discovering topology.
  - IR firing is range-gated to <= 2 physical tiles (1200 mm) using fresh ToF.
  - Exit detection uses strong multi-ray evidence so 2-3-cell-deep junctions are
    not rejected merely because the forward/diagonal ToF sees a long distance.
  - Known-map mode can BFS to a requested goal cell.
  - Node scans never perform IR lateral recovery; IR LOW at a node triggers
    an immediate stationary Gimbal L/F/R topology scan instead.
  - After the final frontier is scanned, DFS stops backtracking and uses the
    confirmed completed map to take the fastest move+turn route back to (0,0).
  - Mid-edge single-IR recovery failure no longer crashes immediately:
    near-node -> accept; otherwise retreat to source cell and rescan/replan.

IMPORTANT:
  Tune CELL_LENGTH_M and TOF_OPEN_THRESHOLD_MM for the real maze geometry.
  Known-map mode assumes the same physical start/root and initial orientation.
"""

from robomaster import robot, blaster
import argparse
import heapq
import json
import math
import os
import queue
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

# Corridor wall-follow band.
# Calibration/logs show the corridor is roughly 25-27 cm wide.  13 cm remains
# the nominal reference for authority selection, but lateral control no longer
# tries to hold exactly 13.0 cm every cycle.  Instead, the robot coasts inside
# a comfort band and only nudges laterally when it becomes too close/far.
#
# This reduces the old constant left/right "pushing" while preserving a safe
# wall-follow reference.
#
# IMPORTANT CONTROL-LAW RULE:
# Both Sharp sensors may be READ for supervision/arbitration, but ONLY ONE
# Sharp sensor owns lateral control authority at a time.
CENTER_TARGET_CM = 13.0                 # nominal only / authority selection
SHARP_FOLLOW_NEAR_CM = 11.5             # closer -> nudge AWAY from wall
SHARP_FOLLOW_FAR_CM = 15.0              # farther -> nudge TOWARD wall
SHARP_FOLLOW_KP = 0.030                 # proportional gain outside safe band
SHARP_FOLLOW_MIN_STRAFE_MPS = 0.025     # minimum useful correction
MAX_CENTER_STRAFE_MPS = 0.065           # softer than old continuous centering

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
# Keep the IR/Gimbal supervisor consistent with the DFS topology classifier.
IR_BOTH_ROUTE_OPEN_MM = 600.0
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

# Physical field geometry.  One floor tile / maze pitch is 60 cm.
# Keep this separate from CELL_LENGTH_M because the real chassis may need to
# travel slightly less than the nominal 60 cm pitch to stop near cell centre.
GRID_TILE_M = 0.60
GRID_TILE_MM = GRID_TILE_M * 1000.0

# Tuned odometry displacement for one logical cell move.
# The map still uses a 60 cm physical grid; the chassis stops after ~55 cm
# because of its real starting/stopping geometry.
CELL_LENGTH_M = 0.55

# Reaching this fraction counts as arriving at the next cell if ToF
# sees the end wall slightly earlier than odometry expects.
CELL_SUCCESS_FRACTION = 0.82

# ToF at cell center:
# <= threshold -> WALL
# > threshold  -> OPEN
#
# *** TUNE THIS FOR THE REAL MAZE ***
# Maze cell pitch is 60 cm.
# At a node, a wall belonging to the current cell should normally appear
# much closer than one full grid pitch; an OPEN direction should see through
# into the next cell.  Use 600 mm as the topology-open threshold.
TOF_OPEN_THRESHOLD_MM = 600

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


# ============================================================
# FAKE-EXIT / OPEN-BOUNDARY GUARD
# ============================================================
# The maze dimensions are unknown. A wide opening at the outer boundary can
# otherwise look like one more grid edge and make DFS continue outside.
#
# Explore-mode policy:
#   * wide/open boundary => EXIT_CANDIDATE
#   * never traverse EXIT_CANDIDATE during the mapping pass
#   * keep exploring the interior
#   * after exploration, return to known root/entrance (0, 0)
#
# We intentionally do NOT try to decide "real exit" vs "fake exit" during the
# mapping pass. Both are deferred, which prevents either one from luring DFS
# outside the maze.
EXIT_GUARD_ENABLED = True

# After turning toward a selected edge, if either side Sharp still sees a wall
# in this range, it still looks like a normal corridor and no expensive fan
# scan is needed.
EXIT_CORRIDOR_WALL_MAX_CM = 22.0

# Only suspicious edges (both side walls absent/far) get this wide fan scan.
# Every ray still uses TOF_SCAN_SAMPLES = 7.
#
# IMPORTANT: a normal route is OPEN at > 1 tile (~600 mm), but that must NOT
# be enough to call it an EXIT.  Internal T/+ junctions and a 2-3-cell-deep
# corridor can easily produce >600 mm on several rays.  EXIT therefore needs:
#   (1) broad-open evidence on both sides, AND
#   (2) several rays that stay open much farther into the room.
EXIT_FAN_ANGLES_DEG = (-60.0, -45.0, -30.0, +30.0, +45.0, +60.0)
EXIT_FAN_OPEN_MM = GRID_TILE_MM * 1.50          # 900 mm = 1.5 tiles
EXIT_FAN_MIN_OPEN_PER_SIDE = 2
EXIT_FAN_STRONG_OPEN_MM = GRID_TILE_MM * 2.50   # 1500 mm = 2.5 tiles
EXIT_FAN_MIN_STRONG_TOTAL = 4                   # out of 6 rays
EXIT_FAN_MIN_STRONG_PER_SIDE = 1

# A very close center obstruction is treated as obstacle/noise, not an exit.
# If a box/person partly blocks a true opening but is farther than this, the
# side fan can still identify the broad opening.
EXIT_FRONT_MIN_SAFE_MM = 300.0

# Fake exits in the real field can look like a NORMAL corridor at the robot:
# both Sharp sensors still see the two side walls, but those walls terminate
# together a short distance ahead.  Side-looking Sharp alone cannot detect
# that geometry.
#
# Before crossing a straight corridor frontier, probe shallow forward angles.
# A continuing corridor should make these rays hit the same side walls at a
# predictable range.  If the measured ray goes MUCH farther than that
# prediction, the side wall probably ENDS ahead.
EXIT_WALL_END_PROBE_ENABLED = True
EXIT_WALL_END_ANGLES_LEFT = (-15.0, -10.0)
EXIT_WALL_END_ANGLES_RIGHT = (+10.0, +15.0)

# Only arm the shallow probe when FRONT is sufficiently open.  This avoids
# wasting time at short wall approaches.
EXIT_WALL_END_FRONT_ARM_MM = 850.0

# Measured ToF must exceed BOTH:
#   predicted continuing-wall hit * ratio
#   predicted continuing-wall hit + margin
# to vote that the wall has ended.
EXIT_WALL_END_RATIO = 1.45
EXIT_WALL_END_MARGIN_MM = 220.0
EXIT_WALL_END_MIN_MEASURED_MM = 750.0

# One 7-sample-median ray per side is enough to raise suspicion, but BOTH
# physical sides must independently vote "wall ended".
EXIT_WALL_END_MIN_VOTES_PER_SIDE = 1

# IMPORTANT V8.9:
# Do NOT use the stationary projected-wall-end detector as a reason to stop a
# normal corridor edge. Real maze grid walls can be segmented and can look as
# if they "end ahead" from a stationary shallow-angle scan.
#
# Instead, detect the real/fake exit WHILE crossing the edge:
#   both Sharp walls present at move start
#   -> both disappear together during the middle of the cell
#   -> confirm for several consecutive samples
#   -> stop + wide fan scan
#   -> if broad open space is confirmed, retreat to source and defer edge.
EXIT_MOTION_GUARD_ENABLED = True
# Ignore early wall gaps.  Junction corners / segmented foam near the source
# used to look like an exit as soon as both Sharp readings disappeared.
EXIT_MOTION_MIN_TRAVEL_M = 0.26
# Stop exit classification before the logical cell centre so there is still
# room to retreat if the opening is a real/fake boundary.
EXIT_MOTION_MAX_TRAVEL_M = 0.48
# 4 consecutive control samples (~0.20 s at CONTROL_DT=0.05) are required.
EXIT_MOTION_LOST_CONFIRM_COUNT = 4

# If one wall disappears late in the edge, slow down and gather more evidence
# instead of immediately turning that wall gap into EXIT_CANDIDATE.
EXIT_MOTION_CAUTION_START_M = 0.24
EXIT_MOTION_CAUTION_SPEED_MPS = 0.08


# ============================================================
# OPEN-AREA TRAP FALLBACK
# ============================================================
# Last-resort containment: if the robot nevertheless reaches a "cell" that
# looks like open floor rather than a maze cell, immediately go back to parent
# BEFORE DFS can choose W/N/E and continue outside.
OPEN_AREA_TRAP_ENABLED = True
OPEN_AREA_TRAP_MIN_OPEN_FAN_RAYS = 4      # out of 6
# A legitimate corridor can be ~3 tiles deep.  Do not call that open floor.
# 3.75 tiles = 2250 mm gives a buffer beyond a 3-tile sightline.
OPEN_AREA_TRAP_LONG_MM = GRID_TILE_MM * 3.75
OPEN_AREA_TRAP_MIN_LONG_FAN_RAYS = 4
OPEN_AREA_TRAP_SIDE_WALL_MAX_CM = 22.0


# ============================================================
# ADAPTIVE START / STAGING ANCHOR
# ============================================================
# (0,0) is treated as a launch anchor, not as an ordinary maze topology cell.
# This lets the SAME code start from:
#
#   A) a normal corridor with side walls, OR
#   B) a completely open staging area with the maze straight ahead.
#
# Startup contract:
#   chassis initially faces INTO the maze.
#
# At root:
#   FRONT = known maze ingress
#   BACK  = known entrance / return side
#   LEFT/RIGHT are ignored as DFS frontiers even if the staging area is open.
START_ANCHOR_ENABLED = True
START_MAZE_INGRESS_DIR = 0   # N relative to startup heading
START_FORCE_FRONT_OPEN = True

# Startup diagnostics only. They do not reject the launch.
START_OPEN_MM = 600.0


# ============================================================
# KNOWN ENTRANCE CORRIDOR
# ============================================================
# The robot starts at root=(0,0), facing INTO the maze.
# Therefore the physical entrance is root-back.
#
# The entrance can have the same geometry as an exit:
# two side walls form a corridor and terminate at the outside opening.
# We identify this edge from startup context, NOT from exit geometry.
ENTRANCE_CORRIDOR_PROFILE_ENABLED = True

# Gimbal yaw angles relative to chassis used to profile the corridor behind
# the robot without moving the chassis.  +180 is directly backward.
# All rays still use TOF_SCAN_SAMPLES = 7.
ENTRANCE_PROFILE_ANGLES_DEG = (
    +150.0,
    +165.0,
    +180.0,
    +195.0,
    +210.0,
)

# Center/back ray should normally see at least one grid pitch into/out through
# the entrance corridor.  Failure only produces a warning: the entrance is
# still trusted because it is known from the starting condition.
ENTRANCE_OPEN_VERIFY_MM = 600.0

# Used only as descriptive map metadata.
ENTRANCE_CORRIDOR_SIDE_WALL_MAX_CM = 22.0


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
# TARGET VISION / TARGET MEMORY
# ============================================================
# Vision is deliberately separated from the DFS/ToF topology classifier.
# A colored patch is NOT accepted as a target merely because its HSV matches:
# it must also pass geometry/quality gates and remain stable for several frames.
TARGET_VISION_ENABLED = True
TARGET_PREVIEW_ENABLED = True
TARGET_WINDOW_NAME = "RoboMaster Target Vision"

# Camera stream. RoboMaster EP 360p is normally 640x360 and is light enough
# for simultaneous DFS + OpenCV on a typical competition laptop.
TARGET_CAMERA_RESOLUTION = "360p"

# Only objects whose CENTRE lies inside this box are eligible.
# The gimbal sweep deliberately moves side-wall targets INTO this central ROI.
TARGET_ROI_X_MIN = 0.20
TARGET_ROI_X_MAX = 0.80
TARGET_ROI_Y_MIN = 0.16
TARGET_ROI_Y_MAX = 0.88

# ------------------------------------------------------------
# FOAM-WALL SPATIAL GATE
# ------------------------------------------------------------
# Competition-field constraint: a legitimate target will not appear ABOVE
# the visible top edge of the white foam maze wall.  We therefore estimate a
# per-column foam-wall top profile and reject HSV/SDK candidates whose centre
# lies above that profile.  This is a geometry gate, not another color score.
#
# FAIL-OPEN is intentional: if the wall top cannot be estimated at a candidate
# x-position (occlusion, extreme pitch, unusual lighting), normal ROI/shape/
# temporal checks still run instead of silently deleting a real target.
TARGET_FOAM_GATE_ENABLED = True
TARGET_FOAM_FAIL_CLOSED = False

# White/grey foam segmentation in HSV.  Hue is ignored because low-saturation
# whites/greys have unstable hue.  These values were chosen to keep the foam
# visible in the supplied competition-room frame while suppressing most floor.
TARGET_FOAM_HSV_LOW = (0, 0, 115)
TARGET_FOAM_HSV_HIGH = (180, 48, 255)
TARGET_FOAM_OPEN_KERNEL = 3
TARGET_FOAM_CLOSE_KERNEL = 9

# Connected foam-like regions must be substantial and extend far enough down
# the image to avoid accepting ceiling panels/lights as the maze wall.
TARGET_FOAM_MIN_COMPONENT_AREA_PX = 1800
TARGET_FOAM_MIN_COMPONENT_WIDTH_PX = 35
TARGET_FOAM_MIN_COMPONENT_HEIGHT_PX = 35
TARGET_FOAM_COMPONENT_MIN_BOTTOM_FRAC = 0.42

# Build a smooth wall-top profile and bridge only short gaps caused by target
# signs, seams or small occlusions.
TARGET_FOAM_PROFILE_MAX_INTERP_GAP_PX = 64
TARGET_FOAM_PROFILE_MEDIAN_RADIUS_PX = 7
TARGET_FOAM_PROFILE_SAMPLE_RADIUS_PX = 12
TARGET_FOAM_PROFILE_TTL_SEC = 0.55
TARGET_FOAM_MIN_ROI_COVERAGE = 0.12

# Hard candidate gate.  The candidate centre must be below the local wall top
# plus a small tolerance, and at least this fraction of its box must remain
# below that boundary.
TARGET_FOAM_CENTER_MARGIN_PX = 4
TARGET_FOAM_MIN_BBOX_BELOW_FRAC = 0.55

# HSV starting ranges. These MUST be tuned under the real room lighting.
# Red uses two hue bands because HSV hue wraps around 0/180 in OpenCV.
TARGET_HSV_RANGES = {
    "RED": [
        ((0, 95, 65), (11, 255, 255)),
        ((169, 95, 65), (180, 255, 255)),
    ],
    "YELLOW": [
        ((18, 100, 80), (38, 255, 255)),
    ],
    "GREEN": [
        ((38, 75, 55), (90, 255, 255)),
    ],
    "BLUE": [
        ((92, 85, 55), (138, 255, 255)),
    ],
}

# Reject tiny color speckles and huge regions such as a yellow wall.
# Area is measured as a fraction of the ROI area.
TARGET_MIN_AREA_FRAC_ROI = 0.0018
TARGET_MAX_AREA_FRAC_ROI = 0.20

# Shape-quality gates. A target should be a clean planar geometric sign.
TARGET_MIN_SOLIDITY = 0.86
TARGET_RECT_MIN_FILL = 0.68
TARGET_RECT_MAX_CORNER_COS = 0.42
TARGET_CIRCLE_MIN_CIRCULARITY = 0.72
TARGET_CIRCLE_ASPECT_MIN = 0.70
TARGET_CIRCLE_ASPECT_MAX = 1.42
TARGET_POLY_EPS_FRAC = 0.035
TARGET_BORDER_MARGIN_PX = 5

# Four required shape classes.
TARGET_SQUARE_ASPECT_MIN = 0.78
TARGET_SQUARE_ASPECT_MAX = 1.28
TARGET_RECT_ASPECT_MIN = 1.28

# Temporal confirmation is the main anti-noise gate.
TARGET_HISTORY_SEC = 0.90
TARGET_CONFIRM_FRAMES = 3
TARGET_CONFIRM_MAX_CENTER_STD = 0.045
TARGET_CONFIRM_MAX_AREA_CV = 0.38

# ------------------------------------------------------------
# SAMPLED TARGET WINDOW (adapted from the friend's Round-1 policy)
# ------------------------------------------------------------
# Do not decide from a single frame.  At each gimbal pose, collect a bounded
# camera window, require repeated evidence, and only then promote it to LOCK.
#
# The values mirror the useful behavior exposed by final_round1_tof_camera_01:
#   confidence >= 0.50  -> worth holding / investigating
#   confidence >= 0.60  -> eligible to save once temporally verified
#   10 sampled frames   -> one observation window
#   3 verified frames   -> minimum temporal agreement
#   max 3 windows / 3 s -> do not stare forever at noise
TARGET_MIN_CONFIDENCE = 0.50
TARGET_SAVE_CONFIDENCE = 0.60
TARGET_SAMPLE_FRAMES = 10
TARGET_VERIFY_FRAMES = 3
TARGET_HOLD_MAX_SEC = 3.0
TARGET_HOLD_MAX_WINDOWS = 3

# 360p video is usually much faster than this.  The timeout prevents a camera
# hiccup from blocking DFS forever while still allowing 10 fresh frames.
TARGET_SAMPLE_WINDOW_TIMEOUT_SEC = 0.85
TARGET_SAMPLE_POLL_SEC = 0.015

# At every newly scanned DFS cell the turret checks only two downward
# pitch levels. Positive/upward target-search poses are intentionally removed.
#
# RoboMaster convention used by this file:
#   negative pitch = down
#
# Final-run vision pitch policy:
#   * -5 deg    : normal / shallow-down view
#   * -15 deg   : deepest allowed view around +/-45 deg yaw.  At these
#                 diagonal poses a deeper tilt can point the ToF/IR geometry
#                 into nearby hardware / maze edges and create bad returns.
#   * -22.5 deg : deep/close-target view allowed at FRONT and near-side
#                 (+/-82 deg, approximately 90 deg) poses.
#
# Keep LEFT / diagonal / FRONT / diagonal / RIGHT coverage, but apply a
# yaw-dependent pitch floor so only the +/-45 deg search is limited.
TARGET_SWEEP_PITCH_NORMAL_DEG = -5.0
TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG = -15.0
TARGET_SWEEP_PITCH_DEEP_DEG = -22.5
TARGET_DIAGONAL_YAW_MIN_DEG = 25.0
TARGET_DIAGONAL_YAW_MAX_DEG = 65.0

TARGET_SWEEP_POSES = (
    (-82.0, TARGET_SWEEP_PITCH_NORMAL_DEG),
    (-82.0, TARGET_SWEEP_PITCH_DEEP_DEG),

    (-45.0, TARGET_SWEEP_PITCH_NORMAL_DEG),
    (-45.0, TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG),

    (  0.0, TARGET_SWEEP_PITCH_NORMAL_DEG),
    (  0.0, TARGET_SWEEP_PITCH_DEEP_DEG),

    (+45.0, TARGET_SWEEP_PITCH_NORMAL_DEG),
    (+45.0, TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG),

    (+82.0, TARGET_SWEEP_PITCH_NORMAL_DEG),
    (+82.0, TARGET_SWEEP_PITCH_DEEP_DEG),
)

TARGET_SWEEP_SETTLE_SEC = 0.07
TARGET_SWEEP_OBSERVE_SEC = 0.24
TARGET_SCAN_COOLDOWN_SEC = 0.80

# ------------------------------------------------------------
# TRANSIENT GLIMPSE + RETURN-PASS RE-SCAN
# ------------------------------------------------------------
# A target can cross the ROI for only one or two frames while the gimbal is
# sweeping.  Do not immediately throw that evidence away: hold the current
# pose briefly, then remember an unresolved GLIMPSE if it still cannot pass
# the normal temporal confirmation gate.
TARGET_GLIMPSE_ENABLED = True
TARGET_GLIMPSE_LINGER_SEC = 0.38
TARGET_GLIMPSE_MIN_SCORE = 0.48
TARGET_GLIMPSE_BEARING_MERGE_DEG = 18.0
TARGET_GLIMPSE_MAX_PER_CELL = 8

# After DFS has completely explored the current region, the fast trip home can
# view the same physical wall/target from the opposite travel direction.
# Re-scan ONLY cells that contain an unresolved glimpse or an already locked
# target observation, instead of repeating the full 10-pose sweep everywhere.
TARGET_RETURN_RESCAN_ENABLED = True
# Fine-grained switches.  Keep both True for maximum robustness; set the DFS
# one False if you want all second-look work postponed until exploration ends.
TARGET_DFS_BACKTRACK_RESCAN_ENABLED = True
TARGET_FAST_RETURN_RESCAN_ENABLED = True
TARGET_RETURN_RESCAN_GLIMPSES = True
TARGET_RETURN_RESCAN_LOCKED = True
TARGET_RETURN_RESCAN_MAX_ITEMS_PER_CELL = 5
TARGET_RETURN_RESCAN_OBSERVE_SEC = 0.30
TARGET_RETURN_RESCAN_LINGER_SEC = 0.32

# Targeted re-scan begins from the saved absolute bearing.  The return heading
# is automatically removed, so the same target is viewed from the new chassis
# orientation. Yaw may still be bracketed.  Pitch is selected dynamically:
# +/-45-ish yaw is capped at -15 deg, while FRONT and near-90 side views may
# still use the deep -22.5 deg level.
TARGET_RETURN_RESCAN_YAW_OFFSETS_DEG = (0.0, -10.0, +10.0)

# Multiple ranged observations of one confirmed target are fused with a median
# grid estimate.  This is deliberately robust to one bad ToF return.
TARGET_FUSION_MAX_SAMPLES = 12

# After a stable candidate is found, move it closer to image centre before
# recording ToF/bearing. This is a memory/measurement lock, not a firing lock.
TARGET_LOCK_MAX_STEPS = 3
TARGET_LOCK_CENTER_TOL_X = 0.055
TARGET_LOCK_CENTER_TOL_Y = 0.070
TARGET_LOCK_HFOV_DEG = 80.0
TARGET_LOCK_VFOV_DEG = 48.0
TARGET_LOCK_MAX_YAW_STEP_DEG = 16.0
TARGET_LOCK_MAX_PITCH_STEP_DEG = 10.0
TARGET_LOCK_YAW_LIMIT_DEG = 88.0
# Lock may fine-adjust while centering a confirmed candidate, but it is never
# allowed to tilt above the shallow -5 deg search posture.
TARGET_LOCK_PITCH_MIN_DEG = -25.0
TARGET_LOCK_PITCH_MAX_DEG = TARGET_SWEEP_PITCH_NORMAL_DEG
TARGET_LOCK_SETTLE_SEC = 0.10

# Dedupe only observations that are physically close. Identical color/shape
# targets elsewhere in the maze are intentionally allowed.
TARGET_DEDUPE_GRID_DIST = 0.55
TARGET_DEDUPE_BEARING_DEG = 22.0
TARGET_DEDUPE_SAME_CELL_ONLY_IF_NO_RANGE = True

# SDK marker detections are independent of HSV geometry targets. A current SDK
# marker suppresses overlapping red HSV boxes so the same red sign is not saved
# twice as two different target types.
SDK_MARKER_ENABLED = True
SDK_MARKER_TTL_SEC = 0.55
SDK_MARKER_CONFIRM_FRAMES = 2

TARGET_LATEST_JSON = Path("maps") / "latest_targets.json"


# ============================================================
# TARGET FIRE POLICY / INFRARED BLASTER
# ============================================================
# Target detection still records EVERY confirmed target.  Firing is a separate
# operator-selected policy configured from Mission Control before the robot moves.
# A target that is not selected is still mapped/saved, but can never reach fire().
TARGET_AUTO_FIRE_ENABLED = True
TARGET_FIRE_TYPE_NAME = "INFRARED"
TARGET_INFRARED_SHOTS = 1
TARGET_FIRE_ONCE_PER_TARGET = True
TARGET_FIRE_SETTLE_SEC = 0.12

# Assignment constraint: target must be fired from <= 2 floor tiles.
# 1 tile = 60 cm -> 2 tiles = 120 cm = 1200 mm.
# The FINAL fire gate re-samples ToF after target lock; stale memory range is
# never enough to authorize a shot.  Unknown range is blocked for safety.
TARGET_FIRE_MAX_TILES = 2.0
TARGET_FIRE_MAX_RANGE_MM = GRID_TILE_MM * TARGET_FIRE_MAX_TILES
TARGET_FIRE_RANGE_SAMPLES = 7
TARGET_FIRE_RANGE_RECHECK_SEC = 0.05

TARGET_FIRE_COLORS = ("RED", "YELLOW", "GREEN", "BLUE")
TARGET_FIRE_SHAPES = (
    "SQUARE",
    "RECT_VERTICAL",
    "RECT_HORIZONTAL",
    "CIRCLE",
)


# ============================================================
# PERSISTENT MAP / KNOWN-MAP NAVIGATION
# ============================================================

MAP_SCHEMA = "robomaster_dfs_grid_map"
MAP_SCHEMA_VERSION = 1
MAP_DIR = Path("maps")
MAP_LATEST_JSON = MAP_DIR / "latest_map.json"
MAP_LATEST_ASCII = MAP_DIR / "latest_map.txt"
MAP_LATEST_SVG = MAP_DIR / "latest_map.svg"

# Save latest_map.json/.txt/.svg after every meaningful topology change.
# This protects the learned map even if the run is interrupted later.
MAP_AUTOSAVE = True

# After the LAST unexplored frontier has been scanned, do NOT continue
# unwinding the DFS parent stack all the way home.  Plan a direct return
# through the completed map instead.
FAST_RETURN_HOME_AFTER_DFS = True
FAST_RETURN_MAX_REPLANS = 8

# Estimated action times used only by the home-route planner.
# They let the planner prefer a route that may have the same number of cells
# but fewer costly 90/180-degree turns.
FAST_RETURN_MOVE_EST_SEC = 4.0
FAST_RETURN_TURN_90_EST_SEC = 3.0
FAST_RETURN_TURN_180_EST_SEC = 5.0


# ============================================================
# TRANSIENT MOTION-ABORT RECOVERY
# ============================================================
# If a side-IR recovery cannot be cleared while the robot is still between
# cells, do not crash the whole mission and do not permanently mark that edge
# as a wall.  Back out along the just-traversed corridor, return close to the
# source cell, then rescan/replan.
MOTION_ABORT_RETREAT_SPEED_MPS = 0.10
MOTION_ABORT_HOME_TOL_M = 0.055
MOTION_ABORT_PROGRESS_EPS_M = 0.020
MOTION_ABORT_TIMEOUT_SEC = 6.0


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
# MISSION CONTROL GUI
# ============================================================

class MissionControlGUI:
    """
    Thread-isolated Tkinter mission-control window.

    The RoboMaster/DFS code remains on the mission thread.  Tkinter owns its
    own GUI thread and receives immutable snapshots through a Queue, so the UI
    never iterates live DFS dictionaries while they are being modified.

    During normal exploration the window is read-only.  After the robot has
    returned to START and one or more EXIT_CANDIDATE edges exist, the operator
    can select a specific candidate, preview its shortest confirmed route,
    then explicitly continue through that edge or finish the mission.
    """

    POLL_MS = 100

    def __init__(self, explorer):
        self.explorer = explorer
        self.messages = queue.Queue(maxsize=80)
        self.ready_event = threading.Event()
        self.closed_event = threading.Event()
        self.decision_event = threading.Event()
        self.target_policy_event = threading.Event()

        self.available = False
        self.thread = None
        self.start_error = None

        self.decision = None
        self.selected_candidate_key = None
        self.target_policy = None

        # GUI-thread-only fields are initialized in _run().
        self.root = None
        self.canvas = None
        self.status_var = None
        self.mode_var = None
        self.telemetry_var = None
        self.target_runtime_var = None
        self.fire_runtime_var = None
        self.fire_policy_summary_var = None
        self.target_vars = {}
        self.sdk_enabled_var = None
        self.sdk_labels_var = None
        self.auto_fire_var = None
        self.notebook = None
        self.targets_tab = None
        self.candidate_list = None
        self.candidate_detail_var = None
        self.continue_button = None
        self.finish_button = None
        self.estop_button = None

        self.latest_snapshot = None
        self.candidate_options = []
        self.candidate_by_index = []
        self.selected_option = None
        self.marker_hits = []
        self.mission_complete = False

    # --------------------------------------------------------
    # Public / mission-thread API
    # --------------------------------------------------------

    def start(self, timeout=3.0):
        if self.thread is not None:
            return self.available

        self.thread = threading.Thread(
            target=self._run,
            name='RoboMasterMissionControlGUI',
            daemon=True,
        )
        self.thread.start()
        self.ready_event.wait(timeout=max(0.2, float(timeout)))

        if not self.available:
            if self.start_error:
                print(f'[GUI WARN] Mission Control unavailable: {self.start_error}')
            else:
                print('[GUI WARN] Mission Control did not become ready; using terminal fallback.')

        return self.available

    def _post(self, kind, payload=None):
        if not self.available and kind != 'close':
            return False

        item = (kind, payload)
        try:
            self.messages.put_nowait(item)
            return True
        except queue.Full:
            # Keep the newest state.  Dropping an old visual snapshot is fine;
            # decision messages are retried after freeing one slot.
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(item)
                return True
            except queue.Full:
                return False

    def post_snapshot(self, snapshot):
        return self._post('snapshot', snapshot)

    def post_mission_complete(self, snapshot=None):
        if snapshot is not None:
            self.post_snapshot(snapshot)
        return self._post('mission_complete', None)

    def request_exit_decision(self, options, snapshot=None):
        """
        Block the mission thread while the GUI stays responsive.

        Returns:
            ('continue', ((x, y), dir_index))
            ('finish', None)
            None  -> GUI unavailable/closed, caller should use terminal fallback
        """
        if not self.available or self.closed_event.is_set():
            return None

        self.decision = None
        self.selected_candidate_key = None
        self.decision_event.clear()

        payload = {
            'options': options,
            'snapshot': snapshot,
        }
        if not self._post('exit_decision', payload):
            return None

        while self.explorer.running and self.available:
            if self.decision_event.wait(0.10):
                break

        if not self.decision_event.is_set():
            return None

        return self.decision, self.selected_candidate_key

    def close(self):
        if self.available:
            self._post('close', None)

    def wait_closed(self, timeout=None):
        return self.closed_event.wait(timeout=timeout)

    def request_target_fire_policy(self):
        """Wait until the operator explicitly arms a target-fire policy.

        The GUI is already visible before the robot moves.  This method blocks
        only the mission thread; Tk remains responsive on its own thread.
        """
        if not self.available or self.closed_event.is_set():
            return None

        while self.explorer.running and self.available:
            if self.target_policy_event.wait(0.10):
                break

        if not self.target_policy_event.is_set():
            return None
        return dict(self.target_policy or {})

    # --------------------------------------------------------
    # GUI thread
    # --------------------------------------------------------

    def _run(self):
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox

            self.tk = tk
            self.ttk = ttk
            self.messagebox = messagebox

            root = tk.Tk()
            self.root = root
            root.title('RoboMaster Maze Mission Control')
            root.geometry('1240x800')
            root.minsize(980, 660)
            root.configure(bg='#0d1117')

            style = ttk.Style(root)
            try:
                style.theme_use('clam')
            except Exception:
                pass

            style.configure('MC.TFrame', background='#0d1117')
            style.configure('Panel.TFrame', background='#161b22')
            style.configure(
                'MC.TLabel', background='#0d1117', foreground='#e6edf3',
                font=('Segoe UI', 10),
            )
            style.configure(
                'Title.TLabel', background='#0d1117', foreground='#f0f6fc',
                font=('Segoe UI Semibold', 16),
            )
            style.configure(
                'Status.TLabel', background='#161b22', foreground='#58a6ff',
                font=('Segoe UI Semibold', 11), padding=(10, 8),
            )
            style.configure(
                'Panel.TLabel', background='#161b22', foreground='#c9d1d9',
                font=('Segoe UI', 10),
            )
            style.configure(
                'Section.TLabel', background='#161b22', foreground='#f0f6fc',
                font=('Segoe UI Semibold', 11),
            )
            style.configure(
                'Hint.TLabel', background='#161b22', foreground='#8b949e',
                font=('Segoe UI', 9),
            )
            style.configure('Accent.TButton', font=('Segoe UI Semibold', 10), padding=(10, 8))
            style.configure('Danger.TButton', font=('Segoe UI Semibold', 10), padding=(10, 8))
            style.configure('Fire.TButton', font=('Segoe UI Semibold', 10), padding=(10, 8))
            style.configure('MC.TCheckbutton', background='#161b22', foreground='#e6edf3')
            style.map('MC.TCheckbutton', background=[('active', '#161b22')])
            style.configure('MC.TNotebook', background='#0d1117', borderwidth=0)
            style.configure('MC.TNotebook.Tab', padding=(14, 7), font=('Segoe UI Semibold', 10))

            root.columnconfigure(0, weight=1)
            root.rowconfigure(2, weight=1)

            header = ttk.Frame(root, style='MC.TFrame', padding=(16, 12, 16, 8))
            header.grid(row=0, column=0, sticky='ew')
            header.columnconfigure(1, weight=1)

            ttk.Label(
                header,
                text='RoboMaster Maze Mission Control',
                style='Title.TLabel',
            ).grid(row=0, column=0, sticky='w')

            self.mode_var = tk.StringVar(value='GUI READY - TARGETS NOT ARMED')
            ttk.Label(
                header,
                textvariable=self.mode_var,
                style='MC.TLabel',
                anchor='e',
            ).grid(row=0, column=1, sticky='e')

            self.status_var = tk.StringVar(
                value='Select the target types allowed to fire, then ARM before motion.'
            )
            ttk.Label(
                root,
                textvariable=self.status_var,
                style='Status.TLabel',
                anchor='w',
            ).grid(row=1, column=0, sticky='ew', padx=16, pady=(0, 8))

            self.notebook = ttk.Notebook(root, style='MC.TNotebook')
            self.notebook.grid(row=2, column=0, sticky='nsew', padx=16, pady=(0, 10))

            mission_tab = ttk.Frame(self.notebook, style='MC.TFrame')
            self.targets_tab = ttk.Frame(self.notebook, style='MC.TFrame')
            status_tab = ttk.Frame(self.notebook, style='MC.TFrame')
            self.notebook.add(mission_tab, text='MISSION / MAP')
            self.notebook.add(self.targets_tab, text='TARGET FIRE RULES')
            self.notebook.add(status_tab, text='LIVE STATUS')

            # ==============================================================
            # TAB 1: Mission / map
            # ==============================================================
            mission_tab.columnconfigure(0, weight=1)
            mission_tab.columnconfigure(1, weight=0)
            mission_tab.rowconfigure(0, weight=1)

            map_panel = ttk.Frame(mission_tab, style='Panel.TFrame', padding=8)
            map_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
            map_panel.columnconfigure(0, weight=1)
            map_panel.rowconfigure(1, weight=1)

            ttk.Label(
                map_panel, text='Live DFS Map', style='Section.TLabel'
            ).grid(row=0, column=0, sticky='w', padx=4, pady=(2, 8))

            self.canvas = tk.Canvas(
                map_panel,
                bg='#0b0f14',
                highlightthickness=1,
                highlightbackground='#30363d',
                bd=0,
            )
            self.canvas.grid(row=1, column=0, sticky='nsew')
            self.canvas.bind('<Configure>', lambda _e: self._draw_map())
            self.canvas.bind('<Button-1>', self._on_map_click)

            side = ttk.Frame(mission_tab, style='Panel.TFrame', padding=12, width=350)
            side.grid(row=0, column=1, sticky='ns')
            side.grid_propagate(False)
            side.columnconfigure(0, weight=1)

            ttk.Label(side, text='Exit Candidates', style='Section.TLabel').grid(
                row=0, column=0, sticky='w'
            )
            ttk.Label(
                side,
                text=(
                    'Available after the robot returns to START. Select an Exit '
                    'to preview the confirmed shortest route.'
                ),
                style='Panel.TLabel', wraplength=320, justify='left',
            ).grid(row=1, column=0, sticky='ew', pady=(4, 8))

            self.candidate_list = tk.Listbox(
                side, height=9, exportselection=False,
                bg='#0d1117', fg='#e6edf3',
                selectbackground='#1f6feb', selectforeground='white',
                highlightthickness=1, highlightbackground='#30363d',
                relief='flat', font=('Consolas', 10),
            )
            self.candidate_list.grid(row=2, column=0, sticky='ew')
            self.candidate_list.bind('<<ListboxSelect>>', self._on_list_select)

            self.candidate_detail_var = tk.StringVar(value='No Exit selection is active.')
            ttk.Label(
                side, textvariable=self.candidate_detail_var,
                style='Panel.TLabel', wraplength=320, justify='left',
            ).grid(row=3, column=0, sticky='ew', pady=(8, 8))

            self.continue_button = ttk.Button(
                side, text='CONTINUE VIA SELECTED EXIT', style='Accent.TButton',
                command=self._continue_selected, state='disabled',
            )
            self.continue_button.grid(row=4, column=0, sticky='ew', pady=(0, 6))

            self.finish_button = ttk.Button(
                side, text='FINISH MISSION AT START',
                command=self._finish_selected, state='disabled',
            )
            self.finish_button.grid(row=5, column=0, sticky='ew', pady=(0, 10))

            ttk.Separator(side, orient='horizontal').grid(
                row=6, column=0, sticky='ew', pady=(0, 10)
            )

            ttk.Label(side, text='Armed Fire Policy', style='Section.TLabel').grid(
                row=7, column=0, sticky='w'
            )
            self.fire_policy_summary_var = tk.StringVar(
                value='DISARMED\nNo target may fire until TARGET FIRE RULES is armed.'
            )
            ttk.Label(
                side, textvariable=self.fire_policy_summary_var,
                style='Panel.TLabel', wraplength=320, justify='left',
            ).grid(row=8, column=0, sticky='ew', pady=(4, 12))

            self.estop_button = ttk.Button(
                side, text='EMERGENCY STOP', style='Danger.TButton',
                command=self._emergency_stop,
            )
            self.estop_button.grid(row=9, column=0, sticky='ew')

            # ==============================================================
            # TAB 2: Target fire rules
            # ==============================================================
            self.targets_tab.columnconfigure(0, weight=1)
            target_outer = ttk.Frame(self.targets_tab, style='Panel.TFrame', padding=18)
            target_outer.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
            target_outer.columnconfigure(0, weight=1)

            ttk.Label(
                target_outer,
                text='Select targets that are ALLOWED to fire',
                style='Section.TLabel',
            ).grid(row=0, column=0, sticky='w')
            ttk.Label(
                target_outer,
                text=(
                    'Detection and mapping still record every confirmed target. '
                    'Only checked identities are allowed to trigger the blaster. '
                    'Firing mode is locked to INFRARED; no water shots are used. '
                    f'Final fire safety requires fresh ToF <= {TARGET_FIRE_MAX_RANGE_MM/10.0:.0f} cm '
                    '(2 tiles x 60 cm). Farther targets stay saved and are not fired.'
                ),
                style='Panel.TLabel', wraplength=900, justify='left',
            ).grid(row=1, column=0, sticky='ew', pady=(4, 12))

            matrix = ttk.Frame(target_outer, style='Panel.TFrame')
            matrix.grid(row=2, column=0, sticky='w')

            shape_titles = {
                'SQUARE': 'SQUARE',
                'RECT_VERTICAL': 'RECT V',
                'RECT_HORIZONTAL': 'RECT H',
                'CIRCLE': 'CIRCLE',
            }
            ttk.Label(matrix, text='COLOR', style='Section.TLabel').grid(
                row=0, column=0, sticky='w', padx=(0, 16), pady=(0, 6)
            )
            for col, shape in enumerate(TARGET_FIRE_SHAPES, start=1):
                ttk.Label(
                    matrix, text=shape_titles[shape], style='Section.TLabel'
                ).grid(row=0, column=col, padx=12, pady=(0, 6))

            for row, color in enumerate(TARGET_FIRE_COLORS, start=1):
                ttk.Label(matrix, text=color, style='Panel.TLabel').grid(
                    row=row, column=0, sticky='w', padx=(0, 16), pady=5
                )
                for col, shape in enumerate(TARGET_FIRE_SHAPES, start=1):
                    var = tk.BooleanVar(value=False)
                    self.target_vars[(color, shape)] = var
                    ttk.Checkbutton(
                        matrix,
                        variable=var,
                        style='MC.TCheckbutton',
                    ).grid(row=row, column=col, padx=20, pady=5)

            ttk.Separator(target_outer, orient='horizontal').grid(
                row=3, column=0, sticky='ew', pady=14
            )

            sdk_box = ttk.Frame(target_outer, style='Panel.TFrame')
            sdk_box.grid(row=4, column=0, sticky='ew')
            sdk_box.columnconfigure(1, weight=1)

            self.sdk_enabled_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                sdk_box,
                text='Allow RoboMaster SDK markers (red number / symbol markers)',
                variable=self.sdk_enabled_var,
                style='MC.TCheckbutton',
            ).grid(row=0, column=0, columnspan=2, sticky='w')

            ttk.Label(
                sdk_box,
                text='Optional SDK labels:',
                style='Panel.TLabel',
            ).grid(row=1, column=0, sticky='w', pady=(8, 0), padx=(24, 8))
            self.sdk_labels_var = tk.StringVar(value='')
            ttk.Entry(sdk_box, textvariable=self.sdk_labels_var).grid(
                row=1, column=1, sticky='ew', pady=(8, 0)
            )
            ttk.Label(
                sdk_box,
                text='Comma-separated. Leave blank = every SDK marker when enabled.',
                style='Hint.TLabel',
            ).grid(row=2, column=1, sticky='w', pady=(2, 0))

            self.auto_fire_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                target_outer,
                text='Auto-fire selected targets immediately after stable LOCK + center verification',
                variable=self.auto_fire_var,
                style='MC.TCheckbutton',
            ).grid(row=5, column=0, sticky='w', pady=(14, 4))

            ttk.Label(
                target_outer,
                text=(
                    'Safety rule: raw SEARCH / VERIFY detections never fire. '
                    'The target must first pass the existing temporal confirmation and final center gate.'
                ),
                style='Hint.TLabel', wraplength=900, justify='left',
            ).grid(row=6, column=0, sticky='w', pady=(0, 14))

            actions = ttk.Frame(target_outer, style='Panel.TFrame')
            actions.grid(row=7, column=0, sticky='ew')
            actions.columnconfigure(0, weight=1)
            actions.columnconfigure(1, weight=1)
            actions.columnconfigure(2, weight=1)
            actions.columnconfigure(3, weight=1)

            ttk.Button(
                actions, text='SELECT ALL', command=self._select_all_target_boxes
            ).grid(row=0, column=0, sticky='ew', padx=(0, 5))
            ttk.Button(
                actions, text='CLEAR ALL', command=self._clear_all_target_boxes
            ).grid(row=0, column=1, sticky='ew', padx=5)
            ttk.Button(
                actions, text='ARM SELECTED TARGETS', style='Accent.TButton',
                command=self._arm_selected_targets,
            ).grid(row=0, column=2, sticky='ew', padx=5)
            ttk.Button(
                actions, text='ARM FIRE ALL TARGETS', style='Fire.TButton',
                command=self._arm_all_targets,
            ).grid(row=0, column=3, sticky='ew', padx=(5, 0))

            self.target_policy_detail_var = tk.StringVar(
                value='DISARMED - mission will wait here before movement.'
            )
            ttk.Label(
                target_outer,
                textvariable=self.target_policy_detail_var,
                style='Status.TLabel', anchor='w',
            ).grid(row=8, column=0, sticky='ew', pady=(16, 0))

            # ==============================================================
            # TAB 3: Live status
            # ==============================================================
            status_tab.columnconfigure(0, weight=1)
            status_tab.rowconfigure(0, weight=1)
            status_panel = ttk.Frame(status_tab, style='Panel.TFrame', padding=18)
            status_panel.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
            status_panel.columnconfigure(0, weight=1)

            ttk.Label(status_panel, text='Robot / DFS', style='Section.TLabel').grid(
                row=0, column=0, sticky='w'
            )
            self.telemetry_var = tk.StringVar(value='No telemetry yet.')
            ttk.Label(
                status_panel, textvariable=self.telemetry_var,
                style='Panel.TLabel', wraplength=950, justify='left',
            ).grid(row=1, column=0, sticky='ew', pady=(4, 14))

            ttk.Separator(status_panel, orient='horizontal').grid(
                row=2, column=0, sticky='ew', pady=(0, 12)
            )
            ttk.Label(status_panel, text='Target Vision', style='Section.TLabel').grid(
                row=3, column=0, sticky='w'
            )
            self.target_runtime_var = tk.StringVar(value='Vision status unavailable.')
            ttk.Label(
                status_panel, textvariable=self.target_runtime_var,
                style='Panel.TLabel', wraplength=950, justify='left',
            ).grid(row=4, column=0, sticky='ew', pady=(4, 14))

            ttk.Separator(status_panel, orient='horizontal').grid(
                row=5, column=0, sticky='ew', pady=(0, 12)
            )
            ttk.Label(status_panel, text='Infrared Fire', style='Section.TLabel').grid(
                row=6, column=0, sticky='w'
            )
            self.fire_runtime_var = tk.StringVar(value='DISARMED')
            ttk.Label(
                status_panel, textvariable=self.fire_runtime_var,
                style='Panel.TLabel', wraplength=950, justify='left',
            ).grid(row=7, column=0, sticky='ew', pady=(4, 0))

            footer = ttk.Label(
                root,
                text=(
                    'Purple E# = EXIT_CANDIDATE   •   T# = locked target   •   '
                    'T#F = target already fired   •   Orange ? = pending glimpse   •   '
                    'Yellow triangle = robot'
                ),
                style='MC.TLabel', anchor='w',
            )
            footer.grid(row=3, column=0, sticky='ew', padx=16, pady=(0, 10))

            root.protocol('WM_DELETE_WINDOW', self._on_close)

            # The target rules page is intentionally the first page the operator
            # sees; mission motion will wait for an explicit ARM action.
            self.notebook.select(self.targets_tab)

            self.available = True
            self.ready_event.set()
            root.after(self.POLL_MS, self._process_messages)
            root.mainloop()

        except Exception as e:
            self.start_error = e
            self.available = False
            self.ready_event.set()
        finally:
            self.available = False
            if not self.decision_event.is_set():
                self.decision = 'finish'
                self.selected_candidate_key = None
                self.decision_event.set()
            if not self.target_policy_event.is_set():
                self.target_policy = None
                self.target_policy_event.set()

            try:
                self.canvas = None
                self.status_var = None
                self.mode_var = None
                self.telemetry_var = None
                self.target_runtime_var = None
                self.fire_runtime_var = None
                self.fire_policy_summary_var = None
                self.target_vars = {}
                self.sdk_enabled_var = None
                self.sdk_labels_var = None
                self.auto_fire_var = None
                self.notebook = None
                self.targets_tab = None
                self.candidate_list = None
                self.candidate_detail_var = None
                self.continue_button = None
                self.finish_button = None
                self.estop_button = None
                self.root = None
                import gc
                gc.collect()
            except Exception:
                pass

            self.closed_event.set()

    def _select_all_target_boxes(self):
        for var in self.target_vars.values():
            var.set(True)
        if self.sdk_enabled_var is not None:
            self.sdk_enabled_var.set(True)

    def _clear_all_target_boxes(self):
        for var in self.target_vars.values():
            var.set(False)
        if self.sdk_enabled_var is not None:
            self.sdk_enabled_var.set(False)
        if self.sdk_labels_var is not None:
            self.sdk_labels_var.set('')

    def _build_selected_target_policy(self, mode='selected'):
        selected = []
        for (color, shape), var in self.target_vars.items():
            try:
                checked = bool(var.get())
            except Exception:
                checked = False
            if checked:
                selected.append([color, shape])

        labels_raw = ''
        if self.sdk_labels_var is not None:
            try:
                labels_raw = str(self.sdk_labels_var.get())
            except Exception:
                labels_raw = ''
        sdk_labels = [
            item.strip() for item in labels_raw.split(',') if item.strip()
        ]

        sdk_enabled = False
        if self.sdk_enabled_var is not None:
            try:
                sdk_enabled = bool(self.sdk_enabled_var.get())
            except Exception:
                pass

        auto_fire = True
        if self.auto_fire_var is not None:
            try:
                auto_fire = bool(self.auto_fire_var.get())
            except Exception:
                pass

        return {
            'armed': True,
            'mode': str(mode),
            'fire_type': 'infrared',
            'auto_fire': bool(auto_fire),
            'selected_color_shapes': selected,
            'sdk_enabled': bool(sdk_enabled),
            'sdk_labels': sdk_labels,
        }

    @staticmethod
    def _policy_summary_text(policy):
        if not policy or not policy.get('armed'):
            return 'DISARMED'
        range_cm = float(policy.get('max_range_mm', TARGET_FIRE_MAX_RANGE_MM)) / 10.0
        if policy.get('mode') == 'all':
            return (
                'ARMED: ALL TARGETS\n'
                f'INFRARED only / one shot per confirmed target / range <= {range_cm:.0f} cm'
            )

        selected = list(policy.get('selected_color_shapes') or [])
        sdk_enabled = bool(policy.get('sdk_enabled'))
        sdk_labels = list(policy.get('sdk_labels') or [])
        parts = []
        if selected:
            parts.append(f'{len(selected)} color/shape identities')
        if sdk_enabled:
            parts.append('SDK=' + (','.join(sdk_labels) if sdk_labels else 'ALL'))
        if not parts:
            parts.append('NO TARGETS (safe no-fire policy)')
        return (
            'ARMED: ' + ' + '.join(parts)
            + '\nINFRARED only / auto-fire=' + str(bool(policy.get('auto_fire', True)))
            + f' / range <= {range_cm:.0f} cm'
        )

    def _apply_armed_policy(self, policy):
        self.target_policy = dict(policy)
        # Thread-safe explorer setter.  This also makes re-arming during a run
        # immediately update the gate rather than only changing GUI state.
        try:
            self.explorer.set_target_fire_policy(policy)
        except Exception as e:
            self.messagebox.showerror(
                'Fire policy error', f'Could not apply policy: {e}', parent=self.root
            )
            return

        summary = self._policy_summary_text(policy)
        if self.fire_policy_summary_var is not None:
            self.fire_policy_summary_var.set(summary)
        if hasattr(self, 'target_policy_detail_var') and self.target_policy_detail_var is not None:
            self.target_policy_detail_var.set(summary)
        self.mode_var.set('TARGET FIRE POLICY ARMED')
        self.status_var.set('Target fire policy armed. Mission may start / continue.')
        self.target_policy_event.set()

    def _arm_selected_targets(self):
        policy = self._build_selected_target_policy(mode='selected')
        selected_n = len(policy.get('selected_color_shapes') or [])
        sdk_on = bool(policy.get('sdk_enabled'))
        if selected_n == 0 and not sdk_on:
            ok = self.messagebox.askyesno(
                'Arm no-fire policy',
                'No targets are selected. Arm the mission with ALL FIRING BLOCKED?',
                parent=self.root,
            )
            if not ok:
                return
        self._apply_armed_policy(policy)

    def _arm_all_targets(self):
        ok = self.messagebox.askyesno(
            'Arm FIRE ALL targets',
            'Allow every confirmed color/shape target and every SDK marker to fire?\n\n'
            'This does NOT fire now. It arms the automatic INFRARED firing gate.',
            parent=self.root,
        )
        if not ok:
            return
        self._select_all_target_boxes()
        policy = self._build_selected_target_policy(mode='all')
        policy['sdk_enabled'] = True
        policy['sdk_labels'] = []
        policy['auto_fire'] = True
        if self.auto_fire_var is not None:
            self.auto_fire_var.set(True)
        self._apply_armed_policy(policy)

    def _process_messages(self):
        if not self.available or self.root is None:
            return

        processed = 0
        while processed < 30:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break

            processed += 1

            if kind == 'snapshot':
                self._apply_snapshot(payload)

            elif kind == 'exit_decision':
                snapshot = payload.get('snapshot') if isinstance(payload, dict) else None
                if snapshot is not None:
                    self._apply_snapshot(snapshot)
                options = payload.get('options', []) if isinstance(payload, dict) else []
                self._enter_exit_decision(options)

            elif kind == 'mission_complete':
                self.mission_complete = True
                self.mode_var.set('MISSION COMPLETE')
                self.status_var.set('Mission finished. Map remains available for inspection.')
                self.continue_button.configure(state='disabled')
                self.finish_button.configure(state='disabled')

            elif kind == 'close':
                try:
                    self.root.destroy()
                except Exception:
                    pass
                return

        if self.available and self.root is not None:
            self.root.after(self.POLL_MS, self._process_messages)

    def _apply_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return

        self.latest_snapshot = snapshot
        status = snapshot.get('status') or 'RUNNING'
        self.status_var.set(status)

        current = tuple(snapshot.get('current', (0, 0)))
        heading_idx = int(snapshot.get('heading', 0)) % 4
        heading = DIR_NAMES[heading_idx]
        visited_count = len(snapshot.get('visited', []))
        exit_count = len(snapshot.get('exit_candidates', []))
        target_count = len(snapshot.get('targets', []))
        fired_count = sum(
            1 for t in snapshot.get('targets', [])
            if t.get('fire_status') == 'FIRED_IR'
        )
        too_far_count = sum(
            1 for t in snapshot.get('targets', [])
            if t.get('fire_status') == 'WAITING_TOO_FAR'
        )
        glimpse_count = sum(
            1 for g in snapshot.get('target_glimpses', [])
            if not g.get('resolved_target_id')
        )
        pos = snapshot.get('position')
        pos_txt = 'n/a'
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                pos_txt = f'({float(pos[0]):+.2f}, {float(pos[1]):+.2f}) m'
            except Exception:
                pass

        if self.telemetry_var is not None:
            self.telemetry_var.set(
                f'Cell: {current}\n'
                f'Heading: {heading}\n'
                f'Visited: {visited_count}\n'
                f'Exit candidates: {exit_count}\n'
                f'Targets locked: {target_count}\n'
                f'Targets fired (IR): {fired_count}\n'
                f'Targets waiting >120cm: {too_far_count}\n'
                f'Pending glimpses: {glimpse_count}\n'
                f'Chassis odom XY: {pos_txt}'
            )

        if self.target_runtime_var is not None:
            self.target_runtime_var.set(str(snapshot.get('target_status', 'n/a')))

        policy = snapshot.get('target_fire_policy') or self.target_policy
        if policy:
            summary = self._policy_summary_text(policy)
            if self.fire_policy_summary_var is not None:
                self.fire_policy_summary_var.set(summary)

        if self.fire_runtime_var is not None:
            last_event = snapshot.get('last_fire_event', 'No fire event yet.')
            self.fire_runtime_var.set(
                f"{self._policy_summary_text(policy)}\n"
                f"Last event: {last_event}"
            )

        self._draw_map()

    def _enter_exit_decision(self, options):
        self.candidate_options = list(options or [])
        self.candidate_by_index = []
        self.selected_option = None
        self.candidate_list.delete(0, self.tk.END)

        for option in self.candidate_options:
            if not option.get('reachable', True):
                continue
            self.candidate_by_index.append(option)
            cid = option.get('id', '?')
            cell = tuple(option.get('cell', (0, 0)))
            direction = option.get('dir', '?')
            moves = option.get('moves')
            self.candidate_list.insert(
                self.tk.END,
                f'{cid:<3}  {cell!s:<10} -> {direction}   {moves} moves'
            )

        self.mode_var.set('WAITING FOR EXIT SELECTION')
        self.status_var.set(
            'Robot is safely at START. Select the EXIT_CANDIDATE you want to inspect.'
        )
        self.finish_button.configure(state='normal')
        self.continue_button.configure(state='disabled')

        if self.candidate_by_index:
            self.candidate_list.selection_set(0)
            self.candidate_list.activate(0)
            self._select_option(self.candidate_by_index[0])
        else:
            self.candidate_detail_var.set('No reachable EXIT_CANDIDATE.')

        self._draw_map()

    def _on_list_select(self, _event=None):
        selected = self.candidate_list.curselection()
        if not selected:
            return
        idx = int(selected[0])
        if 0 <= idx < len(self.candidate_by_index):
            self._select_option(self.candidate_by_index[idx])

    def _select_option(self, option):
        self.selected_option = option
        key = option.get('key')
        self.selected_candidate_key = key

        cid = option.get('id', '?')
        cell = tuple(option.get('cell', (0, 0)))
        direction = option.get('dir', '?')
        moves = option.get('moves', 0)
        distance_m = option.get('route_distance_m', 0.0)
        front = option.get('front_mm')
        reason = option.get('reason', 'unknown')
        path = option.get('path', [])
        front_txt = 'n/a' if front is None else f'{float(front):.0f} mm'

        self.candidate_detail_var.set(
            f'{cid}: source {cell} -> {direction}\n'
            f'Shortest confirmed route: {moves} cell moves\n'
            f'Approx. route to source: {distance_m:.2f} m\n'
            f'Front ToF when recorded: {front_txt}\n'
            f'Reason: {reason}\n'
            f'Path: {path}'
        )
        self.continue_button.configure(state='normal')

        # Mirror selection in listbox when the map marker was clicked.
        for i, item in enumerate(self.candidate_by_index):
            if item.get('key') == key:
                self.candidate_list.selection_clear(0, self.tk.END)
                self.candidate_list.selection_set(i)
                self.candidate_list.activate(i)
                self.candidate_list.see(i)
                break

        self._draw_map()

    def _continue_selected(self):
        if self.selected_option is None:
            return

        cid = self.selected_option.get('id', '?')
        cell = tuple(self.selected_option.get('cell', (0, 0)))
        direction = self.selected_option.get('dir', '?')

        ok = self.messagebox.askyesno(
            'Confirm EXIT traversal',
            f'Continue via {cid}: {cell} -> {direction}?\n\n'
            'The robot will first follow the displayed shortest confirmed route, '
            'then cross only this approved EXIT edge. Collision safety remains active.',
            parent=self.root,
        )
        if not ok:
            return

        self.decision = 'continue'
        self.selected_candidate_key = self.selected_option.get('key')
        self.continue_button.configure(state='disabled')
        self.finish_button.configure(state='disabled')
        self.mode_var.set('EXIT ROUTE APPROVED')
        self.status_var.set(f"Operator approved {cid}. Robot may leave START.")
        self.decision_event.set()

    def _finish_selected(self):
        ok = self.messagebox.askyesno(
            'Finish mission',
            'Finish the mission at START and do not traverse any deferred EXIT_CANDIDATE?',
            parent=self.root,
        )
        if not ok:
            return

        self.decision = 'finish'
        self.selected_candidate_key = None
        self.continue_button.configure(state='disabled')
        self.finish_button.configure(state='disabled')
        self.mode_var.set('FINISH SELECTED')
        self.status_var.set('Operator selected FINISH at START.')
        self.decision_event.set()

    def _emergency_stop(self):
        ok = self.messagebox.askyesno(
            'Emergency stop',
            'Stop the current mission?\n\nMotion loops will exit and the chassis will be stopped by cleanup.',
            parent=self.root,
        )
        if not ok:
            return

        self.explorer.running = False
        self.decision = 'finish'
        self.selected_candidate_key = None
        self.decision_event.set()
        if not self.target_policy_event.is_set():
            self.target_policy = None
            self.target_policy_event.set()
        self.mode_var.set('STOP REQUESTED')
        self.status_var.set('Emergency stop requested. Waiting for motion loop to stop...')
        self.continue_button.configure(state='disabled')
        self.finish_button.configure(state='disabled')

    def _on_close(self):
        if not self.mission_complete and self.explorer.running:
            ok = self.messagebox.askyesno(
                'Close Mission Control',
                'Closing Mission Control during a mission will request a safe stop. Continue?',
                parent=self.root,
            )
            if not ok:
                return
            self.explorer.running = False
            self.decision = 'finish'
            self.selected_candidate_key = None
            self.decision_event.set()
            if not self.target_policy_event.is_set():
                self.target_policy = None
                self.target_policy_event.set()

        try:
            self.root.destroy()
        except Exception:
            pass

    # --------------------------------------------------------
    # Live map drawing
    # --------------------------------------------------------

    def _on_map_click(self, event):
        if not self.candidate_by_index:
            return

        best = None
        best_d2 = None
        for x, y, option in self.marker_hits:
            d2 = (float(event.x) - x) ** 2 + (float(event.y) - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = option

        if best is not None and best_d2 is not None and best_d2 <= 28.0 ** 2:
            self._select_option(best)

    @staticmethod
    def _dir_vec_screen(direction):
        direction = int(direction) % 4
        if direction == 0:
            return (0, -1)
        if direction == 1:
            return (1, 0)
        if direction == 2:
            return (0, 1)
        return (-1, 0)

    def _draw_map(self):
        canvas = self.canvas
        snap = self.latest_snapshot
        if canvas is None:
            return

        canvas.delete('all')
        self.marker_hits = []

        if not isinstance(snap, dict):
            canvas.create_text(
                24, 24,
                anchor='nw',
                text='Waiting for map data...',
                fill='#8b949e',
                font=('Segoe UI', 12),
            )
            return

        cells = {tuple(c) for c in snap.get('cells', [])}
        if not cells:
            cells.add(tuple(snap.get('root', (0, 0))))

        root_cell = tuple(snap.get('root', (0, 0)))
        current = tuple(snap.get('current', root_cell))
        cells.add(root_cell)
        cells.add(current)

        exits = list(snap.get('exit_candidates', []))
        open_map = {
            tuple(item['cell']): set(int(d) for d in item.get('dirs', []))
            for item in snap.get('open_dirs', [])
        }
        blocked = {
            frozenset((tuple(edge[0]), tuple(edge[1])))
            for edge in snap.get('blocked_edges', [])
            if isinstance(edge, (list, tuple)) and len(edge) == 2
        }
        visited = {tuple(c) for c in snap.get('visited', [])}
        dead = {tuple(c) for c in snap.get('dead_end_cells', [])}

        candidate_keys = {
            (tuple(item.get('cell', (0, 0))), int(item.get('dir_index', 0)) % 4)
            for item in exits
        }

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        w = max(420, int(canvas.winfo_width()))
        h = max(420, int(canvas.winfo_height()))
        pad = 54
        cols = max(1, max_x - min_x + 1)
        rows = max(1, max_y - min_y + 1)
        cell_size = min(96.0, (w - 2 * pad) / cols, (h - 2 * pad) / rows)
        cell_size = max(38.0, cell_size)

        map_w = (max_x - min_x) * cell_size
        map_h = (max_y - min_y) * cell_size
        origin_x = (w - map_w) / 2.0
        origin_y = (h - map_h) / 2.0

        def center(cell):
            x, y = cell
            cx = origin_x + (x - min_x) * cell_size
            cy = origin_y + (max_y - y) * cell_size
            return cx, cy

        # Cells are drawn as a true edge-to-edge grid.  The previous GUI used
        # smaller node boxes plus graph-link lines between their centres, which
        # made the maze look like a graph instead of the physical 60 cm grid.
        # With half == cell_size / 2 every neighbouring cell touches exactly at
        # its shared boundary; OPEN edges are gaps in the wall, not connector
        # lines between nodes.
        half = cell_size * 0.50

        # Cells + wall segments.
        entrance_dir = int(snap.get('known_entrance_dir', 2)) % 4
        ingress_dir = int(snap.get('known_maze_ingress_dir', 0)) % 4

        for cell in sorted(cells, key=lambda p: (p[1], p[0])):
            cx, cy = center(cell)
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

            if cell == root_cell:
                fill = '#173d2a'
            elif cell == current:
                fill = '#5a4714'
            elif cell in dead:
                fill = '#4a2024'
            elif cell in visited:
                fill = '#172b42'
            else:
                fill = '#21262d'

            # Edge-to-edge tile.  A very thin neutral outline keeps individual
            # cells readable while preserving the continuous grid appearance.
            # Confirmed walls are drawn afterward with a much heavier stroke.
            canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=fill,
                outline='#30363d',
                width=1,
            )
            canvas.create_text(
                cx, cy + half - 12,
                text=f'({cell[0]},{cell[1]})',
                fill='#8b949e',
                font=('Consolas', max(8, int(cell_size * 0.11))),
            )

            opens = open_map.get(cell, set())
            for d in range(4):
                vx, vy = DIR_VEC[d]
                nb = (cell[0] + vx, cell[1] + vy)
                edge = frozenset((cell, nb))

                special_open = (
                    (cell, d) in candidate_keys
                    or (cell == root_cell and d == entrance_dir)
                    or (cell == root_cell and d == ingress_dir)
                )
                is_open = (d in opens and edge not in blocked) or special_open
                if is_open:
                    continue

                if d == 0:
                    coords = (x1, y1, x2, y1)
                elif d == 1:
                    coords = (x2, y1, x2, y2)
                elif d == 2:
                    coords = (x1, y2, x2, y2)
                else:
                    coords = (x1, y1, x1, y2)
                canvas.create_line(*coords, fill='#f0f6fc', width=max(2, int(cell_size * 0.055)))

        # Route overlay is intentionally drawn AFTER the grid cells/walls.
        # With edge-to-edge cells there is no inter-node gap anymore, so drawing
        # this underneath the cells would hide the route completely.
        # During normal autonomous motion this comes from the
        # explorer snapshot (return-home / navigation route). During EXIT
        # selection the locally selected candidate preview takes priority.
        selected_path = [tuple(c) for c in snap.get('route_preview', [])]
        selected_dir = None
        if self.selected_option is not None:
            selected_path = [tuple(c) for c in self.selected_option.get('path', [])]
            selected_dir = self.selected_option.get('dir_index')

        if len(selected_path) >= 2:
            pts = []
            for cell in selected_path:
                pts.extend(center(cell))
            canvas.create_line(
                *pts,
                fill='#58a6ff',
                width=7,
                capstyle='round',
                joinstyle='round',
            )

        if selected_path and selected_dir is not None:
            sx, sy = center(selected_path[-1])
            dx, dy = self._dir_vec_screen(selected_dir)
            ex = sx + dx * cell_size * 0.72
            ey = sy + dy * cell_size * 0.72
            canvas.create_line(
                sx, sy, ex, ey,
                fill='#58a6ff', width=7, arrow=self.tk.LAST,
                arrowshape=(12, 14, 6),
            )

        # Blocked edges as red X.
        for edge in blocked:
            pts = list(edge)
            if len(pts) != 2 or pts[0] not in cells or pts[1] not in cells:
                continue
            ax, ay = center(pts[0])
            bx, by = center(pts[1])
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            s = 8
            canvas.create_line(mx - s, my - s, mx + s, my + s, fill='#f85149', width=3)
            canvas.create_line(mx - s, my + s, mx + s, my - s, fill='#f85149', width=3)

        # Exit markers. Prefer the decision-option IDs when available.
        option_by_key = {
            option.get('key'): option for option in self.candidate_options
        }

        for idx, item in enumerate(exits, start=1):
            cell = tuple(item.get('cell', (0, 0)))
            d = int(item.get('dir_index', 0)) % 4
            key = (cell, d)
            option = option_by_key.get(key)
            cid = option.get('id') if option else f'E{idx}'

            cx, cy = center(cell)
            dx, dy = self._dir_vec_screen(d)
            mx = cx + dx * half
            my = cy + dy * half
            ox = cx + dx * cell_size * 0.55
            oy = cy + dy * cell_size * 0.55

            selected = (
                self.selected_option is not None
                and self.selected_option.get('key') == key
            )
            color = '#d2a8ff' if selected else '#a371f7'
            radius = 14 if selected else 11
            canvas.create_line(mx, my, ox, oy, fill=color, width=4, arrow=self.tk.LAST)
            canvas.create_oval(
                ox - radius, oy - radius, ox + radius, oy + radius,
                fill=color, outline='#ffffff', width=2,
            )
            canvas.create_text(
                ox, oy,
                text=str(cid),
                fill='#0d1117',
                font=('Segoe UI Semibold', 8),
            )
            if option is not None:
                self.marker_hits.append((ox, oy, option))

        # Locked target markers. Prefer the ranged grid estimate when it is
        # locally plausible; otherwise show a bearing marker near the source
        # cell so a bad/long ToF return cannot throw the GUI scale off.
        target_color_ui = {
            'RED': '#ff5f56',
            'YELLOW': '#f2cc60',
            'GREEN': '#3fb950',
            'BLUE': '#58a6ff',
        }

        for rec in snap.get('targets', []):
            try:
                source = tuple(rec.get('source_cell', root_cell))
                sx, sy = center(source)
                tx, ty = sx, sy

                est = rec.get('estimated_grid_xy')
                use_est = False
                if isinstance(est, (list, tuple)) and len(est) >= 2:
                    gx, gy = float(est[0]), float(est[1])
                    grid_dist = math.hypot(gx - source[0], gy - source[1])
                    if grid_dist <= 1.8:
                        tx, ty = center((gx, gy))
                        use_est = True

                if not use_est:
                    bearing = rec.get('bearing_deg_from_north')
                    if bearing is not None:
                        rad = math.radians(float(bearing))
                        tx = sx + math.sin(rad) * cell_size * 0.38
                        ty = sy - math.cos(rad) * cell_size * 0.38

                tid = str(rec.get('id', 'T?'))
                if rec.get('fire_status') == 'FIRED_IR':
                    tid += 'F'
                if rec.get('kind') == 'SDK_MARKER':
                    fill = '#d2a8ff'
                    r = max(8, int(cell_size * 0.10))
                    canvas.create_polygon(
                        tx, ty - r,
                        tx + r, ty,
                        tx, ty + r,
                        tx - r, ty,
                        fill=fill,
                        outline='#ffffff',
                        width=2,
                    )
                else:
                    fill = target_color_ui.get(rec.get('color'), '#ffffff')
                    r = max(8, int(cell_size * 0.10))
                    shape = rec.get('shape')
                    if shape == 'CIRCLE':
                        canvas.create_oval(
                            tx - r, ty - r, tx + r, ty + r,
                            fill=fill, outline='#ffffff', width=2,
                        )
                    elif shape == 'RECT_VERTICAL':
                        canvas.create_rectangle(
                            tx - r * 0.65, ty - r,
                            tx + r * 0.65, ty + r,
                            fill=fill, outline='#ffffff', width=2,
                        )
                    elif shape == 'RECT_HORIZONTAL':
                        canvas.create_rectangle(
                            tx - r, ty - r * 0.65,
                            tx + r, ty + r * 0.65,
                            fill=fill, outline='#ffffff', width=2,
                        )
                    else:
                        canvas.create_rectangle(
                            tx - r, ty - r, tx + r, ty + r,
                            fill=fill, outline='#ffffff', width=2,
                        )

                canvas.create_text(
                    tx,
                    ty,
                    text=tid,
                    fill='#0d1117',
                    font=('Segoe UI Semibold', 8),
                )
            except Exception:
                continue

        # Unresolved one/few-frame target glimpses.  These are NOT confirmed
        # targets; the orange G? marker only tells the operator that the return
        # pass has a direction worth re-checking.
        for rec in snap.get('target_glimpses', []):
            if rec.get('resolved_target_id'):
                continue
            try:
                source = tuple(rec.get('source_cell', root_cell))
                sx, sy = center(source)
                bearing = rec.get('bearing_deg_from_north')
                if bearing is None:
                    continue
                rad = math.radians(float(bearing))
                gx = sx + math.sin(rad) * cell_size * 0.30
                gy = sy - math.cos(rad) * cell_size * 0.30
                r = max(7, int(cell_size * 0.085))
                canvas.create_oval(
                    gx - r, gy - r, gx + r, gy + r,
                    fill='#d29922', outline='#ffffff', width=2,
                )
                canvas.create_text(
                    gx, gy,
                    text='?',
                    fill='#0d1117',
                    font=('Segoe UI Semibold', 9),
                )
            except Exception:
                continue

        # Root marker.
        rx, ry = center(root_cell)
        canvas.create_text(
            rx, ry - 2,
            text='S',
            fill='#7ee787',
            font=('Segoe UI Semibold', max(12, int(cell_size * 0.20))),
        )

        # Robot heading triangle.
        cx, cy = center(current)
        heading = int(snap.get('heading', 0)) % 4
        dx, dy = self._dir_vec_screen(heading)
        px, py = -dy, dx
        tip_x = cx + dx * cell_size * 0.25
        tip_y = cy + dy * cell_size * 0.25
        back_x = cx - dx * cell_size * 0.15
        back_y = cy - dy * cell_size * 0.15
        side = cell_size * 0.14
        points = (
            tip_x, tip_y,
            back_x + px * side, back_y + py * side,
            back_x - px * side, back_y - py * side,
        )
        canvas.create_polygon(
            points,
            fill='#f2cc60',
            outline='#ffffff',
            width=2,
        )

        # North indicator.
        canvas.create_text(26, 22, text='N', fill='#e6edf3', font=('Segoe UI Semibold', 12))
        canvas.create_line(26, 56, 26, 32, fill='#e6edf3', width=3, arrow=self.tk.LAST)


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
        self.camera = None
        self.vision = None
        self.blaster = None

        self.state = SharedState()

        # ----------------------------------------------------
        # Target vision / target memory state
        # ----------------------------------------------------
        self.target_vision_enabled = bool(TARGET_VISION_ENABLED)
        self.vision_available = False
        self.vision_stream_started = False
        self.preview_enabled = bool(TARGET_PREVIEW_ENABLED)
        self.vision_thread = None
        self.vision_lock = threading.Lock()
        self.latest_frame_shape = None
        self.latest_detections = []

        # Latest dynamic foam-wall horizon used by both HSV and SDK targets.
        # profile_px is a float32 array of image y-values (NaN = unknown).
        self.latest_foam_profile_px = None
        self.latest_foam_profile_t = 0.0
        self.latest_foam_profile_coverage = 0.0
        self.latest_foam_components = []

        self.detection_history = deque(maxlen=120)
        self.sdk_marker_history = deque(maxlen=120)
        self.latest_sdk_markers = []
        self.detected_targets = []
        self.target_id_seq = 0
        self.target_scanned_cells = set()
        self.target_last_scan_t = {}
        self.target_status_text = "VISION OFF"

        # Operator-selected fire policy. Detection/memory and firing are kept
        # deliberately separate so unselected targets are still mapped but are
        # physically blocked from reaching the blaster API.
        self.fire_policy_lock = threading.Lock()
        self.target_fire_policy = {
            "armed": False,
            "mode": "selected",
            "fire_type": "infrared",
            "max_range_mm": float(TARGET_FIRE_MAX_RANGE_MM),
            "auto_fire": bool(TARGET_AUTO_FIRE_ENABLED),
            "selected_color_shapes": [],
            "sdk_enabled": False,
            "sdk_labels": [],
        }
        self.fired_target_ids = set()
        self.last_fire_event = "DISARMED - waiting for operator target selection"

        # Weak/brief observations that did not yet satisfy temporal lock.
        # These are intentionally remembered so the return pass can look again
        # from the opposite travel direction instead of losing a one-frame hit.
        self.target_glimpses = []
        self.target_glimpse_seq = 0
        self.target_return_rescanned = set()

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

        # Deferred wide-open boundary directions.
        # key = (cell_tuple, absolute_direction_index)
        self.exit_candidates = {}

        # Cells reached only after the operator explicitly approves crossing
        # a deferred EXIT_CANDIDATE. The first cell beyond an approved exit
        # is allowed to be scanned as a real continuation area instead of
        # being immediately rolled back by the OPEN-AREA TRAP fallback.
        # All later edges still use the normal exit guards.
        self.approved_exit_entry_cells = set()

        # Optional live Mission Control GUI.  It receives immutable snapshots
        # from this mission thread and never reads mutable DFS containers
        # directly.
        self.mission_gui = None
        self.gui_status_text = "INITIALIZING"
        self.gui_route_preview = []
        self.operator_selected_exit_candidate = None

        # Start heading is N (0), so root-back is the physical known entrance.
        # It remains excluded from DFS but is drawn/saved explicitly.
        self.known_entrance_dir = REL_BACK % 4
        self.known_maze_ingress_dir = START_MAZE_INGRESS_DIR % 4
        self.entrance_corridor_profile = None
        self.start_anchor_profile = None

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

        # Distinguish a transient mid-edge safety abort from a confirmed wall.
        # DFS uses this to rescan the SAME source cell instead of corrupting
        # the map by permanently blocking the edge.
        self.motion_replan_requested = False
        self.motion_replan_reason = None
        self.motion_exit_candidate_detected = False
        self.transient_runtime_blocked_edges = set()
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
    # TARGET FIRE POLICY / INFRARED BLASTER
    # --------------------------------------------------------

    def get_target_fire_policy(self):
        with self.fire_policy_lock:
            policy = dict(self.target_fire_policy)
            policy["selected_color_shapes"] = [
                list(item) for item in self.target_fire_policy.get("selected_color_shapes", [])
            ]
            policy["sdk_labels"] = list(self.target_fire_policy.get("sdk_labels", []))
            return policy

    def set_target_fire_policy(self, policy):
        """Validate and atomically install the operator-selected fire gate."""
        policy = dict(policy or {})
        mode = str(policy.get("mode", "selected")).lower()
        if mode not in ("selected", "all"):
            mode = "selected"

        selected = []
        seen = set()
        for item in policy.get("selected_color_shapes", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            color = str(item[0]).upper().strip()
            shape = str(item[1]).upper().strip()
            key = (color, shape)
            if color not in TARGET_FIRE_COLORS or shape not in TARGET_FIRE_SHAPES:
                continue
            if key in seen:
                continue
            seen.add(key)
            selected.append([color, shape])

        sdk_labels = []
        seen_labels = set()
        for value in policy.get("sdk_labels", []) or []:
            label = str(value).strip()
            if not label:
                continue
            folded = label.casefold()
            if folded in seen_labels:
                continue
            seen_labels.add(folded)
            sdk_labels.append(label)

        clean = {
            "armed": bool(policy.get("armed", True)),
            "mode": mode,
            "fire_type": "infrared",
            "max_range_mm": float(TARGET_FIRE_MAX_RANGE_MM),
            "auto_fire": bool(policy.get("auto_fire", True)),
            "selected_color_shapes": selected,
            "sdk_enabled": bool(policy.get("sdk_enabled", False)),
            "sdk_labels": sdk_labels,
        }

        if mode == "all":
            clean["sdk_enabled"] = True
            clean["sdk_labels"] = []

        with self.fire_policy_lock:
            self.target_fire_policy = clean

        summary = self._target_fire_policy_summary(clean)
        self.last_fire_event = f"POLICY ARMED: {summary}"
        print(f"[FIRE POLICY] {summary}")
        return clean

    @staticmethod
    def _target_fire_policy_summary(policy):
        if not policy or not policy.get("armed"):
            return "DISARMED"
        range_txt = f"range<={TARGET_FIRE_MAX_RANGE_MM/10.0:.0f}cm"
        if policy.get("mode") == "all":
            return f"ALL TARGETS / IR / {range_txt}"
        selected_n = len(policy.get("selected_color_shapes", []) or [])
        sdk = bool(policy.get("sdk_enabled"))
        labels = list(policy.get("sdk_labels", []) or [])
        sdk_txt = "OFF"
        if sdk:
            sdk_txt = "ALL" if not labels else ",".join(labels)
        return (
            f"SELECTED color/shape={selected_n}, SDK={sdk_txt}, IR, "
            f"range<={TARGET_FIRE_MAX_RANGE_MM/10.0:.0f}cm"
        )

    def target_matches_fire_policy(self, candidate):
        """Return (allowed, reason) without changing robot state."""
        policy = self.get_target_fire_policy()
        if not policy.get("armed"):
            return False, "POLICY_DISARMED"
        if not policy.get("auto_fire", True):
            return False, "AUTO_FIRE_OFF"
        if policy.get("mode") == "all":
            return True, "ALL_TARGETS"

        kind = str(candidate.get("kind", ""))
        if kind == "COLOR_SHAPE":
            key = (
                str(candidate.get("color", "")).upper(),
                str(candidate.get("shape", "")).upper(),
            )
            selected = {
                (str(item[0]).upper(), str(item[1]).upper())
                for item in policy.get("selected_color_shapes", [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            }
            if key in selected:
                return True, "SELECTED_COLOR_SHAPE"
            return False, "COLOR_SHAPE_NOT_SELECTED"

        if kind == "SDK_MARKER":
            if not policy.get("sdk_enabled"):
                return False, "SDK_DISABLED"
            allowed_labels = [
                str(v).casefold() for v in policy.get("sdk_labels", []) if str(v).strip()
            ]
            if not allowed_labels:
                return True, "SDK_ALL_LABELS"
            label = str(candidate.get("label", "")).casefold()
            if label in allowed_labels:
                return True, "SDK_LABEL_SELECTED"
            return False, "SDK_LABEL_NOT_SELECTED"

        return False, "UNKNOWN_TARGET_KIND"

    def _maybe_fire_locked_target(self, record):
        """Final physical firing gate. Called only after a stable target LOCK."""
        if not isinstance(record, dict):
            return False

        tid = str(record.get("id") or "")
        allowed, reason = self.target_matches_fire_policy(record)
        record["fire_policy_match"] = bool(allowed)
        record["fire_policy_reason"] = str(reason)

        if not allowed:
            # Keep an already-fired record as fired when it is re-seen later.
            if record.get("fire_status") != "FIRED_IR":
                record["fire_status"] = "SKIPPED_NOT_SELECTED"
            self.last_fire_event = f"SKIP {tid or '?'}: {reason}"
            print(f"[FIRE BLOCK] {tid or '?'} -> {reason}")
            return False

        if TARGET_FIRE_ONCE_PER_TARGET and (
            tid in self.fired_target_ids or record.get("fire_status") == "FIRED_IR"
        ):
            self.last_fire_event = f"SKIP {tid}: already fired"
            print(f"[FIRE SKIP] {tid}: already fired")
            return False

        # ----------------------------------------------------
        # FINAL ASSIGNMENT RANGE GATE: <= 2 tiles = <= 1200 mm
        # ----------------------------------------------------
        # Do not trust an old target-memory distance.  The target has just been
        # centered/revalidated, so stop and take a fresh median ToF sample at
        # the actual firing pose.  If range is unknown or too far, keep the
        # target in memory but DO NOT shoot; a later closer re-acquisition may
        # still fire it.
        self.stop_chassis()
        time.sleep(max(0.0, float(TARGET_FIRE_RANGE_RECHECK_SEC)))
        live_range_mm = self.sample_tof_median(
            samples=max(3, int(TARGET_FIRE_RANGE_SAMPLES))
        )
        record["fire_range_limit_mm"] = float(TARGET_FIRE_MAX_RANGE_MM)
        record["fire_range_mm"] = (
            None if live_range_mm is None else float(live_range_mm)
        )

        if live_range_mm is None:
            record["fire_status"] = "BLOCKED_RANGE_UNKNOWN"
            record["fire_policy_reason"] = "RANGE_UNKNOWN"
            self.last_fire_event = (
                f"BLOCK {tid or '?'}: range unknown; "
                f"need <= {TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
            )
            print(
                f"[FIRE RANGE BLOCK] {tid or '?'}: NO ToF RANGE -> "
                f"must confirm <= {TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
            )
            return False

        if live_range_mm > TARGET_FIRE_MAX_RANGE_MM:
            record["fire_status"] = "WAITING_TOO_FAR"
            record["fire_policy_reason"] = "TARGET_TOO_FAR"
            self.last_fire_event = (
                f"WAIT {tid or '?'}: {live_range_mm:.0f} mm > "
                f"{TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
            )
            print(
                f"[FIRE RANGE BLOCK] {tid or '?'}: "
                f"{live_range_mm:.0f} mm ({live_range_mm/10.0:.1f} cm) > "
                f"2 tiles / {TARGET_FIRE_MAX_RANGE_MM:.0f} mm -> NO FIRE"
            )
            return False

        # Save the live legal firing distance separately from the mapping ToF.
        print(
            f"[FIRE RANGE OK] {tid or '?'}: "
            f"{live_range_mm:.0f} mm ({live_range_mm/10.0:.1f} cm) <= "
            f"{TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
        )

        if self.blaster is None:
            record["fire_status"] = "BLOCKED_NO_BLASTER"
            self.last_fire_event = f"BLOCK {tid or '?'}: blaster unavailable"
            print(f"[FIRE BLOCK] {tid or '?'}: blaster unavailable")
            return False

        # Target lock already centers/revalidates the candidate. Stop the base
        # once more here so a future caller cannot accidentally fire in motion.
        self.stop_chassis()

        try:
            ok = self.blaster.fire(
                fire_type=blaster.INFRARED_FIRE,
                times=max(1, int(TARGET_INFRARED_SHOTS)),
            )
            if ok is False:
                raise RuntimeError("RoboMaster blaster.fire() returned False")
        except Exception as e:
            record["fire_status"] = "FIRE_FAILED"
            record["fire_error"] = str(e)
            self.last_fire_event = f"FAIL {tid or '?'}: {e}"
            print(f"[FIRE ERROR] {tid or '?'}: {e}")
            return False

        when = datetime.now().isoformat(timespec="milliseconds")
        record["fire_status"] = "FIRED_IR"
        record["fired_at"] = when
        record["fire_type"] = "INFRARED"
        record["fire_range_rule"] = "<=2_tiles_120cm"
        record["fire_times"] = max(1, int(TARGET_INFRARED_SHOTS))
        record["fire_count"] = int(record.get("fire_count", 0)) + 1
        if tid:
            self.fired_target_ids.add(tid)

        desc = (
            f"{record.get('color')} {record.get('shape')}"
            if record.get("kind") == "COLOR_SHAPE"
            else f"SDK {record.get('label')}"
        )
        self.last_fire_event = f"FIRED IR {tid or '?'} {desc}"
        self.target_status_text = f"FIRED IR {tid or '?'} {desc}"
        print(f"[FIRE IR] {tid or '?'} {desc}")
        time.sleep(max(0.0, float(TARGET_FIRE_SETTLE_SEC)))
        return True

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
    # TARGET VISION / TARGET MEMORY
    # --------------------------------------------------------

    @staticmethod
    def _norm_bbox_iou(a, b):
        """IoU for normalized [x, y, w, h] boxes."""
        try:
            ax, ay, aw, ah = [float(v) for v in a]
            bx, by, bw, bh = [float(v) for v in b]
        except Exception:
            return 0.0

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 1e-9 else 0.0

    @staticmethod
    def _angle_diff_deg(a, b):
        return abs(wrap_deg(float(a) - float(b)))

    def _roi_contains_norm_center(self, cx, cy):
        return (
            TARGET_ROI_X_MIN <= float(cx) <= TARGET_ROI_X_MAX
            and TARGET_ROI_Y_MIN <= float(cy) <= TARGET_ROI_Y_MAX
        )

    def _compute_foam_wall_profile(self, frame):
        """Estimate the visible top edge of the white foam maze wall.

        Returns:
            profile_px : np.ndarray shape (frame_width,), y pixel per x column.
                         NaN means the wall top is unknown at that x position.
            coverage   : fraction of the configured target ROI with a valid y.
            components : accepted foam-like component bounding boxes for debug.

        The detector intentionally prefers a conservative FAIL-OPEN policy.
        A missing profile does not by itself reject a target; it only removes
        the extra spatial evidence for that candidate.
        """
        if (
            not TARGET_FOAM_GATE_ENABLED
            or cv2 is None
            or np is None
            or frame is None
        ):
            return None, 0.0, []

        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 1 or frame_w <= 1:
            return None, 0.0, []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(TARGET_FOAM_HSV_LOW, dtype=np.uint8),
            np.array(TARGET_FOAM_HSV_HIGH, dtype=np.uint8),
        )

        open_k = max(1, int(TARGET_FOAM_OPEN_KERNEL))
        close_k = max(1, int(TARGET_FOAM_CLOSE_KERNEL))
        if open_k % 2 == 0:
            open_k += 1
        if close_k % 2 == 0:
            close_k += 1

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((open_k, open_k), dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        profile = np.full(frame_w, np.nan, dtype=np.float32)
        components = []
        min_bottom_y = float(frame_h) * float(TARGET_FOAM_COMPONENT_MIN_BOTTOM_FRAC)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, bw, bh = cv2.boundingRect(contour)

            if area < TARGET_FOAM_MIN_COMPONENT_AREA_PX:
                continue
            if bw < TARGET_FOAM_MIN_COMPONENT_WIDTH_PX:
                continue
            if bh < TARGET_FOAM_MIN_COMPONENT_HEIGHT_PX:
                continue
            if (y + bh) < min_bottom_y:
                # Ceiling panels/lights may be white, but they do not extend
                # into the maze-wall/floor part of the image.
                continue

            components.append({
                "bbox_px": [int(x), int(y), int(bw), int(bh)],
                "area_px": area,
            })

            # Fill only this component in a local mask. RETR_EXTERNAL plus
            # filled contour also closes small holes caused by colored targets
            # mounted on the foam face, preserving the true wall top.
            local = np.zeros((bh, bw), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(local, [shifted], -1, 255, thickness=-1)

            for local_x in range(bw):
                ys = np.flatnonzero(local[:, local_x])
                if ys.size == 0:
                    continue

                px = x + local_x
                top_y = float(y + int(ys[0]))
                if not math.isfinite(float(profile[px])) or top_y < float(profile[px]):
                    profile[px] = top_y

        # Bridge only SHORT missing spans. This repairs seams/sign holes without
        # inventing a wall across a wide corridor opening.
        valid_idx = np.flatnonzero(np.isfinite(profile))
        max_gap = max(0, int(TARGET_FOAM_PROFILE_MAX_INTERP_GAP_PX))
        if valid_idx.size >= 2 and max_gap > 0:
            for left_i, right_i in zip(valid_idx[:-1], valid_idx[1:]):
                gap = int(right_i - left_i - 1)
                if gap <= 0 or gap > max_gap:
                    continue
                profile[left_i:right_i + 1] = np.linspace(
                    float(profile[left_i]),
                    float(profile[right_i]),
                    int(right_i - left_i + 1),
                    dtype=np.float32,
                )

        # Robust 1-D median smoothing while preserving unknown regions.
        radius = max(0, int(TARGET_FOAM_PROFILE_MEDIAN_RADIUS_PX))
        if radius > 0 and np.isfinite(profile).any():
            src = profile.copy()
            smooth = profile.copy()
            for px in np.flatnonzero(np.isfinite(src)):
                lo = max(0, int(px) - radius)
                hi = min(frame_w, int(px) + radius + 1)
                vals = src[lo:hi]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    smooth[int(px)] = float(np.median(vals))
            profile = smooth

        roi_x1 = max(0, min(frame_w - 1, int(round(TARGET_ROI_X_MIN * frame_w))))
        roi_x2 = max(roi_x1 + 1, min(frame_w, int(round(TARGET_ROI_X_MAX * frame_w))))
        roi_slice = profile[roi_x1:roi_x2]
        coverage = (
            float(np.count_nonzero(np.isfinite(roi_slice))) / float(max(1, roi_slice.size))
        )

        return profile, coverage, components

    def _foam_profile_snapshot(self):
        """Thread-safe copy of the latest wall-top estimate."""
        with self.vision_lock:
            profile = self.latest_foam_profile_px
            if profile is not None:
                profile = profile.copy()
            return (
                profile,
                float(self.latest_foam_profile_t),
                float(self.latest_foam_profile_coverage),
                [dict(c) for c in self.latest_foam_components],
                tuple(self.latest_frame_shape) if self.latest_frame_shape else None,
            )

    @staticmethod
    def _foam_wall_y_at_px(profile_px, x_px):
        if profile_px is None or np is None:
            return None

        try:
            width = int(len(profile_px))
            x = int(round(float(x_px)))
        except Exception:
            return None

        if width <= 0 or x < 0 or x >= width:
            return None

        radius = max(0, int(TARGET_FOAM_PROFILE_SAMPLE_RADIUS_PX))
        lo = max(0, x - radius)
        hi = min(width, x + radius + 1)
        vals = np.asarray(profile_px[lo:hi], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _target_passes_foam_gate_px(
        self,
        frame_w,
        frame_h,
        cx_px,
        cy_px,
        bbox_px,
        profile_px=None,
        profile_t=None,
    ):
        """Return (pass_bool, local_wall_y_px, reason, below_fraction)."""
        if not TARGET_FOAM_GATE_ENABLED:
            return True, None, "foam-gate-disabled", 1.0

        if profile_px is None:
            (
                profile_px,
                snap_t,
                coverage,
                _components,
                _shape,
            ) = self._foam_profile_snapshot()
            profile_t = snap_t
        else:
            coverage = None

        now = time.monotonic()
        if profile_t is not None and profile_t > 0.0:
            if now - float(profile_t) > TARGET_FOAM_PROFILE_TTL_SEC:
                if TARGET_FOAM_FAIL_CLOSED:
                    return False, None, "foam-profile-stale", 0.0
                return True, None, "foam-profile-stale-fail-open", 1.0

        if profile_px is None:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-profile-missing", 0.0
            return True, None, "foam-profile-missing-fail-open", 1.0

        # A very low global coverage means the frame probably does not expose a
        # usable foam horizon. Do not trust a tiny accidental white component.
        if coverage is not None and coverage < TARGET_FOAM_MIN_ROI_COVERAGE:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-profile-low-coverage", 0.0
            return True, None, "foam-profile-low-coverage-fail-open", 1.0

        wall_y = self._foam_wall_y_at_px(profile_px, cx_px)
        if wall_y is None:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-local-profile-missing", 0.0
            return True, None, "foam-local-profile-missing-fail-open", 1.0

        try:
            x, y, bw, bh = [float(v) for v in bbox_px]
            cy = float(cy_px)
        except Exception:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, wall_y, "foam-invalid-bbox", 0.0
            return True, wall_y, "foam-invalid-bbox-fail-open", 1.0

        bh = max(1.0, bh)
        cutoff_y = float(wall_y) + float(TARGET_FOAM_CENTER_MARGIN_PX)
        bbox_bottom = y + bh
        below_height = max(0.0, bbox_bottom - max(y, cutoff_y))
        below_fraction = clamp(below_height / bh, 0.0, 1.0)

        if cy < cutoff_y:
            return False, wall_y, "candidate-centre-above-foam", below_fraction

        if below_fraction < TARGET_FOAM_MIN_BBOX_BELOW_FRAC:
            return False, wall_y, "candidate-mostly-above-foam", below_fraction

        return True, wall_y, "foam-pass", below_fraction

    def _target_passes_foam_gate_norm(self, cx, cy, bbox_norm):
        """Foam gate for RoboMaster SDK markers expressed in normalized coords."""
        (
            profile_px,
            profile_t,
            coverage,
            _components,
            shape,
        ) = self._foam_profile_snapshot()

        if shape is None or len(shape) < 2:
            if TARGET_FOAM_FAIL_CLOSED and TARGET_FOAM_GATE_ENABLED:
                return False, None, "foam-frame-shape-missing", 0.0
            return True, None, "foam-frame-shape-missing-fail-open", 1.0

        frame_h, frame_w = int(shape[0]), int(shape[1])
        try:
            bx, by, bw, bh = [float(v) for v in bbox_norm]
            bbox_px = [
                bx * frame_w,
                by * frame_h,
                bw * frame_w,
                bh * frame_h,
            ]
            cx_px = float(cx) * frame_w
            cy_px = float(cy) * frame_h
        except Exception:
            if TARGET_FOAM_FAIL_CLOSED and TARGET_FOAM_GATE_ENABLED:
                return False, None, "foam-sdk-normalization-error", 0.0
            return True, None, "foam-sdk-normalization-error-fail-open", 1.0

        # Apply the same minimum-coverage trust rule used for color targets.
        if (
            TARGET_FOAM_GATE_ENABLED
            and profile_px is not None
            and coverage < TARGET_FOAM_MIN_ROI_COVERAGE
        ):
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-profile-low-coverage", 0.0
            return True, None, "foam-profile-low-coverage-fail-open", 1.0

        return self._target_passes_foam_gate_px(
            frame_w,
            frame_h,
            cx_px,
            cy_px,
            bbox_px,
            profile_px=profile_px,
            profile_t=profile_t,
        )

    def sdk_marker_callback(self, marker_info):
        """
        RoboMaster SDK visual-marker callback.

        Each SDK marker is kept as a separate target class from HSV targets.
        The callback is intentionally lightweight; temporal confirmation is
        performed later by the mission thread.
        """
        now = time.monotonic()
        parsed = []

        try:
            items = marker_info or []
            for raw in items:
                if raw is None or len(raw) < 5:
                    continue

                cx, cy, w, h = [float(v) for v in raw[:4]]
                label = str(raw[4])

                if not self._roi_contains_norm_center(cx, cy):
                    continue

                # SDK marker callback gives normalized center + width/height.
                x = cx - w * 0.5
                y = cy - h * 0.5
                bbox_norm = [x, y, w, h]

                foam_ok, foam_y, foam_reason, foam_below_frac = (
                    self._target_passes_foam_gate_norm(cx, cy, bbox_norm)
                )
                if not foam_ok:
                    continue

                parsed.append({
                    "kind": "SDK_MARKER",
                    "label": label,
                    "center_norm": [cx, cy],
                    "bbox_norm": bbox_norm,
                    "score": 1.0,
                    "foam_gate": foam_reason,
                    "foam_wall_y_px": foam_y,
                    "foam_bbox_below_frac": foam_below_frac,
                })
        except Exception:
            parsed = []

        with self.vision_lock:
            self.latest_sdk_markers = parsed
            self.sdk_marker_history.append((now, parsed))

    def _current_sdk_markers(self):
        now = time.monotonic()
        with self.vision_lock:
            markers = [dict(m) for m in self.latest_sdk_markers]
            history = list(self.sdk_marker_history)

        # If the SDK stopped reporting, do not let stale markers suppress red
        # HSV detections forever.
        if history and now - history[-1][0] <= SDK_MARKER_TTL_SEC:
            return markers
        return []

    def _target_roi_px(self, frame):
        h, w = frame.shape[:2]
        x1 = int(round(TARGET_ROI_X_MIN * w))
        x2 = int(round(TARGET_ROI_X_MAX * w))
        y1 = int(round(TARGET_ROI_Y_MIN * h))
        y2 = int(round(TARGET_ROI_Y_MAX * h))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return x1, y1, x2, y2

    def _classify_target_contour(self, contour):
        area = float(cv2.contourArea(contour))
        if area <= 1.0:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1.0:
            return None

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        if solidity < TARGET_MIN_SOLIDITY:
            return None

        approx = cv2.approxPolyDP(
            contour,
            TARGET_POLY_EPS_FRAC * perimeter,
            True,
        )
        x, y, w, h = cv2.boundingRect(contour)
        fill = area / float(max(1, w * h))
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        shape = None
        quality = 0.0

        # Rectangle family.
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(float)
            corner_cos = []

            for i in range(4):
                prev_p = pts[(i - 1) % 4]
                cur_p = pts[i]
                next_p = pts[(i + 1) % 4]

                v1 = prev_p - cur_p
                v2 = next_p - cur_p
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom <= 1e-9:
                    corner_cos.append(1.0)
                else:
                    corner_cos.append(
                        abs(float(np.dot(v1, v2)) / denom)
                    )

            max_corner_cos = max(corner_cos) if corner_cos else 1.0

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if (
                rw > 1.0
                and rh > 1.0
                and fill >= TARGET_RECT_MIN_FILL
                and max_corner_cos <= TARGET_RECT_MAX_CORNER_COS
            ):
                aspect_rot = max(rw, rh) / max(1e-6, min(rw, rh))

                if (
                    TARGET_SQUARE_ASPECT_MIN
                    <= aspect_rot
                    <= TARGET_SQUARE_ASPECT_MAX
                ):
                    shape = "SQUARE"
                else:
                    # Use the image-axis box only to decide vertical/horizontal.
                    # minAreaRect can swap axes as its angle crosses 45 degrees.
                    axis_aspect = float(w) / max(1.0, float(h))
                    if axis_aspect >= TARGET_RECT_ASPECT_MIN:
                        shape = "RECT_HORIZONTAL"
                    elif (1.0 / max(axis_aspect, 1e-6)) >= TARGET_RECT_ASPECT_MIN:
                        shape = "RECT_VERTICAL"

                quality = (
                    0.45 * solidity
                    + 0.35 * min(1.0, fill)
                    + 0.20 * (1.0 - min(1.0, max_corner_cos))
                )

        # Circle family.
        elif len(approx) >= 5 and circularity >= TARGET_CIRCLE_MIN_CIRCULARITY:
            axis_aspect = float(w) / max(1.0, float(h))
            if TARGET_CIRCLE_ASPECT_MIN <= axis_aspect <= TARGET_CIRCLE_ASPECT_MAX:
                shape = "CIRCLE"
                quality = 0.55 * solidity + 0.45 * min(1.0, circularity)

        if shape is None:
            return None

        return {
            "shape": shape,
            "solidity": solidity,
            "fill": fill,
            "circularity": circularity,
            "quality": quality,
            "bbox_local": [int(x), int(y), int(w), int(h)],
            "area": area,
        }

    def _detect_color_targets(self, frame, foam_profile=None):
        if cv2 is None or np is None or frame is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = self._target_roi_px(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        roi_h, roi_w = roi.shape[:2]
        roi_area = float(max(1, roi_w * roi_h))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        k_open = np.ones((3, 3), dtype=np.uint8)
        k_close = np.ones((5, 5), dtype=np.uint8)
        sdk_markers = self._current_sdk_markers()

        detections = []

        for color_name, ranges in TARGET_HSV_RANGES.items():
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

            for lo, hi in ranges:
                lo_np = np.array(lo, dtype=np.uint8)
                hi_np = np.array(hi, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo_np, hi_np))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                area = float(cv2.contourArea(contour))
                area_frac = area / roi_area

                if (
                    area_frac < TARGET_MIN_AREA_FRAC_ROI
                    or area_frac > TARGET_MAX_AREA_FRAC_ROI
                ):
                    continue

                classified = self._classify_target_contour(contour)
                if classified is None:
                    continue

                lx, ly, bw, bh = classified["bbox_local"]

                # Strong rejection for a giant clipped region such as a yellow
                # wall entering the ROI from an edge.
                if (
                    lx <= TARGET_BORDER_MARGIN_PX
                    or ly <= TARGET_BORDER_MARGIN_PX
                    or lx + bw >= roi_w - TARGET_BORDER_MARGIN_PX
                    or ly + bh >= roi_h - TARGET_BORDER_MARGIN_PX
                ):
                    continue

                gx, gy = x1 + lx, y1 + ly
                cx = gx + bw * 0.5
                cy = gy + bh * 0.5
                cx_n = cx / float(frame_w)
                cy_n = cy / float(frame_h)

                if not self._roi_contains_norm_center(cx_n, cy_n):
                    continue

                foam_ok, foam_y, foam_reason, foam_below_frac = (
                    self._target_passes_foam_gate_px(
                        frame_w,
                        frame_h,
                        cx,
                        cy,
                        [gx, gy, bw, bh],
                        profile_px=foam_profile,
                        profile_t=(time.monotonic() if foam_profile is not None else None),
                    )
                )
                if not foam_ok:
                    continue

                bbox_norm = [
                    gx / float(frame_w),
                    gy / float(frame_h),
                    bw / float(frame_w),
                    bh / float(frame_h),
                ]

                # If the RoboMaster SDK already recognizes a red marker in the
                # same place, trust the SDK identity and suppress the duplicate
                # generic RED geometric target.
                if color_name == "RED":
                    overlap = max(
                        [
                            self._norm_bbox_iou(bbox_norm, m.get("bbox_norm", []))
                            for m in sdk_markers
                        ]
                        or [0.0]
                    )
                    if overlap >= 0.18:
                        continue

                quality = float(classified["quality"])
                score = clamp(
                    0.72 * quality
                    + 0.28 * min(1.0, area_frac / 0.025),
                    0.0,
                    1.0,
                )

                detections.append({
                    "kind": "COLOR_SHAPE",
                    "color": color_name,
                    "shape": classified["shape"],
                    "center_norm": [cx_n, cy_n],
                    "bbox_norm": bbox_norm,
                    "bbox_px": [int(gx), int(gy), int(bw), int(bh)],
                    "area_frac_roi": area_frac,
                    "solidity": classified["solidity"],
                    "fill": classified["fill"],
                    "circularity": classified["circularity"],
                    "score": score,
                    "foam_gate": foam_reason,
                    "foam_wall_y_px": foam_y,
                    "foam_bbox_below_frac": foam_below_frac,
                })

        detections.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return detections

    def _draw_target_overlay(self, frame, detections):
        if cv2 is None or frame is None:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = self._target_roi_px(out)

        (
            foam_profile,
            foam_t,
            foam_coverage,
            foam_components,
            _foam_shape,
        ) = self._foam_profile_snapshot()
        foam_age = time.monotonic() - foam_t if foam_t > 0.0 else float("inf")
        foam_fresh = (
            foam_profile is not None
            and foam_age <= TARGET_FOAM_PROFILE_TTL_SEC
        )
        foam_usable = (
            TARGET_FOAM_GATE_ENABLED
            and foam_fresh
            and foam_coverage >= TARGET_FOAM_MIN_ROI_COVERAGE
        )

        # Shade the part of the search ROI that is geometrically impossible:
        # anything above the detected foam-wall top.  This is only a preview;
        # the actual rejection happens in _target_passes_foam_gate_px().
        if foam_fresh and np is not None:
            shade = out.copy()
            for px in range(x1, min(x2, len(foam_profile))):
                wy = float(foam_profile[px])
                if not math.isfinite(wy):
                    continue
                cutoff = int(round(wy + TARGET_FOAM_CENTER_MARGIN_PX))
                cutoff = max(y1, min(y2, cutoff))
                if cutoff > y1:
                    cv2.line(
                        shade,
                        (px, y1),
                        (px, cutoff),
                        (0, 0, 80),
                        1,
                    )
            out = cv2.addWeighted(out, 0.80, shade, 0.20, 0.0)

            # Draw each contiguous known profile run without connecting across
            # corridor gaps where the wall top is genuinely unknown.
            run = []
            for px in range(x1, min(x2, len(foam_profile))):
                wy = float(foam_profile[px])
                if math.isfinite(wy):
                    run.append((px, int(round(wy))))
                else:
                    if len(run) >= 2:
                        cv2.polylines(
                            out,
                            [np.asarray(run, dtype=np.int32)],
                            False,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    run = []
            if len(run) >= 2:
                cv2.polylines(
                    out,
                    [np.asarray(run, dtype=np.int32)],
                    False,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        gate_text = (
            f"SEARCH ROI | FOAM GATE {'LOCK' if foam_usable else 'FAIL-OPEN'} "
            f"cov={foam_coverage * 100.0:.0f}%"
        )
        cv2.putText(
            out,
            gate_text,
            (x1 + 4, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        bgr_by_color = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 220, 0),
            "BLUE": (255, 100, 0),
        }

        for d in detections:
            x, y, bw, bh = d.get("bbox_px", [0, 0, 0, 0])
            color = bgr_by_color.get(d.get("color"), (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            fire_ok, _fire_reason = self.target_matches_fire_policy(d)
            fire_tag = "FIRE" if fire_ok else "SKIP"
            label = (
                f"{fire_tag} {d.get('color','?')} {d.get('shape','?')} "
                f"{d.get('score',0.0):.2f}"
            )
            cv2.putText(
                out,
                label,
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        for marker in self._current_sdk_markers():
            bx, by, bw, bh = marker.get("bbox_norm", [0, 0, 0, 0])
            x = int(bx * w)
            y = int(by * h)
            ww = int(bw * w)
            hh = int(bh * h)
            cv2.rectangle(out, (x, y), (x + ww, y + hh), (255, 0, 255), 2)
            marker_for_policy = dict(marker)
            marker_for_policy["kind"] = "SDK_MARKER"
            fire_ok, _fire_reason = self.target_matches_fire_policy(marker_for_policy)
            fire_tag = "FIRE" if fire_ok else "SKIP"
            cv2.putText(
                out,
                f"{fire_tag} SDK {marker.get('label','?')}",
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        try:
            gp, gy = self.current_gimbal_relative()
        except Exception:
            gp, gy = None, None

        pose_txt = (
            f"cell={self.current} heading={DIR_NAMES[self.heading]} "
            f"gimbal=({gy if gy is not None else 'n/a'},"
            f"{gp if gp is not None else 'n/a'}) "
            f"saved={len(self.detected_targets)}"
        )

        cv2.putText(
            out,
            pose_txt,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            self.target_status_text[:110],
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        return out

    def _vision_loop(self):
        print("[VISION] frame thread started")

        while self.running and self.vision_stream_started:
            try:
                frame = self.camera.read_cv2_image(
                    strategy="newest",
                    timeout=0.6,
                )
            except Exception as e:
                if self.running and self.vision_stream_started:
                    print(f"[VISION WARN] frame read failed: {e}")
                    time.sleep(0.08)
                continue

            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()

            # Estimate the dynamic white-foam horizon ONCE for this frame.
            # Both the HSV detector and asynchronous SDK-marker callback use
            # this same geometry constraint.
            foam_profile, foam_coverage, foam_components = (
                self._compute_foam_wall_profile(frame)
            )
            with self.vision_lock:
                self.latest_frame_shape = tuple(frame.shape[:2])
                self.latest_foam_profile_px = (
                    foam_profile.copy() if foam_profile is not None else None
                )
                self.latest_foam_profile_t = now
                self.latest_foam_profile_coverage = float(foam_coverage)
                self.latest_foam_components = [dict(c) for c in foam_components]

            detections = self._detect_color_targets(
                frame,
                foam_profile=foam_profile,
            )

            gimbal_p, gimbal_y = self.current_gimbal_relative()
            snap_detections = []

            for d in detections:
                item = dict(d)
                item["cell"] = [int(self.current[0]), int(self.current[1])]
                item["heading"] = int(self.heading) % 4
                item["gimbal_yaw_deg"] = gimbal_y
                item["gimbal_pitch_deg"] = gimbal_p
                snap_detections.append(item)

            with self.vision_lock:
                self.latest_detections = snap_detections
                self.detection_history.append((now, snap_detections))

            if self.preview_enabled:
                try:
                    annotated = self._draw_target_overlay(frame, snap_detections)
                    cv2.imshow(TARGET_WINDOW_NAME, annotated)
                    key = cv2.waitKey(1) & 0xFF

                    # q/ESC closes only the preview, not the robot mission.
                    if key in (ord("q"), 27):
                        self.preview_enabled = False
                        try:
                            cv2.destroyWindow(TARGET_WINDOW_NAME)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[VISION PREVIEW WARN] {e}")
                    self.preview_enabled = False

        try:
            if cv2 is not None:
                cv2.destroyWindow(TARGET_WINDOW_NAME)
        except Exception:
            pass

        print("[VISION] frame thread stopped")

    def start_target_vision(self):
        if not self.target_vision_enabled:
            self.target_status_text = "VISION DISABLED BY CONFIG"
            return False

        if cv2 is None or np is None:
            self.target_status_text = "VISION OFF: install opencv-python + numpy"
            print(
                "[VISION WARN] OpenCV/NumPy unavailable. "
                "DFS continues without target detection."
            )
            return False

        if self.camera is None:
            self.target_status_text = "VISION OFF: camera unavailable"
            return False

        try:
            from robomaster import camera as rm_camera
            resolution = getattr(
                rm_camera,
                "STREAM_360P",
                TARGET_CAMERA_RESOLUTION,
            )
        except Exception:
            resolution = TARGET_CAMERA_RESOLUTION

        try:
            self.camera.start_video_stream(
                display=False,
                resolution=resolution,
            )
            self.vision_stream_started = True
        except Exception as e:
            self.target_status_text = f"VISION STREAM ERROR: {e}"
            print(f"[VISION WARN] cannot start camera stream: {e}")
            return False

        if SDK_MARKER_ENABLED and self.vision is not None:
            try:
                ok = self.vision.sub_detect_info(
                    name="marker",
                    color="red",
                    callback=self.sdk_marker_callback,
                )
                print(f"[VISION] SDK marker subscription = {ok}")
            except Exception as e:
                print(f"[VISION WARN] SDK marker detector unavailable: {e}")

        self.preview_enabled = bool(self.preview_enabled and TARGET_PREVIEW_ENABLED)
        self.vision_available = True
        self.target_status_text = "VISION READY - waiting for target sweep"

        self.vision_thread = threading.Thread(
            target=self._vision_loop,
            name="RoboMasterTargetVision",
            daemon=True,
        )
        self.vision_thread.start()
        print("[VISION] target detector ready")
        return True

    def stop_target_vision(self):
        self.vision_available = False
        self.vision_stream_started = False

        if SDK_MARKER_ENABLED and self.vision is not None:
            try:
                self.vision.unsub_detect_info(name="marker")
            except Exception:
                pass

        if self.camera is not None:
            try:
                self.camera.stop_video_stream()
            except Exception:
                pass

        if self.vision_thread is not None and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)

        try:
            if cv2 is not None:
                cv2.destroyWindow(TARGET_WINDOW_NAME)
        except Exception:
            pass

    def _fresh_target_frame_count(self, since_t):
        """Number of camera frames processed after *since_t*."""
        with self.vision_lock:
            return sum(1 for ts, _ in self.detection_history if ts >= float(since_t))

    def _sampled_pose_candidates(self, window_start, expected_item=None):
        """Collect one bounded target-observation window.

        This intentionally follows the Round-1 style used in the friend's
        launcher: sample several frames, require repeated observations, and
        separate the confidence that is merely interesting from the confidence
        that is strong enough to save.

        Returns:
            {
              "stable_colors": [...],
              "stable_markers": [...],
              "weak": [...],
              "sampled_frames": int,
            }
        """
        window_start = float(window_start)
        deadline = time.monotonic() + TARGET_SAMPLE_WINDOW_TIMEOUT_SEC

        # Wait for fresh camera evidence instead of sleeping a fixed duration.
        # If the camera is slower than expected, timeout keeps DFS responsive.
        while self.running and time.monotonic() < deadline:
            if self._fresh_target_frame_count(window_start) >= TARGET_SAMPLE_FRAMES:
                break
            time.sleep(TARGET_SAMPLE_POLL_SEC)

        with self.vision_lock:
            color_history = [
                (ts, [dict(d) for d in dets])
                for ts, dets in self.detection_history
                if ts >= window_start
            ]
            marker_history = [
                (ts, [dict(m) for m in markers])
                for ts, markers in self.sdk_marker_history
                if ts >= window_start
            ]

        # Cap the color window to the first N fresh camera frames.  This makes
        # "10 samples" deterministic even if the processing thread runs quickly.
        color_history = color_history[:TARGET_SAMPLE_FRAMES]
        sampled_frames = len(color_history)

        # ----- Color/shape tracks -----
        grouped = {}
        for ts, dets in color_history:
            per_frame = {}
            for d in dets:
                if d.get("kind") != "COLOR_SHAPE":
                    continue
                if float(d.get("score", 0.0)) < TARGET_MIN_CONFIDENCE:
                    continue
                if expected_item is not None and not self._match_candidate_to_item(d, expected_item):
                    continue

                key = ("COLOR_SHAPE", d.get("color"), d.get("shape"))
                prev = per_frame.get(key)
                if prev is None or float(d.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    per_frame[key] = d

            for key, d in per_frame.items():
                grouped.setdefault(key, []).append((ts, d))

        stable_colors = []
        weak = []

        for key, items in grouped.items():
            if not items:
                continue

            centers_x = [float(d["center_norm"][0]) for _, d in items]
            centers_y = [float(d["center_norm"][1]) for _, d in items]
            areas = [float(d.get("area_frac_roi", 0.0)) for _, d in items]
            scores = [float(d.get("score", 0.0)) for _, d in items]

            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
            center_std = math.hypot(cx_std, cy_std)

            area_mean = statistics.fmean(areas) if areas else 0.0
            area_std = statistics.pstdev(areas) if len(areas) > 1 else 0.0
            area_cv = area_std / max(area_mean, 1e-6)

            mean_score = statistics.fmean(scores) if scores else 0.0
            best = dict(max(items, key=lambda pair: float(pair[1].get("score", 0.0)))[1])
            best["sample_hits"] = len(items)
            best["sampled_frames"] = sampled_frames
            best["sample_mean_confidence"] = mean_score
            best["confirm_frames"] = len(items)
            best["center_std"] = center_std
            best["area_cv"] = area_cv
            best["temporal_score"] = mean_score
            weak.append(best)

            if (
                len(items) >= TARGET_VERIFY_FRAMES
                and mean_score >= TARGET_SAVE_CONFIDENCE
                and center_std <= TARGET_CONFIRM_MAX_CENTER_STD
                and area_cv <= TARGET_CONFIRM_MAX_AREA_CV
            ):
                stable_colors.append(best)

        # ----- RoboMaster SDK marker tracks -----
        marker_grouped = {}
        for ts, markers in marker_history:
            seen_this_callback = {}
            for m in markers:
                mm = dict(m)
                mm["kind"] = "SDK_MARKER"
                if expected_item is not None and not self._match_candidate_to_item(mm, expected_item):
                    continue
                key = ("SDK_MARKER", str(mm.get("label", "?")))
                seen_this_callback[key] = mm
            for key, m in seen_this_callback.items():
                marker_grouped.setdefault(key, []).append((ts, m))

        stable_markers = []
        for key, items in marker_grouped.items():
            if not items:
                continue

            centers_x = [float(m["center_norm"][0]) for _, m in items]
            centers_y = [float(m["center_norm"][1]) for _, m in items]
            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
            center_std = math.hypot(cx_std, cy_std)

            best = dict(items[-1][1])
            best["sample_hits"] = len(items)
            best["sampled_frames"] = sampled_frames
            best["sample_mean_confidence"] = 1.0
            best["confirm_frames"] = len(items)
            best["temporal_score"] = 1.0
            weak.append(best)

            if (
                len(items) >= TARGET_VERIFY_FRAMES
                and center_std <= TARGET_CONFIRM_MAX_CENTER_STD
            ):
                stable_markers.append(best)

        stable_colors.sort(
            key=lambda d: (
                int(d.get("sample_hits", 0)),
                float(d.get("sample_mean_confidence", 0.0)),
            ),
            reverse=True,
        )
        stable_markers.sort(
            key=lambda d: int(d.get("sample_hits", 0)),
            reverse=True,
        )
        weak.sort(
            key=lambda d: (
                int(d.get("sample_hits", 0)),
                float(d.get("sample_mean_confidence", d.get("score", 0.0))),
            ),
            reverse=True,
        )

        return {
            "stable_colors": stable_colors,
            "stable_markers": stable_markers,
            "weak": weak,
            "sampled_frames": sampled_frames,
        }

    def _observe_target_pose_windowed(
        self,
        cell,
        pose_index,
        total_poses,
        yaw_deg,
        pitch_deg,
        expected_item=None,
        phase="explore",
    ):
        """Observe one stationary gimbal pose using up to three sample windows.

        No evidence -> leave after one window.
        Weak evidence -> HOLD this exact pose and collect another window.
        Stable evidence -> return immediately so target lock can begin.
        """
        hold_started = time.monotonic()
        best_weak = {}

        for window_no in range(1, TARGET_HOLD_MAX_WINDOWS + 1):
            if not self.running:
                break
            if window_no > 1 and (time.monotonic() - hold_started) >= TARGET_HOLD_MAX_SEC:
                break

            self.target_status_text = (
                f"{phase.upper()} SAMPLE cell={tuple(cell)} "
                f"pose={pose_index}/{total_poses} "
                f"window={window_no}/{TARGET_HOLD_MAX_WINDOWS} "
                f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}"
            )
            self.publish_gui_state()

            window_start = time.monotonic()
            result = self._sampled_pose_candidates(
                window_start,
                expected_item=expected_item,
            )

            for cand in result["weak"]:
                ident = self._candidate_identity_key(cand)
                prev = best_weak.get(ident)
                cand_rank = (
                    int(cand.get("sample_hits", 0)),
                    float(cand.get("sample_mean_confidence", cand.get("score", 0.0))),
                )
                prev_rank = (
                    int(prev.get("sample_hits", 0)),
                    float(prev.get("sample_mean_confidence", prev.get("score", 0.0))),
                ) if prev is not None else (-1, -1.0)
                if prev is None or cand_rank > prev_rank:
                    best_weak[ident] = dict(cand)

            if result["stable_markers"] or result["stable_colors"]:
                self.target_status_text = (
                    f"{phase.upper()} VERIFIED cell={tuple(cell)} "
                    f"window={window_no} "
                    f"marker={len(result['stable_markers'])} "
                    f"color={len(result['stable_colors'])}"
                )
                self.publish_gui_state()
                result["weak"] = list(best_weak.values())
                result["windows_used"] = window_no
                return result

            # Nothing even weakly target-like in the complete window:
            # do not waste the remaining hold budget.
            if not result["weak"]:
                result["weak"] = list(best_weak.values())
                result["windows_used"] = window_no
                return result

            # Weak evidence exists.  Stay at this exact yaw/pitch and give it
            # another full sample window instead of immediately sweeping away.
            if (
                window_no < TARGET_HOLD_MAX_WINDOWS
                and (time.monotonic() - hold_started) < TARGET_HOLD_MAX_SEC
            ):
                top = result["weak"][0]
                ident = self._candidate_identity_key(top)
                self.target_status_text = (
                    f"HOLD {ident} "
                    f"{top.get('sample_hits', 0)}/{max(1, result['sampled_frames'])} frames "
                    f"conf={float(top.get('sample_mean_confidence', top.get('score', 0.0))):.2f}"
                )
                self.publish_gui_state()

        return {
            "stable_colors": [],
            "stable_markers": [],
            "weak": list(best_weak.values()),
            "sampled_frames": 0,
            "windows_used": TARGET_HOLD_MAX_WINDOWS,
        }

    def _stable_color_candidates(self, since_t):
        now = time.monotonic()
        cutoff = max(float(since_t), now - TARGET_HISTORY_SEC)

        with self.vision_lock:
            history = [
                (ts, [dict(d) for d in dets])
                for ts, dets in self.detection_history
                if ts >= cutoff
            ]

        grouped = {}

        for ts, dets in history:
            # At most one observation per signature per frame.
            per_frame = {}
            for d in dets:
                if d.get("kind") != "COLOR_SHAPE":
                    continue
                key = (d.get("color"), d.get("shape"))
                prev = per_frame.get(key)
                if prev is None or d.get("score", 0.0) > prev.get("score", 0.0):
                    per_frame[key] = d

            for key, d in per_frame.items():
                grouped.setdefault(key, []).append((ts, d))

        stable = []

        for key, items in grouped.items():
            if len(items) < TARGET_CONFIRM_FRAMES:
                continue

            centers_x = [float(d["center_norm"][0]) for _, d in items]
            centers_y = [float(d["center_norm"][1]) for _, d in items]
            areas = [float(d.get("area_frac_roi", 0.0)) for _, d in items]
            scores = [float(d.get("score", 0.0)) for _, d in items]

            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
            center_std = math.hypot(cx_std, cy_std)

            area_mean = statistics.fmean(areas) if areas else 0.0
            area_std = statistics.pstdev(areas) if len(areas) > 1 else 0.0
            area_cv = area_std / max(area_mean, 1e-6)

            if center_std > TARGET_CONFIRM_MAX_CENTER_STD:
                continue
            if area_cv > TARGET_CONFIRM_MAX_AREA_CV:
                continue

            # Use the newest observation, but attach temporal evidence.
            newest = dict(items[-1][1])
            newest["confirm_frames"] = len(items)
            newest["center_std"] = center_std
            newest["area_cv"] = area_cv
            newest["temporal_score"] = statistics.fmean(scores)
            stable.append(newest)

        stable.sort(
            key=lambda d: (
                d.get("confirm_frames", 0),
                d.get("temporal_score", 0.0),
            ),
            reverse=True,
        )
        return stable

    def _stable_sdk_markers(self, since_t):
        now = time.monotonic()
        cutoff = max(float(since_t), now - TARGET_HISTORY_SEC)

        with self.vision_lock:
            history = [
                (ts, [dict(m) for m in markers])
                for ts, markers in self.sdk_marker_history
                if ts >= cutoff
            ]

        grouped = {}

        for ts, markers in history:
            seen_this_frame = {}
            for m in markers:
                key = str(m.get("label", "?"))
                seen_this_frame[key] = m
            for key, m in seen_this_frame.items():
                grouped.setdefault(key, []).append((ts, m))

        result = []

        for label, items in grouped.items():
            if len(items) < SDK_MARKER_CONFIRM_FRAMES:
                continue

            centers_x = [float(m["center_norm"][0]) for _, m in items]
            centers_y = [float(m["center_norm"][1]) for _, m in items]
            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0

            if math.hypot(cx_std, cy_std) > TARGET_CONFIRM_MAX_CENTER_STD:
                continue

            newest = dict(items[-1][1])
            newest["confirm_frames"] = len(items)
            result.append(newest)

        result.sort(key=lambda m: m.get("confirm_frames", 0), reverse=True)
        return result

    def _latest_matching_color(self, candidate, expected_center=None):
        key = (candidate.get("color"), candidate.get("shape"))
        now = time.monotonic()

        with self.vision_lock:
            detections = [dict(d) for d in self.latest_detections]
            history = list(self.detection_history)

        if not history or now - history[-1][0] > 0.45:
            return None

        matches = [
            d for d in detections
            if (
                d.get("kind") == "COLOR_SHAPE"
                and (d.get("color"), d.get("shape")) == key
            )
        ]

        if not matches:
            return None

        if expected_center is None:
            return max(matches, key=lambda d: d.get("score", 0.0))

        ex, ey = expected_center
        return min(
            matches,
            key=lambda d: (
                (float(d["center_norm"][0]) - ex) ** 2
                + (float(d["center_norm"][1]) - ey) ** 2
            ),
        )

    def _latest_matching_marker(self, candidate, expected_center=None):
        label = str(candidate.get("label", "?"))
        markers = [
            m for m in self._current_sdk_markers()
            if str(m.get("label", "?")) == label
        ]
        if not markers:
            return None

        if expected_center is None:
            return markers[0]

        ex, ey = expected_center
        return min(
            markers,
            key=lambda m: (
                (float(m["center_norm"][0]) - ex) ** 2
                + (float(m["center_norm"][1]) - ey) ** 2
            ),
        )

    def _transient_target_candidates(self, since_t):
        """Return even 1-frame target evidence seen since *since_t*.

        This deliberately uses the SAME contour/ROI/shape detector as normal
        confirmation; it only relaxes the temporal frame-count requirement.
        Therefore a raw HSV blob that failed the geometric gates never becomes
        a glimpse candidate.
        """
        now = time.monotonic()
        cutoff = max(float(since_t), now - TARGET_HISTORY_SEC)

        with self.vision_lock:
            color_history = [
                (ts, [dict(d) for d in dets])
                for ts, dets in self.detection_history
                if ts >= cutoff
            ]
            marker_history = [
                (ts, [dict(m) for m in markers])
                for ts, markers in self.sdk_marker_history
                if ts >= cutoff
            ]

        grouped = {}

        for ts, dets in color_history:
            per_frame = {}
            for d in dets:
                if d.get("kind") != "COLOR_SHAPE":
                    continue
                key = ("COLOR_SHAPE", d.get("color"), d.get("shape"))
                prev = per_frame.get(key)
                if prev is None or float(d.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    per_frame[key] = d
            for key, d in per_frame.items():
                grouped.setdefault(key, []).append((ts, d))

        for ts, markers in marker_history:
            per_frame = {}
            for m in markers:
                key = ("SDK_MARKER", str(m.get("label", "?")))
                per_frame[key] = m
            for key, m in per_frame.items():
                grouped.setdefault(key, []).append((ts, m))

        result = []
        for key, items in grouped.items():
            if not items:
                continue
            # Prefer the strongest color observation, newest SDK observation.
            if key[0] == "COLOR_SHAPE":
                best = max(items, key=lambda pair: float(pair[1].get("score", 0.0)))[1]
                if float(best.get("score", 0.0)) < TARGET_GLIMPSE_MIN_SCORE:
                    continue
            else:
                best = items[-1][1]

            cand = dict(best)
            cand["kind"] = key[0]
            cand["glimpse_frames"] = len(items)
            cand["glimpse_first_t"] = float(items[0][0])
            cand["glimpse_last_t"] = float(items[-1][0])
            result.append(cand)

        result.sort(
            key=lambda d: (
                int(d.get("glimpse_frames", 0)),
                float(d.get("score", 1.0)),
            ),
            reverse=True,
        )
        return result

    def _candidate_identity_key(self, candidate):
        if candidate.get("kind") == "SDK_MARKER":
            return ("SDK_MARKER", str(candidate.get("label", "?")))
        return (
            "COLOR_SHAPE",
            str(candidate.get("color", "?")),
            str(candidate.get("shape", "?")),
        )

    def _record_target_glimpse(
        self,
        candidate,
        reason="transient",
        gimbal_yaw=None,
        gimbal_pitch=None,
    ):
        """Remember weak target evidence for a targeted return-pass re-scan."""
        if not TARGET_GLIMPSE_ENABLED or candidate is None:
            return None

        cell = tuple(self.current)
        heading = int(self.heading) % 4

        if gimbal_pitch is None or gimbal_yaw is None:
            gp, gy = self.current_gimbal_relative()
            if gimbal_pitch is None:
                gimbal_pitch = gp
            if gimbal_yaw is None:
                gimbal_yaw = gy

        if gimbal_yaw is None:
            return None

        absolute_bearing = wrap_deg(float(heading) * 90.0 + float(gimbal_yaw))
        ident = self._candidate_identity_key(candidate)
        now_iso = datetime.now().isoformat(timespec="milliseconds")

        # Merge repeated brief hits from the same cell and approximately the
        # same physical direction so one noisy sweep cannot create dozens of G#.
        for rec in self.target_glimpses:
            if tuple(rec.get("source_cell", [])) != cell:
                continue
            if tuple(rec.get("identity", [])) != tuple(ident):
                continue
            old_bearing = rec.get("bearing_deg_from_north")
            if old_bearing is None:
                continue
            if self._angle_diff_deg(old_bearing, absolute_bearing) > TARGET_GLIMPSE_BEARING_MERGE_DEG:
                continue

            rec["last_seen_at"] = now_iso
            rec["seen_events"] = int(rec.get("seen_events", 1)) + 1
            rec["max_frames_seen"] = max(
                int(rec.get("max_frames_seen", 0)),
                int(candidate.get("glimpse_frames", candidate.get("confirm_frames", 1))),
            )
            if float(candidate.get("score", 0.0)) >= float(rec.get("score", 0.0)):
                rec["center_norm"] = candidate.get("center_norm")
                rec["bbox_norm"] = candidate.get("bbox_norm")
                rec["score"] = float(candidate.get("score", 0.0))
                rec["gimbal_yaw_deg"] = float(gimbal_yaw)
                rec["gimbal_pitch_deg"] = None if gimbal_pitch is None else float(gimbal_pitch)
                rec["bearing_deg_from_north"] = absolute_bearing
                rec["reason"] = str(reason)
            return rec

        cell_count = sum(
            1 for rec in self.target_glimpses
            if tuple(rec.get("source_cell", [])) == cell
            and not rec.get("resolved_target_id")
        )
        if cell_count >= TARGET_GLIMPSE_MAX_PER_CELL:
            return None

        self.target_glimpse_seq += 1
        rec = {
            "id": f"G{self.target_glimpse_seq}",
            "identity": list(ident),
            "kind": candidate.get("kind"),
            "color": candidate.get("color"),
            "shape": candidate.get("shape"),
            "label": candidate.get("label"),
            "source_cell": [int(cell[0]), int(cell[1])],
            "robot_heading_index": heading,
            "robot_heading": DIR_NAMES[heading],
            "gimbal_yaw_deg": float(gimbal_yaw),
            "gimbal_pitch_deg": None if gimbal_pitch is None else float(gimbal_pitch),
            "bearing_deg_from_north": absolute_bearing,
            "center_norm": candidate.get("center_norm"),
            "bbox_norm": candidate.get("bbox_norm"),
            "score": float(candidate.get("score", 1.0)),
            "max_frames_seen": int(candidate.get("glimpse_frames", candidate.get("confirm_frames", 1))),
            "seen_events": 1,
            "reason": str(reason),
            "resolved_target_id": None,
            "first_seen_at": now_iso,
            "last_seen_at": now_iso,
        }
        self.target_glimpses.append(rec)
        print(
            f"[TARGET GLIMPSE] {rec['id']} cell={cell} "
            f"identity={ident} bearing={absolute_bearing:+.1f} "
            f"pitch={gimbal_pitch} reason={reason}"
        )
        self.target_status_text = (
            f"GLIMPSE {rec['id']} at {cell}; will re-check on return"
        )
        self.save_targets()
        self.publish_gui_state()
        return rec

    def _resolve_matching_glimpses(self, target_record):
        if not target_record:
            return
        ident = self._candidate_identity_key(target_record)
        target_cell = tuple(target_record.get("source_cell", []))
        target_bearing = target_record.get("bearing_deg_from_north")

        for g in self.target_glimpses:
            if g.get("resolved_target_id"):
                continue
            if tuple(g.get("identity", [])) != tuple(ident):
                continue
            if tuple(g.get("source_cell", [])) != target_cell:
                continue
            gb = g.get("bearing_deg_from_north")
            if gb is not None and target_bearing is not None:
                if self._angle_diff_deg(gb, target_bearing) > TARGET_GLIMPSE_BEARING_MERGE_DEG:
                    continue
            g["resolved_target_id"] = target_record.get("id")
            g["resolved_at"] = datetime.now().isoformat(timespec="milliseconds")

    def _match_candidate_to_item(self, candidate, item):
        if candidate is None or item is None:
            return False
        if item.get("kind") == "SDK_MARKER":
            return (
                candidate.get("kind") == "SDK_MARKER"
                and str(candidate.get("label")) == str(item.get("label"))
            )
        return (
            candidate.get("kind") == "COLOR_SHAPE"
            and candidate.get("color") == item.get("color")
            and candidate.get("shape") == item.get("shape")
        )

    def _return_rescan_items_for_cell(self, cell):
        cell = tuple(cell)
        items = []

        if TARGET_RETURN_RESCAN_GLIMPSES:
            for g in self.target_glimpses:
                if g.get("resolved_target_id"):
                    continue
                if tuple(g.get("source_cell", [])) == cell:
                    item = dict(g)
                    item["return_item_type"] = "glimpse"
                    items.append(item)

        if TARGET_RETURN_RESCAN_LOCKED:
            for t in self.detected_targets:
                if tuple(t.get("revisit_cell", t.get("source_cell", []))) == cell:
                    item = dict(t)
                    item["return_item_type"] = "locked"
                    items.append(item)

        # Weak unresolved evidence first, then confirmed targets for refinement.
        items.sort(
            key=lambda x: (
                0 if x.get("return_item_type") == "glimpse" else 1,
                -float(x.get("score", 0.0)),
            )
        )
        return items[:TARGET_RETURN_RESCAN_MAX_ITEMS_PER_CELL]

    def _fusion_update_target(self, target):
        history = list(target.get("observation_history", []))[-TARGET_FUSION_MAX_SAMPLES:]
        target["observation_history"] = history
        valid = [
            h for h in history
            if isinstance(h.get("estimated_grid_xy"), (list, tuple))
            and len(h.get("estimated_grid_xy")) >= 2
        ]
        if not valid:
            target["position_sample_count"] = 0
            return

        xs = [float(h["estimated_grid_xy"][0]) for h in valid]
        ys = [float(h["estimated_grid_xy"][1]) for h in valid]
        mx = statistics.median(xs)
        my = statistics.median(ys)
        target["estimated_grid_xy"] = [mx, my]
        target["position_sample_count"] = len(valid)

        if len(valid) > 1:
            radial = [math.hypot(x - mx, y - my) for x, y in zip(xs, ys)]
            target["position_spread_cells"] = statistics.median(radial)
        else:
            target["position_spread_cells"] = 0.0

    def rescan_targets_on_return(self, cell, phase="return"):
        """Targeted vision check from the return travel heading.

        The chassis is already facing the next confirmed return edge when this
        function is called.  Saved absolute target bearing is converted into a
        NEW relative gimbal yaw, so the same wall is viewed from the reverse
        traversal orientation.  This is useful both for recovering one-frame
        outward-pass glimpses and for adding a second ranged observation to a
        previously locked target.
        """
        if (
            not TARGET_RETURN_RESCAN_ENABLED
            or not self.vision_available
            or not self.running
        ):
            return []

        cell = tuple(cell)
        items = self._return_rescan_items_for_cell(cell)
        if not items:
            return []

        found = []
        self.stop_chassis()
        print(
            f"\n[TARGET RETURN RESCAN] phase={phase} cell={cell} "
            f"heading={DIR_NAMES[self.heading]} items={len(items)}"
        )

        for item in items:
            if not self.running:
                break

            item_id = str(item.get("id", "?"))
            key = (item_id, cell, int(self.heading) % 4, str(phase))
            if key in self.target_return_rescanned:
                continue
            self.target_return_rescanned.add(key)

            bearing = item.get("bearing_deg_from_north")
            if bearing is None:
                continue
            rel_yaw = wrap_deg(float(bearing) - float(self.heading) * 90.0)

            # Gimbal cannot look directly behind the chassis. A future cell or
            # another return heading may bring this bearing inside its side FOV.
            if abs(rel_yaw) > TARGET_LOCK_YAW_LIMIT_DEG:
                print(
                    f"[TARGET RETURN SKIP] {item_id}: relative yaw "
                    f"{rel_yaw:+.1f} outside gimbal limit"
                )
                continue

            base_pitch = item.get("gimbal_pitch_deg")
            try:
                base_pitch = float(base_pitch)
            except Exception:
                base_pitch = TARGET_SWEEP_PITCH_NORMAL_DEG

            candidate_poses = []
            # Re-check at two downward pitch levels, but choose the deep level
            # from the actual yaw.  Around +/-45 deg the deepest legal search
            # pitch is -15 deg; front and near-90 side views retain -22.5 deg.
            pitch_levels = list(self._target_pitch_levels_for_yaw(rel_yaw))
            pitch_levels.sort(key=lambda p: abs(float(p) - base_pitch))
            for pp in pitch_levels:
                candidate_poses.append((rel_yaw, float(pp)))

            # Keep the useful yaw bracket.  The final per-pose clamp below
            # applies the same +/-45 deg protection after the yaw offset.
            yaw_bracket_pitch = float(pitch_levels[0])
            for yoff in TARGET_RETURN_RESCAN_YAW_OFFSETS_DEG[1:]:
                candidate_poses.append((rel_yaw + float(yoff), yaw_bracket_pitch))

            deduped = []
            seen_pose = set()
            for yy, pp in candidate_poses:
                yy = clamp(yy, -TARGET_LOCK_YAW_LIMIT_DEG, TARGET_LOCK_YAW_LIMIT_DEG)
                pp = self._clamp_target_search_pitch(yy, pp)
                pose_key = (round(yy, 1), round(pp, 1))
                if pose_key in seen_pose:
                    continue
                seen_pose.add(pose_key)
                deduped.append((yy, pp))

            confirmed = None
            for pose_no, (yy, pp) in enumerate(deduped, start=1):
                self.target_status_text = (
                    f"RETURN RECHECK {item_id} cell={cell} "
                    f"pose={pose_no}/{len(deduped)} yaw={yy:+.0f} pitch={pp:+.0f}"
                )
                self.publish_gui_state()

                self.gimbal_goto(yy, pitch_deg=pp, force=True)
                time.sleep(TARGET_SWEEP_SETTLE_SEC)

                sampled = self._observe_target_pose_windowed(
                    cell=cell,
                    pose_index=pose_no,
                    total_poses=len(deduped),
                    yaw_deg=yy,
                    pitch_deg=pp,
                    expected_item=item,
                    phase="return",
                )
                stable_markers = list(sampled.get("stable_markers", []))
                stable_colors = list(sampled.get("stable_colors", []))

                preferred_id = (
                    item.get("id")
                    if item.get("return_item_type") == "locked"
                    else None
                )

                if stable_markers:
                    cand = dict(stable_markers[0])
                    cand["kind"] = "SDK_MARKER"
                    confirmed = self._lock_sdk_marker(
                        cand,
                        preferred_target_id=preferred_id,
                        observation_pass="return",
                    )
                elif stable_colors:
                    cand = dict(stable_colors[0])
                    cand["kind"] = "COLOR_SHAPE"
                    confirmed = self._lock_color_candidate(
                        cand,
                        preferred_target_id=preferred_id,
                        observation_pass="return",
                    )

                if confirmed is not None:
                    found.append(confirmed)
                    if item.get("return_item_type") == "glimpse":
                        for g in self.target_glimpses:
                            if g.get("id") == item.get("id"):
                                g["resolved_target_id"] = confirmed.get("id")
                                g["resolved_at"] = datetime.now().isoformat(timespec="milliseconds")
                                break
                    print(
                        f"[TARGET RETURN CONFIRMED] {item_id} -> "
                        f"{confirmed.get('id')} samples="
                        f"{confirmed.get('position_sample_count', 1)}"
                    )
                    break

            if confirmed is None and item.get("return_item_type") == "glimpse":
                # Keep the evidence for logs; do not promote it to a target.
                for g in self.target_glimpses:
                    if g.get("id") == item.get("id"):
                        g["return_recheck_attempts"] = int(g.get("return_recheck_attempts", 0)) + 1
                        break

        self.gimbal_front_down(force=True)
        self.save_targets()
        self.publish_gui_state()
        return found

    @staticmethod
    def _target_yaw_is_diagonal(yaw_deg):
        """True when target-search yaw is in the protected +/-45 deg zone."""
        a = abs(wrap_deg(float(yaw_deg)))
        return TARGET_DIAGONAL_YAW_MIN_DEG <= a <= TARGET_DIAGONAL_YAW_MAX_DEG

    def _target_pitch_levels_for_yaw(self, yaw_deg):
        """Return the two legal target-search pitches for this yaw."""
        deep = (
            TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG
            if self._target_yaw_is_diagonal(yaw_deg)
            else TARGET_SWEEP_PITCH_DEEP_DEG
        )
        return (TARGET_SWEEP_PITCH_NORMAL_DEG, deep)

    def _clamp_target_search_pitch(self, yaw_deg, pitch_deg):
        """Yaw-dependent pitch guard used only by target vision motions.

        Do NOT put this clamp inside generic gimbal_goto(): ToF topology scans
        need their own geometry.  This guard protects target search/re-lock only.
        """
        min_pitch = (
            TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG
            if self._target_yaw_is_diagonal(yaw_deg)
            else TARGET_LOCK_PITCH_MIN_DEG
        )
        return clamp(
            float(pitch_deg),
            float(min_pitch),
            float(TARGET_LOCK_PITCH_MAX_DEG),
        )

    def _target_lock_adjust(self, center_norm):
        p_now, y_now = self.current_gimbal_relative()
        if p_now is None or y_now is None:
            return False

        cx, cy = [float(v) for v in center_norm]
        ex = cx - 0.5
        ey = cy - 0.5

        if (
            abs(ex) <= TARGET_LOCK_CENTER_TOL_X
            and abs(ey) <= TARGET_LOCK_CENTER_TOL_Y
        ):
            return True

        yaw_step = clamp(
            ex * TARGET_LOCK_HFOV_DEG,
            -TARGET_LOCK_MAX_YAW_STEP_DEG,
            TARGET_LOCK_MAX_YAW_STEP_DEG,
        )
        pitch_step = clamp(
            ey * TARGET_LOCK_VFOV_DEG,
            -TARGET_LOCK_MAX_PITCH_STEP_DEG,
            TARGET_LOCK_MAX_PITCH_STEP_DEG,
        )

        y_target = clamp(
            y_now + yaw_step,
            -TARGET_LOCK_YAW_LIMIT_DEG,
            TARGET_LOCK_YAW_LIMIT_DEG,
        )
        # Positive image Y means target is below center. RoboMaster negative
        # pitch points down, hence subtract pitch_step.
        p_target = self._clamp_target_search_pitch(
            y_target,
            p_now - pitch_step,
        )

        self.gimbal_goto(
            y_target,
            pitch_deg=p_target,
            force=True,
        )
        time.sleep(TARGET_LOCK_SETTLE_SEC)
        return False

    def _estimate_target_position(
        self, cell, heading, gimbal_yaw, tof_mm, gimbal_pitch=0.0
    ):
        if tof_mm is None or gimbal_yaw is None:
            return None, None

        try:
            slant_range_m = float(tof_mm) / 1000.0
            if not math.isfinite(slant_range_m) or slant_range_m <= 0.02:
                return None, None
        except Exception:
            return None, None

        # ToF range follows the gimbal ray.  When the camera/ToF is looking
        # noticeably up or down, using the raw slant range directly as planar
        # XY distance pushes the target too far away on the map.  Project the
        # ray onto the horizontal plane first.
        try:
            pitch_deg = float(gimbal_pitch or 0.0)
        except Exception:
            pitch_deg = 0.0
        horizontal_range_m = slant_range_m * abs(math.cos(math.radians(pitch_deg)))

        bearing_deg = wrap_deg(float(heading) * 90.0 + float(gimbal_yaw))
        rad = math.radians(bearing_deg)
        dx = math.sin(rad)
        dy = math.cos(rad)

        grid_range = horizontal_range_m / max(CELL_LENGTH_M, 1e-6)
        gx = float(cell[0]) + dx * grid_range
        gy = float(cell[1]) + dy * grid_range

        return [gx, gy], bearing_deg

    def _same_target_identity(self, a, b):
        if a.get("kind") != b.get("kind"):
            return False

        if a.get("kind") == "COLOR_SHAPE":
            return (
                a.get("color") == b.get("color")
                and a.get("shape") == b.get("shape")
            )

        if a.get("kind") == "SDK_MARKER":
            return str(a.get("label")) == str(b.get("label"))

        return False

    def _find_duplicate_target(self, record):
        for existing in self.detected_targets:
            if not self._same_target_identity(existing, record):
                continue

            pa = existing.get("estimated_grid_xy")
            pb = record.get("estimated_grid_xy")

            if pa is not None and pb is not None:
                dist = math.hypot(
                    float(pa[0]) - float(pb[0]),
                    float(pa[1]) - float(pb[1]),
                )
                if dist <= TARGET_DEDUPE_GRID_DIST:
                    return existing
                continue

            # Without range, only merge a repeated observation from the same
            # logical cell and roughly the same bearing.
            if TARGET_DEDUPE_SAME_CELL_ONLY_IF_NO_RANGE:
                if tuple(existing.get("source_cell", [])) != tuple(
                    record.get("source_cell", [])
                ):
                    continue

            ba = existing.get("bearing_deg_from_north")
            bb = record.get("bearing_deg_from_north")
            if ba is not None and bb is not None:
                if self._angle_diff_deg(ba, bb) <= TARGET_DEDUPE_BEARING_DEG:
                    return existing

        return None

    def _record_target(
        self,
        candidate,
        tof_mm,
        preferred_target_id=None,
        observation_pass="explore",
    ):
        cell = tuple(self.current)
        heading = int(self.heading) % 4
        gimbal_p, gimbal_y = self.current_gimbal_relative()
        grid_xy, bearing = self._estimate_target_position(
            cell,
            heading,
            gimbal_y,
            tof_mm,
            gimbal_pitch=gimbal_p,
        )

        pos = self.state.get_position()

        observation = {
            "pass": str(observation_pass),
            "at": datetime.now().isoformat(timespec="milliseconds"),
            "source_cell": [int(cell[0]), int(cell[1])],
            "robot_heading_index": heading,
            "gimbal_yaw_deg": gimbal_y,
            "gimbal_pitch_deg": gimbal_p,
            "bearing_deg_from_north": bearing,
            "tof_mm": None if tof_mm is None else float(tof_mm),
            "estimated_grid_xy": grid_xy,
            "score": float(candidate.get("temporal_score", candidate.get("score", 1.0))),
        }

        record = {
            "id": None,
            "kind": candidate.get("kind"),
            "color": candidate.get("color"),
            "shape": candidate.get("shape"),
            "label": candidate.get("label"),
            "source_cell": [int(cell[0]), int(cell[1])],
            # Revisit this confirmed cell during the later attack/reacquire pass.
            # It is more trustworthy than blindly navigating to a projected
            # target point derived from one ToF ray.
            "revisit_cell": [int(cell[0]), int(cell[1])],
            "robot_heading_index": heading,
            "robot_heading": DIR_NAMES[heading],
            "gimbal_yaw_deg": gimbal_y,
            "gimbal_pitch_deg": gimbal_p,
            "reacquire_hint": {
                "heading_index": heading,
                "heading": DIR_NAMES[heading],
                "gimbal_yaw_deg": gimbal_y,
                "gimbal_pitch_deg": gimbal_p,
            },
            "bearing_deg_from_north": bearing,
            "tof_mm": None if tof_mm is None else float(tof_mm),
            "tof_position_is_approximate": True,
            "estimated_grid_xy": grid_xy,
            # Never fire from this memory record alone. The attack pass should
            # return to revisit_cell and reacquire/revalidate the live target.
            "attack_requires_reacquire": True,
            "bbox_norm": candidate.get("bbox_norm"),
            "center_norm": candidate.get("center_norm"),
            "score": float(candidate.get("temporal_score", candidate.get("score", 1.0))),
            "confirm_frames": int(candidate.get("confirm_frames", 0)),
            "chassis_odom_xyz": (
                None if pos is None else [float(v) for v in pos]
            ),
            "first_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "last_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "observations": 1,
            "observation_history": [observation],
            "position_sample_count": 1 if grid_xy is not None else 0,
            "position_spread_cells": 0.0,
        }

        duplicate = None
        if preferred_target_id is not None:
            duplicate = next(
                (t for t in self.detected_targets if t.get("id") == preferred_target_id),
                None,
            )
        if duplicate is None:
            duplicate = self._find_duplicate_target(record)

        if duplicate is not None:
            duplicate["last_seen_at"] = record["last_seen_at"]
            duplicate["observations"] = int(duplicate.get("observations", 1)) + 1
            duplicate.setdefault("observation_history", []).append(observation)

            # Keep the newest live reacquisition hint, but fuse physical XY
            # from ALL valid ranged observations using a robust median.
            duplicate["reacquire_hint"] = dict(record.get("reacquire_hint", {}))
            duplicate["last_observation_pass"] = str(observation_pass)
            duplicate["bbox_norm"] = record.get("bbox_norm")
            duplicate["center_norm"] = record.get("center_norm")
            duplicate["gimbal_yaw_deg"] = record.get("gimbal_yaw_deg")
            duplicate["gimbal_pitch_deg"] = record.get("gimbal_pitch_deg")
            duplicate["bearing_deg_from_north"] = record.get("bearing_deg_from_north")
            duplicate["tof_mm"] = record.get("tof_mm")
            self._fusion_update_target(duplicate)
            self._resolve_matching_glimpses(duplicate)

            self.target_status_text = (
                f"RE-SEEN {duplicate['id']} "
                f"{duplicate.get('color') or duplicate.get('label')} "
                f"{duplicate.get('shape') or 'SDK'}"
            )
            print(
                f"[TARGET] duplicate -> {duplicate['id']} "
                f"observations={duplicate['observations']}"
            )
            self._maybe_fire_locked_target(duplicate)
            self.save_targets()
            self.publish_gui_state()
            return duplicate

        self.target_id_seq += 1
        record["id"] = f"T{self.target_id_seq}"
        record["last_observation_pass"] = str(observation_pass)
        self.detected_targets.append(record)
        self._resolve_matching_glimpses(record)

        desc = (
            f"{record.get('color')} {record.get('shape')}"
            if record.get("kind") == "COLOR_SHAPE"
            else f"SDK {record.get('label')}"
        )
        self.target_status_text = (
            f"LOCKED {record['id']} {desc} at cell={cell} "
            f"bearing={bearing if bearing is not None else 'n/a'}"
        )

        print("\n[TARGET LOCKED]")
        print(f"  ID       : {record['id']}")
        print(f"  Type     : {desc}")
        print(f"  Cell     : {cell}")
        print(f"  Gimbal   : yaw={gimbal_y} pitch={gimbal_p}")
        print(f"  ToF      : {record['tof_mm']} mm")
        print(f"  Bearing  : {record['bearing_deg_from_north']}")
        print(f"  Grid est.: {record['estimated_grid_xy']}")

        self._maybe_fire_locked_target(record)
        self.save_targets()
        self.publish_gui_state()
        return record

    def _lock_color_candidate(
        self, candidate, preferred_target_id=None, observation_pass="explore"
    ):
        expected = list(candidate.get("center_norm", [0.5, 0.5]))
        current = dict(candidate)

        for _ in range(TARGET_LOCK_MAX_STEPS):
            latest = self._latest_matching_color(current, expected)
            if latest is None:
                self.target_status_text = "candidate lost during color lock"
                print("[TARGET REJECT] color candidate lost during lock")
                return None

            current.update(latest)
            expected = list(current.get("center_norm", expected))
            centered = self._target_lock_adjust(expected)
            if centered:
                break

        latest = self._latest_matching_color(current, expected)
        if latest is None:
            print("[TARGET REJECT] color target not visible after lock")
            return None

        current.update(latest)
        cx, cy = current.get("center_norm", [0.0, 0.0])
        if (
            abs(float(cx) - 0.5) > TARGET_LOCK_CENTER_TOL_X * 1.7
            or abs(float(cy) - 0.5) > TARGET_LOCK_CENTER_TOL_Y * 1.7
        ):
            print("[TARGET REJECT] color target failed final center gate")
            return None

        tof_mm = self.sample_tof_median()
        return self._record_target(
            current,
            tof_mm,
            preferred_target_id=preferred_target_id,
            observation_pass=observation_pass,
        )

    def _lock_sdk_marker(
        self, candidate, preferred_target_id=None, observation_pass="explore"
    ):
        expected = list(candidate.get("center_norm", [0.5, 0.5]))
        current = dict(candidate)

        for _ in range(TARGET_LOCK_MAX_STEPS):
            latest = self._latest_matching_marker(current, expected)
            if latest is None:
                self.target_status_text = "SDK marker lost during lock"
                print("[TARGET REJECT] SDK marker lost during lock")
                return None

            current.update(latest)
            expected = list(current.get("center_norm", expected))
            centered = self._target_lock_adjust(expected)
            if centered:
                break

        latest = self._latest_matching_marker(current, expected)
        if latest is None:
            print("[TARGET REJECT] SDK marker not visible after lock")
            return None

        current.update(latest)
        tof_mm = self.sample_tof_median()
        return self._record_target(
            current,
            tof_mm,
            preferred_target_id=preferred_target_id,
            observation_pass=observation_pass,
        )

    def scan_targets_at_cell(self, cell, force=False):
        """Search/lock/remember targets while the chassis is stationary.

        Each gimbal pose now uses a sampled-window policy:
          1) collect 10 fresh camera frames,
          2) require >=3 consistent frames,
          3) require mean confidence >=0.60 before lock/save,
          4) if confidence >=0.50 appears but is not yet verified, HOLD the
             same pose for another window (max 3 windows / 3 seconds),
          5) unresolved evidence becomes a G# glimpse for return re-check.

        This prevents the old failure mode where the turret swept away just as
        a valid target first entered the ROI.
        """
        if not self.vision_available or not self.running:
            return []

        cell = tuple(cell)
        now = time.monotonic()
        last_t = self.target_last_scan_t.get(cell)

        if (
            not force
            and cell in self.target_scanned_cells
            and last_t is not None
            and now - last_t < TARGET_SCAN_COOLDOWN_SEC
        ):
            return []

        self.stop_chassis()
        self.target_last_scan_t[cell] = now
        found = []

        print(f"\n[TARGET SWEEP] cell={cell} heading={DIR_NAMES[self.heading]}")
        self.target_status_text = f"SEARCHING targets at cell={cell}"
        self.publish_gui_state()

        total_poses = len(TARGET_SWEEP_POSES)
        for pose_index, (yaw_deg, pitch_deg) in enumerate(TARGET_SWEEP_POSES, start=1):
            if not self.running:
                break

            self.target_status_text = (
                f"SEARCHING cell={cell} pose={pose_index}/{total_poses} "
                f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}"
            )
            self.publish_gui_state()

            self.gimbal_goto(
                yaw_deg,
                pitch_deg=pitch_deg,
                force=True,
            )
            time.sleep(TARGET_SWEEP_SETTLE_SEC)

            sampled = self._observe_target_pose_windowed(
                cell=cell,
                pose_index=pose_index,
                total_poses=total_poses,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                expected_item=None,
                phase="explore",
            )

            marker_candidates = list(sampled.get("stable_markers", []))
            color_candidates = list(sampled.get("stable_colors", []))
            weak_candidates = list(sampled.get("weak", []))

            locked_identity = set()
            stable_identity = set()

            # SDK marker identity is more specific than generic red geometry.
            for candidate in marker_candidates[:2]:
                candidate = dict(candidate)
                candidate["kind"] = "SDK_MARKER"
                ident = self._candidate_identity_key(candidate)
                stable_identity.add(ident)

                rec = self._lock_sdk_marker(candidate)
                if rec is not None:
                    found.append(rec)
                    locked_identity.add(ident)
                else:
                    self._record_target_glimpse(
                        candidate,
                        reason="sdk_verified_but_lock_failed",
                        gimbal_yaw=yaw_deg,
                        gimbal_pitch=pitch_deg,
                    )

                # A lock moves the turret. Return to the survey pose before
                # considering another candidate from this sampled window.
                self.gimbal_goto(
                    yaw_deg,
                    pitch_deg=pitch_deg,
                    force=True,
                )
                time.sleep(TARGET_LOCK_SETTLE_SEC)

            for candidate in color_candidates[:3]:
                candidate = dict(candidate)
                candidate["kind"] = "COLOR_SHAPE"
                ident = self._candidate_identity_key(candidate)
                stable_identity.add(ident)

                rec = self._lock_color_candidate(candidate)
                if rec is not None:
                    found.append(rec)
                    locked_identity.add(ident)
                else:
                    self._record_target_glimpse(
                        candidate,
                        reason="color_verified_but_lock_failed",
                        gimbal_yaw=yaw_deg,
                        gimbal_pitch=pitch_deg,
                    )

                self.gimbal_goto(
                    yaw_deg,
                    pitch_deg=pitch_deg,
                    force=True,
                )
                time.sleep(TARGET_LOCK_SETTLE_SEC)

            # Any >=0.50 evidence that used the hold budget but never reached
            # the >=0.60 / 3-frame save gate is deliberately remembered.
            for cand in weak_candidates:
                ident = self._candidate_identity_key(cand)
                if ident in locked_identity or ident in stable_identity:
                    continue
                self._record_target_glimpse(
                    cand,
                    reason=(
                        f"sample_window_unconfirmed:"
                        f"{cand.get('sample_hits', 0)}/"
                        f"{cand.get('sampled_frames', TARGET_SAMPLE_FRAMES)}"
                    ),
                    gimbal_yaw=yaw_deg,
                    gimbal_pitch=pitch_deg,
                )

        self.target_scanned_cells.add(cell)

        # Restore the exact motion-ready ToF posture used by the DFS safety code.
        self.gimbal_front_down(force=True)

        if found:
            unique_ids = sorted({r.get("id") for r in found if r.get("id")})
            self.target_status_text = (
                f"TARGET SWEEP DONE cell={cell}: {', '.join(unique_ids)}"
            )
        else:
            pending_here = sum(
                1 for g in self.target_glimpses
                if tuple(g.get("source_cell", [])) == cell
                and not g.get("resolved_target_id")
            )
            self.target_status_text = (
                f"TARGET SWEEP DONE cell={cell}: none"
                + (f" / {pending_here} glimpse" if pending_here else "")
            )

        self.publish_gui_state()
        return found

    def save_targets(self):
        try:
            TARGET_LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "schema": "robomaster_target_memory",
                "schema_version": 3,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "count": len(self.detected_targets),
                "target_fire_policy": self.get_target_fire_policy(),
                "last_fire_event": self.last_fire_event,
                "targets": self.detected_targets,
                "pending_glimpse_count": sum(
                    1 for g in self.target_glimpses
                    if not g.get("resolved_target_id")
                ),
                "glimpses": self.target_glimpses,
            }

            tmp = TARGET_LATEST_JSON.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, TARGET_LATEST_JSON)
        except Exception as e:
            print(f"[TARGET SAVE WARN] {e}")

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
        print(
            f" Maze       : dynamic size, physical grid=60cm, "
            f"OPEN if ToF >= {TOF_OPEN_THRESHOLD_MM}mm"
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
        self.camera = getattr(self.ep_robot, "camera", None)
        self.vision = getattr(self.ep_robot, "vision", None)
        self.blaster = getattr(self.ep_robot, "blaster", None)
        if self.blaster is None:
            print("[FIRE WARN] RoboMaster blaster module unavailable; firing will stay blocked.")
        else:
            print("[FIRE] blaster ready; fire type locked to INFRARED")

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

        # Camera/SDK marker processing is started only after the gimbal and yaw
        # references are stable. Vision failure is non-fatal to DFS navigation.
        self.start_target_vision()

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
            self.stop_target_vision()
        except Exception:
            pass

        try:
            self.save_targets()
        except Exception:
            pass

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
        """Generate y from ONE Sharp sensor using a comfort-band controller.

        Unlike the old exact-13-cm controller, this emits NO lateral command
        while the active wall lies inside SHARP_FOLLOW_NEAR_CM..FAR_CM.  The
        chassis therefore does not keep pushing left/right for small noise.

        y > 0 -> slide RIGHT
        y < 0 -> slide LEFT
        """
        if authority == "LEFT":
            dist = left_cm
            if dist is None:
                return 0.0, "AUTH_LEFT_NO_DATA"

            # Safe comfort band: coast straight, let yaw hold do the work.
            if SHARP_FOLLOW_NEAR_CM <= dist <= SHARP_FOLLOW_FAR_CM:
                return 0.0, "AUTH_LEFT_BAND"

            if dist < SHARP_FOLLOW_NEAR_CM:
                # LEFT wall too close -> nudge RIGHT.
                outside = SHARP_FOLLOW_NEAR_CM - dist
                y = max(
                    SHARP_FOLLOW_MIN_STRAFE_MPS,
                    SHARP_FOLLOW_KP * outside,
                )
                if dist <= AUTHORITY_HARD_CM:
                    y = max(y, AUTHORITY_HARD_STRAFE_MPS)
                    mode = "AUTH_LEFT_HARD"
                else:
                    mode = "AUTH_LEFT_NEAR"
                return (
                    clamp(y, 0.0, AUTHORITY_HARD_STRAFE_MPS),
                    mode,
                )

            # LEFT wall too far -> gently move LEFT to keep wall-follow.
            outside = dist - SHARP_FOLLOW_FAR_CM
            y = -max(
                SHARP_FOLLOW_MIN_STRAFE_MPS,
                SHARP_FOLLOW_KP * outside,
            )
            return (
                clamp(y, -MAX_CENTER_STRAFE_MPS, 0.0),
                "AUTH_LEFT_FAR",
            )

        if authority == "RIGHT":
            dist = right_cm
            if dist is None:
                return 0.0, "AUTH_RIGHT_NO_DATA"

            if SHARP_FOLLOW_NEAR_CM <= dist <= SHARP_FOLLOW_FAR_CM:
                return 0.0, "AUTH_RIGHT_BAND"

            if dist < SHARP_FOLLOW_NEAR_CM:
                # RIGHT wall too close -> nudge LEFT.
                outside = SHARP_FOLLOW_NEAR_CM - dist
                y = -max(
                    SHARP_FOLLOW_MIN_STRAFE_MPS,
                    SHARP_FOLLOW_KP * outside,
                )
                if dist <= AUTHORITY_HARD_CM:
                    y = min(y, -AUTHORITY_HARD_STRAFE_MPS)
                    mode = "AUTH_RIGHT_HARD"
                else:
                    mode = "AUTH_RIGHT_NEAR"
                return (
                    clamp(y, -AUTHORITY_HARD_STRAFE_MPS, 0.0),
                    mode,
                )

            # RIGHT wall too far -> gently move RIGHT to keep wall-follow.
            outside = dist - SHARP_FOLLOW_FAR_CM
            y = max(
                SHARP_FOLLOW_MIN_STRAFE_MPS,
                SHARP_FOLLOW_KP * outside,
            )
            return (
                clamp(y, 0.0, MAX_CENTER_STRAFE_MPS),
                "AUTH_RIGHT_FAR",
            )

        return 0.0, "NO_SHARP_AUTHORITY"

    def retreat_to_move_start(self, start_pos, reason=""):
        """
        Back out of a transient mid-edge safety event.

        The robot keeps the same logical heading and drives x<0 toward the
        source cell while:
          - yaw hold keeps orientation locked
          - one Sharp authority may still center laterally
          - absolute chassis position confirms that distance to the original
            start point is decreasing

        We intentionally do NOT run the digital-IR recovery state machine
        while retreating; otherwise the same side IR could recursively trigger
        the exact recovery that we are trying to escape from.

        Rear obstacle sensing is not available, so this maneuver is used only
        to retrace the corridor the robot has just traversed moments earlier.
        """
        self.stop_chassis()
        self.gimbal_front_down()

        if start_pos is None:
            print("[MOTION RETREAT] no original start pose -> cannot retreat")
            return False

        pos = self.state.get_position()

        if pos is None:
            print("[MOTION RETREAT] position telemetry unavailable")
            return False

        def distance_to_start(p):
            return math.hypot(
                p[0] - start_pos[0],
                p[1] - start_pos[1],
            )

        initial_distance = distance_to_start(pos)

        if initial_distance <= MOTION_ABORT_HOME_TOL_M:
            print(
                f"[MOTION RETREAT] already near source cell "
                f"({initial_distance:.3f} m)"
            )
            return True

        print(
            f"[MOTION RETREAT] reason={reason or 'transient IR'} "
            f"distance_to_source={initial_distance:.3f} m "
            f"-> reverse at {MOTION_ABORT_RETREAT_SPEED_MPS:.2f} m/s"
        )

        target_yaw = self.yaw_ref_deg
        best_distance = initial_distance
        last_debug = 0.0
        t0 = time.monotonic()

        while self.running:
            now = time.monotonic()
            pos = self.state.get_position()

            if pos is None:
                self.stop_chassis()
                print("[MOTION RETREAT FAIL] position telemetry lost")
                return False

            dist = distance_to_start(pos)

            if dist <= MOTION_ABORT_HOME_TOL_M:
                self.stop_chassis()
                print(
                    f"[MOTION RETREAT OK] back near source cell: "
                    f"{dist:.3f} m"
                )
                return True

            if now - t0 > MOTION_ABORT_TIMEOUT_SEC:
                self.stop_chassis()
                print(
                    f"[MOTION RETREAT FAIL] timeout; "
                    f"still {dist:.3f} m from source"
                )
                return False

            # Detect going the wrong way / odometry disagreement.
            if dist < best_distance:
                best_distance = dist
            elif dist > best_distance + MOTION_ABORT_PROGRESS_EPS_M:
                self.stop_chassis()
                print(
                    f"[MOTION RETREAT FAIL] distance increased "
                    f"best={best_distance:.3f} now={dist:.3f} m"
                )
                return False

            left_cm, right_cm, _, _ = self.read_sharp_cm()

            # Keep the same single-authority rule during retreat as during
            # normal forward motion. update_sharp_authority() RETURNS the
            # current owner and corridor_lateral_command() requires it.
            authority = self.update_sharp_authority(
                left_cm,
                right_cm
            )

            y_cmd, center_mode = self.corridor_lateral_command(
                left_cm,
                right_cm,
                authority
            )

            z_cmd = self.yaw_hold_command(
                target_yaw,
                stationary=False
            )

            self.chassis.drive_speed(
                x=-MOTION_ABORT_RETREAT_SPEED_MPS,
                y=y_cmd,
                z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT,
            )

            if now - last_debug >= 0.25:
                print(
                    f"[MOTION RETREAT] remaining={dist:.3f}m "
                    f"mode={center_mode} "
                    f"y={y_cmd:+.2f} z={z_cmd:+.1f}"
                )
                last_debug = now

            time.sleep(0.03)

        self.stop_chassis()
        return False

    def request_motion_replan_from_source(
        self,
        start_pos,
        traveled,
        reason,
    ):
        """
        Handle an ambiguous/transient safety abort without poisoning the map.

        Near the destination node:
            accept arrival; node scan will classify topology.

        Mid-edge:
            retreat to the source cell and ask DFS to rescan/replan there.
        """
        self.stop_chassis()

        if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
            print(
                f"[MOTION REPLAN] {reason}; traveled={traveled:.3f}m "
                ">= node-success threshold -> ACCEPT NODE EARLY"
            )
            self.ir_replan_requested = False
            self.motion_replan_requested = False
            self.motion_replan_reason = None
            return "ACCEPT_NODE"

        print(
            f"[MOTION REPLAN] {reason}; traveled={traveled:.3f}m "
            "before node -> RETREAT TO SOURCE CELL"
        )

        retreat_ok = self.retreat_to_move_start(
            start_pos,
            reason=reason,
        )

        if not retreat_ok:
            self.stop_chassis()
            return "RETREAT_FAILED"

        self.ir_replan_requested = False
        self.motion_replan_requested = True
        self.motion_replan_reason = reason

        # Release wall authority because the source-cell geometry will be
        # observed again from a fresh stationary scan.
        self._set_sharp_authority(None, "motion abort returned to source")

        return "REPLAN_SOURCE"

    def evaluate_exit_fan(self, fan):
        """Return robust broad-open evidence for an EXIT candidate.

        A single 600-mm OPEN threshold is topology evidence only; it is not
        enough to distinguish an outside boundary from an internal 3-way/+
        junction.  Exit evidence therefore requires both near-broad coverage
        and multiple much-longer rays.
        """
        left_group = (-60.0, -45.0, -30.0)
        right_group = (+30.0, +45.0, +60.0)

        def count(group, threshold):
            return sum(
                1 for angle in group
                if fan.get(angle) is not None
                and fan[angle] >= threshold
            )

        left_open = count(left_group, EXIT_FAN_OPEN_MM)
        right_open = count(right_group, EXIT_FAN_OPEN_MM)
        left_strong = count(left_group, EXIT_FAN_STRONG_OPEN_MM)
        right_strong = count(right_group, EXIT_FAN_STRONG_OPEN_MM)
        strong_total = left_strong + right_strong

        wide_boundary = (
            left_open >= EXIT_FAN_MIN_OPEN_PER_SIDE
            and right_open >= EXIT_FAN_MIN_OPEN_PER_SIDE
            and left_strong >= EXIT_FAN_MIN_STRONG_PER_SIDE
            and right_strong >= EXIT_FAN_MIN_STRONG_PER_SIDE
            and strong_total >= EXIT_FAN_MIN_STRONG_TOTAL
        )

        return {
            "left_open": left_open,
            "right_open": right_open,
            "left_strong": left_strong,
            "right_strong": right_strong,
            "strong_total": strong_total,
            "wide_boundary": bool(wide_boundary),
        }

    def confirm_in_motion_exit_candidate(
        self,
        source_cell,
        abs_dir,
        start_pos,
        traveled,
        left_cm,
        right_cm,
    ):
        """
        Called only after BOTH side walls that were present at move start have
        disappeared together during the middle of the edge.

        Stop, verify broad-open geometry with the 6-ray fan, then retreat to
        the source node if confirmed.
        """
        self.stop_chassis()

        front_mm = self.sample_tof_median()

        print(
            f"[EXIT MOTION] BOTH side walls disappeared at "
            f"d={traveled:.3f}m on "
            f"{source_cell}->{DIR_NAMES[abs_dir]}"
        )

        if (
            front_mm is not None
            and front_mm < EXIT_FRONT_MIN_SAFE_MM
        ):
            print(
                f"[EXIT MOTION] front={front_mm:.0f}mm too close "
                "-> obstacle/noise, NOT exit"
            )
            return False

        fan = self.scan_exit_fan()

        fan_eval = self.evaluate_exit_fan(fan)
        left_open = fan_eval["left_open"]
        right_open = fan_eval["right_open"]
        strong_total = fan_eval["strong_total"]
        wide_boundary = fan_eval["wide_boundary"]

        print(
            f"[EXIT MOTION] fan votes "
            f"OPEN L={left_open}/3 R={right_open}/3 "
            f"STRONG>={EXIT_FAN_STRONG_OPEN_MM:.0f}mm "
            f"{strong_total}/6 -> "
            f"{'BOUNDARY' if wide_boundary else 'JUNCTION/DEEP PATH'}"
        )

        if not wide_boundary:
            print(
                "[EXIT MOTION] broad opening NOT confirmed "
                "-> resume normal cell motion"
            )
            return False

        self.record_exit_candidate(
            cell=source_cell,
            abs_dir=abs_dir,
            front_mm=front_mm,
            left_cm=left_cm,
            right_cm=right_cm,
            fan_mm=fan,
            reason="both_side_walls_disappeared_during_motion",
            wall_end_probe={
                "detected_at_travel_m": float(traveled),
                "confirm_count": int(EXIT_MOTION_LOST_CONFIRM_COUNT),
            },
        )

        print(
            "[EXIT MOTION] confirmed wide boundary -> RETREAT TO SOURCE "
            "instead of leaving maze"
        )

        retreat_ok = self.retreat_to_move_start(
            start_pos,
            reason="confirmed EXIT_CANDIDATE during edge traversal",
        )

        if not retreat_ok:
            raise RuntimeError(
                "Exit candidate was detected but robot could not safely "
                "retreat to the source cell."
            )

        self.motion_exit_candidate_detected = True
        self.motion_replan_requested = False
        self.motion_replan_reason = None
        self.ir_replan_requested = False
        self._set_sharp_authority(None, "returned from exit candidate")

        return True

    def move_one_cell(
        self,
        source_cell=None,
        abs_dir=None,
        detect_exit=False,
    ):
        """
        Continuous corridor motion:
          - forward drive
          - Sharp wall-follow / nearest-wall priority
          - yaw hold
          - front ToF collision stop
          - stop after CELL_LENGTH_M odometry displacement
        """

        self.motion_replan_requested = False
        self.motion_replan_reason = None
        self.motion_exit_candidate_detected = False

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

        # In-motion fake-exit monitor:
        # only meaningful when the selected edge starts as a straight
        # corridor with BOTH side walls physically present.
        start_left_wall = (
            left_cm is not None
            and left_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )
        start_right_wall = (
            right_cm is not None
            and right_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )

        source_straight_corridor = False

        if source_cell is not None and abs_dir is not None:
            source_open = set(self.open_dirs.get(source_cell, []))
            source_left_dir = (abs_dir - 1) % 4
            source_right_dir = (abs_dir + 1) % 4

            source_straight_corridor = (
                source_left_dir not in source_open
                and source_right_dir not in source_open
            )

        exit_motion_armed = bool(
            detect_exit
            and EXIT_MOTION_GUARD_ENABLED
            and source_cell is not None
            and abs_dir is not None
            and source_straight_corridor
            and start_left_wall
            and start_right_wall
            and not self.is_known_entrance_edge(source_cell, abs_dir)
            and not self.is_known_maze_ingress_edge(
                source_cell,
                abs_dir
            )
        )

        exit_lost_both_count = 0
        exit_motion_checked = False
        exit_caution_slow = False

        if exit_motion_armed:
            print(
                f"[EXIT MOTION] armed for "
                f"{source_cell}->{DIR_NAMES[abs_dir]} "
                f"window={EXIT_MOTION_MIN_TRAVEL_M:.2f}-"
                f"{EXIT_MOTION_MAX_TRAVEL_M:.2f}m"
            )

        print(
            f"[MOVE] start authority={authority} "
            f"sharp_band={SHARP_FOLLOW_NEAR_CM:.1f}-"
            f"{SHARP_FOLLOW_FAR_CM:.1f}cm "
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

                result = self.request_motion_replan_from_source(
                    start_pos=start_pos,
                    traveled=traveled,
                    reason=(
                        f"IR sequence says FRONT blocked "
                        f"(hint={action})"
                    ),
                )

                if result == "ACCEPT_NODE":
                    return True

                if result == "REPLAN_SOURCE":
                    return False

                raise RuntimeError(
                    "IR sequence blocked motion and retreat to the source "
                    "cell failed."
                )

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

                result = self.request_motion_replan_from_source(
                    start_pos=start_pos,
                    traveled=traveled,
                    reason=(
                        f"BOTH-IR says FRONT blocked "
                        f"(hint={action})"
                    ),
                )

                if result == "ACCEPT_NODE":
                    return True

                if result == "REPLAN_SOURCE":
                    return False

                raise RuntimeError(
                    "BOTH-IR blocked motion and retreat to the source "
                    "cell failed."
                )

            # ONE LOW keeps the unambiguous opposite-slide recovery.
            if (ir_l_low ^ ir_r_low) and not both_override:
                self.stop_chassis()

                recovery_t0 = time.monotonic()
                ok = self.ir_clearance_recovery(context="MOVE")
                t0 += time.monotonic() - recovery_t0

                if not ok:
                    result = self.request_motion_replan_from_source(
                        start_pos=start_pos,
                        traveled=traveled,
                        reason="single-IR clearance recovery failed",
                    )

                    if result == "ACCEPT_NODE":
                        return True

                    if result == "REPLAN_SOURCE":
                        return False

                    # Retreat itself failed.  This is the one case where
                    # continuing autonomously would be unsafe.
                    raise RuntimeError(
                        "Single-IR recovery failed and the robot could not "
                        "safely retreat to the source cell."
                    )

                # Recovery can become BOTH LOW / side-blocked and ask for a
                # route replan.  Treat it as a transient geometry event, not a
                # confirmed permanent wall.
                if self.ir_replan_requested:
                    result = self.request_motion_replan_from_source(
                        start_pos=start_pos,
                        traveled=traveled,
                        reason=(
                            "IR/Gimbal requested replan after "
                            "single-IR recovery"
                        ),
                    )

                    if result == "ACCEPT_NODE":
                        return True

                    if result == "REPLAN_SOURCE":
                        return False

                    raise RuntimeError(
                        "IR/Gimbal requested a motion replan but the robot "
                        "could not safely retreat to the source cell."
                    )

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

            # ------------------------------------------------
            # IN-MOTION FAKE-EXIT GUARD
            # ------------------------------------------------
            if (
                exit_motion_armed
                and not exit_motion_checked
                and EXIT_MOTION_MIN_TRAVEL_M
                <= traveled
                <= EXIT_MOTION_MAX_TRAVEL_M
            ):
                left_wall_now = (
                    left_cm is not None
                    and left_cm <= EXIT_CORRIDOR_WALL_MAX_CM
                )
                right_wall_now = (
                    right_cm is not None
                    and right_cm <= EXIT_CORRIDOR_WALL_MAX_CM
                )

                # Once one wall disappears in the latter half of an edge,
                # slow down.  This is not yet an exit decision.
                one_or_both_lost = (
                    (not left_wall_now) or (not right_wall_now)
                )

                exit_caution_slow = bool(
                    traveled >= EXIT_MOTION_CAUTION_START_M
                    and one_or_both_lost
                )

                if not left_wall_now and not right_wall_now:
                    exit_lost_both_count += 1

                    print(
                        f"[EXIT MOTION] BOTH walls lost "
                        f"{exit_lost_both_count}/"
                        f"{EXIT_MOTION_LOST_CONFIRM_COUNT} "
                        f"at d={traveled:.3f}m"
                    )
                else:
                    exit_lost_both_count = 0

                    if exit_caution_slow:
                        missing = (
                            "LEFT" if not left_wall_now else "RIGHT"
                        )
                        print(
                            f"[EXIT MOTION] {missing} wall lost at "
                            f"d={traveled:.3f}m -> CAUTION SLOW"
                        )

                if (
                    exit_lost_both_count
                    >= EXIT_MOTION_LOST_CONFIRM_COUNT
                ):
                    exit_motion_checked = True

                    confirmed = self.confirm_in_motion_exit_candidate(
                        source_cell=source_cell,
                        abs_dir=abs_dir,
                        start_pos=start_pos,
                        traveled=traveled,
                        left_cm=left_cm,
                        right_cm=right_cm,
                    )

                    if confirmed:
                        return False

                    # A noisy/normal geometry event was checked once and did
                    # not confirm as an exit. Do not keep rescanning fan during
                    # the same edge traversal.
                    exit_lost_both_count = 0

            elif (
                exit_motion_armed
                and traveled > EXIT_MOTION_MAX_TRAVEL_M
            ):
                # Very close to the target cell center. Keep the already-set
                # caution speed, but do not start another expensive fan scan.
                exit_motion_checked = True

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

            if exit_motion_armed and exit_caution_slow:
                x_cmd = min(
                    x_cmd,
                    EXIT_MOTION_CAUTION_SPEED_MPS
                )

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
            self.publish_gui_state()
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

        # Heading changes are reflected immediately in Mission Control even
        # before the next topology autosave.
        self.publish_gui_state()


    # --------------------------------------------------------
    # ADAPTIVE START / STAGING ANCHOR
    # --------------------------------------------------------

    def is_known_maze_ingress_edge(self, cell, abs_dir):
        return (
            START_ANCHOR_ENABLED
            and tuple(cell) == tuple(self.root)
            and int(abs_dir) % 4
                == int(self.known_maze_ingress_dir) % 4
        )

    def capture_start_anchor_profile(self):
        """
        Classify startup geometry for logging/map metadata only.

        The launch direction does NOT depend on this classification.
        The robot is assumed to be manually placed facing into the maze.

        This deliberately supports:
          - corridor start
          - one-sided-wall start
          - fully open staging-area start
        """
        if not START_ANCHOR_ENABLED:
            return None

        if self.start_anchor_profile is not None:
            return self.start_anchor_profile

        self.gimbal_front_down()
        front_mm = self.sample_tof_median()

        left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

        # Get coarse L/R ToF without moving chassis.
        left_mm = self.scan_tof_at_yaw(-90.0)
        right_mm = self.scan_tof_at_yaw(+90.0)
        self.gimbal_front_down()

        left_sharp_wall = (
            left_cm is not None
            and left_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )
        right_sharp_wall = (
            right_cm is not None
            and right_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )

        left_open = (
            left_mm is not None
            and left_mm >= START_OPEN_MM
        )
        front_open = (
            front_mm is not None
            and front_mm >= START_OPEN_MM
        )
        right_open = (
            right_mm is not None
            and right_mm >= START_OPEN_MM
        )

        if left_sharp_wall and right_sharp_wall:
            start_type = "CORRIDOR"
        elif left_sharp_wall or right_sharp_wall:
            start_type = "ONE_SIDE_WALL"
        elif left_open and right_open:
            start_type = "OPEN_STAGING"
        else:
            start_type = "MIXED"

        self.start_anchor_profile = {
            "cell": [int(self.root[0]), int(self.root[1])],
            "type": start_type,
            "maze_ingress_dir_index": int(self.known_maze_ingress_dir),
            "maze_ingress_dir": DIR_NAMES[self.known_maze_ingress_dir],
            "entrance_return_dir_index": int(self.known_entrance_dir),
            "entrance_return_dir": DIR_NAMES[self.known_entrance_dir],
            "front_mm": (
                None if front_mm is None else float(front_mm)
            ),
            "left_tof_mm": (
                None if left_mm is None else float(left_mm)
            ),
            "right_tof_mm": (
                None if right_mm is None else float(right_mm)
            ),
            "sharp_left_cm": (
                None if left_cm is None else float(left_cm)
            ),
            "sharp_right_cm": (
                None if right_cm is None else float(right_cm)
            ),
            "front_open": bool(front_open),
            "left_open": bool(left_open),
            "right_open": bool(right_open),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }

        print("\n================ START ANCHOR PROFILE ======================")
        print(f"Start type    : {start_type}")
        print(
            f"Maze ingress  : {DIR_NAMES[self.known_maze_ingress_dir]} "
            "(FORCED from startup orientation)"
        )
        print(
            f"Return side   : {DIR_NAMES[self.known_entrance_dir]}"
        )
        print(
            f"ToF L/F/R     : "
            f"{left_mm}/{front_mm}/{right_mm} mm"
        )
        print(
            f"Sharp L/R     : "
            f"{left_cm if left_cm is not None else 'far'} / "
            f"{right_cm if right_cm is not None else 'far'} cm"
        )
        print(
            "[START ANCHOR] LEFT/RIGHT openness at (0,0) will NOT "
            "become DFS branches."
        )
        print("===========================================================")

        return self.start_anchor_profile

    def apply_root_start_policy(self, scanned_open_dirs):
        """
        Root topology override.

        Regardless of whether the starting area is a corridor or completely
        open, only the physically-known forward maze-ingress edge is allowed
        into DFS.  The back edge remains the known return/entrance direction.

        This prevents an open staging area from being interpreted as a
        3-way/4-way maze intersection.
        """
        if not START_ANCHOR_ENABLED:
            return list(scanned_open_dirs)

        if self.current != self.root:
            return list(scanned_open_dirs)

        forced = []

        ingress = self.known_maze_ingress_dir

        # We trust placement/orientation more than startup side geometry.
        if START_FORCE_FRONT_OPEN:
            forced.append(ingress)
        elif ingress in scanned_open_dirs:
            forced.append(ingress)

        print(
            f"[START ANCHOR] raw root openings="
            f"{[DIR_NAMES[d] for d in scanned_open_dirs]} "
            f"-> DFS openings="
            f"{[DIR_NAMES[d] for d in forced]}"
        )

        return forced

    # --------------------------------------------------------
    # KNOWN ENTRANCE CORRIDOR
    # --------------------------------------------------------

    def is_known_entrance_edge(self, cell, abs_dir):
        return (
            tuple(cell) == tuple(self.root)
            and int(abs_dir) % 4 == int(self.known_entrance_dir) % 4
        )

    def capture_entrance_corridor_profile(self):
        """
        Observe the physical entrance corridor once at startup.

        Important design rule:
        the entrance identity comes from START CONTEXT, not from its shape.
        An exit/fake-exit may look geometrically identical, so geometry alone
        cannot distinguish them reliably.

        The chassis stays still.  The gimbal scans behind the robot while
        stationary yaw hold prevents reaction-torque drift.
        """
        if not ENTRANCE_CORRIDOR_PROFILE_ENABLED:
            return None

        if self.entrance_corridor_profile is not None:
            return self.entrance_corridor_profile

        print("\n================ ENTRANCE CORRIDOR PROFILE ================")
        print(f"Root         : {self.root}")
        print(
            f"Known entry  : {DIR_NAMES[self.known_entrance_dir]} "
            "(back of startup heading)"
        )
        print("Chassis      : HOLD POSITION")
        print("Gimbal       : scan entrance behind robot")
        print("===========================================================")

        self.stop_chassis()

        # Current side-wall signature at the root.
        left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

        fan = {}

        for angle in ENTRANCE_PROFILE_ANGLES_DEG:
            mm = self.scan_tof_at_yaw(angle)
            fan[angle] = mm

            state = (
                "OPEN"
                if mm is not None and mm >= ENTRANCE_OPEN_VERIFY_MM
                else "NEAR/CLOSED"
            )

            value_text = "NO DATA" if mm is None else f"{mm:.0f} mm"

            print(
                f"  [ENTRANCE] yaw={angle:+6.1f}° "
                f"{value_text:>10} -> {state}"
            )

        self.gimbal_front_down()

        back_mm = fan.get(+180.0)

        left_wall = (
            left_cm is not None
            and left_cm <= ENTRANCE_CORRIDOR_SIDE_WALL_MAX_CM
        )
        right_wall = (
            right_cm is not None
            and right_cm <= ENTRANCE_CORRIDOR_SIDE_WALL_MAX_CM
        )

        back_open = (
            back_mm is not None
            and back_mm >= ENTRANCE_OPEN_VERIFY_MM
        )

        # "corridor-like" is descriptive only.  It is NOT required to trust
        # the entrance because the entrance is known from the start setup.
        corridor_like = bool(left_wall and right_wall)

        self.entrance_corridor_profile = {
            "cell": [int(self.root[0]), int(self.root[1])],
            "dir_index": int(self.known_entrance_dir),
            "dir": DIR_NAMES[self.known_entrance_dir],
            "status": "KNOWN_ENTRANCE_CORRIDOR",
            "trusted_from_start_context": True,
            "back_center_mm": (
                None if back_mm is None else float(back_mm)
            ),
            "back_open_verified": bool(back_open),
            "corridor_side_walls_verified": bool(corridor_like),
            "sharp_left_cm": (
                None if left_cm is None else float(left_cm)
            ),
            "sharp_right_cm": (
                None if right_cm is None else float(right_cm)
            ),
            "sharp_left_adc": (
                None if left_adc is None else float(left_adc)
            ),
            "sharp_right_adc": (
                None if right_adc is None else float(right_adc)
            ),
            "fan_mm": {
                str(angle): (
                    None if fan.get(angle) is None
                    else float(fan.get(angle))
                )
                for angle in ENTRANCE_PROFILE_ANGLES_DEG
            },
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }

        ltxt = "far" if left_cm is None else f"{left_cm:.1f} cm"
        rtxt = "far" if right_cm is None else f"{right_cm:.1f} cm"
        btxt = "NO DATA" if back_mm is None else f"{back_mm:.0f} mm"

        print(
            f"[ENTRANCE PROFILE] L={ltxt} R={rtxt} "
            f"BACK={btxt}"
        )
        print(
            f"[ENTRANCE PROFILE] corridor_side_walls="
            f"{corridor_like} back_open={back_open}"
        )
        print(
            "[ENTRANCE PROFILE] identity=KNOWN FROM START CONTEXT; "
            "DFS will never treat this edge as an exit candidate."
        )

        if not back_open:
            print(
                "[ENTRANCE WARN] back ray is not >= "
                f"{ENTRANCE_OPEN_VERIFY_MM:.0f} mm. "
                "This may be caused by start placement or an object behind; "
                "the entrance remains protected from DFS."
            )

        if MAP_AUTOSAVE:
            self.save_map(final=False)

        return self.entrance_corridor_profile

    # --------------------------------------------------------
    # FAKE EXIT / OPEN-BOUNDARY GUARD
    # --------------------------------------------------------

    def exit_candidate_key(self, cell, abs_dir):
        return (tuple(cell), int(abs_dir) % 4)

    def is_exit_candidate(self, cell, abs_dir):
        return self.exit_candidate_key(cell, abs_dir) in self.exit_candidates

    def record_exit_candidate(
        self,
        cell,
        abs_dir,
        front_mm,
        left_cm,
        right_cm,
        fan_mm=None,
        reason="wide_boundary",
        wall_end_probe=None,
    ):
        key = self.exit_candidate_key(cell, abs_dir)

        fan_mm = fan_mm or {}
        wall_end_probe = wall_end_probe or {}

        self.exit_candidates[key] = {
            "cell": [int(cell[0]), int(cell[1])],
            "dir_index": int(abs_dir) % 4,
            "dir": DIR_NAMES[int(abs_dir) % 4],
            "status": "DEFERRED_DURING_EXPLORE",
            "reason": str(reason),
            "front_mm": None if front_mm is None else float(front_mm),
            "sharp_left_cm": None if left_cm is None else float(left_cm),
            "sharp_right_cm": None if right_cm is None else float(right_cm),
            "fan_mm": {
                str(angle): (
                    None if fan_mm.get(angle) is None
                    else float(fan_mm.get(angle))
                )
                for angle in EXIT_FAN_ANGLES_DEG
                if angle in fan_mm
            },
            "wall_end_probe": wall_end_probe,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        print(
            f"[EXIT CANDIDATE] cell={cell} dir={DIR_NAMES[abs_dir]} "
            f"reason={reason} "
            "-> DEFERRED; DFS WILL NOT LEAVE MAZE HERE"
        )

        if MAP_AUTOSAVE:
            self.save_map(final=False)

    @staticmethod
    def predicted_side_wall_hit_mm(side_cm, angle_deg):
        """
        If a perfectly straight side wall continues forward forever, a ray at
        `angle_deg` should hit it at approximately:

            range = perpendicular_side_distance / sin(|angle|)

        Sharp provides the perpendicular side distance at the robot.
        """
        if side_cm is None:
            return None

        angle_rad = math.radians(abs(float(angle_deg)))
        sin_v = math.sin(angle_rad)

        if sin_v <= 1e-6:
            return None

        return float(side_cm) * 10.0 / sin_v

    def wall_end_ray_vote(self, measured_mm, expected_mm):
        """
        True when the ToF ray travelled substantially farther than it should
        have if the currently-detected side wall continued ahead.
        """
        if measured_mm is None or expected_mm is None:
            return False

        threshold = max(
            EXIT_WALL_END_MIN_MEASURED_MM,
            expected_mm * EXIT_WALL_END_RATIO,
            expected_mm + EXIT_WALL_END_MARGIN_MM,
        )

        return measured_mm >= threshold

    def scan_wall_end_probe(self, left_cm, right_cm):
        """
        Detect the exact geometry shown by the real fake-exit:

            |       |
            | ROBOT |
            |       |
            |       |
            END   END
               OPEN

        Sharp still sees both nearby side walls.  Shallow forward ToF rays
        reveal whether those walls continue beyond the next grid boundary.
        """
        evidence = {
            "left": [],
            "right": [],
        }

        print(
            "[EXIT END-PROBE] side walls exist NOW; checking whether "
            "they terminate together ahead"
        )

        for side, side_cm, angles in (
            ("left", left_cm, EXIT_WALL_END_ANGLES_LEFT),
            ("right", right_cm, EXIT_WALL_END_ANGLES_RIGHT),
        ):
            for angle in angles:
                expected = self.predicted_side_wall_hit_mm(
                    side_cm,
                    angle,
                )
                measured = self.scan_tof_at_yaw(angle)

                vote = self.wall_end_ray_vote(
                    measured,
                    expected,
                )

                threshold = None
                if expected is not None:
                    threshold = max(
                        EXIT_WALL_END_MIN_MEASURED_MM,
                        expected * EXIT_WALL_END_RATIO,
                        expected + EXIT_WALL_END_MARGIN_MM,
                    )

                rec = {
                    "angle_deg": float(angle),
                    "measured_mm": (
                        None if measured is None else float(measured)
                    ),
                    "expected_continuing_wall_mm": (
                        None if expected is None else float(expected)
                    ),
                    "wall_end_threshold_mm": (
                        None if threshold is None else float(threshold)
                    ),
                    "wall_ended_vote": bool(vote),
                }

                evidence[side].append(rec)

                mtxt = "NO DATA" if measured is None else f"{measured:.0f}"
                etxt = "N/A" if expected is None else f"{expected:.0f}"
                ttxt = "N/A" if threshold is None else f"{threshold:.0f}"

                print(
                    f"    [END {side.upper():5s}] "
                    f"{angle:+5.1f}° measured={mtxt:>7} "
                    f"expected_wall={etxt:>7} "
                    f"vote_threshold={ttxt:>7} "
                    f"-> {'END' if vote else 'CONTINUE'}"
                )

        self.gimbal_front_down()

        left_votes = sum(
            1 for rec in evidence["left"]
            if rec["wall_ended_vote"]
        )
        right_votes = sum(
            1 for rec in evidence["right"]
            if rec["wall_ended_vote"]
        )

        evidence["left_votes"] = left_votes
        evidence["right_votes"] = right_votes

        both_ended = (
            left_votes >= EXIT_WALL_END_MIN_VOTES_PER_SIDE
            and right_votes >= EXIT_WALL_END_MIN_VOTES_PER_SIDE
        )

        evidence["both_walls_ended"] = bool(both_ended)

        print(
            f"[EXIT END-PROBE] votes "
            f"LEFT={left_votes}/{len(EXIT_WALL_END_ANGLES_LEFT)} "
            f"RIGHT={right_votes}/{len(EXIT_WALL_END_ANGLES_RIGHT)} "
            f"-> {'BOTH WALLS END' if both_ended else 'CORRIDOR CONTINUES'}"
        )

        return both_ended, evidence

    def scan_exit_fan(self):
        """
        Chassis is already facing the selected DFS direction.
        Scan a wide front fan with the same robust 7-sample median.
        """
        fan = {}

        for angle in EXIT_FAN_ANGLES_DEG:
            mm = self.scan_tof_at_yaw(angle)
            fan[angle] = mm

            state = (
                "OPEN"
                if mm is not None and mm >= EXIT_FAN_OPEN_MM
                else "BLOCKED"
            )

            value_text = "NO DATA" if mm is None else f"{mm:.0f} mm"

            print(
                f"    [EXIT FAN] {angle:+5.1f} deg "
                f"{value_text:>10} -> {state}"
            )

        self.gimbal_front_down()
        return fan

    def exit_guard_before_explore_edge(self, cell, abs_dir):
        """
        Conservative PRE-MOVE exit guard.

        V8.9 rule:
        NEVER classify a normal corridor as an exit merely because shallow
        rays predict that its current side walls end somewhere ahead.

        Before motion we only defer an edge when the robot is ALREADY sitting
        at an obviously broad/open boundary with BOTH side corridor walls gone.
        The common photo-like fake-exit case (walls still beside robot, ending
        ahead) is handled by the IN-MOTION guard inside move_one_cell().
        """
        if self.is_known_entrance_edge(cell, abs_dir):
            print(
                f"[ENTRANCE GUARD] {cell}->{DIR_NAMES[abs_dir]} "
                "is KNOWN_ENTRANCE_CORRIDOR"
            )
            return False

        if self.is_known_maze_ingress_edge(cell, abs_dir):
            print(
                f"[START ANCHOR] {cell}->{DIR_NAMES[abs_dir]} "
                "is KNOWN MAZE INGRESS -> bypass exit classifier"
            )
            return False

        if not EXIT_GUARD_ENABLED:
            return False

        if self.is_exit_candidate(cell, abs_dir):
            print(
                f"[EXIT GUARD] {cell}->{DIR_NAMES[abs_dir]} "
                "already recorded as EXIT_CANDIDATE"
            )
            return True

        self.gimbal_front_down()

        front_mm = self.sample_tof_median()
        left_cm, right_cm, _, _ = self.read_sharp_cm()

        left_wall = (
            left_cm is not None
            and left_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )
        right_wall = (
            right_cm is not None
            and right_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )

        left_text = "far" if left_cm is None else f"{left_cm:.1f}"
        right_text = "far" if right_cm is None else f"{right_cm:.1f}"
        front_text = "NO DATA" if front_mm is None else f"{front_mm:.0f}"

        rel_left_abs = (abs_dir - 1) % 4
        rel_right_abs = (abs_dir + 1) % 4
        known_open = set(self.open_dirs.get(cell, []))

        side_topology_left_open = rel_left_abs in known_open
        side_topology_right_open = rel_right_abs in known_open

        print(
            f"[EXIT GUARD PRE] cell={cell} dir={DIR_NAMES[abs_dir]} "
            f"front={front_text}mm L={left_text}cm R={right_text}cm "
            f"sideTopo="
            f"{'OPEN' if side_topology_left_open else 'WALL'}/"
            f"{'OPEN' if side_topology_right_open else 'WALL'}"
        )

        if (
            front_mm is not None
            and front_mm < EXIT_FRONT_MIN_SAFE_MM
        ):
            print(
                "[EXIT GUARD PRE] close front object -> normal obstacle logic"
            )
            return False

        # THIS IS THE KEY FIX:
        # If either/both side walls still exist at the robot, allow the robot
        # to START moving.  We will watch whether BOTH walls disappear during
        # the edge traversal instead of guessing from projected geometry.
        if left_wall or right_wall:
            print(
                "[EXIT GUARD PRE] corridor wall exists NOW "
                "-> allow move; in-motion wall-loss monitor armed"
            )
            return False

        # A mapped intersection is not an outside boundary.
        if side_topology_left_open or side_topology_right_open:
            print(
                "[EXIT GUARD PRE] mapped side opening/intersection "
                "-> allow normal edge"
            )
            return False

        # Only the rare case where both walls are ALREADY absent before moving
        # gets the old broad fan verification.
        print(
            "[EXIT GUARD PRE] both side walls already absent "
            "-> broad fan verification"
        )

        fan = self.scan_exit_fan()

        fan_eval = self.evaluate_exit_fan(fan)
        left_open = fan_eval["left_open"]
        right_open = fan_eval["right_open"]
        strong_total = fan_eval["strong_total"]
        wide_boundary = fan_eval["wide_boundary"]

        print(
            f"[EXIT GUARD PRE] fan votes "
            f"OPEN L={left_open}/3 R={right_open}/3 "
            f"STRONG>={EXIT_FAN_STRONG_OPEN_MM:.0f}mm "
            f"{strong_total}/6 -> "
            f"{'BOUNDARY' if wide_boundary else 'JUNCTION/DEEP PATH'}"
        )

        if not wide_boundary:
            return False

        self.record_exit_candidate(
            cell=cell,
            abs_dir=abs_dir,
            front_mm=front_mm,
            left_cm=left_cm,
            right_cm=right_cm,
            fan_mm=fan,
            reason="broad_open_boundary_before_motion",
            wall_end_probe={},
        )
        return True

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

        # ----------------------------------------------------
        # NODE-SCAN IR POLICY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # Once move_one_cell() has accepted that the robot has ARRIVED at a
        # cell/node, IR LOW is no longer treated as a command to slide.
        #
        # Why:
        #   - a wall can legitimately be very close at a junction/dead end
        #   - move_one_cell() may accept the cell near the front wall
        #   - sliding here can move the robot away from the intended node
        #   - it can also create the old conflict:
        #         "cell reached" -> IR slide -> front too close -> RuntimeError
        #
        # At a node, IR therefore acts only as a HIGH-PRIORITY TRIGGER:
        #     STOP -> keep pose -> Gimbal scan LEFT / FRONT / RIGHT
        #
        # Recovery/sliding is still used during translation, during an
        # explicit recovery slide, and after turns when corner clearance is
        # actually required.
        pre_l_low, pre_r_low, pre_l_raw, pre_r_raw = self.read_ir_filtered()
        pre_dual_event, pre_dual_reason = self.consume_ir_dual_sequence()

        if pre_dual_event:
            self.stop_chassis()
            print(
                f"[IR NODE SCAN {cell}] sequential event: "
                f"{pre_dual_reason} -> HOLD POSITION + GIMBAL L/F/R"
            )

        elif pre_l_low and pre_r_low:
            self.stop_chassis()
            print(
                f"[IR NODE SCAN {cell}] BOTH LOW "
                f"IR_L={pre_l_raw} IR_R={pre_r_raw} "
                "-> HOLD POSITION + GIMBAL L/F/R"
            )

        elif pre_l_low or pre_r_low:
            self.stop_chassis()

            side = "LEFT" if pre_l_low else "RIGHT"

            print(
                f"[IR NODE SCAN {cell}] {side} LOW "
                f"IR_L={pre_l_raw} IR_R={pre_r_raw} "
                "-> NO SLIDE; HOLD POSITION + GIMBAL L/F/R"
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
    # OPEN-AREA TRAP FALLBACK
    # --------------------------------------------------------

    def detect_open_area_trap(self, cell):
        """
        Detect a pseudo-cell that is actually outside the maze.

        Trigger conditions are intentionally strong:
          - not root
          - cell has a parent (we just came through a known edge)
          - LEFT/FRONT/RIGHT all classified OPEN
          - neither Sharp sees a normal nearby side wall
          - wide diagonal fan is broadly open

        A legitimate intersection normally has nearby corner walls that stop
        several diagonal fan rays.  Open floor outside the maze usually does
        not.
        """
        if not OPEN_AREA_TRAP_ENABLED:
            return None

        if tuple(cell) == tuple(self.root):
            return None

        parent = self.parent.get(cell)

        if parent is None:
            return None

        scan = self.cell_scan_mm.get(cell, {})

        l = scan.get("LEFT")
        f = scan.get("FRONT")
        r = scan.get("RIGHT")

        all_three_open = all(
            value is not None and value >= TOF_OPEN_THRESHOLD_MM
            for value in (l, f, r)
        )

        if not all_three_open:
            return None

        left_cm, right_cm, _, _ = self.read_sharp_cm()

        left_wall = (
            left_cm is not None
            and left_cm <= OPEN_AREA_TRAP_SIDE_WALL_MAX_CM
        )
        right_wall = (
            right_cm is not None
            and right_cm <= OPEN_AREA_TRAP_SIDE_WALL_MAX_CM
        )

        if left_wall or right_wall:
            return None

        print(
            f"[OPEN AREA TRAP] suspicious cell={cell}: "
            f"L/F/R={l:.0f}/{f:.0f}/{r:.0f}mm, "
            "Sharp side walls absent -> WIDE FAN VERIFY"
        )

        fan = self.scan_exit_fan()

        fan_eval = self.evaluate_exit_fan(fan)

        open_count = sum(
            1
            for angle in EXIT_FAN_ANGLES_DEG
            if fan.get(angle) is not None
            and fan[angle] >= EXIT_FAN_OPEN_MM
        )

        long_count = sum(
            1
            for angle in EXIT_FAN_ANGLES_DEG
            if fan.get(angle) is not None
            and fan[angle] >= OPEN_AREA_TRAP_LONG_MM
        )

        broad_open = (
            fan_eval["wide_boundary"]
            and open_count >= OPEN_AREA_TRAP_MIN_OPEN_FAN_RAYS
            and long_count >= OPEN_AREA_TRAP_MIN_LONG_FAN_RAYS
        )

        print(
            f"[OPEN AREA TRAP] fan open={open_count}/6 "
            f"strong>={EXIT_FAN_STRONG_OPEN_MM:.0f}mm="
            f"{fan_eval['strong_total']}/6 "
            f"very-long>={OPEN_AREA_TRAP_LONG_MM:.0f}mm={long_count}/6 "
            f"-> {'OUTSIDE SUSPECTED' if broad_open else 'VALID INTERSECTION/DEEP PATH'}"
        )

        if not broad_open:
            return None

        return {
            "parent": parent,
            "left_cm": left_cm,
            "right_cm": right_cm,
            "scan": dict(scan),
            "fan": dict(fan),
        }

    def rollback_open_area_cell(self, cell, stack, evidence):
        """
        We already crossed one edge too far.  Do not explore another direction.
        Return through the exact edge we just used, mark that parent->cell
        opening as EXIT_CANDIDATE, and remove the outside pseudo-cell from map.
        """
        parent = evidence["parent"]
        incoming_dir = direction_between(parent, cell)
        back_dir = direction_between(cell, parent)

        print(
            f"[OPEN AREA TRAP] {cell} is treated as OUTSIDE. "
            f"Immediate return {cell}->{parent}; DO NOT EXPLORE SIDE BRANCHES."
        )

        self.record_exit_candidate(
            cell=parent,
            abs_dir=incoming_dir,
            front_mm=None,
            left_cm=evidence.get("left_cm"),
            right_cm=evidence.get("right_cm"),
            fan_mm=evidence.get("fan", {}),
            reason="open_area_detected_after_boundary_crossing",
            wall_end_probe={
                "outside_cell": [int(cell[0]), int(cell[1])],
                "outside_scan_mm": evidence.get("scan", {}),
            },
        )

        self.turn_to_direction(back_dir)

        ok = self.move_one_cell()

        if not ok:
            self.stop_chassis()
            raise RuntimeError(
                f"Open-area trap detected at {cell}, but emergency "
                f"return to parent {parent} failed."
            )

        # Physically back at parent.
        self.current = parent

        if stack and stack[-1] == cell:
            stack.pop()

        self.visited.discard(cell)
        self.dead_end_cells.discard(cell)
        self.parent.pop(cell, None)
        self.open_dirs.pop(cell, None)
        self.cell_scan_mm.pop(cell, None)

        # The physical opening remains represented by EXIT_CANDIDATE,
        # not by DFS open_dirs.
        self.open_dirs[parent] = [
            d
            for d in self.open_dirs.get(parent, [])
            if d != incoming_dir
        ]

        print(
            f"[OPEN AREA TRAP] back inside at {parent}; "
            f"removed pseudo-cell {cell} from learned map"
        )

        if MAP_AUTOSAVE:
            self.save_map(final=False)

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

        for (cell, _abs_dir) in self.exit_candidates.keys():
            cells.add(cell)

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
                "exit_candidate_dirs": [
                    DIR_NAMES[d]
                    for (candidate_cell, d) in self.exit_candidates.keys()
                    if candidate_cell == cell
                ],
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
                "sharp_follow_near_cm": SHARP_FOLLOW_NEAR_CM,
                "sharp_follow_far_cm": SHARP_FOLLOW_FAR_CM,
                "exit_guard_enabled": EXIT_GUARD_ENABLED,
                "exit_corridor_wall_max_cm": EXIT_CORRIDOR_WALL_MAX_CM,
                "exit_fan_angles_deg": list(EXIT_FAN_ANGLES_DEG),
                "exit_fan_open_mm": EXIT_FAN_OPEN_MM,
                "exit_fan_min_open_per_side": EXIT_FAN_MIN_OPEN_PER_SIDE,
                "exit_fan_strong_open_mm": EXIT_FAN_STRONG_OPEN_MM,
                "exit_fan_min_strong_total": EXIT_FAN_MIN_STRONG_TOTAL,
                "exit_motion_min_travel_m": EXIT_MOTION_MIN_TRAVEL_M,
                "exit_motion_lost_confirm_count": EXIT_MOTION_LOST_CONFIRM_COUNT,
                "exit_wall_end_probe_enabled": EXIT_WALL_END_PROBE_ENABLED,
                "exit_wall_end_angles_left": list(EXIT_WALL_END_ANGLES_LEFT),
                "exit_wall_end_angles_right": list(EXIT_WALL_END_ANGLES_RIGHT),
                "exit_wall_end_front_arm_mm": EXIT_WALL_END_FRONT_ARM_MM,
                "exit_wall_end_ratio": EXIT_WALL_END_RATIO,
                "exit_wall_end_margin_mm": EXIT_WALL_END_MARGIN_MM,
                "exit_wall_end_min_measured_mm": EXIT_WALL_END_MIN_MEASURED_MM,
                "entrance_corridor_profile_enabled": ENTRANCE_CORRIDOR_PROFILE_ENABLED,
                "entrance_profile_angles_deg": list(ENTRANCE_PROFILE_ANGLES_DEG),
                "entrance_open_verify_mm": ENTRANCE_OPEN_VERIFY_MM,
                "exit_motion_guard_enabled": EXIT_MOTION_GUARD_ENABLED,
                "exit_motion_min_travel_m": EXIT_MOTION_MIN_TRAVEL_M,
                "exit_motion_max_travel_m": EXIT_MOTION_MAX_TRAVEL_M,
                "exit_motion_lost_confirm_count": EXIT_MOTION_LOST_CONFIRM_COUNT,
                "exit_motion_caution_start_m": EXIT_MOTION_CAUTION_START_M,
                "exit_motion_caution_speed_mps": EXIT_MOTION_CAUTION_SPEED_MPS,
                "open_area_trap_enabled": OPEN_AREA_TRAP_ENABLED,
                "open_area_trap_min_open_fan_rays": OPEN_AREA_TRAP_MIN_OPEN_FAN_RAYS,
                "open_area_trap_long_mm": OPEN_AREA_TRAP_LONG_MM,
                "open_area_trap_min_long_fan_rays": OPEN_AREA_TRAP_MIN_LONG_FAN_RAYS,
                "start_anchor_enabled": START_ANCHOR_ENABLED,
                "start_maze_ingress_dir": START_MAZE_INGRESS_DIR,
                "start_force_front_open": START_FORCE_FRONT_OPEN,
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
            "start_anchor": {
                "enabled": bool(START_ANCHOR_ENABLED),
                "cell": [int(self.root[0]), int(self.root[1])],
                "maze_ingress_dir_index": int(self.known_maze_ingress_dir),
                "maze_ingress_dir": DIR_NAMES[self.known_maze_ingress_dir],
                "profile": self.start_anchor_profile,
            },
            "known_entrance": {
                "cell": [int(self.root[0]), int(self.root[1])],
                "dir_index": int(self.known_entrance_dir),
                "dir": DIR_NAMES[self.known_entrance_dir],
                "status": "KNOWN_ENTRANCE_CORRIDOR",
                "profile": self.entrance_corridor_profile,
            },
            "exit_candidates": [
                rec
                for _, rec in sorted(
                    self.exit_candidates.items(),
                    key=lambda item: (
                        item[0][0][1],
                        item[0][0][0],
                        item[0][1],
                    ),
                )
            ],
            "targets": self.detected_targets,
            "target_glimpses": self.target_glimpses,
            "target_vision_policy": {
                "enabled": bool(self.target_vision_enabled),
                "roi_norm": [
                    TARGET_ROI_X_MIN,
                    TARGET_ROI_Y_MIN,
                    TARGET_ROI_X_MAX,
                    TARGET_ROI_Y_MAX,
                ],
                "confirm_frames": TARGET_CONFIRM_FRAMES,
                "foam_gate": {
                    "enabled": TARGET_FOAM_GATE_ENABLED,
                    "fail_closed": TARGET_FOAM_FAIL_CLOSED,
                    "hsv_low": list(TARGET_FOAM_HSV_LOW),
                    "hsv_high": list(TARGET_FOAM_HSV_HIGH),
                    "center_margin_px": TARGET_FOAM_CENTER_MARGIN_PX,
                    "min_bbox_below_frac": TARGET_FOAM_MIN_BBOX_BELOW_FRAC,
                },
                "glimpse_linger_sec": TARGET_GLIMPSE_LINGER_SEC,
                "return_rescan_enabled": TARGET_RETURN_RESCAN_ENABLED,
                "sweep_poses_yaw_pitch_deg": [
                    [float(yaw), float(pitch)]
                    for yaw, pitch in TARGET_SWEEP_POSES
                ],
            },
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
        lines.append(
            f"Known maze ingress: {self.root} -> "
            f"{DIR_NAMES[self.known_maze_ingress_dir]}"
        )
        lines.append(
            f"Known entrance/return: {self.root} -> "
            f"{DIR_NAMES[self.known_entrance_dir]}"
        )

        if self.entrance_corridor_profile:
            lines.append(
                "  profile: "
                f"back={self.entrance_corridor_profile.get('back_center_mm')} mm, "
                f"corridor_walls="
                f"{self.entrance_corridor_profile.get('corridor_side_walls_verified')}"
            )

        if self.exit_candidates:
            lines.append("Deferred EXIT_CANDIDATES (real/fake not traversed):")
            for (cell, d), rec in sorted(
                self.exit_candidates.items(),
                key=lambda item: (
                    item[0][0][1],
                    item[0][0][0],
                    item[0][1],
                ),
            ):
                lines.append(
                    f"  {cell} -> {DIR_NAMES[d]} "
                    f"front={rec.get('front_mm')} mm"
                )

        lines.append("")
        return "\n".join(lines)


    def render_svg_map(self):
        """
        Render a wall map as SVG.

        Visual language:
          - each mapped cell is a square
          - black thick lines = confirmed walls
          - gaps = confirmed open directions
          - red X = blocked/failed edge
          - green = root/start, blue = visited, red = hard dead-end
          - orange arrow = open frontier to an unmapped cell
        """
        cells = self.mapped_cells()

        if not cells:
            return """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="240" viewBox="0 0 800 240">
  <rect width="100%" height="100%" fill="white"/>
  <text x="40" y="70" font-family="Arial, sans-serif" font-size="28" fill="#111">RoboMaster Maze Map</text>
  <text x="40" y="120" font-family="Arial, sans-serif" font-size="22" fill="#666">(map empty)</text>
</svg>
"""

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        cell_px = 96
        margin = 88
        header_h = 120
        legend_h = 150
        wall_stroke = 8
        thin_stroke = 2

        cols = max_x - min_x + 1
        rows = max_y - min_y + 1

        width = margin * 2 + cols * cell_px + 1
        height = header_h + rows * cell_px + legend_h + 1

        def cell_xy(cell):
            x, y = cell
            px = margin + (x - min_x) * cell_px
            py = header_h + (max_y - y) * cell_px
            return px, py

        def side_segment(px, py, d):
            if d == 0:   # N
                return (px, py, px + cell_px, py)
            if d == 1:   # E
                return (px + cell_px, py, px + cell_px, py + cell_px)
            if d == 2:   # S
                return (px, py + cell_px, px + cell_px, py + cell_px)
            # W
            return (px, py, px, py + cell_px)

        def opening_midpoint(px, py, d):
            if d == 0:
                return (px + cell_px / 2, py)
            if d == 1:
                return (px + cell_px, py + cell_px / 2)
            if d == 2:
                return (px + cell_px / 2, py + cell_px)
            return (px, py + cell_px / 2)

        def cell_fill(cell):
            if cell == self.root:
                return "#d7f8d0"
            if cell in self.dead_end_cells:
                return "#ffd7d7"
            if cell in self.visited:
                return "#d9e9ff"
            return "#efefef"

        svg = []
        append = svg.append

        append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        append('<rect width="100%" height="100%" fill="white"/>')

        # Title / meta
        append('<text x="28" y="42" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#111">RoboMaster Maze Map</text>')
        append(f'<text x="28" y="74" font-family="Arial, sans-serif" font-size="18" fill="#444">Cells: {len(cells)} | Complete: {self.map_complete} | Root: {self.root} | Grid pitch: {CELL_LENGTH_M:.2f} m | Open threshold: {TOF_OPEN_THRESHOLD_MM} mm</text>')
        append('<text x="28" y="100" font-family="Arial, sans-serif" font-size="16" fill="#666">North is up. Thick black edges are walls. Gaps are open passages.</text>')

        # Border around map area
        append(f'<rect x="{margin - 10}" y="{header_h - 10}" width="{cols * cell_px + 20}" height="{rows * cell_px + 20}" fill="none" stroke="#cfcfcf" stroke-width="2"/>')

        # Light grid background
        for c in range(cols + 1):
            x = margin + c * cell_px
            append(f'<line x1="{x}" y1="{header_h}" x2="{x}" y2="{header_h + rows * cell_px}" stroke="#f1f1f1" stroke-width="1"/>')
        for r in range(rows + 1):
            y = header_h + r * cell_px
            append(f'<line x1="{margin}" y1="{y}" x2="{margin + cols * cell_px}" y2="{y}" stroke="#f1f1f1" stroke-width="1"/>')

        # Draw cells
        for cell in sorted(cells, key=lambda p: (p[1], p[0])):
            px, py = cell_xy(cell)
            fill = cell_fill(cell)
            append(f'<rect x="{px}" y="{py}" width="{cell_px}" height="{cell_px}" fill="{fill}" fill-opacity="0.55" stroke="none"/>')

            # cell label / coordinate
            label = "S" if cell == self.root else ("D" if cell in self.dead_end_cells else "o")
            append(f'<text x="{px + 8}" y="{py + 22}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#222">{label}</text>')
            append(f'<text x="{px + 8}" y="{py + cell_px - 10}" font-family="Consolas, monospace" font-size="13" fill="#333">({cell[0]},{cell[1]})</text>')

            # Optional scan values
            scan = self.cell_scan_mm.get(cell)
            if isinstance(scan, dict):
                small = []
                if scan.get("LEFT") is not None:
                    small.append(f"L{int(scan['LEFT'])}")
                if scan.get("FRONT") is not None:
                    small.append(f"F{int(scan['FRONT'])}")
                if scan.get("RIGHT") is not None:
                    small.append(f"R{int(scan['RIGHT'])}")
                if small:
                    scan_txt = " ".join(small)
                    append(f'<text x="{px + 8}" y="{py + 40}" font-family="Consolas, monospace" font-size="11" fill="#555">{scan_txt}</text>')

            open_dirs = set(self.open_dirs.get(cell, []))

            # Open-frontier marker: direction is open from this cell, but the
            # destination cell has not been mapped yet.
            for d in open_dirs:
                nb = neighbor(cell, d)
                if nb in cells or self.is_blocked(cell, nb):
                    continue

                mx, my = opening_midpoint(px, py, d)
                if d == 0:
                    points = f"{mx},{my - 14} {mx - 8},{my - 2} {mx + 8},{my - 2}"
                elif d == 1:
                    points = f"{mx + 14},{my} {mx + 2},{my - 8} {mx + 2},{my + 8}"
                elif d == 2:
                    points = f"{mx},{my + 14} {mx - 8},{my + 2} {mx + 8},{my + 2}"
                else:
                    points = f"{mx - 14},{my} {mx - 2},{my - 8} {mx - 2},{my + 8}"

                append(f'<polygon points="{points}" fill="#ff9800" fill-opacity="0.95"/>')

            # Walls. EXIT_CANDIDATE and the known entrance are physical
            # openings, even though they are deliberately excluded from DFS.
            for d in range(4):
                nb = neighbor(cell, d)

                candidate_opening = self.is_exit_candidate(cell, d)
                entrance_opening = (
                    cell == self.root
                    and d == self.known_entrance_dir
                )

                ingress_opening = (
                    cell == self.root
                    and d == self.known_maze_ingress_dir
                )

                is_open = (
                    (d in open_dirs and not self.is_blocked(cell, nb))
                    or candidate_opening
                    or entrance_opening
                    or ingress_opening
                )

                if not is_open:
                    x1, y1, x2, y2 = side_segment(px, py, d)
                    append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111" stroke-width="{wall_stroke}" stroke-linecap="square"/>')

            # Deferred wide-open boundary markers.
            for d in range(4):
                if not self.is_exit_candidate(cell, d):
                    continue

                mx, my = opening_midpoint(px, py, d)
                append(f'<circle cx="{mx}" cy="{my}" r="10" fill="#9c27b0" stroke="white" stroke-width="2"/>')
                append(f'<text x="{mx + 12}" y="{my - 12}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#9c27b0">EXIT?</text>')

            # Known maze-ingress marker.
            if cell == self.root:
                d = self.known_maze_ingress_dir
                mx, my = opening_midpoint(px, py, d)
                append(f'<circle cx="{mx}" cy="{my}" r="10" fill="#1565c0" stroke="white" stroke-width="2"/>')
                append(f'<text x="{mx + 12}" y="{my - 12}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#1565c0">MAZE</text>')

            # Known root entrance marker.
            if cell == self.root:
                mx, my = opening_midpoint(
                    px, py, self.known_entrance_dir
                )
                append(f'<circle cx="{mx}" cy="{my}" r="10" fill="#2e7d32" stroke="white" stroke-width="2"/>')
                append(f'<text x="{mx + 12}" y="{my - 12}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#2e7d32">IN CORRIDOR</text>')

        # Blocked/failed edges as red X between cells
        for edge in sorted(self.blocked_edges):
            a, b = edge
            if a not in cells or b not in cells:
                continue

            ax, ay = cell_xy(a)
            bx, by = cell_xy(b)
            cx = (ax + bx) / 2 + cell_px / 2
            cy = (ay + by) / 2 + cell_px / 2
            s = 12
            append(f'<line x1="{cx - s}" y1="{cy - s}" x2="{cx + s}" y2="{cy + s}" stroke="#d32f2f" stroke-width="4"/>')
            append(f'<line x1="{cx - s}" y1="{cy + s}" x2="{cx + s}" y2="{cy - s}" stroke="#d32f2f" stroke-width="4"/>')

        # North arrow
        nx = width - 70
        ny = 55
        append(f'<line x1="{nx}" y1="{ny + 28}" x2="{nx}" y2="{ny - 10}" stroke="#111" stroke-width="4"/>')
        append(f'<polygon points="{nx},{ny - 24} {nx - 10},{ny - 4} {nx + 10},{ny - 4}" fill="#111"/>')
        append(f'<text x="{nx - 7}" y="{ny + 50}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111">N</text>')

        # Legend
        lx = 28
        ly = header_h + rows * cell_px + 38
        append(f'<text x="{lx}" y="{ly - 10}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111">Legend</text>')

        # start
        append(f'<rect x="{lx}" y="{ly + 6}" width="22" height="22" fill="#d7f8d0" stroke="#666"/>')
        append(f'<text x="{lx + 34}" y="{ly + 23}" font-family="Arial, sans-serif" font-size="16" fill="#333">Start / root</text>')

        # visited
        append(f'<rect x="{lx + 200}" y="{ly + 6}" width="22" height="22" fill="#d9e9ff" stroke="#666"/>')
        append(f'<text x="{lx + 234}" y="{ly + 23}" font-family="Arial, sans-serif" font-size="16" fill="#333">Visited cell</text>')

        # dead end
        append(f'<rect x="{lx + 390}" y="{ly + 6}" width="22" height="22" fill="#ffd7d7" stroke="#666"/>')
        append(f'<text x="{lx + 424}" y="{ly + 23}" font-family="Arial, sans-serif" font-size="16" fill="#333">Dead-end cell</text>')

        # wall sample
        y2 = ly + 58
        append(f'<line x1="{lx}" y1="{y2}" x2="{lx + 26}" y2="{y2}" stroke="#111" stroke-width="{wall_stroke}" />')
        append(f'<text x="{lx + 34}" y="{y2 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Wall</text>')

        append(f'<line x1="{lx + 200}" y1="{y2}" x2="{lx + 226}" y2="{y2}" stroke="#d32f2f" stroke-width="4" />')
        append(f'<line x1="{lx + 200}" y1="{y2 + 12}" x2="{lx + 226}" y2="{y2 - 12}" stroke="#d32f2f" stroke-width="4" />')
        append(f'<text x="{lx + 234}" y="{y2 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Blocked / failed edge</text>')

        append(f'<polygon points="{lx + 430},{y2 - 14} {lx + 422},{y2 - 2} {lx + 438},{y2 - 2}" fill="#ff9800"/>')
        append(f'<text x="{lx + 448}" y="{y2 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Open frontier to unmapped area</text>')

        y3 = y2 + 34
        append(f'<circle cx="{lx + 10}" cy="{y3}" r="8" fill="#9c27b0"/>')
        append(f'<text x="{lx + 34}" y="{y3 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Deferred EXIT_CANDIDATE</text>')
        append(f'<circle cx="{lx + 390}" cy="{y3}" r="8" fill="#2e7d32"/>')
        append(f'<text x="{lx + 414}" y="{y3 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Known entrance corridor / safe return</text>')

        append('</svg>')
        return "\n".join(svg)

    def save_map(self, final=False):
        """
        Save reusable JSON + human-readable ASCII + SVG wall map.

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

        svg_text = self.render_svg_map()

        tmp_svg = MAP_LATEST_SVG.with_suffix(".svg.tmp")
        tmp_svg.write_text(svg_text, encoding="utf-8")
        os.replace(tmp_svg, MAP_LATEST_SVG)

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

            svg_snapshot = MAP_DIR / f"maze_{stamp}.svg"
            svg_snapshot.write_text(svg_text, encoding="utf-8")

        print(
            f"[MAP SAVE] cells={len(payload['cells'])} "
            f"complete={payload['complete']} -> {MAP_LATEST_JSON}"
        )
        print(f"[MAP SAVE] wall image -> {MAP_LATEST_SVG}")

        if snapshot is not None:
            print(f"[MAP SNAPSHOT] {snapshot}")

        # The SVG/JSON remain the persistent outputs, while Mission Control
        # receives the same topology immediately for live rendering.
        self.publish_gui_state()

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
        self.exit_candidates = {}

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

        start_anchor = data.get("start_anchor")
        if isinstance(start_anchor, dict):
            try:
                self.known_maze_ingress_dir = int(
                    start_anchor.get(
                        "maze_ingress_dir_index",
                        self.known_maze_ingress_dir,
                    )
                ) % 4
            except Exception:
                pass

            profile = start_anchor.get("profile")
            if isinstance(profile, dict):
                self.start_anchor_profile = dict(profile)

        entrance = data.get("known_entrance")
        if isinstance(entrance, dict):
            try:
                self.known_entrance_dir = int(
                    entrance.get("dir_index", self.known_entrance_dir)
                ) % 4
            except Exception:
                pass

            profile = entrance.get("profile")
            if isinstance(profile, dict):
                self.entrance_corridor_profile = dict(profile)

        for rec in data.get("exit_candidates", []):
            try:
                cell_v = rec["cell"]
                cell = (int(cell_v[0]), int(cell_v[1]))
                d = int(rec["dir_index"]) % 4
                self.exit_candidates[(cell, d)] = dict(rec)
            except Exception:
                continue

        # Restore target memory when replaying a saved map. Target records are
        # descriptive; known-map motion does not depend on them.
        self.detected_targets = [
            dict(rec)
            for rec in data.get("targets", [])
            if isinstance(rec, dict)
        ]
        self.target_id_seq = 0
        for rec in self.detected_targets:
            tid = str(rec.get("id", ""))
            if tid.startswith("T"):
                try:
                    self.target_id_seq = max(
                        self.target_id_seq,
                        int(tid[1:]),
                    )
                except Exception:
                    pass

        self.target_glimpses = [
            dict(rec)
            for rec in data.get("target_glimpses", [])
            if isinstance(rec, dict)
        ]
        self.target_glimpse_seq = 0
        for rec in self.target_glimpses:
            gid = str(rec.get("id", ""))
            if gid.startswith("G"):
                try:
                    self.target_glimpse_seq = max(
                        self.target_glimpse_seq,
                        int(gid[1:]),
                    )
                except Exception:
                    pass

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
        print(f"Exit cand.  : {len(self.exit_candidates)}")
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

                if self.motion_replan_requested:
                    reason = self.motion_replan_reason
                    self.motion_replan_requested = False
                    self.motion_replan_reason = None

                    raise RuntimeError(
                        f"Known-map route was safely aborted and returned to "
                        f"{current}: {reason}. The environment should be "
                        "re-observed before trusting the saved route."
                    )

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

    def has_unvisited_frontier(self):
        """
        True while the explored graph still contains something DFS must visit.

        A visited-but-not-yet-scanned cell also counts as unfinished.
        """
        for cell in self.visited:
            if cell not in self.open_dirs:
                return True

            for d in self.open_dirs.get(cell, []):
                nb = neighbor(cell, d)

                if self.is_blocked(cell, nb):
                    continue

                if nb not in self.visited:
                    return True

        return False

    def confirmed_explored_neighbors(self, cell):
        """
        Safe graph used for the FAST RETURN HOME planner.

        An edge is accepted when:
          1) it is a parent-child edge that the robot physically traversed, OR
          2) BOTH endpoint scans agree that the edge is open.

        This means shortcuts/loops can be used for a faster return, while an
        unconfirmed one-sided ToF opening is not blindly trusted.
        """
        result = []

        for d in range(4):
            nb = neighbor(cell, d)

            if nb not in self.visited:
                continue

            if self.is_blocked(cell, nb):
                continue

            if self.edge_key(cell, nb) in self.transient_runtime_blocked_edges:
                continue

            physically_traversed = (
                self.parent.get(cell) == nb
                or self.parent.get(nb) == cell
            )

            reverse_d = (d + 2) % 4

            mutually_scanned_open = (
                d in self.open_dirs.get(cell, [])
                and reverse_d in self.open_dirs.get(nb, [])
            )

            if physically_traversed or mutually_scanned_open:
                result.append((d, nb))

        return result

    @staticmethod
    def estimated_turn_time(from_heading, to_heading):
        delta = (to_heading - from_heading) % 4

        if delta == 0:
            return 0.0

        if delta == 2:
            return FAST_RETURN_TURN_180_EST_SEC

        return FAST_RETURN_TURN_90_EST_SEC

    def plan_fastest_return(self, start, start_heading, goal):
        """
        Dijkstra over (cell, heading), not just cell.

        Cost ~= chassis travel time + physical turn time.
        Therefore among map routes it can choose one with fewer turns instead
        of blindly taking a cell-count-only DFS parent path.
        """
        if start == goal:
            return [], 0.0

        # state = (cell, heading)
        start_state = (start, start_heading % 4)

        pq = [(0.0, start[0], start[1], start_heading % 4)]
        best = {start_state: 0.0}
        previous = {start_state: None}
        previous_action = {}

        goal_state = None

        while pq:
            cost, x, y, heading = heapq.heappop(pq)
            cell = (x, y)
            state = (cell, heading)

            if cost > best.get(state, float("inf")) + 1e-9:
                continue

            if cell == goal:
                goal_state = state
                break

            for d, nb in self.confirmed_explored_neighbors(cell):
                step_cost = (
                    FAST_RETURN_MOVE_EST_SEC
                    + self.estimated_turn_time(heading, d)
                )

                next_state = (nb, d)
                new_cost = cost + step_cost

                if new_cost + 1e-9 < best.get(next_state, float("inf")):
                    best[next_state] = new_cost
                    previous[next_state] = state
                    previous_action[next_state] = d

                    heapq.heappush(
                        pq,
                        (new_cost, nb[0], nb[1], d)
                    )

        if goal_state is None:
            raise RuntimeError(
                f"No confirmed route from {start} to {goal}."
            )

        reversed_steps = []
        cur = goal_state

        while cur != start_state:
            prev = previous[cur]

            if prev is None:
                raise RuntimeError("Fast-return path reconstruction failed.")

            d = previous_action[cur]
            target_cell = cur[0]
            reversed_steps.append((d, target_cell))
            cur = prev

        reversed_steps.reverse()
        return reversed_steps, best[goal_state]

    def fast_return_home(self):
        """
        Return from the final DFS cell to root=(0,0) using the fastest
        confirmed route currently known.

        No topology scans are repeated.  Real-time IR/Sharp/ToF/yaw safety
        remains active.  If a saved shortcut becomes blocked, that edge is
        marked blocked and the route is replanned from the current cell.
        """
        if self.current == self.root:
            print("[RETURN HOME] already at root.")
            return

        print("\n================ FAST RETURN HOME ================")
        print(f"Current : {self.current}")
        print(f"Home    : {self.root}")
        print("Planner : confirmed-map Dijkstra (move + turn time)")
        print("Safety  : IR + Sharp + front ToF + yaw hold remain active")
        print("==================================================")
        self.set_gui_status("RETURNING TO START")

        replans = 0
        self.transient_runtime_blocked_edges.clear()

        while self.running and self.current != self.root:
            plan, estimated_sec = self.plan_fastest_return(
                self.current,
                self.heading,
                self.root,
            )

            route_cells = [self.current] + [target for _, target in plan]
            self.set_gui_status("RETURNING TO START", route=route_cells)

            print(
                f"[RETURN PLAN] moves={len(plan)} "
                f"estimated_action_time={estimated_sec:.1f}s"
            )
            print(f"[RETURN PLAN] {route_cells}")

            need_replan = False

            for d, target in plan:
                current = self.current

                print(
                    f"\n[RETURN] {current} -> {target} "
                    f"dir={DIR_NAMES[d]}"
                )

                self.turn_to_direction(d)

                # AFTER_TURN IR/Gimbal may decide that the mapped direction
                # is no longer safe.  Preserve the original safety priority:
                # reject the edge BEFORE spending time on a vision re-check.
                if self.ir_replan_requested:
                    self.ir_replan_requested = False
                    self.stop_chassis()

                    print(
                        f"[RETURN REPLAN] current sensors reject "
                        f"{current}->{target}; mark blocked"
                    )

                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                # The return heading provides a genuinely different viewpoint.
                # Re-check only cells that had a brief glimpse or a locked
                # target, then restore FRONT/-5 before motion.
                try:
                    if TARGET_FAST_RETURN_RESCAN_ENABLED:
                        self.rescan_targets_on_return(current, phase="fast_return")
                except Exception as e:
                    print(f"[TARGET RETURN WARN] cell={current}: {e}")
                    try:
                        self.gimbal_front_down(force=True)
                    except Exception:
                        pass

                ok = self.move_one_cell()

                if not ok:
                    self.stop_chassis()

                    if self.motion_replan_requested:
                        reason = self.motion_replan_reason
                        self.motion_replan_requested = False
                        self.motion_replan_reason = None

                        print(
                            f"[RETURN REPLAN] transient safety abort "
                            f"{current}->{target}: {reason}; "
                            "returned to source, do NOT permanently block edge"
                        )

                        # Avoid immediately selecting the exact same edge
                        # again during this return attempt, without persisting
                        # it as a permanent WALL in the learned map.
                        self.transient_runtime_blocked_edges.add(
                            self.edge_key(current, target)
                        )
                        need_replan = True
                        break

                    print(
                        f"[RETURN REPLAN] failed edge "
                        f"{current}->{target}; mark blocked"
                    )

                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                self.current = target

                print(f"[RETURN] arrived {self.current}")

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                if self.current == self.root:
                    break

            if self.current == self.root:
                break

            if not need_replan:
                raise RuntimeError(
                    "Fast-return plan ended before reaching home."
                )

            replans += 1

            if replans > FAST_RETURN_MAX_REPLANS:
                raise RuntimeError(
                    "Fast return exceeded replan limit; robot stopped."
                )

        self.stop_chassis()

        if self.current == self.root:
            try:
                if TARGET_FAST_RETURN_RESCAN_ENABLED:
                    self.rescan_targets_on_return(self.root, phase="fast_return")
            except Exception as e:
                print(f"[TARGET RETURN WARN] root={self.root}: {e}")

        self.gimbal_front_down()

        if self.current == self.root:
            print(
                f"\n[RETURN HOME OK] reached {self.root} "
                f"heading={DIR_NAMES[self.heading]}"
            )
            self.set_gui_status("AT START - RETURN COMPLETE", route=[self.root])

    def shortest_confirmed_path(self, start, goal):
        """
        BFS shortest path over the already explored/confirmed graph.

        This is intentionally different from plan_fastest_return(): here the
        operator asked for the shortest route in CELL COUNT back to an
        EXIT_CANDIDATE source cell.  Safety checks still run while executing
        the path.
        """
        start = tuple(start)
        goal = tuple(goal)

        if start == goal:
            return [start]

        if start not in self.visited:
            raise ValueError(f"Start cell not explored: {start}")

        if goal not in self.visited:
            raise ValueError(f"Goal cell not explored: {goal}")

        q = deque([start])
        came_from = {start: None}

        while q:
            cell = q.popleft()

            if cell == goal:
                break

            for _d, nb in self.confirmed_explored_neighbors(cell):
                if nb in came_from:
                    continue

                came_from[nb] = cell
                q.append(nb)

        if goal not in came_from:
            raise RuntimeError(
                f"No confirmed shortest path from {start} to {goal}."
            )

        path = []
        cur = goal

        while cur is not None:
            path.append(cur)
            cur = came_from[cur]

        path.reverse()
        return path

    def navigate_shortest_confirmed_to(self, goal):
        """
        Navigate to one already-explored cell using the shortest confirmed
        path by number of grid edges.

        If a live safety sensor rejects an old edge, replan from the current
        cell instead of forcing the saved topology.
        """
        goal = tuple(goal)

        if self.current == goal:
            print(f"[EXIT ROUTE] already at candidate source {goal}")
            return True

        print("\n================ EXIT ROUTE =====================")
        print(f"Current : {self.current}")
        print(f"Target  : {goal}")
        print("Planner : BFS shortest confirmed path (minimum cells)")
        print("Safety  : IR + Sharp + front ToF + yaw hold remain active")
        print("==================================================")

        replans = 0
        self.transient_runtime_blocked_edges.clear()

        while self.running and self.current != goal:
            try:
                path = self.shortest_confirmed_path(self.current, goal)
            except Exception as e:
                self.stop_chassis()
                print(
                    f"[EXIT ROUTE] no safe confirmed route from "
                    f"{self.current} to {goal}: {e}"
                )
                return False

            print(
                f"[EXIT ROUTE PLAN] moves={max(0, len(path) - 1)} "
                f"path={path}"
            )

            need_replan = False

            for target in path[1:]:
                current = self.current
                d = direction_between(current, target)

                print(
                    f"\n[EXIT ROUTE] {current} -> {target} "
                    f"dir={DIR_NAMES[d]}"
                )

                self.turn_to_direction(d)

                if self.ir_replan_requested:
                    self.ir_replan_requested = False
                    self.stop_chassis()
                    print(
                        f"[EXIT ROUTE REPLAN] sensors reject "
                        f"{current}->{target}; mark blocked"
                    )
                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                ok = self.move_one_cell()

                if not ok:
                    self.stop_chassis()

                    if self.motion_replan_requested:
                        reason = self.motion_replan_reason
                        self.motion_replan_requested = False
                        self.motion_replan_reason = None

                        print(
                            f"[EXIT ROUTE REPLAN] transient abort "
                            f"{current}->{target}: {reason}"
                        )

                        self.transient_runtime_blocked_edges.add(
                            self.edge_key(current, target)
                        )
                        need_replan = True
                        break

                    print(
                        f"[EXIT ROUTE REPLAN] failed edge "
                        f"{current}->{target}; mark blocked"
                    )
                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                self.current = target
                print(f"[EXIT ROUTE] arrived {self.current}")

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                if self.current == goal:
                    break

            if self.current == goal:
                break

            if not need_replan:
                return False

            replans += 1

            if replans > FAST_RETURN_MAX_REPLANS:
                print("[EXIT ROUTE] replan limit exceeded")
                return False

        self.stop_chassis()
        self.gimbal_front_down()
        return self.current == goal

    def choose_nearest_exit_candidate(self):
        """
        Choose the reachable deferred EXIT_CANDIDATE whose SOURCE cell is
        closest to the current robot position in confirmed grid-edge count.
        """
        if not self.exit_candidates:
            return None

        ranked = []

        # Do not let an old transient block from the home trip poison the
        # operator-requested route selection.
        self.transient_runtime_blocked_edges.clear()

        for (cell, d), rec in self.exit_candidates.items():
            try:
                path = self.shortest_confirmed_path(self.current, cell)
            except Exception as e:
                print(
                    f"[EXIT ROUTE] skip unreachable candidate "
                    f"{cell}->{DIR_NAMES[d]}: {e}"
                )
                continue

            ranked.append(
                (
                    len(path) - 1,
                    cell[1],
                    cell[0],
                    d,
                    tuple(cell),
                    rec,
                    path,
                )
            )

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[:4])
        moves, _y, _x, d, cell, rec, path = ranked[0]

        print("\n[EXIT SELECT] nearest deferred candidate")
        print(
            f"[EXIT SELECT] source={cell} dir={DIR_NAMES[d]} "
            f"shortest_moves={moves}"
        )
        print(f"[EXIT SELECT] route={path}")

        return cell, d, rec

    def prompt_after_return_home(self):
        """
        Ask the operator what to do only AFTER the robot is safely back at
        root.  Mission Control is preferred; terminal selection is the safe
        fallback if Tkinter/display is unavailable.
        """
        if self.current != self.root or not self.exit_candidates:
            self.operator_selected_exit_candidate = None
            return "finish"

        options = self.build_exit_candidate_options()
        reachable = [opt for opt in options if opt.get('reachable')]

        if not reachable:
            print('[EXIT DECISION] no reachable EXIT_CANDIDATE from START')
            self.operator_selected_exit_candidate = None
            return 'finish'

        self.set_gui_status('WAITING FOR EXIT SELECTION', route=[self.root])

        if self.mission_gui is not None and self.mission_gui.available:
            result = self.mission_gui.request_exit_decision(
                options,
                snapshot=self.build_gui_snapshot(),
            )

            if result is not None:
                action, candidate_key = result
                if action == 'continue' and candidate_key in self.exit_candidates:
                    self.operator_selected_exit_candidate = candidate_key
                    return 'continue'

                self.operator_selected_exit_candidate = None
                return 'finish'

        # ----------------------------------------------------
        # Terminal fallback: still allows choosing ANY candidate.
        # ----------------------------------------------------
        print("\n================ EXIT DECISION ==================")
        print(f"Deferred EXIT_CANDIDATES: {len(self.exit_candidates)}")
        print("Robot is back at START (0, 0).")
        print("  [0] FINISH MISSION")

        for index, opt in enumerate(reachable, start=1):
            print(
                f"  [{index}] {opt['id']} source={opt['cell']} "
                f"dir={opt['dir']} shortest={opt['moves']} moves "
                f"front={opt.get('front_mm')}mm"
            )
            print(f"      path={opt['path']}")

        print("==================================================")

        while self.running:
            try:
                raw = input(
                    f"Select 0=FINISH or 1-{len(reachable)}=EXIT [default 0]: "
                ).strip().lower()
            except EOFError:
                print("[EXIT DECISION] no interactive input -> FINISH")
                self.operator_selected_exit_candidate = None
                return "finish"

            if raw in ("", "0", "f", "finish", "end", "stop"):
                self.operator_selected_exit_candidate = None
                return "finish"

            try:
                index = int(raw)
            except ValueError:
                print("Please enter a candidate number or 0 to finish.")
                continue

            if 1 <= index <= len(reachable):
                selected = reachable[index - 1]
                self.operator_selected_exit_candidate = selected['key']
                print(
                    f"[EXIT SELECT] operator chose {selected['id']} "
                    f"{selected['cell']}->{selected['dir']}"
                )
                return "continue"

            print("Selection out of range.")

        self.operator_selected_exit_candidate = None
        return "finish"

    def cross_selected_exit_candidate(self, candidate_key=None):
        """
        Route to the operator-selected deferred candidate, cross that one edge
        with the normal collision/safety layers still active, and attach the
        new cell to the existing DFS graph so exploration can resume there.

        The exit classifier itself is bypassed only for this operator-approved
        edge; subsequent edges use normal EXIT_CANDIDATE protection again.
        """
        if candidate_key is None:
            candidate_key = self.operator_selected_exit_candidate

        if candidate_key is None:
            # Compatibility/safety fallback for non-GUI callers.
            selected = self.choose_nearest_exit_candidate()
            if selected is None:
                print("[EXIT CONTINUE] no reachable EXIT_CANDIDATE remains")
                return False
            source_cell, abs_dir, _rec = selected
            candidate_key = self.exit_candidate_key(source_cell, abs_dir)
        else:
            source_cell = tuple(candidate_key[0])
            abs_dir = int(candidate_key[1]) % 4
            candidate_key = self.exit_candidate_key(source_cell, abs_dir)
            _rec = self.exit_candidates.get(candidate_key)
            if _rec is None:
                print(
                    f"[EXIT CONTINUE] selected candidate no longer exists: "
                    f"{source_cell}->{DIR_NAMES[abs_dir]}"
                )
                return False

        try:
            preview_path = self.shortest_confirmed_path(self.current, source_cell)
        except Exception as e:
            print(f"[EXIT CONTINUE] selected candidate unreachable: {e}")
            return False

        self.set_gui_status(
            f"NAVIGATING TO SELECTED EXIT {source_cell}->{DIR_NAMES[abs_dir]}",
            route=preview_path,
        )

        if not self.navigate_shortest_confirmed_to(source_cell):
            print(
                f"[EXIT CONTINUE] could not reach candidate source "
                f"{source_cell}"
            )
            return False

        destination = neighbor(source_cell, abs_dir)
        key = self.exit_candidate_key(source_cell, abs_dir)

        print("\n================ CROSS APPROVED EXIT ============")
        print(f"Source      : {source_cell}")
        print(f"Direction   : {DIR_NAMES[abs_dir]}")
        print(f"Destination : {destination}")
        print("Exit guard  : bypassed for THIS edge only")
        print("Safety      : IR + Sharp + front ToF + yaw hold ACTIVE")
        print("==================================================")
        self.set_gui_status(
            f"CROSSING APPROVED EXIT {source_cell}->{DIR_NAMES[abs_dir]}",
            route=[source_cell],
        )

        self.turn_to_direction(abs_dir)

        if self.ir_replan_requested:
            self.ir_replan_requested = False
            self.stop_chassis()
            print(
                "[EXIT CONTINUE] live IR/Gimbal safety rejects the approved "
                "exit edge; not forcing motion"
            )
            return False

        # detect_exit=False is deliberate: the operator has explicitly
        # approved this single deferred edge. Collision safety remains active.
        ok = self.move_one_cell(
            source_cell=source_cell,
            abs_dir=abs_dir,
            detect_exit=False,
        )

        if not ok:
            self.stop_chassis()
            print("[EXIT CONTINUE] approved exit crossing failed safely")
            return False

        # Commit the approved connection only after a successful crossing.
        self.exit_candidates.pop(key, None)

        if abs_dir not in self.open_dirs.get(source_cell, []):
            self.open_dirs.setdefault(source_cell, []).append(abs_dir)

        if destination not in self.visited:
            self.parent[destination] = source_cell
            self.visited.add(destination)
        elif destination not in self.parent:
            self.parent[destination] = source_cell

        self.current = destination
        self.approved_exit_entry_cells.add(destination)
        self.map_complete = False

        # Force a fresh topology scan beyond the approved boundary.
        self.open_dirs.pop(destination, None)
        self.cell_scan_mm.pop(destination, None)
        self.dead_end_cells.discard(destination)

        print(
            f"[EXIT CONTINUE OK] crossed to {destination}; "
            "resuming DFS from the new side"
        )
        self.operator_selected_exit_candidate = None
        self.set_gui_status(
            f"EXIT CROSSED - RESUMING DFS AT {destination}",
            route=[destination],
        )

        if MAP_AUTOSAVE:
            self.save_map(final=False)

        return True

    # --------------------------------------------------------
    # MISSION CONTROL GUI SNAPSHOTS
    # --------------------------------------------------------

    def attach_mission_gui(self, gui):
        self.mission_gui = gui
        self.publish_gui_state()

    def set_gui_status(self, status, route=None):
        self.gui_status_text = str(status)
        if route is not None:
            self.gui_route_preview = [tuple(c) for c in route]
        self.publish_gui_state()

    def build_gui_snapshot(self):
        """Build an immutable GUI snapshot on the mission thread."""
        cells = sorted(self.mapped_cells(), key=lambda p: (p[1], p[0]))
        pos = self.state.get_position()

        exit_records = []
        for (cell, d), rec in sorted(
            self.exit_candidates.items(),
            key=lambda item: (
                item[0][0][1], item[0][0][0], item[0][1]
            ),
        ):
            exit_records.append({
                'cell': [int(cell[0]), int(cell[1])],
                'dir_index': int(d),
                'dir': DIR_NAMES[d],
                'front_mm': rec.get('front_mm'),
                'reason': rec.get('reason'),
            })

        return {
            'status': self.gui_status_text,
            'root': [int(self.root[0]), int(self.root[1])],
            'current': [int(self.current[0]), int(self.current[1])],
            'heading': int(self.heading) % 4,
            'position': None if pos is None else [float(v) for v in pos],
            'cells': [[int(c[0]), int(c[1])] for c in cells],
            'visited': [
                [int(c[0]), int(c[1])]
                for c in sorted(self.visited, key=lambda p: (p[1], p[0]))
            ],
            'dead_end_cells': [
                [int(c[0]), int(c[1])]
                for c in sorted(self.dead_end_cells, key=lambda p: (p[1], p[0]))
            ],
            'open_dirs': [
                {
                    'cell': [int(cell[0]), int(cell[1])],
                    'dirs': [int(d) for d in dirs],
                }
                for cell, dirs in sorted(
                    self.open_dirs.items(), key=lambda item: (item[0][1], item[0][0])
                )
            ],
            'blocked_edges': [
                [
                    [int(a[0]), int(a[1])],
                    [int(b[0]), int(b[1])],
                ]
                for a, b in sorted(self.blocked_edges)
            ],
            'exit_candidates': exit_records,
            'known_entrance_dir': int(self.known_entrance_dir),
            'known_maze_ingress_dir': int(self.known_maze_ingress_dir),
            'route_preview': [
                [int(c[0]), int(c[1])] for c in self.gui_route_preview
            ],
            'targets': [dict(rec) for rec in self.detected_targets],
            'target_glimpses': [dict(rec) for rec in self.target_glimpses],
            'target_status': self.target_status_text,
            'target_fire_policy': self.get_target_fire_policy(),
            'last_fire_event': self.last_fire_event,
            'map_complete': bool(self.map_complete),
        }

    def publish_gui_state(self):
        gui = self.mission_gui
        if gui is None or not gui.available:
            return False
        try:
            return gui.post_snapshot(self.build_gui_snapshot())
        except Exception as e:
            print(f'[GUI WARN] state publish failed: {e}')
            return False

    def build_exit_candidate_options(self):
        """
        Build all reachable deferred exits for operator selection.

        Candidates are ranked by confirmed shortest-path cell count from the
        robot's current location, but the GUI never auto-selects a winner for
        motion; the operator can choose any reachable E# entry.
        """
        options = []
        self.transient_runtime_blocked_edges.clear()

        raw = []
        for (cell, d), rec in self.exit_candidates.items():
            key = (tuple(cell), int(d) % 4)
            try:
                path = self.shortest_confirmed_path(self.current, cell)
                reachable = True
                moves = len(path) - 1
            except Exception as e:
                path = []
                reachable = False
                moves = 10 ** 9
                error = str(e)
            else:
                error = None

            raw.append((
                0 if reachable else 1,
                moves,
                cell[1],
                cell[0],
                d,
                key,
                rec,
                path,
                error,
            ))

        raw.sort(key=lambda item: item[:5])

        for index, item in enumerate(raw, start=1):
            _unreachable, moves, _y, _x, d, key, rec, path, error = item
            cell = key[0]
            reachable = error is None
            shown_moves = (len(path) - 1) if reachable else None
            options.append({
                'id': f'E{index}',
                'key': key,
                'cell': tuple(cell),
                'dir_index': int(d),
                'dir': DIR_NAMES[d],
                'reachable': reachable,
                'moves': shown_moves,
                'route_distance_m': (
                    float(shown_moves) * CELL_LENGTH_M if reachable else 0.0
                ),
                'path': [tuple(c) for c in path],
                'front_mm': rec.get('front_mm'),
                'reason': rec.get('reason'),
                'error': error,
            })

        return options

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

        print(
            f"Known maze ingress: {self.root} -> "
            f"{DIR_NAMES[self.known_maze_ingress_dir]}"
        )
        print(
            f"Known entrance/return: {self.root} -> "
            f"{DIR_NAMES[self.known_entrance_dir]}"
        )

        if self.entrance_corridor_profile:
            print(
                "  Entrance profile: "
                f"back={self.entrance_corridor_profile.get('back_center_mm')}mm "
                f"side_walls="
                f"{self.entrance_corridor_profile.get('corridor_side_walls_verified')}"
            )

        if self.exit_candidates:
            print("Deferred EXIT_CANDIDATES:")
            for (cell, d), rec in sorted(
                self.exit_candidates.items(),
                key=lambda item: (
                    item[0][0][1],
                    item[0][0][0],
                    item[0][1],
                ),
            ):
                print(
                    f"  {cell} -> {DIR_NAMES[d]} "
                    f"front={rec.get('front_mm')}mm "
                    f"reason={rec.get('reason')}"
                )

        print("=================================================")
        print(self.render_ascii_map())

    # --------------------------------------------------------
    # DFS
    # --------------------------------------------------------

    def run_dfs(self, resume=False):
        if not resume:
            self.visited = {self.root}
            self.parent = {self.root: None}
            self.current = self.root

            stack = [self.root]
        else:
            # Continue from the cell just beyond an operator-approved
            # EXIT_CANDIDATE without erasing the map learned so far.
            stack = [self.current]

        exploration_finished = False

        if not resume:
            self.set_gui_status("EXPLORING MAZE", route=[self.current])
            print("\n[DFS] START")
            print(f"[DFS] root={self.root}, heading={DIR_NAMES[self.heading]}")

            # Root is a launch/staging anchor. The robot is manually placed
            # facing INTO the maze, so FRONT is the known ingress even when the
            # staging area is completely open.
            self.capture_start_anchor_profile()

            # The physical return side is still profiled independently.
            self.capture_entrance_corridor_profile()
        else:
            self.set_gui_status("EXPLORING BEYOND APPROVED EXIT", route=[self.current])
            print("\n[DFS] RESUME THROUGH APPROVED EXIT")
            print(
                f"[DFS] resume_cell={self.current}, "
                f"heading={DIR_NAMES[self.heading]}"
            )

        while self.running and stack:
            cell = stack[-1]
            self.current = cell

            if cell not in self.open_dirs:
                scanned_dirs = self.scan_cell(cell)

                # Search targets only while safely stationary at a confirmed
                # DFS node. The target sweep restores gimbal FRONT/-5 before
                # any chassis motion, so topology/motion behavior stays intact.
                try:
                    self.scan_targets_at_cell(cell)
                except Exception as e:
                    # Vision must never destroy a valid maze run.
                    print(f"[TARGET SWEEP WARN] cell={cell}: {e}")
                    self.target_status_text = f"VISION WARN at {cell}: {e}"
                    try:
                        self.gimbal_front_down(force=True)
                    except Exception:
                        pass

                if cell == self.root:
                    scanned_dirs = self.apply_root_start_policy(
                        scanned_dirs
                    )

                self.open_dirs[cell] = scanned_dirs

                # Last-resort containment. If an edge-level detector missed
                # the boundary, do NOT let DFS choose another direction from
                # an open outside area.
                if cell in self.approved_exit_entry_cells:
                    outside_evidence = None
                    print(
                        f"[OPEN AREA TRAP] {cell} is the first cell beyond "
                        "an operator-approved EXIT_CANDIDATE -> allow scan"
                    )
                else:
                    outside_evidence = self.detect_open_area_trap(cell)

                if outside_evidence is not None:
                    self.rollback_open_area_cell(
                        cell,
                        stack,
                        outside_evidence,
                    )
                    continue

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

            # ------------------------------------------------
            # GLOBAL EXPLORATION-COMPLETE CHECK
            # ------------------------------------------------
            # If NO visited cell has an unvisited open neighbor anymore, the
            # maze has been fully explored.  Do NOT keep unwinding the DFS
            # parent stack just to get back to root.  Break here and use the
            # completed map to take the fastest confirmed route home.
            if not self.has_unvisited_frontier():
                exploration_finished = True

                print(
                    f"\n[DFS] ALL FRONTIERS COMPLETE at {cell}. "
                    "Exploration finished."
                )

                if cell != self.root and FAST_RETURN_HOME_AFTER_DFS:
                    print(
                        "[DFS] skip remaining DFS-stack backtracking; "
                        "FAST RETURN HOME will plan directly to (0, 0)."
                    )

                break

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

                # A dead-end escape naturally gives the camera the opposite
                # travel orientation.  Re-check only remembered target evidence
                # from this cell before leaving it.
                try:
                    if TARGET_DFS_BACKTRACK_RESCAN_ENABLED:
                        self.rescan_targets_on_return(cell, phase="dfs_backtrack")
                except Exception as e:
                    print(f"[TARGET REVERSE WARN] cell={cell}: {e}")
                    try:
                        self.gimbal_front_down(force=True)
                    except Exception:
                        pass

                # Known parent edge: no exit classification while escaping
                # a dead end. We physically traversed this edge on entry.
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

                # The physical entrance is known from startup and may look
                # exactly like an exit corridor. It is never an Explore
                # frontier and must never be traversed outward.
                if self.is_known_entrance_edge(cell, next_dir):
                    print(
                        f"[DFS] DEFER {cell}->{next_cell} "
                        f"dir={DIR_NAMES[next_dir]} as KNOWN_ENTRANCE"
                    )

                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if d != next_dir
                    ]

                    if MAP_AUTOSAVE:
                        self.save_map(final=False)

                    continue

                # Root->front is a known maze ingress based on placement.
                # It must be allowed even if the starting zone is wide open.
                root_ingress = self.is_known_maze_ingress_edge(
                    cell,
                    next_dir,
                )

                # Unknown-size boundary protection:
                # real exits and fake exits are deferred only after ingress.
                if (
                    not root_ingress
                    and self.exit_guard_before_explore_edge(
                        cell,
                        next_dir,
                    )
                ):
                    print(
                        f"[DFS] DEFER {cell}->{next_cell} "
                        f"dir={DIR_NAMES[next_dir]} as EXIT_CANDIDATE"
                    )

                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if d != next_dir
                    ]

                    if MAP_AUTOSAVE:
                        self.save_map(final=False)

                    continue

                ok = self.move_one_cell(
                    source_cell=cell,
                    abs_dir=next_dir,
                    detect_exit=True,
                )

                if not ok:
                    # Confirmed exit candidate was detected DURING motion and
                    # the robot has already retreated to this source cell.
                    if self.motion_exit_candidate_detected:
                        self.motion_exit_candidate_detected = False

                        print(
                            f"[DFS] EXIT_CANDIDATE caught during motion "
                            f"{cell}->{next_cell}; "
                            "do not create destination cell"
                        )

                        self.open_dirs[cell] = [
                            d for d in self.open_dirs[cell]
                            if d != next_dir
                        ]

                        if MAP_AUTOSAVE:
                            self.save_map(final=False)

                        continue

                    # A transient IR/corner event may have safely returned the
                    # robot to the SAME source cell.  In that case do not poison
                    # the topology by permanently blocking the edge.  Throw
                    # away this cell's scan and observe it again.
                    if self.motion_replan_requested:
                        reason = self.motion_replan_reason

                        print(
                            f"[DFS] transient motion abort at {cell}: "
                            f"{reason} -> RESCAN SAME CELL; edge is NOT blocked"
                        )

                        self.motion_replan_requested = False
                        self.motion_replan_reason = None
                        self.ir_replan_requested = False
                        self.open_dirs.pop(cell, None)

                        if MAP_AUTOSAVE:
                            self.save_map(final=False)

                        continue

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
                exploration_finished = True
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

            # Normal DFS backtracking is also a useful reverse viewpoint.
            # This catches a G# glimpse before the robot leaves the branch,
            # without doing a full vision sweep at every revisited cell.
            try:
                if TARGET_DFS_BACKTRACK_RESCAN_ENABLED:
                    self.rescan_targets_on_return(cell, phase="dfs_backtrack")
            except Exception as e:
                print(f"[TARGET REVERSE WARN] cell={cell}: {e}")
                try:
                    self.gimbal_front_down(force=True)
                except Exception:
                    pass

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

        # Ctrl+C / external stop must never label a partial map as complete.
        if not self.running:
            self.stop_chassis()

            if MAP_AUTOSAVE:
                self.save_map(final=False)

            print("[DFS] stopped before full exploration completed.")
            return

        # Defensive fallback: if the loop ended naturally, verify the global
        # graph really has no remaining frontier.
        if not exploration_finished:
            exploration_finished = not self.has_unvisited_frontier()

        if not exploration_finished:
            self.stop_chassis()
            raise RuntimeError(
                "DFS loop ended while an unexplored frontier still exists."
            )

        self.map_complete = True
        self.set_gui_status("CURRENT REGION FULLY EXPLORED")

        # Save the COMPLETED learned map before driving home.
        self.stop_chassis()
        self.gimbal_front_down()
        self.print_map_summary()
        self.save_map(final=True)

        # The robot can finish exploration at the far end of the last branch.
        # Use the completed graph to return directly instead of DFS-parent
        # backtracking all the way to root.
        if FAST_RETURN_HOME_AFTER_DFS and self.current != self.root:
            self.fast_return_home()

        elif self.current == self.root:
            print("[RETURN HOME] DFS completed at root; no return trip needed.")

        # ----------------------------------------------------
        # OPERATOR DECISION AFTER RETURNING HOME
        # ----------------------------------------------------
        # Deferred exits are intentionally not crossed during the first pass.
        # Once the robot is safely back at START, the operator can finish the
        # mission or explicitly choose which deferred exit to continue through.
        if self.running and self.current == self.root and self.exit_candidates:
            decision = self.prompt_after_return_home()

            if decision == "continue":
                self.map_complete = False

                if self.cross_selected_exit_candidate(
                    self.operator_selected_exit_candidate
                ):
                    # Preserve all learned topology and continue DFS from the
                    # newly reached cell beyond the approved candidate.
                    self.run_dfs(resume=True)
                    return

                print(
                    "[EXIT CONTINUE] unable to enter a candidate safely; "
                    "mission ends at START."
                )
            else:
                print("[MISSION] operator selected FINISH at START.")
                self.set_gui_status("MISSION FINISHED AT START", route=[self.root])


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

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help=(
            "Disable Mission Control GUI and use terminal-only exit selection. "
            "GUI is enabled by default."
        ),
    )

    parser.add_argument(
        "--no-target-vision",
        action="store_true",
        help=(
            "Disable camera target detection/sweep. Maze exploration still runs."
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help=(
            "Keep target detection active but do not open the OpenCV preview."
        ),
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    explorer = DFSMazeExplorer()
    explorer.target_vision_enabled = not args.no_target_vision
    explorer.preview_enabled = not args.no_preview
    mission_gui = None

    # GUI starts before robot connection so the operator sees connection/
    # initialization state as part of the same mission dashboard.
    if not args.no_gui:
        mission_gui = MissionControlGUI(explorer)
        if mission_gui.start():
            explorer.attach_mission_gui(mission_gui)
            explorer.set_gui_status('CONNECTING TO ROBOMASTER')
        else:
            mission_gui = None

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
            explorer.set_gui_status('KNOWN MAP LOADED')

        explorer.connect()

        # The robot is connected and stationary here.  Do not enter DFS/known
        # navigation until the operator has explicitly armed a target-fire rule.
        if mission_gui is not None and mission_gui.available:
            explorer.set_gui_status('WAITING FOR TARGET FIRE RULES - ROBOT STATIONARY')
            fire_policy = mission_gui.request_target_fire_policy()
            if fire_policy is None:
                raise RuntimeError(
                    'Mission Control closed before target fire policy was armed'
                )
            explorer.set_target_fire_policy(fire_policy)
            explorer.set_gui_status('TARGET FIRE RULES ARMED - STARTING MISSION')
        else:
            # Terminal/no-GUI runs fail safe: mapping/navigation may continue,
            # but the physical blaster remains disarmed.
            explorer.set_target_fire_policy({
                'armed': True,
                'mode': 'selected',
                'fire_type': 'infrared',
                'auto_fire': False,
                'selected_color_shapes': [],
                'sdk_enabled': False,
                'sdk_labels': [],
            })
            print('[FIRE POLICY] --no-gui => physical firing DISABLED')

        if mode == "explore":
            explorer.run_dfs()
        else:
            explorer.set_gui_status('RUNNING KNOWN-MAP NAVIGATION')
            explorer.run_known_map(goal=args.goal)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        explorer.running = False
        explorer.set_gui_status('STOPPED BY CTRL+C')

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        explorer.set_gui_status(f'ERROR: {type(e).__name__}: {e}')

    finally:
        explorer.cleanup()

        if mission_gui is not None and mission_gui.available:
            # Keep the final map visible until the operator closes the window.
            try:
                mission_gui.post_mission_complete(explorer.build_gui_snapshot())
                print('[GUI] Mission complete. Close Mission Control to exit.')
                mission_gui.wait_closed()
            except KeyboardInterrupt:
                mission_gui.close()



if __name__ == "__main__":
    main()
