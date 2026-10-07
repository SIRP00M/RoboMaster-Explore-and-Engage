"""Shared tuning values, arena profiles and atomic output writing."""

import json
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

# Arena geometry is intentionally profile-driven instead of encoding any
# particular exit coordinate in DFS logic.  The same engine supports arbitrary
# rectangular arenas, translated logical origins, and arbitrary start cells.
#
# The built-in profile is only the default for today's field; it can be replaced
# by --arena-config or individual CLI overrides without changing mapping code.
DEFAULT_ARENA_PROFILE = {
    "name": "competition_default",
    "width_cells": 6,
    "height_cells": 6,
    "x_min": 0,
    "y_min": 0,
    "start_cell": [1, 0],
    "boundary_guard": True,
}

ARENA_PROFILE_NAME = str(DEFAULT_ARENA_PROFILE["name"])
GRID_WIDTH_CELLS = int(DEFAULT_ARENA_PROFILE["width_cells"])
GRID_HEIGHT_CELLS = int(DEFAULT_ARENA_PROFILE["height_cells"])
GRID_X_MIN = int(DEFAULT_ARENA_PROFILE["x_min"])
GRID_Y_MIN = int(DEFAULT_ARENA_PROFILE["y_min"])
GRID_X_MAX = GRID_X_MIN + GRID_WIDTH_CELLS - 1
GRID_Y_MAX = GRID_Y_MIN + GRID_HEIGHT_CELLS - 1
ROOT_CELL = tuple(int(v) for v in DEFAULT_ARENA_PROFILE["start_cell"][:2])
FIELD_BOUNDARY_GUARD_ENABLED = bool(DEFAULT_ARENA_PROFILE["boundary_guard"])


def apply_arena_profile(profile, source="runtime"):
    """Apply a rectangular arena description to the generic DFS engine.

    Supported keys: width_cells, height_cells, x_min, y_min, start_cell,
    boundary_guard and optional name.  No fake-exit coordinate is accepted or
    needed: perimeter rejection follows only from the configured rectangle.
    """
    global ARENA_PROFILE_NAME
    global GRID_WIDTH_CELLS, GRID_HEIGHT_CELLS
    global GRID_X_MIN, GRID_Y_MIN, GRID_X_MAX, GRID_Y_MAX
    global ROOT_CELL, FIELD_BOUNDARY_GUARD_ENABLED, MAX_VISITED_CELLS

    p = dict(DEFAULT_ARENA_PROFILE)
    if isinstance(profile, dict):
        for key in (
            "name", "width_cells", "height_cells", "x_min", "y_min",
            "start_cell", "boundary_guard",
        ):
            if key in profile and profile[key] is not None:
                p[key] = profile[key]

    width = int(p["width_cells"])
    height = int(p["height_cells"])
    x_min = int(p["x_min"])
    y_min = int(p["y_min"])
    if width <= 0 or height <= 0:
        raise ValueError("arena width/height must be positive")

    start = p.get("start_cell")
    if not isinstance(start, (list, tuple)) or len(start) < 2:
        raise ValueError("arena start_cell must contain x,y")
    start = (int(start[0]), int(start[1]))
    x_max = x_min + width - 1
    y_max = y_min + height - 1
    if not (x_min <= start[0] <= x_max and y_min <= start[1] <= y_max):
        raise ValueError(
            "arena start_cell {} is outside x={}..{}, y={}..{}".format(
                start, x_min, x_max, y_min, y_max
            )
        )

    ARENA_PROFILE_NAME = str(p.get("name") or "arena")
    GRID_WIDTH_CELLS = width
    GRID_HEIGHT_CELLS = height
    GRID_X_MIN = x_min
    GRID_Y_MIN = y_min
    GRID_X_MAX = x_max
    GRID_Y_MAX = y_max
    ROOT_CELL = start
    FIELD_BOUNDARY_GUARD_ENABLED = bool(p.get("boundary_guard", True))
    # The runaway guard follows the configured field size, not one fixed maze.
    try:
        MAX_VISITED_CELLS = width * height
    except Exception:
        pass

    print(
        "[ARENA] profile={} source={} size={}x{} bounds=x{}..{} y{}..{} start={} boundary_guard={}".format(
            ARENA_PROFILE_NAME, source, width, height, x_min, x_max, y_min, y_max,
            ROOT_CELL, FIELD_BOUNDARY_GUARD_ENABLED,
        )
    )


def load_arena_profile(path=None, overrides=None):
    """Load an arena profile from JSON and/or CLI-style overrides."""
    profile = dict(DEFAULT_ARENA_PROFILE)
    source = "built-in default"
    if path:
        pth = Path(path)
        payload = json.loads(pth.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("arena config must be a JSON object")
        profile.update(payload)
        source = str(pth)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if value is not None:
                profile[key] = value
        if any(v is not None for v in overrides.values()):
            source += "+CLI"
    apply_arena_profile(profile, source=source)
    return profile


def current_arena_profile():
    """Return the active arena geometry as a serializable profile."""
    return {
        "name": str(ARENA_PROFILE_NAME),
        "width_cells": int(GRID_WIDTH_CELLS),
        "height_cells": int(GRID_HEIGHT_CELLS),
        "x_min": int(GRID_X_MIN),
        "y_min": int(GRID_Y_MIN),
        "start_cell": [int(ROOT_CELL[0]), int(ROOT_CELL[1])],
        "boundary_guard": bool(FIELD_BOUNDARY_GUARD_ENABLED),
    }

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
DFS_EXPLORE_SPEED_MPS = 0.31  # tile/slip-safe
DFS_KNOWN_SPEED_MPS = 0.48  # tile/slip-safe
DFS_EXPLORE_APPROACH_MIN_MPS = 0.10
DFS_KNOWN_APPROACH_MIN_MPS = 0.13
DFS_EXPLORE_APPROACH_SLOW_M = 0.17
DFS_KNOWN_APPROACH_SLOW_M = 0.19

# Compatibility aliases.  Any legacy helper that still reads these constants
# gets the conservative exploration profile rather than the 0.35 m/s fast path.
FORWARD_SPEED_MPS = DFS_EXPLORE_SPEED_MPS
SLOW_FORWARD_SPEED_MPS = 0.12
CELL_APPROACH_SLOW_M = DFS_EXPLORE_APPROACH_SLOW_M
CELL_APPROACH_MIN_MPS = DFS_EXPLORE_APPROACH_MIN_MPS

# Front-ToF approach/brake policy.  Do not jump directly from forward motion to
# reverse when a wall appears.  Slow progressively, stop, then let the gimbal
# inspect LEFT/FRONT/RIGHT because the wall can be the end of a valid DFS cell.
FRONT_BRAKE_START_MM = 540.0  # brake earlier on smooth tile
FRONT_CRAWL_START_MM = 270.0
FRONT_STOP_SCAN_MM = 165.0
# Faster known-edge travel starts braking sooner.  STOP_SCAN stays unchanged: it
# is a physical clearance/sensing limit and must not become looser with speed.
FRONT_BRAKE_START_FAST_MM = 610.0
FRONT_CRAWL_START_FAST_MM = 290.0
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
# Smooth-tile traction guard: limit only acceleration into forward motion.
# Safety/braking commands are allowed to reduce speed immediately.
SLIPPERY_TILE_MODE = True
MOVE_FORWARD_ACCEL_LIMIT_MPS2 = 0.80
POST_MOVE_TILE_SETTLE_SEC = 0.08
DRIVE_COMMAND_TIMEOUT = 0.25
POSITION_STALE_SEC = 0.60
ATTITUDE_STALE_SEC = 0.40
MAX_CELL_TIME_SEC = max(6.0, (CELL_LENGTH_M / DFS_EXPLORE_SPEED_MPS) * 2.8)
# Bounded recovery for a cell that times out only because side-safety maneuvers
# consumed the motion watchdog while the front path is still sensor-confirmed OPEN.
MOVE_TIMEOUT_OPEN_RETRIES = 2
MOVE_TIMEOUT_OPEN_EXTENSION_SEC = 3.0

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
# If the IR-requested escape direction is physically blocked by the opposite
# Sharp sensor, suppress that same IR request longer. This prevents a ping-pong
# where RIGHT IR repeatedly asks for LEFT strafe while LEFT Sharp is already at
# 5-7 cm (and vice versa).
IR_DESTINATION_VETO_COOLDOWN_SEC = 1.20
IR_DESTINATION_VETO_CM = 8.5

# Competition sunlight hardening: the digital side IRs can false-trigger under
# strong ambient IR.  Treat a one-sided LOW as a secondary hint only.  A real
# side/corner hazard must also be corroborated by the Sharp sensor on the SAME
# side being close.  Front ToF + Sharp remain the primary collision sensors.
IR_SUNLIGHT_GUARD_ENABLED = True
IR_SUNLIGHT_CORROBORATE_MAX_CM = 10.0
IR_SUNLIGHT_IGNORE_COOLDOWN_SEC = 0.65

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
TURN_MAX_DPS = 75.0  # reduce mecanum turn overshoot on tile
TURN_FINE_MAX_DPS = 31.0
TURN_FINE_ZONE_DEG = 14.0
TURN_MIN_DPS = 6.5
TURN_FINE_MIN_DPS = 3.5
TURN_TOLERANCE_DEG = 0.90
TURN_SETTLE_SEC = 0.12
TURN_CONTROL_HZ = 40.0
TURN_I_LIMIT = 7.0
TURN_I_ZONE_DEG = 12.0
TURN_D_ALPHA = 0.72
TURN_TIMEOUT_90_SEC = 3.8
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
# FAKE-EXIT / WIDE-OPEN GUARD
# ============================================================
# Competition fields may contain a fake exit: once the robot crosses the opening,
# the next logical cell can look like a large open area rather than a 60-cm maze
# corridor.  Never finish the map just because an opening looks like an exit.
# First require a strict wide-open signature: L/F/R are OPEN, BACK is the
# physically-traversed edge, and both +/-45-degree diagonal rays are also long.
FAKE_EXIT_GUARD_ENABLED = False  # configured rectangular boundary guard is authoritative
FAKE_EXIT_DIAG_YAWS_DEG = (-45.0, +45.0)
FAKE_EXIT_DIAG_MIN_CENTER_MM = 900.0
FAKE_EXIT_REQUIRE_ALL_CARDINAL_OPEN = True


# ============================================================
# MAP / LIMIT GUARDS
# ============================================================
MAP_DIR = Path("maps")
MAP_LATEST_JSON = MAP_DIR / "latest_map.json"
MAP_LATEST_ASCII = MAP_DIR / "latest_map.txt"
MAP_LATEST_SVG = MAP_DIR / "latest_map.svg"
# Chronological motion history.  Unlike the topology graph, this preserves
# exactly WHICH known/open edge was physically traversed and in what order.
BREADCRUMB_LATEST_JSON = MAP_DIR / "latest_breadcrumb.json"
MAP_AUTOSAVE = True

# Last-resort runaway guard.  It does not define maze geometry; it only prevents
# a bad sensor from producing an unbounded graph forever.
MAX_VISITED_CELLS = GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS
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
TARGET_SIDE_SHIFT_SPEED_MPS = 0.185  # tile/slip-safe        # remaining >65 mm
TARGET_SIDE_SHIFT_MED_SPEED_MPS = 0.120    # remaining <=65 mm
TARGET_SIDE_SHIFT_SLOW_MPS = 0.065         # remaining <=32 mm
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
TARGET_SIDE_SHIFT_RETURN_SPEED_MPS = 0.29
TARGET_SIDE_SHIFT_RETURN_MED_SPEED_MPS = 0.17
TARGET_SIDE_SHIFT_RETURN_SLOW_MPS = 0.08
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
TARGET_DEADEND_FRONT_BACKSHIFT_SPEED_MPS = 0.21
TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_MPS = 0.12
TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_ZONE_M = 0.060
TARGET_DEADEND_FRONT_BACKSHIFT_TIMEOUT_SEC = 3.6
TARGET_DEADEND_FRONT_RETURN_SPEED_MPS = 0.25
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


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
