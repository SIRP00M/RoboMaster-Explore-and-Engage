#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Maze grid geometry, boundary guard, start anchor, and map persistence configs."""

from pathlib import Path
from config.motion import FORWARD_SPEED_MPS

# ------------------------------------------------------------
# CELL / DFS GEOMETRY
# ------------------------------------------------------------
GRID_TILE_M = 0.60
GRID_TILE_MM = GRID_TILE_M * 1000.0

# Tuned odometry displacement for one logical cell move
CELL_LENGTH_M = 0.55
CELL_SUCCESS_FRACTION = 0.82

# ToF threshold at cell center for topology classification
TOF_OPEN_THRESHOLD_MM = 600
DEAD_END_THRESHOLD_MM = 100.0

MAX_CELL_TIME_SEC = max(5.0, (CELL_LENGTH_M / FORWARD_SPEED_MPS) * 2.5)

# ------------------------------------------------------------
# FAKE-EXIT / OPEN-BOUNDARY GUARD
# ------------------------------------------------------------
EXIT_GUARD_ENABLED = True
EXIT_CORRIDOR_WALL_MAX_CM = 22.0

EXIT_FAN_ANGLES_DEG = (-60.0, -45.0, -30.0, +30.0, +45.0, +60.0)
EXIT_FAN_OPEN_MM = GRID_TILE_MM * 1.50          # 900 mm
EXIT_FAN_MIN_OPEN_PER_SIDE = 2
EXIT_FAN_STRONG_OPEN_MM = GRID_TILE_MM * 2.50   # 1500 mm
EXIT_FAN_MIN_STRONG_TOTAL = 4
EXIT_FAN_MIN_STRONG_PER_SIDE = 1
EXIT_FRONT_MIN_SAFE_MM = 300.0

EXIT_WALL_END_PROBE_ENABLED = True
EXIT_WALL_END_ANGLES_LEFT = (-15.0, -10.0)
EXIT_WALL_END_ANGLES_RIGHT = (+10.0, +15.0)
EXIT_WALL_END_FRONT_ARM_MM = 850.0
EXIT_WALL_END_RATIO = 1.45
EXIT_WALL_END_MARGIN_MM = 220.0
EXIT_WALL_END_MIN_MEASURED_MM = 750.0
EXIT_WALL_END_MIN_VOTES_PER_SIDE = 1

# In-motion exit guard
EXIT_MOTION_GUARD_ENABLED = True
EXIT_MOTION_MIN_TRAVEL_M = 0.26
EXIT_MOTION_MAX_TRAVEL_M = 0.48
EXIT_MOTION_LOST_CONFIRM_COUNT = 4
EXIT_MOTION_CAUTION_START_M = 0.24
EXIT_MOTION_CAUTION_SPEED_MPS = 0.08

# ------------------------------------------------------------
# OPEN-AREA TRAP FALLBACK
# ------------------------------------------------------------
OPEN_AREA_TRAP_ENABLED = True
OPEN_AREA_TRAP_MIN_OPEN_FAN_RAYS = 4
OPEN_AREA_TRAP_LONG_MM = GRID_TILE_MM * 3.75
OPEN_AREA_TRAP_MIN_LONG_FAN_RAYS = 4
OPEN_AREA_TRAP_SIDE_WALL_MAX_CM = 22.0

# ------------------------------------------------------------
# ADAPTIVE START / STAGING ANCHOR & KNOWN ENTRANCE
# ------------------------------------------------------------
START_ANCHOR_ENABLED = True
START_MAZE_INGRESS_DIR = 0   # N relative to startup heading
START_FORCE_FRONT_OPEN = True
START_OPEN_MM = 600.0

ENTRANCE_CORRIDOR_PROFILE_ENABLED = True
ENTRANCE_PROFILE_ANGLES_DEG = (
    +150.0,
    +165.0,
    +180.0,
    +195.0,
    +210.0,
)
ENTRANCE_OPEN_VERIFY_MM = 600.0
ENTRANCE_CORRIDOR_SIDE_WALL_MAX_CM = 22.0
ROOT_BACK_IS_WALL = True

# ------------------------------------------------------------
# GIMBAL SCANNING
# ------------------------------------------------------------
TOF_SCAN_SAMPLES = 7
TOF_SCAN_INTERVAL_SEC = 0.050
GIMBAL_SETTLE_SEC = 0.040
GIMBAL_ACTION_TIMEOUT_SEC = 1.80
GIMBAL_ANGLE_TOL_DEG = 2.0
GIMBAL_PITCH_DEG = -5.0
GIMBAL_PITCH_SPEED = 120
GIMBAL_YAW_SPEED = 180
IR_FAST_FRONT_FIRST = True

# ------------------------------------------------------------
# PERSISTENT MAP STORAGE
# ------------------------------------------------------------
MAP_SCHEMA = "robomaster_dfs_grid_map"
MAP_SCHEMA_VERSION = 1
MAP_DIR = Path("maps")
MAP_LATEST_JSON = MAP_DIR / "latest_map.json"
MAP_LATEST_ASCII = MAP_DIR / "latest_map.txt"
MAP_LATEST_SVG = MAP_DIR / "latest_map.svg"
MAP_AUTOSAVE = True

# ------------------------------------------------------------
# FAST RETURN HOME
# ------------------------------------------------------------
FAST_RETURN_HOME_AFTER_DFS = True
FAST_RETURN_MAX_REPLANS = 8
FAST_RETURN_MOVE_EST_SEC = 4.0
FAST_RETURN_TURN_90_EST_SEC = 3.0
FAST_RETURN_TURN_180_EST_SEC = 5.0
