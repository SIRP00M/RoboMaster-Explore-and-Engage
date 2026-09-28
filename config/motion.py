#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Chassis motion, corridor centering, turning, and recovery configurations."""

# ------------------------------------------------------------
# IR HARD-SAFETY / CORNER-CLEARANCE RECOVERY
# ------------------------------------------------------------
IR_FILTER_SAMPLES = 3

# Sequential dual-side IR latch window
IR_DUAL_EVENT_WINDOW_SEC = 1.50

# Small recovery nudge
IR_RECOVERY_SLIDE_M = 0.025
IR_RECOVERY_SLIDE_SPEED_MPS = 0.08
IR_RECOVERY_SLIDE_TIMEOUT_SEC = 1.2
IR_RECOVERY_MAX_ATTEMPTS = 3
IR_RECOVERY_SETTLE_SEC = 0.10

# Destination-side Sharp interlock while sliding
IR_SLIDE_DEST_SHARP_STOP_CM = 10.0

# Post-slide ToF clearance requirement
IR_GIMBAL_SIDE_CLEAR_MM = 100.0

# ------------------------------------------------------------
# CORRIDOR CONTROL & SHARP CENTERING
# ------------------------------------------------------------
CENTER_TARGET_CM = 13.0                 # Nominal only / authority selection
SHARP_FOLLOW_NEAR_CM = 11.5             # Closer -> nudge AWAY from wall
SHARP_FOLLOW_FAR_CM = 15.0              # Farther -> nudge TOWARD wall
SHARP_FOLLOW_KP = 0.030                 # Proportional gain outside safe band
SHARP_FOLLOW_MIN_STRAFE_MPS = 0.025     # Minimum useful correction
MAX_CENTER_STRAFE_MPS = 0.065           # Maximum continuous strafe

# Authority arbitration
AUTHORITY_MIN_HOLD_SEC = 0.75
AUTHORITY_DANGER_CM = 10.0
AUTHORITY_HARD_CM = 7.0
AUTHORITY_FAR_RELEASE_CM = 21.0
AUTHORITY_SWITCH_MARGIN_CM = 1.0
AUTHORITY_HARD_STRAFE_MPS = 0.13

# Both-IR low behavior
IR_BOTH_ROUTE_OPEN_MM = 600.0
IR_BOTH_FRONT_OVERRIDE_SEC = 1.00

# ------------------------------------------------------------
# SPEED & CELL APPROACH
# ------------------------------------------------------------
FORWARD_SPEED_MPS = 0.16
SLOW_FORWARD_SPEED_MPS = 0.10

CELL_APPROACH_SLOW_M = 0.10
CELL_APPROACH_MIN_MPS = 0.07

# Collision prevention
FRONT_HARD_STOP_MM = 160
FRONT_SLOW_MM = 330

# ------------------------------------------------------------
# TRANSIENT MOTION-ABORT RECOVERY
# ------------------------------------------------------------
MOTION_ABORT_RETREAT_SPEED_MPS = 0.10
MOTION_ABORT_HOME_TOL_M = 0.055
MOTION_ABORT_PROGRESS_EPS_M = 0.020
MOTION_ABORT_TIMEOUT_SEC = 6.0

# ------------------------------------------------------------
# YAW HOLD / DRIFT CORRECTION
# ------------------------------------------------------------
YAW_HOLD_ENABLED = True
YAW_HOLD_KP = 1.8
YAW_HOLD_MAX_DPS = 22.0
YAW_HOLD_DEADBAND_DEG = 0.35

STATIONARY_YAW_HOLD_KP = 2.8
STATIONARY_YAW_HOLD_MAX_DPS = 28.0
STATIONARY_YAW_HOLD_HZ = 30.0
STATIONARY_SETTLE_SEC = 0.12

YAW_DRIVE_SIGN = 1.0

# ------------------------------------------------------------
# CLOSED-LOOP CHASSIS TURN
# ------------------------------------------------------------
TURN_KP = 1.10
TURN_MAX_DPS = 45.0
TURN_MIN_DPS = 8.0
TURN_TOLERANCE_DEG = 1.2
TURN_SETTLE_SEC = 0.18
TURN_CONTROL_HZ = 30.0
TURN_TIMEOUT_90_SEC = 5.0
TURN_TIMEOUT_180_SEC = 8.0
TURN_DEBUG_PERIOD_SEC = 0.20
