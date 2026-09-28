#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hardware and sensor wiring configuration for RoboMaster EP."""

# Robot connection mode: "ap" (Direct Wi-Fi AP) or "sta" (Router networking)
CONN_TYPE = "ap"

SENSOR_PORT = 1

# Digital IR obstacle sensors (ACTIVE LOW)
#   0 = WALL / obstacle detected
#   1 = clear
IR_LEFT_ID = 1
IR_RIGHT_ID = 4

# Analog Sharp GP2Y0A41SK0F distance sensors
SHARP_LEFT_ID = 2
SHARP_RIGHT_ID = 3

# CAN bus ToF distance sensor mounted on gimbal
TOF_INDEX = 0
TOF_FREQ_HZ = 20

# Telemetry subscription frequencies
POSITION_FREQ_HZ = 20
ATTITUDE_FREQ_HZ = 50
ESC_FREQ_HZ = 10
STATUS_FREQ_HZ = 5
GIMBAL_ANGLE_FREQ_HZ = 20

# Low-level control loop timing & safety timeouts
CONTROL_DT = 0.05
DRIVE_COMMAND_TIMEOUT = 0.25
POSITION_WAIT_TIMEOUT = 3.0
DEBUG_MOVE_PRINT_PERIOD_SEC = 0.25
