#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Interactive sensor test utility for RoboMaster EP."""

import sys
import time
import threading
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robomaster import robot
from config.hardware import (
    CONN_TYPE,
    SENSOR_PORT,
    IR_LEFT_ID,
    IR_RIGHT_ID,
    SHARP_LEFT_ID,
    SHARP_RIGHT_ID,
    TOF_INDEX,
)
from config.calibration import LEFT_CAL, RIGHT_CAL
from src.hardware.sensors import adc_to_cm

tof_distance_mm = None
tof_lock = threading.Lock()


def tof_callback(distance_info):
    global tof_distance_mm
    if distance_info and len(distance_info) > 0:
        with tof_lock:
            tof_distance_mm = distance_info[0]


def ir_state(raw):
    """IR Active LOW: 0 = WALL, 1 = OPEN."""
    return "WALL" if raw == 0 else "OPEN"


def main():
    print("==============================================")
    print(" RoboMaster Sensor Diagnostic")
    print("==============================================")
    print(f" Port {SENSOR_PORT}:")
    print(f"   ID {IR_LEFT_ID}   = IR Left")
    print(f"   ID {SHARP_LEFT_ID}   = Sharp Left")
    print(f"   ID {SHARP_RIGHT_ID}   = Sharp Right")
    print(f"   ID {IR_RIGHT_ID}   = IR Right")
    print("==============================================")

    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type=CONN_TYPE)
        sensor_adapter = ep_robot.sensor_adapter
        distance_sensor = ep_robot.sensor
        distance_sensor.sub_distance(freq=20, callback=tof_callback)

        print("[OK] Connected. Streaming live sensor readings (Ctrl+C to quit)...")
        time.sleep(1.0)

        while True:
            # Digital IR
            ir_l = sensor_adapter.get_io(id=IR_LEFT_ID, port=SENSOR_PORT)
            ir_r = sensor_adapter.get_io(id=IR_RIGHT_ID, port=SENSOR_PORT)

            # Analog Sharp ADC
            adc_l = sensor_adapter.get_adc(id=SHARP_LEFT_ID, port=SENSOR_PORT)
            adc_r = sensor_adapter.get_adc(id=SHARP_RIGHT_ID, port=SENSOR_PORT)

            cm_l = adc_to_cm(adc_l, LEFT_CAL)
            cm_r = adc_to_cm(adc_r, RIGHT_CAL)

            cm_l_str = f"{cm_l:5.1f}cm" if cm_l is not None else "  --- "
            cm_r_str = f"{cm_r:5.1f}cm" if cm_r is not None else "  --- "

            with tof_lock:
                tof_val = tof_distance_mm

            tof_str = f"{tof_val:4d}mm" if tof_val is not None else " ---"

            sys.stdout.write(
                f"\r[IR L: {ir_state(ir_l):4s}] "
                f"[Sharp L: {adc_l:4d} ({cm_l_str})]  |  "
                f"[ToF: {tof_str}]  |  "
                f"[Sharp R: {adc_r:4d} ({cm_r_str})] "
                f"[IR R: {ir_state(ir_r):4s}]"
            )
            sys.stdout.flush()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[STOP] Test finished.")
    finally:
        try:
            ep_robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
