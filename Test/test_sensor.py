from robomaster import robot
import time
import threading

PORT = 1

# ==========================================
# Sensor Adapter
# ==========================================

IR_LEFT_ID = 1
SHARP_LEFT_ID = 2
SHARP_RIGHT_ID = 3
IR_RIGHT_ID = 4

# ==========================================
# ToF CAN Bus
# ToF #1 = distance_info[0]
# ==========================================

tof_distance_mm = None
tof_lock = threading.Lock()


def tof_callback(distance_info):
    global tof_distance_mm

    if distance_info and len(distance_info) > 0:
        with tof_lock:
            tof_distance_mm = distance_info[0]


def ir_state(raw):
    """
    IR Active LOW

    0 = WALL
    1 = OPEN
    """

    if raw == 0:
        return "WALL"

    return "OPEN"


# ==========================================
# MAIN
# ==========================================

ep_robot = robot.Robot()

try:

    print("==============================================")
    print(" RoboMaster Sensor Test")
    print("==============================================")
    print(" ID1 Port1 = IR")
    print(" ID2 Port1 = Sharp")
    print(" ID3 Port1 = Sharp")
    print(" ID4 Port1 = IR")
    print(" ToF #1    = CAN Bus")
    print("==============================================")

    print("\nConnecting to RoboMaster...")

    ep_robot.initialize(conn_type="ap")

    sensor_adapter = ep_robot.sensor_adaptor
    distance_sensor = ep_robot.sensor

    print("Connected!")

    # ==========================================
    # Subscribe ToF through CAN Bus
    # ==========================================

    result = distance_sensor.sub_distance(
        freq=10,
        callback=tof_callback
    )

    print(f"ToF subscription: {result}")
    print("Reading sensors...")
    print("Press Ctrl+C to stop\n")

    while True:

        # ==========================================
        # IR ID1
        # ==========================================

        try:
            ir_left_raw = sensor_adapter.get_io(
                id=IR_LEFT_ID,
                port=PORT
            )

            ir_left = ir_state(ir_left_raw)

        except Exception as e:
            ir_left_raw = -1
            ir_left = f"ERROR:{e}"

        # ==========================================
        # Sharp ID2
        # ==========================================

        try:
            sharp_left_adc = sensor_adapter.get_adc(
                id=SHARP_LEFT_ID,
                port=PORT
            )

            sharp_left_voltage = (
                sharp_left_adc / 1023.0
            ) * 3.3

        except Exception:
            sharp_left_adc = -1
            sharp_left_voltage = -1

        # ==========================================
        # Sharp ID3
        # ==========================================

        try:
            sharp_right_adc = sensor_adapter.get_adc(
                id=SHARP_RIGHT_ID,
                port=PORT
            )

            sharp_right_voltage = (
                sharp_right_adc / 1023.0
            ) * 3.3

        except Exception:
            sharp_right_adc = -1
            sharp_right_voltage = -1

        # ==========================================
        # IR ID4
        # ==========================================

        try:
            ir_right_raw = sensor_adapter.get_io(
                id=IR_RIGHT_ID,
                port=PORT
            )

            ir_right = ir_state(ir_right_raw)

        except Exception as e:
            ir_right_raw = -1
            ir_right = f"ERROR:{e}"

        # ==========================================
        # Read latest ToF value
        # ==========================================

        with tof_lock:
            tof_mm = tof_distance_mm

        if tof_mm is not None:
            tof_cm = tof_mm / 10.0
            tof_text = f"{tof_mm:4d} mm ({tof_cm:5.1f} cm)"
        else:
            tof_text = "NO DATA"

        # ==========================================
        # DISPLAY
        # ==========================================

        print(
            f"IR1: {ir_left:<4} ({ir_left_raw})"
            f" | Sharp2: ADC={sharp_left_adc:4d}"
            f" {sharp_left_voltage:.3f}V"
            f" | Sharp3: ADC={sharp_right_adc:4d}"
            f" {sharp_right_voltage:.3f}V"
            f" | IR4: {ir_right:<4} ({ir_right_raw})"
            f" | ToF1: {tof_text}"
        )

        time.sleep(0.2)


except KeyboardInterrupt:

    print("\nStopped.")


except Exception as e:

    print(f"\nERROR: {e}")


finally:

    print("Stopping ToF subscription...")

    try:
        ep_robot.sensor.unsub_distance()
    except Exception:
        pass

    print("Disconnecting RoboMaster...")

    ep_robot.close()

    print("RoboMaster disconnected.")