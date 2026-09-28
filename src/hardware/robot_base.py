#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hardware lifecycle, low-level robot telemetry and gimbal controls."""

import math
import threading
import time
from datetime import datetime
from collections import deque
from pathlib import Path

from robomaster import robot

from config import *
from src.core.geometry import wrap_deg, clamp, REL_LEFT, REL_FRONT, REL_RIGHT, REL_BACK, DIR_NAMES
from src.core.state import SharedState


class HardwareMixin:
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


    def stop_chassis(self):
        if self.chassis is not None:
            self.chassis.drive_speed(
                x=0.0,
                y=0.0,
                z=0.0,
                timeout=DRIVE_COMMAND_TIMEOUT
            )


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


