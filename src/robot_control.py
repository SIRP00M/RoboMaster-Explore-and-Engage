"""RoboMaster SDK lifecycle, PID, heading, gimbal and chassis motion."""

from . import config as cfg
import math
import statistics
import time
from datetime import datetime
from robomaster import robot
from .sensors import (
    angle_diff_deg,
    angular_spread_deg,
    circular_mean_deg,
    clamp,
    fmt_deg,
    neighbor,
    tof_is_open_from_center,
    tof_topology_center_mm,
    wrap_deg,
)


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


class RobotControlMixin:
    """Robot connection, yaw/gimbal control and cell motion."""

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
        self._set_drive_command_telemetry(0.0, 0.0, 0.0, "STOP")
        if self.chassis is None:
            return
        try:
            self.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=cfg.DRIVE_COMMAND_TIMEOUT)
        except Exception as exc:
            self.fault("STOP", f"drive_speed(0) failed: {type(exc).__name__}: {exc}", "no further command")

    def drive_speed_resilient(self, x=0.0, y=0.0, z=0.0, timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="DRIVE CMD"):
        """Send a chassis velocity command with bounded retries.

        A transient SDK/Wi-Fi exception is treated as a transport fault, not as
        proof that the robot pose is lost. The caller decides what to do only
        after all retries fail.
        """
        if self.chassis is None:
            self.fault(label, "chassis unavailable", "command not sent")
            return False
        for attempt in range(1, cfg.SDK_COMMAND_RETRIES + 1):
            try:
                self.chassis.drive_speed(
                    x=float(x), y=float(y), z=float(z), timeout=float(timeout)
                )
                self._set_drive_command_telemetry(x, y, z, label)
                label_u = str(label or "").upper()
                if "TARGET" in label_u:
                    self.active_motion_profile_name = "TARGET_SHIFT"
                elif "IR+SHARP" in label_u or "IR " in label_u:
                    self.active_motion_profile_name = "IR_RECOVERY"
                elif "SHARP" in label_u and "MOVE" not in label_u:
                    self.active_motion_profile_name = "SHARP_RECOVERY"
                elif "RETREAT" in label_u:
                    self.active_motion_profile_name = "RETREAT"
                elif "TURN" in label_u:
                    self.active_motion_profile_name = "TURN"
                elif "YAW ALIGN" in label_u:
                    self.active_motion_profile_name = "YAW_ALIGN"
                return True
            except Exception as exc:
                self.fault(
                    label,
                    f"attempt {attempt}/{cfg.SDK_COMMAND_RETRIES}: {type(exc).__name__}: {exc}",
                    "retry command" if attempt < cfg.SDK_COMMAND_RETRIES else "caller recovery",
                )
                self.safe_stop()
                if attempt < cfg.SDK_COMMAND_RETRIES:
                    time.sleep(cfg.SDK_RETRY_DELAY_SEC * attempt)
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
                self.chassis.sub_position(freq=cfg.POSITION_FREQ_HZ, callback=self.position_callback)
            elif name == "attitude" and self.chassis is not None:
                try:
                    self.chassis.unsub_attitude()
                except Exception:
                    pass
                self.chassis.sub_attitude(freq=cfg.ATTITUDE_FREQ_HZ, callback=self.attitude_callback)
            elif name == "tof" and self.distance_sensor is not None:
                try:
                    self.distance_sensor.unsub_distance()
                except Exception:
                    pass
                self.distance_sensor.sub_distance(freq=cfg.TOF_FREQ_HZ, callback=self.tof_callback)
            elif name == "gimbal" and self.gimbal is not None:
                try:
                    self.gimbal.unsub_angle()
                except Exception:
                    pass
                self.gimbal.sub_angle(freq=cfg.GIMBAL_ANGLE_FREQ_HZ, callback=self.gimbal_callback)
            elif name == "status" and self.chassis is not None:
                try:
                    self.chassis.unsub_status()
                except Exception:
                    pass
                self.chassis.sub_status(freq=cfg.STATUS_FREQ_HZ, callback=self.status_callback)
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

        for attempt in range(1, cfg.TELEMETRY_RECOVERY_ATTEMPTS + 1):
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

            deadline = time.monotonic() + cfg.TELEMETRY_RECOVERY_WAIT_SEC
            while self.running and time.monotonic() < deadline:
                if ready():
                    time.sleep(cfg.TELEMETRY_RECOVERY_SETTLE_SEC)
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
        for attempt in range(1, cfg.GIMBAL_SCAN_RETRIES + 2):
            if self.gimbal_front_down(force=True) and self.gimbal_front_safe_for_motion():
                return True
            # moveto may acknowledge while pitch remains around -15deg.  Use
            # feedback velocity control before resorting to physical recenter.
            if self.gimbal_velocity_recover(
                0.0,pitch_deg=cfg.GIMBAL_PITCH_DEG,timeout_sec=1.6,require_motion_safe=True
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

    def connect(self):
        print("============================================================")
        print(" RoboMaster DFS MAP-ONLY / RESILIENT RESET")
        print("============================================================")
        print(f" Grid       : {cfg.GRID_TILE_M:.2f} m")
        raw_open_equiv = max(0.0, cfg.TOF_OPEN_THRESHOLD_MM - cfg.TOF_FORWARD_FROM_CENTER_M*1000.0)
        print(f" ToF OPEN   : center > {cfg.TOF_OPEN_THRESHOLD_MM:.0f} mm (raw approx > {raw_open_equiv:.0f} mm)")
        print(f" ToF origin : +{cfg.TOF_FORWARD_FROM_CENTER_M*100.0:.1f} cm from robot centre; safety uses RAW ToF")
        print(f" Muzzle     : +{cfg.FIRE_MUZZLE_FORWARD_FROM_CENTER_M*100.0:.1f} cm; +{cfg.FIRE_MUZZLE_AHEAD_OF_TOF_M*100.0:.1f} cm ahead of ToF")
        print(f" Runtime    : RUN pose={cfg.ROOT_CELL}, FRONT=N | field={cfg.GRID_WIDTH_CELLS}x{cfg.GRID_HEIGHT_CELLS}")
        print(" Features   : DFS + map + ToF node-probe + resilient gimbal + Sharp + IR + runtime-zero yaw PID")
        print(" DFS speed  : explore={:.2f} m/s | known/backtrack={:.2f} m/s".format(cfg.DFS_EXPLORE_SPEED_MPS, cfg.DFS_KNOWN_SPEED_MPS))
        print(" Targets    : Lab-CLAHE + HSV/LAB + shape/temporal + Foam-board gate + CENTER -> ToF geometry -> selectable fire")
        print(" Fire mode  : {}".format(self.get_fire_mode()))
        print(" Pose rule  : side radar uses temporary lateral viewpoint shifts; always returns node anchor; no forward target attack")
        print("============================================================")

        init_ok = False
        for attempt in range(1, cfg.STARTUP_CONNECT_RETRIES + 1):
            try:
                print(f"[CONNECT] AP attempt {attempt}/{cfg.STARTUP_CONNECT_RETRIES}")
                self.ep_robot.initialize(conn_type=cfg.CONN_TYPE)
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
            ("ToF", lambda: self.distance_sensor.sub_distance(freq=cfg.TOF_FREQ_HZ, callback=self.tof_callback)),
            ("position", lambda: self.chassis.sub_position(freq=cfg.POSITION_FREQ_HZ, callback=self.position_callback)),
            ("attitude", lambda: self.chassis.sub_attitude(freq=cfg.ATTITUDE_FREQ_HZ, callback=self.attitude_callback)),
            ("gimbal", lambda: self.gimbal.sub_angle(freq=cfg.GIMBAL_ANGLE_FREQ_HZ, callback=self.gimbal_callback)),
            ("status", lambda: self.chassis.sub_status(freq=cfg.STATUS_FREQ_HZ, callback=self.status_callback)),
        ]
        for label, fn in subscriptions:
            try:
                result = fn()
                print(f"[SUB] {label}: {result}")
            except Exception as exc:
                self.fault("SUBSCRIBE", f"{label}: {type(exc).__name__}: {exc}", "telemetry wait will verify")

        self.safe_stop()

        yaw = self.wait_for_telemetry("chassis yaw", self.current_yaw, cfg.STARTUP_TELEMETRY_WAIT_SEC)
        pos = self.wait_for_telemetry("chassis position", self.current_position, cfg.STARTUP_TELEMETRY_WAIT_SEC)
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
        print(f"[READY] runtime yaw-zero raw={self.base_yaw_deg:+.2f} deg -> logical yaw=0.00 deg; root={self.root}; heading=N; field={cfg.GRID_WIDTH_CELLS}x{cfg.GRID_HEIGHT_CELLS}")
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
        for attempt in range(1, cfg.RUNTIME_YAW_ZERO_RETRIES + 1):
            samples = []
            deadline = time.monotonic() + max(1.0, cfg.RUNTIME_YAW_ZERO_SAMPLES * cfg.RUNTIME_YAW_ZERO_INTERVAL_SEC * 3.0)
            while self.running and len(samples) < cfg.RUNTIME_YAW_ZERO_SAMPLES and time.monotonic() < deadline:
                yaw = self.current_yaw()
                if yaw is not None and math.isfinite(yaw):
                    samples.append(float(yaw))
                time.sleep(cfg.RUNTIME_YAW_ZERO_INTERVAL_SEC)

            center = circular_mean_deg(samples)
            spread = angular_spread_deg(samples, center) if center is not None else float("inf")
            if center is not None and len(samples) >= max(5, cfg.RUNTIME_YAW_ZERO_SAMPLES // 2) and spread <= cfg.RUNTIME_YAW_ZERO_MAX_SPREAD_DEG:
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
        return cfg.YAW_DRIVE_SIGN * pid.step(err)

    def lateral_yaw_hold_command(self, target_yaw=None):
        """Stronger yaw hold for pure mecanum lateral recovery.

        IR/Sharp recovery can create much more reaction torque than straight
        motion.  Keep this controller independent from the normal straight PID
        so normal forward travel is not made twitchy.
        """
        err = self.yaw_error_deg(target_yaw)
        if err is None or abs(err) <= cfg.LATERAL_YAW_HOLD_DEADBAND_DEG:
            return 0.0
        z = cfg.YAW_DRIVE_SIGN * cfg.LATERAL_YAW_HOLD_KP * float(err)
        z = clamp(z, -cfg.LATERAL_YAW_HOLD_MAX_DPS, +cfg.LATERAL_YAW_HOLD_MAX_DPS)
        if 0.0 < abs(z) < cfg.LATERAL_YAW_HOLD_MIN_DPS:
            z = math.copysign(cfg.LATERAL_YAW_HOLD_MIN_DPS, z)
        return z

    def align_heading_stationary(self, target_yaw=None, timeout_sec=cfg.STATIONARY_ALIGN_TIMEOUT_SEC,
                                 tolerance_deg=cfg.STATIONARY_SETTLE_TOL_DEG, settle_sec=cfg.STATIONARY_SETTLE_SEC):
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
        dt = 1.0 / cfg.STATIONARY_HOLD_HZ

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
                    z = cfg.YAW_DRIVE_SIGN * self.pid_stationary.step(err)
                    min_dps = (
                        cfg.STATIONARY_YAW_FINE_MIN_DPS
                        if abs(err) <= cfg.STATIONARY_YAW_FINE_ZONE_DEG
                        else cfg.STATIONARY_YAW_MIN_DPS
                    )
                    if abs(z) < min_dps:
                        z = math.copysign(min_dps, z if abs(z) > 1e-9 else err)
                    if not self.drive_speed_resilient(
                        x=0.0, y=0.0, z=z, timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="YAW ALIGN CMD"
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

    def hold_heading_stationary(self, duration=cfg.STATIONARY_SETTLE_SEC):
        # Compatibility wrapper: duration remains a minimum settle request, but
        # the controller now verifies actual yaw instead of merely waiting.
        timeout = max(cfg.STATIONARY_ALIGN_TIMEOUT_SEC, float(duration) + 0.45)
        return self.align_heading_stationary(
            self.yaw_ref_deg, timeout_sec=timeout,
            tolerance_deg=cfg.STATIONARY_SETTLE_TOL_DEG,
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

        dt = 1.0 / cfg.TURN_CONTROL_HZ
        start = time.monotonic()
        settled_since = None
        ok = False
        last_yaw_t = time.monotonic()
        try:
            while self.running and time.monotonic() - start < float(timeout_sec):
                yaw = self.current_yaw()
                if yaw is None:
                    self.safe_stop()
                    if time.monotonic() - last_yaw_t > cfg.ATTITUDE_STALE_SEC * 2.0:
                        self.fault("TURN", "attitude telemetry stale", "stop turn and retry")
                        break
                    time.sleep(dt)
                    continue
                last_yaw_t = time.monotonic()

                err = wrap_deg(target_yaw - yaw)
                if abs(err) <= cfg.TURN_TOLERANCE_DEG:
                    self.safe_stop()
                    if settled_since is None:
                        settled_since = time.monotonic()
                    if time.monotonic() - settled_since >= cfg.TURN_SETTLE_SEC:
                        ok = True
                        break
                else:
                    settled_since = None
                    limit = cfg.TURN_FINE_MAX_DPS if abs(err) <= cfg.TURN_FINE_ZONE_DEG else cfg.TURN_MAX_DPS
                    z = cfg.YAW_DRIVE_SIGN * self.pid_turn.step(err, output_limit=limit)
                    min_dps = cfg.TURN_FINE_MIN_DPS if abs(err) <= 4.0 else cfg.TURN_MIN_DPS
                    if abs(z) < min_dps:
                        z = math.copysign(min_dps, z if abs(z) > 1e-9 else err)
                    if not self.drive_speed_resilient(
                        x=0.0, y=0.0, z=z, timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="TURN CMD"
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
                target_yaw, timeout_sec=cfg.STATIONARY_ALIGN_TIMEOUT_SEC,
                tolerance_deg=cfg.STATIONARY_SETTLE_TOL_DEG, settle_sec=cfg.STATIONARY_SETTLE_SEC
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
        if err <= cfg.CARDINAL_SNAP_TOL_DEG:
            self.yaw_ref_deg = target
            if self.align_heading_stationary(
                target, timeout_sec=cfg.STATIONARY_ALIGN_TIMEOUT_SEC,
                tolerance_deg=cfg.STATIONARY_SETTLE_TOL_DEG, settle_sec=cfg.STATIONARY_SETTLE_SEC
            ):
                self.heading = d
                final_err = self.yaw_error_deg(target)
                self.fault("POSE RECOVER", f"aligned to {cfg.DIR_NAMES[d]} residual={fmt_deg(final_err)}deg", "resume")
                return d
            self.fault("POSE RECOVER", f"nearest cardinal {cfg.DIR_NAMES[d]} found but PID alignment failed", "do not snap pose")
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
            if same_heading_err is None or abs(same_heading_err) > cfg.PRE_MOVE_ALIGN_TOL_DEG:
                if not self.align_heading_stationary(
                    target_yaw, timeout_sec=cfg.PRE_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.07
                ):
                    self.fault("TURN", f"already facing {cfg.DIR_NAMES[target_dir]} but yaw could not re-align", "retry/defer")
                    return False
            self.gimbal_front_down()
            if self.gimbal_front_safe_for_motion():
                return True
            if self.recover_gimbal_front():
                self.fault("TURN", "gimbal front recovered after transient failure", "resume")
                return True
            return False

        delta = (target_dir - self.heading) % 4
        timeout = cfg.TURN_TIMEOUT_180_SEC if delta == 2 else cfg.TURN_TIMEOUT_90_SEC

        for attempt in range(cfg.TURN_RECOVERY_ATTEMPTS + 1):
            scale = 1.0 if attempt == 0 else cfg.TURN_RECOVERY_TIMEOUT_SCALE
            print(f"[TURN] {cfg.DIR_NAMES[self.heading]} -> {cfg.DIR_NAMES[target_dir]} attempt={attempt+1}")
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
                print(f"[TURN OK] {cfg.DIR_NAMES[target_dir]} logical_yaw={fmt_deg(logical)}deg residual={fmt_deg(final_err)}deg")
                return True
            self.fault("TURN", f"failed to settle on {cfg.DIR_NAMES[target_dir]} attempt {attempt+1}", "retry")
            time.sleep(0.12)

        snapped = self.recover_to_nearest_cardinal()
        if snapped == target_dir:
            self.pid_straight.reset()
            return True
        self.safe_stop()
        return False

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
                abs(wrap_deg(float(y))) <= cfg.STARTUP_GIMBAL_ACCEPT_YAW_DEG
                and cfg.STARTUP_GIMBAL_ACCEPT_PITCH_MIN_DEG
                    <= float(p)
                    <= cfg.STARTUP_GIMBAL_ACCEPT_PITCH_MAX_DEG
            )
            return bool(ok), float(p), float(y)

        for attempt in range(1, int(cfg.STARTUP_GIMBAL_RECENTER_ATTEMPTS) + 1):
            action_ok = False
            try:
                action = self.gimbal.recenter(
                    pitch_speed=min(float(cfg.GIMBAL_PITCH_SPEED), 110.0),
                    yaw_speed=min(float(cfg.GIMBAL_YAW_SPEED), 160.0),
                )
                action_ok = self._wait_action(
                    action, cfg.GIMBAL_ACTION_TIMEOUT_SEC, "GIMBAL RECENTER"
                )
            except Exception as exc:
                self.fault(
                    "GIMBAL RECENTER",
                    f"attempt {attempt}: {type(exc).__name__}: {exc}",
                    "verify feedback / velocity fallback",
                )

            # The action object can report False even though the mechanism moved.
            # Give feedback a short independent settle window.
            deadline = time.monotonic() + cfg.STARTUP_GIMBAL_RECENTER_VERIFY_SEC
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
                f"{cfg.STARTUP_GIMBAL_RECENTER_ATTEMPTS} action_ok={action_ok} "
                f"raw_p={p} raw_y={y} -> closed-loop fallback"
            )

            # Do not repeat a failed SDK action for seconds.  Use the telemetry
            # controller that was already proven to move the real gimbal.
            if self.gimbal_velocity_recover(
                0.0,
                pitch_deg=cfg.GIMBAL_PITCH_DEG,
                timeout_sec=cfg.STARTUP_GIMBAL_FALLBACK_TIMEOUT_SEC,
                require_motion_safe=True,
            ):
                ok, p, y = feedback_safe()
                if ok or (
                    p is not None and y is not None
                    and abs(wrap_deg(float(y))) <= cfg.GIMBAL_MOTION_FRONT_YAW_TOL_DEG
                    and cfg.GIMBAL_MOTION_SAFE_PITCH_MIN_DEG
                        <= float(p)
                        <= cfg.GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
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
            # RoboMaster yaw feedback can cross the +/-180 seam.  All target/topology
            # logic in this program uses signed bearings, so keep yaw canonical.
            return p, wrap_deg(y)
        # IMPORTANT: subtracting two raw angles can produce values such as +246.9
        # even though the physical bearing is -113.1 deg.  The continuous target
        # sweep compares bearings arithmetically, so always wrap the relative yaw.
        return p - self.gimbal_zero_pitch_raw, wrap_deg(y - self.gimbal_zero_yaw_raw)

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
        yaw_ok = abs(wrap_deg(y - yaw_deg)) <= cfg.GIMBAL_YAW_TOL_DEG
        pitch_exact = abs(p - pitch_deg) <= cfg.GIMBAL_PITCH_EXACT_TOL_DEG
        pitch_safe = (
            cfg.GIMBAL_TOF_SAFE_PITCH_MIN_DEG
            <= p
            <= cfg.GIMBAL_TOF_SAFE_PITCH_MAX_DEG
        )
        return bool(yaw_ok and (pitch_exact or pitch_safe))

    def gimbal_front_safe_for_motion(self):
        """Front ToF only needs forward yaw + a safe pitch, not exact -5 deg."""
        p, y = self.current_gimbal_relative()
        if p is None or y is None:
            return False
        return bool(
            abs(wrap_deg(y)) <= cfg.GIMBAL_MOTION_FRONT_YAW_TOL_DEG
            and cfg.GIMBAL_MOTION_SAFE_PITCH_MIN_DEG <= p <= cfg.GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
        )

    def gimbal_velocity_recover(self, yaw_deg, pitch_deg=cfg.GIMBAL_PITCH_DEG, timeout_sec=None, require_motion_safe=False):
        """Closed-loop velocity fallback for a stuck/lying SDK gimbal action.

        This uses sub_angle feedback directly.  It is intentionally available to
        topology/motion code, not only the camera radar, because the field log
        showed moveto() could leave +/-90deg scans at the correct yaw but a large
        pitch offset, or occasionally leave yaw tens of degrees short.
        """
        if self.gimbal is None:
            return False
        yaw_deg=clamp(float(yaw_deg),cfg.GIMBAL_SOFT_YAW_MIN_DEG,cfg.GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg=clamp(float(pitch_deg),cfg.GIMBAL_SOFT_PITCH_MIN_DEG,cfg.GIMBAL_SOFT_PITCH_MAX_DEG)
        p0,y0=self.current_gimbal_relative()
        yaw_dist=90.0 if y0 is None else abs(wrap_deg(yaw_deg-float(y0)))
        pitch_dist=10.0 if p0 is None else abs(float(pitch_deg)-float(p0))
        if timeout_sec is None:
            timeout_sec=max(
                0.9,
                yaw_dist/max(30.0,cfg.GIMBAL_RECOVERY_YAW_MAX_DPS*0.75)+0.65,
                pitch_dist/max(10.0,cfg.GIMBAL_RECOVERY_PITCH_MAX_DPS*0.65)+0.55,
            )
        deadline=time.monotonic()+float(timeout_sec)
        stable=0
        try:
            while self.running and time.monotonic()<deadline:
                p,y=self.current_gimbal_relative()
                if p is None or y is None:
                    time.sleep(0.03); continue
                ey=wrap_deg(yaw_deg-float(y)); ep=float(pitch_deg)-float(p)
                yaw_ok=abs(ey)<=cfg.GIMBAL_RECOVERY_YAW_TOL_DEG
                pitch_ok=abs(ep)<=cfg.GIMBAL_RECOVERY_PITCH_TOL_DEG
                if require_motion_safe:
                    pitch_ok = pitch_ok or (
                        cfg.GIMBAL_MOTION_SAFE_PITCH_MIN_DEG <= float(p) <= cfg.GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
                    )
                if yaw_ok and pitch_ok:
                    stable+=1
                    try: self.gimbal.drive_speed(pitch_speed=0.0,yaw_speed=0.0)
                    except Exception: pass
                    if stable>=cfg.GIMBAL_RECOVERY_SETTLE_SAMPLES:
                        return True
                    time.sleep(0.035); continue
                stable=0
                ys=0.0 if yaw_ok else clamp(2.7*ey,-cfg.GIMBAL_RECOVERY_YAW_MAX_DPS,+cfg.GIMBAL_RECOVERY_YAW_MAX_DPS)
                ps=0.0 if pitch_ok else clamp(2.4*ep,-cfg.GIMBAL_RECOVERY_PITCH_MAX_DPS,+cfg.GIMBAL_RECOVERY_PITCH_MAX_DPS)
                if 0.0<abs(ys)<cfg.GIMBAL_RECOVERY_YAW_MIN_DPS:
                    ys=math.copysign(cfg.GIMBAL_RECOVERY_YAW_MIN_DPS,ys)
                if 0.0<abs(ps)<cfg.GIMBAL_RECOVERY_PITCH_MIN_DPS:
                    ps=math.copysign(cfg.GIMBAL_RECOVERY_PITCH_MIN_DPS,ps)
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
        yaw_ok=abs(wrap_deg(float(y)-yaw_deg))<=cfg.GIMBAL_RECOVERY_YAW_TOL_DEG
        if require_motion_safe:
            pitch_ok=cfg.GIMBAL_MOTION_SAFE_PITCH_MIN_DEG <= float(p) <= cfg.GIMBAL_MOTION_SAFE_PITCH_MAX_DEG
        else:
            pitch_ok=(abs(float(p)-pitch_deg)<=cfg.GIMBAL_RECOVERY_PITCH_TOL_DEG or
                      cfg.GIMBAL_TOF_SAFE_PITCH_MIN_DEG <= float(p) <= cfg.GIMBAL_TOF_SAFE_PITCH_MAX_DEG)
        return bool(yaw_ok and pitch_ok)

    def gimbal_goto(self, yaw_deg, pitch_deg=cfg.GIMBAL_PITCH_DEG, force=False):
        """Topology/motion gimbal positioning with one-shot SDK + fast fallback.

        Repeating an SDK moveto() that has already failed costs several seconds
        and did not improve the 2026-10-01 field behavior.  Try it once, verify
        live feedback, then switch immediately to closed-loop velocity recovery.
        """
        if self.gimbal is None:
            return False
        yaw_deg = clamp(float(yaw_deg), cfg.GIMBAL_SOFT_YAW_MIN_DEG, cfg.GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg = clamp(float(pitch_deg), cfg.GIMBAL_SOFT_PITCH_MIN_DEG, cfg.GIMBAL_SOFT_PITCH_MAX_DEG)
        if not force and self.gimbal_at_target(yaw_deg, pitch_deg):
            return True

        raw_p = pitch_deg + (self.gimbal_zero_pitch_raw or 0.0)
        raw_y = yaw_deg + (self.gimbal_zero_yaw_raw or 0.0)

        try:
            action = self.gimbal.moveto(
                pitch=raw_p, yaw=raw_y,
                pitch_speed=cfg.GIMBAL_PITCH_SPEED,
                yaw_speed=cfg.GIMBAL_YAW_SPEED,
            )
            self._wait_action(action, cfg.GIMBAL_ACTION_TIMEOUT_SEC, "GIMBAL MOVETO")
        except Exception as exc:
            self.fault("GIMBAL MOVETO", f"{type(exc).__name__}: {exc}", "velocity fallback")

        time.sleep(cfg.GIMBAL_SETTLE_SEC)
        if self.gimbal_at_target(yaw_deg, pitch_deg):
            p, y = self.current_gimbal_relative()
            if p is not None and abs(p - pitch_deg) > cfg.GIMBAL_PITCH_EXACT_TOL_DEG:
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
        return self.gimbal_goto(0.0, cfg.GIMBAL_PITCH_DEG, force=force)

    def sample_fresh_tof(self, samples=cfg.TOF_SCAN_SAMPLES, timeout=cfg.TOF_SCAN_TIMEOUT_SEC):
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
                        time.monotonic() - stamp <= cfg.FRONT_TOF_STALE_SEC
                        and value is not None
                        and math.isfinite(value)
                        and cfg.TOF_VALID_MIN_MM <= value <= cfg.TOF_VALID_MAX_MM
                    ):
                        values.append(float(value))
            time.sleep(cfg.TOF_SCAN_INTERVAL_SEC)

        if not values:
            return None
        return float(statistics.median(values))

    def scan_tof_at_yaw(self, yaw_deg):
        for attempt in range(1, cfg.GIMBAL_SCAN_RETRIES + 2):
            if not self.gimbal_goto(yaw_deg, cfg.GIMBAL_PITCH_DEG, force=(attempt > 1)):
                self.fault("SCAN", f"gimbal could not reach yaw={yaw_deg:+.1f}", "retry")
                continue
            mm = self.sample_fresh_tof()
            if mm is not None:
                return mm
            self.fault("SCAN", f"no fresh ToF at yaw={yaw_deg:+.1f} attempt {attempt}", "retry")
        return None

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

        if remaining <= cfg.LATERAL_ZERO_REMAIN_M or x_abs < 1e-6:
            target = 0.0
        else:
            if remaining < cfg.LATERAL_FADE_START_M:
                span = max(1e-6, cfg.LATERAL_FADE_START_M - cfg.LATERAL_ZERO_REMAIN_M)
                fade = clamp((remaining - cfg.LATERAL_ZERO_REMAIN_M) / span, 0.0, 1.0)
                ratio = cfg.LATERAL_MAX_RATIO_APPROACH
            else:
                fade = 1.0
                ratio = cfg.LATERAL_MAX_RATIO_CRUISE
            target = raw_y * fade
            cap = x_abs * ratio
            target = clamp(target, -cap, +cap)

        max_delta = cfg.LATERAL_SLEW_MPS2 * cfg.CONTROL_DT
        y = self._slew(float(previous_y), target, max_delta)
        # Do not let slew memory keep lateral motion alive inside the final
        # straight-stop zone.  Force it to zero there.
        if remaining <= cfg.LATERAL_ZERO_REMAIN_M:
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
            target_yaw, timeout_sec=cfg.PRE_MOVE_ALIGN_TIMEOUT_SEC,
            tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.10
        )
        self.pid_straight.reset()

        deadline = time.monotonic() + cfg.RETREAT_TIMEOUT_SEC
        ramp_start = time.monotonic()
        last_metric = None
        last_progress_t = time.monotonic()
        last_debug = 0.0

        while self.running and time.monotonic() < deadline:
            pos = self.current_position()
            if pos is None:
                self.safe_stop()
                time.sleep(cfg.CONTROL_DT)
                continue

            fwd = self.cell_forward_progress(start_pos, pos, target_yaw)
            lat = self.cell_lateral_offset(start_pos, pos, target_yaw)
            metric = math.hypot(fwd, lat)

            if abs(fwd) <= cfg.RETREAT_FORWARD_TOL_M and abs(lat) <= cfg.RETREAT_LATERAL_TOL_M:
                self.safe_stop()
                self.align_heading_stationary(
                    target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.10
                )
                print(f"[RETREAT OK] source fwd={fwd:+.3f}m lat={lat:+.3f}m")
                return True

            if last_metric is None or metric < last_metric - 0.008:
                last_progress_t = time.monotonic()
                last_metric = metric
            elif time.monotonic() - last_progress_t > 1.6:
                # Do not immediately abort: if forward is already nearly home,
                # spend the remaining time correcting only lateral displacement.
                if abs(fwd) > cfg.RETREAT_SOFT_FORWARD_TOL_M or abs(lat) > cfg.RETREAT_SOFT_LATERAL_TOL_M:
                    self.fault(
                        "RETREAT",
                        f"progress slow fwd={fwd:+.3f}m lat={lat:+.3f}m",
                        "continue low-speed pose closure",
                    )
                last_progress_t = time.monotonic()
                last_metric = metric

            # Forward return command.  Ramp in gently from zero so a front-stop
            # never turns into an instantaneous full-speed reverse command.
            if abs(fwd) <= cfg.RETREAT_FORWARD_TOL_M:
                x_cmd = 0.0
            else:
                desired = -math.copysign(cfg.RETREAT_SPEED_MPS, fwd)
                if abs(fwd) < cfg.RETREAT_NEAR_SOURCE_M:
                    desired = -math.copysign(
                        max(cfg.RETREAT_MIN_SPEED_MPS, cfg.RETREAT_SPEED_MPS * abs(fwd) / cfg.RETREAT_NEAR_SOURCE_M),
                        fwd,
                    )
                ramp = clamp((time.monotonic() - ramp_start) / max(1e-6, cfg.RETREAT_RAMP_SEC), 0.0, 1.0)
                x_cmd = desired * ramp

            # Close lateral drift independently. Positive lat means robot ended to
            # the RIGHT of the intended line, therefore command negative y.
            if abs(lat) <= cfg.RETREAT_LATERAL_TOL_M:
                y_cmd = 0.0
            else:
                y_cmd = clamp(-cfg.RETREAT_LATERAL_KP * lat, -cfg.RETREAT_LATERAL_MAX_MPS, +cfg.RETREAT_LATERAL_MAX_MPS)
                # Near the source give lateral closure priority over x.
                if abs(fwd) <= cfg.RETREAT_NEAR_SOURCE_M:
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
                x=x_cmd, y=y_cmd, z=z_cmd, timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="RETREAT CMD"
            ):
                break
            time.sleep(cfg.CONTROL_DT)

        self.safe_stop()
        pos = self.current_position(fresh=False)
        if pos is not None:
            fwd = self.cell_forward_progress(start_pos, pos, target_yaw)
            lat = self.cell_lateral_offset(start_pos, pos, target_yaw)
            if abs(fwd) <= cfg.RETREAT_SOFT_FORWARD_TOL_M and abs(lat) <= cfg.RETREAT_SOFT_LATERAL_TOL_M:
                self.align_heading_stationary(
                    target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                    tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08
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
        time.sleep(cfg.FRONT_NODE_SCAN_SETTLE_SEC)
        self.align_heading_stationary(
            target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
            tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08
        )
        print(
            f"[NODE PROBE] stopped at fwd={traveled:.3f}m front={tof_front:.0f}mm; "
            "Gimbal L/F/R scan"
        )
        result = {}
        valid = 0
        for label, yaw_deg, _rel in cfg.SCAN_RELATIVE_ORDER:
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
            valid >= cfg.FRONT_NODE_CAPTURE_MIN_VALID_RAYS
            and front_wall
            and traveled >= cfg.FRONT_NODE_CAPTURE_MIN_PROGRESS_M
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
            abs(fwd) <= cfg.MOVE_EXCEPTION_SOURCE_FWD_TOL_M
            and abs(lat) <= cfg.MOVE_EXCEPTION_SOURCE_LAT_TOL_M
        ):
            self.align_heading_stationary(
                target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08,
            )
            self.fault(
                "MOVE EXCEPTION RECOVER",
                f"proved source anchor fwd={fwd:+.3f}m lat={lat:+.3f}m",
                "retry/defer edge without ending mission",
            )
            return self.MOVE_BLOCKED_RETURNED

        if (
            abs(fwd - target_distance) <= cfg.MOVE_EXCEPTION_DEST_FWD_TOL_M
            and abs(lat) <= cfg.MOVE_EXCEPTION_DEST_LAT_TOL_M
        ):
            self.align_heading_stationary(
                target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08,
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
        cruise = cfg.DFS_KNOWN_SPEED_MPS if fast else cfg.DFS_EXPLORE_SPEED_MPS
        approach_min = cfg.DFS_KNOWN_APPROACH_MIN_MPS if fast else cfg.DFS_EXPLORE_APPROACH_MIN_MPS
        approach_zone = cfg.DFS_KNOWN_APPROACH_SLOW_M if fast else cfg.DFS_EXPLORE_APPROACH_SLOW_M
        brake_start = cfg.FRONT_BRAKE_START_FAST_MM if fast else cfg.FRONT_BRAKE_START_MM
        crawl_start = cfg.FRONT_CRAWL_START_FAST_MM if fast else cfg.FRONT_CRAWL_START_MM
        timeout = max(4.5, (cfg.CELL_LENGTH_M / max(0.05, cruise)) * 3.2)
        return {
            "name": name, "fast": fast, "cruise": float(cruise),
            "approach_min": float(approach_min), "approach_zone": float(approach_zone),
            "brake_start_mm": float(brake_start), "crawl_start_mm": float(crawl_start),
            "timeout_sec": float(timeout),
        }

    def move_one_cell(self, source_cell, abs_dir, motion_profile="EXPLORE"):
        """Public exception boundary for one physical edge traversal.

        Every confirmed MOVE_ARRIVED is committed to the chronological breadcrumb
        here, so EXPLORE, shortcut relocation, backtrack and Round-2 shortest travel
        all share the same trustworthy recorder.
        """
        abs_dir = int(abs_dir) % 4
        profile = self._motion_profile(motion_profile)
        start_pos = self.current_position(fresh=False)
        target_yaw = self.desired_yaw_for_heading(abs_dir)
        learned_distance = self.remembered_edge_distance(source_cell, abs_dir)
        target_distance = learned_distance if learned_distance is not None else cfg.CELL_LENGTH_M
        result = self.MOVE_POSE_UNCERTAIN
        try:
            result = self._move_one_cell_impl(source_cell, abs_dir, profile)
        except Exception as exc:
            self.safe_stop()
            detail = f"{type(exc).__name__}: {exc}"
            self.fault(
                "MOVE UNEXPECTED", detail,
                "prove source/destination pose, otherwise retreat",
            )
            try:
                result = self._recover_after_translation_exception(
                    start_pos, target_yaw, target_distance, detail
                )
            except Exception as recover_exc:
                self.safe_stop()
                self.fault(
                    "MOVE RECOVER",
                    f"{type(recover_exc).__name__}: {recover_exc}",
                    "pose cannot be proven",
                )
                result = self.MOVE_POSE_UNCERTAIN

        if result == self.MOVE_ARRIVED:
            try:
                self._record_breadcrumb(
                    source_cell, abs_dir, profile, distance_m=self.last_move_distance_m
                )
            except Exception as exc:
                # Breadcrumb is diagnostic/history data: never cancel motion because
                # the recorder itself had a software/file-shape problem.
                self.fault(
                    "BREADCRUMB", f"{type(exc).__name__}: {exc}",
                    "arrival remains valid; continue mission",
                )
        return result

    def _move_one_cell_impl(self, source_cell, abs_dir, profile=None):
        if not self.pose_trusted or not self.running:
            return self.MOVE_STOPPED

        profile = dict(profile or self._motion_profile("EXPLORE"))
        self.active_motion_profile_name = str(profile.get("name", "EXPLORE"))
        cruise_speed = float(profile.get("cruise", cfg.DFS_EXPLORE_SPEED_MPS))
        approach_min_speed = float(profile.get("approach_min", cfg.DFS_EXPLORE_APPROACH_MIN_MPS))
        approach_slow_m = float(profile.get("approach_zone", cfg.DFS_EXPLORE_APPROACH_SLOW_M))
        brake_start_mm = float(profile.get("brake_start_mm", cfg.FRONT_BRAKE_START_MM))
        crawl_start_mm = float(profile.get("crawl_start_mm", cfg.FRONT_CRAWL_START_MM))
        move_timeout_sec = float(profile.get("timeout_sec", cfg.MAX_CELL_TIME_SEC))
        print("[MOVE PROFILE] {} cruise={:.2f}m/s approach={:.2f}m brake={:.0f}/{:.0f}mm".format(
            profile.get("name", "EXPLORE"), cruise_speed, approach_slow_m,
            brake_start_mm, crawl_start_mm,
        ))

        abs_dir = int(abs_dir) % 4
        self.last_move_distance_m = None
        self.last_stop_probe = None
        learned_distance = self.remembered_edge_distance(source_cell, abs_dir)
        target_distance = learned_distance if learned_distance is not None else cfg.CELL_LENGTH_M
        expected_yaw = self.desired_yaw_for_heading(abs_dir)
        if expected_yaw is None:
            self.fault("MOVE PREP", "runtime yaw zero unavailable", "do not translate")
            return self.MOVE_BLOCKED_RETURNED

        # Never translate merely because the discrete heading variable says so.
        # Physically PID-align the chassis to the requested runtime cardinal first.
        self.yaw_ref_deg = expected_yaw
        prep_err = self.yaw_error_deg(expected_yaw)
        if prep_err is None or abs(prep_err) > cfg.PRE_MOVE_ALIGN_TOL_DEG:
            if not self.align_heading_stationary(
                expected_yaw, timeout_sec=cfg.PRE_MOVE_ALIGN_TIMEOUT_SEC,
                tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.07
            ):
                self.fault("MOVE PREP", f"yaw not aligned for {cfg.DIR_NAMES[abs_dir]}", "do not translate")
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
        for _ in range(cfg.SHARP_FILTER_SAMPLES):
            self.read_sharp_cm()
            time.sleep(0.015)

        start_t = time.monotonic()
        stale_tof_since = None
        yaw_bad_since = None
        last_debug = 0.0
        last_y_cmd = 0.0
        last_x_cmd = 0.0
        last_cmd_t = start_t
        side_escape_since = None
        side_escape_sign = 0.0
        side_escape_ignore_until = 0.0
        ir_retrigger_ignore_until = 0.0
        timeout_open_retries = 0

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
                        start_t += cfg.TELEMETRY_RECOVERY_WAIT_SEC
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
                    start_t += cfg.TELEMETRY_RECOVERY_WAIT_SEC
                    continue
                self.fault(
                    "MOVE HOLD", "attitude telemetry unavailable during translation",
                    "STOP in place; automatic reverse disabled",
                )
                return self.MOVE_POSE_UNCERTAIN
            if abs(yaw_err) >= cfg.MOVE_YAW_ABORT_ERROR_DEG:
                if yaw_bad_since is None:
                    yaw_bad_since = now
                elif now - yaw_bad_since >= cfg.MOVE_YAW_ABORT_CONFIRM_SEC:
                    self.safe_stop()
                    self.fault(
                        "MOVE YAW",
                        "diverged by {:+.1f}deg at d={:.3f}m".format(yaw_err, traveled),
                        "STOP/re-align; automatic reverse disabled",
                    )
                    if self.align_heading_stationary(
                        target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                        tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG, settle_sec=0.08,
                    ):
                        self.pid_straight.reset()
                        yaw_bad_since = None
                        continue
                    return self.MOVE_POSE_UNCERTAIN
            else:
                yaw_bad_since = None

            if traveled >= target_distance:
                self.safe_stop()
                if cfg.SLIPPERY_TILE_MODE:
                    time.sleep(cfg.POST_MOVE_TILE_SETTLE_SEC)
                # Remove the small yaw residual created by wheel inertia before the
                # next node scan.  Failure is logged but arrival position remains valid.
                post_err = self.yaw_error_deg(target_yaw)
                aligned = True
                if post_err is None or abs(post_err) > cfg.POST_MOVE_ALIGN_TOL_DEG:
                    aligned = self.align_heading_stationary(
                        target_yaw, timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                        tolerance_deg=cfg.POST_MOVE_ALIGN_TOL_DEG, settle_sec=0.05
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
                if traveled >= target_distance * cfg.CELL_SUCCESS_FRACTION:
                    self.align_heading_stationary(
                        target_yaw,
                        timeout_sec=cfg.POST_MOVE_ALIGN_TIMEOUT_SEC,
                        tolerance_deg=cfg.PRE_MOVE_ALIGN_TOL_DEG,
                        settle_sec=0.08,
                    )
                    self.last_move_distance_m = traveled
                    self.fault("MOVE TIMEOUT", f"near node at {traveled:.3f}m", "accept node")
                    return self.MOVE_ARRIVED

                tof_timeout = self.latest_tof(fresh=True)
                probe = None
                if tof_timeout is not None:
                    probe_t0 = time.monotonic()
                    probe = self.stopped_front_topology_probe(traveled, tof_timeout, target_yaw)
                    # Sensor probing is recovery time, not commanded forward travel.
                    start_t += max(0.0, time.monotonic() - probe_t0)

                # If the front ray is still OPEN and yaw/odometry are healthy, this
                # was most likely a watchdog timeout caused by IR/Sharp side recovery,
                # not a lost pose. Give a bounded extra forward window and suppress
                # the IR request that may have been fighting the Sharp controller.
                front_state = None
                if isinstance(probe, dict):
                    states = probe.get("states", {}) or {}
                    front_state = states.get("FRONT")

                if (
                    isinstance(probe, dict)
                    and probe.get("valid", 0) >= 2
                    and str(front_state).upper() == "OPEN"
                    and timeout_open_retries < cfg.MOVE_TIMEOUT_OPEN_RETRIES
                ):
                    timeout_open_retries += 1
                    side_escape_sign = 0.0
                    side_escape_since = None
                    last_y_cmd = 0.0
                    self.pid_straight.reset()
                    cooldown_until = time.monotonic() + cfg.IR_DESTINATION_VETO_COOLDOWN_SEC
                    ir_retrigger_ignore_until = max(ir_retrigger_ignore_until, cooldown_until)
                    side_escape_ignore_until = max(
                        side_escape_ignore_until,
                        time.monotonic() + cfg.SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC,
                    )
                    self.gimbal_front_down(force=True)
                    self.align_heading_stationary(
                        target_yaw,
                        timeout_sec=0.45,
                        tolerance_deg=max(0.85, cfg.PRE_MOVE_ALIGN_TOL_DEG),
                        settle_sec=0.04,
                    )
                    # Start a fresh bounded watchdog window without changing the
                    # odometry anchor; traveled still measures from the true source.
                    move_timeout_sec = max(
                        float(profile.get("timeout_sec", cfg.MAX_CELL_TIME_SEC)),
                        cfg.MOVE_TIMEOUT_OPEN_EXTENSION_SEC,
                    )
                    start_t = time.monotonic()
                    self.fault(
                        "MOVE TIMEOUT EXTEND",
                        "front OPEN at {:.3f}m; recovery {}/{}".format(
                            traveled, timeout_open_retries, cfg.MOVE_TIMEOUT_OPEN_RETRIES
                        ),
                        "suppress IR/Sharp ping-pong and continue same edge",
                    )
                    continue

                # A sensor-confirmed node after meaningful progress may still be
                # accepted using the existing conservative rule.
                if (
                    isinstance(probe, dict)
                    and traveled >= cfg.SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M
                    and probe.get("valid", 0) >= 2
                    and probe.get("node_like")
                ):
                    self.last_move_distance_m = max(cfg.SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
                    self.last_stop_probe = dict(probe.get("rays", {}))
                    self.fault(
                        "MOVE TIMEOUT HOLD",
                        "timeout at {:.3f}m".format(traveled),
                        "accept sensor-confirmed physical node",
                    )
                    return self.MOVE_ARRIVED

                # Last resort: prove the source again instead of immediately killing
                # the whole mission. A successful closed-loop retreat preserves the
                # DFS anchor, so the edge can be deferred and other branches explored.
                if self.retreat_to_source(
                    start_pos,
                    "cell timeout at {:.3f}m after bounded OPEN retries".format(traveled),
                    target_yaw=target_yaw,
                ):
                    self.fault(
                        "MOVE TIMEOUT RETURN",
                        "returned to source after timeout at {:.3f}m".format(traveled),
                        "defer edge; continue DFS",
                    )
                    return self.MOVE_BLOCKED_RETURNED

                self.fault(
                    "MOVE TIMEOUT HOLD",
                    "timeout at {:.3f}m; source retreat could not be proven".format(traveled),
                    "STOP in place; pose uncertain",
                )
                return self.MOVE_POSE_UNCERTAIN

            tof = self.latest_tof()
            if tof is None:
                if stale_tof_since is None:
                    stale_tof_since = now
                self.safe_stop()
                if now - stale_tof_since >= cfg.FRONT_TOF_STALE_SEC:
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
                    time.sleep(cfg.CONTROL_DT)
                    continue
            else:
                stale_tof_since = None

            if tof <= cfg.FRONT_STOP_SCAN_MM:
                # On a previously proven OPEN edge, reaching the expected node is
                # already supported by odometry + stored topology.  Do not spend
                # competition time rescanning L/F/R merely because a wall is close
                # beyond that known node.  An EARLY close obstacle still falls
                # through to the full stationary probe below.
                if bool(profile.get("fast")) and traveled >= target_distance * cfg.CELL_SUCCESS_FRACTION:
                    self.safe_stop()
                    self.last_move_distance_m = traveled
                    self.align_heading_stationary(
                        target_yaw, timeout_sec=0.70,
                        tolerance_deg=max(0.45, cfg.POST_MOVE_ALIGN_TOL_DEG), settle_sec=0.05,
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

                if traveled >= target_distance * cfg.CELL_SUCCESS_FRACTION:
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
                if traveled >= cfg.SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M and probe.get("valid", 0) >= 2:
                    self.last_move_distance_m = max(cfg.SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
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
                # Fast sunlight pre-gate BEFORE safe_stop: a lone digital IR LOW
                # does not interrupt forward motion unless same-side Sharp agrees
                # that the wall/corner is genuinely close.
                _sg_lcm, _sg_rcm, _sg_la, _sg_ra = self.read_sharp_cm()
                _sg_source = _sg_lcm if l_low else _sg_rcm
                if (
                    cfg.IR_SUNLIGHT_GUARD_ENABLED
                    and (_sg_source is None or _sg_source > cfg.IR_SUNLIGHT_CORROBORATE_MAX_CM)
                ):
                    ir_retrigger_ignore_until = (
                        time.monotonic() + cfg.IR_SUNLIGHT_IGNORE_COOLDOWN_SEC
                    )
                    print(
                        "[IR SUN GUARD MOVE] {} LOW raw=({},{}), source Sharp={}cm -> "
                        "ignore as uncorroborated/ambient IR".format(
                            "LEFT" if l_low else "RIGHT", l_raw, r_raw, _sg_source
                        )
                    )
                else:
                    self.safe_stop()
                    ir_t0 = time.monotonic()
                    ir_status = self.ir_simple_side_clear(target_yaw, context="MOVE")
                    start_t += max(0.0, time.monotonic() - ir_t0)
                    self.pid_straight.reset()
                    self.sharp_authority = None
                    if ir_status == "SUNLIGHT":
                        ir_retrigger_ignore_until = (
                            time.monotonic() + cfg.IR_SUNLIGHT_IGNORE_COOLDOWN_SEC
                        )
                    elif ir_status == "CLEAR":
                        # Prevent a sticky angled IR from immediately causing a second
                        # stop a few centimetres later.  Sharp/ToF remain live during
                        # this short cooldown.
                        ir_retrigger_ignore_until = time.monotonic() + cfg.IR_SIMPLE_RETRIGGER_COOLDOWN_SEC
                        continue
                    elif ir_status == "VETO":
                        # The requested IR strafe points into a close opposite wall.
                        # Ignore this IR direction briefly and let the Sharp controller
                        # below perform the physically safe escape instead.
                        ir_retrigger_ignore_until = (
                            time.monotonic() + cfg.IR_DESTINATION_VETO_COOLDOWN_SEC
                        )
                        print(
                            "[IR SIMPLE MOVE] destination Sharp veto -> suppress IR "
                            "retrigger; Sharp/ToF own recovery"
                        )
                    else:
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
                and left_cm <= cfg.SHARP_EMERGENCY_CM and right_cm <= cfg.SHARP_EMERGENCY_CM
            ):
                self.safe_stop()
                probe = self.stopped_front_topology_probe(traveled, tof, target_yaw)
                if traveled >= target_distance * cfg.CELL_SUCCESS_FRACTION or probe.get("node_like"):
                    self.last_move_distance_m = max(cfg.SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
                    return self.MOVE_ARRIVED
                self.fault(
                    "SHARP HOLD",
                    "both Sharp at emergency floor at d={:.3f}m".format(traveled),
                    "STOP/PROBE; no full retreat",
                )
                side_escape_ignore_until = time.monotonic() + cfg.SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC
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
                    tof_cmd = cfg.FRONT_MIN_BRAKE_SPEED_MPS
                else:
                    span = max(1.0, brake_start_mm - crawl_start_mm)
                    alpha = clamp((tof - crawl_start_mm) / span, 0.0, 1.0)
                    tof_cmd = cfg.FRONT_MIN_BRAKE_SPEED_MPS + alpha * (cruise_speed - cfg.FRONT_MIN_BRAKE_SPEED_MPS)
                x_cmd = min(x_cmd, tof_cmd)

            # A genuinely close side wall is handled as a pure lateral escape.
            # Never mix x forward with a large y correction; that was the main
            # source of the visible diagonal motion immediately before stopping.
            # Hysteretic side-escape state: once armed at <= trigger, remain in
            # pure-strafe mode until that SAME side reaches the clear threshold.
            escape_dist = None
            if side_escape_sign == 0.0 and now >= side_escape_ignore_until:
                if left_cm is not None and left_cm <= cfg.SHARP_SIDE_ESCAPE_TRIGGER_CM:
                    side_escape_sign, escape_dist = +1.0, left_cm
                if right_cm is not None and right_cm <= cfg.SHARP_SIDE_ESCAPE_TRIGGER_CM:
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
                if watched is None or watched >= cfg.SHARP_SIDE_ESCAPE_CLEAR_CM:
                    if side_escape_since is not None:
                        start_t += max(0.0, now - side_escape_since)
                    side_escape_sign = 0.0
                    side_escape_since = None
                    last_y_cmd = 0.0

            if side_escape_sign != 0.0:
                destination = right_cm if side_escape_sign > 0.0 else left_cm
                destination_tight = bool(
                    destination is not None
                    and destination <= cfg.SHARP_SIDE_ESCAPE_TRIGGER_CM
                )
                escape_timeout = bool(
                    side_escape_since is not None
                    and now - side_escape_since > cfg.SHARP_SIDE_ESCAPE_MAX_SEC
                )
                if destination_tight or escape_timeout:
                    self.safe_stop()
                    if side_escape_since is not None:
                        start_t += max(0.0, now - side_escape_since)
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
                        and source_now >= cfg.SHARP_SIDE_ESCAPE_FAST_RELEASE_CM
                        and tof is not None
                        and tof > max(cfg.FRONT_BRAKE_START_MM, cfg.TOF_OPEN_THRESHOLD_MM)
                    )
                    if fast_release:
                        side_escape_sign = 0.0
                        side_escape_since = None
                        side_escape_ignore_until = time.monotonic() + cfg.SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC
                        last_y_cmd = 0.0
                        yaw_after_escape = self.yaw_error_deg(target_yaw)
                        if (
                            yaw_after_escape is None
                            or abs(yaw_after_escape) > cfg.LATERAL_RECOVERY_REALIGN_TRIGGER_DEG
                        ):
                            self.align_heading_stationary(
                                target_yaw, timeout_sec=0.40,
                                tolerance_deg=max(0.75, cfg.PRE_MOVE_ALIGN_TOL_DEG),
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
                    if traveled >= target_distance * cfg.CELL_SUCCESS_FRACTION or probe.get("node_like"):
                        self.last_move_distance_m = max(cfg.SENSOR_NO_RETREAT_MIN_NODE_PROGRESS_M, traveled)
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
                    side_escape_ignore_until = time.monotonic() + cfg.SHARP_SIDE_ESCAPE_RETRY_COOLDOWN_SEC
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
                y_cmd = side_escape_sign * cfg.SHARP_SIDE_ESCAPE_SPEED_MPS
                last_y_cmd = y_cmd
            else:
                y_cmd = self.shape_lateral_command(raw_y_cmd, x_cmd, remaining, last_y_cmd)
                last_y_cmd = y_cmd

            # Smooth-tile traction: avoid instant 0 -> cruise wheel-speed steps.
            # Deceleration is deliberately NOT rate-limited so obstacle/safety
            # braking can still reduce speed immediately.
            if cfg.SLIPPERY_TILE_MODE and x_cmd > last_x_cmd:
                dt_cmd = max(0.005, min(0.20, now - last_cmd_t))
                x_cmd = min(x_cmd, last_x_cmd + cfg.MOVE_FORWARD_ACCEL_LIMIT_MPS2 * dt_cmd)
            last_x_cmd = float(x_cmd)
            last_cmd_t = now

            z_cmd = (
                self.lateral_yaw_hold_command(target_yaw)
                if side_escape_sign != 0.0
                else self.yaw_hold_command(target_yaw)
            )
            if not self.drive_speed_resilient(
                x=x_cmd, y=y_cmd, z=z_cmd, timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="MOVE CMD"
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
            time.sleep(cfg.CONTROL_DT)

        self.safe_stop()
        return self.MOVE_STOPPED
