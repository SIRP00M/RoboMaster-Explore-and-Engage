"""Sensor state, callbacks, geometry, Sharp and IR processing."""

from . import config as cfg
import math
import statistics
import threading
import time


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
        planar_m = (raw_m * abs(math.cos(math.radians(float(pitch_deg))))) + float(cfg.TOF_FORWARD_FROM_CENTER_M)
        return max(0.0, planar_m * 1000.0)
    except Exception:
        return None


def tof_topology_center_mm(raw_tof_mm):
    """Robot-centre planar range used only for OPEN/WALL topology."""
    return tof_center_planar_mm(raw_tof_mm, cfg.GIMBAL_PITCH_DEG)


def tof_is_open_from_center(raw_tof_mm):
    centre_mm = tof_topology_center_mm(raw_tof_mm)
    return bool(centre_mm is not None and centre_mm > cfg.TOF_OPEN_THRESHOLD_MM)


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
    dx, dy = cfg.DIR_VEC[int(direction) % 4]
    return (cell[0] + dx, cell[1] + dy)


def cell_in_field(cell):
    """True only for logical cells inside the currently configured arena."""
    try:
        x, y = int(cell[0]), int(cell[1])
    except Exception:
        return False
    return cfg.GRID_X_MIN <= x <= cfg.GRID_X_MAX and cfg.GRID_Y_MIN <= y <= cfg.GRID_Y_MAX


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    for d, (vx, vy) in cfg.DIR_VEC.items():
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
    if not math.isfinite(adc) or adc < cfg.SHARP_MIN_PLAUSIBLE_ADC:
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
    if not math.isfinite(raw) or raw < cfg.SHARP_MIN_PLAUSIBLE_ADC:
        return False
    try:
        far_adc = float(calibration[-1][1])
    except Exception:
        return False
    return raw < far_adc


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


class SensorMixin:
    """Sensor callbacks, freshness, filtering and side clearance."""

    def tof_callback(self, distance_info):
        try:
            if distance_info and len(distance_info) > cfg.TOF_INDEX:
                v = float(distance_info[cfg.TOF_INDEX])
                if math.isfinite(v) and cfg.TOF_VALID_MIN_MM <= v <= cfg.TOF_VALID_MAX_MM:
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
            logical_pos = (raw[0] - ox, raw[1] - oy, raw[2] - oz)
            self.state.set_position(logical_pos)

            now = time.monotonic()
            prev = self._speed_prev_position
            if prev is not None:
                pt, pp = prev
                dt = now - float(pt)
                if 0.015 <= dt <= 0.50:
                    vx = (float(logical_pos[0]) - float(pp[0])) / dt
                    vy = (float(logical_pos[1]) - float(pp[1])) / dt
                    speed = math.hypot(vx, vy)
                    if math.isfinite(speed) and speed <= 2.5:
                        with self.speed_telemetry_lock:
                            a = 0.38
                            old_vx = float(self.odom_velocity.get("vx", 0.0))
                            old_vy = float(self.odom_velocity.get("vy", 0.0))
                            fv = a * vx + (1.0 - a) * old_vx
                            fw = a * vy + (1.0 - a) * old_vy
                            self.odom_velocity = {
                                "vx": fv, "vy": fw, "speed": math.hypot(fv, fw), "t": now
                            }
            self._speed_prev_position = (now, logical_pos)
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
        return self._fresh(item, cfg.POSITION_STALE_SEC) if fresh else (None if item is None else item[0])

    def current_yaw(self, fresh=True):
        item = self.state.get_attitude()
        value = self._fresh(item, cfg.ATTITUDE_STALE_SEC) if fresh else (None if item is None else item[0])
        return None if value is None else float(value[0])

    def current_gimbal_raw(self, fresh=True):
        item = self.state.get_gimbal()
        value = self._fresh(item, 0.50) if fresh else (None if item is None else item[0])
        if value is None:
            return None, None
        return float(value[0]), float(value[1])

    def latest_tof(self, fresh=True):
        item = self.state.get_tof()
        value = self._fresh(item, cfg.FRONT_TOF_STALE_SEC) if fresh else (None if item is None else item[0])
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

    def read_sharp_adc(self):
        left = right = None
        try:
            left = self.sensor_adapter.get_adc(id=cfg.SHARP_LEFT_ID, port=cfg.SENSOR_PORT)
        except Exception:
            pass
        try:
            right = self.sensor_adapter.get_adc(id=cfg.SHARP_RIGHT_ID, port=cfg.SENSOR_PORT)
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
        return adc_to_cm(la, cfg.LEFT_CAL), adc_to_cm(ra, cfg.RIGHT_CAL), la, ra

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
                raw = self.sensor_adapter.get_adc(id=cfg.SHARP_LEFT_ID, port=cfg.SENSOR_PORT)
                if raw is not None:
                    l_adc_last = float(raw)
                    cm = adc_to_cm(l_adc_last, cfg.LEFT_CAL)
                    if cm is not None and math.isfinite(float(cm)):
                        l_cm_vals.append(float(cm))
            except Exception:
                pass
            try:
                raw = self.sensor_adapter.get_adc(id=cfg.SHARP_RIGHT_ID, port=cfg.SENSOR_PORT)
                if raw is not None:
                    r_adc_last = float(raw)
                    cm = adc_to_cm(r_adc_last, cfg.RIGHT_CAL)
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
            l_raw = self.sensor_adapter.get_io(id=cfg.IR_LEFT_ID, port=cfg.SENSOR_PORT)
        except Exception:
            pass
        try:
            r_raw = self.sensor_adapter.get_io(id=cfg.IR_RIGHT_ID, port=cfg.SENSOR_PORT)
        except Exception:
            pass
        return l_raw == 0, r_raw == 0, l_raw, r_raw

    def read_ir_filtered(self, samples=cfg.IR_FILTER_SAMPLES):
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

        if current is not None and now - self.sharp_authority_since < cfg.SHARP_AUTHORITY_HOLD_SEC:
            current_dist = left_cm if current == "LEFT" else right_cm
            if current_dist is not None:
                return current

        if left_cm is None and right_cm is None:
            new = None
        elif left_cm is None:
            new = "RIGHT"
        elif right_cm is None:
            new = "LEFT"
        elif left_cm <= cfg.SHARP_AUTHORITY_DANGER_CM or right_cm <= cfg.SHARP_AUTHORITY_DANGER_CM:
            new = "LEFT" if left_cm <= right_cm else "RIGHT"
        else:
            lerr = abs(left_cm - cfg.CENTER_TARGET_CM)
            rerr = abs(right_cm - cfg.CENTER_TARGET_CM)
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
            if d < cfg.SHARP_FOLLOW_NEAR_CM:
                y = max(cfg.SHARP_FOLLOW_MIN_STRAFE_MPS, cfg.SHARP_FOLLOW_KP * (cfg.SHARP_FOLLOW_NEAR_CM - d))
                if d <= cfg.SHARP_EMERGENCY_CM:
                    y = max(y, cfg.SHARP_HARD_STRAFE_MPS)
                return clamp(y, 0.0, cfg.SHARP_HARD_STRAFE_MPS), auth
            if d > cfg.SHARP_FOLLOW_FAR_CM and d < cfg.SHARP_AUTHORITY_RELEASE_CM:
                y = -max(cfg.SHARP_FOLLOW_MIN_STRAFE_MPS, cfg.SHARP_FOLLOW_KP * (d - cfg.SHARP_FOLLOW_FAR_CM))
                return clamp(y, -cfg.SHARP_MAX_STRAFE_MPS, 0.0), auth
            return 0.0, auth

        if auth == "RIGHT":
            d = right_cm
            if d is None:
                return 0.0, auth
            if d < cfg.SHARP_FOLLOW_NEAR_CM:
                y = -max(cfg.SHARP_FOLLOW_MIN_STRAFE_MPS, cfg.SHARP_FOLLOW_KP * (cfg.SHARP_FOLLOW_NEAR_CM - d))
                if d <= cfg.SHARP_EMERGENCY_CM:
                    y = min(y, -cfg.SHARP_HARD_STRAFE_MPS)
                return clamp(y, -cfg.SHARP_HARD_STRAFE_MPS, 0.0), auth
            if d > cfg.SHARP_FOLLOW_FAR_CM and d < cfg.SHARP_AUTHORITY_RELEASE_CM:
                y = max(cfg.SHARP_FOLLOW_MIN_STRAFE_MPS, cfg.SHARP_FOLLOW_KP * (d - cfg.SHARP_FOLLOW_FAR_CM))
                return clamp(y, 0.0, cfg.SHARP_MAX_STRAFE_MPS), auth
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
        l_low, r_low, l_raw, r_raw = self.read_ir_filtered(samples=cfg.IR_FILTER_SAMPLES)

        if not l_low and not r_low:
            return "CLEAR"

        # Read Sharp before stopping/escaping so sunlight-induced digital IR LOW
        # cannot by itself force a chassis maneuver.
        l_pre_cm, r_pre_cm, _l_pre_adc, _r_pre_adc = self.read_sharp_cm()

        if l_low and r_low:
            if (
                cfg.IR_SUNLIGHT_GUARD_ENABLED
                and (l_pre_cm is None or l_pre_cm > cfg.IR_SUNLIGHT_CORROBORATE_MAX_CM)
                and (r_pre_cm is None or r_pre_cm > cfg.IR_SUNLIGHT_CORROBORATE_MAX_CM)
            ):
                print(
                    "[IR SUN GUARD {}] BOTH LOW but Sharp L={} R={}cm -> "
                    "ambient-light suspect; ignore digital IR".format(
                        context, l_pre_cm, r_pre_cm
                    )
                )
                return "SUNLIGHT"
            self.safe_stop()
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

        source_pre_cm = l_pre_cm if sign > 0.0 else r_pre_cm
        if (
            cfg.IR_SUNLIGHT_GUARD_ENABLED
            and (source_pre_cm is None or source_pre_cm > cfg.IR_SUNLIGHT_CORROBORATE_MAX_CM)
        ):
            print(
                "[IR SUN GUARD {}] {} LOW but source Sharp={}cm > {:.1f}cm -> "
                "ignore IR; Sharp/ToF own safety".format(
                    context, source_name, source_pre_cm, cfg.IR_SUNLIGHT_CORROBORATE_MAX_CM
                )
            )
            return "SUNLIGHT"

        self.safe_stop()

        # IR is only a corner/edge feeler; Sharp owns physical side clearance.
        # If IR asks us to strafe INTO a side that is already too close, veto the
        # maneuver before moving.  This is the exact failure seen in the field log:
        # RIGHT IR kept requesting LEFT while LEFT Sharp was ~5 cm.
        l0_cm, r0_cm = l_pre_cm, r_pre_cm
        source0_cm = l0_cm if sign > 0.0 else r0_cm
        dest0_cm = r0_cm if sign > 0.0 else l0_cm
        if (
            dest0_cm is not None
            and dest0_cm <= cfg.IR_DESTINATION_VETO_CM
            and source0_cm is not None
            and source0_cm >= cfg.IR_SIMPLE_SOURCE_SHARP_CLEAR_CM
        ):
            print(
                "[IR+SHARP {} VETO] {} LOW wants {} but {} Sharp={:.1f}cm "
                "while source Sharp={:.1f}cm -> Sharp owns; no IR strafe".format(
                    context, source_name, direction_name, dest_name,
                    dest0_cm, source0_cm,
                )
            )
            return "VETO"

        anchor = self.current_position()
        if anchor is None:
            print(
                "[IR SIMPLE {}] {} LOW but no odometry -> STOP only".format(
                    context, source_name
                )
            )
            return "HOLD"

        deadline = time.monotonic() + float(cfg.IR_SIMPLE_TIMEOUT_SEC)
        clear_count = 0
        sharp_clear_count = 0
        moved = 0.0
        stop_reason = "TIMEOUT"
        last_source_cm = None
        last_dest_cm = None

        print(
            "[IR+SHARP {}] {} LOW -> strafe {} until IR clears or {} Sharp >= {:.1f}cm".format(
                context, source_name, direction_name,
                source_name, cfg.IR_SIMPLE_SOURCE_SHARP_CLEAR_CM,
            )
        )

        while self.running and time.monotonic() < deadline:
            pos = self.current_position()
            if pos is None:
                stop_reason = "NO_ODOM"
                break

            moved = abs(self.cell_lateral_offset(anchor, pos, target_yaw))
            if moved >= cfg.IR_SIMPLE_MAX_LATERAL_M:
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
                and source_cm >= cfg.IR_SIMPLE_SOURCE_SHARP_CLEAR_CM
                and moved >= cfg.IR_SIMPLE_SOURCE_SHARP_CLEAR_MIN_MOVE_M
            )
            sharp_clear_count = sharp_clear_count + 1 if sharp_safe else 0

            if clear_count >= cfg.IR_SIMPLE_CLEAR_CONFIRM:
                stop_reason = "IR_CLEAR"
                break
            if sharp_clear_count >= cfg.IR_SIMPLE_CLEAR_CONFIRM:
                stop_reason = "SOURCE_SHARP_CLEAR"
                break

            if dest_cm is not None and dest_cm <= cfg.IR_SIMPLE_DEST_SHARP_STOP_CM:
                stop_reason = "{}_SHARP_{:.1f}CM".format(dest_name, dest_cm)
                break

            # Adaptive escape speed: run fast while destination clearance is
            # healthy, slow down as the opposite wall approaches.
            speed = cfg.IR_SIMPLE_STRAFE_SPEED_MPS
            if dest_cm is None or dest_cm > cfg.IR_SIMPLE_DEST_SHARP_CAUTION_CM + 2.0:
                speed = cfg.IR_SIMPLE_STRAFE_FAST_MPS
            elif dest_cm <= cfg.IR_SIMPLE_DEST_SHARP_CAUTION_CM:
                speed = cfg.IR_SIMPLE_STRAFE_SLOW_MPS

            if not self.drive_speed_resilient(
                x=0.0,
                y=sign * speed,
                z=self.lateral_yaw_hold_command(target_yaw),
                timeout=cfg.DRIVE_COMMAND_TIMEOUT,
                label="IR+SHARP STRAFE",
            ):
                stop_reason = "DRIVE_FAIL"
                break

            time.sleep(cfg.CONTROL_DT)

        self.safe_stop()

        # Do not pay a stationary PID settle on every tiny corner nudge.  Only
        # re-align when the lateral maneuver actually displaced yaw visibly.
        post_err = self.yaw_error_deg(target_yaw)
        if post_err is None or abs(post_err) > cfg.LATERAL_RECOVERY_REALIGN_TRIGGER_DEG:
            self.align_heading_stationary(
                target_yaw,
                timeout_sec=min(0.45, cfg.POST_MOVE_ALIGN_TIMEOUT_SEC),
                tolerance_deg=max(0.75, cfg.PRE_MOVE_ALIGN_TOL_DEG),
                settle_sec=0.04,
            )

        l3, r3, lr3, rr3 = self.read_ir_filtered(samples=2)
        source_still_low = l3 if sign > 0.0 else r3

        # Sharp-confirmed clearance is accepted even if the angled digital IR
        # remains LOW for a moment; front ToF + Sharp continue guarding motion.
        source_sharp_clear = bool(
            last_source_cm is not None
            and last_source_cm >= cfg.IR_SIMPLE_SOURCE_SHARP_CLEAR_CM
            and moved >= cfg.IR_SIMPLE_SOURCE_SHARP_CLEAR_MIN_MOVE_M
        )
        if not source_still_low:
            stop_reason = "IR_CLEAR"
            result = "CLEAR"
        elif source_sharp_clear:
            stop_reason = "SOURCE_SHARP_CLEAR"
            result = "CLEAR"
        elif (
            last_dest_cm is not None
            and last_dest_cm <= cfg.IR_DESTINATION_VETO_CM
        ):
            # Destination-side Sharp veto: do not immediately retry the same IR
            # request on the next control cycle. The main Sharp escape controller
            # will move away from the close wall instead.
            result = "VETO"
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
