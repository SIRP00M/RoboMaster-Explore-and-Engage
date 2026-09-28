#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Motion control, yaw lock, turning, corridor centering, and recovery."""

import math
import statistics
import threading
import time

from robomaster import robot

from config import *
from src.core.geometry import wrap_deg, clamp, DIR_NAMES, DIR_VEC, neighbor


class MotionMixin:
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


