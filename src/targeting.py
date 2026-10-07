"""Target sweeps, viewpoint shifts, aim/fire gates and target memory."""

from . import config as cfg
import json
import math
import time
from .config import atomic_write_text
from datetime import datetime
from robomaster import blaster
from .sensors import (
    angle_diff_deg,
    clamp,
    direction_between,
    fmt_deg,
    sharp_raw_means_far,
    tof_center_planar_mm,
    tof_is_open_from_center,
    wrap_deg,
)


class TargetingMixin:
    """Sweeps, viewpoint shifts, aiming, firing and target records."""

    def _stop_gimbal_velocity(self):
        if self.owner.gimbal is None:
            return
        try:
            self.owner.gimbal.drive_speed(pitch_speed=0.0,yaw_speed=0.0)
        except Exception:
            pass

    def _goto_target_pose_strict(self, yaw_deg, pitch_deg, timeout_sec=1.35, allow_soft=False):
        """Fast feedback-driven target-search pose.

        V20 used SDK moveto() here and the generic gimbal_goto() could retry the
        same failed action several times before falling back to velocity control.
        Field log 2026-10-01 showed the turret sitting around -53.6 deg while
        FRONT wanted -45 deg, wasting seconds and causing whole sectors to skip.

        Target SEARCH does not require topology-grade exact pitch.  Drive the
        turret directly from sub_angle feedback, detect a real stall, and permit
        a bounded soft yaw hand-off because the following continuous sweep will
        still cross the acquisition cone.
        """
        yaw_deg = clamp(float(yaw_deg), cfg.GIMBAL_SOFT_YAW_MIN_DEG, cfg.GIMBAL_SOFT_YAW_MAX_DEG)
        pitch_deg = clamp(float(pitch_deg), cfg.TARGET_CENTER_PITCH_HARD_MIN_DEG, cfg.TARGET_CENTER_PITCH_MAX_DEG)

        p0, y0 = self.owner.current_gimbal_relative()
        yaw_distance = 90.0 if y0 is None else abs(wrap_deg(yaw_deg - float(y0)))
        pitch_distance = 10.0 if p0 is None else abs(float(pitch_deg) - float(p0))
        dynamic_timeout = max(
            float(timeout_sec),
            yaw_distance / max(45.0, cfg.GIMBAL_RECOVERY_YAW_MAX_DPS * 0.82) + 0.40,
            pitch_distance / max(14.0, cfg.GIMBAL_RECOVERY_PITCH_MAX_DPS * 0.75) + 0.35,
        )
        deadline = time.monotonic() + max(0.35, dynamic_timeout)
        stable = 0
        last_progress_t = time.monotonic()
        last_yaw = y0
        stall_pulses = 0

        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                p_now, y_now = self.owner.current_gimbal_relative()
                if p_now is None or y_now is None:
                    self._stop_gimbal_velocity()
                    time.sleep(0.04)
                    continue

                ey = wrap_deg(yaw_deg - float(y_now))
                ep = float(pitch_deg) - float(p_now)
                yaw_ok = abs(ey) <= cfg.TARGET_SEARCH_POSE_YAW_TOL_DEG
                pitch_ok = abs(ep) <= cfg.TARGET_SEARCH_POSE_PITCH_TOL_DEG
                if yaw_ok and pitch_ok:
                    stable += 1
                    self._stop_gimbal_velocity()
                    if stable >= 2:
                        return True
                    time.sleep(0.04)
                    continue
                stable = 0

                # Progress watchdog: repeated low-speed commands around a sticky
                # point were seen to leave the EP gimbal sitting at -53.x deg.
                if last_yaw is None or angle_diff_deg(float(y_now), float(last_yaw)) >= cfg.TARGET_GIMBAL_STALL_MIN_PROGRESS_DEG:
                    last_yaw = float(y_now)
                    last_progress_t = time.monotonic()
                    stall_pulses = 0
                elif abs(ey) > cfg.TARGET_SEARCH_POSE_YAW_TOL_DEG and (time.monotonic() - last_progress_t) >= cfg.TARGET_GIMBAL_STALL_WINDOW_SEC:
                    self._stop_gimbal_velocity()
                    time.sleep(0.045)
                    stall_pulses += 1
                    last_progress_t = time.monotonic()

                ys = 0.0 if yaw_ok else clamp(3.0 * ey, -cfg.GIMBAL_RECOVERY_YAW_MAX_DPS, +cfg.GIMBAL_RECOVERY_YAW_MAX_DPS)
                ps = 0.0 if pitch_ok else clamp(2.2 * ep, -cfg.GIMBAL_RECOVERY_PITCH_MAX_DPS, +cfg.GIMBAL_RECOVERY_PITCH_MAX_DPS)

                # Strong enough to overcome gimbal stiction, but only while more
                # than a few degrees away.  Near target we let proportional speed
                # settle instead of hammering it at 120-300 dps.
                yaw_floor = 22.0 if abs(ey) > 6.0 else 10.0
                if 0.0 < abs(ys) < yaw_floor:
                    ys = math.copysign(yaw_floor, ys)
                if 0.0 < abs(ps) < 3.0:
                    ps = math.copysign(3.0, ps)
                if stall_pulses > 0 and abs(ey) > 6.0:
                    ys = math.copysign(min(cfg.GIMBAL_RECOVERY_YAW_MAX_DPS, 85.0), ey)

                self.owner.gimbal.drive_speed(pitch_speed=float(ps), yaw_speed=float(ys))
                time.sleep(0.040)
        except Exception as exc:
            self.owner.fault(
                "TARGET POSE",
                f"yaw={yaw_deg:+.1f} pitch={pitch_deg:+.1f}: {type(exc).__name__}: {exc}",
                "soft-check pose / continue mission",
            )
        finally:
            self._stop_gimbal_velocity()

        p_now, y_now = self.owner.current_gimbal_relative()
        if p_now is None or y_now is None:
            return False
        yaw_err = abs(wrap_deg(float(y_now) - yaw_deg))
        pitch_err = abs(float(p_now) - pitch_deg)
        if yaw_err <= cfg.TARGET_SEARCH_POSE_YAW_TOL_DEG and pitch_err <= cfg.TARGET_SEARCH_POSE_PITCH_TOL_DEG:
            return True
        if allow_soft and yaw_err <= cfg.TARGET_SEARCH_POSE_SOFT_YAW_TOL_DEG:
            print(
                f"[TARGET GIMBAL SOFT HANDOFF] targetYaw={yaw_deg:+.1f} actual={float(y_now):+.1f} "
                f"yawErr={yaw_err:.1f} pitch={float(p_now):+.1f} -> sweep continues"
            )
            return True
        return False

    def _aim_velocity(self, center_norm, yaw_min=None, yaw_max=None):
        try:
            cx,cy=[float(v) for v in center_norm]
        except Exception:
            self._stop_gimbal_velocity(); return False
        desired_x=0.5+cfg.TARGET_AIM_OFFSET_X; desired_y=0.5+cfg.TARGET_AIM_OFFSET_Y
        ex=cx-desired_x; ey=cy-desired_y
        yaw_speed=0.0 if abs(ex)<=cfg.TARGET_AIM_DEADBAND_X else clamp(ex*cfg.TARGET_AIM_YAW_GAIN_DPS,-cfg.TARGET_AIM_YAW_MAX_DPS,cfg.TARGET_AIM_YAW_MAX_DPS)
        # V7: horizontal servo keeps the existing limits, while vertical
        # centering gets its own deeper/flexible pitch authority.  This is only
        # called after SHOOT-INTENT, never during DFS topology scans.
        pitch_speed=0.0 if abs(ey)<=cfg.TARGET_AIM_DEADBAND_Y else clamp(
            -ey*cfg.TARGET_CENTER_PITCH_GAIN_DPS,
            -cfg.TARGET_CENTER_PITCH_MAX_DPS,
            +cfg.TARGET_CENTER_PITCH_MAX_DPS,
        )
        p_now,y_now=self.owner.current_gimbal_relative()
        if y_now is not None:
            # Global mechanical/software limit intersected with the sector's
            # +/-30deg crosshair envelope.  The narrow +/-10deg acquisition
            # cone is intentionally NOT used here: once shoot-intent is earned,
            # the crosshair may move farther to put the real target centre on aim.
            lo=-float(cfg.TARGET_AIM_YAW_LIMIT_DEG)
            hi=+float(cfg.TARGET_AIM_YAW_LIMIT_DEG)
            if yaw_min is not None:
                lo=max(lo,float(yaw_min))
            if yaw_max is not None:
                hi=min(hi,float(yaw_max))
            if y_now<=lo and yaw_speed<0: yaw_speed=0.0
            if y_now>=hi and yaw_speed>0: yaw_speed=0.0
        if p_now is not None:
            # V7 FLEX CENTER PITCH: the old -14deg limit was correct for search
            # but too shallow for close targets low in the camera.  During AIM
            # only, brake near -20.5deg and permit a hard floor down to -22.5deg.
            # Search resumes at -10deg immediately after target service.
            if p_now<=cfg.TARGET_CENTER_PITCH_HARD_MIN_DEG and pitch_speed<0:
                pitch_speed=0.0
            elif pitch_speed<0 and p_now < (cfg.TARGET_CENTER_PITCH_SOFT_MIN_DEG + cfg.TARGET_CENTER_PITCH_BRAKE_ZONE_DEG):
                headroom=max(0.0,float(p_now)-cfg.TARGET_CENTER_PITCH_HARD_MIN_DEG)
                pitch_speed=max(float(pitch_speed),-max(2.0,6.0*headroom))
            if p_now>=cfg.TARGET_CENTER_PITCH_MAX_DEG and pitch_speed>0:
                pitch_speed=0.0
        try:
            self.owner.gimbal.drive_speed(pitch_speed=float(pitch_speed),yaw_speed=float(yaw_speed))
            return True
        except Exception as exc:
            self.owner.fault("TARGET AIM",f"drive_speed failed: {type(exc).__name__}: {exc}","abort target lock only")
            return False

    def _aim_and_lock(self,candidate, aim_yaw_min=None, aim_yaw_max=None):
        current=dict(candidate); expected=list(current.get("center_norm",(0.5,0.5)))
        self.owner.safe_stop(); self._set_focus_roi(current)
        # Keep the chassis on its DFS cardinal while the turret servos.  The log
        # showed repeated target locks could otherwise accumulate ~12deg base yaw.
        chassis_hold_yaw=self.owner.desired_yaw_for_heading(self.owner.heading)
        self.owner.pid_straight.reset()
        deadline=time.monotonic()+cfg.TARGET_AIM_TIMEOUT_SEC
        grace_used=False
        last_seq=-1; lost_since=None; centered_frames=0
        last_err=(None,None); verify_failures=0
        print(f"[TARGET ACQUIRE] {current.get('color')} {current.get('shape')} score={current.get('score',0):.2f}")
        if aim_yaw_min is not None and aim_yaw_max is not None:
            print(
                "[TARGET CROSSHAIR ENVELOPE] yaw={:+.1f}..{:+.1f}deg ".format(
                    float(aim_yaw_min), float(aim_yaw_max),
                )
                + "pitchCenter={:+.1f}..{:+.1f}deg hardDown={:+.1f}deg".format(
                    float(cfg.TARGET_CENTER_PITCH_SOFT_MIN_DEG),
                    float(cfg.TARGET_CENTER_PITCH_MAX_DEG),
                    float(cfg.TARGET_CENTER_PITCH_HARD_MIN_DEG),
                )
            )
        try:
            while self.owner.running and self.running and time.monotonic()<deadline:
                # Counter gimbal reaction torque with zero-translation chassis
                # yaw hold.  Failure is non-fatal to target service; the sector
                # hand-off performs a final stationary alignment as well.
                if chassis_hold_yaw is not None:
                    try:
                        z_hold=self.owner.yaw_hold_command(chassis_hold_yaw,stationary=False)
                        self.owner.drive_speed_resilient(
                            x=0.0,y=0.0,z=z_hold,timeout=cfg.DRIVE_COMMAND_TIMEOUT,
                            label="TARGET AIM YAW HOLD",
                        )
                    except Exception:
                        pass
                with self.lock: seq=self.frame_seq
                if seq==last_seq:
                    time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
                last_seq=seq
                latest=self._latest_match(current,expected,allow_blob_fallback=True)
                if latest is None:
                    self._stop_gimbal_velocity(); centered_frames=0
                    if lost_since is None: lost_since=time.monotonic()
                    if time.monotonic()-lost_since>cfg.TARGET_AIM_LOST_GRACE_SEC:
                        print("[TARGET LOST] lock grace expired")
                        return None
                    time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
                lost_since=None; current.update(latest); expected=list(current.get("center_norm",expected)); self._set_focus_roi(current)
                cx,cy=[float(v) for v in expected]
                ex=cx-(0.5+cfg.TARGET_AIM_OFFSET_X); ey=cy-(0.5+cfg.TARGET_AIM_OFFSET_Y)
                last_err=(ex,ey)
                if abs(ex)<=cfg.TARGET_AIM_CENTER_TOL_X and abs(ey)<=cfg.TARGET_AIM_CENTER_TOL_Y:
                    centered_frames+=1; self._stop_gimbal_velocity()
                else:
                    centered_frames=0
                    if not self._aim_velocity(expected, yaw_min=aim_yaw_min, yaw_max=aim_yaw_max): return None
                self.status=f"AIM {current.get('color')} {current.get('shape')} err=({ex:+.3f},{ey:+.3f}) hold={centered_frames}/{cfg.TARGET_AIM_CENTER_HOLD_FRAMES}"
                if centered_frames>=cfg.TARGET_AIM_CENTER_HOLD_FRAMES:
                    # V15: the quick-gate already established a multi-frame real target,
                    # and AIM itself has now held the BBOX centre on the crosshair for
                    # 3 consecutive fresh frames.  Do not add another verification wait
                    # here: proceed directly to fresh ToF -> camera/muzzle compensation.
                    self._stop_gimbal_velocity()
                    locked = dict(current)
                    locked["crosshair_centered"] = True
                    locked["crosshair_error_norm"] = [float(ex), float(ey)]
                    locked["center_hold_frames"] = int(centered_frames)
                    print(
                        "[TARGET CENTER LOCK] {} {} bbox-center err=({:+.3f},{:+.3f}) "
                        "hold={}/{} -> ToF/fire pipeline".format(
                            locked.get("color"), locked.get("shape"), ex, ey,
                            centered_frames, cfg.TARGET_AIM_CENTER_HOLD_FRAMES,
                        )
                    )
                    self._set_focus_roi(locked)
                    return locked
                time.sleep(cfg.TARGET_AIM_POLL_SEC)
            # One short grace only when the target is genuinely almost centered.
            ex,ey=last_err
            if (
                not grace_used and ex is not None and ey is not None
                and abs(ex) <= cfg.TARGET_AIM_NEAR_CENTER_GRACE_ERR
                and abs(ey) <= cfg.TARGET_AIM_NEAR_CENTER_GRACE_ERR
                and self.owner.running and self.running
            ):
                grace_used=True
                deadline=time.monotonic()+cfg.TARGET_AIM_NEAR_CENTER_GRACE_SEC
                print(
                    f"[TARGET AIM GRACE] near center err=({ex:+.3f},{ey:+.3f}) "
                    f"+{cfg.TARGET_AIM_NEAR_CENTER_GRACE_SEC:.2f}s"
                )
                while self.owner.running and self.running and time.monotonic()<deadline:
                    with self.lock: seq=self.frame_seq
                    if seq==last_seq:
                        time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
                    last_seq=seq
                    latest=self._latest_match(current,expected,allow_blob_fallback=True)
                    if latest is None:
                        time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
                    current.update(latest); expected=list(current.get("center_norm",expected)); self._set_focus_roi(current)
                    cx,cy=[float(v) for v in expected]
                    ex=cx-(0.5+cfg.TARGET_AIM_OFFSET_X); ey=cy-(0.5+cfg.TARGET_AIM_OFFSET_Y)
                    last_err=(ex,ey)
                    if abs(ex)<=cfg.TARGET_AIM_CENTER_TOL_X and abs(ey)<=cfg.TARGET_AIM_CENTER_TOL_Y:
                        centered_frames+=1; self._stop_gimbal_velocity()
                    else:
                        centered_frames=0
                        if not self._aim_velocity(expected, yaw_min=aim_yaw_min, yaw_max=aim_yaw_max): break
                    if centered_frames>=cfg.TARGET_AIM_CENTER_HOLD_FRAMES:
                        locked=dict(current)
                        locked["crosshair_centered"]=True
                        locked["crosshair_error_norm"]=[float(ex),float(ey)]
                        locked["center_hold_frames"]=int(centered_frames)
                        print(
                            f"[TARGET CENTER LOCK] {locked.get('color')} {locked.get('shape')} "
                            f"bbox-center err=({ex:+.3f},{ey:+.3f}) hold={centered_frames}/{cfg.TARGET_AIM_CENTER_HOLD_FRAMES} -> ToF/fire pipeline"
                        )
                        self._set_focus_roi(locked)
                        return locked
                    time.sleep(cfg.TARGET_AIM_POLL_SEC)

            p_end,y_end=self.owner.current_gimbal_relative()
            ex,ey=last_err
            print(
                "[TARGET AIM TIMEOUT] yaw={} pitch={} err=({}, {}) verifyFail={}".format(
                    fmt_deg(y_end), fmt_deg(p_end),
                    "NA" if ex is None else "{:+.3f}".format(ex),
                    "NA" if ey is None else "{:+.3f}".format(ey),
                    verify_failures,
                )
            )
            return None
        finally:
            self._stop_gimbal_velocity(); self._clear_focus_roi()
            self.owner.safe_stop()
            if chassis_hold_yaw is not None:
                try:
                    aim_exit_err = self.owner.yaw_error_deg(chassis_hold_yaw)
                    if aim_exit_err is None or abs(aim_exit_err) > cfg.TARGET_CHASSIS_REALIGN_TRIGGER_DEG:
                        self.owner.align_heading_stationary(
                            chassis_hold_yaw,timeout_sec=cfg.TARGET_CHASSIS_REALIGN_TIMEOUT_SEC,
                            tolerance_deg=max(0.90,cfg.PRE_MOVE_ALIGN_TOL_DEG),settle_sec=0.035,
                        )
                except Exception:
                    pass

    def _temporary_side_shift(self, direction, anchor_pos=None, preposition_yaw=None):
        """Move the chassis laterally ~35 cm to create the side-camera viewpoint.

        V20.7 fixes the old semantic mistake: 35 cm is the *chassis displacement*
        from the logical node anchor, not a 350 mm wall-ToF target.  Therefore a
        side shift is attempted even when the side is completely OPEN.

        Speed depends only on remaining odometric shift.  Fresh destination Sharp
        is sampled at ~50 Hz and can only emergency-STOP the excursion.
        """
        token = {
            "direction": direction,
            "anchor_pos": anchor_pos,
            "target_yaw": self.owner.desired_yaw_for_heading(self.owner.heading),
            "shift_m": 0.0,
            "result": "NO_SHIFT",
            "dest_sharp_cm": None,
        }
        if not cfg.TARGET_SIDE_SHIFT_ENABLED or direction not in ("LEFT", "RIGHT"):
            return token

        self.owner.safe_stop()
        if token["target_yaw"] is None:
            token["result"] = "NO_YAW_REF"
            return token
        if anchor_pos is None:
            anchor_pos = self.owner.current_position()
            token["anchor_pos"] = anchor_pos
        if anchor_pos is None:
            token["result"] = "NO_POSITION"
            return token

        # Only pay for a stationary realign if chassis yaw is visibly off.
        shift_yaw_err = self.owner.yaw_error_deg(token["target_yaw"])
        if shift_yaw_err is None or abs(shift_yaw_err) > cfg.TARGET_CHASSIS_REALIGN_TRIGGER_DEG:
            self.owner.align_heading_stationary(
                token["target_yaw"], timeout_sec=0.45,
                tolerance_deg=max(0.90, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
            )
        self.owner.pid_straight.reset()

        sign = +1.0 if direction == "RIGHT" else -1.0
        dest_side = "RIGHT" if sign > 0 else "LEFT"

        # Hide gimbal reposition time underneath the 35 cm chassis slide.  The
        # later sector sweep still verifies feedback before it starts, so failure
        # here only loses the overlap optimization, never target coverage.
        if preposition_yaw is not None:
            try:
                self.owner.gimbal.moveto(
                    pitch=float(cfg.TARGET_SEARCH_PITCH_DEG), yaw=float(preposition_yaw),
                    pitch_speed=float(cfg.GIMBAL_PITCH_SPEED), yaw_speed=float(cfg.GIMBAL_YAW_SPEED),
                )
            except Exception:
                pass

        start_t = time.monotonic()
        last_log = 0.0
        missing_sharp_cycles = 0
        stop_reason = "FULL_35CM"

        while self.owner.running and self.running:
            now = time.monotonic()
            pos = self.owner.current_position()
            if pos is None:
                stop_reason = "POSITION_STALE"
                break

            lateral = self.owner.cell_lateral_offset(anchor_pos, pos, token["target_yaw"])
            shifted = max(0.0, sign * float(lateral))
            token["shift_m"] = shifted
            remaining = max(0.0, cfg.TARGET_SIDE_SHIFT_DISTANCE_M - shifted)

            if shifted >= cfg.TARGET_SIDE_SHIFT_DISTANCE_M - cfg.TARGET_SIDE_SHIFT_STOP_TOL_M:
                stop_reason = "FULL_35CM"
                break
            if shifted >= cfg.TARGET_SIDE_SHIFT_MAX_M:
                stop_reason = "ODOM_MAX_GUARD"
                break
            if now - start_t >= cfg.TARGET_SIDE_SHIFT_TIMEOUT_SEC:
                stop_reason = "SHIFT_TIMEOUT"
                break

            # Fresh, unfiltered destination Sharp.  FAR>CAL means safely beyond
            # the calibrated close-range sensor span, not a missing sensor.
            left_cm, right_cm, la_raw, ra_raw = self.owner.read_sharp_cm_fast(
                samples=cfg.TARGET_SIDE_SHIFT_FAST_SHARP_SAMPLES,
                interval_sec=cfg.TARGET_SIDE_SHIFT_FAST_SHARP_INTERVAL_SEC,
            )
            dest_cm = right_cm if sign > 0 else left_cm
            dest_raw = ra_raw if sign > 0 else la_raw
            dest_cal = cfg.RIGHT_CAL if sign > 0 else cfg.LEFT_CAL
            dest_far = (dest_cm is None and sharp_raw_means_far(dest_raw, dest_cal))
            token["dest_sharp_cm"] = dest_cm

            if dest_far:
                missing_sharp_cycles = 0
            elif dest_cm is None:
                missing_sharp_cycles += 1
                if missing_sharp_cycles >= cfg.TARGET_SIDE_SHIFT_SHARP_MISSING_MAX_CYCLES:
                    stop_reason = f"{dest_side}_SHARP_UNAVAILABLE"
                    self.owner.safe_stop()
                    break
            else:
                missing_sharp_cycles = 0
                if dest_cm <= cfg.TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM:
                    stop_reason = f"{dest_side}_SHARP_EMERGENCY_STOP"
                    self.owner.safe_stop()
                    break

            # Exact user-requested four-stage profile, based on *remaining chassis
            # displacement* rather than wall-ToF distance.
            if remaining <= cfg.TARGET_SIDE_SHIFT_CRAWL_REMAIN_M:
                speed = cfg.TARGET_SIDE_SHIFT_CRAWL_MPS
            elif remaining <= cfg.TARGET_SIDE_SHIFT_SLOW_REMAIN_M:
                speed = cfg.TARGET_SIDE_SHIFT_SLOW_MPS
            elif remaining <= cfg.TARGET_SIDE_SHIFT_MED_REMAIN_M:
                speed = cfg.TARGET_SIDE_SHIFT_MED_SPEED_MPS
            else:
                speed = cfg.TARGET_SIDE_SHIFT_SPEED_MPS

            z_cmd = self.owner.lateral_yaw_hold_command(token["target_yaw"])
            if not self.owner.drive_speed_resilient(
                x=0.0, y=sign * speed, z=z_cmd,
                timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="TARGET FULL35 SHIFT",
            ):
                stop_reason = "DRIVE_COMMAND_FAILED"
                break

            if now - last_log >= 0.18:
                sharp_text = (
                    "FAR>CAL" if dest_far else
                    ("NA" if dest_cm is None else f"{dest_cm:.1f}cm")
                )
                print(
                    f"[TARGET FULL35 {dest_side}] d={shifted:.3f}m "
                    f"remain={remaining*1000:.0f}mm {dest_side}SharpFAST={sharp_text} "
                    f"v={speed:.3f}m/s"
                )
                last_log = now
            time.sleep(cfg.TARGET_SIDE_SHIFT_CONTROL_DT)

        self.owner.safe_stop()
        # No extra ToF sample/settle here: the following sweep already validates
        # the gimbal pose and the camera thread is continuously live.
        token["result"] = stop_reason
        print(
            f"[TARGET FULL35 DONE] dir={direction} shift={token['shift_m']:.3f}m "
            f"destSharp={token.get('dest_sharp_cm')} result={stop_reason}"
        )
        return token

    def _is_confirmed_dead_end_cell(self, cell):
        """True only when stationary topology confirmed L/F/R are all walls.

        The BACK edge is the already-traversed way out of the cell.  UNKNOWN is
        deliberately not accepted as a wall: a failed ToF ray must never trigger
        a 35 cm target-viewpoint excursion.
        """
        cell = tuple(cell)
        # A temporary reverse is allowed only when BACK is an already-traversed
        # open edge.  This deliberately excludes the root/staging cell and any
        # boxed/uncertain pose where reversing 35 cm has not been proven safe.
        parent = self.owner.parent.get(cell)
        if parent is None:
            return False
        back_dir = direction_between(cell, parent)
        if back_dir is None or self.owner.edge_state.get((cell, back_dir)) != "OPEN":
            return False

        scan = self.owner.cell_scan_mm.get(cell, {})
        for label in ("LEFT", "FRONT", "RIGHT"):
            mm = scan.get(label)
            if mm is None:
                return False
            try:
                mm = float(mm)
            except Exception:
                return False
            if (not math.isfinite(mm)) or tof_is_open_from_center(mm):
                return False
        return True

    def _temporary_deadend_front_backshift(self, cell, anchor_pos=None):
        """Back ~35 cm only for the FRONT target scan of a confirmed dead-end.

        This is a target-viewpoint excursion, not DFS motion.  Odometry remains
        referenced to the logical node anchor and the already-traversed BACK path
        is used.  The robot is returned to the same anchor after the FRONT sector.
        """
        token = {
            "kind": "DEADEND_FRONT_BACKSHIFT",
            "cell": tuple(cell),
            "anchor_pos": anchor_pos,
            "target_yaw": self.owner.desired_yaw_for_heading(self.owner.heading),
            "shift_m": 0.0,
            "result": "NO_SHIFT",
            "anchor_front_tof_mm": self.owner.cell_scan_mm.get(tuple(cell), {}).get("FRONT"),
        }
        if not cfg.TARGET_DEADEND_FRONT_BACKSHIFT_ENABLED:
            token["result"] = "DISABLED"
            return token
        if not self._is_confirmed_dead_end_cell(cell):
            token["result"] = "NOT_CONFIRMED_DEADEND"
            return token

        self.owner.safe_stop()
        target_yaw = token.get("target_yaw")
        if target_yaw is None:
            token["result"] = "NO_YAW_REF"
            return token
        if anchor_pos is None:
            anchor_pos = self.owner.current_position()
            token["anchor_pos"] = anchor_pos
        if anchor_pos is None:
            token["result"] = "NO_POSITION"
            return token

        self.owner.align_heading_stationary(
            target_yaw, timeout_sec=0.9,
            tolerance_deg=max(0.55, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.08,
        )
        self.owner.pid_straight.reset()

        start_t = time.monotonic()
        last_log = 0.0
        stop_reason = "TARGET_DISTANCE"

        while self.owner.running and self.running:
            now = time.monotonic()
            pos = self.owner.current_position()
            if pos is None:
                stop_reason = "POSITION_STALE"
                break

            fwd = float(self.owner.cell_forward_progress(anchor_pos, pos, target_yaw))
            lat = float(self.owner.cell_lateral_offset(anchor_pos, pos, target_yaw))
            backed = max(0.0, -fwd)
            token["shift_m"] = backed

            if backed >= cfg.TARGET_DEADEND_FRONT_BACKSHIFT_M:
                stop_reason = "BACKSHIFT_35CM"
                break
            if now - start_t >= cfg.TARGET_DEADEND_FRONT_BACKSHIFT_TIMEOUT_SEC:
                stop_reason = "TIMEOUT_BEFORE_35CM"
                break

            remaining = max(0.0, cfg.TARGET_DEADEND_FRONT_BACKSHIFT_M - backed)
            x_mag = cfg.TARGET_DEADEND_FRONT_BACKSHIFT_SPEED_MPS
            if remaining <= cfg.TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_ZONE_M:
                x_mag = cfg.TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_MPS

            # Keep the temporary reverse on the same centreline.  This uses only
            # small odometry lateral correction; IR is intentionally not involved.
            y_cmd = clamp(
                -cfg.TARGET_DEADEND_FRONT_LATERAL_KP * lat,
                -cfg.TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
                +cfg.TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
            )
            if abs(lat) <= 0.010:
                y_cmd = 0.0
            z_cmd = self.owner.yaw_hold_command(target_yaw, stationary=False)

            if not self.owner.drive_speed_resilient(
                x=-x_mag, y=y_cmd, z=z_cmd,
                timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="TARGET DEADEND BACKSHIFT",
            ):
                stop_reason = "DRIVE_CMD_FAIL"
                break

            if now - last_log >= 0.30:
                print(
                    f"[TARGET DEADEND BACKSHIFT] cell={tuple(cell)} "
                    f"back={backed:.3f}/{cfg.TARGET_DEADEND_FRONT_BACKSHIFT_M:.3f}m "
                    f"lat={lat:+.3f}m x={-x_mag:+.3f} y={y_cmd:+.3f}"
                )
                last_log = now
            time.sleep(cfg.TARGET_SWEEP_CONTROL_DT)

        self.owner.safe_stop()
        time.sleep(0.06)
        token["result"] = stop_reason
        print(
            f"[TARGET DEADEND BACKSHIFT DONE] cell={tuple(cell)} "
            f"back={token['shift_m']:.3f}m result={stop_reason}"
        )
        return token

    def _return_from_deadend_front_backshift(self, token):
        """Drive forward from the temporary dead-end viewpoint to its anchor."""
        if not isinstance(token, dict):
            return True
        anchor = token.get("anchor_pos")
        target_yaw = token.get("target_yaw")
        if anchor is None or target_yaw is None:
            return True
        if float(token.get("shift_m", 0.0) or 0.0) <= 0.005:
            return True

        self.owner.safe_stop()
        self.owner.pid_straight.reset()

        # Return toward the dead-end wall with ToF facing FRONT.  Odometry is
        # authoritative for the anchor; ToF is only an overshoot/collision veto
        # referenced to the stationary front range measured at that anchor.
        self._goto_target_pose_strict(0.0, cfg.GIMBAL_PITCH_DEG, timeout_sec=1.0)
        anchor_front_tof = token.get("anchor_front_tof_mm")
        try:
            anchor_front_tof = float(anchor_front_tof) if anchor_front_tof is not None else None
        except Exception:
            anchor_front_tof = None

        deadline = time.monotonic() + cfg.TARGET_DEADEND_FRONT_RETURN_TIMEOUT_SEC
        last_log = 0.0

        while self.owner.running and self.running and time.monotonic() < deadline:
            pos = self.owner.current_position()
            if pos is None:
                break
            fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
            lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))

            if (
                abs(fwd) <= cfg.TARGET_DEADEND_FRONT_RETURN_FWD_TOL_M
                and abs(lat) <= cfg.TARGET_DEADEND_FRONT_RETURN_LAT_TOL_M
            ):
                self.owner.safe_stop()
                self.owner.align_heading_stationary(
                    target_yaw, timeout_sec=0.8,
                    tolerance_deg=max(0.55, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.07,
                )
                print(
                    f"[TARGET DEADEND ANCHOR RETURN] fwd={fwd:+.3f}m "
                    f"lat={lat:+.3f}m -> OK"
                )
                return True

            remaining = max(0.0, -fwd)
            x_cmd = cfg.TARGET_DEADEND_FRONT_RETURN_SPEED_MPS
            if remaining <= cfg.TARGET_DEADEND_FRONT_BACKSHIFT_SLOW_ZONE_M:
                x_cmd = cfg.TARGET_DEADEND_FRONT_RETURN_SLOW_MPS

            # If odometry says we have already passed the anchor, stop rather
            # than oscillating toward the dead-end wall.
            if fwd > cfg.TARGET_DEADEND_FRONT_RETURN_FWD_TOL_M:
                break

            # Secondary wall guard.  Do not drive materially closer to the wall
            # than the original anchor even if odometry has drifted.
            front_now = self.owner.latest_tof(fresh=True)
            if (
                anchor_front_tof is not None
                and front_now is not None
                and front_now <= max(55.0, anchor_front_tof - 25.0)
            ):
                print(
                    f"[TARGET DEADEND ANCHOR RETURN GUARD] front={front_now:.0f}mm "
                    f"anchorFront={anchor_front_tof:.0f}mm -> STOP"
                )
                break

            y_cmd = clamp(
                -cfg.TARGET_DEADEND_FRONT_LATERAL_KP * lat,
                -cfg.TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
                +cfg.TARGET_DEADEND_FRONT_LATERAL_MAX_MPS,
            )
            if abs(lat) <= 0.010:
                y_cmd = 0.0
            z_cmd = self.owner.yaw_hold_command(target_yaw, stationary=False)

            if not self.owner.drive_speed_resilient(
                x=+x_cmd, y=y_cmd, z=z_cmd,
                timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="TARGET DEADEND ANCHOR RETURN",
            ):
                break

            now = time.monotonic()
            if now - last_log >= 0.30:
                print(
                    f"[TARGET DEADEND ANCHOR RETURN] fwd={fwd:+.3f}m "
                    f"lat={lat:+.3f}m x={x_cmd:+.3f} y={y_cmd:+.3f}"
                )
                last_log = now
            time.sleep(cfg.TARGET_SWEEP_CONTROL_DT)

        self.owner.safe_stop()
        pos = self.owner.current_position(fresh=False)
        if pos is None:
            self.owner.fault(
                "TARGET DEADEND ANCHOR",
                "position unavailable after FRONT backshift return",
                "remain stopped / DFS will re-align",
            )
            return False
        fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
        lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))
        soft = (
            abs(fwd) <= cfg.TARGET_DEADEND_FRONT_RETURN_SOFT_TOL_M
            and abs(lat) <= cfg.TARGET_DEADEND_FRONT_RETURN_SOFT_TOL_M
        )
        self.owner.align_heading_stationary(
            target_yaw, timeout_sec=0.9,
            tolerance_deg=max(0.70, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.07,
        )
        if soft:
            self.owner.fault(
                "TARGET DEADEND ANCHOR",
                f"soft return fwd={fwd:+.3f}m lat={lat:+.3f}m",
                "accept anchor tolerance and continue",
            )
            return True
        self.owner.fault(
            "TARGET DEADEND ANCHOR",
            f"return residual fwd={fwd:+.3f}m lat={lat:+.3f}m",
            "remain stopped; DFS will re-align before translation",
        )
        return False

    def _return_to_sector_anchor(self, token):
        """Fast return over the just-proven side-shift path."""
        if not isinstance(token, dict):
            return True
        anchor = token.get("anchor_pos")
        target_yaw = token.get("target_yaw")
        if anchor is None or target_yaw is None:
            return True

        self.owner.safe_stop()
        self.owner.pid_straight.reset()
        deadline = time.monotonic() + cfg.TARGET_SIDE_SHIFT_RETURN_TIMEOUT_SEC
        last_log = 0.0
        missing = 0

        while self.owner.running and self.running and time.monotonic() < deadline:
            pos = self.owner.current_position()
            if pos is None:
                break
            lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))
            fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
            remaining = abs(lat)

            if remaining <= cfg.TARGET_SIDE_SHIFT_RETURN_LAT_TOL_M:
                self.owner.safe_stop()
                yaw_err = self.owner.yaw_error_deg(target_yaw)
                # Do not burn ~0.5-0.8 s settling a yaw that is already good; the
                # next sector/move has its own cardinal verification.
                if yaw_err is None or abs(yaw_err) > 1.35:
                    self.owner.align_heading_stationary(
                        target_yaw, timeout_sec=0.40,
                        tolerance_deg=max(0.85, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
                    )
                ok = abs(fwd) <= cfg.TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M
                print(
                    f"[TARGET ANCHOR FAST RETURN] lat={lat:+.3f}m fwd={fwd:+.3f}m "
                    f"-> {'OK' if ok else 'LATERAL_OK/FWD_DRIFT'}"
                )
                return True

            if remaining <= cfg.TARGET_SIDE_SHIFT_RETURN_SLOW_ZONE_M:
                speed = cfg.TARGET_SIDE_SHIFT_RETURN_SLOW_MPS
            elif remaining <= cfg.TARGET_SIDE_SHIFT_RETURN_MED_ZONE_M:
                speed = cfg.TARGET_SIDE_SHIFT_RETURN_MED_SPEED_MPS
            else:
                speed = cfg.TARGET_SIDE_SHIFT_RETURN_SPEED_MPS
            y_cmd = math.copysign(speed, -lat)

            left_cm, right_cm, la_raw, ra_raw = self.owner.read_sharp_cm_fast(
                samples=cfg.TARGET_SIDE_SHIFT_FAST_SHARP_SAMPLES,
                interval_sec=cfg.TARGET_SIDE_SHIFT_FAST_SHARP_INTERVAL_SEC,
            )
            dest_cm = right_cm if y_cmd > 0 else left_cm
            dest_raw = ra_raw if y_cmd > 0 else la_raw
            dest_cal = cfg.RIGHT_CAL if y_cmd > 0 else cfg.LEFT_CAL
            dest_far = (dest_cm is None and sharp_raw_means_far(dest_raw, dest_cal))
            if dest_far:
                missing = 0
            elif dest_cm is None:
                missing += 1
                # Returning to anchor is required; tolerate one missing callback,
                # then reduce rather than blindly keeping 0.38 m/s.
                if missing >= 2:
                    y_cmd = math.copysign(min(abs(y_cmd), 0.10), y_cmd)
            else:
                missing = 0
                if dest_cm <= cfg.TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM:
                    self.owner.safe_stop()
                    print(
                        f"[TARGET ANCHOR RETURN BLOCK] y={y_cmd:+.3f} "
                        f"destSharpFAST={dest_cm:.1f}cm"
                    )
                    break

            z_cmd = self.owner.lateral_yaw_hold_command(target_yaw)
            if not self.owner.drive_speed_resilient(
                x=0.0, y=y_cmd, z=z_cmd,
                timeout=cfg.DRIVE_COMMAND_TIMEOUT, label="TARGET FAST ANCHOR RETURN",
            ):
                break

            now = time.monotonic()
            if now - last_log >= 0.20:
                print(
                    f"[TARGET ANCHOR FAST RETURN] lat={lat:+.3f}m "
                    f"remain={remaining:.3f}m v={abs(y_cmd):.2f}m/s"
                )
                last_log = now
            time.sleep(cfg.TARGET_SIDE_SHIFT_CONTROL_DT)

        self.owner.safe_stop()
        pos = self.owner.current_position(fresh=False)
        if pos is None:
            self.owner.fault("TARGET ANCHOR", "return ended without position", "continue cautiously")
            return False
        lat = float(self.owner.cell_lateral_offset(anchor, pos, target_yaw))
        fwd = float(self.owner.cell_forward_progress(anchor, pos, target_yaw))
        soft = (
            abs(lat) <= cfg.TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M
            and abs(fwd) <= cfg.TARGET_SIDE_SHIFT_RETURN_SOFT_TOL_M
        )
        yaw_err = self.owner.yaw_error_deg(target_yaw)
        if yaw_err is None or abs(yaw_err) > 1.35:
            self.owner.align_heading_stationary(
                target_yaw, timeout_sec=0.40,
                tolerance_deg=max(0.85, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
            )
        if soft:
            self.owner.fault(
                "TARGET ANCHOR",
                f"fast return soft tolerance lat={lat:+.3f}m fwd={fwd:+.3f}m",
                "accept node and continue",
            )
            return True
        self.owner.fault(
            "TARGET ANCHOR",
            f"return residual lat={lat:+.3f}m fwd={fwd:+.3f}m",
            "remain stopped; DFS will re-align before translation",
        )
        return False

    def _candidate_probably_already_fired(self, candidate, detect_yaw):
        """Cheap pre-AIM check for a previously fired physical target.

        Prefer estimated world position using the currently-fresh gimbal ToF.
        Fall back to same-cell absolute bearing only.  Shape is intentionally
        ignored because oblique views can change SQUARE <-> RECT classification.
        """
        color = str(candidate.get("color") or "").upper()
        if not color:
            return None
        current_cell = tuple(self.owner.current)
        abs_bearing = wrap_deg(float(self.owner.heading) * 90.0 + float(detect_yaw))

        # Use the latest ToF without waiting for another multi-sample range read.
        est_xy = None
        tof_now = self.owner.latest_tof(fresh=True)
        if tof_now is not None:
            try:
                est_xy, _ = self._estimate_position(float(tof_now))
            except Exception:
                est_xy = None

        for old in self.targets:
            if str(old.get("color") or "").upper() != color:
                continue
            cand_shape = str(candidate.get("shape") or "").upper()
            old_shape = str(old.get("shape") or "").upper()
            rect_family = {"SQUARE", "RECT_HORIZONTAL", "RECT_VERTICAL"}
            if cand_shape != old_shape and not (cand_shape in rect_family and old_shape in rect_family):
                continue
            if not (
                str(old.get("fire_status") or "").startswith("FIRED_")
                or str(old.get("id") or "") in self.fired_target_ids
            ):
                continue

            old_xy = old.get("estimated_grid_xy")
            if est_xy is not None and old_xy is not None:
                try:
                    if math.hypot(
                        float(est_xy[0]) - float(old_xy[0]),
                        float(est_xy[1]) - float(old_xy[1]),
                    ) <= cfg.TARGET_FAST_FIRED_GRID_DIST:
                        return old
                except Exception:
                    pass

            old_cell = tuple(old.get("source_cell", ()))
            old_b = old.get("detected_bearing_deg_from_north")
            if old_b is None:
                old_b = old.get("bearing_deg_from_north")
            if old_cell == current_cell and old_b is not None:
                if angle_diff_deg(float(old_b), abs_bearing) <= cfg.TARGET_FAST_FIRED_SAME_CELL_BEARING_DEG:
                    return old
        return None

    def _candidate_suppressed_in_sector(self, candidate, yaw_now, handled):
        for item in handled:
            if isinstance(item, dict):
                color=item.get("color"); shape=item.get("shape")
                sweep_yaw=item.get("sweep_yaw"); radius=float(item.get("radius",cfg.TARGET_SWEEP_REPEAT_SUPPRESS_DEG))
            else:
                try:
                    color,shape,sweep_yaw=item[:3]; radius=cfg.TARGET_SWEEP_REPEAT_SUPPRESS_DEG
                except Exception:
                    continue
            if color != candidate.get("color"):
                continue
            if isinstance(item, dict) and item.get("ignore_shape"):
                pass
            elif shape != candidate.get("shape"):
                continue
            if sweep_yaw is not None and yaw_now is not None:
                if angle_diff_deg(float(sweep_yaw), float(yaw_now)) <= float(radius):
                    return True
        return False

    def _continuous_sector_sweep(self, sector_name, start_yaw, end_yaw, fire_lo, fire_hi):
        """Sweep one sector continuously, interrupting only for a stable target."""
        found = []
        handled = []
        interrupts = 0
        direction = +1.0 if end_yaw >= start_yaw else -1.0

        if not self._goto_target_pose_strict(start_yaw, cfg.TARGET_SEARCH_PITCH_DEG, timeout_sec=0.80, allow_soft=True):
            self.owner.fault(
                "TARGET SWEEP",
                f"{sector_name}: cannot reach start yaw={start_yaw:+.1f}",
                "skip sector",
            )
            return found

        time.sleep(cfg.TARGET_SCAN_SETTLE_SEC)
        gate_since = time.monotonic()
        deadline = time.monotonic() + cfg.TARGET_SWEEP_TIMEOUT_SEC
        last_debug = 0.0
        preferred_cone = self._sector_preferred_fire_cone(sector_name)
        pref_text = "NA" if preferred_cone is None else "{:+.0f}..{:+.0f}".format(*preferred_cone)
        print(
            f"[TARGET SWEEP {sector_name}] {start_yaw:+.0f}->{end_yaw:+.0f}deg "
            f"pitch={cfg.TARGET_SEARCH_PITCH_DEG:+.0f} preferred={pref_text} "
            f"softAcq={fire_lo:+.0f}..{fire_hi:+.0f} "
            f"aimEnvelope={self._sector_aim_envelope(sector_name)}"
        )

        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                p_now, y_now = self.owner.current_gimbal_relative()
                if p_now is None or y_now is None:
                    self._stop_gimbal_velocity()
                    time.sleep(0.03)
                    continue

                remaining = direction * (float(end_yaw) - float(y_now))
                if remaining <= 1.5:
                    break

                candidate = self._quick_gate_candidate_since(
                    gate_since, min_frames=cfg.TARGET_SEARCH_QUICK_GATE_FRAMES
                )
                if candidate is not None:
                    if self._candidate_suppressed_in_sector(candidate, y_now, handled):
                        gate_since = time.monotonic()
                    else:
                        candidate = dict(candidate)
                        detect_yaw = float(y_now)
                        old_fired = self._candidate_probably_already_fired(candidate, detect_yaw)
                        if old_fired is not None:
                            print(
                                f"[TARGET FAST SKIP FIRED] {old_fired.get('id')} "
                                f"{candidate.get('color')} detectYaw={detect_yaw:+.1f} "
                                f"shapeNow={candidate.get('shape')} -> keep sweeping"
                            )
                            handled.append({
                                "color": candidate.get("color"),
                                "shape": candidate.get("shape"),
                                "sweep_yaw": detect_yaw,
                                "radius": max(32.0, cfg.TARGET_SWEEP_REPEAT_SUPPRESS_DEG),
                                "ignore_shape": True,
                            })
                            gate_since = time.monotonic()
                            time.sleep(cfg.TARGET_SWEEP_CONTROL_DT)
                            continue

                        self._stop_gimbal_velocity()
                        service_started = time.monotonic()
                        acquisition_cone = (float(fire_lo), float(fire_hi))
                        preferred_cone = self._sector_preferred_fire_cone(sector_name)
                        aim_envelope = self._sector_aim_envelope(sector_name)
                        shoot_intent = self._yaw_in_cone(detect_yaw, acquisition_cone)
                        in_preferred = self._yaw_in_cone(detect_yaw, preferred_cone)
                        candidate["detected_sweep_yaw_deg"] = detect_yaw
                        candidate["detected_sector"] = sector_name
                        candidate["shoot_intent"] = bool(shoot_intent)
                        candidate["preferred_acquisition_cone_deg"] = (
                            None if preferred_cone is None else [float(preferred_cone[0]), float(preferred_cone[1])]
                        )
                        candidate["detected_in_preferred_cone"] = bool(in_preferred)
                        candidate["acquisition_cone_deg"] = [float(fire_lo), float(fire_hi)]
                        candidate["aim_envelope_deg"] = (
                            None if aim_envelope is None
                            else [float(aim_envelope[0]), float(aim_envelope[1])]
                        )

                        locked = None
                        fire_record = None
                        fire_success = False
                        if not shoot_intent:
                            # Outside the narrow +/-10deg acquisition cone: keep
                            # the target in memory, but do not spend time dragging
                            # the crosshair toward it and never authorize fire.
                            print(
                                f"[TARGET MEMORY {sector_name}] yaw={detect_yaw:+.1f} "
                                f"{candidate.get('color')} {candidate.get('shape')} "
                                f"outside softAcq={fire_lo:+.0f}..{fire_hi:+.0f} -> remember only"
                            )
                            tof_mm = self.owner.sample_fresh_tof(samples=cfg.TARGET_FIRE_RANGE_SAMPLES)
                            record = self._record_target(
                                candidate,
                                tof_mm,
                                sector_name=sector_name,
                                detection_yaw_deg=detect_yaw,
                                fire_cone=acquisition_cone,
                                shoot_intent=False,
                                aim_envelope=aim_envelope,
                            )
                            if record is not None and not any(
                                r.get("id") == record.get("id")
                                for r in found if isinstance(r, dict)
                            ):
                                found.append(record)
                        else:
                            acq_kind = "PREFERRED" if in_preferred else "SOFT"
                            print(
                                f"[TARGET HIT {sector_name}] yaw={detect_yaw:+.1f} "
                                f"{candidate.get('color')} {candidate.get('shape')} "
                                f"inside {acq_kind} acquisition -> SHOOT-INTENT LOCK"
                            )
                            aim_lo = None if aim_envelope is None else aim_envelope[0]
                            aim_hi = None if aim_envelope is None else aim_envelope[1]
                            locked = self._aim_and_lock(
                                candidate, aim_yaw_min=aim_lo, aim_yaw_max=aim_hi
                            )
                            if locked is not None:
                                tof_mm = self.owner.sample_fresh_tof(samples=cfg.TARGET_FIRE_RANGE_SAMPLES)
                                record = self._record_target(
                                    locked,
                                    tof_mm,
                                    sector_name=sector_name,
                                    detection_yaw_deg=detect_yaw,
                                    fire_cone=acquisition_cone,
                                    shoot_intent=True,
                                    aim_envelope=aim_envelope,
                                )
                                if record is not None:
                                    if not any(r.get("id") == record.get("id") for r in found if isinstance(r, dict)):
                                        found.append(record)
                                    fire_record = record
                                    fire_success = bool(self._maybe_fire(
                                        record,
                                        sector_name=sector_name,
                                        fire_cone=acquisition_cone,
                                    ))

                        interrupts += 1
                        # Suppress around the ORIGINAL sweep bearing.  Auto-Aim
                        # may rotate tens of degrees, so its final lock yaw must
                        # never become the sweep-progress coordinate.
                        fire_status = (
                            str(fire_record.get("fire_status") or "")
                            if isinstance(fire_record, dict) else ""
                        )
                        if fire_success or fire_status.startswith("FIRED_"):
                            suppress_radius = cfg.TARGET_SWEEP_REPEAT_SUPPRESS_DEG
                        elif fire_status == "DEFER_BAD_ANGLE":
                            # Critical: do NOT burn the good-angle opportunity.
                            # Skip only the immediate duplicate frames around the
                            # same bearing, then allow reacquisition farther along.
                            suppress_radius = cfg.TARGET_BAD_ANGLE_RETRY_SUPPRESS_DEG
                        elif locked is not None:
                            suppress_radius = cfg.TARGET_SWEEP_FAIL_SUPPRESS_DEG
                        elif not shoot_intent:
                            suppress_radius = cfg.TARGET_SWEEP_REPEAT_SUPPRESS_DEG
                        else:
                            suppress_radius = cfg.TARGET_SWEEP_FAIL_SUPPRESS_DEG

                        handled.append({
                            "color": candidate.get("color"),
                            "shape": candidate.get("shape"),
                            "sweep_yaw": detect_yaw,
                            "radius": suppress_radius,
                            # Once it has been locked, perspective may flip
                            # SQUARE <-> RECT while the sweep continues.
                            "ignore_shape": bool(fire_status == "DEFER_BAD_ANGLE"),
                        })

                        if interrupts >= cfg.TARGET_SWEEP_MAX_INTERRUPTS_PER_SECTOR:
                            print(f"[TARGET SWEEP {sector_name}] interrupt cap reached -> continue mission")
                            break

                        # Resume from the DETECTION bearing, not the post-Aim
                        # yaw.  This guarantees monotonic sector progress.
                        resume_yaw = detect_yaw + direction * cfg.TARGET_SWEEP_RESUME_ADVANCE_DEG
                        resume_yaw = clamp(resume_yaw, min(start_yaw, end_yaw), max(start_yaw, end_yaw))
                        if direction * (float(end_yaw) - resume_yaw) <= 1.5:
                            break
                        self._goto_target_pose_strict(
                            resume_yaw, cfg.TARGET_SEARCH_PITCH_DEG, timeout_sec=0.55, allow_soft=True
                        )
                        # Target lock / verification time must not consume the
                        # sector's continuous-sweep watchdog budget.
                        deadline += max(0.0, time.monotonic() - service_started)
                        gate_since = time.monotonic()
                        continue

                # Continuous velocity sweep + closed-loop pitch hold at -10 deg.
                remaining_abs = abs(float(end_yaw) - float(y_now))
                yaw_speed = cfg.TARGET_SWEEP_SPEED_DPS
                if remaining_abs < cfg.TARGET_SWEEP_FINE_ZONE_DEG:
                    yaw_speed = max(
                        cfg.TARGET_SWEEP_MIN_DPS,
                        min(cfg.TARGET_SWEEP_SPEED_DPS, 2.5 * remaining_abs),
                    )
                yaw_speed *= direction

                pitch_err = float(cfg.TARGET_SEARCH_PITCH_DEG) - float(p_now)
                pitch_speed = clamp(2.4 * pitch_err, -20.0, +20.0)
                if abs(pitch_err) <= 0.55:
                    pitch_speed = 0.0
                elif 0.0 < abs(pitch_speed) < 2.5:
                    pitch_speed = math.copysign(2.5, pitch_speed)

                self.owner.gimbal.drive_speed(
                    pitch_speed=float(pitch_speed), yaw_speed=float(yaw_speed)
                )

                now = time.monotonic()
                if now - last_debug >= 0.50:
                    print(
                        f"[TARGET SWEEP {sector_name}] yaw={y_now:+.1f} "
                        f"pitch={p_now:+.1f} speed={yaw_speed:+.1f}dps"
                    )
                    last_debug = now
                time.sleep(cfg.TARGET_SWEEP_CONTROL_DT)
        except Exception as exc:
            self.owner.fault(
                "TARGET SWEEP",
                f"{sector_name}: {type(exc).__name__}: {exc}",
                "stop gimbal / keep target memory / continue DFS",
            )
        finally:
            self._stop_gimbal_velocity()

        # V20.1: do NOT spend another strict moveto/retry at sector end.  The
        # next sector performs its own feedback-driven start positioning.  This
        # avoids the -53.6 -> -45 stall seen in the field log and removes a large
        # chunk of dead time per cell.
        p_end, y_end = self.owner.current_gimbal_relative()
        if y_end is not None and abs(wrap_deg(float(end_yaw) - float(y_end))) > cfg.TARGET_SEARCH_POSE_SOFT_YAW_TOL_DEG:
            self.owner.fault(
                "TARGET SWEEP END",
                f"{sector_name}: ended yaw={float(y_end):+.1f}, expected {float(end_yaw):+.1f}",
                "next sector will reacquire from feedback",
            )
        return found

    def scan_cell(self, cell, force=False):
        if not self.available or not self.owner.running or not self.owner.pose_trusted:
            return []
        cell = tuple(cell)
        if not force and cell in self.scanned_cells:
            return []

        self.owner.safe_stop()
        target_scan_started = time.monotonic()
        self.last_scan_t[cell] = target_scan_started
        found = []
        cell_anchor = self.owner.current_position()
        heading_yaw = self.owner.desired_yaw_for_heading(self.owner.heading)
        if heading_yaw is not None:
            initial_err = self.owner.yaw_error_deg(heading_yaw)
            if initial_err is None or abs(initial_err) > cfg.PRE_MOVE_ALIGN_TOL_DEG:
                self.owner.align_heading_stationary(
                    heading_yaw, timeout_sec=0.65,
                    tolerance_deg=max(0.70, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.05,
                )

        dead_end_front_shift = self._is_confirmed_dead_end_cell(cell)
        dead_end_note = " DEADEND-FRONT-BACKSHIFT=35cm" if dead_end_front_shift else ""
        print(
            f"\n[TARGET 3-SECTOR FAST35] cell={cell} "
            f"pitch={cfg.TARGET_SEARCH_PITCH_DEG:+.0f}deg "
            f"LEFT(-110..-45) FRONT(-45..+45) RIGHT(+45..+110)"
            f"{dead_end_note}"
        )

        # V20.7: no redundant FRONT pose before LEFT.  Side shift overlaps the
        # gimbal move toward that sector's start angle, saving one mechanical
        # positioning wait per cell.

        try:
            for sector_name, start_yaw, end_yaw, slide_dir, fire_lo, fire_hi in cfg.TARGET_RADAR_SECTORS:
                if not self.owner.running or not self.owner.pose_trusted:
                    break

                # Always start a sector with the chassis cardinal corrected.
                if heading_yaw is not None:
                    sector_err = self.owner.yaw_error_deg(heading_yaw)
                    if sector_err is None or abs(sector_err) > cfg.PRE_MOVE_ALIGN_TOL_DEG:
                        self.owner.align_heading_stationary(
                            heading_yaw, timeout_sec=0.60,
                            tolerance_deg=max(0.75, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.045,
                        )

                token = {
                    "direction": None,
                    "anchor_pos": cell_anchor,
                    "target_yaw": heading_yaw,
                    "shift_m": 0.0,
                    "result": "ANCHOR",
                }
                front_backshift_token = None
                if slide_dir is not None:
                    token = self._temporary_side_shift(slide_dir, anchor_pos=cell_anchor, preposition_yaw=start_yaw)
                elif sector_name == "FRONT" and dead_end_front_shift:
                    front_backshift_token = self._temporary_deadend_front_backshift(
                        cell, anchor_pos=cell_anchor
                    )

                try:
                    sector_found = self._continuous_sector_sweep(
                        sector_name, start_yaw, end_yaw, fire_lo, fire_hi
                    )
                    for rec in sector_found:
                        if not any(r.get("id") == rec.get("id") for r in found if isinstance(r, dict)):
                            found.append(rec)
                finally:
                    # LEFT/RIGHT and dead-end FRONT are viewpoint excursions
                    # only.  Always restore the logical node anchor before the
                    # next sector and before any DFS translation.
                    if slide_dir is not None:
                        self._return_to_sector_anchor(token)
                    elif front_backshift_token is not None:
                        self._return_from_deadend_front_backshift(front_backshift_token)

                # Re-check chassis yaw after large gimbal movement.  This
                # counters the small base reaction observed when the turret sweeps.
                if heading_yaw is not None:
                    yaw_before = self.owner.yaw_error_deg(heading_yaw)
                    # V20: turret reaction is usually tiny.  Only stop for a chassis
                    # re-align when it actually exceeds the competition tolerance.
                    if yaw_before is None or abs(yaw_before) > cfg.TARGET_CHASSIS_REALIGN_TRIGGER_DEG:
                        self.owner.align_heading_stationary(
                            heading_yaw, timeout_sec=cfg.TARGET_CHASSIS_REALIGN_TIMEOUT_SEC,
                            tolerance_deg=max(0.90, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.035,
                        )
                    yaw_after = self.owner.yaw_error_deg(heading_yaw)
                    print(
                        f"[TARGET CHASSIS YAW] after {sector_name}: "
                        f"err before={fmt_deg(yaw_before)} after={fmt_deg(yaw_after)}"
                    )

            self.scanned_cells.add(cell)
            target_scan_sec = time.monotonic() - target_scan_started
            self.status = f"3-SECTOR 9MIN cell={cell} found={len(found)} t={target_scan_sec:.1f}s"
            print(f"[PERF TARGET SCAN] cell={cell} sec={target_scan_sec:.2f} found={len(found)}")
            self.save_targets()
            return found
        except Exception as exc:
            self.owner.fault(
                "TARGET SERVICE",
                f"cell={cell}: {type(exc).__name__}: {exc}",
                "stop target service / continue DFS",
            )
            return found
        finally:
            self._stop_gimbal_velocity()
            self._clear_focus_roi()
            try:
                if not self.owner.gimbal_front_safe_for_motion():
                    self.owner.recover_gimbal_front()
            except Exception:
                pass

    def _round2_matching_fired(self, records, hint):
        """Return True when this scan fired the same color/shape expected by a hint."""
        hc = str(hint.get("color") or "").upper()
        hs = str(hint.get("shape") or "").upper()
        for rec in records or []:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("color") or "").upper() != hc:
                continue
            if str(rec.get("shape") or "").upper() != hs:
                continue
            if str(rec.get("fire_status") or "").startswith("FIRED_"):
                return True
        return False

    def scan_round2_hint(self, hint):
        """Replay one proven Round-1 firing direction from the current logical cell.

        Fast path: sweep only +/- ROUND2_NARROW_SWEEP_HALF_DEG around the saved
        detection bearing.  Robust fallback: scan the original full LEFT/FRONT/RIGHT
        sector once if the narrow replay does not reacquire/fire the expected class.
        """
        if not self.available or not self.owner.running or not self.owner.pose_trusted:
            return False
        if not isinstance(hint, dict):
            return False

        sector = str(hint.get("sector") or hint.get("scan_sector") or "FRONT").upper()
        cfg = None
        for row in cfg.TARGET_RADAR_SECTORS:
            if str(row[0]).upper() == sector:
                cfg = row
                break
        if cfg is None:
            self.owner.fault("ROUND2 HINT", f"unknown sector={sector}", "skip hint")
            return False

        sector_name, full_start, full_end, slide_dir, fire_lo, fire_hi = cfg
        cell = tuple(self.owner.current)
        self.owner.safe_stop()
        cell_anchor = self.owner.current_position()
        heading_yaw = self.owner.desired_yaw_for_heading(self.owner.heading)
        if heading_yaw is not None:
            self.owner.align_heading_stationary(
                heading_yaw, timeout_sec=0.8,
                tolerance_deg=max(0.60, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.06,
            )

        saved_yaw = hint.get("detected_sweep_yaw_deg")
        if saved_yaw is None:
            saved_yaw = hint.get("lock_yaw_deg")
        if saved_yaw is None:
            saved_yaw = {"LEFT": -90.0, "FRONT": 0.0, "RIGHT": 90.0}.get(sector_name, 0.0)
        try:
            saved_yaw = float(saved_yaw)
        except Exception:
            saved_yaw = 0.0
        lo_bound, hi_bound = min(float(full_start), float(full_end)), max(float(full_start), float(full_end))
        narrow_lo = clamp(saved_yaw - cfg.ROUND2_NARROW_SWEEP_HALF_DEG, lo_bound, hi_bound)
        narrow_hi = clamp(saved_yaw + cfg.ROUND2_NARROW_SWEEP_HALF_DEG, lo_bound, hi_bound)
        if float(full_end) >= float(full_start):
            narrow_start, narrow_end = narrow_lo, narrow_hi
        else:
            narrow_start, narrow_end = narrow_hi, narrow_lo

        print(
            "\n[ROUND2 HINT] cell={} {} {} sector={} savedYaw={:+.1f} narrow={:+.1f}..{:+.1f}".format(
                cell, hint.get("color"), hint.get("shape"), sector_name,
                saved_yaw, narrow_start, narrow_end,
            )
        )

        self._goto_target_pose_strict(0.0, cfg.TARGET_SEARCH_PITCH_DEG, timeout_sec=0.9)
        token = {
            "direction": None, "anchor_pos": cell_anchor, "target_yaw": heading_yaw,
            "shift_m": 0.0, "result": "ANCHOR",
        }
        front_backshift_token = None
        all_found = []
        try:
            if slide_dir is not None:
                token = self._temporary_side_shift(slide_dir, anchor_pos=cell_anchor)
            elif sector_name == "FRONT" and self._is_confirmed_dead_end_cell(cell):
                front_backshift_token = self._temporary_deadend_front_backshift(
                    cell, anchor_pos=cell_anchor
                )

            found = self._continuous_sector_sweep(
                sector_name, narrow_start, narrow_end, fire_lo, fire_hi
            )
            all_found.extend(found or [])
            success = self._round2_matching_fired(all_found, hint)

            if (
                not success and cfg.ROUND2_FALLBACK_FULL_SECTOR
                and self.owner.running and self.owner.pose_trusted
            ):
                print(
                    "[ROUND2 FALLBACK] {} {} not fired in narrow window -> full {} sector".format(
                        hint.get("color"), hint.get("shape"), sector_name
                    )
                )
                found2 = self._continuous_sector_sweep(
                    sector_name, full_start, full_end, fire_lo, fire_hi
                )
                all_found.extend(found2 or [])
                success = self._round2_matching_fired(all_found, hint)
            return bool(success)
        except Exception as exc:
            self.owner.fault(
                "ROUND2 TARGET",
                "cell={} {} {}: {}: {}".format(
                    cell, hint.get("color"), hint.get("shape"), type(exc).__name__, exc
                ),
                "restore anchor / continue next hint",
            )
            return False
        finally:
            self._stop_gimbal_velocity()
            self._clear_focus_roi()
            try:
                if slide_dir is not None:
                    self._return_to_sector_anchor(token)
                elif front_backshift_token is not None:
                    self._return_from_deadend_front_backshift(front_backshift_token)
            except Exception as exc:
                self.owner.fault(
                    "ROUND2 ANCHOR", f"{type(exc).__name__}: {exc}",
                    "stop target replay / preserve logical pose",
                )
            try:
                if heading_yaw is not None:
                    self.owner.align_heading_stationary(
                        heading_yaw, timeout_sec=0.8,
                        tolerance_deg=max(0.60, cfg.PRE_MOVE_ALIGN_TOL_DEG), settle_sec=0.05,
                    )
                self.owner.gimbal_front_down(force=True)
            except Exception:
                pass

    def _estimate_position(self,tof_mm):
        p,y=self.owner.current_gimbal_relative()
        if tof_mm is None or y is None:
            return None,None
        try:
            pitch=float(p or 0.0)
            planar_mm = tof_center_planar_mm(tof_mm, pitch)
            if planar_mm is None:
                return None,None
            planar = planar_mm / 1000.0
        except Exception:
            return None,None
        bearing=wrap_deg(float(self.owner.heading)*90.0+float(y))
        rad=math.radians(bearing); dx=math.sin(rad); dy=math.cos(rad)
        grid_range=planar/max(cfg.GRID_TILE_M,1e-6)
        cell=tuple(self.owner.current)
        return [float(cell[0])+dx*grid_range,float(cell[1])+dy*grid_range],bearing

    @staticmethod
    def _same_identity(a,b):
        # V20.7: perspective can turn the same SQUARE into a horizontal/vertical
        # RECT, but a CIRCLE should never merge with that rectangular family.
        if a.get("color") != b.get("color"):
            return False
        sa = str(a.get("shape") or "").upper()
        sb = str(b.get("shape") or "").upper()
        if sa == sb:
            return True
        rect_family = {"SQUARE", "RECT_HORIZONTAL", "RECT_VERTICAL"}
        return sa in rect_family and sb in rect_family

    def _find_duplicate(self,record):
        for old in self.targets:
            if not self._same_identity(old,record): continue
            a=old.get("estimated_grid_xy"); b=record.get("estimated_grid_xy")
            if a is not None and b is not None:
                if math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))<=cfg.TARGET_DEDUPE_GRID_DIST:
                    return old
                continue
            if tuple(old.get("source_cell",()))!=tuple(record.get("source_cell",())):
                continue
            ba=old.get("bearing_deg_from_north"); bb=record.get("bearing_deg_from_north")
            if ba is not None and bb is not None and angle_diff_deg(ba,bb)<=cfg.TARGET_DEDUPE_BEARING_DEG:
                return old
        return None

    def _record_target(
        self, candidate, tof_mm, sector_name=None, detection_yaw_deg=None,
        fire_cone=None, shoot_intent=None, aim_envelope=None
    ):
        grid_xy, bearing = self._estimate_position(tof_mm)
        cell = tuple(self.owner.current)
        p, y = self.owner.current_gimbal_relative()
        cone = fire_cone if fire_cone is not None else self._sector_fire_cone(sector_name)
        preferred_cone = self._sector_preferred_fire_cone(sector_name)
        if aim_envelope is None:
            aim_envelope = self._sector_aim_envelope(sector_name)

        detect_yaw = y if detection_yaw_deg is None else float(detection_yaw_deg)
        detected_in_acq = self._yaw_in_cone(detect_yaw, cone)
        detected_in_preferred = self._yaw_in_cone(detect_yaw, preferred_cone)
        if shoot_intent is not None:
            detected_in_acq = bool(shoot_intent) and bool(detected_in_acq)
        in_aim_envelope = self._yaw_in_cone(y, aim_envelope)

        center = candidate.get("center_norm")
        crosshair_centered = bool(candidate.get("crosshair_centered", False))
        crosshair_error = candidate.get("crosshair_error_norm")
        if crosshair_error is None and center is not None:
            try:
                cx, cy = [float(v) for v in center]
                ex = cx - (0.5 + cfg.TARGET_AIM_OFFSET_X)
                ey = cy - (0.5 + cfg.TARGET_AIM_OFFSET_Y)
                crosshair_error = [float(ex), float(ey)]
                if detected_in_acq:
                    crosshair_centered = bool(
                        abs(ex) <= cfg.TARGET_AIM_CENTER_TOL_X
                        and abs(ey) <= cfg.TARGET_AIM_CENTER_TOL_Y
                    )
            except Exception:
                pass

        if detected_in_acq and in_aim_envelope and crosshair_centered:
            fire_status = "PENDING"
        elif not detected_in_acq:
            fire_status = "MEMORY_OUTSIDE_ACQ_CONE"
        elif not in_aim_envelope:
            fire_status = "MEMORY_OUTSIDE_AIM_ENVELOPE"
        else:
            fire_status = "MEMORY_CROSSHAIR_NOT_CENTERED"

        rec = {
            "id": None,
            "kind": "COLOR_SHAPE",
            "color": candidate.get("color"),
            "shape": candidate.get("shape"),
            "source_cell": [int(cell[0]), int(cell[1])],
            "heading": cfg.DIR_NAMES[int(self.owner.heading) % 4],
            "scan_sector": sector_name,
            "detected_sweep_yaw_deg": detect_yaw,
            "detected_bearing_deg_from_north": (
                None if detect_yaw is None
                else wrap_deg(float(self.owner.heading) * 90.0 + float(detect_yaw))
            ),
            "gimbal_yaw_deg": y,
            "gimbal_pitch_deg": p,
            "lock_yaw_deg": y,
            "bearing_deg_from_north": bearing,
            # Legacy field names are retained for compatibility, but in V5
            # fire_cone means the narrow ACQUISITION cone at detection time.
            "fire_cone_deg": None if cone is None else [float(cone[0]), float(cone[1])],
            "in_fire_cone": bool(detected_in_acq),
            "acquisition_cone_deg": None if cone is None else [float(cone[0]), float(cone[1])],
            "preferred_acquisition_cone_deg": (
                None if preferred_cone is None else [float(preferred_cone[0]), float(preferred_cone[1])]
            ),
            "detected_in_preferred_cone": bool(detected_in_preferred),
            "soft_acquisition": bool(detected_in_acq and not detected_in_preferred),
            "detected_in_acquisition_cone": bool(detected_in_acq),
            "shoot_intent": bool(detected_in_acq),
            "aim_envelope_deg": (
                None if aim_envelope is None
                else [float(aim_envelope[0]), float(aim_envelope[1])]
            ),
            "in_aim_envelope": bool(in_aim_envelope),
            "crosshair_centered": bool(crosshair_centered),
            "crosshair_error_norm": crosshair_error,
            "tof_mm": None if tof_mm is None else float(tof_mm),
            "estimated_grid_xy": grid_xy,
            "center_norm": center,
            "bbox_norm": candidate.get("bbox_norm"),
            "score": float(candidate.get("temporal_score", candidate.get("score", 0.0))),
            "color_confidence": float(candidate.get("color_confidence", 0.0)),
            "shape_confidence": float(candidate.get("shape_confidence", 0.0)),
            "median_hsv": candidate.get("median_hsv"),
            "median_lab": candidate.get("median_lab"),
            "confirm_frames": int(candidate.get("confirm_frames", 0)),
            "first_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "last_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "observations": 1,
            "fire_status": fire_status,
        }
        old = self._find_duplicate(rec)
        if old is not None:
            old["last_seen_at"] = rec["last_seen_at"]
            old["observations"] = int(old.get("observations", 1)) + 1
            for k in (
                "tof_mm", "estimated_grid_xy", "bearing_deg_from_north",
                "detected_bearing_deg_from_north",
                "gimbal_yaw_deg", "gimbal_pitch_deg", "lock_yaw_deg",
                "center_norm", "bbox_norm", "score", "color_confidence",
                "shape_confidence", "median_hsv", "median_lab",
                "scan_sector", "detected_sweep_yaw_deg", "fire_cone_deg",
                "in_fire_cone", "acquisition_cone_deg",
                "preferred_acquisition_cone_deg", "detected_in_preferred_cone",
                "soft_acquisition", "detected_in_acquisition_cone", "shoot_intent",
                "aim_envelope_deg", "in_aim_envelope",
                "good_fire_cone_deg", "good_fire_angle",
                "deferred_lock_yaw_deg", "deferred_fire_pose_yaw_deg",
                "crosshair_centered", "crosshair_error_norm",
            ):
                if rec.get(k) is not None:
                    old[k] = rec[k]
            if not str(old.get("fire_status") or "").startswith("FIRED_"):
                old["fire_status"] = fire_status
            return old

        self.target_seq += 1
        rec["id"] = "T{}".format(self.target_seq)
        self.targets.append(rec)
        if detected_in_acq:
            print(
                "[TARGET LOCKED] {} {} {} sector={} detectYaw={} lockYaw={} "
                "acqCone={} aimEnvelope={} crosshair={} ToF={}".format(
                    rec["id"], rec["color"], rec["shape"], sector_name,
                    fmt_deg(detect_yaw), fmt_deg(y), rec["acquisition_cone_deg"],
                    rec["aim_envelope_deg"], rec["crosshair_centered"], rec["tof_mm"],
                )
            )
        else:
            print(
                "[TARGET REMEMBERED] {} {} {} sector={} detectYaw={} outside acqCone={} ToF={}".format(
                    rec["id"], rec["color"], rec["shape"], sector_name,
                    fmt_deg(detect_yaw), rec["acquisition_cone_deg"], rec["tof_mm"],
                )
            )
        return rec

    def _target_upper_point_norm(self, candidate):
        """Return the SAME target's upper-biased aim point in normalized image coordinates."""
        if not isinstance(candidate, dict):
            return None
        bbox = candidate.get("bbox_norm")
        try:
            x, y, w, h = [float(v) for v in bbox]
            if w > 0.0 and h > 0.0:
                return (
                    x + 0.5 * w,
                    y + clamp(float(cfg.TARGET_UPPER_HIT_Y_RATIO), 0.05, 0.50) * h,
                )
        except Exception:
            pass
        # A missing bbox should not invent a shot point.  The center lock remains
        # valid memory, but physical fire waits for a real target box.
        return None

    def _aim_point_velocity(self, point_norm, yaw_min=None, yaw_max=None):
        """Servo an arbitrary visual point onto the camera crosshair during target service."""
        try:
            px, py = [float(v) for v in point_norm]
        except Exception:
            self._stop_gimbal_velocity()
            return False

        desired_x = 0.5 + cfg.TARGET_AIM_OFFSET_X
        desired_y = 0.5 + cfg.TARGET_AIM_OFFSET_Y
        ex = px - desired_x
        ey = py - desired_y

        yaw_speed = 0.0 if abs(ex) <= cfg.TARGET_AIM_DEADBAND_X else clamp(
            ex * cfg.TARGET_AIM_YAW_GAIN_DPS,
            -cfg.TARGET_AIM_YAW_MAX_DPS,
            +cfg.TARGET_AIM_YAW_MAX_DPS,
        )
        pitch_speed = 0.0 if abs(ey) <= cfg.TARGET_AIM_DEADBAND_Y else clamp(
            -ey * cfg.TARGET_CENTER_PITCH_GAIN_DPS,
            -cfg.TARGET_CENTER_PITCH_MAX_DPS,
            +cfg.TARGET_CENTER_PITCH_MAX_DPS,
        )

        p_now, y_now = self.owner.current_gimbal_relative()
        if y_now is not None:
            lo = -float(cfg.TARGET_AIM_YAW_LIMIT_DEG)
            hi = +float(cfg.TARGET_AIM_YAW_LIMIT_DEG)
            if yaw_min is not None:
                lo = max(lo, float(yaw_min))
            if yaw_max is not None:
                hi = min(hi, float(yaw_max))
            if y_now <= lo and yaw_speed < 0:
                yaw_speed = 0.0
            if y_now >= hi and yaw_speed > 0:
                yaw_speed = 0.0

        if p_now is not None:
            if p_now <= cfg.TARGET_CENTER_PITCH_HARD_MIN_DEG and pitch_speed < 0:
                pitch_speed = 0.0
            elif pitch_speed < 0 and p_now < (
                cfg.TARGET_CENTER_PITCH_SOFT_MIN_DEG + cfg.TARGET_CENTER_PITCH_BRAKE_ZONE_DEG
            ):
                headroom = max(0.0, float(p_now) - cfg.TARGET_CENTER_PITCH_HARD_MIN_DEG)
                pitch_speed = max(float(pitch_speed), -max(2.0, 6.0 * headroom))
            if p_now >= cfg.TARGET_CENTER_PITCH_MAX_DEG and pitch_speed > 0:
                pitch_speed = 0.0

        try:
            self.owner.gimbal.drive_speed(
                pitch_speed=float(pitch_speed),
                yaw_speed=float(yaw_speed),
            )
            return True
        except Exception as exc:
            self.owner.fault(
                "TARGET UPPER AIM",
                "drive_speed failed: {}: {}".format(type(exc).__name__, exc),
                "block this shot / keep target memory",
            )
            return False

    def _aim_upper_same_target(self, record, aim_envelope=None):
        """CENTER is already locked; now move the SAME target's upper point onto the crosshair."""
        if not cfg.TARGET_UPPER_AIM_ENABLED:
            record["upper_hit_verified"] = False
            record["upper_hit_disabled"] = True
            return True
        if not isinstance(record, dict):
            return False

        current = dict(record)
        expected = list(current.get("center_norm") or (0.5, 0.5))
        yaw_min = yaw_max = None
        if isinstance(aim_envelope, (tuple, list)) and len(aim_envelope) >= 2:
            yaw_min, yaw_max = float(aim_envelope[0]), float(aim_envelope[1])

        point = self._target_upper_point_norm(current)
        if point is None:
            record["upper_hit_verified"] = False
            record["upper_hit_error"] = "NO_BBOX"
            return False

        chassis_hold_yaw = self.owner.desired_yaw_for_heading(self.owner.heading)
        self.owner.pid_straight.reset()
        deadline = time.monotonic() + float(cfg.TARGET_UPPER_AIM_TIMEOUT_SEC)
        last_seq = -1
        lost_since = None
        stable_frames = 0
        last_err = (None, None)
        self._set_focus_roi(current)

        print(
            "[TARGET UPPER AIM] {} same-target point={:.0f}% from top; "
            "burst={} cadence={:.2f}s".format(
                record.get("id", "?"),
                100.0 * float(cfg.TARGET_UPPER_HIT_Y_RATIO),
                self.owner.get_fire_burst_count(),
                float(cfg.TARGET_FIRE_BURST_INTERVAL_SEC),
            )
        )

        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                if chassis_hold_yaw is not None:
                    try:
                        z_hold = self.owner.yaw_hold_command(chassis_hold_yaw, stationary=False)
                        self.owner.drive_speed_resilient(
                            x=0.0, y=0.0, z=z_hold,
                            timeout=cfg.DRIVE_COMMAND_TIMEOUT,
                            label="TARGET UPPER YAW HOLD",
                        )
                    except Exception:
                        pass

                with self.lock:
                    seq = self.frame_seq
                if seq == last_seq:
                    time.sleep(cfg.TARGET_AIM_POLL_SEC)
                    continue
                last_seq = seq

                latest = self._latest_match(current, expected, allow_blob_fallback=False)
                if latest is None:
                    self._stop_gimbal_velocity()
                    stable_frames = 0
                    if lost_since is None:
                        lost_since = time.monotonic()
                    if time.monotonic() - lost_since > cfg.TARGET_UPPER_AIM_LOST_GRACE_SEC:
                        record["upper_hit_verified"] = False
                        record["upper_hit_error"] = "TARGET_LOST"
                        print("[TARGET UPPER LOST] same target absent too long")
                        return False
                    time.sleep(cfg.TARGET_AIM_POLL_SEC)
                    continue

                lost_since = None
                current.update(latest)
                expected = list(current.get("center_norm", expected))
                self._set_focus_roi(current)
                point = self._target_upper_point_norm(current)
                if point is None:
                    record["upper_hit_verified"] = False
                    record["upper_hit_error"] = "NO_BBOX_DURING_TRACK"
                    return False

                px, py = point
                ex = float(px) - (0.5 + cfg.TARGET_AIM_OFFSET_X)
                ey = float(py) - (0.5 + cfg.TARGET_AIM_OFFSET_Y)
                last_err = (ex, ey)

                if abs(ex) <= cfg.TARGET_UPPER_AIM_TOL_X and abs(ey) <= cfg.TARGET_UPPER_AIM_TOL_Y:
                    stable_frames += 1
                    self._stop_gimbal_velocity()
                else:
                    stable_frames = 0
                    if not self._aim_point_velocity(point, yaw_min=yaw_min, yaw_max=yaw_max):
                        record["upper_hit_verified"] = False
                        record["upper_hit_error"] = "SERVO_FAILED"
                        return False

                self.status = (
                    "UPPER {} {} err=({:+.3f},{:+.3f}) hold={}/{}".format(
                        current.get("color"), current.get("shape"), ex, ey,
                        stable_frames, cfg.TARGET_UPPER_AIM_HOLD_FRAMES,
                    )
                )

                if stable_frames >= cfg.TARGET_UPPER_AIM_HOLD_FRAMES:
                    self._stop_gimbal_velocity()
                    verified = self._verify_exact(current, since_t=time.monotonic())
                    if verified is None:
                        stable_frames = 0
                        continue
                    vpoint = self._target_upper_point_norm(verified)
                    if vpoint is None:
                        stable_frames = 0
                        continue
                    vex = float(vpoint[0]) - (0.5 + cfg.TARGET_AIM_OFFSET_X)
                    vey = float(vpoint[1]) - (0.5 + cfg.TARGET_AIM_OFFSET_Y)
                    if abs(vex) > cfg.TARGET_UPPER_AIM_TOL_X or abs(vey) > cfg.TARGET_UPPER_AIM_TOL_Y:
                        current.update(verified)
                        expected = list(verified.get("center_norm", expected))
                        stable_frames = 0
                        continue

                    current.update(verified)
                    record["upper_hit_verified"] = True
                    record["upper_hit_y_ratio"] = float(cfg.TARGET_UPPER_HIT_Y_RATIO)
                    record["upper_hit_error_norm"] = [float(vex), float(vey)]
                    record["upper_lock_center_norm"] = current.get("center_norm")
                    record["upper_lock_bbox_norm"] = current.get("bbox_norm")
                    p_up, y_up = self.owner.current_gimbal_relative()
                    record["upper_lock_pitch_deg"] = p_up
                    record["upper_lock_yaw_deg"] = y_up
                    print(
                        "[TARGET UPPER LOCK] {} yaw={} pitch={} err=({:+.3f},{:+.3f})".format(
                            record.get("id", "?"), fmt_deg(y_up), fmt_deg(p_up), vex, vey
                        )
                    )
                    return True

                time.sleep(cfg.TARGET_AIM_POLL_SEC)

            ex, ey = last_err
            p_end, y_end = self.owner.current_gimbal_relative()
            record["upper_hit_verified"] = False
            record["upper_hit_error"] = "TIMEOUT"
            print(
                "[TARGET UPPER TIMEOUT] yaw={} pitch={} err=({}, {})".format(
                    fmt_deg(y_end), fmt_deg(p_end),
                    "NA" if ex is None else "{:+.3f}".format(ex),
                    "NA" if ey is None else "{:+.3f}".format(ey),
                )
            )
            return False
        finally:
            self._stop_gimbal_velocity()
            self._clear_focus_roi()
            self.owner.safe_stop()
            if chassis_hold_yaw is not None:
                try:
                    self.owner.align_heading_stationary(
                        chassis_hold_yaw,
                        timeout_sec=0.85,
                        tolerance_deg=max(0.65, cfg.PRE_MOVE_ALIGN_TOL_DEG),
                        settle_sec=0.06,
                    )
                except Exception:
                    pass

    def _compute_fire_solution(self, range_mm, fire_mode, yaw_now, pitch_now):
        """Compute physical muzzle LOS from a CAMERA-centred target.

        ``range_mm`` is the RAW slant range measured at the ToF origin.  The
        sensor is +8 cm forward of robot centre while the muzzle is +15 cm, so
        the muzzle is 7 cm farther forward than the ToF/camera longitudinal
        plane.  We reconstruct the target point from the camera-centred Gimbal
        pitch, then solve the pitch from the muzzle to that same point.

        At pitch~=0 this gives the requested geometry:
            ToF 600 mm -> robot-centre target ~= 680 mm
                       -> muzzle target      ~= 530 mm
        """
        try:
            raw_slant_m = max(0.02, float(range_mm) / 1000.0)
            yaw_now = float(yaw_now)
            pitch_now = float(pitch_now)
        except Exception:
            return None

        pitch_rad = math.radians(pitch_now)
        raw_planar_m = raw_slant_m * abs(math.cos(pitch_rad))

        # Target planar position measured from the chassis/Gimbal yaw centre.
        center_to_target_planar_m = (
            float(cfg.FIRE_TOF_FORWARD_FROM_CENTER_M) + raw_planar_m
        )

        # Camera is assumed to share the ToF forward plane until separately
        # measured.  Because the camera crosshair is already centred, its pitch
        # defines target height relative to the camera optical centre.
        camera_to_target_planar_m = max(0.005,
            center_to_target_planar_m - float(cfg.FIRE_CAMERA_FORWARD_FROM_CENTER_M)
        )
        target_z_from_camera_m = camera_to_target_planar_m * math.tan(pitch_rad)

        # Convert the same target point into the physical muzzle frame.
        muzzle_to_target_planar_m = (
            center_to_target_planar_m - float(cfg.FIRE_MUZZLE_FORWARD_FROM_CENTER_M)
        )
        # A target behind/effectively at the muzzle plane is not a valid ballistic
        # solution even if the ToF itself still returns a number.
        if muzzle_to_target_planar_m <= 0.015:
            return None

        target_z_from_muzzle_m = (
            float(cfg.FIRE_CAMERA_ABOVE_MUZZLE_M) + target_z_from_camera_m
        )
        geometric_fire_pitch = math.degrees(math.atan2(
            target_z_from_muzzle_m, muzzle_to_target_planar_m
        ))

        # Keep the old 'parallax' telemetry meaning: how much pitch changes from
        # the camera-centred pose to the physical muzzle LOS.
        parallax_deg = wrap_deg(geometric_fire_pitch - pitch_now)
        parallax_deg = clamp(parallax_deg, -cfg.FIRE_COMP_MAX_ABS_DEG, +cfg.FIRE_COMP_MAX_ABS_DEG)

        # ToF-vs-camera vertical separation remains diagnostic only.
        tof_to_camera_m = float(cfg.FIRE_TOF_ABOVE_CAMERA_M)
        tof_camera_parallax_deg = math.degrees(math.atan2(
            abs(tof_to_camera_m), max(0.005, camera_to_target_planar_m)
        ))
        tof_camera_parallax_deg *= (
            math.copysign(1.0, tof_to_camera_m) if tof_to_camera_m != 0 else 0.0
        )

        water_extra = float(cfg.WATER_EXTRA_PITCH_DEG) if str(fire_mode).upper() == "WATER" else 0.0
        target_pitch = clamp(
            pitch_now + parallax_deg + water_extra,
            cfg.FIRE_COMP_PITCH_MIN_DEG,
            cfg.FIRE_COMP_PITCH_MAX_DEG,
        )

        muzzle_slant_m = math.hypot(muzzle_to_target_planar_m, target_z_from_muzzle_m)
        center_slant_m = math.hypot(center_to_target_planar_m,
                                   target_z_from_camera_m + float(cfg.FIRE_CAMERA_ABOVE_MUZZLE_M))
        solution = {
            "fire_mode": str(fire_mode).upper(),
            "tof_range_mm": float(range_mm),
            "tof_raw_slant_mm": float(range_mm),
            "tof_raw_planar_mm": float(raw_planar_m * 1000.0),
            "robot_center_to_target_planar_mm": float(center_to_target_planar_m * 1000.0),
            "muzzle_to_target_planar_mm": float(muzzle_to_target_planar_m * 1000.0),
            "camera_to_target_planar_mm": float(camera_to_target_planar_m * 1000.0),
            "camera_lock_yaw_deg": yaw_now,
            "camera_lock_pitch_deg": pitch_now,
            "camera_forward_from_center_m": float(cfg.FIRE_CAMERA_FORWARD_FROM_CENTER_M),
            "tof_forward_from_center_m": float(cfg.FIRE_TOF_FORWARD_FROM_CENTER_M),
            "muzzle_forward_from_center_m": float(cfg.FIRE_MUZZLE_FORWARD_FROM_CENTER_M),
            "muzzle_ahead_of_tof_m": float(cfg.FIRE_MUZZLE_AHEAD_OF_TOF_M),
            "camera_above_muzzle_m": float(cfg.FIRE_CAMERA_ABOVE_MUZZLE_M),
            "tof_above_muzzle_m": float(cfg.FIRE_TOF_ABOVE_MUZZLE_M),
            "tof_above_camera_m": tof_to_camera_m,
            "camera_muzzle_parallax_pitch_deg": float(parallax_deg),
            "tof_camera_parallax_deg": float(tof_camera_parallax_deg),
            "water_extra_pitch_deg": float(water_extra),
            "fire_yaw_deg": yaw_now,
            "fire_pitch_deg": float(target_pitch),
            "muzzle_to_target_slant_m": float(muzzle_slant_m),
            "robot_center_to_target_slant_m": float(center_slant_m),
        }
        return solution

    def _move_to_fire_solution(self, solution):
        if not cfg.FIRE_AIM_GEOMETRY_ENABLED or not isinstance(solution, dict):
            return True
        yaw_target = solution.get("fire_yaw_deg")
        pitch_target = solution.get("fire_pitch_deg")
        if yaw_target is None or pitch_target is None:
            return False

        p0, y0 = self.owner.current_gimbal_relative()
        if p0 is None or y0 is None:
            return False
        if (
            abs(wrap_deg(float(y0) - float(yaw_target))) <= cfg.FIRE_COMP_YAW_TOL_DEG
            and abs(float(p0) - float(pitch_target)) <= cfg.FIRE_COMP_PITCH_TOL_DEG
        ):
            return True

        deadline = time.monotonic() + float(cfg.FIRE_COMP_TIMEOUT_SEC)
        stable = 0
        try:
            while self.owner.running and self.running and time.monotonic() < deadline:
                p_now, y_now = self.owner.current_gimbal_relative()
                if p_now is None or y_now is None:
                    time.sleep(0.025)
                    continue
                ey = wrap_deg(float(yaw_target) - float(y_now))
                ep = float(pitch_target) - float(p_now)
                if abs(ey) <= cfg.FIRE_COMP_YAW_TOL_DEG and abs(ep) <= cfg.FIRE_COMP_PITCH_TOL_DEG:
                    self._stop_gimbal_velocity()
                    stable += 1
                    if stable >= cfg.FIRE_COMP_SETTLE_SAMPLES:
                        return True
                    time.sleep(cfg.FIRE_COMP_SETTLE_SEC)
                    continue
                stable = 0
                ys = clamp(2.8 * ey, -30.0, +30.0)
                ps = clamp(3.0 * ep, -28.0, +28.0)
                if 0.0 < abs(ys) < 3.0:
                    ys = math.copysign(3.0, ys)
                if 0.0 < abs(ps) < 2.5:
                    ps = math.copysign(2.5, ps)
                # Never drive beyond the dedicated firing-pose pitch envelope.
                if float(p_now) <= cfg.FIRE_COMP_PITCH_MIN_DEG and ps < 0:
                    ps = 0.0
                if float(p_now) >= cfg.FIRE_COMP_PITCH_MAX_DEG and ps > 0:
                    ps = 0.0
                self.owner.gimbal.drive_speed(pitch_speed=float(ps), yaw_speed=float(ys))
                time.sleep(0.030)
        except Exception as exc:
            self.owner.fault(
                "FIRE AIM",
                "{}: {}".format(type(exc).__name__, exc),
                "do not fire / resume target sweep",
            )
        finally:
            self._stop_gimbal_velocity()

        p_now, y_now = self.owner.current_gimbal_relative()
        return bool(
            p_now is not None and y_now is not None
            and abs(wrap_deg(float(y_now) - float(yaw_target))) <= 2.5
            and abs(float(p_now) - float(pitch_target)) <= 1.5
        )

    def _maybe_fire(self, record, sector_name=None, fire_cone=None):
        if not cfg.TARGET_AUTO_FIRE_ENABLED or not isinstance(record, dict):
            return False
        tid = str(record.get("id") or "?")

        # GUI target rule: detection/memory still runs for every target, but the
        # blaster is permitted only for one of the 16 selected color/shape pairs.
        if not self.owner.target_allowed(record.get("color"), record.get("shape")):
            record["fire_status"] = "SKIPPED_TARGET_FILTER"
            record["selected_for_fire"] = False
            self.last_fire_event = "{} SKIP: {} {} disabled".format(
                tid, record.get("color"), record.get("shape")
            )
            self.save_targets()
            return False
        record["selected_for_fire"] = True

        if cfg.TARGET_FIRE_ONCE_PER_TARGET and (
            tid in self.fired_target_ids or str(record.get("fire_status") or "").startswith("FIRED_")
        ):
            return False

        sector = str(sector_name or record.get("scan_sector") or "").upper()
        cone = fire_cone if fire_cone is not None else self._sector_fire_cone(sector)
        aim_envelope = self._sector_aim_envelope(sector)
        detect_yaw = record.get("detected_sweep_yaw_deg")
        _p_now, yaw_now = self.owner.current_gimbal_relative()

        # V15 bearing policy:
        #   soft +/-20deg cone is checked at DETECTION time to establish
        #   shoot-intent (the inner +/-10deg remains PREFERRED); after that the crosshair may servo as far as +/-30deg
        #   from the sector axis to put the target centre on the reticle.
        preferred_cone = self._sector_preferred_fire_cone(sector)
        detected_in_acq = self._yaw_in_cone(detect_yaw, cone)
        detected_in_preferred = self._yaw_in_cone(detect_yaw, preferred_cone)
        in_aim_envelope = self._yaw_in_cone(yaw_now, aim_envelope)
        crosshair_centered = bool(record.get("crosshair_centered", False))

        record["scan_sector"] = sector
        record["lock_yaw_deg"] = yaw_now
        record["fire_cone_deg"] = None if cone is None else [float(cone[0]), float(cone[1])]
        record["in_fire_cone"] = bool(detected_in_acq)
        record["acquisition_cone_deg"] = record["fire_cone_deg"]
        record["preferred_acquisition_cone_deg"] = (
            None if preferred_cone is None else [float(preferred_cone[0]), float(preferred_cone[1])]
        )
        record["detected_in_preferred_cone"] = bool(detected_in_preferred)
        record["soft_acquisition"] = bool(detected_in_acq and not detected_in_preferred)
        record["detected_in_acquisition_cone"] = bool(detected_in_acq)
        record["shoot_intent"] = bool(detected_in_acq)
        record["aim_envelope_deg"] = (
            None if aim_envelope is None
            else [float(aim_envelope[0]), float(aim_envelope[1])]
        )
        record["in_aim_envelope"] = bool(in_aim_envelope)

        if not detected_in_acq:
            record["fire_status"] = "MEMORY_OUTSIDE_ACQ_CONE"
            print(
                "[FIRE MEMORY] {}: detectYaw={} outside {} acquisition cone={} -> remember only".format(
                    tid, fmt_deg(detect_yaw), sector, record["acquisition_cone_deg"]
                )
            )
            self.save_targets()
            return False

        if not in_aim_envelope:
            record["fire_status"] = "MEMORY_OUTSIDE_AIM_ENVELOPE"
            print(
                "[FIRE MEMORY] {}: Crosshair needed lockYaw={} outside {} aimEnvelope={} -> remember only".format(
                    tid, fmt_deg(yaw_now), sector, record["aim_envelope_deg"]
                )
            )
            self.save_targets()
            return False

        if not crosshair_centered:
            record["fire_status"] = "MEMORY_CROSSHAIR_NOT_CENTERED"
            print(
                "[FIRE MEMORY] {}: shoot-intent OK but crosshair is not verified at target centre -> remember only".format(tid)
            )
            self.save_targets()
            return False

        # V20.7 GOOD-ANGLE FIRE: wide detection/aim is useful for remembering
        # targets, but a water shot from a steep oblique angle is not.  Gate on
        # the FINAL lock yaw (not detectYaw) so Auto-Aim cannot drag a candidate
        # from a good detection bearing into a bad physical firing bearing.
        good_fire_cone = self._sector_good_fire_cone(sector)
        good_fire_angle = self._yaw_in_cone(yaw_now, good_fire_cone)
        record["good_fire_cone_deg"] = (
            None if good_fire_cone is None
            else [float(good_fire_cone[0]), float(good_fire_cone[1])]
        )
        record["good_fire_angle"] = bool(good_fire_angle)
        if not good_fire_angle:
            record["fire_status"] = "DEFER_BAD_ANGLE"
            record["deferred_lock_yaw_deg"] = yaw_now
            self.last_fire_event = "{} DEFER: BAD ANGLE {}".format(tid, fmt_deg(yaw_now))
            print(
                "[FIRE DEFER ANGLE] {}: {} lockYaw={} outside GOOD={} -> NO SHOT; keep target alive for a better view".format(
                    tid, sector, fmt_deg(yaw_now), record["good_fire_cone_deg"]
                )
            )
            self.save_targets()
            return False

        # V15 CENTER-ONLY: BBOX centre must sit on the camera crosshair for
        # 3 consecutive frames. Upper-target bias is disabled; range/parallax
        # compensation begins immediately after the centre lock.
        if cfg.TARGET_UPPER_AIM_ENABLED:
            if not self._aim_upper_same_target(record, aim_envelope=aim_envelope):
                record["fire_status"] = "BLOCKED_UPPER_AIM_NOT_LOCKED"
                self.last_fire_event = "{} BLOCKED: UPPER AIM".format(tid)
                self.save_targets()
                return False

        self.owner.safe_stop()
        time.sleep(cfg.TARGET_FIRE_RANGE_RECHECK_SEC)
        live = self.owner.sample_fresh_tof(samples=cfg.TARGET_FIRE_RANGE_SAMPLES)
        # RAW ToF remains the competition/safety range gate for backwards
        # compatibility.  Physical centre/muzzle ranges are added after the
        # fire solution is computed and are used for aiming geometry.
        record["fire_range_mm"] = None if live is None else float(live)
        record["fire_range_raw_tof_mm"] = None if live is None else float(live)
        record["fire_range_preferred_mm"] = float(cfg.TARGET_FIRE_PREFERRED_RANGE_MM)
        record["fire_range_limit_mm"] = float(cfg.TARGET_FIRE_MAX_RANGE_MM)

        if live is None:
            record["fire_status"] = "BLOCKED_RANGE_UNKNOWN"
            print("[FIRE BLOCK] {}: no fresh ToF".format(tid))
            return False
        if live > cfg.TARGET_FIRE_MAX_RANGE_MM:
            record["fire_status"] = "WAITING_TOO_FAR"
            print(
                "[FIRE HOLD] {}: {:.0f} mm > {:.0f} mm (remember target, do not fire)".format(
                    tid, live, cfg.TARGET_FIRE_MAX_RANGE_MM
                )
            )
            return False

        # No software minimum range: any valid fresh ToF <=1200 mm is eligible.
        # <=350 mm is the preferred high-confidence zone; 350-1200 mm is a legal
        # fallback and is explicitly labelled as such in telemetry/target memory.
        if sector == "FRONT":
            record["fire_range_zone"] = (
                "FRONT_PREFERRED_LE_350MM"
                if live <= cfg.TARGET_FRONT_SWEET_SPOT_MM
                else "FRONT_FALLBACK_350_TO_1200MM"
            )
        else:
            record["fire_range_zone"] = (
                "SIDE_PREFERRED_LE_350MM" if live <= cfg.TARGET_FIRE_PREFERRED_RANGE_MM
                else "SIDE_FALLBACK_350_TO_1200MM"
            )

        if live <= cfg.TARGET_FIRE_PREFERRED_RANGE_MM:
            print(
                "[FIRE RANGE] {}: {:.0f} mm -> PREFERRED <= {:.0f} mm".format(
                    tid, live, cfg.TARGET_FIRE_PREFERRED_RANGE_MM
                )
            )
        else:
            print(
                "[FIRE RANGE] {}: {:.0f} mm -> FALLBACK (preferred <= {:.0f}, hard <= {:.0f} mm)".format(
                    tid, live, cfg.TARGET_FIRE_PREFERRED_RANGE_MM, cfg.TARGET_FIRE_MAX_RANGE_MM
                )
            )

        # Camera crosshair is centered at this point. Convert that camera LOS
        # into the physical muzzle LOS using fresh ToF range + measured offsets.
        fire_mode = self.owner.get_fire_mode()
        p_lock, y_lock = self.owner.current_gimbal_relative()
        if p_lock is None or y_lock is None:
            record["fire_status"] = "BLOCKED_NO_GIMBAL_POSE"
            return False
        solution = self._compute_fire_solution(live, fire_mode, y_lock, p_lock)
        record["fire_mode"] = fire_mode
        record["fire_aim_solution"] = solution
        self.last_aim_solution = {} if solution is None else dict(solution)
        if solution is not None:
            record["fire_range_center_mm"] = solution.get("robot_center_to_target_planar_mm")
            record["fire_range_muzzle_mm"] = solution.get("muzzle_to_target_planar_mm")
        if solution is None:
            record["fire_status"] = "BLOCKED_NO_FIRE_SOLUTION"
            print("[FIRE BLOCK] {}: could not compute fire geometry".format(tid))
            return False
        if cfg.FIRE_AIM_GEOMETRY_ENABLED:
            print(
                "[FIRE SOLUTION] {} mode={} ToF(raw)={:.0f}mm center={:.0f}mm muzzle={:.0f}mm "
                "camPitch={:+.2f} parallax={:+.2f}deg -> firePitch={:+.2f}deg ToF-Cam={:+.2f}deg".format(
                    tid, fire_mode, live,
                    float(solution["robot_center_to_target_planar_mm"]),
                    float(solution["muzzle_to_target_planar_mm"]),
                    float(solution["camera_lock_pitch_deg"]),
                    float(solution["camera_muzzle_parallax_pitch_deg"]),
                    float(solution["fire_pitch_deg"]),
                    float(solution["tof_camera_parallax_deg"]),
                )
            )
            if not self._move_to_fire_solution(solution):
                record["fire_status"] = "BLOCKED_FIRE_POSE_NOT_REACHED"
                self.last_fire_event = "{} BLOCKED: FIRE POSE".format(tid)
                print("[FIRE BLOCK] {}: compensated fire pose not reached".format(tid))
                return False

        if not cfg.TARGET_REAL_FIRE_ENABLED:
            record["fire_status"] = "READY_DRY_RUN"
            print(
                "[FIRE DRY] {}: detectYaw={} lockYaw={} range={:.0f} mm zone={}".format(
                    tid, fmt_deg(detect_yaw), fmt_deg(yaw_now), live,
                    record["fire_range_zone"],
                )
            )
            return True
        if self.owner.blaster is None:
            record["fire_status"] = "BLOCKED_NO_BLASTER"
            print("[FIRE BLOCK] {}: blaster unavailable".format(tid))
            return False

        fire_mode = self.owner.get_fire_mode()
        if fire_mode == "WATER":
            fire_type = getattr(blaster, "WATER_FIRE", None)
        else:
            fire_mode = "INFRARED"
            fire_type = getattr(blaster, "INFRARED_FIRE", None)
        fire_times = self.owner.get_fire_burst_count()

        if fire_type is None:
            record["fire_status"] = "BLOCKED_FIRE_TYPE_UNAVAILABLE"
            self.last_fire_event = "{} BLOCKED: {} API unavailable".format(tid, fire_mode)
            print("[FIRE BLOCK] {}: {} constant unavailable in RoboMaster SDK".format(tid, fire_mode))
            return False

        # Re-check the ACTUAL yaw after fire-pose compensation.  Pitch
        # compensation should not change yaw, but feedback drift/coupling can.
        # Never let such drift turn a previously good lock into an oblique shot.
        _p_fire_now, y_fire_now = self.owner.current_gimbal_relative()
        good_fire_cone = self._sector_good_fire_cone(sector)
        if not self._yaw_in_cone(y_fire_now, good_fire_cone):
            record["fire_status"] = "DEFER_BAD_ANGLE"
            record["deferred_fire_pose_yaw_deg"] = y_fire_now
            record["good_fire_cone_deg"] = (
                None if good_fire_cone is None
                else [float(good_fire_cone[0]), float(good_fire_cone[1])]
            )
            self.last_fire_event = "{} DEFER: FIRE POSE ANGLE {}".format(tid, fmt_deg(y_fire_now))
            print(
                "[FIRE DEFER ANGLE] {}: finalFireYaw={} outside GOOD={} after compensation -> NO SHOT".format(
                    tid, fmt_deg(y_fire_now), record["good_fire_cone_deg"]
                )
            )
            self.save_targets()
            return False

        actual_shots = 0
        try:
            self._stop_gimbal_velocity()
            for shot_idx in range(int(fire_times)):
                ok = self.owner.blaster.fire(
                    fire_type=fire_type,
                    times=1,
                )
                if ok is False:
                    raise RuntimeError("blaster.fire returned False at burst shot {}".format(shot_idx + 1))
                actual_shots += 1
                print(
                    "[FIRE BURST] {} {} shot {}/{} cadence={:.2f}s".format(
                        tid, fire_mode, shot_idx + 1, fire_times,
                        float(cfg.TARGET_FIRE_BURST_INTERVAL_SEC),
                    )
                )
                if shot_idx + 1 < int(fire_times):
                    time.sleep(float(cfg.TARGET_FIRE_BURST_INTERVAL_SEC))
        except Exception as exc:
            record["fire_status"] = "FIRE_FAILED"
            record["fire_error"] = str(exc)
            record["fire_times_requested"] = int(fire_times)
            record["fire_times_actual"] = int(actual_shots)
            self.last_fire_event = "{} FAILED: {}".format(tid, exc)
            self.owner.fault(
                "FIRE",
                "{}: {}: {}".format(tid, type(exc).__name__, exc),
                "continue DFS / target remains remembered",
            )
            return False

        record["fire_status"] = "FIRED_{}".format(fire_mode)
        record["fired_at"] = datetime.now().isoformat(timespec="milliseconds")
        record["fire_times"] = int(actual_shots)
        record["fire_times_requested"] = int(fire_times)
        record["fire_burst_interval_sec"] = float(cfg.TARGET_FIRE_BURST_INTERVAL_SEC)
        record["fire_mode"] = fire_mode

        # Round-2 hint: save the LOGICAL node anchor plus the exact direction
        # from which a shot was proven to work.  Side/dead-end viewpoint shifts
        # are intentionally not baked into coordinates; round 2 replays the same
        # sector setup so temporary mecanum shifts remain safety-guarded by ToF/Sharp.
        fire_p, fire_y = self.owner.current_gimbal_relative()
        source_cell = tuple(self.owner.current)
        heading_idx = int(self.owner.heading) % 4
        record["fire_anchor"] = {
            "cell": [int(source_cell[0]), int(source_cell[1])],
            "heading_index": heading_idx,
            "heading": cfg.DIR_NAMES[heading_idx],
            "sector": str(record.get("scan_sector") or sector or "FRONT").upper(),
            "detected_sweep_yaw_deg": record.get("detected_sweep_yaw_deg"),
            "lock_yaw_deg": record.get("lock_yaw_deg"),
            "actual_fire_gimbal_yaw_deg": fire_y,
            "actual_fire_gimbal_pitch_deg": fire_p,
            "absolute_bearing_deg_from_north": (
                None if fire_y is None
                else wrap_deg(float(heading_idx) * 90.0 + float(fire_y))
            ),
            "range_mm": float(live),
            "raw_tof_range_mm": float(live),
            "robot_center_range_mm": (
                None if solution is None else solution.get("robot_center_to_target_planar_mm")
            ),
            "muzzle_range_mm": (
                None if solution is None else solution.get("muzzle_to_target_planar_mm")
            ),
            "target_estimated_grid_xy": record.get("estimated_grid_xy"),
            "color": record.get("color"),
            "shape": record.get("shape"),
            "source_target_id": tid,
            "fired_at": record.get("fired_at"),
        }
        self.fired_target_ids.add(tid)
        muzzle_mm = None if solution is None else solution.get("muzzle_to_target_planar_mm")
        self.last_fire_event = "{} FIRED {} x{} ToF={:.0f}mm Muzzle={}".format(
            tid, fire_mode, actual_shots, live,
            "NA" if muzzle_mm is None else "{:.0f}mm".format(float(muzzle_mm)),
        )
        print(
            "[FIRE {}] {} {} {} x{} sector={} detectYaw={} lockYaw={} ToF(raw)={:.0f}mm "
            "center={} muzzle={} zone={}".format(
                fire_mode, tid, record.get("color"), record.get("shape"), actual_shots,
                record.get("scan_sector"), fmt_deg(detect_yaw), fmt_deg(yaw_now), live,
                "NA" if solution is None else "{:.0f}mm".format(float(solution["robot_center_to_target_planar_mm"])),
                "NA" if muzzle_mm is None else "{:.0f}mm".format(float(muzzle_mm)),
                record["fire_range_zone"],
            )
        )
        time.sleep(cfg.TARGET_FIRE_SETTLE_SEC)
        self.save_targets()
        return True

    def save_targets(self):
        try:
            payload={
                "schema":"robomaster_hardened_target_memory","schema_version":2,
                "updated_at":datetime.now().isoformat(timespec="seconds"),
                "selected_fire_mode": self.owner.get_fire_mode(),
                "selected_fire_burst_count": self.owner.get_fire_burst_count(),
                "selected_target_classes": [
                    {"color": c, "shape": sh}
                    for c, sh in self.owner.get_target_selection()
                ],
                "fire_burst_interval_sec": float(cfg.TARGET_FIRE_BURST_INTERVAL_SEC),
                "upper_aim": {
                    "enabled": bool(cfg.TARGET_UPPER_AIM_ENABLED),
                    "hit_y_ratio_from_top": float(cfg.TARGET_UPPER_HIT_Y_RATIO),
                    "hold_frames": int(cfg.TARGET_UPPER_AIM_HOLD_FRAMES),
                },
                "aim_geometry": {
                    "camera_above_muzzle_m": cfg.FIRE_CAMERA_ABOVE_MUZZLE_M,
                    "tof_above_muzzle_m": cfg.FIRE_TOF_ABOVE_MUZZLE_M,
                    "tof_above_camera_m": cfg.FIRE_TOF_ABOVE_CAMERA_M,
                    "tof_forward_from_center_m": cfg.FIRE_TOF_FORWARD_FROM_CENTER_M,
                    "camera_forward_from_center_m": cfg.FIRE_CAMERA_FORWARD_FROM_CENTER_M,
                    "muzzle_forward_from_center_m": cfg.FIRE_MUZZLE_FORWARD_FROM_CENTER_M,
                    "muzzle_ahead_of_tof_m": cfg.FIRE_MUZZLE_AHEAD_OF_TOF_M,
                    "water_extra_pitch_deg": cfg.WATER_EXTRA_PITCH_DEG,
                },
                "detector":"Lab-CLAHE + broad HSV + LAB color confidence + shape + temporal + foam board gate",
                "real_fire_enabled":bool(cfg.TARGET_REAL_FIRE_ENABLED),
                "fire_range_preferred_mm":float(cfg.TARGET_FIRE_PREFERRED_RANGE_MM),
                "fire_range_limit_mm":float(cfg.TARGET_FIRE_MAX_RANGE_MM),
                "scan_pitch_deg":float(cfg.TARGET_SEARCH_PITCH_DEG),
                "radar_sectors":[
                    {
                        "name":name, "sweep_deg":[start,end],
                        "pre_slide":slide,
                        "preferred_acquisition_cone_deg":list(self._sector_preferred_fire_cone(name) or ()),
                        "acquisition_cone_deg":[lo,hi],
                        "aim_envelope_deg":list(self._sector_aim_envelope(name) or ()),
                    }
                    for name,start,end,slide,lo,hi in cfg.TARGET_RADAR_SECTORS
                ],
                "side_shift":{
                    "enabled":bool(cfg.TARGET_SIDE_SHIFT_ENABLED),
                    "viewpoint_shift_m":float(cfg.TARGET_SIDE_SHIFT_DISTANCE_M),
                    "destination_sharp_hard_stop_cm":float(cfg.TARGET_SIDE_SHIFT_DEST_HARD_STOP_CM),
                    "max_shift_m":float(cfg.TARGET_SIDE_SHIFT_MAX_M),
                    "policy":"odometry ~35cm viewpoint shift; fresh Sharp emergency stop",
                },
                "foam_gate":{
                    "enabled":bool(cfg.TARGET_FOAM_GATE_ENABLED),"fail_closed":bool(cfg.TARGET_FOAM_FAIL_CLOSED),
                    "hsv_low":list(cfg.TARGET_FOAM_HSV_LOW),"hsv_high":list(cfg.TARGET_FOAM_HSV_HIGH),
                    "center_margin_px":cfg.TARGET_FOAM_CENTER_MARGIN_PX,"min_bbox_below_frac":cfg.TARGET_FOAM_MIN_BBOX_BELOW_FRAC,
                },
                "count":len(self.targets),"targets":self.targets,
            }
            atomic_write_text(cfg.TARGET_LATEST_JSON,json.dumps(payload,ensure_ascii=False,indent=2))
        except Exception as exc:
            self.owner.fault("TARGET SAVE",f"{type(exc).__name__}: {exc}","continue")
