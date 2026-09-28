#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Infrared blaster control, fire policy enforcement, and range gating."""

import math
import time
from datetime import datetime

from robomaster import blaster as rm_blaster

from config import *
from src.core.geometry import wrap_deg


class BlasterMixin:
    def get_target_fire_policy(self):
        with self.fire_policy_lock:
            policy = dict(self.target_fire_policy)
            policy["selected_color_shapes"] = [
                list(item) for item in self.target_fire_policy.get("selected_color_shapes", [])
            ]
            policy["sdk_labels"] = list(self.target_fire_policy.get("sdk_labels", []))
            return policy


    def set_target_fire_policy(self, policy):
        """Validate and atomically install the operator-selected fire gate."""
        policy = dict(policy or {})
        mode = str(policy.get("mode", "selected")).lower()
        if mode not in ("selected", "all"):
            mode = "selected"

        selected = []
        seen = set()
        for item in policy.get("selected_color_shapes", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            color = str(item[0]).upper().strip()
            shape = str(item[1]).upper().strip()
            key = (color, shape)
            if color not in TARGET_FIRE_COLORS or shape not in TARGET_FIRE_SHAPES:
                continue
            if key in seen:
                continue
            seen.add(key)
            selected.append([color, shape])

        sdk_labels = []
        seen_labels = set()
        for value in policy.get("sdk_labels", []) or []:
            label = str(value).strip()
            if not label:
                continue
            folded = label.casefold()
            if folded in seen_labels:
                continue
            seen_labels.add(folded)
            sdk_labels.append(label)

        clean = {
            "armed": bool(policy.get("armed", True)),
            "mode": mode,
            "fire_type": "infrared",
            "max_range_mm": float(TARGET_FIRE_MAX_RANGE_MM),
            "auto_fire": bool(TARGET_AUTO_FIRE_ENABLED) and bool(policy.get("auto_fire", True)),
            "selected_color_shapes": selected,
            "sdk_enabled": bool(policy.get("sdk_enabled", False)),
            "sdk_labels": sdk_labels,
        }

        if mode == "all":
            clean["sdk_enabled"] = True
            clean["sdk_labels"] = []

        with self.fire_policy_lock:
            self.target_fire_policy = clean

        summary = self._target_fire_policy_summary(clean)
        self.last_fire_event = f"POLICY ARMED: {summary}"
        print(f"[FIRE POLICY] {summary}")
        return clean


    def _target_fire_policy_summary(self, policy):
        if not policy or not policy.get("armed"):
            return "DISARMED"
        range_txt = f"range<={TARGET_FIRE_MAX_RANGE_MM/10.0:.0f}cm"
        if policy.get("mode") == "all":
            return f"ALL TARGETS / IR / {range_txt}"
        selected_n = len(policy.get("selected_color_shapes", []) or [])
        sdk = bool(policy.get("sdk_enabled"))
        labels = list(policy.get("sdk_labels", []) or [])
        sdk_txt = "OFF"
        if sdk:
            sdk_txt = "ALL" if not labels else ",".join(labels)
        return (
            f"SELECTED color/shape={selected_n}, SDK={sdk_txt}, IR, "
            f"range<={TARGET_FIRE_MAX_RANGE_MM/10.0:.0f}cm"
        )


    def target_matches_fire_policy(self, candidate):
        """Return (allowed, reason) without changing robot state."""
        policy = self.get_target_fire_policy()
        if not policy.get("armed"):
            return False, "POLICY_DISARMED"
        if not policy.get("auto_fire", True):
            return False, "AUTO_FIRE_OFF"
        if policy.get("mode") == "all":
            return True, "ALL_TARGETS"

        kind = str(candidate.get("kind", ""))
        if kind == "COLOR_SHAPE":
            key = (
                str(candidate.get("color", "")).upper(),
                str(candidate.get("shape", "")).upper(),
            )
            selected = {
                (str(item[0]).upper(), str(item[1]).upper())
                for item in policy.get("selected_color_shapes", [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            }
            if key in selected:
                return True, "SELECTED_COLOR_SHAPE"
            return False, "COLOR_SHAPE_NOT_SELECTED"

        if kind == "SDK_MARKER":
            if not policy.get("sdk_enabled"):
                return False, "SDK_DISABLED"
            allowed_labels = [
                str(v).casefold() for v in policy.get("sdk_labels", []) if str(v).strip()
            ]
            if not allowed_labels:
                return True, "SDK_ALL_LABELS"
            label = str(candidate.get("label", "")).casefold()
            if label in allowed_labels:
                return True, "SDK_LABEL_SELECTED"
            return False, "SDK_LABEL_NOT_SELECTED"

        return False, "UNKNOWN_TARGET_KIND"


    def _maybe_fire_locked_target(self, record):
        """Final physical firing gate. Called only after a stable target LOCK."""
        if not isinstance(record, dict):
            return False

        tid = str(record.get("id") or "")
        allowed, reason = self.target_matches_fire_policy(record)
        record["fire_policy_match"] = bool(allowed)
        record["fire_policy_reason"] = str(reason)

        if not allowed:
            # Keep an already-fired record as fired when it is re-seen later.
            if record.get("fire_status") != "FIRED_IR":
                record["fire_status"] = "SKIPPED_NOT_SELECTED"
            self.last_fire_event = f"SKIP {tid or '?'}: {reason}"
            print(f"[FIRE BLOCK] {tid or '?'} -> {reason}")
            return False

        if TARGET_FIRE_ONCE_PER_TARGET and (
            tid in self.fired_target_ids or record.get("fire_status") == "FIRED_IR"
        ):
            self.last_fire_event = f"SKIP {tid}: already fired"
            print(f"[FIRE SKIP] {tid}: already fired")
            return False

        # ----------------------------------------------------
        # FINAL ASSIGNMENT RANGE GATE: <= 2 tiles = <= 1200 mm
        # ----------------------------------------------------
        # Do not trust an old target-memory distance.  The target has just been
        # centered/revalidated, so stop and take a fresh median ToF sample at
        # the actual firing pose.  If range is unknown or too far, keep the
        # target in memory but DO NOT shoot; a later closer re-acquisition may
        # still fire it.
        self.stop_chassis()
        time.sleep(max(0.0, float(TARGET_FIRE_RANGE_RECHECK_SEC)))
        live_range_mm = self.sample_tof_median(
            samples=max(3, int(TARGET_FIRE_RANGE_SAMPLES))
        )
        record["fire_range_limit_mm"] = float(TARGET_FIRE_MAX_RANGE_MM)
        record["fire_range_mm"] = (
            None if live_range_mm is None else float(live_range_mm)
        )

        if live_range_mm is None:
            record["fire_status"] = "BLOCKED_RANGE_UNKNOWN"
            record["fire_policy_reason"] = "RANGE_UNKNOWN"
            self.last_fire_event = (
                f"BLOCK {tid or '?'}: range unknown; "
                f"need <= {TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
            )
            print(
                f"[FIRE RANGE BLOCK] {tid or '?'}: NO ToF RANGE -> "
                f"must confirm <= {TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
            )
            return False

        if live_range_mm > TARGET_FIRE_MAX_RANGE_MM:
            record["fire_status"] = "WAITING_TOO_FAR"
            record["fire_policy_reason"] = "TARGET_TOO_FAR"
            self.last_fire_event = (
                f"WAIT {tid or '?'}: {live_range_mm:.0f} mm > "
                f"{TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
            )
            print(
                f"[FIRE RANGE BLOCK] {tid or '?'}: "
                f"{live_range_mm:.0f} mm ({live_range_mm/10.0:.1f} cm) > "
                f"2 tiles / {TARGET_FIRE_MAX_RANGE_MM:.0f} mm -> NO FIRE"
            )
            return False

        # Save the live legal firing distance separately from the mapping ToF.
        print(
            f"[FIRE RANGE OK] {tid or '?'}: "
            f"{live_range_mm:.0f} mm ({live_range_mm/10.0:.1f} cm) <= "
            f"{TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
        )

        if self.blaster is None:
            record["fire_status"] = "BLOCKED_NO_BLASTER"
            self.last_fire_event = f"BLOCK {tid or '?'}: blaster unavailable"
            print(f"[FIRE BLOCK] {tid or '?'}: blaster unavailable")
            return False

        # Target lock already centers/revalidates the candidate. Stop the base
        # once more here so a future caller cannot accidentally fire in motion.
        self.stop_chassis()

        try:
            ok = self.blaster.fire(
                fire_type=rm_blaster.INFRARED_FIRE,
                times=max(1, int(TARGET_INFRARED_SHOTS)),
            )
            if ok is False:
                raise RuntimeError("RoboMaster blaster.fire() returned False")
        except Exception as e:
            record["fire_status"] = "FIRE_FAILED"
            record["fire_error"] = str(e)
            self.last_fire_event = f"FAIL {tid or '?'}: {e}"
            print(f"[FIRE ERROR] {tid or '?'}: {e}")
            return False

        when = datetime.now().isoformat(timespec="milliseconds")
        record["fire_status"] = "FIRED_IR"
        record["fired_at"] = when
        record["fire_type"] = "INFRARED"
        record["fire_range_rule"] = "<=2_tiles_120cm"
        record["fire_times"] = max(1, int(TARGET_INFRARED_SHOTS))
        record["fire_count"] = int(record.get("fire_count", 0)) + 1
        if tid:
            self.fired_target_ids.add(tid)

        desc = (
            f"{record.get('color')} {record.get('shape')}"
            if record.get("kind") == "COLOR_SHAPE"
            else f"SDK {record.get('label')}"
        )
        self.last_fire_event = f"FIRED IR {tid or '?'} {desc}"
        self.target_status_text = f"FIRED IR {tid or '?'} {desc}"
        print(f"[FIRE IR] {tid or '?'} {desc}")
        time.sleep(max(0.0, float(TARGET_FIRE_SETTLE_SEC)))
        return True


