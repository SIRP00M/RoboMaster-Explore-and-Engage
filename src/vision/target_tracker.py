#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Target observation, glimpse memory, lock adjustment, and fusion tracking."""

import json
import math
import statistics
import time
from pathlib import Path

from config import *
from src.core.geometry import wrap_deg, clamp, DIR_VEC


class TargetTrackerMixin:
    def _sampled_pose_candidates(self, window_start, expected_item=None):
        """Collect one bounded target-observation window.

        This intentionally follows the Round-1 style used in the friend's
        launcher: sample several frames, require repeated observations, and
        separate the confidence that is merely interesting from the confidence
        that is strong enough to save.

        Returns:
            {
              "stable_colors": [...],
              "stable_markers": [...],
              "weak": [...],
              "sampled_frames": int,
            }
        """
        window_start = float(window_start)
        deadline = time.monotonic() + TARGET_SAMPLE_WINDOW_TIMEOUT_SEC

        # Wait for fresh camera evidence instead of sleeping a fixed duration.
        # If the camera is slower than expected, timeout keeps DFS responsive.
        while self.running and time.monotonic() < deadline:
            if self._fresh_target_frame_count(window_start) >= TARGET_SAMPLE_FRAMES:
                break
            time.sleep(TARGET_SAMPLE_POLL_SEC)

        with self.vision_lock:
            color_history = [
                (ts, [dict(d) for d in dets])
                for ts, dets in self.detection_history
                if ts >= window_start
            ]
            marker_history = [
                (ts, [dict(m) for m in markers])
                for ts, markers in self.sdk_marker_history
                if ts >= window_start
            ]

        # Cap the color window to the first N fresh camera frames.  This makes
        # "10 samples" deterministic even if the processing thread runs quickly.
        color_history = color_history[:TARGET_SAMPLE_FRAMES]
        sampled_frames = len(color_history)

        # ----- Color/shape tracks -----
        grouped = {}
        for ts, dets in color_history:
            per_frame = {}
            for d in dets:
                if d.get("kind") != "COLOR_SHAPE":
                    continue
                if float(d.get("score", 0.0)) < TARGET_MIN_CONFIDENCE:
                    continue
                if expected_item is not None and not self._match_candidate_to_item(d, expected_item):
                    continue

                key = ("COLOR_SHAPE", d.get("color"), d.get("shape"))
                prev = per_frame.get(key)
                if prev is None or float(d.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    per_frame[key] = d

            for key, d in per_frame.items():
                grouped.setdefault(key, []).append((ts, d))

        stable_colors = []
        weak = []

        for key, items in grouped.items():
            if not items:
                continue

            centers_x = [float(d["center_norm"][0]) for _, d in items]
            centers_y = [float(d["center_norm"][1]) for _, d in items]
            areas = [float(d.get("area_frac_roi", 0.0)) for _, d in items]
            scores = [float(d.get("score", 0.0)) for _, d in items]

            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
            center_std = math.hypot(cx_std, cy_std)

            area_mean = statistics.fmean(areas) if areas else 0.0
            area_std = statistics.pstdev(areas) if len(areas) > 1 else 0.0
            area_cv = area_std / max(area_mean, 1e-6)

            mean_score = statistics.fmean(scores) if scores else 0.0
            best = dict(max(items, key=lambda pair: float(pair[1].get("score", 0.0)))[1])
            best["sample_hits"] = len(items)
            best["sampled_frames"] = sampled_frames
            best["sample_mean_confidence"] = mean_score
            best["confirm_frames"] = len(items)
            best["center_std"] = center_std
            best["area_cv"] = area_cv
            best["temporal_score"] = mean_score
            weak.append(best)

            if (
                len(items) >= TARGET_VERIFY_FRAMES
                and mean_score >= TARGET_SAVE_CONFIDENCE
                and center_std <= TARGET_CONFIRM_MAX_CENTER_STD
                and area_cv <= TARGET_CONFIRM_MAX_AREA_CV
            ):
                stable_colors.append(best)

        # ----- RoboMaster SDK marker tracks -----
        marker_grouped = {}
        for ts, markers in marker_history:
            seen_this_callback = {}
            for m in markers:
                mm = dict(m)
                mm["kind"] = "SDK_MARKER"
                if expected_item is not None and not self._match_candidate_to_item(mm, expected_item):
                    continue
                key = ("SDK_MARKER", str(mm.get("label", "?")))
                seen_this_callback[key] = mm
            for key, m in seen_this_callback.items():
                marker_grouped.setdefault(key, []).append((ts, m))

        stable_markers = []
        for key, items in marker_grouped.items():
            if not items:
                continue

            centers_x = [float(m["center_norm"][0]) for _, m in items]
            centers_y = [float(m["center_norm"][1]) for _, m in items]
            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
            center_std = math.hypot(cx_std, cy_std)

            best = dict(items[-1][1])
            best["sample_hits"] = len(items)
            best["sampled_frames"] = sampled_frames
            best["sample_mean_confidence"] = 1.0
            best["confirm_frames"] = len(items)
            best["temporal_score"] = 1.0
            weak.append(best)

            if (
                len(items) >= TARGET_VERIFY_FRAMES
                and center_std <= TARGET_CONFIRM_MAX_CENTER_STD
            ):
                stable_markers.append(best)

        stable_colors.sort(
            key=lambda d: (
                int(d.get("sample_hits", 0)),
                float(d.get("sample_mean_confidence", 0.0)),
            ),
            reverse=True,
        )
        stable_markers.sort(
            key=lambda d: int(d.get("sample_hits", 0)),
            reverse=True,
        )
        weak.sort(
            key=lambda d: (
                int(d.get("sample_hits", 0)),
                float(d.get("sample_mean_confidence", d.get("score", 0.0))),
            ),
            reverse=True,
        )

        return {
            "stable_colors": stable_colors,
            "stable_markers": stable_markers,
            "weak": weak,
            "sampled_frames": sampled_frames,
        }


    def _observe_target_pose_windowed(
        self,
        cell,
        pose_index,
        total_poses,
        yaw_deg,
        pitch_deg,
        expected_item=None,
        phase="explore",
    ):
        """Observe one stationary gimbal pose using up to three sample windows.

        No evidence -> leave after one window.
        Weak evidence -> HOLD this exact pose and collect another window.
        Stable evidence -> return immediately so target lock can begin.
        """
        hold_started = time.monotonic()
        best_weak = {}

        for window_no in range(1, TARGET_HOLD_MAX_WINDOWS + 1):
            if not self.running:
                break
            if window_no > 1 and (time.monotonic() - hold_started) >= TARGET_HOLD_MAX_SEC:
                break

            self.target_status_text = (
                f"{phase.upper()} SAMPLE cell={tuple(cell)} "
                f"pose={pose_index}/{total_poses} "
                f"window={window_no}/{TARGET_HOLD_MAX_WINDOWS} "
                f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}"
            )
            self.publish_gui_state()

            window_start = time.monotonic()
            result = self._sampled_pose_candidates(
                window_start,
                expected_item=expected_item,
            )

            for cand in result["weak"]:
                ident = self._candidate_identity_key(cand)
                prev = best_weak.get(ident)
                cand_rank = (
                    int(cand.get("sample_hits", 0)),
                    float(cand.get("sample_mean_confidence", cand.get("score", 0.0))),
                )
                prev_rank = (
                    int(prev.get("sample_hits", 0)),
                    float(prev.get("sample_mean_confidence", prev.get("score", 0.0))),
                ) if prev is not None else (-1, -1.0)
                if prev is None or cand_rank > prev_rank:
                    best_weak[ident] = dict(cand)

            if result["stable_markers"] or result["stable_colors"]:
                self.target_status_text = (
                    f"{phase.upper()} VERIFIED cell={tuple(cell)} "
                    f"window={window_no} "
                    f"marker={len(result['stable_markers'])} "
                    f"color={len(result['stable_colors'])}"
                )
                self.publish_gui_state()
                result["weak"] = list(best_weak.values())
                result["windows_used"] = window_no
                return result

            # Nothing even weakly target-like in the complete window:
            # do not waste the remaining hold budget.
            if not result["weak"]:
                result["weak"] = list(best_weak.values())
                result["windows_used"] = window_no
                return result

            # Weak evidence exists.  Stay at this exact yaw/pitch and give it
            # another full sample window instead of immediately sweeping away.
            if (
                window_no < TARGET_HOLD_MAX_WINDOWS
                and (time.monotonic() - hold_started) < TARGET_HOLD_MAX_SEC
            ):
                top = result["weak"][0]
                ident = self._candidate_identity_key(top)
                self.target_status_text = (
                    f"HOLD {ident} "
                    f"{top.get('sample_hits', 0)}/{max(1, result['sampled_frames'])} frames "
                    f"conf={float(top.get('sample_mean_confidence', top.get('score', 0.0))):.2f}"
                )
                self.publish_gui_state()

        return {
            "stable_colors": [],
            "stable_markers": [],
            "weak": list(best_weak.values()),
            "sampled_frames": 0,
            "windows_used": TARGET_HOLD_MAX_WINDOWS,
        }


    def _stable_color_candidates(self, since_t):
        now = time.monotonic()
        cutoff = max(float(since_t), now - TARGET_HISTORY_SEC)

        with self.vision_lock:
            history = [
                (ts, [dict(d) for d in dets])
                for ts, dets in self.detection_history
                if ts >= cutoff
            ]

        grouped = {}

        for ts, dets in history:
            # At most one observation per signature per frame.
            per_frame = {}
            for d in dets:
                if d.get("kind") != "COLOR_SHAPE":
                    continue
                key = (d.get("color"), d.get("shape"))
                prev = per_frame.get(key)
                if prev is None or d.get("score", 0.0) > prev.get("score", 0.0):
                    per_frame[key] = d

            for key, d in per_frame.items():
                grouped.setdefault(key, []).append((ts, d))

        stable = []

        for key, items in grouped.items():
            if len(items) < TARGET_CONFIRM_FRAMES:
                continue

            centers_x = [float(d["center_norm"][0]) for _, d in items]
            centers_y = [float(d["center_norm"][1]) for _, d in items]
            areas = [float(d.get("area_frac_roi", 0.0)) for _, d in items]
            scores = [float(d.get("score", 0.0)) for _, d in items]

            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0
            center_std = math.hypot(cx_std, cy_std)

            area_mean = statistics.fmean(areas) if areas else 0.0
            area_std = statistics.pstdev(areas) if len(areas) > 1 else 0.0
            area_cv = area_std / max(area_mean, 1e-6)

            if center_std > TARGET_CONFIRM_MAX_CENTER_STD:
                continue
            if area_cv > TARGET_CONFIRM_MAX_AREA_CV:
                continue

            # Use the newest observation, but attach temporal evidence.
            newest = dict(items[-1][1])
            newest["confirm_frames"] = len(items)
            newest["center_std"] = center_std
            newest["area_cv"] = area_cv
            newest["temporal_score"] = statistics.fmean(scores)
            stable.append(newest)

        stable.sort(
            key=lambda d: (
                d.get("confirm_frames", 0),
                d.get("temporal_score", 0.0),
            ),
            reverse=True,
        )
        return stable


    def _stable_sdk_markers(self, since_t):
        now = time.monotonic()
        cutoff = max(float(since_t), now - TARGET_HISTORY_SEC)

        with self.vision_lock:
            history = [
                (ts, [dict(m) for m in markers])
                for ts, markers in self.sdk_marker_history
                if ts >= cutoff
            ]

        grouped = {}

        for ts, markers in history:
            seen_this_frame = {}
            for m in markers:
                key = str(m.get("label", "?"))
                seen_this_frame[key] = m
            for key, m in seen_this_frame.items():
                grouped.setdefault(key, []).append((ts, m))

        result = []

        for label, items in grouped.items():
            if len(items) < SDK_MARKER_CONFIRM_FRAMES:
                continue

            centers_x = [float(m["center_norm"][0]) for _, m in items]
            centers_y = [float(m["center_norm"][1]) for _, m in items]
            cx_std = statistics.pstdev(centers_x) if len(centers_x) > 1 else 0.0
            cy_std = statistics.pstdev(centers_y) if len(centers_y) > 1 else 0.0

            if math.hypot(cx_std, cy_std) > TARGET_CONFIRM_MAX_CENTER_STD:
                continue

            newest = dict(items[-1][1])
            newest["confirm_frames"] = len(items)
            result.append(newest)

        result.sort(key=lambda m: m.get("confirm_frames", 0), reverse=True)
        return result


    def _latest_matching_color(self, candidate, expected_center=None):
        key = (candidate.get("color"), candidate.get("shape"))
        now = time.monotonic()

        with self.vision_lock:
            detections = [dict(d) for d in self.latest_detections]
            history = list(self.detection_history)

        if not history or now - history[-1][0] > 0.45:
            return None

        matches = [
            d for d in detections
            if (
                d.get("kind") == "COLOR_SHAPE"
                and (d.get("color"), d.get("shape")) == key
            )
        ]

        if not matches:
            return None

        if expected_center is None:
            return max(matches, key=lambda d: d.get("score", 0.0))

        ex, ey = expected_center
        return min(
            matches,
            key=lambda d: (
                (float(d["center_norm"][0]) - ex) ** 2
                + (float(d["center_norm"][1]) - ey) ** 2
            ),
        )


    def _latest_matching_marker(self, candidate, expected_center=None):
        label = str(candidate.get("label", "?"))
        markers = [
            m for m in self._current_sdk_markers()
            if str(m.get("label", "?")) == label
        ]
        if not markers:
            return None

        if expected_center is None:
            return markers[0]

        ex, ey = expected_center
        return min(
            markers,
            key=lambda m: (
                (float(m["center_norm"][0]) - ex) ** 2
                + (float(m["center_norm"][1]) - ey) ** 2
            ),
        )


    def _transient_target_candidates(self, since_t):
        """Return even 1-frame target evidence seen since *since_t*.

        This deliberately uses the SAME contour/ROI/shape detector as normal
        confirmation; it only relaxes the temporal frame-count requirement.
        Therefore a raw HSV blob that failed the geometric gates never becomes
        a glimpse candidate.
        """
        now = time.monotonic()
        cutoff = max(float(since_t), now - TARGET_HISTORY_SEC)

        with self.vision_lock:
            color_history = [
                (ts, [dict(d) for d in dets])
                for ts, dets in self.detection_history
                if ts >= cutoff
            ]
            marker_history = [
                (ts, [dict(m) for m in markers])
                for ts, markers in self.sdk_marker_history
                if ts >= cutoff
            ]

        grouped = {}

        for ts, dets in color_history:
            per_frame = {}
            for d in dets:
                if d.get("kind") != "COLOR_SHAPE":
                    continue
                key = ("COLOR_SHAPE", d.get("color"), d.get("shape"))
                prev = per_frame.get(key)
                if prev is None or float(d.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    per_frame[key] = d
            for key, d in per_frame.items():
                grouped.setdefault(key, []).append((ts, d))

        for ts, markers in marker_history:
            per_frame = {}
            for m in markers:
                key = ("SDK_MARKER", str(m.get("label", "?")))
                per_frame[key] = m
            for key, m in per_frame.items():
                grouped.setdefault(key, []).append((ts, m))

        result = []
        for key, items in grouped.items():
            if not items:
                continue
            # Prefer the strongest color observation, newest SDK observation.
            if key[0] == "COLOR_SHAPE":
                best = max(items, key=lambda pair: float(pair[1].get("score", 0.0)))[1]
                if float(best.get("score", 0.0)) < TARGET_GLIMPSE_MIN_SCORE:
                    continue
            else:
                best = items[-1][1]

            cand = dict(best)
            cand["kind"] = key[0]
            cand["glimpse_frames"] = len(items)
            cand["glimpse_first_t"] = float(items[0][0])
            cand["glimpse_last_t"] = float(items[-1][0])
            result.append(cand)

        result.sort(
            key=lambda d: (
                int(d.get("glimpse_frames", 0)),
                float(d.get("score", 1.0)),
            ),
            reverse=True,
        )
        return result


    def _candidate_identity_key(self, candidate):
        if candidate.get("kind") == "SDK_MARKER":
            return ("SDK_MARKER", str(candidate.get("label", "?")))
        return (
            "COLOR_SHAPE",
            str(candidate.get("color", "?")),
            str(candidate.get("shape", "?")),
        )


    def _record_target_glimpse(
        self,
        candidate,
        reason="transient",
        gimbal_yaw=None,
        gimbal_pitch=None,
    ):
        """Remember weak target evidence for a targeted return-pass re-scan."""
        if not TARGET_GLIMPSE_ENABLED or candidate is None:
            return None

        cell = tuple(self.current)
        heading = int(self.heading) % 4

        if gimbal_pitch is None or gimbal_yaw is None:
            gp, gy = self.current_gimbal_relative()
            if gimbal_pitch is None:
                gimbal_pitch = gp
            if gimbal_yaw is None:
                gimbal_yaw = gy

        if gimbal_yaw is None:
            return None

        absolute_bearing = wrap_deg(float(heading) * 90.0 + float(gimbal_yaw))
        ident = self._candidate_identity_key(candidate)
        now_iso = datetime.now().isoformat(timespec="milliseconds")

        # Merge repeated brief hits from the same cell and approximately the
        # same physical direction so one noisy sweep cannot create dozens of G#.
        for rec in self.target_glimpses:
            if tuple(rec.get("source_cell", [])) != cell:
                continue
            if tuple(rec.get("identity", [])) != tuple(ident):
                continue
            old_bearing = rec.get("bearing_deg_from_north")
            if old_bearing is None:
                continue
            if self._angle_diff_deg(old_bearing, absolute_bearing) > TARGET_GLIMPSE_BEARING_MERGE_DEG:
                continue

            rec["last_seen_at"] = now_iso
            rec["seen_events"] = int(rec.get("seen_events", 1)) + 1
            rec["max_frames_seen"] = max(
                int(rec.get("max_frames_seen", 0)),
                int(candidate.get("glimpse_frames", candidate.get("confirm_frames", 1))),
            )
            if float(candidate.get("score", 0.0)) >= float(rec.get("score", 0.0)):
                rec["center_norm"] = candidate.get("center_norm")
                rec["bbox_norm"] = candidate.get("bbox_norm")
                rec["score"] = float(candidate.get("score", 0.0))
                rec["gimbal_yaw_deg"] = float(gimbal_yaw)
                rec["gimbal_pitch_deg"] = None if gimbal_pitch is None else float(gimbal_pitch)
                rec["bearing_deg_from_north"] = absolute_bearing
                rec["reason"] = str(reason)
            return rec

        cell_count = sum(
            1 for rec in self.target_glimpses
            if tuple(rec.get("source_cell", [])) == cell
            and not rec.get("resolved_target_id")
        )
        if cell_count >= TARGET_GLIMPSE_MAX_PER_CELL:
            return None

        self.target_glimpse_seq += 1
        rec = {
            "id": f"G{self.target_glimpse_seq}",
            "identity": list(ident),
            "kind": candidate.get("kind"),
            "color": candidate.get("color"),
            "shape": candidate.get("shape"),
            "label": candidate.get("label"),
            "source_cell": [int(cell[0]), int(cell[1])],
            "robot_heading_index": heading,
            "robot_heading": DIR_NAMES[heading],
            "gimbal_yaw_deg": float(gimbal_yaw),
            "gimbal_pitch_deg": None if gimbal_pitch is None else float(gimbal_pitch),
            "bearing_deg_from_north": absolute_bearing,
            "center_norm": candidate.get("center_norm"),
            "bbox_norm": candidate.get("bbox_norm"),
            "score": float(candidate.get("score", 1.0)),
            "max_frames_seen": int(candidate.get("glimpse_frames", candidate.get("confirm_frames", 1))),
            "seen_events": 1,
            "reason": str(reason),
            "resolved_target_id": None,
            "first_seen_at": now_iso,
            "last_seen_at": now_iso,
        }
        self.target_glimpses.append(rec)
        print(
            f"[TARGET GLIMPSE] {rec['id']} cell={cell} "
            f"identity={ident} bearing={absolute_bearing:+.1f} "
            f"pitch={gimbal_pitch} reason={reason}"
        )
        self.target_status_text = (
            f"GLIMPSE {rec['id']} at {cell}; will re-check on return"
        )
        self.save_targets()
        self.publish_gui_state()
        return rec


    def _resolve_matching_glimpses(self, target_record):
        if not target_record:
            return
        ident = self._candidate_identity_key(target_record)
        target_cell = tuple(target_record.get("source_cell", []))
        target_bearing = target_record.get("bearing_deg_from_north")

        for g in self.target_glimpses:
            if g.get("resolved_target_id"):
                continue
            if tuple(g.get("identity", [])) != tuple(ident):
                continue
            if tuple(g.get("source_cell", [])) != target_cell:
                continue
            gb = g.get("bearing_deg_from_north")
            if gb is not None and target_bearing is not None:
                if self._angle_diff_deg(gb, target_bearing) > TARGET_GLIMPSE_BEARING_MERGE_DEG:
                    continue
            g["resolved_target_id"] = target_record.get("id")
            g["resolved_at"] = datetime.now().isoformat(timespec="milliseconds")


    def _match_candidate_to_item(self, candidate, item):
        if candidate is None or item is None:
            return False
        if item.get("kind") == "SDK_MARKER":
            return (
                candidate.get("kind") == "SDK_MARKER"
                and str(candidate.get("label")) == str(item.get("label"))
            )
        return (
            candidate.get("kind") == "COLOR_SHAPE"
            and candidate.get("color") == item.get("color")
            and candidate.get("shape") == item.get("shape")
        )


    def _return_rescan_items_for_cell(self, cell):
        cell = tuple(cell)
        items = []

        if TARGET_RETURN_RESCAN_GLIMPSES:
            for g in self.target_glimpses:
                if g.get("resolved_target_id"):
                    continue
                if tuple(g.get("source_cell", [])) == cell:
                    item = dict(g)
                    item["return_item_type"] = "glimpse"
                    items.append(item)

        if TARGET_RETURN_RESCAN_LOCKED:
            for t in self.detected_targets:
                if tuple(t.get("revisit_cell", t.get("source_cell", []))) == cell:
                    item = dict(t)
                    item["return_item_type"] = "locked"
                    items.append(item)

        # Weak unresolved evidence first, then confirmed targets for refinement.
        items.sort(
            key=lambda x: (
                0 if x.get("return_item_type") == "glimpse" else 1,
                -float(x.get("score", 0.0)),
            )
        )
        return items[:TARGET_RETURN_RESCAN_MAX_ITEMS_PER_CELL]


    def _fusion_update_target(self, target):
        history = list(target.get("observation_history", []))[-TARGET_FUSION_MAX_SAMPLES:]
        target["observation_history"] = history
        valid = [
            h for h in history
            if isinstance(h.get("estimated_grid_xy"), (list, tuple))
            and len(h.get("estimated_grid_xy")) >= 2
        ]
        if not valid:
            target["position_sample_count"] = 0
            return

        xs = [float(h["estimated_grid_xy"][0]) for h in valid]
        ys = [float(h["estimated_grid_xy"][1]) for h in valid]
        mx = statistics.median(xs)
        my = statistics.median(ys)
        target["estimated_grid_xy"] = [mx, my]
        target["position_sample_count"] = len(valid)

        if len(valid) > 1:
            radial = [math.hypot(x - mx, y - my) for x, y in zip(xs, ys)]
            target["position_spread_cells"] = statistics.median(radial)
        else:
            target["position_spread_cells"] = 0.0


    def rescan_targets_on_return(self, cell, phase="return"):
        """Targeted vision check from the return travel heading.

        The chassis is already facing the next confirmed return edge when this
        function is called.  Saved absolute target bearing is converted into a
        NEW relative gimbal yaw, so the same wall is viewed from the reverse
        traversal orientation.  This is useful both for recovering one-frame
        outward-pass glimpses and for adding a second ranged observation to a
        previously locked target.
        """
        if (
            not TARGET_RETURN_RESCAN_ENABLED
            or not self.vision_available
            or not self.running
        ):
            return []

        cell = tuple(cell)
        items = self._return_rescan_items_for_cell(cell)
        if not items:
            return []

        found = []
        self.stop_chassis()
        print(
            f"\n[TARGET RETURN RESCAN] phase={phase} cell={cell} "
            f"heading={DIR_NAMES[self.heading]} items={len(items)}"
        )

        for item in items:
            if not self.running:
                break

            item_id = str(item.get("id", "?"))
            key = (item_id, cell, int(self.heading) % 4, str(phase))
            if key in self.target_return_rescanned:
                continue
            self.target_return_rescanned.add(key)

            bearing = item.get("bearing_deg_from_north")
            if bearing is None:
                continue
            rel_yaw = wrap_deg(float(bearing) - float(self.heading) * 90.0)

            # Gimbal cannot look directly behind the chassis. A future cell or
            # another return heading may bring this bearing inside its side FOV.
            if abs(rel_yaw) > TARGET_LOCK_YAW_LIMIT_DEG:
                print(
                    f"[TARGET RETURN SKIP] {item_id}: relative yaw "
                    f"{rel_yaw:+.1f} outside gimbal limit"
                )
                continue

            base_pitch = item.get("gimbal_pitch_deg")
            try:
                base_pitch = float(base_pitch)
            except Exception:
                base_pitch = TARGET_SWEEP_PITCH_NORMAL_DEG

            candidate_poses = []
            # Re-check at two downward pitch levels, but choose the deep level
            # from the actual yaw.  Around +/-45 deg the deepest legal search
            # pitch is -15 deg; front and near-90 side views retain -22.5 deg.
            pitch_levels = list(self._target_pitch_levels_for_yaw(rel_yaw))
            pitch_levels.sort(key=lambda p: abs(float(p) - base_pitch))
            for pp in pitch_levels:
                candidate_poses.append((rel_yaw, float(pp)))

            # Keep the useful yaw bracket.  The final per-pose clamp below
            # applies the same +/-45 deg protection after the yaw offset.
            yaw_bracket_pitch = float(pitch_levels[0])
            for yoff in TARGET_RETURN_RESCAN_YAW_OFFSETS_DEG[1:]:
                candidate_poses.append((rel_yaw + float(yoff), yaw_bracket_pitch))

            deduped = []
            seen_pose = set()
            for yy, pp in candidate_poses:
                yy = clamp(yy, -TARGET_LOCK_YAW_LIMIT_DEG, TARGET_LOCK_YAW_LIMIT_DEG)
                pp = self._clamp_target_search_pitch(yy, pp)
                pose_key = (round(yy, 1), round(pp, 1))
                if pose_key in seen_pose:
                    continue
                seen_pose.add(pose_key)
                deduped.append((yy, pp))

            confirmed = None
            for pose_no, (yy, pp) in enumerate(deduped, start=1):
                self.target_status_text = (
                    f"RETURN RECHECK {item_id} cell={cell} "
                    f"pose={pose_no}/{len(deduped)} yaw={yy:+.0f} pitch={pp:+.0f}"
                )
                self.publish_gui_state()

                self.gimbal_goto(yy, pitch_deg=pp, force=True)
                time.sleep(TARGET_SWEEP_SETTLE_SEC)

                sampled = self._observe_target_pose_windowed(
                    cell=cell,
                    pose_index=pose_no,
                    total_poses=len(deduped),
                    yaw_deg=yy,
                    pitch_deg=pp,
                    expected_item=item,
                    phase="return",
                )
                stable_markers = list(sampled.get("stable_markers", []))
                stable_colors = list(sampled.get("stable_colors", []))

                preferred_id = (
                    item.get("id")
                    if item.get("return_item_type") == "locked"
                    else None
                )

                if stable_markers:
                    cand = dict(stable_markers[0])
                    cand["kind"] = "SDK_MARKER"
                    confirmed = self._lock_sdk_marker(
                        cand,
                        preferred_target_id=preferred_id,
                        observation_pass="return",
                    )
                elif stable_colors:
                    cand = dict(stable_colors[0])
                    cand["kind"] = "COLOR_SHAPE"
                    confirmed = self._lock_color_candidate(
                        cand,
                        preferred_target_id=preferred_id,
                        observation_pass="return",
                    )

                if confirmed is not None:
                    found.append(confirmed)
                    if item.get("return_item_type") == "glimpse":
                        for g in self.target_glimpses:
                            if g.get("id") == item.get("id"):
                                g["resolved_target_id"] = confirmed.get("id")
                                g["resolved_at"] = datetime.now().isoformat(timespec="milliseconds")
                                break
                    print(
                        f"[TARGET RETURN CONFIRMED] {item_id} -> "
                        f"{confirmed.get('id')} samples="
                        f"{confirmed.get('position_sample_count', 1)}"
                    )
                    break

            if confirmed is None and item.get("return_item_type") == "glimpse":
                # Keep the evidence for logs; do not promote it to a target.
                for g in self.target_glimpses:
                    if g.get("id") == item.get("id"):
                        g["return_recheck_attempts"] = int(g.get("return_recheck_attempts", 0)) + 1
                        break

        self.gimbal_front_down(force=True)
        self.save_targets()
        self.publish_gui_state()
        return found


    def _target_yaw_is_diagonal(self, yaw_deg):
        """True when target-search yaw is in the protected +/-45 deg zone."""
        a = abs(wrap_deg(float(yaw_deg)))
        return TARGET_DIAGONAL_YAW_MIN_DEG <= a <= TARGET_DIAGONAL_YAW_MAX_DEG


    def _target_pitch_levels_for_yaw(self, yaw_deg):
        """Return the two legal target-search pitches for this yaw."""
        deep = (
            TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG
            if self._target_yaw_is_diagonal(yaw_deg)
            else TARGET_SWEEP_PITCH_DEEP_DEG
        )
        return (TARGET_SWEEP_PITCH_NORMAL_DEG, deep)


    def _clamp_target_search_pitch(self, yaw_deg, pitch_deg):
        """Yaw-dependent pitch guard used only by target vision motions.

        Do NOT put this clamp inside generic gimbal_goto(): ToF topology scans
        need their own geometry.  This guard protects target search/re-lock only.
        """
        min_pitch = (
            TARGET_SWEEP_PITCH_DIAGONAL_LOW_DEG
            if self._target_yaw_is_diagonal(yaw_deg)
            else TARGET_LOCK_PITCH_MIN_DEG
        )
        return clamp(
            float(pitch_deg),
            float(min_pitch),
            float(TARGET_LOCK_PITCH_MAX_DEG),
        )


    def _target_lock_adjust(self, center_norm):
        p_now, y_now = self.current_gimbal_relative()
        if p_now is None or y_now is None:
            return False

        cx, cy = [float(v) for v in center_norm]
        ex = cx - 0.5
        ey = cy - 0.5

        if (
            abs(ex) <= TARGET_LOCK_CENTER_TOL_X
            and abs(ey) <= TARGET_LOCK_CENTER_TOL_Y
        ):
            return True

        yaw_step = clamp(
            ex * TARGET_LOCK_HFOV_DEG,
            -TARGET_LOCK_MAX_YAW_STEP_DEG,
            TARGET_LOCK_MAX_YAW_STEP_DEG,
        )
        pitch_step = clamp(
            ey * TARGET_LOCK_VFOV_DEG,
            -TARGET_LOCK_MAX_PITCH_STEP_DEG,
            TARGET_LOCK_MAX_PITCH_STEP_DEG,
        )

        y_target = clamp(
            y_now + yaw_step,
            -TARGET_LOCK_YAW_LIMIT_DEG,
            TARGET_LOCK_YAW_LIMIT_DEG,
        )
        # Positive image Y means target is below center. RoboMaster negative
        # pitch points down, hence subtract pitch_step.
        p_target = self._clamp_target_search_pitch(
            y_target,
            p_now - pitch_step,
        )

        self.gimbal_goto(
            y_target,
            pitch_deg=p_target,
            force=True,
        )
        time.sleep(TARGET_LOCK_SETTLE_SEC)
        return False


    def _estimate_target_position(
        self, cell, heading, gimbal_yaw, tof_mm, gimbal_pitch=0.0
    ):
        if tof_mm is None or gimbal_yaw is None:
            return None, None

        try:
            slant_range_m = float(tof_mm) / 1000.0
            if not math.isfinite(slant_range_m) or slant_range_m <= 0.02:
                return None, None
        except Exception:
            return None, None

        # ToF range follows the gimbal ray.  When the camera/ToF is looking
        # noticeably up or down, using the raw slant range directly as planar
        # XY distance pushes the target too far away on the map.  Project the
        # ray onto the horizontal plane first.
        try:
            pitch_deg = float(gimbal_pitch or 0.0)
        except Exception:
            pitch_deg = 0.0
        horizontal_range_m = slant_range_m * abs(math.cos(math.radians(pitch_deg)))

        bearing_deg = wrap_deg(float(heading) * 90.0 + float(gimbal_yaw))
        rad = math.radians(bearing_deg)
        dx = math.sin(rad)
        dy = math.cos(rad)

        grid_range = horizontal_range_m / max(CELL_LENGTH_M, 1e-6)
        gx = float(cell[0]) + dx * grid_range
        gy = float(cell[1]) + dy * grid_range

        return [gx, gy], bearing_deg


    def _same_target_identity(self, a, b):
        if a.get("kind") != b.get("kind"):
            return False

        if a.get("kind") == "COLOR_SHAPE":
            return (
                a.get("color") == b.get("color")
                and a.get("shape") == b.get("shape")
            )

        if a.get("kind") == "SDK_MARKER":
            return str(a.get("label")) == str(b.get("label"))

        return False


    def _find_duplicate_target(self, record):
        for existing in self.detected_targets:
            if not self._same_target_identity(existing, record):
                continue

            pa = existing.get("estimated_grid_xy")
            pb = record.get("estimated_grid_xy")

            if pa is not None and pb is not None:
                dist = math.hypot(
                    float(pa[0]) - float(pb[0]),
                    float(pa[1]) - float(pb[1]),
                )
                if dist <= TARGET_DEDUPE_GRID_DIST:
                    return existing
                continue

            # Without range, only merge a repeated observation from the same
            # logical cell and roughly the same bearing.
            if TARGET_DEDUPE_SAME_CELL_ONLY_IF_NO_RANGE:
                if tuple(existing.get("source_cell", [])) != tuple(
                    record.get("source_cell", [])
                ):
                    continue

            ba = existing.get("bearing_deg_from_north")
            bb = record.get("bearing_deg_from_north")
            if ba is not None and bb is not None:
                if self._angle_diff_deg(ba, bb) <= TARGET_DEDUPE_BEARING_DEG:
                    return existing

        return None


    def _record_target(
        self,
        candidate,
        tof_mm,
        preferred_target_id=None,
        observation_pass="explore",
    ):
        cell = tuple(self.current)
        heading = int(self.heading) % 4
        gimbal_p, gimbal_y = self.current_gimbal_relative()
        grid_xy, bearing = self._estimate_target_position(
            cell,
            heading,
            gimbal_y,
            tof_mm,
            gimbal_pitch=gimbal_p,
        )

        pos = self.state.get_position()

        observation = {
            "pass": str(observation_pass),
            "at": datetime.now().isoformat(timespec="milliseconds"),
            "source_cell": [int(cell[0]), int(cell[1])],
            "robot_heading_index": heading,
            "gimbal_yaw_deg": gimbal_y,
            "gimbal_pitch_deg": gimbal_p,
            "bearing_deg_from_north": bearing,
            "tof_mm": None if tof_mm is None else float(tof_mm),
            "estimated_grid_xy": grid_xy,
            "score": float(candidate.get("temporal_score", candidate.get("score", 1.0))),
        }

        record = {
            "id": None,
            "kind": candidate.get("kind"),
            "color": candidate.get("color"),
            "shape": candidate.get("shape"),
            "label": candidate.get("label"),
            "source_cell": [int(cell[0]), int(cell[1])],
            # Revisit this confirmed cell during the later attack/reacquire pass.
            # It is more trustworthy than blindly navigating to a projected
            # target point derived from one ToF ray.
            "revisit_cell": [int(cell[0]), int(cell[1])],
            "robot_heading_index": heading,
            "robot_heading": DIR_NAMES[heading],
            "gimbal_yaw_deg": gimbal_y,
            "gimbal_pitch_deg": gimbal_p,
            "reacquire_hint": {
                "heading_index": heading,
                "heading": DIR_NAMES[heading],
                "gimbal_yaw_deg": gimbal_y,
                "gimbal_pitch_deg": gimbal_p,
            },
            "bearing_deg_from_north": bearing,
            "tof_mm": None if tof_mm is None else float(tof_mm),
            "tof_position_is_approximate": True,
            "estimated_grid_xy": grid_xy,
            # Never fire from this memory record alone. The attack pass should
            # return to revisit_cell and reacquire/revalidate the live target.
            "attack_requires_reacquire": True,
            "bbox_norm": candidate.get("bbox_norm"),
            "center_norm": candidate.get("center_norm"),
            "score": float(candidate.get("temporal_score", candidate.get("score", 1.0))),
            "confirm_frames": int(candidate.get("confirm_frames", 0)),
            "chassis_odom_xyz": (
                None if pos is None else [float(v) for v in pos]
            ),
            "first_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "last_seen_at": datetime.now().isoformat(timespec="milliseconds"),
            "observations": 1,
            "observation_history": [observation],
            "position_sample_count": 1 if grid_xy is not None else 0,
            "position_spread_cells": 0.0,
        }

        duplicate = None
        if preferred_target_id is not None:
            duplicate = next(
                (t for t in self.detected_targets if t.get("id") == preferred_target_id),
                None,
            )
        if duplicate is None:
            duplicate = self._find_duplicate_target(record)

        if duplicate is not None:
            duplicate["last_seen_at"] = record["last_seen_at"]
            duplicate["observations"] = int(duplicate.get("observations", 1)) + 1
            duplicate.setdefault("observation_history", []).append(observation)

            # Keep the newest live reacquisition hint, but fuse physical XY
            # from ALL valid ranged observations using a robust median.
            duplicate["reacquire_hint"] = dict(record.get("reacquire_hint", {}))
            duplicate["last_observation_pass"] = str(observation_pass)
            duplicate["bbox_norm"] = record.get("bbox_norm")
            duplicate["center_norm"] = record.get("center_norm")
            duplicate["gimbal_yaw_deg"] = record.get("gimbal_yaw_deg")
            duplicate["gimbal_pitch_deg"] = record.get("gimbal_pitch_deg")
            duplicate["bearing_deg_from_north"] = record.get("bearing_deg_from_north")
            duplicate["tof_mm"] = record.get("tof_mm")
            self._fusion_update_target(duplicate)
            self._resolve_matching_glimpses(duplicate)

            self.target_status_text = (
                f"RE-SEEN {duplicate['id']} "
                f"{duplicate.get('color') or duplicate.get('label')} "
                f"{duplicate.get('shape') or 'SDK'}"
            )
            print(
                f"[TARGET] duplicate -> {duplicate['id']} "
                f"observations={duplicate['observations']}"
            )
            self._maybe_fire_locked_target(duplicate)
            self.save_targets()
            self.publish_gui_state()
            return duplicate

        self.target_id_seq += 1
        record["id"] = f"T{self.target_id_seq}"
        record["last_observation_pass"] = str(observation_pass)
        self.detected_targets.append(record)
        self._resolve_matching_glimpses(record)

        desc = (
            f"{record.get('color')} {record.get('shape')}"
            if record.get("kind") == "COLOR_SHAPE"
            else f"SDK {record.get('label')}"
        )
        self.target_status_text = (
            f"LOCKED {record['id']} {desc} at cell={cell} "
            f"bearing={bearing if bearing is not None else 'n/a'}"
        )

        print("\n[TARGET LOCKED]")
        print(f"  ID       : {record['id']}")
        print(f"  Type     : {desc}")
        print(f"  Cell     : {cell}")
        print(f"  Gimbal   : yaw={gimbal_y} pitch={gimbal_p}")
        print(f"  ToF      : {record['tof_mm']} mm")
        print(f"  Bearing  : {record['bearing_deg_from_north']}")
        print(f"  Grid est.: {record['estimated_grid_xy']}")

        self._maybe_fire_locked_target(record)
        self.save_targets()
        self.publish_gui_state()
        return record


    def _lock_color_candidate(
        self, candidate, preferred_target_id=None, observation_pass="explore"
    ):
        expected = list(candidate.get("center_norm", [0.5, 0.5]))
        current = dict(candidate)

        for _ in range(TARGET_LOCK_MAX_STEPS):
            latest = self._latest_matching_color(current, expected)
            if latest is None:
                self.target_status_text = "candidate lost during color lock"
                print("[TARGET REJECT] color candidate lost during lock")
                return None

            current.update(latest)
            expected = list(current.get("center_norm", expected))
            centered = self._target_lock_adjust(expected)
            if centered:
                break

        latest = self._latest_matching_color(current, expected)
        if latest is None:
            print("[TARGET REJECT] color target not visible after lock")
            return None

        current.update(latest)
        cx, cy = current.get("center_norm", [0.0, 0.0])
        if (
            abs(float(cx) - 0.5) > TARGET_LOCK_CENTER_TOL_X * 1.7
            or abs(float(cy) - 0.5) > TARGET_LOCK_CENTER_TOL_Y * 1.7
        ):
            print("[TARGET REJECT] color target failed final center gate")
            return None

        tof_mm = self.sample_tof_median()
        return self._record_target(
            current,
            tof_mm,
            preferred_target_id=preferred_target_id,
            observation_pass=observation_pass,
        )


    def _lock_sdk_marker(
        self, candidate, preferred_target_id=None, observation_pass="explore"
    ):
        expected = list(candidate.get("center_norm", [0.5, 0.5]))
        current = dict(candidate)

        for _ in range(TARGET_LOCK_MAX_STEPS):
            latest = self._latest_matching_marker(current, expected)
            if latest is None:
                self.target_status_text = "SDK marker lost during lock"
                print("[TARGET REJECT] SDK marker lost during lock")
                return None

            current.update(latest)
            expected = list(current.get("center_norm", expected))
            centered = self._target_lock_adjust(expected)
            if centered:
                break

        latest = self._latest_matching_marker(current, expected)
        if latest is None:
            print("[TARGET REJECT] SDK marker not visible after lock")
            return None

        current.update(latest)
        tof_mm = self.sample_tof_median()
        return self._record_target(
            current,
            tof_mm,
            preferred_target_id=preferred_target_id,
            observation_pass=observation_pass,
        )


    def scan_targets_at_cell(self, cell, force=False):
        """Search/lock/remember targets while the chassis is stationary.

        Each gimbal pose now uses a sampled-window policy:
          1) collect 10 fresh camera frames,
          2) require >=3 consistent frames,
          3) require mean confidence >=0.60 before lock/save,
          4) if confidence >=0.50 appears but is not yet verified, HOLD the
             same pose for another window (max 3 windows / 3 seconds),
          5) unresolved evidence becomes a G# glimpse for return re-check.

        This prevents the old failure mode where the turret swept away just as
        a valid target first entered the ROI.
        """
        if not self.vision_available or not self.running:
            return []

        cell = tuple(cell)
        now = time.monotonic()
        last_t = self.target_last_scan_t.get(cell)

        if (
            not force
            and cell in self.target_scanned_cells
            and last_t is not None
            and now - last_t < TARGET_SCAN_COOLDOWN_SEC
        ):
            return []

        self.stop_chassis()
        self.target_last_scan_t[cell] = now
        found = []

        print(f"\n[TARGET SWEEP] cell={cell} heading={DIR_NAMES[self.heading]}")
        self.target_status_text = f"SEARCHING targets at cell={cell}"
        self.publish_gui_state()

        total_poses = len(TARGET_SWEEP_POSES)
        for pose_index, (yaw_deg, pitch_deg) in enumerate(TARGET_SWEEP_POSES, start=1):
            if not self.running:
                break

            self.target_status_text = (
                f"SEARCHING cell={cell} pose={pose_index}/{total_poses} "
                f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}"
            )
            self.publish_gui_state()

            self.gimbal_goto(
                yaw_deg,
                pitch_deg=pitch_deg,
                force=True,
            )
            time.sleep(TARGET_SWEEP_SETTLE_SEC)

            sampled = self._observe_target_pose_windowed(
                cell=cell,
                pose_index=pose_index,
                total_poses=total_poses,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                expected_item=None,
                phase="explore",
            )

            marker_candidates = list(sampled.get("stable_markers", []))
            color_candidates = list(sampled.get("stable_colors", []))
            weak_candidates = list(sampled.get("weak", []))

            locked_identity = set()
            stable_identity = set()

            # SDK marker identity is more specific than generic red geometry.
            for candidate in marker_candidates[:2]:
                candidate = dict(candidate)
                candidate["kind"] = "SDK_MARKER"
                ident = self._candidate_identity_key(candidate)
                stable_identity.add(ident)

                rec = self._lock_sdk_marker(candidate)
                if rec is not None:
                    found.append(rec)
                    locked_identity.add(ident)
                else:
                    self._record_target_glimpse(
                        candidate,
                        reason="sdk_verified_but_lock_failed",
                        gimbal_yaw=yaw_deg,
                        gimbal_pitch=pitch_deg,
                    )

                # A lock moves the turret. Return to the survey pose before
                # considering another candidate from this sampled window.
                self.gimbal_goto(
                    yaw_deg,
                    pitch_deg=pitch_deg,
                    force=True,
                )
                time.sleep(TARGET_LOCK_SETTLE_SEC)

            for candidate in color_candidates[:3]:
                candidate = dict(candidate)
                candidate["kind"] = "COLOR_SHAPE"
                ident = self._candidate_identity_key(candidate)
                stable_identity.add(ident)

                rec = self._lock_color_candidate(candidate)
                if rec is not None:
                    found.append(rec)
                    locked_identity.add(ident)
                else:
                    self._record_target_glimpse(
                        candidate,
                        reason="color_verified_but_lock_failed",
                        gimbal_yaw=yaw_deg,
                        gimbal_pitch=pitch_deg,
                    )

                self.gimbal_goto(
                    yaw_deg,
                    pitch_deg=pitch_deg,
                    force=True,
                )
                time.sleep(TARGET_LOCK_SETTLE_SEC)

            # Any >=0.50 evidence that used the hold budget but never reached
            # the >=0.60 / 3-frame save gate is deliberately remembered.
            for cand in weak_candidates:
                ident = self._candidate_identity_key(cand)
                if ident in locked_identity or ident in stable_identity:
                    continue
                self._record_target_glimpse(
                    cand,
                    reason=(
                        f"sample_window_unconfirmed:"
                        f"{cand.get('sample_hits', 0)}/"
                        f"{cand.get('sampled_frames', TARGET_SAMPLE_FRAMES)}"
                    ),
                    gimbal_yaw=yaw_deg,
                    gimbal_pitch=pitch_deg,
                )

        self.target_scanned_cells.add(cell)

        # Restore the exact motion-ready ToF posture used by the DFS safety code.
        self.gimbal_front_down(force=True)

        if found:
            unique_ids = sorted({r.get("id") for r in found if r.get("id")})
            self.target_status_text = (
                f"TARGET SWEEP DONE cell={cell}: {', '.join(unique_ids)}"
            )
        else:
            pending_here = sum(
                1 for g in self.target_glimpses
                if tuple(g.get("source_cell", [])) == cell
                and not g.get("resolved_target_id")
            )
            self.target_status_text = (
                f"TARGET SWEEP DONE cell={cell}: none"
                + (f" / {pending_here} glimpse" if pending_here else "")
            )

        self.publish_gui_state()
        return found


    def save_targets(self):
        try:
            TARGET_LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "schema": "robomaster_target_memory",
                "schema_version": 3,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "count": len(self.detected_targets),
                "target_fire_policy": self.get_target_fire_policy(),
                "last_fire_event": self.last_fire_event,
                "targets": self.detected_targets,
                "pending_glimpse_count": sum(
                    1 for g in self.target_glimpses
                    if not g.get("resolved_target_id")
                ),
                "glimpses": self.target_glimpses,
            }

            tmp = TARGET_LATEST_JSON.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, TARGET_LATEST_JSON)
        except Exception as e:
            print(f"[TARGET SAVE WARN] {e}")


