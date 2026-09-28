#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenCV camera vision pipeline, foam wall horizon gate, and contour classification."""

import math
import threading
import time

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

from config import *
from src.core.geometry import wrap_deg, clamp


class VisionMixin:
    def _norm_bbox_iou(self, a, b):
        """IoU for normalized [x, y, w, h] boxes."""
        try:
            ax, ay, aw, ah = [float(v) for v in a]
            bx, by, bw, bh = [float(v) for v in b]
        except Exception:
            return 0.0

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 1e-9 else 0.0


    def _angle_diff_deg(self, a, b):
        return abs(wrap_deg(float(a) - float(b)))


    def _roi_contains_norm_center(self, cx, cy):
        return (
            TARGET_ROI_X_MIN <= float(cx) <= TARGET_ROI_X_MAX
            and TARGET_ROI_Y_MIN <= float(cy) <= TARGET_ROI_Y_MAX
        )


    def _compute_foam_wall_profile(self, frame):
        """Estimate the visible top edge of the white foam maze wall.

        Returns:
            profile_px : np.ndarray shape (frame_width,), y pixel per x column.
                         NaN means the wall top is unknown at that x position.
            coverage   : fraction of the configured target ROI with a valid y.
            components : accepted foam-like component bounding boxes for debug.

        The detector intentionally prefers a conservative FAIL-OPEN policy.
        A missing profile does not by itself reject a target; it only removes
        the extra spatial evidence for that candidate.
        """
        if (
            not TARGET_FOAM_GATE_ENABLED
            or cv2 is None
            or np is None
            or frame is None
        ):
            return None, 0.0, []

        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 1 or frame_w <= 1:
            return None, 0.0, []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(TARGET_FOAM_HSV_LOW, dtype=np.uint8),
            np.array(TARGET_FOAM_HSV_HIGH, dtype=np.uint8),
        )

        open_k = max(1, int(TARGET_FOAM_OPEN_KERNEL))
        close_k = max(1, int(TARGET_FOAM_CLOSE_KERNEL))
        if open_k % 2 == 0:
            open_k += 1
        if close_k % 2 == 0:
            close_k += 1

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((open_k, open_k), dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        profile = np.full(frame_w, np.nan, dtype=np.float32)
        components = []
        min_bottom_y = float(frame_h) * float(TARGET_FOAM_COMPONENT_MIN_BOTTOM_FRAC)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, bw, bh = cv2.boundingRect(contour)

            if area < TARGET_FOAM_MIN_COMPONENT_AREA_PX:
                continue
            if bw < TARGET_FOAM_MIN_COMPONENT_WIDTH_PX:
                continue
            if bh < TARGET_FOAM_MIN_COMPONENT_HEIGHT_PX:
                continue
            if (y + bh) < min_bottom_y:
                # Ceiling panels/lights may be white, but they do not extend
                # into the maze-wall/floor part of the image.
                continue

            components.append({
                "bbox_px": [int(x), int(y), int(bw), int(bh)],
                "area_px": area,
            })

            # Fill only this component in a local mask. RETR_EXTERNAL plus
            # filled contour also closes small holes caused by colored targets
            # mounted on the foam face, preserving the true wall top.
            local = np.zeros((bh, bw), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= x
            shifted[:, :, 1] -= y
            cv2.drawContours(local, [shifted], -1, 255, thickness=-1)

            for local_x in range(bw):
                ys = np.flatnonzero(local[:, local_x])
                if ys.size == 0:
                    continue

                px = x + local_x
                top_y = float(y + int(ys[0]))
                if not math.isfinite(float(profile[px])) or top_y < float(profile[px]):
                    profile[px] = top_y

        # Bridge only SHORT missing spans. This repairs seams/sign holes without
        # inventing a wall across a wide corridor opening.
        valid_idx = np.flatnonzero(np.isfinite(profile))
        max_gap = max(0, int(TARGET_FOAM_PROFILE_MAX_INTERP_GAP_PX))
        if valid_idx.size >= 2 and max_gap > 0:
            for left_i, right_i in zip(valid_idx[:-1], valid_idx[1:]):
                gap = int(right_i - left_i - 1)
                if gap <= 0 or gap > max_gap:
                    continue
                profile[left_i:right_i + 1] = np.linspace(
                    float(profile[left_i]),
                    float(profile[right_i]),
                    int(right_i - left_i + 1),
                    dtype=np.float32,
                )

        # Robust 1-D median smoothing while preserving unknown regions.
        radius = max(0, int(TARGET_FOAM_PROFILE_MEDIAN_RADIUS_PX))
        if radius > 0 and np.isfinite(profile).any():
            src = profile.copy()
            smooth = profile.copy()
            for px in np.flatnonzero(np.isfinite(src)):
                lo = max(0, int(px) - radius)
                hi = min(frame_w, int(px) + radius + 1)
                vals = src[lo:hi]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    smooth[int(px)] = float(np.median(vals))
            profile = smooth

        roi_x1 = max(0, min(frame_w - 1, int(round(TARGET_ROI_X_MIN * frame_w))))
        roi_x2 = max(roi_x1 + 1, min(frame_w, int(round(TARGET_ROI_X_MAX * frame_w))))
        roi_slice = profile[roi_x1:roi_x2]
        coverage = (
            float(np.count_nonzero(np.isfinite(roi_slice))) / float(max(1, roi_slice.size))
        )

        return profile, coverage, components


    def _foam_profile_snapshot(self):
        """Thread-safe copy of the latest wall-top estimate."""
        with self.vision_lock:
            profile = self.latest_foam_profile_px
            if profile is not None:
                profile = profile.copy()
            return (
                profile,
                float(self.latest_foam_profile_t),
                float(self.latest_foam_profile_coverage),
                [dict(c) for c in self.latest_foam_components],
                tuple(self.latest_frame_shape) if self.latest_frame_shape else None,
            )


    def _foam_wall_y_at_px(self, profile_px, x_px):
        if profile_px is None or np is None:
            return None

        try:
            width = int(len(profile_px))
            x = int(round(float(x_px)))
        except Exception:
            return None

        if width <= 0 or x < 0 or x >= width:
            return None

        radius = max(0, int(TARGET_FOAM_PROFILE_SAMPLE_RADIUS_PX))
        lo = max(0, x - radius)
        hi = min(width, x + radius + 1)
        vals = np.asarray(profile_px[lo:hi], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.median(vals))


    def _target_passes_foam_gate_px(
        self,
        frame_w,
        frame_h,
        cx_px,
        cy_px,
        bbox_px,
        profile_px=None,
        profile_t=None,
    ):
        """Return (pass_bool, local_wall_y_px, reason, below_fraction)."""
        if not TARGET_FOAM_GATE_ENABLED:
            return True, None, "foam-gate-disabled", 1.0

        if profile_px is None:
            (
                profile_px,
                snap_t,
                coverage,
                _components,
                _shape,
            ) = self._foam_profile_snapshot()
            profile_t = snap_t
        else:
            coverage = None

        now = time.monotonic()
        if profile_t is not None and profile_t > 0.0:
            if now - float(profile_t) > TARGET_FOAM_PROFILE_TTL_SEC:
                if TARGET_FOAM_FAIL_CLOSED:
                    return False, None, "foam-profile-stale", 0.0
                return True, None, "foam-profile-stale-fail-open", 1.0

        if profile_px is None:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-profile-missing", 0.0
            return True, None, "foam-profile-missing-fail-open", 1.0

        # A very low global coverage means the frame probably does not expose a
        # usable foam horizon. Do not trust a tiny accidental white component.
        if coverage is not None and coverage < TARGET_FOAM_MIN_ROI_COVERAGE:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-profile-low-coverage", 0.0
            return True, None, "foam-profile-low-coverage-fail-open", 1.0

        wall_y = self._foam_wall_y_at_px(profile_px, cx_px)
        if wall_y is None:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-local-profile-missing", 0.0
            return True, None, "foam-local-profile-missing-fail-open", 1.0

        try:
            x, y, bw, bh = [float(v) for v in bbox_px]
            cy = float(cy_px)
        except Exception:
            if TARGET_FOAM_FAIL_CLOSED:
                return False, wall_y, "foam-invalid-bbox", 0.0
            return True, wall_y, "foam-invalid-bbox-fail-open", 1.0

        bh = max(1.0, bh)
        cutoff_y = float(wall_y) + float(TARGET_FOAM_CENTER_MARGIN_PX)
        bbox_bottom = y + bh
        below_height = max(0.0, bbox_bottom - max(y, cutoff_y))
        below_fraction = clamp(below_height / bh, 0.0, 1.0)

        if cy < cutoff_y:
            return False, wall_y, "candidate-centre-above-foam", below_fraction

        if below_fraction < TARGET_FOAM_MIN_BBOX_BELOW_FRAC:
            return False, wall_y, "candidate-mostly-above-foam", below_fraction

        return True, wall_y, "foam-pass", below_fraction


    def _target_passes_foam_gate_norm(self, cx, cy, bbox_norm):
        """Foam gate for RoboMaster SDK markers expressed in normalized coords."""
        (
            profile_px,
            profile_t,
            coverage,
            _components,
            shape,
        ) = self._foam_profile_snapshot()

        if shape is None or len(shape) < 2:
            if TARGET_FOAM_FAIL_CLOSED and TARGET_FOAM_GATE_ENABLED:
                return False, None, "foam-frame-shape-missing", 0.0
            return True, None, "foam-frame-shape-missing-fail-open", 1.0

        frame_h, frame_w = int(shape[0]), int(shape[1])
        try:
            bx, by, bw, bh = [float(v) for v in bbox_norm]
            bbox_px = [
                bx * frame_w,
                by * frame_h,
                bw * frame_w,
                bh * frame_h,
            ]
            cx_px = float(cx) * frame_w
            cy_px = float(cy) * frame_h
        except Exception:
            if TARGET_FOAM_FAIL_CLOSED and TARGET_FOAM_GATE_ENABLED:
                return False, None, "foam-sdk-normalization-error", 0.0
            return True, None, "foam-sdk-normalization-error-fail-open", 1.0

        # Apply the same minimum-coverage trust rule used for color targets.
        if (
            TARGET_FOAM_GATE_ENABLED
            and profile_px is not None
            and coverage < TARGET_FOAM_MIN_ROI_COVERAGE
        ):
            if TARGET_FOAM_FAIL_CLOSED:
                return False, None, "foam-profile-low-coverage", 0.0
            return True, None, "foam-profile-low-coverage-fail-open", 1.0

        return self._target_passes_foam_gate_px(
            frame_w,
            frame_h,
            cx_px,
            cy_px,
            bbox_px,
            profile_px=profile_px,
            profile_t=profile_t,
        )


    def _current_sdk_markers(self):
        now = time.monotonic()
        with self.vision_lock:
            markers = [dict(m) for m in self.latest_sdk_markers]
            history = list(self.sdk_marker_history)

        # If the SDK stopped reporting, do not let stale markers suppress red
        # HSV detections forever.
        if history and now - history[-1][0] <= SDK_MARKER_TTL_SEC:
            return markers
        return []


    def _target_roi_px(self, frame):
        h, w = frame.shape[:2]
        x1 = int(round(TARGET_ROI_X_MIN * w))
        x2 = int(round(TARGET_ROI_X_MAX * w))
        y1 = int(round(TARGET_ROI_Y_MIN * h))
        y2 = int(round(TARGET_ROI_Y_MAX * h))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return x1, y1, x2, y2


    def _classify_target_contour(self, contour):
        area = float(cv2.contourArea(contour))
        if area <= 1.0:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1.0:
            return None

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        if solidity < TARGET_MIN_SOLIDITY:
            return None

        approx = cv2.approxPolyDP(
            contour,
            TARGET_POLY_EPS_FRAC * perimeter,
            True,
        )
        x, y, w, h = cv2.boundingRect(contour)
        fill = area / float(max(1, w * h))
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        shape = None
        quality = 0.0

        # Rectangle family.
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(float)
            corner_cos = []

            for i in range(4):
                prev_p = pts[(i - 1) % 4]
                cur_p = pts[i]
                next_p = pts[(i + 1) % 4]

                v1 = prev_p - cur_p
                v2 = next_p - cur_p
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom <= 1e-9:
                    corner_cos.append(1.0)
                else:
                    corner_cos.append(
                        abs(float(np.dot(v1, v2)) / denom)
                    )

            max_corner_cos = max(corner_cos) if corner_cos else 1.0

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if (
                rw > 1.0
                and rh > 1.0
                and fill >= TARGET_RECT_MIN_FILL
                and max_corner_cos <= TARGET_RECT_MAX_CORNER_COS
            ):
                aspect_rot = max(rw, rh) / max(1e-6, min(rw, rh))

                if (
                    TARGET_SQUARE_ASPECT_MIN
                    <= aspect_rot
                    <= TARGET_SQUARE_ASPECT_MAX
                ):
                    shape = "SQUARE"
                else:
                    # Use the image-axis box only to decide vertical/horizontal.
                    # minAreaRect can swap axes as its angle crosses 45 degrees.
                    axis_aspect = float(w) / max(1.0, float(h))
                    if axis_aspect >= TARGET_RECT_ASPECT_MIN:
                        shape = "RECT_HORIZONTAL"
                    elif (1.0 / max(axis_aspect, 1e-6)) >= TARGET_RECT_ASPECT_MIN:
                        shape = "RECT_VERTICAL"

                quality = (
                    0.45 * solidity
                    + 0.35 * min(1.0, fill)
                    + 0.20 * (1.0 - min(1.0, max_corner_cos))
                )

        # Circle family.
        elif len(approx) >= 5 and circularity >= TARGET_CIRCLE_MIN_CIRCULARITY:
            axis_aspect = float(w) / max(1.0, float(h))
            if TARGET_CIRCLE_ASPECT_MIN <= axis_aspect <= TARGET_CIRCLE_ASPECT_MAX:
                shape = "CIRCLE"
                quality = 0.55 * solidity + 0.45 * min(1.0, circularity)

        if shape is None:
            return None

        return {
            "shape": shape,
            "solidity": solidity,
            "fill": fill,
            "circularity": circularity,
            "quality": quality,
            "bbox_local": [int(x), int(y), int(w), int(h)],
            "area": area,
        }


    def _detect_color_targets(self, frame, foam_profile=None):
        if cv2 is None or np is None or frame is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = self._target_roi_px(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        roi_h, roi_w = roi.shape[:2]
        roi_area = float(max(1, roi_w * roi_h))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        k_open = np.ones((3, 3), dtype=np.uint8)
        k_close = np.ones((5, 5), dtype=np.uint8)
        sdk_markers = self._current_sdk_markers()

        detections = []

        for color_name, ranges in TARGET_HSV_RANGES.items():
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

            for lo, hi in ranges:
                lo_np = np.array(lo, dtype=np.uint8)
                hi_np = np.array(hi, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo_np, hi_np))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                area = float(cv2.contourArea(contour))
                area_frac = area / roi_area

                if (
                    area_frac < TARGET_MIN_AREA_FRAC_ROI
                    or area_frac > TARGET_MAX_AREA_FRAC_ROI
                ):
                    continue

                classified = self._classify_target_contour(contour)
                if classified is None:
                    continue

                lx, ly, bw, bh = classified["bbox_local"]

                # Strong rejection for a giant clipped region such as a yellow
                # wall entering the ROI from an edge.
                if (
                    lx <= TARGET_BORDER_MARGIN_PX
                    or ly <= TARGET_BORDER_MARGIN_PX
                    or lx + bw >= roi_w - TARGET_BORDER_MARGIN_PX
                    or ly + bh >= roi_h - TARGET_BORDER_MARGIN_PX
                ):
                    continue

                gx, gy = x1 + lx, y1 + ly
                cx = gx + bw * 0.5
                cy = gy + bh * 0.5
                cx_n = cx / float(frame_w)
                cy_n = cy / float(frame_h)

                if not self._roi_contains_norm_center(cx_n, cy_n):
                    continue

                foam_ok, foam_y, foam_reason, foam_below_frac = (
                    self._target_passes_foam_gate_px(
                        frame_w,
                        frame_h,
                        cx,
                        cy,
                        [gx, gy, bw, bh],
                        profile_px=foam_profile,
                        profile_t=(time.monotonic() if foam_profile is not None else None),
                    )
                )
                if not foam_ok:
                    continue

                bbox_norm = [
                    gx / float(frame_w),
                    gy / float(frame_h),
                    bw / float(frame_w),
                    bh / float(frame_h),
                ]

                # If the RoboMaster SDK already recognizes a red marker in the
                # same place, trust the SDK identity and suppress the duplicate
                # generic RED geometric target.
                if color_name == "RED":
                    overlap = max(
                        [
                            self._norm_bbox_iou(bbox_norm, m.get("bbox_norm", []))
                            for m in sdk_markers
                        ]
                        or [0.0]
                    )
                    if overlap >= 0.18:
                        continue

                quality = float(classified["quality"])
                score = clamp(
                    0.72 * quality
                    + 0.28 * min(1.0, area_frac / 0.025),
                    0.0,
                    1.0,
                )

                detections.append({
                    "kind": "COLOR_SHAPE",
                    "color": color_name,
                    "shape": classified["shape"],
                    "center_norm": [cx_n, cy_n],
                    "bbox_norm": bbox_norm,
                    "bbox_px": [int(gx), int(gy), int(bw), int(bh)],
                    "area_frac_roi": area_frac,
                    "solidity": classified["solidity"],
                    "fill": classified["fill"],
                    "circularity": classified["circularity"],
                    "score": score,
                    "foam_gate": foam_reason,
                    "foam_wall_y_px": foam_y,
                    "foam_bbox_below_frac": foam_below_frac,
                })

        detections.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return detections


    def _draw_target_overlay(self, frame, detections):
        if cv2 is None or frame is None:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = self._target_roi_px(out)

        (
            foam_profile,
            foam_t,
            foam_coverage,
            foam_components,
            _foam_shape,
        ) = self._foam_profile_snapshot()
        foam_age = time.monotonic() - foam_t if foam_t > 0.0 else float("inf")
        foam_fresh = (
            foam_profile is not None
            and foam_age <= TARGET_FOAM_PROFILE_TTL_SEC
        )
        foam_usable = (
            TARGET_FOAM_GATE_ENABLED
            and foam_fresh
            and foam_coverage >= TARGET_FOAM_MIN_ROI_COVERAGE
        )

        # Shade the part of the search ROI that is geometrically impossible:
        # anything above the detected foam-wall top.  This is only a preview;
        # the actual rejection happens in _target_passes_foam_gate_px().
        if foam_fresh and np is not None:
            shade = out.copy()
            for px in range(x1, min(x2, len(foam_profile))):
                wy = float(foam_profile[px])
                if not math.isfinite(wy):
                    continue
                cutoff = int(round(wy + TARGET_FOAM_CENTER_MARGIN_PX))
                cutoff = max(y1, min(y2, cutoff))
                if cutoff > y1:
                    cv2.line(
                        shade,
                        (px, y1),
                        (px, cutoff),
                        (0, 0, 80),
                        1,
                    )
            out = cv2.addWeighted(out, 0.80, shade, 0.20, 0.0)

            # Draw each contiguous known profile run without connecting across
            # corridor gaps where the wall top is genuinely unknown.
            run = []
            for px in range(x1, min(x2, len(foam_profile))):
                wy = float(foam_profile[px])
                if math.isfinite(wy):
                    run.append((px, int(round(wy))))
                else:
                    if len(run) >= 2:
                        cv2.polylines(
                            out,
                            [np.asarray(run, dtype=np.int32)],
                            False,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    run = []
            if len(run) >= 2:
                cv2.polylines(
                    out,
                    [np.asarray(run, dtype=np.int32)],
                    False,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        gate_text = (
            f"SEARCH ROI | FOAM GATE {'LOCK' if foam_usable else 'FAIL-OPEN'} "
            f"cov={foam_coverage * 100.0:.0f}%"
        )
        cv2.putText(
            out,
            gate_text,
            (x1 + 4, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        bgr_by_color = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 220, 0),
            "BLUE": (255, 100, 0),
        }

        for d in detections:
            x, y, bw, bh = d.get("bbox_px", [0, 0, 0, 0])
            color = bgr_by_color.get(d.get("color"), (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            fire_ok, _fire_reason = self.target_matches_fire_policy(d)
            fire_tag = "FIRE" if fire_ok else "SKIP"
            label = (
                f"{fire_tag} {d.get('color','?')} {d.get('shape','?')} "
                f"{d.get('score',0.0):.2f}"
            )
            cv2.putText(
                out,
                label,
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        for marker in self._current_sdk_markers():
            bx, by, bw, bh = marker.get("bbox_norm", [0, 0, 0, 0])
            x = int(bx * w)
            y = int(by * h)
            ww = int(bw * w)
            hh = int(bh * h)
            cv2.rectangle(out, (x, y), (x + ww, y + hh), (255, 0, 255), 2)
            marker_for_policy = dict(marker)
            marker_for_policy["kind"] = "SDK_MARKER"
            fire_ok, _fire_reason = self.target_matches_fire_policy(marker_for_policy)
            fire_tag = "FIRE" if fire_ok else "SKIP"
            cv2.putText(
                out,
                f"{fire_tag} SDK {marker.get('label','?')}",
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        try:
            gp, gy = self.current_gimbal_relative()
        except Exception:
            gp, gy = None, None

        pose_txt = (
            f"cell={self.current} heading={DIR_NAMES[self.heading]} "
            f"gimbal=({gy if gy is not None else 'n/a'},"
            f"{gp if gp is not None else 'n/a'}) "
            f"saved={len(self.detected_targets)}"
        )

        cv2.putText(
            out,
            pose_txt,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            self.target_status_text[:110],
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        return out


    def _vision_loop(self):
        print("[VISION] frame thread started")

        while self.running and self.vision_stream_started:
            try:
                frame = self.camera.read_cv2_image(
                    strategy="newest",
                    timeout=0.6,
                )
            except Exception as e:
                if self.running and self.vision_stream_started:
                    print(f"[VISION WARN] frame read failed: {e}")
                    time.sleep(0.08)
                continue

            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()

            # Estimate the dynamic white-foam horizon ONCE for this frame.
            # Both the HSV detector and asynchronous SDK-marker callback use
            # this same geometry constraint.
            foam_profile, foam_coverage, foam_components = (
                self._compute_foam_wall_profile(frame)
            )
            with self.vision_lock:
                self.latest_frame_shape = tuple(frame.shape[:2])
                self.latest_foam_profile_px = (
                    foam_profile.copy() if foam_profile is not None else None
                )
                self.latest_foam_profile_t = now
                self.latest_foam_profile_coverage = float(foam_coverage)
                self.latest_foam_components = [dict(c) for c in foam_components]

            detections = self._detect_color_targets(
                frame,
                foam_profile=foam_profile,
            )

            gimbal_p, gimbal_y = self.current_gimbal_relative()
            snap_detections = []

            for d in detections:
                item = dict(d)
                item["cell"] = [int(self.current[0]), int(self.current[1])]
                item["heading"] = int(self.heading) % 4
                item["gimbal_yaw_deg"] = gimbal_y
                item["gimbal_pitch_deg"] = gimbal_p
                snap_detections.append(item)

            with self.vision_lock:
                self.latest_detections = snap_detections
                self.detection_history.append((now, snap_detections))

            if self.preview_enabled:
                try:
                    annotated = self._draw_target_overlay(frame, snap_detections)
                    cv2.imshow(TARGET_WINDOW_NAME, annotated)
                    key = cv2.waitKey(1) & 0xFF

                    # q/ESC closes only the preview, not the robot mission.
                    if key in (ord("q"), 27):
                        self.preview_enabled = False
                        try:
                            cv2.destroyWindow(TARGET_WINDOW_NAME)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[VISION PREVIEW WARN] {e}")
                    self.preview_enabled = False

        try:
            if cv2 is not None:
                cv2.destroyWindow(TARGET_WINDOW_NAME)
        except Exception:
            pass

        print("[VISION] frame thread stopped")


    def start_target_vision(self):
        if not self.target_vision_enabled:
            self.target_status_text = "VISION DISABLED BY CONFIG"
            return False

        if cv2 is None or np is None:
            self.target_status_text = "VISION OFF: install opencv-python + numpy"
            print(
                "[VISION WARN] OpenCV/NumPy unavailable. "
                "DFS continues without target detection."
            )
            return False

        if self.camera is None:
            self.target_status_text = "VISION OFF: camera unavailable"
            return False

        try:
            from robomaster import camera as rm_camera
            resolution = getattr(
                rm_camera,
                "STREAM_360P",
                TARGET_CAMERA_RESOLUTION,
            )
        except Exception:
            resolution = TARGET_CAMERA_RESOLUTION

        try:
            self.camera.start_video_stream(
                display=False,
                resolution=resolution,
            )
            self.vision_stream_started = True
        except Exception as e:
            self.target_status_text = f"VISION STREAM ERROR: {e}"
            print(f"[VISION WARN] cannot start camera stream: {e}")
            return False

        if SDK_MARKER_ENABLED and self.vision is not None:
            try:
                ok = self.vision.sub_detect_info(
                    name="marker",
                    color="red",
                    callback=self.sdk_marker_callback,
                )
                print(f"[VISION] SDK marker subscription = {ok}")
            except Exception as e:
                print(f"[VISION WARN] SDK marker detector unavailable: {e}")

        self.preview_enabled = bool(self.preview_enabled and TARGET_PREVIEW_ENABLED)
        self.vision_available = True
        self.target_status_text = "VISION READY - waiting for target sweep"

        self.vision_thread = threading.Thread(
            target=self._vision_loop,
            name="RoboMasterTargetVision",
            daemon=True,
        )
        self.vision_thread.start()
        print("[VISION] target detector ready")
        return True


    def stop_target_vision(self):
        self.vision_available = False
        self.vision_stream_started = False

        if SDK_MARKER_ENABLED and self.vision is not None:
            try:
                self.vision.unsub_detect_info(name="marker")
            except Exception:
                pass

        if self.camera is not None:
            try:
                self.camera.stop_video_stream()
            except Exception:
                pass

        if self.vision_thread is not None and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)

        try:
            if cv2 is not None:
                cv2.destroyWindow(TARGET_WINDOW_NAME)
        except Exception:
            pass


    def _fresh_target_frame_count(self, since_t):
        """Number of camera frames processed after *since_t*."""
        with self.vision_lock:
            return sum(1 for ts, _ in self.detection_history if ts >= float(since_t))


