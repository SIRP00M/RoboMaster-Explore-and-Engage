#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive HSV target-color calibration for TARGET_HSV_RANGES.

Point the camera at a target and either CLICK a spot or DRAG a box over a
whole colored patch; the tool samples the HSV under it and grows a
min/max range that covers everything you've marked for the active color.
The mask preview updates live so you can see coverage before saving --
no guessing, no re-runs.

Efficient by default:
  - starts from whatever TARGET_HSV_RANGES already sits in V16B.py, so a
    same-day touch-up is a few drags, not starting from a blank slate
    (--seed-from to point elsewhere, --no-seed to start empty)
  - auto-picks the RoboMaster camera, falling back to a USB webcam if the
    robot isn't connected, so this also works as an indoor dry run
  - only the ACTIVE color's mask is computed per frame

    python Test/calibrate_vision.py                    # RoboMaster camera, or webcam 0 if none
    python Test/calibrate_vision.py --source 0          # force webcam
    python Test/calibrate_vision.py --source photo.jpg  # a still image
    python Test/calibrate_vision.py --apply V16B.py      # also patch the file in place on save

Controls
--------
    1-4         select RED / YELLOW / GREEN / BLUE
    drag        sample a whole patch (5th-95th percentile HSV inside the box)
    click       sample a single point (small patch under the cursor)
    z           undo the last sample for the active color
    c           clear CLICKED samples for the active color (keeps the seed)
    x           clear the SEEDED range for the active color (keeps clicks)
    r           clear both, for the active color
    p           cycle preview: frame / mask / side-by-side
    s           save JSON + snippet (+ patch --apply file if given)
    q / ESC     quit
"""

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    sys.exit(f"calibrate_vision requires opencv-python and numpy: {exc}")

COLOR_NAMES = ["RED", "YELLOW", "GREEN", "BLUE"]
COLOR_KEYS = {ord("1"): "RED", ord("2"): "YELLOW", ord("3"): "GREEN", ord("4"): "BLUE"}
COLOR_BGR = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0), "BLUE": (255, 120, 0)}

WINDOW = "HSV Calibrator"
PATCH_RADIUS_DEFAULT = 5
DRAG_MIN_PX = 6
HUE_WRAP_SPLIT = 90.0
VARNAME = "TARGET_HSV_RANGES"


# ------------------------------------------------------------
# Frame sources
# ------------------------------------------------------------
class WebcamSource:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class VideoFileSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video file {path}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class ImageSource:
    def __init__(self, path):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read image {path}")
        self.frame = frame

    def read(self):
        return self.frame.copy()

    def release(self):
        pass


class RoboMasterSource:
    def __init__(self, conn_type="ap"):
        from robomaster import robot, camera as rm_camera

        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type=conn_type)
        self.camera = self.ep_robot.camera
        resolution = getattr(rm_camera, "STREAM_360P", "360p")
        self.camera.start_video_stream(display=False, resolution=resolution)

    def read(self):
        try:
            return self.camera.read_cv2_image(strategy="newest", timeout=0.6)
        except Exception:
            return None

    def release(self):
        try:
            self.camera.stop_video_stream()
        finally:
            self.ep_robot.close()


def make_source(spec, conn_type):
    if spec == "auto":
        try:
            return RoboMasterSource(conn_type=conn_type)
        except Exception as exc:
            print(f"[SOURCE] RoboMaster unavailable ({exc}); falling back to webcam 0")
            return WebcamSource(0)
    if spec == "robomaster":
        return RoboMasterSource(conn_type=conn_type)
    if isinstance(spec, str) and spec.isdigit():
        return WebcamSource(spec)
    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"source not found: {spec}")
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return ImageSource(path)
    return VideoFileSource(path)


# ------------------------------------------------------------
# Existing TARGET_HSV_RANGES <-> this tool
# ------------------------------------------------------------
def load_seed_ranges(path):
    """Parse TARGET_HSV_RANGES out of ``path`` with ast.literal_eval (no exec)."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(rf"{VARNAME}\s*=\s*(\{{.*?\n\}})", text, re.DOTALL)
    if not m:
        return {}
    try:
        raw = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return {}
    return {name: [(tuple(lo), tuple(hi)) for lo, hi in ranges] for name, ranges in raw.items()}


def patch_file(path, ranges_by_color):
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(rf"{VARNAME}\s*=\s*(\{{.*?\n\}})", text, re.DOTALL)
    if not m:
        print(f"[APPLY] no {VARNAME} block found in {path}; left it untouched")
        return False
    backup = Path(str(path) + ".hsv_bak")
    backup.write_text(text, encoding="utf-8")
    new_block = format_block(ranges_by_color)
    patched = text[: m.start(1)] + new_block + text[m.end(1) :]
    Path(path).write_text(patched, encoding="utf-8")
    print(f"[APPLY] patched {path} ({VARNAME}); previous version backed up to {backup}")
    return True


# ------------------------------------------------------------
# HSV range bookkeeping
# ------------------------------------------------------------
class Calibrator:
    def __init__(self, colors, seed=None, patch_radius=PATCH_RADIUS_DEFAULT):
        self.colors = list(colors)
        self.samples = {name: [] for name in self.colors}
        self.seed = {name: list(seed.get(name, [])) for name in self.colors} if seed else {n: [] for n in self.colors}
        self.active = self.colors[0]
        self.patch_radius = patch_radius
        self.h_margin, self.s_margin, self.v_margin = 8, 30, 30
        self.hover_hsv = None

    def sample_at(self, hsv_frame, x, y):
        h, w = hsv_frame.shape[:2]
        r = self.patch_radius
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        patch = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            return None
        med = np.median(patch, axis=0)
        return tuple(float(v) for v in med)

    def add_point(self, hsv_frame, x, y):
        hsv = self.sample_at(hsv_frame, x, y)
        if hsv is not None:
            self.samples[self.active].append(hsv)
        return hsv

    def add_region(self, hsv_frame, x0, y0, x1, y1):
        h, w = hsv_frame.shape[:2]
        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))
        region = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
        if region.size == 0:
            return None
        lo = np.percentile(region, 5, axis=0)
        hi = np.percentile(region, 95, axis=0)
        self.samples[self.active].append(tuple(float(v) for v in lo))
        self.samples[self.active].append(tuple(float(v) for v in hi))
        return lo, hi

    def undo(self):
        if self.samples[self.active]:
            self.samples[self.active].pop()

    def clear_samples(self):
        self.samples[self.active] = []

    def clear_seed(self):
        self.seed[self.active] = []

    def clear_all(self):
        self.clear_samples()
        self.clear_seed()

    def ranges_for(self, name):
        clicked = compute_ranges(self.samples[name], self.h_margin, self.s_margin, self.v_margin)
        return merge_ranges(self.seed.get(name, []) + clicked)

    def all_ranges(self):
        return {name: self.ranges_for(name) for name in self.colors if self.ranges_for(name)}


def compute_ranges(samples, h_margin, s_margin, v_margin):
    """Accumulated (h, s, v) samples -> one or two HSV bound tuples.

    Two ranges come back when the samples straddle the hue wraparound
    (RED sitting on both sides of 0/179), matching what the detector
    already expects for such colors.
    """
    if not samples:
        return []
    hues = [s[0] for s in samples]
    low_side = [s for s in samples if s[0] < HUE_WRAP_SPLIT]
    high_side = [s for s in samples if s[0] >= HUE_WRAP_SPLIT]
    wraps = low_side and high_side and min(hues) <= 15 and max(hues) >= 165 and (max(hues) - min(hues)) > 90
    if not wraps:
        return [_bounds(samples, h_margin, s_margin, v_margin)]
    return [_bounds(low_side, h_margin, s_margin, v_margin), _bounds(high_side, h_margin, s_margin, v_margin)]


def _bounds(samples, h_margin, s_margin, v_margin):
    hs, ss, vs = (s[0] for s in samples), (s[1] for s in samples), (s[2] for s in samples)
    hs, ss, vs = list(hs), list(ss), list(vs)
    lo = (int(max(0, min(hs) - h_margin)), int(max(0, min(ss) - s_margin)), int(max(0, min(vs) - v_margin)))
    hi = (int(min(179, max(hs) + h_margin)), int(min(255, max(ss) + s_margin)), int(min(255, max(vs) + v_margin)))
    return (lo, hi)


def merge_ranges(ranges):
    """Union a mixed bag of (lo, hi) tuples, keeping the low/high hue-wrap buckets separate."""
    if not ranges:
        return []
    low_bucket = [r for r in ranges if r[0][0] < HUE_WRAP_SPLIT]
    high_bucket = [r for r in ranges if r[0][0] >= HUE_WRAP_SPLIT]
    out = []
    for bucket in (low_bucket, high_bucket):
        if not bucket:
            continue
        los, his = [r[0] for r in bucket], [r[1] for r in bucket]
        lo = tuple(int(min(v[i] for v in los)) for i in range(3))
        hi = tuple(int(max(v[i] for v in his)) for i in range(3))
        out.append((lo, hi))
    return out


# ------------------------------------------------------------
# Rendering / output
# ------------------------------------------------------------
def build_mask(hsv_frame, ranges):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv_frame, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    return mask


def draw_hud(frame, calib, coverage_pct):
    y = 20
    for name in calib.colors:
        n = len(calib.samples[name])
        ranges = calib.ranges_for(name)
        marker = ">" if name == calib.active else " "
        text = f"{marker}{name}: {n} clicks"
        if ranges:
            text += " " + " ".join(f"{lo}-{hi}" for lo, hi in ranges)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_BGR[name], 1, cv2.LINE_AA)
        y += 16

    if calib.hover_hsv is not None:
        h, s, v = calib.hover_hsv
        cv2.putText(frame, f"cursor HSV=({h:.0f},{s:.0f},{v:.0f})  coverage={coverage_pct:.1f}%",
                    (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def format_block(ranges_by_color):
    lines = [f"{VARNAME} = {{"]
    for name, ranges in ranges_by_color.items():
        if len(ranges) == 1:
            lo, hi = ranges[0]
            lines.append(f'    "{name}": [({tuple(lo)}, {tuple(hi)})],')
        else:
            lines.append(f'    "{name}": [')
            for lo, hi in ranges:
                lines.append(f"        ({tuple(lo)}, {tuple(hi)}),")
            lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def save(calib, out_path, apply_path):
    ranges_by_color = calib.all_ranges()
    if not ranges_by_color:
        print("[SAVE] no ranges yet (no seed, no samples) -- nothing to save")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {name: [[list(lo), list(hi)] for lo, hi in ranges] for name, ranges in ranges_by_color.items()},
        indent=2,
    ))

    snippet = format_block(ranges_by_color)
    snippet_path = out_path.with_suffix(".py")
    snippet_path.write_text(snippet + "\n")

    print(f"[SAVE] wrote {out_path}")
    print(f"[SAVE] wrote {snippet_path}")
    print(snippet)

    if apply_path:
        patch_file(apply_path, ranges_by_color)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", help="'auto' (RoboMaster, else webcam), 'robomaster', a webcam index, or an image/video path")
    ap.add_argument("--conn-type", default="ap")
    ap.add_argument("--seed-from", default="V16B.py", help=f"file to read the starting {VARNAME} from")
    ap.add_argument("--no-seed", action="store_true", help="start with empty ranges instead of seeding from --seed-from")
    ap.add_argument("--apply", default=None, help="also patch this file's TARGET_HSV_RANGES in place on save")
    ap.add_argument("--out", default="calibration/vision_hsv.json")
    args = ap.parse_args()

    seed = {}
    if not args.no_seed and Path(args.seed_from).exists():
        seed = load_seed_ranges(args.seed_from)
        if seed:
            print(f"[SEED] loaded {VARNAME} from {args.seed_from}: {list(seed)}")

    source = make_source(args.source, args.conn_type)
    calib = Calibrator(COLOR_NAMES, seed=seed)
    preview_mode = 0  # 0=frame, 1=mask, 2=side-by-side
    drag_start = [None]
    drag_now = [None]
    last_hsv_frame = [None]

    def on_mouse(event, x, y, flags, userdata):
        if last_hsv_frame[0] is None:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            calib.hover_hsv = calib.sample_at(last_hsv_frame[0], x, y)
            if drag_start[0] is not None:
                drag_now[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            drag_start[0] = (x, y)
            drag_now[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if drag_start[0] is None:
                return
            x0, y0 = drag_start[0]
            drag_start[0] = None
            drag_now[0] = None
            if abs(x - x0) < DRAG_MIN_PX and abs(y - y0) < DRAG_MIN_PX:
                hsv = calib.add_point(last_hsv_frame[0], x, y)
                if hsv is not None:
                    print(f"[{calib.active}] point HSV=({hsv[0]:.0f},{hsv[1]:.0f},{hsv[2]:.0f})")
            else:
                found = calib.add_region(last_hsv_frame[0], x0, y0, x, y)
                if found is not None:
                    lo, hi = found
                    print(f"[{calib.active}] region 5-95pct HSV lo=({lo[0]:.0f},{lo[1]:.0f},{lo[2]:.0f}) "
                          f"hi=({hi[0]:.0f},{hi[1]:.0f},{hi[2]:.0f})")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("HSV Calibrator ready. 1-4=color  drag/click=sample  z=undo  c=clear-clicks  "
          "x=clear-seed  r=clear-all  p=preview  s=save  q=quit")

    try:
        while True:
            frame = source.read()
            if frame is None:
                time.sleep(0.02)
                continue

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            last_hsv_frame[0] = hsv_frame

            active_ranges = calib.ranges_for(calib.active)
            mask = build_mask(hsv_frame, active_ranges)
            coverage_pct = 100.0 * cv2.countNonZero(mask) / mask.size

            display = frame.copy()
            draw_hud(display, calib, coverage_pct)
            if drag_start[0] is not None and drag_now[0] is not None:
                cv2.rectangle(display, drag_start[0], drag_now[0], COLOR_BGR[calib.active], 1)

            if preview_mode in (1, 2):
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                shown = mask_bgr if preview_mode == 1 else np.hstack([display, mask_bgr])
            else:
                shown = display

            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key in COLOR_KEYS:
                calib.active = COLOR_KEYS[key]
            elif key == ord("z"):
                calib.undo()
            elif key == ord("c"):
                calib.clear_samples()
            elif key == ord("x"):
                calib.clear_seed()
            elif key == ord("r"):
                calib.clear_all()
            elif key == ord("p"):
                preview_mode = (preview_mode + 1) % 3
            elif key == ord("s"):
                save(calib, args.out, args.apply)
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
