#!/usr/bin/env python3
"""
Measure the ToF's and Sharps' effective offset from the gimbal's yaw pivot.

_estimate_position() in V16B.py turns a ToF slant range straight into a
grid position, treating the reading as if it started exactly at the cell
centre. It doesn't: the ToF module sits some distance off the gimbal's
own rotation axis, so every target/wall position it derives is off by
that fixed amount. The same question applies to the two Sharp sensors.

Stand the robot centred in a cell with walls on its LEFT and RIGHT, square
to them (front/back walls too, if the cell has them) and run this. Left +
right always sums to one cell width, wherever exactly the robot sits
across the cell, so the gap between that sum and a real cell width IS the
sensor's own offset -- no ruler needed. (Same trick used to seed the first
ToF point in calibrate_tof.py, and the same one the sibling maze-runner
project's calibrate_mounts.py is built on.)

    python Test/calibrate_mounts.py
    python Test/calibrate_mounts.py --samples 40
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from robomaster import robot
except ModuleNotFoundError:
    robot = None

CELL_SIZE_M = 0.60  # matches GRID_TILE_M in V16B.py
DEFAULT_PITCH_DEG = -5.0
TOF_FREQ_HZ = 20
VALID_MIN_MM = 20.0
VALID_MAX_MM = 4000.0
NEAR_WALL_FRAC = 0.75  # a reading beyond this fraction of a cell isn't "the adjacent wall"

# Copied from V16B.py (LEFT_CAL/RIGHT_CAL, SHARP_MIN_PLAUSIBLE_ADC) so this
# script stays a standalone Test/ tool with no import-time dependency on it.
LEFT_CAL = [
    (4.0, 864.5), (6.0, 610.0), (8.0, 475.0), (10.0, 378.0),
    (12.0, 316.0), (14.0, 274.0), (16.0, 247.0), (18.0, 224.0),
    (20.0, 212.0), (22.0, 198.0), (24.0, 182.0),
]
RIGHT_CAL = [
    (4.0, 822.0), (6.0, 589.0), (8.0, 455.0), (10.0, 374.0),
    (12.0, 313.0), (14.0, 276.0), (16.0, 239.0), (18.0, 205.0),
    (20.0, 174.0), (22.0, 161.0), (24.0, 133.0),
]
SHARP_LEFT_ID, SHARP_RIGHT_ID, SENSOR_PORT = 2, 3, 1
SHARP_MIN_PLAUSIBLE_ADC = 20.0


def adc_to_cm(adc, calibration):
    if adc is None:
        return None
    if not math.isfinite(adc) or adc < SHARP_MIN_PLAUSIBLE_ADC:
        return None
    near_cm, near_adc = calibration[0]
    _, far_adc = calibration[-1]
    if adc >= near_adc:
        return near_cm
    if adc < far_adc:
        return None
    for (d1, a1), (d2, a2) in zip(calibration, calibration[1:]):
        if a1 >= adc >= a2:
            return (d1 + d2) * 0.5 if abs(a1 - a2) < 1e-9 else d1 + (a1 - adc) / (a1 - a2) * (d2 - d1)
    return None


def median_tof(ep_robot, latest, lock, yaw_deg, pitch_deg, samples) -> Optional[float]:
    ep_robot.gimbal.moveto(pitch=pitch_deg, yaw=yaw_deg, pitch_speed=60, yaw_speed=90).wait_for_completed()
    values, seen_ts = [], None
    deadline = time.monotonic() + 3.0
    while len(values) < samples and time.monotonic() < deadline:
        with lock:
            item = latest[0]
        if item is not None:
            mm, ts = item
            if ts != seen_ts and VALID_MIN_MM <= mm <= VALID_MAX_MM:
                seen_ts = ts
                values.append(mm)
        time.sleep(0.02)
    return statistics.median(values) / 1000.0 if values else None  # -> metres


def median_sharp(ep_robot, samples) -> Tuple[Optional[float], Optional[float]]:
    lefts, rights = [], []
    for _ in range(samples):
        try:
            l_adc = ep_robot.sensor_adaptor.get_adc(id=SHARP_LEFT_ID, port=SENSOR_PORT)
        except Exception:
            l_adc = None
        try:
            r_adc = ep_robot.sensor_adaptor.get_adc(id=SHARP_RIGHT_ID, port=SENSOR_PORT)
        except Exception:
            r_adc = None
        l_cm, r_cm = adc_to_cm(l_adc, LEFT_CAL), adc_to_cm(r_adc, RIGHT_CAL)
        if l_cm is not None:
            lefts.append(l_cm)
        if r_cm is not None:
            rights.append(r_cm)
        time.sleep(0.02)
    l = statistics.median(lefts) / 100.0 if lefts else None  # -> metres
    r = statistics.median(rights) / 100.0 if rights else None
    return l, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conn-type", default="ap", choices=("ap", "sta", "rndis"))
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_DEG)
    ap.add_argument("--out", default="calibration/mounts.json")
    args = ap.parse_args()

    if robot is None:
        raise RuntimeError("RoboMaster SDK not found in the active Python environment")

    print("=" * 68)
    print(" SENSOR MOUNT OFFSET (gimbal pivot vs. ToF / Sharp)")
    print("=" * 68)
    print("Stand the robot centred in a cell with walls LEFT and RIGHT")
    print("(front/back too if present), square to them.")
    print("=" * 68)

    import threading
    lock = threading.Lock()
    latest = [None]  # (mm, monotonic_ts), boxed so the callback can rebind it

    def on_distance(info):
        if info:
            with lock:
                latest[0] = (float(info[0]), time.monotonic())

    ep_robot = robot.Robot()
    try:
        print(f"\n[CONNECT] conn_type='{args.conn_type}' ...")
        ep_robot.initialize(conn_type=args.conn_type)
        ep_robot.gimbal.recenter().wait_for_completed()
        ep_robot.sensor.sub_distance(freq=TOF_FREQ_HZ, callback=on_distance)

        raw = {}
        for name, yaw in (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", -90.0)):
            raw[name] = median_tof(ep_robot, latest, lock, yaw, args.pitch, args.samples)
        ep_robot.gimbal.moveto(pitch=args.pitch, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
        sharp_l, sharp_r = median_sharp(ep_robot, args.samples * 2)
    finally:
        try:
            ep_robot.sensor.unsub_distance()
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass

    print("\nraw ToF (m):", {k: None if v is None else round(v, 3) for k, v in raw.items()})
    print(f"Sharp (m): left={sharp_l} right={sharp_r}")

    out = {}
    half = CELL_SIZE_M / 2.0
    limit = CELL_SIZE_M * NEAR_WALL_FRAC

    e, w = raw["E"], raw["W"]
    if e is None or w is None or max(e, w) > limit:
        print("\nNo usable walls left+right (ToF at +/-90deg): tof_pivot_offset_m not computed.")
    else:
        out["tof_pivot_offset_m"] = round((CELL_SIZE_M - (e + w)) / 2.0, 4)

    n, s = raw["N"], raw["S"]
    if n is not None and s is not None and max(n, s) < limit:
        out["tof_pivot_forward_m"] = round((s - n) / 2.0, 4)
    else:
        print("No walls front+back: tof_pivot_forward_m not computed.")

    if sharp_l is None or sharp_r is None:
        print("A Sharp gave no reading: sharp offsets not computed.")
    elif "tof_pivot_offset_m" in out:
        # Sharp's own calibration already reports metres from its own face;
        # any leftover gap from one cell width is a lateral offset vs. the
        # SAME pivot the ToF trick just anchored, not a Sharp bias per se.
        offset = round((CELL_SIZE_M - (sharp_l + sharp_r)) / 2.0, 4)
        out["sharp_left_offset_m"] = offset
        out["sharp_right_offset_m"] = offset

    if not out:
        print("\nNothing usable measured (need walls on at least left+right). Nothing saved.")
        return 1

    out["measured_at"] = datetime.now().isoformat(timespec="seconds")
    out["source"] = "python Test/calibrate_mounts.py (robot centred in a cell, walls left+right)"
    out["raw"] = {"tof": raw, "sharp_left_m": sharp_l, "sharp_right_m": sharp_r}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nSaved {out_path}:")
    print(json.dumps({k: v for k, v in out.items() if k.endswith("_m")}, indent=2))
    if "tof_pivot_offset_m" in out:
        print(
            f"\nThis is a HORIZONTAL offset, not the vertical FIRE_CAMERA_ABOVE_MUZZLE_M /"
            f"\nFIRE_TOF_ABOVE_MUZZLE_M constants already in V16B.py -- those stay ruler-measured."
            f"\nTo use it, add tof_pivot_offset_m ({out['tof_pivot_offset_m']*1000:+.1f} mm) to the"
            f"\nraw ToF (in metres) before _estimate_position() treats it as distance-from-pivot."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
