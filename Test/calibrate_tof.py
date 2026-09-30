#!/usr/bin/env python3
"""
RoboMaster front ToF raw-vs-true distance calibration.

The ToF is not linear near its minimum range: a single fixed offset that
looks right at 0.3 m can be several cm off by 0.9 m. This builds a
piecewise-linear (raw_mm -> true_mm) table instead, the same idea as the
existing Sharp calibration (LEFT_CAL/RIGHT_CAL) but for the front ToF.

Default (auto), no ruler needed after setup: stand the robot centred in a
maze cell with walls on its LEFT, RIGHT and FRONT, square to them. Left-raw
+ right-raw always sums to one cell width regardless of exactly where the
robot sits across the cell, so that fixes the ToF's own offset at short
range and, with it, the true distance to the wall ahead -- point 1, no
ruler. From there it's interactive and grid-quantized, since the maze's
own cells are already a known ruler: press 'n' and the robot backs up one
full grid cell (--cell-size, 0.60 m by default) on odometry (holding
heading straight with the IMU) and records a ToF point there -- true
distance is always point-1's distance plus however many grid cells back,
no separate measuring step. This mirrors calibrate_tof.py in the sibling
maze-runner project (tools/calibrate_tof.py), adapted to drive directly on
the RoboMaster SDK and to step by grid cell instead of a preset list.

    python Test/calibrate_tof.py                # auto point 1, then 'n' to step back a grid cell
    python Test/calibrate_tof.py --cell-size 0.5 # a different maze cell size
    python Test/calibrate_tof.py --manual        # place the wall by hand for every point instead
    python Test/calibrate_tof.py --manual --distances 40 60 90 120

Interactive (auto) controls: n=back up one grid cell & record it, z=undo
the last point (and re-arm that grid step for a retry), s=save without
quitting, q=save & quit.

Manual mode (or the automatic fallback when there are no side walls to
find point 1 from): R=record/re-record N samples, Enter=accept & next,
Q=save & quit.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from robomaster import robot
except ModuleNotFoundError:
    robot = None

from backup_driver import BackupDriver

CELL_SIZE_M = 0.60  # matches GRID_TILE_M in V16B.py
DEFAULT_DISTANCES_CM = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 120.0, 150.0)
DEFAULT_SAMPLES = 40
DEFAULT_PITCH_DEG = -5.0
TOF_FREQ_HZ = 20
VALID_MIN_MM = 20.0
VALID_MAX_MM = 4000.0
SIDE_WALL_MAX_FRAC = 0.75  # a side reading beyond this fraction of a cell isn't "the wall"


def read_key() -> str:
    """Read one key without requiring Enter first."""
    import os as _os
    if _os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        return "ENTER" if ch in ("\r", "\n") else ch.upper()

    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "ENTER" if ch in ("\r", "\n") else ch.upper()


class ToFReader:
    def __init__(self, ep_robot):
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[float, float]] = None  # (mm, monotonic_ts)
        self._sensor = ep_robot.sensor
        self._sensor.sub_distance(freq=TOF_FREQ_HZ, callback=self._callback)

    def _callback(self, info) -> None:
        if info:
            with self._lock:
                self._latest = (float(info[0]), time.monotonic())

    def median(self, samples: int, timeout: float = 3.0) -> Optional[float]:
        values: List[float] = []
        seen_ts: Optional[float] = None
        deadline = time.monotonic() + timeout
        while len(values) < samples and time.monotonic() < deadline:
            with self._lock:
                item = self._latest
            if item is not None:
                mm, ts = item
                if ts != seen_ts and VALID_MIN_MM <= mm <= VALID_MAX_MM:
                    seen_ts = ts
                    values.append(mm)
            time.sleep(0.02)
        return statistics.median(values) if values else None

    def close(self) -> None:
        try:
            self._sensor.unsub_distance()
        except Exception:
            pass


def auto_first_point(
    ep_robot, reader: ToFReader, pitch_deg: float, samples: int, cell_size_m: float = CELL_SIZE_M,
) -> Optional[Tuple[float, float]]:
    """(raw_mm, true_mm) for the wall dead ahead, found from the side walls.

    Returns None if there are no usable walls on both sides (open floor,
    doorway, etc.) -- caller should fall back to fully manual placement.
    """
    gimbal = ep_robot.gimbal
    limit_mm = cell_size_m * 1000.0 * SIDE_WALL_MAX_FRAC

    gimbal.moveto(pitch=pitch_deg, yaw=-90.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
    left = reader.median(samples)
    gimbal.moveto(pitch=pitch_deg, yaw=90.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
    right = reader.median(samples)
    gimbal.moveto(pitch=pitch_deg, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()

    if left is None or right is None or left > limit_mm or right > limit_mm:
        return None

    offset_mm = (cell_size_m * 1000.0 - (left + right)) / 2.0
    front = reader.median(samples)
    if front is None:
        return None
    true_mm = front + offset_mm
    print(
        f"[AUTO] side walls left={left:.1f} right={right:.1f} mm -> "
        f"ToF offset {offset_mm:+.1f} mm; wall ahead raw={front:.1f} -> true={true_mm:.1f} mm"
    )
    return front, true_mm


def interactive_grid_collect(
    ep_robot, reader: ToFReader, pitch_deg: float, samples: int,
    start_true_cm: float, cell_size_m: float, points: Dict[float, float],
    save_fn,
) -> None:
    """'n' backs up one grid cell and records a point there; 'z' undoes the
    last one (and re-arms that grid step so 'n' re-measures the same spot);
    's' saves without quitting; 'q' saves and returns.

    True distance for step k is always start_true_cm + k * cell_size_m --
    the maze's own cells are the ruler, so no separate measuring is needed.
    """
    span_m = cell_size_m  # printed as a per-press reminder, not a hard cap
    print(f"\nGrid step = {cell_size_m * 100.0:.0f} cm. Each 'n' backs up one cell "
          f"({span_m * 100.0:.0f} cm) -- make sure that's clear behind it.")
    print("n=back up one grid cell & record   z=undo last (re-arm)   s=save   q=save & quit")

    driver = BackupDriver(ep_robot)
    step = 0
    try:
        while True:
            key = read_key()
            if key == "N":
                step += 1
                target_total_m = step * cell_size_m
                traveled_m = driver.back_up_to(target_total_m)
                real_true_cm = start_true_cm + (traveled_m or 0.0) * 100.0
                ep_robot.gimbal.moveto(pitch=pitch_deg, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
                raw = reader.median(samples)
                if raw is None:
                    print(f"  step {step} (~{real_true_cm:.1f} cm): no fresh ToF reading -- not recorded, "
                          f"press 'n' again to retry from here")
                    step -= 1
                    continue
                print(f"  step {step}: {real_true_cm:.1f} cm  raw={raw:.1f} mm  "
                      f"(shift {real_true_cm * 10.0 - raw:+.1f})")
                points[round(real_true_cm, 2)] = raw
            elif key == "Z":
                if points and len(points) > 1:  # never undo point 1
                    dropped = max(points)
                    del points[dropped]
                    step = max(0, step - 1)
                    print(f"  undid {dropped:.1f} cm -- press 'n' to re-measure that grid step")
                else:
                    print("  nothing to undo")
            elif key == "S":
                save_fn(points)
            elif key == "Q":
                save_fn(points)
                break
    finally:
        driver.close()


def collect(reader: ToFReader, distances_cm: List[float], samples: int, points: Dict[float, float]) -> bool:
    """Fill ``points`` (true_cm -> raw_mm) interactively. Returns False on quit."""
    for true_cm in distances_cm:
        if true_cm in points:
            continue
        accepted = False
        while not accepted:
            print(f"\nSet the wall to {true_cm:.1f} cm from the gimbal (tape/ruler). "
                  f"R=record  Q=save & quit")
            key = read_key()
            if key == "Q":
                return False
            if key != "R":
                continue
            raw = reader.median(samples)
            if raw is None:
                print("  no fresh ToF reading -- check the wall is in range and in view")
                continue
            print(f"  raw = {raw:.1f} mm  (shift {true_cm * 10.0 - raw:+.1f} mm)")
            print("  Enter=accept & next   R=re-record   Q=save & quit")
            while True:
                confirm = read_key()
                if confirm == "ENTER":
                    points[true_cm] = raw
                    accepted = True
                    break
                if confirm == "R":
                    break
                if confirm == "Q":
                    return False
    return True


def build_table(points: Dict[float, float]) -> List[Tuple[float, float]]:
    pairs = sorted((raw, true_cm * 10.0) for true_cm, raw in points.items())
    table = sorted(set(pairs))
    if len(table) < 2:
        raise ValueError("need at least two distinct calibration points")

    bad = [(r2, d2) for (r1, d1), (r2, d2) in zip(table, table[1:]) if d2 <= d1]
    if bad:
        print("[WARN] true distance should only ever INCREASE as raw reading increases; "
              "it didn't at:")
        for r, d in bad:
            print(f"         raw={r:.1f} mm  true={d:.1f} mm")
        print("       Something likely went wrong physically around there (robot moved by")
        print("       hand, hit something, wheel slip). Consider dropping those points and")
        print("       re-measuring that range before trusting this table.")
    return table


def save_points(points: Dict[float, float], out_path: str, source_label: str) -> bool:
    if len(points) < 2:
        print(f"\n[SAVE] only {len(points)} point(s) so far; need >= 2 to build a table -- keep collecting")
        return False

    table = build_table(points)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "points": [{"raw_mm": r, "true_mm": d} for r, d in table],
        "source": source_label,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    snippet = format_snippet(table)
    snippet_path = path.with_suffix(".py")
    snippet_path.write_text(snippet + "\n")

    print(f"\n[SAVE] {len(table)} points -> {path} and {snippet_path}")
    print("Paste into V16B.py next to LEFT_CAL/RIGHT_CAL:\n")
    print(snippet)
    return True


def format_snippet(table: List[Tuple[float, float]]) -> str:
    lines = ["TOF_CAL = [  # (raw_mm, true_mm) -- python Test/calibrate_tof.py"]
    for raw, true in table:
        lines.append(f"    ({raw:.1f}, {true:.1f}),")
    lines.append("]")
    lines.append("")
    lines.append("")
    lines.append("def tof_calibrated_mm(raw_mm, table=TOF_CAL):")
    lines.append('    """raw ToF mm -> corrected mm; same interpolate/extrapolate-by-shift as adc_to_cm."""')
    lines.append("    if raw_mm is None:")
    lines.append("        return None")
    lines.append("    if raw_mm <= table[0][0]:")
    lines.append("        return raw_mm + (table[0][1] - table[0][0])")
    lines.append("    if raw_mm >= table[-1][0]:")
    lines.append("        return raw_mm + (table[-1][1] - table[-1][0])")
    lines.append("    for (r1, d1), (r2, d2) in zip(table, table[1:]):")
    lines.append("        if r1 <= raw_mm <= r2:")
    lines.append("            return d1 + (d2 - d1) * (raw_mm - r1) / (r2 - r1)")
    lines.append("    return raw_mm")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conn-type", default="ap", choices=("ap", "sta", "rndis"))
    ap.add_argument("--cell-size", type=float, default=CELL_SIZE_M, help="maze grid cell size in metres (auto mode step)")
    ap.add_argument("--distances", type=float, nargs="+", default=list(DEFAULT_DISTANCES_CM),
                     help="--manual only: true distances in cm to place the wall at")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_DEG)
    ap.add_argument("--manual", action="store_true", help="place the wall by hand for every point (no self-driving)")
    ap.add_argument("--max", type=float, default=None, help="--manual only: drop distances beyond this cm")
    ap.add_argument("--out", default="calibration/tof.json")
    args = ap.parse_args()

    if robot is None:
        raise RuntimeError("RoboMaster SDK not found in the active Python environment")

    points: Dict[float, float] = {}
    source_label = "python Test/calibrate_tof.py" + (" --manual" if args.manual else "")

    print("=" * 68)
    print(" ToF RAW-VS-TRUE DISTANCE CALIBRATION")
    print("=" * 68)
    if args.manual:
        distances = sorted(d for d in args.distances if args.max is None or d <= args.max + 1e-9)
        print(f"Place a wall at each of {distances} cm from the gimbal by hand.")
        print("R=record | Enter=accept/next | Q=save & quit")
    else:
        print("Stand the robot centred in a cell with walls LEFT, RIGHT and FRONT,")
        print("square to them: point 1 comes from those walls, then 'n' backs it up")
        print(f"one grid cell ({args.cell_size * 100.0:.0f} cm) at a time, recording as it goes.")
        print("Falls back to manual placement if there are no side walls for point 1.")
    print("=" * 68)

    ep_robot = robot.Robot()
    reader = None
    already_saved = False
    try:
        print(f"\n[CONNECT] conn_type='{args.conn_type}' ...")
        ep_robot.initialize(conn_type=args.conn_type)
        ep_robot.gimbal.recenter().wait_for_completed()
        reader = ToFReader(ep_robot)
        print("[CONNECT] RoboMaster connected, ToF streaming")

        start_true_cm = None
        if not args.manual:
            found = auto_first_point(ep_robot, reader, args.pitch, args.samples, cell_size_m=args.cell_size)
            if found is not None:
                raw, true_mm = found
                start_true_cm = round(true_mm / 10.0, 2)
                points[start_true_cm] = raw
            else:
                print("[AUTO] no usable side walls -- falling back to fully manual placement")

        ep_robot.gimbal.moveto(pitch=args.pitch, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()

        if start_true_cm is not None:
            interactive_grid_collect(
                ep_robot, reader, args.pitch, args.samples, start_true_cm, args.cell_size, points,
                save_fn=lambda pts: save_points(pts, args.out, source_label),
            )
            already_saved = True  # interactive_grid_collect() only returns via its own 'q' save
        else:
            distances = sorted(d for d in args.distances if args.max is None or d <= args.max + 1e-9)
            collect(reader, distances, args.samples, points)
    except KeyboardInterrupt:
        print("\n[CTRL+C] saving progress...")
    finally:
        if reader is not None:
            reader.close()
        try:
            ep_robot.close()
        except Exception:
            pass

    # Manual/fallback path, or an auto-mode Ctrl+C before it reached its own
    # 'q' save, still need a save here; the normal interactive 'q' path does
    # not, to avoid printing the same save twice.
    if already_saved:
        return 0
    if not save_points(points, args.out, source_label):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
