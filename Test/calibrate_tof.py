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
ruler. The robot then backs itself up on odometry (chassis position +
attitude, holding heading straight) through the rest of --distances,
re-reading the ToF at each stop; the true distance used is where it
REALLY stopped (odometry), not the nominal target. This mirrors
calibrate_tof.py in the sibling maze-runner project (tools/calibrate_tof.py),
adapted to drive directly on the RoboMaster SDK instead of that project's
own motion stack.

    python Test/calibrate_tof.py                          # auto point 1 + self-driving backup
    python Test/calibrate_tof.py --distances 40 60 90 120  # only these (cm) after point 1
    python Test/calibrate_tof.py --manual                  # place the wall by hand for every point
    python Test/calibrate_tof.py --max 90                  # only 0.6 m clear behind it

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

CELL_SIZE_M = 0.60  # matches GRID_TILE_M in V16B.py
DEFAULT_DISTANCES_CM = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 120.0, 150.0)
DEFAULT_SAMPLES = 40
DEFAULT_PITCH_DEG = -5.0
TOF_FREQ_HZ = 20
VALID_MIN_MM = 20.0
VALID_MAX_MM = 4000.0
SIDE_WALL_MAX_FRAC = 0.75  # a side reading beyond this fraction of a cell isn't "the wall"

# Auto backup drive.
POSITION_FREQ_HZ = 20
ATTITUDE_FREQ_HZ = 20
BACKUP_SPEED_MPS = 0.09
BACKUP_SLOW_ZONE_M = 0.06
BACKUP_MIN_SPEED_MPS = 0.03
BACKUP_YAW_KP = 1.2
BACKUP_YAW_MAX_DPS = 25.0
BACKUP_ARRIVE_TOL_M = 0.01
BACKUP_TIMEOUT_SEC = 15.0


def wrap_deg(angle: float) -> float:
    angle = float(angle)
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


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


class BackupDriver:
    """Drives straight backward on odometry, holding heading with the IMU.

    Position is zeroed by the SDK at the moment sub_position() is called,
    so subscribing right after point 1 makes that position the origin: all
    later targets are driven to as an ABSOLUTE displacement from it (not
    chained from the previous stop), so odometry error doesn't compound
    hop to hop -- same principle as the sibling project's back_up().
    """

    def __init__(self, ep_robot):
        self._chassis = ep_robot.chassis
        self._pos_lock = threading.Lock()
        self._att_lock = threading.Lock()
        self._x: Optional[float] = None
        self._yaw: Optional[float] = None
        self._chassis.sub_position(freq=POSITION_FREQ_HZ, callback=self._on_position)
        self._chassis.sub_attitude(freq=ATTITUDE_FREQ_HZ, callback=self._on_attitude)
        self.base_yaw = self._wait_for_yaw()

    def _on_position(self, info) -> None:
        if info and len(info) >= 1:
            with self._pos_lock:
                self._x = float(info[0])

    def _on_attitude(self, info) -> None:
        if info and len(info) >= 1:
            with self._att_lock:
                self._yaw = float(info[0])

    def _wait_for_yaw(self, timeout: float = 2.0) -> Optional[float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._att_lock:
                if self._yaw is not None:
                    return self._yaw
            time.sleep(0.02)
        return None

    def traveled_back_m(self) -> float:
        with self._pos_lock:
            x = self._x
        return 0.0 if x is None else -x

    def back_up_to(self, target_total_m: float) -> Optional[float]:
        """Drive until traveled_back_m() reaches target_total_m (absolute,
        from this driver's origin). Returns the real traveled distance."""
        deadline = time.monotonic() + BACKUP_TIMEOUT_SEC
        while time.monotonic() < deadline:
            traveled = self.traveled_back_m()
            remaining = target_total_m - traveled
            if remaining <= BACKUP_ARRIVE_TOL_M:
                break
            speed = BACKUP_SPEED_MPS if remaining > BACKUP_SLOW_ZONE_M else max(
                BACKUP_MIN_SPEED_MPS, BACKUP_SPEED_MPS * remaining / BACKUP_SLOW_ZONE_M
            )
            with self._att_lock:
                yaw = self._yaw
            z = 0.0
            if yaw is not None and self.base_yaw is not None:
                err = wrap_deg(self.base_yaw - yaw)
                z = max(-BACKUP_YAW_MAX_DPS, min(BACKUP_YAW_MAX_DPS, BACKUP_YAW_KP * err))
            self._chassis.drive_speed(x=-speed, y=0.0, z=z, timeout=0.3)
            time.sleep(0.04)
        self._chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.3)
        time.sleep(0.15)
        return self.traveled_back_m()

    def close(self) -> None:
        try:
            self._chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.3)
        except Exception:
            pass
        try:
            self._chassis.unsub_position()
        except Exception:
            pass
        try:
            self._chassis.unsub_attitude()
        except Exception:
            pass


def auto_first_point(ep_robot, reader: ToFReader, pitch_deg: float, samples: int) -> Optional[Tuple[float, float]]:
    """(raw_mm, true_mm) for the wall dead ahead, found from the side walls.

    Returns None if there are no usable walls on both sides (open floor,
    doorway, etc.) -- caller should fall back to fully manual placement.
    """
    gimbal = ep_robot.gimbal
    limit_mm = CELL_SIZE_M * 1000.0 * SIDE_WALL_MAX_FRAC

    gimbal.moveto(pitch=pitch_deg, yaw=-90.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
    left = reader.median(samples)
    gimbal.moveto(pitch=pitch_deg, yaw=90.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
    right = reader.median(samples)
    gimbal.moveto(pitch=pitch_deg, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()

    if left is None or right is None or left > limit_mm or right > limit_mm:
        return None

    offset_mm = (CELL_SIZE_M * 1000.0 - (left + right)) / 2.0
    front = reader.median(samples)
    if front is None:
        return None
    true_mm = front + offset_mm
    print(
        f"[AUTO] side walls left={left:.1f} right={right:.1f} mm -> "
        f"ToF offset {offset_mm:+.1f} mm; wall ahead raw={front:.1f} -> true={true_mm:.1f} mm"
    )
    return front, true_mm


def auto_collect(
    ep_robot, reader: ToFReader, pitch_deg: float, samples: int,
    start_true_cm: float, distances_cm: List[float], points: Dict[float, float],
) -> None:
    """Self-driving backup: fills ``points`` for every distance > start_true_cm."""
    targets = [d for d in distances_cm if d > start_true_cm + 2.0]
    if not targets:
        return

    span_m = (max(targets) - start_true_cm) / 100.0
    print(f"\nIt will back up about {span_m:.2f} m from here: keep that much clear behind it.")
    input("Press Enter to start the self-driving backup (Ctrl+C to abort)...")

    driver = BackupDriver(ep_robot)
    try:
        for true_cm in targets:
            target_total_m = (true_cm - start_true_cm) / 100.0
            traveled_m = driver.back_up_to(target_total_m)
            real_true_cm = start_true_cm + (traveled_m or 0.0) * 100.0
            ep_robot.gimbal.moveto(pitch=pitch_deg, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()
            raw = reader.median(samples)
            if raw is None:
                print(f"  {real_true_cm:.1f} cm: no fresh ToF reading -- skipped")
                continue
            print(f"  {real_true_cm:.1f} cm (target {true_cm:.1f}): raw = {raw:.1f} mm "
                  f"(shift {real_true_cm * 10.0 - raw:+.1f})")
            points[round(real_true_cm, 2)] = raw
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
    return table


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
    ap.add_argument("--distances", type=float, nargs="+", default=list(DEFAULT_DISTANCES_CM),
                     help="true distances in cm to add AFTER the auto/first point")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_DEG)
    ap.add_argument("--manual", action="store_true", help="place the wall by hand for every point (no self-driving)")
    ap.add_argument("--max", type=float, default=None, help="drop distances beyond this cm (little room behind it)")
    ap.add_argument("--out", default="calibration/tof.json")
    args = ap.parse_args()

    if robot is None:
        raise RuntimeError("RoboMaster SDK not found in the active Python environment")

    points: Dict[float, float] = {}
    distances = sorted(d for d in args.distances if args.max is None or d <= args.max + 1e-9)

    print("=" * 68)
    print(" ToF RAW-VS-TRUE DISTANCE CALIBRATION")
    print("=" * 68)
    if args.manual:
        print(f"Place a wall at each of {distances} cm from the gimbal by hand.")
    else:
        print("Stand the robot centred in a cell with walls LEFT, RIGHT and FRONT,")
        print("square to them: point 1 comes from those walls, then it backs itself")
        print(f"up through {distances} cm on odometry. Falls back to manual placement")
        print("if there are no side walls to find point 1 from.")
    print("R=record | Enter=accept/next | Q=save & quit  (manual points only)")
    print("=" * 68)

    ep_robot = robot.Robot()
    reader = None
    try:
        print(f"\n[CONNECT] conn_type='{args.conn_type}' ...")
        ep_robot.initialize(conn_type=args.conn_type)
        ep_robot.gimbal.recenter().wait_for_completed()
        reader = ToFReader(ep_robot)
        print("[CONNECT] RoboMaster connected, ToF streaming")

        start_true_cm = None
        if not args.manual:
            found = auto_first_point(ep_robot, reader, args.pitch, args.samples)
            if found is not None:
                raw, true_mm = found
                start_true_cm = round(true_mm / 10.0, 2)
                points[start_true_cm] = raw
            else:
                print("[AUTO] no usable side walls -- falling back to fully manual placement")

        ep_robot.gimbal.moveto(pitch=args.pitch, yaw=0.0, pitch_speed=60, yaw_speed=90).wait_for_completed()

        if start_true_cm is not None:
            auto_collect(ep_robot, reader, args.pitch, args.samples, start_true_cm, distances, points)
            remaining = [d for d in distances if d not in points and d <= start_true_cm + 2.0]
            if remaining:
                print(f"\n{remaining} cm (at/before point 1) still need manual placement:")
                collect(reader, remaining, args.samples, points)
        else:
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

    if len(points) < 2:
        print(f"\nOnly {len(points)} point(s) accepted; need >= 2. Nothing saved.")
        return 1

    table = build_table(points)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "points": [{"raw_mm": r, "true_mm": d} for r, d in table],
        "source": "python Test/calibrate_tof.py" + (" --manual" if args.manual else ""),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    snippet = format_snippet(table)
    snippet_path = out_path.with_suffix(".py")
    snippet_path.write_text(snippet + "\n")

    print(f"\nSaved {len(table)} points to {out_path}")
    print(f"Snippet written to {snippet_path} -- paste into V16B.py next to LEFT_CAL/RIGHT_CAL:\n")
    print(snippet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
