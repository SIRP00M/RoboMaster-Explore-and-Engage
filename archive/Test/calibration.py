#!/usr/bin/env python3
"""
RoboMaster Sharp GP2Y0A41SK0F (41SK) raw calibration collector

LEFT  Sharp -> Sensor Adapter ID 2, Port 1, AD
RIGHT Sharp -> Sensor Adapter ID 3, Port 1, AD

Default sequence: 4, 6, 8, ... 30 cm
R = record/re-record 100 samples
Enter = accept current capture and continue
Q = save progress and quit

No median/EMA/distance conversion is applied while collecting; raw ADC is saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from robomaster import robot
except ModuleNotFoundError:
    robot = None

SENSOR_MODEL = "Sharp GP2Y0A41SK0F"
NOMINAL_MIN_CM = 4
NOMINAL_MAX_CM = 30

LEFT_SENSOR_ID = 2
RIGHT_SENSOR_ID = 3
SENSOR_PORT = 1

# 41SK is a short-range 4-30 cm sensor. 2 cm spacing gives a denser LUT
# than the old 21YK calibration and better captures the nonlinear near field.
DEFAULT_START_CM = 4
DEFAULT_STOP_CM = 30
DEFAULT_STEP_CM = 2

DEFAULT_SAMPLE_COUNT = 100
DEFAULT_SAMPLE_INTERVAL_SEC = 0.02


def read_key() -> str:
    """Read one key without requiring Enter first."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "ENTER"
        return ch.upper()

    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\r", "\n"):
        return "ENTER"
    return ch.upper()


def make_summary(values: List[float]) -> Dict[str, float]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) >= 2 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def collect_samples(sensor_adapter, sensor_id: int, count: int, interval_sec: float) -> List[dict]:
    rows = []
    print(f"\n[RECORD] ID={sensor_id}, Port={SENSOR_PORT}, samples={count}")
    print("[RECORD] Keep wall/target still...")

    for i in range(count):
        ts = time.time()
        raw = sensor_adapter.get_adc(id=sensor_id, port=SENSOR_PORT)
        adc = float(raw)
        rows.append({
            "sample_index": i + 1,
            "timestamp_unix": ts,
            "adc_raw": adc,
        })

        if (i + 1) % 10 == 0 or i == 0 or i + 1 == count:
            print(f"\r  {i + 1:3d}/{count}  ADC={adc:8.3f}", end="", flush=True)
        if interval_sec > 0:
            time.sleep(interval_sec)

    print()
    return rows


def save_outputs(output_dir: Path, session_name: str, records: List[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{session_name}.json"
    json_path.write_text(
        json.dumps({"metadata": metadata, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    raw_csv = output_dir / f"{session_name}_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["side", "sensor_id", "port", "distance_cm", "sample_index", "timestamp_unix", "adc_raw"])
        for rec in records:
            for sample in rec["samples"]:
                w.writerow([
                    rec["side"], rec["sensor_id"], rec["port"], rec["distance_cm"],
                    sample["sample_index"], f'{sample["timestamp_unix"]:.6f}', sample["adc_raw"]
                ])

    summary_csv = output_dir / f"{session_name}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["side", "sensor_id", "port", "distance_cm", "count", "mean_adc", "median_adc", "stdev_adc", "min_adc", "max_adc"])
        for rec in records:
            s = rec["summary"]
            w.writerow([
                rec["side"], rec["sensor_id"], rec["port"], rec["distance_cm"],
                s["count"], s["mean"], s["median"], s["stdev"], s["min"], s["max"]
            ])


def build_distances(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")
    values = list(range(start, stop + 1, step))
    if values[-1] != stop:
        values.append(stop)
    return values


def main() -> None:
    p = argparse.ArgumentParser(description="Collect raw Sharp GP2Y0A41SK0F (41SK) ADC calibration data via RoboMaster Sensor Adapter")
    p.add_argument("--conn-type", default="ap", choices=("ap", "sta", "rndis"))
    p.add_argument("--start", type=int, default=DEFAULT_START_CM)
    p.add_argument("--stop", type=int, default=DEFAULT_STOP_CM)
    p.add_argument("--step", type=int, default=DEFAULT_STEP_CM)
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    p.add_argument("--interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC)
    p.add_argument("--output-dir", default="sharp_calibration_runs")
    args = p.parse_args()

    if robot is None:
        raise RuntimeError("RoboMaster SDK not found in the active Python environment")

    distances = build_distances(args.start, args.stop, args.step)
    output_dir = Path(args.output_dir)
    now = datetime.now()
    session_name = f"gp2y0a41sk0f_calibration_{now.strftime('%Y%m%d_%H%M%S')}"

    metadata = {
        "created_at": now.isoformat(timespec="seconds"),
        "sensor_model": SENSOR_MODEL,
        "connection": args.conn_type,
        "left_sensor_id": LEFT_SENSOR_ID,
        "right_sensor_id": RIGHT_SENSOR_ID,
        "sensor_port": SENSOR_PORT,
        "distance_sequence_cm": distances,
        "samples_per_distance": args.samples,
        "sample_interval_sec": args.interval,
        "nominal_range_cm": [NOMINAL_MIN_CM, NOMINAL_MAX_CM],
        "note": "Raw ADC only. GP2Y0A41SK0F nominal measuring range is 4-30 cm.",
    }

    sides = [("LEFT", LEFT_SENSOR_ID), ("RIGHT", RIGHT_SENSOR_ID)]
    records: List[dict] = []

    print("=" * 68)
    print(" SHARP GP2Y0A41SK0F (41SK) RAW CALIBRATION")
    print("=" * 68)
    print(f"LEFT  : ID {LEFT_SENSOR_ID}, Port {SENSOR_PORT}")
    print(f"RIGHT : ID {RIGHT_SENSOR_ID}, Port {SENSOR_PORT}")
    print(f"Distances: {distances}")
    print(f"Samples/point: {args.samples}")
    print("R=record | Enter=accept/next | Q=save & quit")
    print("Nominal sensor range: 4-30 cm")
    print("=" * 68)

    ep_robot = robot.Robot()

    try:
        print(f"\n[CONNECT] conn_type='{args.conn_type}' ...")
        ep_robot.initialize(conn_type=args.conn_type)
        sensor_adapter = ep_robot.sensor_adaptor
        print("[CONNECT] RoboMaster connected")

        for side_name, sensor_id in sides:
            print("\n" + "#" * 68)
            print(f" {side_name} SENSOR - ID={sensor_id}, Port={SENSOR_PORT}")
            print("#" * 68)
            if side_name == "RIGHT":
                print("LEFT complete. Move setup to RIGHT sensor.")

            for distance_cm in distances:
                while True:
                    print(f"\n[{side_name}] Set wall/target to {distance_cm} cm")
                    print("Press R to record, or Q to save & quit")
                    key = read_key()

                    if key == "Q":
                        save_outputs(output_dir, session_name, records, metadata)
                        print(f"\n[SAVED] {output_dir.resolve()}")
                        return
                    if key != "R":
                        continue

                    samples = collect_samples(sensor_adapter, sensor_id, args.samples, args.interval)
                    values = [float(x["adc_raw"]) for x in samples]
                    summary = make_summary(values)
                    latest = {
                        "side": side_name,
                        "sensor_id": sensor_id,
                        "port": SENSOR_PORT,
                        "distance_cm": distance_cm,
                        "summary": summary,
                        "samples": samples,
                    }

                    print(f"\n[DONE] {side_name} @ {distance_cm} cm")
                    print(f"  mean   = {summary['mean']:.3f}")
                    print(f"  median = {summary['median']:.3f}")
                    print(f"  stdev  = {summary['stdev']:.3f}")
                    print(f"  min/max= {summary['min']:.3f} / {summary['max']:.3f}")
                    print("Enter = accept & next | R = re-record | Q = save & quit")

                    while True:
                        confirm = read_key()
                        if confirm == "ENTER":
                            records = [r for r in records if not (r["side"] == side_name and r["distance_cm"] == distance_cm)]
                            records.append(latest)
                            save_outputs(output_dir, session_name, records, metadata)
                            print(f"\n[ACCEPTED] {side_name} {distance_cm} cm")
                            accepted = True
                            break
                        if confirm == "R":
                            accepted = False
                            break
                        if confirm == "Q":
                            save_outputs(output_dir, session_name, records, metadata)
                            print(f"\n[SAVED] {output_dir.resolve()}")
                            return

                    if accepted:
                        break
                    # otherwise loop and re-record the same distance

        save_outputs(output_dir, session_name, records, metadata)
        print("\n" + "=" * 68)
        print(" CALIBRATION CAPTURE COMPLETE")
        print("=" * 68)
        print(f"Measurement points: {len(records)}")
        print(f"Saved in: {output_dir.resolve()}")
        print(f"  {session_name}.json")
        print(f"  {session_name}_raw.csv")
        print(f"  {session_name}_summary.csv")

    except KeyboardInterrupt:
        print("\n[CTRL+C] Saving progress...")
        save_outputs(output_dir, session_name, records, metadata)
        print(f"[SAVED] {output_dir.resolve()}")
    finally:
        try:
            ep_robot.close()
            print("[DISCONNECT] RoboMaster closed")
        except Exception:
            pass


if __name__ == "__main__":
    main()
