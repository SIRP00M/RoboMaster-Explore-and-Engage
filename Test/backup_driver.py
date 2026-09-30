#!/usr/bin/env python3
"""Shared straight-backward odometry drive, used by calibrate_tof.py and
calibrate_vision.py's 'n' (back up and resample) key.

Kept in one place after the same bug (see BackupDriver's docstring) had to
be fixed in two near-identical copies -- a second fix would be just as
easy to apply to only one of them again.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

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


class BackupDriver:
    """Drives straight backward on odometry, holding heading with the IMU.

    chassis.sub_position() does NOT zero itself at subscribe time -- it
    reports whatever the chassis has accumulated since power-on/last reset
    (same reason V16B.py's own position_callback rebases to its first
    sample). This rebases to its own first sample as the origin, so every
    back_up_to() target is an ABSOLUTE displacement from THAT point (not
    chained from the previous stop), so odometry error doesn't compound
    call to call. Skipping this rebase previously let a backup drive well
    past its intended stop, since "remaining" was measured against a
    leftover offset instead of "since now".
    """

    def __init__(self, ep_robot):
        self._chassis = ep_robot.chassis
        self._pos_lock = threading.Lock()
        self._att_lock = threading.Lock()
        self._x0: Optional[float] = None  # raw x at first sample -- our origin
        self._x: Optional[float] = None
        self._yaw: Optional[float] = None
        self._chassis.sub_position(freq=POSITION_FREQ_HZ, callback=self._on_position)
        self._chassis.sub_attitude(freq=ATTITUDE_FREQ_HZ, callback=self._on_attitude)
        self.base_yaw = self._wait_for_yaw()

    def _on_position(self, info) -> None:
        if info and len(info) >= 1:
            x = float(info[0])
            with self._pos_lock:
                if self._x0 is None:
                    self._x0 = x
                self._x = x

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
            x, x0 = self._x, self._x0
        return 0.0 if x is None or x0 is None else x0 - x

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
