#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sensor sampling, filtering, ADC conversion, and event latching."""

import math
import statistics
import time

from config import *
from src.core.geometry import wrap_deg
from src.hardware.sensors import adc_to_cm


class SensorManagerMixin:
    def sample_tof_median(self, samples=TOF_SCAN_SAMPLES):
        values = []

        for _ in range(samples):
            v = self.state.get_tof()

            if v is not None and math.isfinite(v) and v > 0:
                values.append(float(v))

            time.sleep(TOF_SCAN_INTERVAL_SEC)

        if not values:
            return None

        return float(statistics.median(values))


    def scan_tof_at_yaw(self, yaw_deg):
        self.gimbal_point_relative_from_front(yaw_deg)
        return self.sample_tof_median()


    def read_sharp_adc(self):
        left = None
        right = None

        try:
            left = self.sensor_adapter.get_adc(
                id=SHARP_LEFT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        try:
            right = self.sensor_adapter.get_adc(
                id=SHARP_RIGHT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        if left is not None:
            try:
                self.left_adc_hist.append(float(left))
            except Exception:
                pass

        if right is not None:
            try:
                self.right_adc_hist.append(float(right))
            except Exception:
                pass

        left_med = (
            statistics.median(self.left_adc_hist)
            if self.left_adc_hist else None
        )
        right_med = (
            statistics.median(self.right_adc_hist)
            if self.right_adc_hist else None
        )

        return left_med, right_med


    def read_sharp_cm(self):
        left_adc, right_adc = self.read_sharp_adc()

        left_cm = adc_to_cm(left_adc, LEFT_CAL)
        right_cm = adc_to_cm(right_adc, RIGHT_CAL)

        return left_cm, right_cm, left_adc, right_adc


    def read_ir_once(self):
        """
        Returns:
            left_low, right_low, left_raw, right_raw

        Sensors are ACTIVE LOW:
            raw == 0 -> obstacle / wall
            raw == 1 -> clear
        """
        left_raw = None
        right_raw = None

        try:
            left_raw = self.sensor_adapter.get_io(
                id=IR_LEFT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        try:
            right_raw = self.sensor_adapter.get_io(
                id=IR_RIGHT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        left_low = (left_raw == 0)
        right_low = (right_raw == 0)

        return left_low, right_low, left_raw, right_raw


    def _update_ir_event_latch(self, left_low, right_low):
        """
        Record LOW edges and detect a dual-side sequence.

        A scan is requested when:
          - LEFT and RIGHT become LOW together, OR
          - one side becomes LOW while the other side is already LOW, OR
          - LEFT and RIGHT LOW edges occur within IR_DUAL_EVENT_WINDOW_SEC,
            even if the first side has already cleared.

        Continuous LOW does not repeatedly create new events; a new LOW edge
        is required after the previous event has been consumed.
        """
        now = time.monotonic()

        left_edge = bool(left_low and not self.ir_prev_left_low)
        right_edge = bool(right_low and not self.ir_prev_right_low)

        if left_edge:
            self.ir_last_left_event_t = now

        if right_edge:
            self.ir_last_right_event_t = now

        dual = False
        reason = None

        # Simultaneous / overlapping event.
        if (left_edge and right_low) or (right_edge and left_low):
            dual = True
            reason = "IR sides overlapped"

        # Sequential event: first side may already have cleared.
        if (
            self.ir_last_left_event_t is not None
            and self.ir_last_right_event_t is not None
        ):
            dt = abs(
                self.ir_last_left_event_t - self.ir_last_right_event_t
            )

            if dt <= IR_DUAL_EVENT_WINDOW_SEC:
                dual = True

                if reason is None:
                    if self.ir_last_left_event_t < self.ir_last_right_event_t:
                        reason = f"LEFT then RIGHT within {dt:.2f}s"
                    elif self.ir_last_right_event_t < self.ir_last_left_event_t:
                        reason = f"RIGHT then LEFT within {dt:.2f}s"
                    else:
                        reason = "LEFT + RIGHT same-time event"

        if dual:
            self.ir_dual_sequence_pending = True
            self.ir_dual_sequence_reason = reason

        self.ir_prev_left_low = bool(left_low)
        self.ir_prev_right_low = bool(right_low)


    def consume_ir_dual_sequence(self):
        """
        Consume one latched dual-side event.

        The timestamps are cleared so the same pair of old edges cannot
        repeatedly force route scans.
        """
        if not self.ir_dual_sequence_pending:
            return False, None

        reason = self.ir_dual_sequence_reason

        self.ir_dual_sequence_pending = False
        self.ir_dual_sequence_reason = None
        self.ir_last_left_event_t = None
        self.ir_last_right_event_t = None

        return True, reason


    def read_ir_filtered(self, samples=IR_FILTER_SAMPLES):
        """
        Majority filter for digital IR.

        Besides returning current LOW states, this updates the edge/event latch
        used to detect LEFT->RIGHT or RIGHT->LEFT events that happen close
        together in time.
        """
        left_hits = 0
        right_hits = 0
        left_last = None
        right_last = None

        samples = max(1, int(samples))

        for _ in range(samples):
            l_low, r_low, l_raw, r_raw = self.read_ir_once()

            left_last = l_raw
            right_last = r_raw

            if l_low:
                left_hits += 1
            if r_low:
                right_hits += 1

            time.sleep(0.015)

        needed = samples // 2 + 1

        left_low = left_hits >= needed
        right_low = right_hits >= needed

        self._update_ir_event_latch(left_low, right_low)

        return (
            left_low,
            right_low,
            left_last,
            right_last,
        )


