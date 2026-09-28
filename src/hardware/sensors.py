#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sensor helper functions including Sharp ADC-to-cm interpolation."""

import math
from config.calibration import SHARP_MIN_PLAUSIBLE_ADC


def adc_to_cm(adc, calibration):
    """
    Piecewise-linear interpolation using monotonic calibration points.

    Returns:
        float cm : within reliable calibrated region
        None     : ADC says wall is farther than reliable calibration region,
                   sensor value is invalid, or wall is effectively unavailable
                   for wall-follow.
    """
    if adc is None:
        return None

    try:
        adc = float(adc)
    except Exception:
        return None

    if not math.isfinite(adc) or adc < SHARP_MIN_PLAUSIBLE_ADC:
        return None

    # Calibration is near -> far, ADC high -> low.
    near_cm, near_adc = calibration[0]
    far_cm, far_adc = calibration[-1]

    # Closer than nearest calibration point.
    if adc >= near_adc:
        return near_cm

    # Farther than reliable calibrated range.
    if adc < far_adc:
        return None

    for i in range(len(calibration) - 1):
        d1, a1 = calibration[i]
        d2, a2 = calibration[i + 1]

        # a1 >= adc >= a2
        if a1 >= adc >= a2:
            if abs(a1 - a2) < 1e-9:
                return (d1 + d2) * 0.5

            t = (a1 - adc) / (a1 - a2)
            return d1 + t * (d2 - d1)

    return None
