#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharp GP2Y0A41SK0F calibration lookup tables (distance in cm vs ADC reading).

Measured median ADC from 100-sample calibration dataset.
Only the monotonic/reliable section (4 - 24 cm) is used for control.
Values below the lowest ADC reading (~24 cm) are treated as "wall too far".
"""

LEFT_CAL = [
    (4.0, 864.5),
    (6.0, 610.0),
    (8.0, 475.0),
    (10.0, 378.0),
    (12.0, 316.0),
    (14.0, 274.0),
    (16.0, 247.0),
    (18.0, 224.0),
    (20.0, 212.0),
    (22.0, 198.0),
    (24.0, 182.0),
]

RIGHT_CAL = [
    (4.0, 822.0),
    (6.0, 589.0),
    (8.0, 455.0),
    (10.0, 374.0),
    (12.0, 313.0),
    (14.0, 276.0),
    (16.0, 239.0),
    (18.0, 205.0),
    (20.0, 174.0),
    (22.0, 161.0),
    (24.0, 133.0),
]

SHARP_FILTER_SAMPLES = 5
SHARP_MIN_PLAUSIBLE_ADC = 20
