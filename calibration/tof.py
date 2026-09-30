TOF_CAL = [  # (raw_mm, true_mm) -- python Test/calibrate_tof.py
    (125.0, 286.5),
    (837.0, 880.0),
    (1481.0, 1479.8),
]


def tof_calibrated_mm(raw_mm, table=TOF_CAL):
    """raw ToF mm -> corrected mm; same interpolate/extrapolate-by-shift as adc_to_cm."""
    if raw_mm is None:
        return None
    if raw_mm <= table[0][0]:
        return raw_mm + (table[0][1] - table[0][0])
    if raw_mm >= table[-1][0]:
        return raw_mm + (table[-1][1] - table[-1][0])
    for (r1, d1), (r2, d2) in zip(table, table[1:]):
        if r1 <= raw_mm <= r2:
            return d1 + (d2 - d1) * (raw_mm - r1) / (r2 - r1)
    return raw_mm
