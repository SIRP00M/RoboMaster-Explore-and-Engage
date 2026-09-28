#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Grid geometry, direction vectors, and angle normalization utilities."""

DIR_NAMES = ("N", "E", "S", "W")

DIR_VEC = {
    0: (0, 1),   # N
    1: (1, 0),   # E
    2: (0, -1),  # S
    3: (-1, 0),  # W
}

REL_LEFT = -1
REL_FRONT = 0
REL_RIGHT = 1
REL_BACK = 2


def clamp(v, lo, hi):
    """Clamp value v into range [lo, hi]."""
    return max(lo, min(hi, v))


def wrap_deg(angle):
    """Normalize angle in degrees into (-180.0, 180.0]."""
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def direction_between(a, b):
    """Return cardinal direction (0=N, 1=E, 2=S, 3=W) from cell a to adjacent cell b."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    for d, (vx, vy) in DIR_VEC.items():
        if (dx, dy) == (vx, vy):
            return d

    raise ValueError(f"Cells are not adjacent: {a} -> {b}")


def neighbor(cell, direction):
    """Return coordinate tuple of neighboring cell in given direction."""
    dx, dy = DIR_VEC[direction]
    return (cell[0] + dx, cell[1] + dy)
