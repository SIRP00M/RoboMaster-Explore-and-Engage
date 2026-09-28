#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Infrared Blaster and target fire policy configuration."""

from config.maze import GRID_TILE_MM

TARGET_AUTO_FIRE_ENABLED = True
TARGET_FIRE_TYPE_NAME = "INFRARED"
TARGET_INFRARED_SHOTS = 1
TARGET_FIRE_ONCE_PER_TARGET = True
TARGET_FIRE_SETTLE_SEC = 0.12

# Range gate constraint: fire only when <= 2 physical floor tiles (120 cm)
TARGET_FIRE_MAX_TILES = 2.0
TARGET_FIRE_MAX_RANGE_MM = GRID_TILE_MM * TARGET_FIRE_MAX_TILES
TARGET_FIRE_RANGE_SAMPLES = 7
TARGET_FIRE_RANGE_RECHECK_SEC = 0.05

TARGET_FIRE_COLORS = ("RED", "YELLOW", "GREEN", "BLUE")
TARGET_FIRE_SHAPES = (
    "SQUARE",
    "RECT_VERTICAL",
    "RECT_HORIZONTAL",
    "CIRCLE",
)
