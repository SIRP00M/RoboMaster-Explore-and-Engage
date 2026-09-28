#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Thread-safe telemetry state container for sensor feedback and chassis attitude."""

import threading


class SharedState:
    """Synchronized shared state for robot sensors and odometry telemetry."""

    def __init__(self):
        self.lock = threading.Lock()

        self.tof_mm = None
        self.position = None          # (x, y, z)
        self.attitude = None          # (yaw, pitch, roll)

        # Gimbal feedback:
        # (pitch relative chassis, yaw relative chassis,
        #  pitch ground, yaw ground)
        self.gimbal_angle = None

        # Raw chassis ESC / motor encoder-level telemetry.
        self.esc_speed = None         # rpm[4]
        self.esc_angle = None         # motor-angle raw[4]
        self.esc_timestamp = None
        self.esc_state = None

        # Chassis status flags, including slip flag.
        self.chassis_status = None

    def set_tof(self, value):
        with self.lock:
            self.tof_mm = value

    def get_tof(self):
        with self.lock:
            return self.tof_mm

    def set_position(self, value):
        with self.lock:
            self.position = value

    def get_position(self):
        with self.lock:
            return self.position

    def set_attitude(self, value):
        with self.lock:
            self.attitude = value

    def get_attitude(self):
        with self.lock:
            return self.attitude

    def set_gimbal_angle(self, value):
        with self.lock:
            self.gimbal_angle = value

    def get_gimbal_angle(self):
        with self.lock:
            return self.gimbal_angle

    def set_esc(self, speed, angle, timestamp, state):
        with self.lock:
            self.esc_speed = tuple(speed) if speed is not None else None
            self.esc_angle = tuple(angle) if angle is not None else None
            self.esc_timestamp = timestamp
            self.esc_state = state

    def get_esc(self):
        with self.lock:
            return (
                self.esc_speed,
                self.esc_angle,
                self.esc_timestamp,
                self.esc_state,
            )

    def set_chassis_status(self, value):
        with self.lock:
            self.chassis_status = tuple(value) if value is not None else None

    def get_chassis_status(self):
        with self.lock:
            return self.chassis_status
