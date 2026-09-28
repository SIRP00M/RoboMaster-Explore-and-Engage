#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Maze graph topology, wall detection, boundary guards, SVG/ASCII renderers, and JSON storage."""

import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from config import *
from src.core.geometry import (
    wrap_deg, clamp, DIR_NAMES, DIR_VEC, neighbor, direction_between,
    REL_LEFT, REL_FRONT, REL_RIGHT, REL_BACK
)


class GridMapMixin:
    def is_known_maze_ingress_edge(self, cell, abs_dir):
        return (
            START_ANCHOR_ENABLED
            and tuple(cell) == tuple(self.root)
            and int(abs_dir) % 4
                == int(self.known_maze_ingress_dir) % 4
        )


    def capture_start_anchor_profile(self):
        """
        Classify startup geometry for logging/map metadata only.

        The launch direction does NOT depend on this classification.
        The robot is assumed to be manually placed facing into the maze.

        This deliberately supports:
          - corridor start
          - one-sided-wall start
          - fully open staging-area start
        """
        if not START_ANCHOR_ENABLED:
            return None

        if self.start_anchor_profile is not None:
            return self.start_anchor_profile

        self.gimbal_front_down()
        front_mm = self.sample_tof_median()

        left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

        # Get coarse L/R ToF without moving chassis.
        left_mm = self.scan_tof_at_yaw(-90.0)
        right_mm = self.scan_tof_at_yaw(+90.0)
        self.gimbal_front_down()

        left_sharp_wall = (
            left_cm is not None
            and left_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )
        right_sharp_wall = (
            right_cm is not None
            and right_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )

        left_open = (
            left_mm is not None
            and left_mm >= START_OPEN_MM
        )
        front_open = (
            front_mm is not None
            and front_mm >= START_OPEN_MM
        )
        right_open = (
            right_mm is not None
            and right_mm >= START_OPEN_MM
        )

        if left_sharp_wall and right_sharp_wall:
            start_type = "CORRIDOR"
        elif left_sharp_wall or right_sharp_wall:
            start_type = "ONE_SIDE_WALL"
        elif left_open and right_open:
            start_type = "OPEN_STAGING"
        else:
            start_type = "MIXED"

        self.start_anchor_profile = {
            "cell": [int(self.root[0]), int(self.root[1])],
            "type": start_type,
            "maze_ingress_dir_index": int(self.known_maze_ingress_dir),
            "maze_ingress_dir": DIR_NAMES[self.known_maze_ingress_dir],
            "entrance_return_dir_index": int(self.known_entrance_dir),
            "entrance_return_dir": DIR_NAMES[self.known_entrance_dir],
            "front_mm": (
                None if front_mm is None else float(front_mm)
            ),
            "left_tof_mm": (
                None if left_mm is None else float(left_mm)
            ),
            "right_tof_mm": (
                None if right_mm is None else float(right_mm)
            ),
            "sharp_left_cm": (
                None if left_cm is None else float(left_cm)
            ),
            "sharp_right_cm": (
                None if right_cm is None else float(right_cm)
            ),
            "front_open": bool(front_open),
            "left_open": bool(left_open),
            "right_open": bool(right_open),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }

        print("\n================ START ANCHOR PROFILE ======================")
        print(f"Start type    : {start_type}")
        print(
            f"Maze ingress  : {DIR_NAMES[self.known_maze_ingress_dir]} "
            "(FORCED from startup orientation)"
        )
        print(
            f"Return side   : {DIR_NAMES[self.known_entrance_dir]}"
        )
        print(
            f"ToF L/F/R     : "
            f"{left_mm}/{front_mm}/{right_mm} mm"
        )
        print(
            f"Sharp L/R     : "
            f"{left_cm if left_cm is not None else 'far'} / "
            f"{right_cm if right_cm is not None else 'far'} cm"
        )
        print(
            "[START ANCHOR] LEFT/RIGHT openness at (0,0) will NOT "
            "become DFS branches."
        )
        print("===========================================================")

        return self.start_anchor_profile


    def apply_root_start_policy(self, scanned_open_dirs):
        """
        Root topology override.

        Regardless of whether the starting area is a corridor or completely
        open, only the physically-known forward maze-ingress edge is allowed
        into DFS.  The back edge remains the known return/entrance direction.

        This prevents an open staging area from being interpreted as a
        3-way/4-way maze intersection.
        """
        if not START_ANCHOR_ENABLED:
            return list(scanned_open_dirs)

        if self.current != self.root:
            return list(scanned_open_dirs)

        forced = []

        ingress = self.known_maze_ingress_dir

        # We trust placement/orientation more than startup side geometry.
        if START_FORCE_FRONT_OPEN:
            forced.append(ingress)
        elif ingress in scanned_open_dirs:
            forced.append(ingress)

        print(
            f"[START ANCHOR] raw root openings="
            f"{[DIR_NAMES[d] for d in scanned_open_dirs]} "
            f"-> DFS openings="
            f"{[DIR_NAMES[d] for d in forced]}"
        )

        return forced


    def is_known_entrance_edge(self, cell, abs_dir):
        return (
            tuple(cell) == tuple(self.root)
            and int(abs_dir) % 4 == int(self.known_entrance_dir) % 4
        )


    def capture_entrance_corridor_profile(self):
        """
        Observe the physical entrance corridor once at startup.

        Important design rule:
        the entrance identity comes from START CONTEXT, not from its shape.
        An exit/fake-exit may look geometrically identical, so geometry alone
        cannot distinguish them reliably.

        The chassis stays still.  The gimbal scans behind the robot while
        stationary yaw hold prevents reaction-torque drift.
        """
        if not ENTRANCE_CORRIDOR_PROFILE_ENABLED:
            return None

        if self.entrance_corridor_profile is not None:
            return self.entrance_corridor_profile

        print("\n================ ENTRANCE CORRIDOR PROFILE ================")
        print(f"Root         : {self.root}")
        print(
            f"Known entry  : {DIR_NAMES[self.known_entrance_dir]} "
            "(back of startup heading)"
        )
        print("Chassis      : HOLD POSITION")
        print("Gimbal       : scan entrance behind robot")
        print("===========================================================")

        self.stop_chassis()

        # Current side-wall signature at the root.
        left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

        fan = {}

        for angle in ENTRANCE_PROFILE_ANGLES_DEG:
            mm = self.scan_tof_at_yaw(angle)
            fan[angle] = mm

            state = (
                "OPEN"
                if mm is not None and mm >= ENTRANCE_OPEN_VERIFY_MM
                else "NEAR/CLOSED"
            )

            value_text = "NO DATA" if mm is None else f"{mm:.0f} mm"

            print(
                f"  [ENTRANCE] yaw={angle:+6.1f}° "
                f"{value_text:>10} -> {state}"
            )

        self.gimbal_front_down()

        back_mm = fan.get(+180.0)

        left_wall = (
            left_cm is not None
            and left_cm <= ENTRANCE_CORRIDOR_SIDE_WALL_MAX_CM
        )
        right_wall = (
            right_cm is not None
            and right_cm <= ENTRANCE_CORRIDOR_SIDE_WALL_MAX_CM
        )

        back_open = (
            back_mm is not None
            and back_mm >= ENTRANCE_OPEN_VERIFY_MM
        )

        # "corridor-like" is descriptive only.  It is NOT required to trust
        # the entrance because the entrance is known from the start setup.
        corridor_like = bool(left_wall and right_wall)

        self.entrance_corridor_profile = {
            "cell": [int(self.root[0]), int(self.root[1])],
            "dir_index": int(self.known_entrance_dir),
            "dir": DIR_NAMES[self.known_entrance_dir],
            "status": "KNOWN_ENTRANCE_CORRIDOR",
            "trusted_from_start_context": True,
            "back_center_mm": (
                None if back_mm is None else float(back_mm)
            ),
            "back_open_verified": bool(back_open),
            "corridor_side_walls_verified": bool(corridor_like),
            "sharp_left_cm": (
                None if left_cm is None else float(left_cm)
            ),
            "sharp_right_cm": (
                None if right_cm is None else float(right_cm)
            ),
            "sharp_left_adc": (
                None if left_adc is None else float(left_adc)
            ),
            "sharp_right_adc": (
                None if right_adc is None else float(right_adc)
            ),
            "fan_mm": {
                str(angle): (
                    None if fan.get(angle) is None
                    else float(fan.get(angle))
                )
                for angle in ENTRANCE_PROFILE_ANGLES_DEG
            },
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }

        ltxt = "far" if left_cm is None else f"{left_cm:.1f} cm"
        rtxt = "far" if right_cm is None else f"{right_cm:.1f} cm"
        btxt = "NO DATA" if back_mm is None else f"{back_mm:.0f} mm"

        print(
            f"[ENTRANCE PROFILE] L={ltxt} R={rtxt} "
            f"BACK={btxt}"
        )
        print(
            f"[ENTRANCE PROFILE] corridor_side_walls="
            f"{corridor_like} back_open={back_open}"
        )
        print(
            "[ENTRANCE PROFILE] identity=KNOWN FROM START CONTEXT; "
            "DFS will never treat this edge as an exit candidate."
        )

        if not back_open:
            print(
                "[ENTRANCE WARN] back ray is not >= "
                f"{ENTRANCE_OPEN_VERIFY_MM:.0f} mm. "
                "This may be caused by start placement or an object behind; "
                "the entrance remains protected from DFS."
            )

        if MAP_AUTOSAVE:
            self.save_map(final=False)

        return self.entrance_corridor_profile


    def exit_candidate_key(self, cell, abs_dir):
        return (tuple(cell), int(abs_dir) % 4)


    def is_exit_candidate(self, cell, abs_dir):
        return self.exit_candidate_key(cell, abs_dir) in self.exit_candidates


    def record_exit_candidate(
        self,
        cell,
        abs_dir,
        front_mm,
        left_cm,
        right_cm,
        fan_mm=None,
        reason="wide_boundary",
        wall_end_probe=None,
    ):
        key = self.exit_candidate_key(cell, abs_dir)

        fan_mm = fan_mm or {}
        wall_end_probe = wall_end_probe or {}

        self.exit_candidates[key] = {
            "cell": [int(cell[0]), int(cell[1])],
            "dir_index": int(abs_dir) % 4,
            "dir": DIR_NAMES[int(abs_dir) % 4],
            "status": "DEFERRED_DURING_EXPLORE",
            "reason": str(reason),
            "front_mm": None if front_mm is None else float(front_mm),
            "sharp_left_cm": None if left_cm is None else float(left_cm),
            "sharp_right_cm": None if right_cm is None else float(right_cm),
            "fan_mm": {
                str(angle): (
                    None if fan_mm.get(angle) is None
                    else float(fan_mm.get(angle))
                )
                for angle in EXIT_FAN_ANGLES_DEG
                if angle in fan_mm
            },
            "wall_end_probe": wall_end_probe,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        print(
            f"[EXIT CANDIDATE] cell={cell} dir={DIR_NAMES[abs_dir]} "
            f"reason={reason} "
            "-> DEFERRED; DFS WILL NOT LEAVE MAZE HERE"
        )

        if MAP_AUTOSAVE:
            self.save_map(final=False)


    def predicted_side_wall_hit_mm(self, side_cm, angle_deg):
        """
        If a perfectly straight side wall continues forward forever, a ray at
        `angle_deg` should hit it at approximately:

            range = perpendicular_side_distance / sin(|angle|)

        Sharp provides the perpendicular side distance at the robot.
        """
        if side_cm is None:
            return None

        angle_rad = math.radians(abs(float(angle_deg)))
        sin_v = math.sin(angle_rad)

        if sin_v <= 1e-6:
            return None

        return float(side_cm) * 10.0 / sin_v


    def wall_end_ray_vote(self, measured_mm, expected_mm):
        """
        True when the ToF ray travelled substantially farther than it should
        have if the currently-detected side wall continued ahead.
        """
        if measured_mm is None or expected_mm is None:
            return False

        threshold = max(
            EXIT_WALL_END_MIN_MEASURED_MM,
            expected_mm * EXIT_WALL_END_RATIO,
            expected_mm + EXIT_WALL_END_MARGIN_MM,
        )

        return measured_mm >= threshold


    def scan_wall_end_probe(self, left_cm, right_cm):
        """
        Detect the exact geometry shown by the real fake-exit:

            |       |
            | ROBOT |
            |       |
            |       |
            END   END
               OPEN

        Sharp still sees both nearby side walls.  Shallow forward ToF rays
        reveal whether those walls continue beyond the next grid boundary.
        """
        evidence = {
            "left": [],
            "right": [],
        }

        print(
            "[EXIT END-PROBE] side walls exist NOW; checking whether "
            "they terminate together ahead"
        )

        for side, side_cm, angles in (
            ("left", left_cm, EXIT_WALL_END_ANGLES_LEFT),
            ("right", right_cm, EXIT_WALL_END_ANGLES_RIGHT),
        ):
            for angle in angles:
                expected = self.predicted_side_wall_hit_mm(
                    side_cm,
                    angle,
                )
                measured = self.scan_tof_at_yaw(angle)

                vote = self.wall_end_ray_vote(
                    measured,
                    expected,
                )

                threshold = None
                if expected is not None:
                    threshold = max(
                        EXIT_WALL_END_MIN_MEASURED_MM,
                        expected * EXIT_WALL_END_RATIO,
                        expected + EXIT_WALL_END_MARGIN_MM,
                    )

                rec = {
                    "angle_deg": float(angle),
                    "measured_mm": (
                        None if measured is None else float(measured)
                    ),
                    "expected_continuing_wall_mm": (
                        None if expected is None else float(expected)
                    ),
                    "wall_end_threshold_mm": (
                        None if threshold is None else float(threshold)
                    ),
                    "wall_ended_vote": bool(vote),
                }

                evidence[side].append(rec)

                mtxt = "NO DATA" if measured is None else f"{measured:.0f}"
                etxt = "N/A" if expected is None else f"{expected:.0f}"
                ttxt = "N/A" if threshold is None else f"{threshold:.0f}"

                print(
                    f"    [END {side.upper():5s}] "
                    f"{angle:+5.1f}° measured={mtxt:>7} "
                    f"expected_wall={etxt:>7} "
                    f"vote_threshold={ttxt:>7} "
                    f"-> {'END' if vote else 'CONTINUE'}"
                )

        self.gimbal_front_down()

        left_votes = sum(
            1 for rec in evidence["left"]
            if rec["wall_ended_vote"]
        )
        right_votes = sum(
            1 for rec in evidence["right"]
            if rec["wall_ended_vote"]
        )

        evidence["left_votes"] = left_votes
        evidence["right_votes"] = right_votes

        both_ended = (
            left_votes >= EXIT_WALL_END_MIN_VOTES_PER_SIDE
            and right_votes >= EXIT_WALL_END_MIN_VOTES_PER_SIDE
        )

        evidence["both_walls_ended"] = bool(both_ended)

        print(
            f"[EXIT END-PROBE] votes "
            f"LEFT={left_votes}/{len(EXIT_WALL_END_ANGLES_LEFT)} "
            f"RIGHT={right_votes}/{len(EXIT_WALL_END_ANGLES_RIGHT)} "
            f"-> {'BOTH WALLS END' if both_ended else 'CORRIDOR CONTINUES'}"
        )

        return both_ended, evidence


    def scan_exit_fan(self):
        """
        Chassis is already facing the selected DFS direction.
        Scan a wide front fan with the same robust 7-sample median.
        """
        fan = {}

        for angle in EXIT_FAN_ANGLES_DEG:
            mm = self.scan_tof_at_yaw(angle)
            fan[angle] = mm

            state = (
                "OPEN"
                if mm is not None and mm >= EXIT_FAN_OPEN_MM
                else "BLOCKED"
            )

            value_text = "NO DATA" if mm is None else f"{mm:.0f} mm"

            print(
                f"    [EXIT FAN] {angle:+5.1f} deg "
                f"{value_text:>10} -> {state}"
            )

        self.gimbal_front_down()
        return fan


    def exit_guard_before_explore_edge(self, cell, abs_dir):
        """
        Conservative PRE-MOVE exit guard.

        V8.9 rule:
        NEVER classify a normal corridor as an exit merely because shallow
        rays predict that its current side walls end somewhere ahead.

        Before motion we only defer an edge when the robot is ALREADY sitting
        at an obviously broad/open boundary with BOTH side corridor walls gone.
        The common photo-like fake-exit case (walls still beside robot, ending
        ahead) is handled by the IN-MOTION guard inside move_one_cell().
        """
        if self.is_known_entrance_edge(cell, abs_dir):
            print(
                f"[ENTRANCE GUARD] {cell}->{DIR_NAMES[abs_dir]} "
                "is KNOWN_ENTRANCE_CORRIDOR"
            )
            return False

        if self.is_known_maze_ingress_edge(cell, abs_dir):
            print(
                f"[START ANCHOR] {cell}->{DIR_NAMES[abs_dir]} "
                "is KNOWN MAZE INGRESS -> bypass exit classifier"
            )
            return False

        if not EXIT_GUARD_ENABLED:
            return False

        if self.is_exit_candidate(cell, abs_dir):
            print(
                f"[EXIT GUARD] {cell}->{DIR_NAMES[abs_dir]} "
                "already recorded as EXIT_CANDIDATE"
            )
            return True

        self.gimbal_front_down()

        front_mm = self.sample_tof_median()
        left_cm, right_cm, _, _ = self.read_sharp_cm()

        left_wall = (
            left_cm is not None
            and left_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )
        right_wall = (
            right_cm is not None
            and right_cm <= EXIT_CORRIDOR_WALL_MAX_CM
        )

        left_text = "far" if left_cm is None else f"{left_cm:.1f}"
        right_text = "far" if right_cm is None else f"{right_cm:.1f}"
        front_text = "NO DATA" if front_mm is None else f"{front_mm:.0f}"

        rel_left_abs = (abs_dir - 1) % 4
        rel_right_abs = (abs_dir + 1) % 4
        known_open = set(self.open_dirs.get(cell, []))

        side_topology_left_open = rel_left_abs in known_open
        side_topology_right_open = rel_right_abs in known_open

        print(
            f"[EXIT GUARD PRE] cell={cell} dir={DIR_NAMES[abs_dir]} "
            f"front={front_text}mm L={left_text}cm R={right_text}cm "
            f"sideTopo="
            f"{'OPEN' if side_topology_left_open else 'WALL'}/"
            f"{'OPEN' if side_topology_right_open else 'WALL'}"
        )

        if (
            front_mm is not None
            and front_mm < EXIT_FRONT_MIN_SAFE_MM
        ):
            print(
                "[EXIT GUARD PRE] close front object -> normal obstacle logic"
            )
            return False

        # THIS IS THE KEY FIX:
        # If either/both side walls still exist at the robot, allow the robot
        # to START moving.  We will watch whether BOTH walls disappear during
        # the edge traversal instead of guessing from projected geometry.
        if left_wall or right_wall:
            print(
                "[EXIT GUARD PRE] corridor wall exists NOW "
                "-> allow move; in-motion wall-loss monitor armed"
            )
            return False

        # A mapped intersection is not an outside boundary.
        if side_topology_left_open or side_topology_right_open:
            print(
                "[EXIT GUARD PRE] mapped side opening/intersection "
                "-> allow normal edge"
            )
            return False

        # Only the rare case where both walls are ALREADY absent before moving
        # gets the old broad fan verification.
        print(
            "[EXIT GUARD PRE] both side walls already absent "
            "-> broad fan verification"
        )

        fan = self.scan_exit_fan()

        fan_eval = self.evaluate_exit_fan(fan)
        left_open = fan_eval["left_open"]
        right_open = fan_eval["right_open"]
        strong_total = fan_eval["strong_total"]
        wide_boundary = fan_eval["wide_boundary"]

        print(
            f"[EXIT GUARD PRE] fan votes "
            f"OPEN L={left_open}/3 R={right_open}/3 "
            f"STRONG>={EXIT_FAN_STRONG_OPEN_MM:.0f}mm "
            f"{strong_total}/6 -> "
            f"{'BOUNDARY' if wide_boundary else 'JUNCTION/DEEP PATH'}"
        )

        if not wide_boundary:
            return False

        self.record_exit_candidate(
            cell=cell,
            abs_dir=abs_dir,
            front_mm=front_mm,
            left_cm=left_cm,
            right_cm=right_cm,
            fan_mm=fan,
            reason="broad_open_boundary_before_motion",
            wall_end_probe={},
        )
        return True


    def scan_cell(self, cell):
        """
        Scan LEFT / FRONT / RIGHT with the gimbal ToF.

        Normal maze classification:
            distance > TOF_OPEN_THRESHOLD_MM -> OPEN
            otherwise                         -> WALL

        Additional close-range dead-end override:
            LEFT <= DEAD_END_THRESHOLD_MM
            AND FRONT <= DEAD_END_THRESHOLD_MM
            AND RIGHT <= DEAD_END_THRESHOLD_MM

        When that close-range condition is true, DFS does not attempt any
        forward/side branch.  It immediately reverses toward the parent cell.
        """
        print(
            f"\n[SCAN] cell={cell} heading={DIR_NAMES[self.heading]}"
        )

        # ----------------------------------------------------
        # NODE-SCAN IR POLICY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # Once move_one_cell() has accepted that the robot has ARRIVED at a
        # cell/node, IR LOW is no longer treated as a command to slide.
        #
        # Why:
        #   - a wall can legitimately be very close at a junction/dead end
        #   - move_one_cell() may accept the cell near the front wall
        #   - sliding here can move the robot away from the intended node
        #   - it can also create the old conflict:
        #         "cell reached" -> IR slide -> front too close -> RuntimeError
        #
        # At a node, IR therefore acts only as a HIGH-PRIORITY TRIGGER:
        #     STOP -> keep pose -> Gimbal scan LEFT / FRONT / RIGHT
        #
        # Recovery/sliding is still used during translation, during an
        # explicit recovery slide, and after turns when corner clearance is
        # actually required.
        pre_l_low, pre_r_low, pre_l_raw, pre_r_raw = self.read_ir_filtered()
        pre_dual_event, pre_dual_reason = self.consume_ir_dual_sequence()

        if pre_dual_event:
            self.stop_chassis()
            print(
                f"[IR NODE SCAN {cell}] sequential event: "
                f"{pre_dual_reason} -> HOLD POSITION + GIMBAL L/F/R"
            )

        elif pre_l_low and pre_r_low:
            self.stop_chassis()
            print(
                f"[IR NODE SCAN {cell}] BOTH LOW "
                f"IR_L={pre_l_raw} IR_R={pre_r_raw} "
                "-> HOLD POSITION + GIMBAL L/F/R"
            )

        elif pre_l_low or pre_r_low:
            self.stop_chassis()

            side = "LEFT" if pre_l_low else "RIGHT"

            print(
                f"[IR NODE SCAN {cell}] {side} LOW "
                f"IR_L={pre_l_raw} IR_R={pre_r_raw} "
                "-> NO SLIDE; HOLD POSITION + GIMBAL L/F/R"
            )

        # A cell scan itself is a fresh replan, so consume any old hint.
        self.ir_replan_requested = False
        self.ir_route_hint = None

        relative_scans = [
            ("LEFT",  -90.0, REL_LEFT),
            ("FRONT",   0.0, REL_FRONT),
            ("RIGHT", +90.0, REL_RIGHT),
        ]

        ordered_open_dirs = []
        scan_mm = {}

        for label, yaw_deg, rel_dir in relative_scans:
            before_yaw = self.current_yaw()
            mm = self.scan_tof_at_yaw(yaw_deg)
            after_yaw = self.current_yaw()
            yaw_err = self.yaw_error_deg()

            scan_mm[label] = mm

            print(
                f"  [YAW] before={before_yaw} after={after_yaw} "
                f"target={self.yaw_ref_deg} err={yaw_err}"
            )

            if mm is None:
                is_open = False
                print(f"  {label:<5}: NO DATA -> CLOSED for safety")
            else:
                is_open = mm > TOF_OPEN_THRESHOLD_MM
                state = "OPEN" if is_open else "WALL"

                print(
                    f"  {label:<5}: {mm:7.1f} mm -> {state}"
                )

            abs_dir = (self.heading + rel_dir) % 4

            if is_open:
                ordered_open_dirs.append(abs_dir)

        # Save actual measurements for later debugging.
        self.cell_scan_mm[cell] = dict(scan_mm)

        # ----------------------------------------------------
        # HARD DEAD-END DETECTION
        # ----------------------------------------------------
        all_valid = all(
            scan_mm.get(k) is not None
            for k in ("LEFT", "FRONT", "RIGHT")
        )

        hard_dead_end = (
            all_valid
            and scan_mm["LEFT"] <= DEAD_END_THRESHOLD_MM
            and scan_mm["FRONT"] <= DEAD_END_THRESHOLD_MM
            and scan_mm["RIGHT"] <= DEAD_END_THRESHOLD_MM
        )

        if hard_dead_end:
            self.dead_end_cells.add(cell)

            print(
                "  [DEAD END] CLOSE WALLS ON ALL 3 SIDES "
                f"(threshold={DEAD_END_THRESHOLD_MM:.0f} mm)"
            )
            print(
                f"             L={scan_mm['LEFT']:.0f} "
                f"F={scan_mm['FRONT']:.0f} "
                f"R={scan_mm['RIGHT']:.0f} mm"
            )

            # Any apparent side/front OPEN caused by a bad ToF sample must not
            # be trusted once the explicit dead-end condition has fired.
            ordered_open_dirs = []

        else:
            self.dead_end_cells.discard(cell)

        # Parent direction is a guaranteed known connection because the robot
        # physically came through that edge.
        p = self.parent.get(cell)

        if p is not None:
            back_dir = direction_between(cell, p)

            if back_dir not in ordered_open_dirs:
                ordered_open_dirs.append(back_dir)

        elif not ROOT_BACK_IS_WALL:
            # Optional root back scan if the entrance must also be explored.
            mm = self.scan_tof_at_yaw(180.0)

            if mm is not None and mm > TOF_OPEN_THRESHOLD_MM:
                ordered_open_dirs.append(
                    (self.heading + REL_BACK) % 4
                )

        # Always put the turret physically back on the new chassis front and
        # restore pitch -5 degrees before any chassis motion.
        self.gimbal_front_down()

        print(
            "  open absolute dirs:",
            [DIR_NAMES[d] for d in ordered_open_dirs]
        )

        return ordered_open_dirs


    def detect_open_area_trap(self, cell):
        """
        Detect a pseudo-cell that is actually outside the maze.

        Trigger conditions are intentionally strong:
          - not root
          - cell has a parent (we just came through a known edge)
          - LEFT/FRONT/RIGHT all classified OPEN
          - neither Sharp sees a normal nearby side wall
          - wide diagonal fan is broadly open

        A legitimate intersection normally has nearby corner walls that stop
        several diagonal fan rays.  Open floor outside the maze usually does
        not.
        """
        if not OPEN_AREA_TRAP_ENABLED:
            return None

        if tuple(cell) == tuple(self.root):
            return None

        parent = self.parent.get(cell)

        if parent is None:
            return None

        scan = self.cell_scan_mm.get(cell, {})

        l = scan.get("LEFT")
        f = scan.get("FRONT")
        r = scan.get("RIGHT")

        all_three_open = all(
            value is not None and value >= TOF_OPEN_THRESHOLD_MM
            for value in (l, f, r)
        )

        if not all_three_open:
            return None

        left_cm, right_cm, _, _ = self.read_sharp_cm()

        left_wall = (
            left_cm is not None
            and left_cm <= OPEN_AREA_TRAP_SIDE_WALL_MAX_CM
        )
        right_wall = (
            right_cm is not None
            and right_cm <= OPEN_AREA_TRAP_SIDE_WALL_MAX_CM
        )

        if left_wall or right_wall:
            return None

        print(
            f"[OPEN AREA TRAP] suspicious cell={cell}: "
            f"L/F/R={l:.0f}/{f:.0f}/{r:.0f}mm, "
            "Sharp side walls absent -> WIDE FAN VERIFY"
        )

        fan = self.scan_exit_fan()

        fan_eval = self.evaluate_exit_fan(fan)

        open_count = sum(
            1
            for angle in EXIT_FAN_ANGLES_DEG
            if fan.get(angle) is not None
            and fan[angle] >= EXIT_FAN_OPEN_MM
        )

        long_count = sum(
            1
            for angle in EXIT_FAN_ANGLES_DEG
            if fan.get(angle) is not None
            and fan[angle] >= OPEN_AREA_TRAP_LONG_MM
        )

        broad_open = (
            fan_eval["wide_boundary"]
            and open_count >= OPEN_AREA_TRAP_MIN_OPEN_FAN_RAYS
            and long_count >= OPEN_AREA_TRAP_MIN_LONG_FAN_RAYS
        )

        print(
            f"[OPEN AREA TRAP] fan open={open_count}/6 "
            f"strong>={EXIT_FAN_STRONG_OPEN_MM:.0f}mm="
            f"{fan_eval['strong_total']}/6 "
            f"very-long>={OPEN_AREA_TRAP_LONG_MM:.0f}mm={long_count}/6 "
            f"-> {'OUTSIDE SUSPECTED' if broad_open else 'VALID INTERSECTION/DEEP PATH'}"
        )

        if not broad_open:
            return None

        return {
            "parent": parent,
            "left_cm": left_cm,
            "right_cm": right_cm,
            "scan": dict(scan),
            "fan": dict(fan),
        }


    def rollback_open_area_cell(self, cell, stack, evidence):
        """
        We already crossed one edge too far.  Do not explore another direction.
        Return through the exact edge we just used, mark that parent->cell
        opening as EXIT_CANDIDATE, and remove the outside pseudo-cell from map.
        """
        parent = evidence["parent"]
        incoming_dir = direction_between(parent, cell)
        back_dir = direction_between(cell, parent)

        print(
            f"[OPEN AREA TRAP] {cell} is treated as OUTSIDE. "
            f"Immediate return {cell}->{parent}; DO NOT EXPLORE SIDE BRANCHES."
        )

        self.record_exit_candidate(
            cell=parent,
            abs_dir=incoming_dir,
            front_mm=None,
            left_cm=evidence.get("left_cm"),
            right_cm=evidence.get("right_cm"),
            fan_mm=evidence.get("fan", {}),
            reason="open_area_detected_after_boundary_crossing",
            wall_end_probe={
                "outside_cell": [int(cell[0]), int(cell[1])],
                "outside_scan_mm": evidence.get("scan", {}),
            },
        )

        self.turn_to_direction(back_dir)

        ok = self.move_one_cell()

        if not ok:
            self.stop_chassis()
            raise RuntimeError(
                f"Open-area trap detected at {cell}, but emergency "
                f"return to parent {parent} failed."
            )

        # Physically back at parent.
        self.current = parent

        if stack and stack[-1] == cell:
            stack.pop()

        self.visited.discard(cell)
        self.dead_end_cells.discard(cell)
        self.parent.pop(cell, None)
        self.open_dirs.pop(cell, None)
        self.cell_scan_mm.pop(cell, None)

        # The physical opening remains represented by EXIT_CANDIDATE,
        # not by DFS open_dirs.
        self.open_dirs[parent] = [
            d
            for d in self.open_dirs.get(parent, [])
            if d != incoming_dir
        ]

        print(
            f"[OPEN AREA TRAP] back inside at {parent}; "
            f"removed pseudo-cell {cell} from learned map"
        )

        if MAP_AUTOSAVE:
            self.save_map(final=False)


    def cell_key(self, cell):
        return f"{int(cell[0])},{int(cell[1])}"


    def parse_cell_key(self, value):
        x_str, y_str = str(value).split(",", 1)
        return (int(x_str), int(y_str))


    def edge_key(self, a, b):
        return tuple(sorted((a, b)))


    def mark_blocked(self, a, b):
        self.blocked_edges.add(self.edge_key(a, b))

        if MAP_AUTOSAVE:
            self.save_map(final=False)


    def is_blocked(self, a, b):
        return self.edge_key(a, b) in self.blocked_edges


    def mapped_cells(self):
        cells = set(self.visited)
        cells.update(self.open_dirs.keys())
        cells.update(self.cell_scan_mm.keys())
        cells.update(self.dead_end_cells)

        for edge in self.blocked_edges:
            cells.update(edge)

        for (cell, _abs_dir) in self.exit_candidates.keys():
            cells.add(cell)

        return cells


    def build_map_payload(self):
        """
        JSON map representation designed to be reusable on future runs.

        Coordinate convention:
            +Y = North
            +X = East

        A known-map run assumes the robot is physically placed at `root`
        and initially faces `start_heading`.
        """
        cells = self.mapped_cells()

        cell_records = {}

        for cell in sorted(cells, key=lambda p: (p[1], p[0])):
            dirs = list(self.open_dirs.get(cell, []))

            cell_records[self.cell_key(cell)] = {
                "x": int(cell[0]),
                "y": int(cell[1]),
                "visited": cell in self.visited,
                "dead_end": cell in self.dead_end_cells,
                "open_dirs": [DIR_NAMES[d] for d in dirs],
                "open_dir_indices": [int(d) for d in dirs],
                "open_neighbors": [
                    [int(v) for v in neighbor(cell, d)]
                    for d in dirs
                    if not self.is_blocked(cell, neighbor(cell, d))
                ],
                "tof_scan_mm": self.cell_scan_mm.get(cell),
                "exit_candidate_dirs": [
                    DIR_NAMES[d]
                    for (candidate_cell, d) in self.exit_candidates.keys()
                    if candidate_cell == cell
                ],
            }

        blocked = []

        for a, b in sorted(self.blocked_edges):
            blocked.append([
                [int(a[0]), int(a[1])],
                [int(b[0]), int(b[1])],
            ])

        payload = {
            "schema": MAP_SCHEMA,
            "schema_version": MAP_SCHEMA_VERSION,
            "created_at": self.map_created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "complete": bool(self.map_complete),
            "root": [int(self.root[0]), int(self.root[1])],
            "start_heading": "N",
            "start_heading_index": 0,
            "coordinate_system": {
                "N": [0, 1],
                "E": [1, 0],
                "S": [0, -1],
                "W": [-1, 0],
            },
            "geometry": {
                "cell_length_m": CELL_LENGTH_M,
                "cell_success_fraction": CELL_SUCCESS_FRACTION,
            },
            "sensor_policy": {
                "tof_open_threshold_mm": TOF_OPEN_THRESHOLD_MM,
                "dead_end_threshold_mm": DEAD_END_THRESHOLD_MM,
                "tof_scan_samples": TOF_SCAN_SAMPLES,
                "gimbal_pitch_deg": GIMBAL_PITCH_DEG,
                "sharp_center_target_cm": CENTER_TARGET_CM,
                "sharp_follow_near_cm": SHARP_FOLLOW_NEAR_CM,
                "sharp_follow_far_cm": SHARP_FOLLOW_FAR_CM,
                "exit_guard_enabled": EXIT_GUARD_ENABLED,
                "exit_corridor_wall_max_cm": EXIT_CORRIDOR_WALL_MAX_CM,
                "exit_fan_angles_deg": list(EXIT_FAN_ANGLES_DEG),
                "exit_fan_open_mm": EXIT_FAN_OPEN_MM,
                "exit_fan_min_open_per_side": EXIT_FAN_MIN_OPEN_PER_SIDE,
                "exit_fan_strong_open_mm": EXIT_FAN_STRONG_OPEN_MM,
                "exit_fan_min_strong_total": EXIT_FAN_MIN_STRONG_TOTAL,
                "exit_motion_min_travel_m": EXIT_MOTION_MIN_TRAVEL_M,
                "exit_motion_lost_confirm_count": EXIT_MOTION_LOST_CONFIRM_COUNT,
                "exit_wall_end_probe_enabled": EXIT_WALL_END_PROBE_ENABLED,
                "exit_wall_end_angles_left": list(EXIT_WALL_END_ANGLES_LEFT),
                "exit_wall_end_angles_right": list(EXIT_WALL_END_ANGLES_RIGHT),
                "exit_wall_end_front_arm_mm": EXIT_WALL_END_FRONT_ARM_MM,
                "exit_wall_end_ratio": EXIT_WALL_END_RATIO,
                "exit_wall_end_margin_mm": EXIT_WALL_END_MARGIN_MM,
                "exit_wall_end_min_measured_mm": EXIT_WALL_END_MIN_MEASURED_MM,
                "entrance_corridor_profile_enabled": ENTRANCE_CORRIDOR_PROFILE_ENABLED,
                "entrance_profile_angles_deg": list(ENTRANCE_PROFILE_ANGLES_DEG),
                "entrance_open_verify_mm": ENTRANCE_OPEN_VERIFY_MM,
                "exit_motion_guard_enabled": EXIT_MOTION_GUARD_ENABLED,
                "exit_motion_min_travel_m": EXIT_MOTION_MIN_TRAVEL_M,
                "exit_motion_max_travel_m": EXIT_MOTION_MAX_TRAVEL_M,
                "exit_motion_lost_confirm_count": EXIT_MOTION_LOST_CONFIRM_COUNT,
                "exit_motion_caution_start_m": EXIT_MOTION_CAUTION_START_M,
                "exit_motion_caution_speed_mps": EXIT_MOTION_CAUTION_SPEED_MPS,
                "open_area_trap_enabled": OPEN_AREA_TRAP_ENABLED,
                "open_area_trap_min_open_fan_rays": OPEN_AREA_TRAP_MIN_OPEN_FAN_RAYS,
                "open_area_trap_long_mm": OPEN_AREA_TRAP_LONG_MM,
                "open_area_trap_min_long_fan_rays": OPEN_AREA_TRAP_MIN_LONG_FAN_RAYS,
                "start_anchor_enabled": START_ANCHOR_ENABLED,
                "start_maze_ingress_dir": START_MAZE_INGRESS_DIR,
                "start_force_front_open": START_FORCE_FRONT_OPEN,
            },
            "visited_cells": [
                [int(c[0]), int(c[1])]
                for c in sorted(self.visited, key=lambda p: (p[1], p[0]))
            ],
            "dead_end_cells": [
                [int(c[0]), int(c[1])]
                for c in sorted(self.dead_end_cells)
            ],
            "blocked_edges": blocked,
            "start_anchor": {
                "enabled": bool(START_ANCHOR_ENABLED),
                "cell": [int(self.root[0]), int(self.root[1])],
                "maze_ingress_dir_index": int(self.known_maze_ingress_dir),
                "maze_ingress_dir": DIR_NAMES[self.known_maze_ingress_dir],
                "profile": self.start_anchor_profile,
            },
            "known_entrance": {
                "cell": [int(self.root[0]), int(self.root[1])],
                "dir_index": int(self.known_entrance_dir),
                "dir": DIR_NAMES[self.known_entrance_dir],
                "status": "KNOWN_ENTRANCE_CORRIDOR",
                "profile": self.entrance_corridor_profile,
            },
            "exit_candidates": [
                rec
                for _, rec in sorted(
                    self.exit_candidates.items(),
                    key=lambda item: (
                        item[0][0][1],
                        item[0][0][0],
                        item[0][1],
                    ),
                )
            ],
            "targets": self.detected_targets,
            "target_glimpses": self.target_glimpses,
            "target_vision_policy": {
                "enabled": bool(self.target_vision_enabled),
                "roi_norm": [
                    TARGET_ROI_X_MIN,
                    TARGET_ROI_Y_MIN,
                    TARGET_ROI_X_MAX,
                    TARGET_ROI_Y_MAX,
                ],
                "confirm_frames": TARGET_CONFIRM_FRAMES,
                "foam_gate": {
                    "enabled": TARGET_FOAM_GATE_ENABLED,
                    "fail_closed": TARGET_FOAM_FAIL_CLOSED,
                    "hsv_low": list(TARGET_FOAM_HSV_LOW),
                    "hsv_high": list(TARGET_FOAM_HSV_HIGH),
                    "center_margin_px": TARGET_FOAM_CENTER_MARGIN_PX,
                    "min_bbox_below_frac": TARGET_FOAM_MIN_BBOX_BELOW_FRAC,
                },
                "glimpse_linger_sec": TARGET_GLIMPSE_LINGER_SEC,
                "return_rescan_enabled": TARGET_RETURN_RESCAN_ENABLED,
                "sweep_poses_yaw_pitch_deg": [
                    [float(yaw), float(pitch)]
                    for yaw, pitch in TARGET_SWEEP_POSES
                ],
            },
            "cells": cell_records,
            "usage_note": (
                "Known-map mode assumes the robot starts at the same physical "
                "root position and same North-facing orientation used during "
                "mapping. Safety sensors remain active during replay."
            ),
        }

        return payload


    def render_ascii_map(self):
        """
        Human-readable topological map.

        Legend:
            S = root/start
            D = hard dead end
            o = mapped cell
            ? = mapped record not physically visited
        """
        cells = self.mapped_cells()

        if not cells:
            return "(map empty)\n"

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        width = (max_x - min_x) * 4 + 3
        height = (max_y - min_y) * 2 + 1

        canvas = [[" " for _ in range(width)] for _ in range(height)]

        def xy_to_rc(cell):
            x, y = cell
            col = (x - min_x) * 4 + 1
            row = (max_y - y) * 2
            return row, col

        for cell in cells:
            row, col = xy_to_rc(cell)

            if cell == self.root:
                ch = "S"
            elif cell in self.dead_end_cells:
                ch = "D"
            elif cell in self.visited:
                ch = "o"
            else:
                ch = "?"

            canvas[row][col] = ch

        # Draw only trusted non-blocked links between mapped cells.
        for cell in cells:
            for d in self.open_dirs.get(cell, []):
                nb = neighbor(cell, d)

                if nb not in cells or self.is_blocked(cell, nb):
                    continue

                r1, c1 = xy_to_rc(cell)
                r2, c2 = xy_to_rc(nb)

                if r1 == r2:
                    lo, hi = sorted((c1, c2))
                    for c in range(lo + 1, hi):
                        canvas[r1][c] = "-"
                elif c1 == c2:
                    lo, hi = sorted((r1, r2))
                    for r in range(lo + 1, hi):
                        canvas[r][c1] = "|"

        lines = [
            "RoboMaster DFS Persistent Map",
            "N = up, E = right",
            "Legend: S=start, o=mapped, D=dead-end, ?=known/unvisited",
            "",
        ]
        lines.extend("".join(row).rstrip() for row in canvas)
        lines.append("")
        lines.append(
            f"Known maze ingress: {self.root} -> "
            f"{DIR_NAMES[self.known_maze_ingress_dir]}"
        )
        lines.append(
            f"Known entrance/return: {self.root} -> "
            f"{DIR_NAMES[self.known_entrance_dir]}"
        )

        if self.entrance_corridor_profile:
            lines.append(
                "  profile: "
                f"back={self.entrance_corridor_profile.get('back_center_mm')} mm, "
                f"corridor_walls="
                f"{self.entrance_corridor_profile.get('corridor_side_walls_verified')}"
            )

        if self.exit_candidates:
            lines.append("Deferred EXIT_CANDIDATES (real/fake not traversed):")
            for (cell, d), rec in sorted(
                self.exit_candidates.items(),
                key=lambda item: (
                    item[0][0][1],
                    item[0][0][0],
                    item[0][1],
                ),
            ):
                lines.append(
                    f"  {cell} -> {DIR_NAMES[d]} "
                    f"front={rec.get('front_mm')} mm"
                )

        lines.append("")
        return "\n".join(lines)


    def render_svg_map(self):
        """
        Render a wall map as SVG.

        Visual language:
          - each mapped cell is a square
          - black thick lines = confirmed walls
          - gaps = confirmed open directions
          - red X = blocked/failed edge
          - green = root/start, blue = visited, red = hard dead-end
          - orange arrow = open frontier to an unmapped cell
        """
        cells = self.mapped_cells()

        if not cells:
            return """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="240" viewBox="0 0 800 240">
  <rect width="100%" height="100%" fill="white"/>
  <text x="40" y="70" font-family="Arial, sans-serif" font-size="28" fill="#111">RoboMaster Maze Map</text>
  <text x="40" y="120" font-family="Arial, sans-serif" font-size="22" fill="#666">(map empty)</text>
</svg>
"""

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        cell_px = 96
        margin = 88
        header_h = 120
        legend_h = 150
        wall_stroke = 8
        thin_stroke = 2

        cols = max_x - min_x + 1
        rows = max_y - min_y + 1

        width = margin * 2 + cols * cell_px + 1
        height = header_h + rows * cell_px + legend_h + 1

        def cell_xy(cell):
            x, y = cell
            px = margin + (x - min_x) * cell_px
            py = header_h + (max_y - y) * cell_px
            return px, py

        def side_segment(px, py, d):
            if d == 0:   # N
                return (px, py, px + cell_px, py)
            if d == 1:   # E
                return (px + cell_px, py, px + cell_px, py + cell_px)
            if d == 2:   # S
                return (px, py + cell_px, px + cell_px, py + cell_px)
            # W
            return (px, py, px, py + cell_px)

        def opening_midpoint(px, py, d):
            if d == 0:
                return (px + cell_px / 2, py)
            if d == 1:
                return (px + cell_px, py + cell_px / 2)
            if d == 2:
                return (px + cell_px / 2, py + cell_px)
            return (px, py + cell_px / 2)

        def cell_fill(cell):
            if cell == self.root:
                return "#d7f8d0"
            if cell in self.dead_end_cells:
                return "#ffd7d7"
            if cell in self.visited:
                return "#d9e9ff"
            return "#efefef"

        svg = []
        append = svg.append

        append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        append('<rect width="100%" height="100%" fill="white"/>')

        # Title / meta
        append('<text x="28" y="42" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#111">RoboMaster Maze Map</text>')
        append(f'<text x="28" y="74" font-family="Arial, sans-serif" font-size="18" fill="#444">Cells: {len(cells)} | Complete: {self.map_complete} | Root: {self.root} | Grid pitch: {CELL_LENGTH_M:.2f} m | Open threshold: {TOF_OPEN_THRESHOLD_MM} mm</text>')
        append('<text x="28" y="100" font-family="Arial, sans-serif" font-size="16" fill="#666">North is up. Thick black edges are walls. Gaps are open passages.</text>')

        # Border around map area
        append(f'<rect x="{margin - 10}" y="{header_h - 10}" width="{cols * cell_px + 20}" height="{rows * cell_px + 20}" fill="none" stroke="#cfcfcf" stroke-width="2"/>')

        # Light grid background
        for c in range(cols + 1):
            x = margin + c * cell_px
            append(f'<line x1="{x}" y1="{header_h}" x2="{x}" y2="{header_h + rows * cell_px}" stroke="#f1f1f1" stroke-width="1"/>')
        for r in range(rows + 1):
            y = header_h + r * cell_px
            append(f'<line x1="{margin}" y1="{y}" x2="{margin + cols * cell_px}" y2="{y}" stroke="#f1f1f1" stroke-width="1"/>')

        # Draw cells
        for cell in sorted(cells, key=lambda p: (p[1], p[0])):
            px, py = cell_xy(cell)
            fill = cell_fill(cell)
            append(f'<rect x="{px}" y="{py}" width="{cell_px}" height="{cell_px}" fill="{fill}" fill-opacity="0.55" stroke="none"/>')

            # cell label / coordinate
            label = "S" if cell == self.root else ("D" if cell in self.dead_end_cells else "o")
            append(f'<text x="{px + 8}" y="{py + 22}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#222">{label}</text>')
            append(f'<text x="{px + 8}" y="{py + cell_px - 10}" font-family="Consolas, monospace" font-size="13" fill="#333">({cell[0]},{cell[1]})</text>')

            # Optional scan values
            scan = self.cell_scan_mm.get(cell)
            if isinstance(scan, dict):
                small = []
                if scan.get("LEFT") is not None:
                    small.append(f"L{int(scan['LEFT'])}")
                if scan.get("FRONT") is not None:
                    small.append(f"F{int(scan['FRONT'])}")
                if scan.get("RIGHT") is not None:
                    small.append(f"R{int(scan['RIGHT'])}")
                if small:
                    scan_txt = " ".join(small)
                    append(f'<text x="{px + 8}" y="{py + 40}" font-family="Consolas, monospace" font-size="11" fill="#555">{scan_txt}</text>')

            open_dirs = set(self.open_dirs.get(cell, []))

            # Open-frontier marker: direction is open from this cell, but the
            # destination cell has not been mapped yet.
            for d in open_dirs:
                nb = neighbor(cell, d)
                if nb in cells or self.is_blocked(cell, nb):
                    continue

                mx, my = opening_midpoint(px, py, d)
                if d == 0:
                    points = f"{mx},{my - 14} {mx - 8},{my - 2} {mx + 8},{my - 2}"
                elif d == 1:
                    points = f"{mx + 14},{my} {mx + 2},{my - 8} {mx + 2},{my + 8}"
                elif d == 2:
                    points = f"{mx},{my + 14} {mx - 8},{my + 2} {mx + 8},{my + 2}"
                else:
                    points = f"{mx - 14},{my} {mx - 2},{my - 8} {mx - 2},{my + 8}"

                append(f'<polygon points="{points}" fill="#ff9800" fill-opacity="0.95"/>')

            # Walls. EXIT_CANDIDATE and the known entrance are physical
            # openings, even though they are deliberately excluded from DFS.
            for d in range(4):
                nb = neighbor(cell, d)

                candidate_opening = self.is_exit_candidate(cell, d)
                entrance_opening = (
                    cell == self.root
                    and d == self.known_entrance_dir
                )

                ingress_opening = (
                    cell == self.root
                    and d == self.known_maze_ingress_dir
                )

                is_open = (
                    (d in open_dirs and not self.is_blocked(cell, nb))
                    or candidate_opening
                    or entrance_opening
                    or ingress_opening
                )

                if not is_open:
                    x1, y1, x2, y2 = side_segment(px, py, d)
                    append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111" stroke-width="{wall_stroke}" stroke-linecap="square"/>')

            # Deferred wide-open boundary markers.
            for d in range(4):
                if not self.is_exit_candidate(cell, d):
                    continue

                mx, my = opening_midpoint(px, py, d)
                append(f'<circle cx="{mx}" cy="{my}" r="10" fill="#9c27b0" stroke="white" stroke-width="2"/>')
                append(f'<text x="{mx + 12}" y="{my - 12}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#9c27b0">EXIT?</text>')

            # Known maze-ingress marker.
            if cell == self.root:
                d = self.known_maze_ingress_dir
                mx, my = opening_midpoint(px, py, d)
                append(f'<circle cx="{mx}" cy="{my}" r="10" fill="#1565c0" stroke="white" stroke-width="2"/>')
                append(f'<text x="{mx + 12}" y="{my - 12}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#1565c0">MAZE</text>')

            # Known root entrance marker.
            if cell == self.root:
                mx, my = opening_midpoint(
                    px, py, self.known_entrance_dir
                )
                append(f'<circle cx="{mx}" cy="{my}" r="10" fill="#2e7d32" stroke="white" stroke-width="2"/>')
                append(f'<text x="{mx + 12}" y="{my - 12}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#2e7d32">IN CORRIDOR</text>')

        # Blocked/failed edges as red X between cells
        for edge in sorted(self.blocked_edges):
            a, b = edge
            if a not in cells or b not in cells:
                continue

            ax, ay = cell_xy(a)
            bx, by = cell_xy(b)
            cx = (ax + bx) / 2 + cell_px / 2
            cy = (ay + by) / 2 + cell_px / 2
            s = 12
            append(f'<line x1="{cx - s}" y1="{cy - s}" x2="{cx + s}" y2="{cy + s}" stroke="#d32f2f" stroke-width="4"/>')
            append(f'<line x1="{cx - s}" y1="{cy + s}" x2="{cx + s}" y2="{cy - s}" stroke="#d32f2f" stroke-width="4"/>')

        # North arrow
        nx = width - 70
        ny = 55
        append(f'<line x1="{nx}" y1="{ny + 28}" x2="{nx}" y2="{ny - 10}" stroke="#111" stroke-width="4"/>')
        append(f'<polygon points="{nx},{ny - 24} {nx - 10},{ny - 4} {nx + 10},{ny - 4}" fill="#111"/>')
        append(f'<text x="{nx - 7}" y="{ny + 50}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111">N</text>')

        # Legend
        lx = 28
        ly = header_h + rows * cell_px + 38
        append(f'<text x="{lx}" y="{ly - 10}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111">Legend</text>')

        # start
        append(f'<rect x="{lx}" y="{ly + 6}" width="22" height="22" fill="#d7f8d0" stroke="#666"/>')
        append(f'<text x="{lx + 34}" y="{ly + 23}" font-family="Arial, sans-serif" font-size="16" fill="#333">Start / root</text>')

        # visited
        append(f'<rect x="{lx + 200}" y="{ly + 6}" width="22" height="22" fill="#d9e9ff" stroke="#666"/>')
        append(f'<text x="{lx + 234}" y="{ly + 23}" font-family="Arial, sans-serif" font-size="16" fill="#333">Visited cell</text>')

        # dead end
        append(f'<rect x="{lx + 390}" y="{ly + 6}" width="22" height="22" fill="#ffd7d7" stroke="#666"/>')
        append(f'<text x="{lx + 424}" y="{ly + 23}" font-family="Arial, sans-serif" font-size="16" fill="#333">Dead-end cell</text>')

        # wall sample
        y2 = ly + 58
        append(f'<line x1="{lx}" y1="{y2}" x2="{lx + 26}" y2="{y2}" stroke="#111" stroke-width="{wall_stroke}" />')
        append(f'<text x="{lx + 34}" y="{y2 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Wall</text>')

        append(f'<line x1="{lx + 200}" y1="{y2}" x2="{lx + 226}" y2="{y2}" stroke="#d32f2f" stroke-width="4" />')
        append(f'<line x1="{lx + 200}" y1="{y2 + 12}" x2="{lx + 226}" y2="{y2 - 12}" stroke="#d32f2f" stroke-width="4" />')
        append(f'<text x="{lx + 234}" y="{y2 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Blocked / failed edge</text>')

        append(f'<polygon points="{lx + 430},{y2 - 14} {lx + 422},{y2 - 2} {lx + 438},{y2 - 2}" fill="#ff9800"/>')
        append(f'<text x="{lx + 448}" y="{y2 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Open frontier to unmapped area</text>')

        y3 = y2 + 34
        append(f'<circle cx="{lx + 10}" cy="{y3}" r="8" fill="#9c27b0"/>')
        append(f'<text x="{lx + 34}" y="{y3 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Deferred EXIT_CANDIDATE</text>')
        append(f'<circle cx="{lx + 390}" cy="{y3}" r="8" fill="#2e7d32"/>')
        append(f'<text x="{lx + 414}" y="{y3 + 6}" font-family="Arial, sans-serif" font-size="16" fill="#333">Known entrance corridor / safe return</text>')

        append('</svg>')
        return "\n".join(svg)


    def save_map(self, final=False):
        """
        Save reusable JSON + human-readable ASCII + SVG wall map.

        latest_map.* is overwritten intentionally.
        A timestamped snapshot is also created when final=True.
        """
        if not self.mapped_cells():
            return None

        MAP_DIR.mkdir(parents=True, exist_ok=True)

        payload = self.build_map_payload()

        # Atomic-ish replacement so a power/program interruption is less
        # likely to leave a half-written latest_map.json.
        tmp_json = MAP_LATEST_JSON.with_suffix(".json.tmp")

        with tmp_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        os.replace(tmp_json, MAP_LATEST_JSON)

        ascii_text = self.render_ascii_map()

        tmp_txt = MAP_LATEST_ASCII.with_suffix(".txt.tmp")
        tmp_txt.write_text(ascii_text, encoding="utf-8")
        os.replace(tmp_txt, MAP_LATEST_ASCII)

        svg_text = self.render_svg_map()

        tmp_svg = MAP_LATEST_SVG.with_suffix(".svg.tmp")
        tmp_svg.write_text(svg_text, encoding="utf-8")
        os.replace(tmp_svg, MAP_LATEST_SVG)

        snapshot = None

        if final:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = MAP_DIR / f"maze_{stamp}.json"
            snapshot.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            ascii_snapshot = MAP_DIR / f"maze_{stamp}.txt"
            ascii_snapshot.write_text(ascii_text, encoding="utf-8")

            svg_snapshot = MAP_DIR / f"maze_{stamp}.svg"
            svg_snapshot.write_text(svg_text, encoding="utf-8")

        print(
            f"[MAP SAVE] cells={len(payload['cells'])} "
            f"complete={payload['complete']} -> {MAP_LATEST_JSON}"
        )
        print(f"[MAP SAVE] wall image -> {MAP_LATEST_SVG}")

        if snapshot is not None:
            print(f"[MAP SNAPSHOT] {snapshot}")

        # The SVG/JSON remain the persistent outputs, while Mission Control
        # receives the same topology immediately for live rendering.
        self.publish_gui_state()

        return MAP_LATEST_JSON


    def load_map(self, map_path):
        """
        Load a previously learned grid topology.

        This restores topology only.  Absolute chassis yaw is intentionally
        NOT restored because the robot gets a fresh startup yaw reference on
        every physical run.
        """
        path = Path(map_path)

        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("schema") != MAP_SCHEMA:
            raise ValueError(
                f"Unsupported map schema: {data.get('schema')!r}"
            )

        if int(data.get("schema_version", -1)) != MAP_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported map version: {data.get('schema_version')}"
            )

        root = data.get("root", [0, 0])
        self.root = (int(root[0]), int(root[1]))
        self.current = self.root

        self.open_dirs = {}
        self.cell_scan_mm = {}
        self.dead_end_cells = set()
        self.blocked_edges = set()
        self.known_map_cells = set()
        self.exit_candidates = {}

        cells_data = data.get("cells", {})

        for key, rec in cells_data.items():
            cell = (int(rec["x"]), int(rec["y"]))
            self.known_map_cells.add(cell)

            dirs = rec.get("open_dir_indices")

            if dirs is None:
                dirs = [
                    DIR_NAMES.index(name)
                    for name in rec.get("open_dirs", [])
                ]

            self.open_dirs[cell] = [int(d) % 4 for d in dirs]

            scan = rec.get("tof_scan_mm")
            if scan is not None:
                self.cell_scan_mm[cell] = scan

            if rec.get("dead_end", False):
                self.dead_end_cells.add(cell)

        for edge in data.get("blocked_edges", []):
            if len(edge) != 2:
                continue

            a = (int(edge[0][0]), int(edge[0][1]))
            b = (int(edge[1][0]), int(edge[1][1]))
            self.blocked_edges.add(self.edge_key(a, b))

        start_anchor = data.get("start_anchor")
        if isinstance(start_anchor, dict):
            try:
                self.known_maze_ingress_dir = int(
                    start_anchor.get(
                        "maze_ingress_dir_index",
                        self.known_maze_ingress_dir,
                    )
                ) % 4
            except Exception:
                pass

            profile = start_anchor.get("profile")
            if isinstance(profile, dict):
                self.start_anchor_profile = dict(profile)

        entrance = data.get("known_entrance")
        if isinstance(entrance, dict):
            try:
                self.known_entrance_dir = int(
                    entrance.get("dir_index", self.known_entrance_dir)
                ) % 4
            except Exception:
                pass

            profile = entrance.get("profile")
            if isinstance(profile, dict):
                self.entrance_corridor_profile = dict(profile)

        for rec in data.get("exit_candidates", []):
            try:
                cell_v = rec["cell"]
                cell = (int(cell_v[0]), int(cell_v[1]))
                d = int(rec["dir_index"]) % 4
                self.exit_candidates[(cell, d)] = dict(rec)
            except Exception:
                continue

        # Restore target memory when replaying a saved map. Target records are
        # descriptive; known-map motion does not depend on them.
        self.detected_targets = [
            dict(rec)
            for rec in data.get("targets", [])
            if isinstance(rec, dict)
        ]
        self.target_id_seq = 0
        for rec in self.detected_targets:
            tid = str(rec.get("id", ""))
            if tid.startswith("T"):
                try:
                    self.target_id_seq = max(
                        self.target_id_seq,
                        int(tid[1:]),
                    )
                except Exception:
                    pass

        self.target_glimpses = [
            dict(rec)
            for rec in data.get("target_glimpses", [])
            if isinstance(rec, dict)
        ]
        self.target_glimpse_seq = 0
        for rec in self.target_glimpses:
            gid = str(rec.get("id", ""))
            if gid.startswith("G"):
                try:
                    self.target_glimpse_seq = max(
                        self.target_glimpse_seq,
                        int(gid[1:]),
                    )
                except Exception:
                    pass

        self.map_created_at = data.get(
            "created_at",
            datetime.now().isoformat(timespec="seconds")
        )
        self.map_complete = bool(data.get("complete", False))
        self.loaded_map_path = path

        # Keep loaded map's cell length warning visible, but do not silently
        # mutate the runtime constant.
        saved_cell_length = (
            data.get("geometry", {}).get("cell_length_m")
        )

        print("\n================ MAP LOADED ================")
        print(f"File        : {path}")
        print(f"Cells       : {len(self.known_map_cells)}")
        print(f"Complete    : {self.map_complete}")
        print(f"Root        : {self.root}")
        print(f"Exit cand.  : {len(self.exit_candidates)}")
        print(f"Start facing: {data.get('start_heading', 'N')}")
        print(f"Saved cell  : {saved_cell_length} m")
        print(f"Runtime cell: {CELL_LENGTH_M} m")
        print("============================================")

        return data


