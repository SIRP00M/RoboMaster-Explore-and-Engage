#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autonomous DFS Maze Explorer and Target Engagement Orchestrator.
Combines hardware, sensor fusion, motion control, computer vision,
targeting, blaster, mapping, navigation, and GUI subsystems.
"""

import time
from collections import deque

from config import *
from src.core.geometry import DIR_NAMES, DIR_VEC, neighbor, direction_between
from src.hardware.robot_base import HardwareMixin
from src.hardware.sensor_manager import SensorManagerMixin
from src.motion.motion_controller import MotionMixin
from src.vision.vision_pipeline import VisionMixin
from src.vision.target_tracker import TargetTrackerMixin
from src.blaster.blaster_controller import BlasterMixin
from src.mapping.grid_map import GridMapMixin
from src.mapping.navigator import NavigatorMixin
from src.gui.gui_bridge import GuiBridgeMixin


class DFSMazeExplorer(
    HardwareMixin,
    SensorManagerMixin,
    MotionMixin,
    VisionMixin,
    TargetTrackerMixin,
    BlasterMixin,
    GridMapMixin,
    NavigatorMixin,
    GuiBridgeMixin,
):
    """Integrated autonomous explorer combining all modular subsystems."""

    def run_dfs(self, resume=False):
        if not resume:
            self.visited = {self.root}
            self.parent = {self.root: None}
            self.current = self.root

            stack = [self.root]
        else:
            # Continue from the cell just beyond an operator-approved
            # EXIT_CANDIDATE without erasing the map learned so far.
            stack = [self.current]

        exploration_finished = False

        if not resume:
            self.set_gui_status("EXPLORING MAZE", route=[self.current])
            print("\n[DFS] START")
            print(f"[DFS] root={self.root}, heading={DIR_NAMES[self.heading]}")

            # Root is a launch/staging anchor. The robot is manually placed
            # facing INTO the maze, so FRONT is the known ingress even when the
            # staging area is completely open.
            self.capture_start_anchor_profile()

            # The physical return side is still profiled independently.
            self.capture_entrance_corridor_profile()
        else:
            self.set_gui_status("EXPLORING BEYOND APPROVED EXIT", route=[self.current])
            print("\n[DFS] RESUME THROUGH APPROVED EXIT")
            print(
                f"[DFS] resume_cell={self.current}, "
                f"heading={DIR_NAMES[self.heading]}"
            )

        while self.running and stack:
            cell = stack[-1]
            self.current = cell

            if cell not in self.open_dirs:
                scanned_dirs = self.scan_cell(cell)

                # Search targets only while safely stationary at a confirmed
                # DFS node. The target sweep restores gimbal FRONT/-5 before
                # any chassis motion, so topology/motion behavior stays intact.
                try:
                    self.scan_targets_at_cell(cell)
                except Exception as e:
                    # Vision must never destroy a valid maze run.
                    print(f"[TARGET SWEEP WARN] cell={cell}: {e}")
                    self.target_status_text = f"VISION WARN at {cell}: {e}"
                    try:
                        self.gimbal_front_down(force=True)
                    except Exception:
                        pass

                if cell == self.root:
                    scanned_dirs = self.apply_root_start_policy(
                        scanned_dirs
                    )

                self.open_dirs[cell] = scanned_dirs

                # Last-resort containment. If an edge-level detector missed
                # the boundary, do NOT let DFS choose another direction from
                # an open outside area.
                if cell in self.approved_exit_entry_cells:
                    outside_evidence = None
                    print(
                        f"[OPEN AREA TRAP] {cell} is the first cell beyond "
                        "an operator-approved EXIT_CANDIDATE -> allow scan"
                    )
                else:
                    outside_evidence = self.detect_open_area_trap(cell)

                if outside_evidence is not None:
                    self.rollback_open_area_cell(
                        cell,
                        stack,
                        outside_evidence,
                    )
                    continue

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

            # ------------------------------------------------
            # GLOBAL EXPLORATION-COMPLETE CHECK
            # ------------------------------------------------
            # If NO visited cell has an unvisited open neighbor anymore, the
            # maze has been fully explored.  Do NOT keep unwinding the DFS
            # parent stack just to get back to root.  Break here and use the
            # completed map to take the fastest confirmed route home.
            if not self.has_unvisited_frontier():
                exploration_finished = True

                print(
                    f"\n[DFS] ALL FRONTIERS COMPLETE at {cell}. "
                    "Exploration finished."
                )

                if cell != self.root and FAST_RETURN_HOME_AFTER_DFS:
                    print(
                        "[DFS] skip remaining DFS-stack backtracking; "
                        "FAST RETURN HOME will plan directly to (0, 0)."
                    )

                break

            # ------------------------------------------------
            # HARD DEAD-END OVERRIDE
            # ------------------------------------------------
            # LEFT + FRONT + RIGHT are all <= DEAD_END_THRESHOLD_MM.
            # Do not spend another DFS decision cycle here: reverse 180 deg
            # and immediately go back through the edge we entered from.
            if cell in self.dead_end_cells:
                parent = self.parent.get(cell)

                if parent is None:
                    # At the root there is no mapped parent cell.  Still obey
                    # the requested dead-end behavior by turning around, then
                    # stop the exploration safely at the entrance.
                    reverse_dir = (self.heading + 2) % 4

                    print(
                        f"\n[DEAD END] root {cell}: "
                        f"LEFT/FRONT/RIGHT <= {DEAD_END_THRESHOLD_MM:.0f} mm"
                    )
                    print(
                        f"[DEAD END] TURN 180 "
                        f"{DIR_NAMES[self.heading]} -> {DIR_NAMES[reverse_dir]}"
                    )

                    self.turn_to_direction(reverse_dir)
                    print("[DFS] root is boxed in; exploration complete.")
                    break

                back_dir = direction_between(cell, parent)

                print(
                    f"\n[DEAD END] {cell}: "
                    f"LEFT/FRONT/RIGHT <= {DEAD_END_THRESHOLD_MM:.0f} mm"
                )
                print(
                    f"[DEAD END] TURN 180 + BACKTRACK "
                    f"{cell} -> {parent} dir={DIR_NAMES[back_dir]}"
                )

                # Because the robot entered this cell facing away from parent,
                # back_dir should normally be a 180-degree logical turn.
                self.turn_to_direction(back_dir)

                # A dead-end escape naturally gives the camera the opposite
                # travel orientation.  Re-check only remembered target evidence
                # from this cell before leaving it.
                try:
                    if TARGET_DFS_BACKTRACK_RESCAN_ENABLED:
                        self.rescan_targets_on_return(cell, phase="dfs_backtrack")
                except Exception as e:
                    print(f"[TARGET REVERSE WARN] cell={cell}: {e}")
                    try:
                        self.gimbal_front_down(force=True)
                    except Exception:
                        pass

                # Known parent edge: no exit classification while escaping
                # a dead end. We physically traversed this edge on entry.
                ok = self.move_one_cell()

                if not ok:
                    self.stop_chassis()
                    raise RuntimeError(
                        f"Dead-end backtrack failed: {cell} -> {parent}."
                    )

                stack.pop()
                self.current = parent

                print(f"[DEAD END] escaped; back at {parent}")

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                continue

            # Classic DFS:
            # choose the first open neighbor that has not been visited.
            next_dir = None
            next_cell = None

            for d in self.open_dirs[cell]:
                nb = neighbor(cell, d)

                if self.is_blocked(cell, nb):
                    continue

                if nb not in self.visited:
                    next_dir = d
                    next_cell = nb
                    break

            if next_cell is not None:
                print(
                    f"\n[DFS] EXPLORE {cell} -> {next_cell} "
                    f"dir={DIR_NAMES[next_dir]}"
                )

                self.turn_to_direction(next_dir)

                # BOTH-IR after the turn may reveal that the intended front
                # direction is actually blocked while another branch is open.
                # Stay on the SAME logical cell and rescan/replan.
                if self.ir_replan_requested:
                    print(
                        f"[DFS] IR/Gimbal requests REPLAN at {cell} "
                        f"hint={self.ir_route_hint}"
                    )
                    self.ir_replan_requested = False
                    self.open_dirs.pop(cell, None)
                    continue

                # The physical entrance is known from startup and may look
                # exactly like an exit corridor. It is never an Explore
                # frontier and must never be traversed outward.
                if self.is_known_entrance_edge(cell, next_dir):
                    print(
                        f"[DFS] DEFER {cell}->{next_cell} "
                        f"dir={DIR_NAMES[next_dir]} as KNOWN_ENTRANCE"
                    )

                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if d != next_dir
                    ]

                    if MAP_AUTOSAVE:
                        self.save_map(final=False)

                    continue

                # Root->front is a known maze ingress based on placement.
                # It must be allowed even if the starting zone is wide open.
                root_ingress = self.is_known_maze_ingress_edge(
                    cell,
                    next_dir,
                )

                # Unknown-size boundary protection:
                # real exits and fake exits are deferred only after ingress.
                if (
                    not root_ingress
                    and self.exit_guard_before_explore_edge(
                        cell,
                        next_dir,
                    )
                ):
                    print(
                        f"[DFS] DEFER {cell}->{next_cell} "
                        f"dir={DIR_NAMES[next_dir]} as EXIT_CANDIDATE"
                    )

                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if d != next_dir
                    ]

                    if MAP_AUTOSAVE:
                        self.save_map(final=False)

                    continue

                ok = self.move_one_cell(
                    source_cell=cell,
                    abs_dir=next_dir,
                    detect_exit=True,
                )

                if not ok:
                    # Confirmed exit candidate was detected DURING motion and
                    # the robot has already retreated to this source cell.
                    if self.motion_exit_candidate_detected:
                        self.motion_exit_candidate_detected = False

                        print(
                            f"[DFS] EXIT_CANDIDATE caught during motion "
                            f"{cell}->{next_cell}; "
                            "do not create destination cell"
                        )

                        self.open_dirs[cell] = [
                            d for d in self.open_dirs[cell]
                            if d != next_dir
                        ]

                        if MAP_AUTOSAVE:
                            self.save_map(final=False)

                        continue

                    # A transient IR/corner event may have safely returned the
                    # robot to the SAME source cell.  In that case do not poison
                    # the topology by permanently blocking the edge.  Throw
                    # away this cell's scan and observe it again.
                    if self.motion_replan_requested:
                        reason = self.motion_replan_reason

                        print(
                            f"[DFS] transient motion abort at {cell}: "
                            f"{reason} -> RESCAN SAME CELL; edge is NOT blocked"
                        )

                        self.motion_replan_requested = False
                        self.motion_replan_reason = None
                        self.ir_replan_requested = False
                        self.open_dirs.pop(cell, None)

                        if MAP_AUTOSAVE:
                            self.save_map(final=False)

                        continue

                    print(
                        f"[DFS] edge {cell}->{next_cell} failed; "
                        f"mark blocked and continue."
                    )
                    self.mark_blocked(cell, next_cell)

                    # Remove this false-positive opening from the cell map.
                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if neighbor(cell, d) != next_cell
                    ]

                    if MAP_AUTOSAVE:
                        self.save_map(final=False)

                    continue

                self.parent[next_cell] = cell
                self.visited.add(next_cell)
                stack.append(next_cell)

                print(
                    f"[DFS] ARRIVED {next_cell}; "
                    f"visited={len(self.visited)}"
                )

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                continue

            # No unvisited open neighbor -> backtrack.
            parent = self.parent.get(cell)

            if parent is None:
                print("\n[DFS] Root has no unvisited neighbors.")
                print("[DFS] COMPLETE.")
                exploration_finished = True
                break

            back_dir = direction_between(cell, parent)

            print(
                f"\n[DFS] BACKTRACK {cell} -> {parent} "
                f"dir={DIR_NAMES[back_dir]}"
            )

            self.turn_to_direction(back_dir)

            if self.ir_replan_requested:
                self.ir_replan_requested = False
                self.stop_chassis()
                raise RuntimeError(
                    "IR/Gimbal says the DFS parent/backtrack direction is "
                    f"blocked at {cell}; topology cannot be trusted."
                )

            # Normal DFS backtracking is also a useful reverse viewpoint.
            # This catches a G# glimpse before the robot leaves the branch,
            # without doing a full vision sweep at every revisited cell.
            try:
                if TARGET_DFS_BACKTRACK_RESCAN_ENABLED:
                    self.rescan_targets_on_return(cell, phase="dfs_backtrack")
            except Exception as e:
                print(f"[TARGET REVERSE WARN] cell={cell}: {e}")
                try:
                    self.gimbal_front_down(force=True)
                except Exception:
                    pass

            ok = self.move_one_cell()

            if not ok:
                self.stop_chassis()
                raise RuntimeError(
                    f"Backtrack failed: {cell} -> {parent}. "
                    f"Stopping because DFS topology is no longer reliable."
                )

            stack.pop()
            self.current = parent

            print(f"[DFS] back at {parent}")

            if MAP_AUTOSAVE:
                self.save_map(final=False)

        # Ctrl+C / external stop must never label a partial map as complete.
        if not self.running:
            self.stop_chassis()

            if MAP_AUTOSAVE:
                self.save_map(final=False)

            print("[DFS] stopped before full exploration completed.")
            return

        # Defensive fallback: if the loop ended naturally, verify the global
        # graph really has no remaining frontier.
        if not exploration_finished:
            exploration_finished = not self.has_unvisited_frontier()

        if not exploration_finished:
            self.stop_chassis()
            raise RuntimeError(
                "DFS loop ended while an unexplored frontier still exists."
            )

        self.map_complete = True
        self.set_gui_status("CURRENT REGION FULLY EXPLORED")

        # Save the COMPLETED learned map before driving home.
        self.stop_chassis()
        self.gimbal_front_down()
        self.print_map_summary()
        self.save_map(final=True)

        # The robot can finish exploration at the far end of the last branch.
        # Use the completed graph to return directly instead of DFS-parent
        # backtracking all the way to root.
        if FAST_RETURN_HOME_AFTER_DFS and self.current != self.root:
            self.fast_return_home()

        elif self.current == self.root:
            print("[RETURN HOME] DFS completed at root; no return trip needed.")

        # ----------------------------------------------------
        # OPERATOR DECISION AFTER RETURNING HOME
        # ----------------------------------------------------
        # Deferred exits are intentionally not crossed during the first pass.
        # Once the robot is safely back at START, the operator can finish the
        # mission or explicitly choose which deferred exit to continue through.
        if self.running and self.current == self.root and self.exit_candidates:
            decision = self.prompt_after_return_home()

            if decision == "continue":
                self.map_complete = False

                if self.cross_selected_exit_candidate(
                    self.operator_selected_exit_candidate
                ):
                    # Preserve all learned topology and continue DFS from the
                    # newly reached cell beyond the approved candidate.
                    self.run_dfs(resume=True)
                    return

                print(
                    "[EXIT CONTINUE] unable to enter a candidate safely; "
                    "mission ends at START."
                )
            else:
                print("[MISSION] operator selected FINISH at START.")
                self.set_gui_status("MISSION FINISHED AT START", route=[self.root])


