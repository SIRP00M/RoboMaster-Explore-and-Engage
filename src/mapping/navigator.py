#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Graph pathfinding, BFS/Dijkstra shortest paths, fast return home, and exit selection."""

import heapq
import math
import time
from collections import deque
from pathlib import Path

from config import *
from src.core.geometry import wrap_deg, clamp, DIR_NAMES, DIR_VEC, neighbor, direction_between


class NavigatorMixin:
    def known_neighbors(self, cell):
        """
        Trusted neighbors from the saved graph.

        An edge is usable only if:
          - the direction was saved as open
          - the destination is a known mapped cell
          - the edge is not marked blocked
        """
        result = []

        for d in self.open_dirs.get(cell, []):
            nb = neighbor(cell, d)

            if nb not in self.known_map_cells:
                continue

            if self.is_blocked(cell, nb):
                continue

            result.append((d, nb))

        return result


    def shortest_known_path(self, start, goal):
        """
        BFS shortest path over the known unweighted grid graph.
        Returns a list of cells including start and goal.
        """
        if start not in self.known_map_cells:
            raise ValueError(f"Start cell not in map: {start}")

        if goal not in self.known_map_cells:
            raise ValueError(f"Goal cell not in map: {goal}")

        q = deque([start])
        came_from = {start: None}

        while q:
            cell = q.popleft()

            if cell == goal:
                break

            for _, nb in self.known_neighbors(cell):
                if nb not in came_from:
                    came_from[nb] = cell
                    q.append(nb)

        if goal not in came_from:
            raise RuntimeError(
                f"No known path from {start} to {goal}"
            )

        path = []
        cur = goal

        while cur is not None:
            path.append(cur)
            cur = came_from[cur]

        path.reverse()
        return path


    def execute_known_path(self, path):
        """
        Follow a saved path while retaining all real-time safety layers:
        yaw lock, Sharp corridor authority, IR interlocks and front ToF.
        """
        if not path:
            return

        self.current = path[0]

        for target in path[1:]:
            current = self.current
            d = direction_between(current, target)

            print(
                f"\n[KNOWN] {current} -> {target} "
                f"dir={DIR_NAMES[d]}"
            )

            self.turn_to_direction(d)

            if self.ir_replan_requested:
                self.ir_replan_requested = False
                self.stop_chassis()
                raise RuntimeError(
                    "Saved map route disagrees with current IR/Gimbal "
                    f"observation before {current}->{target}. "
                    "Stopping instead of trusting stale topology."
                )

            ok = self.move_one_cell()

            if not ok:
                self.stop_chassis()

                if self.motion_replan_requested:
                    reason = self.motion_replan_reason
                    self.motion_replan_requested = False
                    self.motion_replan_reason = None

                    raise RuntimeError(
                        f"Known-map route was safely aborted and returned to "
                        f"{current}: {reason}. The environment should be "
                        "re-observed before trusting the saved route."
                    )

                raise RuntimeError(
                    f"Known-map motion failed at edge {current}->{target}. "
                    "The environment may have changed."
                )

            self.current = target

        print(f"\n[KNOWN] reached {self.current}")


    def run_known_map(self, goal=None):
        """
        Reuse an already learned map.

        If goal is provided:
            compute BFS shortest path from root to goal and run it.

        If goal is None:
            perform a full coverage replay of all reachable known cells
            WITHOUT re-scanning topology with the gimbal at every node.
        """
        if not self.known_map_cells:
            raise RuntimeError("No map loaded.")

        self.heading = 0
        self.current = self.root

        if goal is not None:
            path = self.shortest_known_path(self.root, goal)

            print("\n================ KNOWN MAP ROUTE ================")
            print(f"Start: {self.root}")
            print(f"Goal : {goal}")
            print(f"Cells: {len(path)}")
            print("Path :", path)
            print("=================================================")

            self.execute_known_path(path)
            return

        print("\n[KNOWN] FULL MAP REPLAY/COVERAGE")
        print("[KNOWN] topology scans are skipped; safety sensors remain active")

        seen = {self.root}
        stack = [(self.root, 0)]

        while self.running and stack:
            cell, next_index = stack[-1]
            neighbors = self.known_neighbors(cell)

            # Find next unvisited known neighbor.
            chosen = None

            while next_index < len(neighbors):
                d, nb = neighbors[next_index]
                next_index += 1
                stack[-1] = (cell, next_index)

                if nb not in seen:
                    chosen = (d, nb)
                    break

            if chosen is not None:
                d, nb = chosen

                print(
                    f"\n[KNOWN] VISIT {cell} -> {nb} "
                    f"dir={DIR_NAMES[d]}"
                )

                self.turn_to_direction(d)

                if self.ir_replan_requested:
                    self.ir_replan_requested = False
                    raise RuntimeError(
                        f"Current sensors disagree with saved edge {cell}->{nb}"
                    )

                if not self.move_one_cell():
                    raise RuntimeError(
                        f"Could not traverse saved edge {cell}->{nb}"
                    )

                self.current = nb
                seen.add(nb)
                stack.append((nb, 0))
                continue

            # Finished this node: go back to DFS parent in replay stack.
            if len(stack) == 1:
                break

            child = stack.pop()[0]
            parent = stack[-1][0]
            d = direction_between(child, parent)

            print(
                f"\n[KNOWN] BACKTRACK {child} -> {parent} "
                f"dir={DIR_NAMES[d]}"
            )

            self.turn_to_direction(d)

            if self.ir_replan_requested:
                self.ir_replan_requested = False
                raise RuntimeError(
                    f"Current sensors disagree with saved edge {child}->{parent}"
                )

            if not self.move_one_cell():
                raise RuntimeError(
                    f"Could not backtrack saved edge {child}->{parent}"
                )

            self.current = parent

        print(
            f"\n[KNOWN] replay complete: "
            f"{len(seen)}/{len(self.known_map_cells)} cells reached"
        )


    def has_unvisited_frontier(self):
        """
        True while the explored graph still contains something DFS must visit.

        A visited-but-not-yet-scanned cell also counts as unfinished.
        """
        for cell in self.visited:
            if cell not in self.open_dirs:
                return True

            for d in self.open_dirs.get(cell, []):
                nb = neighbor(cell, d)

                if self.is_blocked(cell, nb):
                    continue

                if nb not in self.visited:
                    return True

        return False


    def confirmed_explored_neighbors(self, cell):
        """
        Safe graph used for the FAST RETURN HOME planner.

        An edge is accepted when:
          1) it is a parent-child edge that the robot physically traversed, OR
          2) BOTH endpoint scans agree that the edge is open.

        This means shortcuts/loops can be used for a faster return, while an
        unconfirmed one-sided ToF opening is not blindly trusted.
        """
        result = []

        for d in range(4):
            nb = neighbor(cell, d)

            if nb not in self.visited:
                continue

            if self.is_blocked(cell, nb):
                continue

            if self.edge_key(cell, nb) in self.transient_runtime_blocked_edges:
                continue

            physically_traversed = (
                self.parent.get(cell) == nb
                or self.parent.get(nb) == cell
            )

            reverse_d = (d + 2) % 4

            mutually_scanned_open = (
                d in self.open_dirs.get(cell, [])
                and reverse_d in self.open_dirs.get(nb, [])
            )

            if physically_traversed or mutually_scanned_open:
                result.append((d, nb))

        return result


    def estimated_turn_time(self, from_heading, to_heading):
        delta = (to_heading - from_heading) % 4

        if delta == 0:
            return 0.0

        if delta == 2:
            return FAST_RETURN_TURN_180_EST_SEC

        return FAST_RETURN_TURN_90_EST_SEC


    def plan_fastest_return(self, start, start_heading, goal):
        """
        Dijkstra over (cell, heading), not just cell.

        Cost ~= chassis travel time + physical turn time.
        Therefore among map routes it can choose one with fewer turns instead
        of blindly taking a cell-count-only DFS parent path.
        """
        if start == goal:
            return [], 0.0

        # state = (cell, heading)
        start_state = (start, start_heading % 4)

        pq = [(0.0, start[0], start[1], start_heading % 4)]
        best = {start_state: 0.0}
        previous = {start_state: None}
        previous_action = {}

        goal_state = None

        while pq:
            cost, x, y, heading = heapq.heappop(pq)
            cell = (x, y)
            state = (cell, heading)

            if cost > best.get(state, float("inf")) + 1e-9:
                continue

            if cell == goal:
                goal_state = state
                break

            for d, nb in self.confirmed_explored_neighbors(cell):
                step_cost = (
                    FAST_RETURN_MOVE_EST_SEC
                    + self.estimated_turn_time(heading, d)
                )

                next_state = (nb, d)
                new_cost = cost + step_cost

                if new_cost + 1e-9 < best.get(next_state, float("inf")):
                    best[next_state] = new_cost
                    previous[next_state] = state
                    previous_action[next_state] = d

                    heapq.heappush(
                        pq,
                        (new_cost, nb[0], nb[1], d)
                    )

        if goal_state is None:
            raise RuntimeError(
                f"No confirmed route from {start} to {goal}."
            )

        reversed_steps = []
        cur = goal_state

        while cur != start_state:
            prev = previous[cur]

            if prev is None:
                raise RuntimeError("Fast-return path reconstruction failed.")

            d = previous_action[cur]
            target_cell = cur[0]
            reversed_steps.append((d, target_cell))
            cur = prev

        reversed_steps.reverse()
        return reversed_steps, best[goal_state]


    def fast_return_home(self):
        """
        Return from the final DFS cell to root=(0,0) using the fastest
        confirmed route currently known.

        No topology scans are repeated.  Real-time IR/Sharp/ToF/yaw safety
        remains active.  If a saved shortcut becomes blocked, that edge is
        marked blocked and the route is replanned from the current cell.
        """
        if self.current == self.root:
            print("[RETURN HOME] already at root.")
            return

        print("\n================ FAST RETURN HOME ================")
        print(f"Current : {self.current}")
        print(f"Home    : {self.root}")
        print("Planner : confirmed-map Dijkstra (move + turn time)")
        print("Safety  : IR + Sharp + front ToF + yaw hold remain active")
        print("==================================================")
        self.set_gui_status("RETURNING TO START")

        replans = 0
        self.transient_runtime_blocked_edges.clear()

        while self.running and self.current != self.root:
            plan, estimated_sec = self.plan_fastest_return(
                self.current,
                self.heading,
                self.root,
            )

            route_cells = [self.current] + [target for _, target in plan]
            self.set_gui_status("RETURNING TO START", route=route_cells)

            print(
                f"[RETURN PLAN] moves={len(plan)} "
                f"estimated_action_time={estimated_sec:.1f}s"
            )
            print(f"[RETURN PLAN] {route_cells}")

            need_replan = False

            for d, target in plan:
                current = self.current

                print(
                    f"\n[RETURN] {current} -> {target} "
                    f"dir={DIR_NAMES[d]}"
                )

                self.turn_to_direction(d)

                # AFTER_TURN IR/Gimbal may decide that the mapped direction
                # is no longer safe.  Preserve the original safety priority:
                # reject the edge BEFORE spending time on a vision re-check.
                if self.ir_replan_requested:
                    self.ir_replan_requested = False
                    self.stop_chassis()

                    print(
                        f"[RETURN REPLAN] current sensors reject "
                        f"{current}->{target}; mark blocked"
                    )

                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                # The return heading provides a genuinely different viewpoint.
                # Re-check only cells that had a brief glimpse or a locked
                # target, then restore FRONT/-5 before motion.
                try:
                    if TARGET_FAST_RETURN_RESCAN_ENABLED:
                        self.rescan_targets_on_return(current, phase="fast_return")
                except Exception as e:
                    print(f"[TARGET RETURN WARN] cell={current}: {e}")
                    try:
                        self.gimbal_front_down(force=True)
                    except Exception:
                        pass

                ok = self.move_one_cell()

                if not ok:
                    self.stop_chassis()

                    if self.motion_replan_requested:
                        reason = self.motion_replan_reason
                        self.motion_replan_requested = False
                        self.motion_replan_reason = None

                        print(
                            f"[RETURN REPLAN] transient safety abort "
                            f"{current}->{target}: {reason}; "
                            "returned to source, do NOT permanently block edge"
                        )

                        # Avoid immediately selecting the exact same edge
                        # again during this return attempt, without persisting
                        # it as a permanent WALL in the learned map.
                        self.transient_runtime_blocked_edges.add(
                            self.edge_key(current, target)
                        )
                        need_replan = True
                        break

                    print(
                        f"[RETURN REPLAN] failed edge "
                        f"{current}->{target}; mark blocked"
                    )

                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                self.current = target

                print(f"[RETURN] arrived {self.current}")

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                if self.current == self.root:
                    break

            if self.current == self.root:
                break

            if not need_replan:
                raise RuntimeError(
                    "Fast-return plan ended before reaching home."
                )

            replans += 1

            if replans > FAST_RETURN_MAX_REPLANS:
                raise RuntimeError(
                    "Fast return exceeded replan limit; robot stopped."
                )

        self.stop_chassis()

        if self.current == self.root:
            try:
                if TARGET_FAST_RETURN_RESCAN_ENABLED:
                    self.rescan_targets_on_return(self.root, phase="fast_return")
            except Exception as e:
                print(f"[TARGET RETURN WARN] root={self.root}: {e}")

        self.gimbal_front_down()

        if self.current == self.root:
            print(
                f"\n[RETURN HOME OK] reached {self.root} "
                f"heading={DIR_NAMES[self.heading]}"
            )
            self.set_gui_status("AT START - RETURN COMPLETE", route=[self.root])


    def shortest_confirmed_path(self, start, goal):
        """
        BFS shortest path over the already explored/confirmed graph.

        This is intentionally different from plan_fastest_return(): here the
        operator asked for the shortest route in CELL COUNT back to an
        EXIT_CANDIDATE source cell.  Safety checks still run while executing
        the path.
        """
        start = tuple(start)
        goal = tuple(goal)

        if start == goal:
            return [start]

        if start not in self.visited:
            raise ValueError(f"Start cell not explored: {start}")

        if goal not in self.visited:
            raise ValueError(f"Goal cell not explored: {goal}")

        q = deque([start])
        came_from = {start: None}

        while q:
            cell = q.popleft()

            if cell == goal:
                break

            for _d, nb in self.confirmed_explored_neighbors(cell):
                if nb in came_from:
                    continue

                came_from[nb] = cell
                q.append(nb)

        if goal not in came_from:
            raise RuntimeError(
                f"No confirmed shortest path from {start} to {goal}."
            )

        path = []
        cur = goal

        while cur is not None:
            path.append(cur)
            cur = came_from[cur]

        path.reverse()
        return path


    def navigate_shortest_confirmed_to(self, goal):
        """
        Navigate to one already-explored cell using the shortest confirmed
        path by number of grid edges.

        If a live safety sensor rejects an old edge, replan from the current
        cell instead of forcing the saved topology.
        """
        goal = tuple(goal)

        if self.current == goal:
            print(f"[EXIT ROUTE] already at candidate source {goal}")
            return True

        print("\n================ EXIT ROUTE =====================")
        print(f"Current : {self.current}")
        print(f"Target  : {goal}")
        print("Planner : BFS shortest confirmed path (minimum cells)")
        print("Safety  : IR + Sharp + front ToF + yaw hold remain active")
        print("==================================================")

        replans = 0
        self.transient_runtime_blocked_edges.clear()

        while self.running and self.current != goal:
            try:
                path = self.shortest_confirmed_path(self.current, goal)
            except Exception as e:
                self.stop_chassis()
                print(
                    f"[EXIT ROUTE] no safe confirmed route from "
                    f"{self.current} to {goal}: {e}"
                )
                return False

            print(
                f"[EXIT ROUTE PLAN] moves={max(0, len(path) - 1)} "
                f"path={path}"
            )

            need_replan = False

            for target in path[1:]:
                current = self.current
                d = direction_between(current, target)

                print(
                    f"\n[EXIT ROUTE] {current} -> {target} "
                    f"dir={DIR_NAMES[d]}"
                )

                self.turn_to_direction(d)

                if self.ir_replan_requested:
                    self.ir_replan_requested = False
                    self.stop_chassis()
                    print(
                        f"[EXIT ROUTE REPLAN] sensors reject "
                        f"{current}->{target}; mark blocked"
                    )
                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                ok = self.move_one_cell()

                if not ok:
                    self.stop_chassis()

                    if self.motion_replan_requested:
                        reason = self.motion_replan_reason
                        self.motion_replan_requested = False
                        self.motion_replan_reason = None

                        print(
                            f"[EXIT ROUTE REPLAN] transient abort "
                            f"{current}->{target}: {reason}"
                        )

                        self.transient_runtime_blocked_edges.add(
                            self.edge_key(current, target)
                        )
                        need_replan = True
                        break

                    print(
                        f"[EXIT ROUTE REPLAN] failed edge "
                        f"{current}->{target}; mark blocked"
                    )
                    self.mark_blocked(current, target)
                    need_replan = True
                    break

                self.current = target
                print(f"[EXIT ROUTE] arrived {self.current}")

                if MAP_AUTOSAVE:
                    self.save_map(final=False)

                if self.current == goal:
                    break

            if self.current == goal:
                break

            if not need_replan:
                return False

            replans += 1

            if replans > FAST_RETURN_MAX_REPLANS:
                print("[EXIT ROUTE] replan limit exceeded")
                return False

        self.stop_chassis()
        self.gimbal_front_down()
        return self.current == goal


    def choose_nearest_exit_candidate(self):
        """
        Choose the reachable deferred EXIT_CANDIDATE whose SOURCE cell is
        closest to the current robot position in confirmed grid-edge count.
        """
        if not self.exit_candidates:
            return None

        ranked = []

        # Do not let an old transient block from the home trip poison the
        # operator-requested route selection.
        self.transient_runtime_blocked_edges.clear()

        for (cell, d), rec in self.exit_candidates.items():
            try:
                path = self.shortest_confirmed_path(self.current, cell)
            except Exception as e:
                print(
                    f"[EXIT ROUTE] skip unreachable candidate "
                    f"{cell}->{DIR_NAMES[d]}: {e}"
                )
                continue

            ranked.append(
                (
                    len(path) - 1,
                    cell[1],
                    cell[0],
                    d,
                    tuple(cell),
                    rec,
                    path,
                )
            )

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[:4])
        moves, _y, _x, d, cell, rec, path = ranked[0]

        print("\n[EXIT SELECT] nearest deferred candidate")
        print(
            f"[EXIT SELECT] source={cell} dir={DIR_NAMES[d]} "
            f"shortest_moves={moves}"
        )
        print(f"[EXIT SELECT] route={path}")

        return cell, d, rec


    def prompt_after_return_home(self):
        """
        Ask the operator what to do only AFTER the robot is safely back at
        root.  Mission Control is preferred; terminal selection is the safe
        fallback if Tkinter/display is unavailable.
        """
        if self.current != self.root or not self.exit_candidates:
            self.operator_selected_exit_candidate = None
            return "finish"

        options = self.build_exit_candidate_options()
        reachable = [opt for opt in options if opt.get('reachable')]

        if not reachable:
            print('[EXIT DECISION] no reachable EXIT_CANDIDATE from START')
            self.operator_selected_exit_candidate = None
            return 'finish'

        self.set_gui_status('WAITING FOR EXIT SELECTION', route=[self.root])

        if self.mission_gui is not None and self.mission_gui.available:
            result = self.mission_gui.request_exit_decision(
                options,
                snapshot=self.build_gui_snapshot(),
            )

            if result is not None:
                action, candidate_key = result
                if action == 'continue' and candidate_key in self.exit_candidates:
                    self.operator_selected_exit_candidate = candidate_key
                    return 'continue'

                self.operator_selected_exit_candidate = None
                return 'finish'

        # ----------------------------------------------------
        # Terminal fallback: still allows choosing ANY candidate.
        # ----------------------------------------------------
        print("\n================ EXIT DECISION ==================")
        print(f"Deferred EXIT_CANDIDATES: {len(self.exit_candidates)}")
        print("Robot is back at START (0, 0).")
        print("  [0] FINISH MISSION")

        for index, opt in enumerate(reachable, start=1):
            print(
                f"  [{index}] {opt['id']} source={opt['cell']} "
                f"dir={opt['dir']} shortest={opt['moves']} moves "
                f"front={opt.get('front_mm')}mm"
            )
            print(f"      path={opt['path']}")

        print("==================================================")

        while self.running:
            try:
                raw = input(
                    f"Select 0=FINISH or 1-{len(reachable)}=EXIT [default 0]: "
                ).strip().lower()
            except EOFError:
                print("[EXIT DECISION] no interactive input -> FINISH")
                self.operator_selected_exit_candidate = None
                return "finish"

            if raw in ("", "0", "f", "finish", "end", "stop"):
                self.operator_selected_exit_candidate = None
                return "finish"

            try:
                index = int(raw)
            except ValueError:
                print("Please enter a candidate number or 0 to finish.")
                continue

            if 1 <= index <= len(reachable):
                selected = reachable[index - 1]
                self.operator_selected_exit_candidate = selected['key']
                print(
                    f"[EXIT SELECT] operator chose {selected['id']} "
                    f"{selected['cell']}->{selected['dir']}"
                )
                return "continue"

            print("Selection out of range.")

        self.operator_selected_exit_candidate = None
        return "finish"


    def cross_selected_exit_candidate(self, candidate_key=None):
        """
        Route to the operator-selected deferred candidate, cross that one edge
        with the normal collision/safety layers still active, and attach the
        new cell to the existing DFS graph so exploration can resume there.

        The exit classifier itself is bypassed only for this operator-approved
        edge; subsequent edges use normal EXIT_CANDIDATE protection again.
        """
        if candidate_key is None:
            candidate_key = self.operator_selected_exit_candidate

        if candidate_key is None:
            # Compatibility/safety fallback for non-GUI callers.
            selected = self.choose_nearest_exit_candidate()
            if selected is None:
                print("[EXIT CONTINUE] no reachable EXIT_CANDIDATE remains")
                return False
            source_cell, abs_dir, _rec = selected
            candidate_key = self.exit_candidate_key(source_cell, abs_dir)
        else:
            source_cell = tuple(candidate_key[0])
            abs_dir = int(candidate_key[1]) % 4
            candidate_key = self.exit_candidate_key(source_cell, abs_dir)
            _rec = self.exit_candidates.get(candidate_key)
            if _rec is None:
                print(
                    f"[EXIT CONTINUE] selected candidate no longer exists: "
                    f"{source_cell}->{DIR_NAMES[abs_dir]}"
                )
                return False

        try:
            preview_path = self.shortest_confirmed_path(self.current, source_cell)
        except Exception as e:
            print(f"[EXIT CONTINUE] selected candidate unreachable: {e}")
            return False

        self.set_gui_status(
            f"NAVIGATING TO SELECTED EXIT {source_cell}->{DIR_NAMES[abs_dir]}",
            route=preview_path,
        )

        if not self.navigate_shortest_confirmed_to(source_cell):
            print(
                f"[EXIT CONTINUE] could not reach candidate source "
                f"{source_cell}"
            )
            return False

        destination = neighbor(source_cell, abs_dir)
        key = self.exit_candidate_key(source_cell, abs_dir)

        print("\n================ CROSS APPROVED EXIT ============")
        print(f"Source      : {source_cell}")
        print(f"Direction   : {DIR_NAMES[abs_dir]}")
        print(f"Destination : {destination}")
        print("Exit guard  : bypassed for THIS edge only")
        print("Safety      : IR + Sharp + front ToF + yaw hold ACTIVE")
        print("==================================================")
        self.set_gui_status(
            f"CROSSING APPROVED EXIT {source_cell}->{DIR_NAMES[abs_dir]}",
            route=[source_cell],
        )

        self.turn_to_direction(abs_dir)

        if self.ir_replan_requested:
            self.ir_replan_requested = False
            self.stop_chassis()
            print(
                "[EXIT CONTINUE] live IR/Gimbal safety rejects the approved "
                "exit edge; not forcing motion"
            )
            return False

        # detect_exit=False is deliberate: the operator has explicitly
        # approved this single deferred edge. Collision safety remains active.
        ok = self.move_one_cell(
            source_cell=source_cell,
            abs_dir=abs_dir,
            detect_exit=False,
        )

        if not ok:
            self.stop_chassis()
            print("[EXIT CONTINUE] approved exit crossing failed safely")
            return False

        # Commit the approved connection only after a successful crossing.
        self.exit_candidates.pop(key, None)

        if abs_dir not in self.open_dirs.get(source_cell, []):
            self.open_dirs.setdefault(source_cell, []).append(abs_dir)

        if destination not in self.visited:
            self.parent[destination] = source_cell
            self.visited.add(destination)
        elif destination not in self.parent:
            self.parent[destination] = source_cell

        self.current = destination
        self.approved_exit_entry_cells.add(destination)
        self.map_complete = False

        # Force a fresh topology scan beyond the approved boundary.
        self.open_dirs.pop(destination, None)
        self.cell_scan_mm.pop(destination, None)
        self.dead_end_cells.discard(destination)

        print(
            f"[EXIT CONTINUE OK] crossed to {destination}; "
            "resuming DFS from the new side"
        )
        self.operator_selected_exit_candidate = None
        self.set_gui_status(
            f"EXIT CROSSED - RESUMING DFS AT {destination}",
            route=[destination],
        )

        if MAP_AUTOSAVE:
            self.save_map(final=False)

        return True


