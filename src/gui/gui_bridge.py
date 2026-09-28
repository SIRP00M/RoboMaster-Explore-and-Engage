#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mission Control GUI bridge and snapshot publisher."""

import math
import time

from config import *
from src.core.geometry import DIR_NAMES, DIR_VEC, neighbor


class GuiBridgeMixin:
    def attach_mission_gui(self, gui):
        self.mission_gui = gui
        self.publish_gui_state()


    def set_gui_status(self, status, route=None):
        self.gui_status_text = str(status)
        if route is not None:
            self.gui_route_preview = [tuple(c) for c in route]
        self.publish_gui_state()


    def build_gui_snapshot(self):
        """Build an immutable GUI snapshot on the mission thread."""
        cells = sorted(self.mapped_cells(), key=lambda p: (p[1], p[0]))
        pos = self.state.get_position()

        exit_records = []
        for (cell, d), rec in sorted(
            self.exit_candidates.items(),
            key=lambda item: (
                item[0][0][1], item[0][0][0], item[0][1]
            ),
        ):
            exit_records.append({
                'cell': [int(cell[0]), int(cell[1])],
                'dir_index': int(d),
                'dir': DIR_NAMES[d],
                'front_mm': rec.get('front_mm'),
                'reason': rec.get('reason'),
            })

        return {
            'status': self.gui_status_text,
            'root': [int(self.root[0]), int(self.root[1])],
            'current': [int(self.current[0]), int(self.current[1])],
            'heading': int(self.heading) % 4,
            'position': None if pos is None else [float(v) for v in pos],
            'cells': [[int(c[0]), int(c[1])] for c in cells],
            'visited': [
                [int(c[0]), int(c[1])]
                for c in sorted(self.visited, key=lambda p: (p[1], p[0]))
            ],
            'dead_end_cells': [
                [int(c[0]), int(c[1])]
                for c in sorted(self.dead_end_cells, key=lambda p: (p[1], p[0]))
            ],
            'open_dirs': [
                {
                    'cell': [int(cell[0]), int(cell[1])],
                    'dirs': [int(d) for d in dirs],
                }
                for cell, dirs in sorted(
                    self.open_dirs.items(), key=lambda item: (item[0][1], item[0][0])
                )
            ],
            'blocked_edges': [
                [
                    [int(a[0]), int(a[1])],
                    [int(b[0]), int(b[1])],
                ]
                for a, b in sorted(self.blocked_edges)
            ],
            'exit_candidates': exit_records,
            'known_entrance_dir': int(self.known_entrance_dir),
            'known_maze_ingress_dir': int(self.known_maze_ingress_dir),
            'route_preview': [
                [int(c[0]), int(c[1])] for c in self.gui_route_preview
            ],
            'targets': [dict(rec) for rec in self.detected_targets],
            'target_glimpses': [dict(rec) for rec in self.target_glimpses],
            'target_status': self.target_status_text,
            'target_fire_policy': self.get_target_fire_policy(),
            'last_fire_event': self.last_fire_event,
            'map_complete': bool(self.map_complete),
        }


    def publish_gui_state(self):
        gui = self.mission_gui
        if gui is None or not gui.available:
            return False
        try:
            return gui.post_snapshot(self.build_gui_snapshot())
        except Exception as e:
            print(f'[GUI WARN] state publish failed: {e}')
            return False


    def build_exit_candidate_options(self):
        """
        Build all reachable deferred exits for operator selection.

        Candidates are ranked by confirmed shortest-path cell count from the
        robot's current location, but the GUI never auto-selects a winner for
        motion; the operator can choose any reachable E# entry.
        """
        options = []
        self.transient_runtime_blocked_edges.clear()

        raw = []
        for (cell, d), rec in self.exit_candidates.items():
            key = (tuple(cell), int(d) % 4)
            try:
                path = self.shortest_confirmed_path(self.current, cell)
                reachable = True
                moves = len(path) - 1
            except Exception as e:
                path = []
                reachable = False
                moves = 10 ** 9
                error = str(e)
            else:
                error = None

            raw.append((
                0 if reachable else 1,
                moves,
                cell[1],
                cell[0],
                d,
                key,
                rec,
                path,
                error,
            ))

        raw.sort(key=lambda item: item[:5])

        for index, item in enumerate(raw, start=1):
            _unreachable, moves, _y, _x, d, key, rec, path, error = item
            cell = key[0]
            reachable = error is None
            shown_moves = (len(path) - 1) if reachable else None
            options.append({
                'id': f'E{index}',
                'key': key,
                'cell': tuple(cell),
                'dir_index': int(d),
                'dir': DIR_NAMES[d],
                'reachable': reachable,
                'moves': shown_moves,
                'route_distance_m': (
                    float(shown_moves) * CELL_LENGTH_M if reachable else 0.0
                ),
                'path': [tuple(c) for c in path],
                'front_mm': rec.get('front_mm'),
                'reason': rec.get('reason'),
                'error': error,
            })

        return options


    def print_map_summary(self):
        print("\n================ DFS MAP SUMMARY ================")
        print(f"Visited cells: {len(self.visited)}")
        print("Cells:", sorted(self.visited, key=lambda p: (p[1], p[0])))

        for cell in sorted(self.open_dirs, key=lambda p: (p[1], p[0])):
            dirs = [DIR_NAMES[d] for d in self.open_dirs[cell]]
            print(f"  {cell}: open={dirs}")

        if self.dead_end_cells:
            print("Hard dead-end cells:")
            for cell in sorted(self.dead_end_cells):
                scan = self.cell_scan_mm.get(cell, {})
                print(
                    f"  {cell}: "
                    f"L={scan.get('LEFT')} "
                    f"F={scan.get('FRONT')} "
                    f"R={scan.get('RIGHT')} mm"
                )

        if self.blocked_edges:
            print("Blocked / failed edges:")
            for edge in sorted(self.blocked_edges):
                print(" ", edge)

        print(
            f"Known maze ingress: {self.root} -> "
            f"{DIR_NAMES[self.known_maze_ingress_dir]}"
        )
        print(
            f"Known entrance/return: {self.root} -> "
            f"{DIR_NAMES[self.known_entrance_dir]}"
        )

        if self.entrance_corridor_profile:
            print(
                "  Entrance profile: "
                f"back={self.entrance_corridor_profile.get('back_center_mm')}mm "
                f"side_walls="
                f"{self.entrance_corridor_profile.get('corridor_side_walls_verified')}"
            )

        if self.exit_candidates:
            print("Deferred EXIT_CANDIDATES:")
            for (cell, d), rec in sorted(
                self.exit_candidates.items(),
                key=lambda item: (
                    item[0][0][1],
                    item[0][0][0],
                    item[0][1],
                ),
            ):
                print(
                    f"  {cell} -> {DIR_NAMES[d]} "
                    f"front={rec.get('front_mm')}mm "
                    f"reason={rec.get('reason')}"
                )

        print("=================================================")
        print(self.render_ascii_map())


