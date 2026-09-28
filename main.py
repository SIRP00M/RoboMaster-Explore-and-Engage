#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP Autonomous Maze Exploration & Target Engagement
------------------------------------------------------------
Main Command-Line Interface (CLI) and Mission Launcher.

Usage:
  python3 main.py --mode explore
  python3 main.py --mode auto
  python3 main.py --mode known --map maps/latest_map.json --goal 2,4
  python3 main.py --no-gui
"""

import argparse
from pathlib import Path

from config import MAP_LATEST_JSON
from src.gui.mission_control import MissionControlGUI
from src.strategy.explorer import DFSMazeExplorer


def parse_goal(text):
    """Parse goal string 'x,y' into an integer tuple (x, y)."""
    if text is None:
        return None

    parts = str(text).split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "goal must be x,y, for example --goal 2,4"
        )

    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "goal coordinates must be integers"
        ) from e


def build_arg_parser():
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RoboMaster DFS explorer with persistent reusable grid maps."
    )

    parser.add_argument(
        "--mode",
        choices=("explore", "known", "auto"),
        default="explore",
        help=(
            "explore: learn/save a map; "
            "known: load saved map; "
            "auto: use saved map if present, otherwise explore"
        ),
    )

    parser.add_argument(
        "--map",
        dest="map_path",
        default=str(MAP_LATEST_JSON),
        help="map JSON used by known/auto mode",
    )

    parser.add_argument(
        "--goal",
        type=parse_goal,
        default=None,
        help="known-map target cell x,y. If omitted, replay/cover the full saved map.",
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable Mission Control GUI and use terminal-only exit selection.",
    )

    parser.add_argument(
        "--no-target-vision",
        action="store_true",
        help="Disable camera target detection/sweep. Maze exploration still runs.",
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Keep target detection active but do not open the OpenCV preview.",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    explorer = DFSMazeExplorer()
    explorer.target_vision_enabled = not args.no_target_vision
    explorer.preview_enabled = not args.no_preview
    mission_gui = None

    # GUI starts before robot connection so the operator sees connection/
    # initialization state as part of the same mission dashboard.
    if not args.no_gui:
        mission_gui = MissionControlGUI(explorer)
        if mission_gui.start():
            explorer.attach_mission_gui(mission_gui)
            explorer.set_gui_status('CONNECTING TO ROBOMASTER')
        else:
            # Ready-timeout expired, but the Tk thread may still be mid-init.
            # Tell it to close itself once it comes up so we never leave a
            # disconnected, unattached window on screen for the whole mission.
            mission_gui.close()
            mission_gui = None

    try:
        mode = args.mode

        if mode == "auto":
            if Path(args.map_path).exists():
                mode = "known"
                print(f"[MODE AUTO] found {args.map_path} -> KNOWN MAP")
            else:
                mode = "explore"
                print(f"[MODE AUTO] no map at {args.map_path} -> EXPLORE")

        # Load topology before connecting; physical yaw reference is still
        # captured fresh during connect().
        if mode == "known":
            explorer.load_map(args.map_path)
            explorer.set_gui_status('KNOWN MAP LOADED')

        explorer.connect()

        # The robot is connected and stationary here. Do not enter DFS/known
        # navigation until the operator has explicitly armed a target-fire rule.
        if mission_gui is not None and mission_gui.available:
            explorer.set_gui_status('WAITING FOR TARGET FIRE RULES - ROBOT STATIONARY')
            fire_policy = mission_gui.request_target_fire_policy()
            if fire_policy is None:
                raise RuntimeError(
                    'Mission Control closed before target fire policy was armed'
                )
            explorer.set_target_fire_policy(fire_policy)
            explorer.set_gui_status('TARGET FIRE RULES ARMED - STARTING MISSION')
        else:
            # Terminal/no-GUI runs fail safe: mapping/navigation may continue,
            # but the physical blaster remains disarmed.
            explorer.set_target_fire_policy({
                'armed': True,
                'mode': 'selected',
                'fire_type': 'infrared',
                'auto_fire': False,
                'selected_color_shapes': [],
                'sdk_enabled': False,
                'sdk_labels': [],
            })
            print('[FIRE POLICY] --no-gui => physical firing DISABLED')

        if mode == "explore":
            explorer.run_dfs()
        else:
            explorer.set_gui_status('RUNNING KNOWN-MAP NAVIGATION')
            explorer.run_known_map(goal=args.goal)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        explorer.running = False
        explorer.set_gui_status('STOPPED BY CTRL+C')

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        explorer.set_gui_status(f'ERROR: {type(e).__name__}: {e}')

    finally:
        explorer.cleanup()

        if mission_gui is not None and mission_gui.available:
            # Keep the final map visible until the operator closes the window.
            try:
                mission_gui.post_mission_complete(explorer.build_gui_snapshot())
                print('[GUI] Mission complete. Close Mission Control to exit.')
                mission_gui.wait_closed()
            except KeyboardInterrupt:
                mission_gui.close()


if __name__ == "__main__":
    main()
