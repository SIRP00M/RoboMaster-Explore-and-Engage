"""Start mission control: python main.py. Use --help for all options."""

import argparse
from src import config as cfg
from src.config import load_arena_profile
from src.gui import tk, ttk


def _run_headless(explorer):
    try:
        if explorer.connect():
            explorer.run_selected_mission()
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        explorer.running = False
        explorer.safe_stop()
        explorer.save_map(final=False)
    except Exception as exc:
        explorer.fault("UNEXPECTED", "{}: {}".format(type(exc).__name__, exc), "SAFE STOP + save partial map")
        explorer.enter_safe_pause("unexpected exception contained: {}: {}".format(type(exc).__name__, exc))
    finally:
        explorer.cleanup()


def main():
    parser = argparse.ArgumentParser(description="RoboMaster DFS + Target Mission Control")
    parser.add_argument("--no-gui", action="store_true", help="run immediately without Tk GUI")
    parser.add_argument(
        "--fire", choices=("INFRARED", "WATER"), default=cfg.TARGET_FIRE_MODE_DEFAULT,
        help="initial fire mode (GUI can change it live)",
    )
    parser.add_argument(
        "--shots", type=int, choices=cfg.TARGET_FIRE_BURST_OPTIONS,
        default=cfg.TARGET_FIRE_BURST_DEFAULT,
        help="shots per locked target; GUI can change 1-6 live",
    )
    parser.add_argument(
        "--round", dest="mission_round", choices=(1, 2), type=int, default=1,
        help="1=explore/map/fire and save attack memory, 2=load map + shortest attack",
    )
    parser.add_argument(
        "--arena-config", type=str, default=None,
        help="optional JSON arena profile (width/height/origin/start); overrides built-in profile",
    )
    parser.add_argument("--grid-width", type=int, default=None, help="arena width in logical cells")
    parser.add_argument("--grid-height", type=int, default=None, help="arena height in logical cells")
    parser.add_argument("--grid-x-min", type=int, default=None, help="minimum logical x coordinate")
    parser.add_argument("--grid-y-min", type=int, default=None, help="minimum logical y coordinate")
    parser.add_argument("--start-x", type=int, default=None, help="runtime start-cell x")
    parser.add_argument("--start-y", type=int, default=None, help="runtime start-cell y")
    parser.add_argument(
        "--no-boundary-guard", action="store_true",
        help="disable configured perimeter rejection (normally leave enabled)",
    )
    args = parser.parse_args()

    start_override = None
    if args.start_x is not None or args.start_y is not None:
        if args.start_x is None or args.start_y is None:
            parser.error("--start-x and --start-y must be supplied together")
        start_override = [args.start_x, args.start_y]

    try:
        load_arena_profile(
            args.arena_config,
            overrides={
                "width_cells": args.grid_width,
                "height_cells": args.grid_height,
                "x_min": args.grid_x_min,
                "y_min": args.grid_y_min,
                "start_cell": start_override,
                "boundary_guard": False if args.no_boundary_guard else None,
            },
        )
    except Exception as exc:
        parser.error("invalid arena profile: {}: {}".format(type(exc).__name__, exc))

    from src.navigation import DFSMapOnlyExplorer
    from src.gui import MissionControlGUI

    explorer = DFSMapOnlyExplorer()
    explorer.set_fire_mode(args.fire)
    explorer.set_fire_burst_count(args.shots)
    explorer.set_mission_mode("ROUND{}".format(args.mission_round))
    use_gui = bool(cfg.CONTROL_GUI_ENABLED and not args.no_gui and tk is not None and ttk is not None)
    if use_gui:
        try:
            MissionControlGUI(
                explorer, initial_fire_mode=args.fire, initial_burst=args.shots,
                initial_round="ROUND{}".format(args.mission_round),
            ).run()
            return
        except Exception as exc:
            print("[GUI WARN] {}: {} -> headless mode".format(type(exc).__name__, exc))
    _run_headless(explorer)


if __name__ == "__main__":
    main()
