"""Complete mission-control GUI, arena/speed editors and map display."""

from . import config as cfg
import json
import math
import threading
from .config import apply_arena_profile, current_arena_profile
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except Exception:
    tk = None
    ttk = None
    filedialog = None


class MissionControlGUI:
    def __init__(self, explorer, initial_fire_mode="INFRARED", initial_burst=1, initial_round="ROUND1"):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is unavailable")
        self.explorer = explorer
        self.explorer.set_fire_mode(initial_fire_mode)
        self.explorer.set_fire_burst_count(initial_burst)
        self.explorer.set_mission_mode(initial_round)
        self.root = tk.Tk()
        self.root.title("RoboMaster Mission Control - Round 1 Map / Round 2 Shortest Attack")
        self.root.geometry("1080x760")
        self.root.minsize(940, 650)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.mission_thread = None
        self.mission_started = False
        self.mission_finished_announced = False
        self.status_override = None
        self.closing = False

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.mission_page = ttk.Frame(self.notebook)
        self.target_page = ttk.Frame(self.notebook)
        self.arena_page = ttk.Frame(self.notebook)
        self.speed_page = ttk.Frame(self.notebook)
        self.notebook.add(self.mission_page, text="1. Mission / Live Map")
        self.notebook.add(self.target_page, text="2. Target Selection (16)")
        self.notebook.add(self.arena_page, text="3. Arena / Field Setup")
        self.notebook.add(self.speed_page, text="4. Speed / Motion Tuning")

        # ---------------- Page 1: mission page ----------------
        # Competition controls stay fixed at TOP-CENTER so START/STOP are always
        # reachable without hunting through the side panel.
        top_controls = ttk.Frame(self.mission_page, padding=(8, 8, 8, 2))
        top_controls.pack(fill="x")
        top_center = ttk.Frame(top_controls)
        top_center.pack(anchor="center")
        self.start_btn = ttk.Button(
            top_center, text="START MISSION", width=24, command=self._start_mission
        )
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(
            top_center, text="STOP + SAVE NOW", width=24, command=self._stop_mission, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=6)

        outer = ttk.Frame(self.mission_page, padding=4)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        map_frame = ttk.LabelFrame(outer, text="Live Map | Orange=Breadcrumb | Purple=Round-2 Shortest Preview")
        map_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        map_frame.rowconfigure(0, weight=1)
        map_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            map_frame,
            width=cfg.CONTROL_GUI_CANVAS_W,
            height=cfg.CONTROL_GUI_CANVAS_H,
            background="white",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(outer)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        mode_box = ttk.LabelFrame(side, text="Mission Round", padding=8)
        mode_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.round_var = tk.StringVar(value=self.explorer.mission_mode)
        mode_row = ttk.Frame(mode_box)
        mode_row.pack(fill="x")
        ttk.Radiobutton(
            mode_row, text="Round 1: Explore + Map + Fire",
            variable=self.round_var, value="ROUND1", command=self._round_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_row, text="Round 2: Known Map + Shortest Attack",
            variable=self.round_var, value="ROUND2", command=self._round_changed,
        ).pack(anchor="w")
        ttk.Label(
            mode_box,
            text=(
                "ROUND 2: load ONLY maps/round1_attack_memory.json. "
                "This ONE file contains the learned map + proven firing anchors + saved target classes. "
                "Do NOT load latest_map.json or latest_targets.json separately."
            ),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

        memory_row = ttk.Frame(mode_box)
        memory_row.pack(fill="x", pady=(7, 0))
        ttk.Button(
            memory_row, text="1) LOAD ROUND-1 MAP + SHOT POINTS", command=self._load_round1_snapshot
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            memory_row, text="2) PREVIEW SHORTEST ROUTE", command=self._preview_round2_route
        ).pack(side="left")
        ttk.Button(
            mode_box, text="LOAD DEFAULT + PREVIEW ROUND 2",
            command=self._load_default_and_preview_round2
        ).pack(anchor="w", pady=(7, 0))
        self.memory_status_text = tk.StringVar(
            value="Round-2 snapshot: not loaded (default maps/round1_attack_memory.json)"
        )
        ttk.Label(
            mode_box, textvariable=self.memory_status_text, wraplength=340, justify="left"
        ).pack(anchor="w", pady=(5, 0))
        self.route_preview_text = tk.StringVar(value="Route preview: not built")
        ttk.Label(
            mode_box, textvariable=self.route_preview_text, wraplength=340, justify="left"
        ).pack(anchor="w", pady=(3, 0))

        fire_box = ttk.LabelFrame(side, text="Fire Type / Burst", padding=8)
        fire_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.fire_var = tk.StringVar(value=self.explorer.get_fire_mode())
        ttk.Radiobutton(
            fire_box, text="Infrared", variable=self.fire_var,
            value="INFRARED", command=self._fire_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            fire_box, text="Water Blaster", variable=self.fire_var,
            value="WATER", command=self._fire_changed,
        ).pack(anchor="w")
        ttk.Separator(fire_box, orient="horizontal").pack(fill="x", pady=4)
        self.burst_var = tk.IntVar(value=self.explorer.get_fire_burst_count())
        burst_row = ttk.Frame(fire_box)
        burst_row.pack(anchor="w")
        ttk.Label(burst_row, text="Shots:").pack(side="left", padx=(0, 6))
        self.burst_combo = ttk.Combobox(
            burst_row, textvariable=self.burst_var,
            values=cfg.TARGET_FIRE_BURST_OPTIONS, state="readonly", width=5,
        )
        self.burst_combo.pack(side="left")
        self.burst_combo.bind("<<ComboboxSelected>>", self._burst_changed)
        ttk.Label(
            fire_box,
            text="Range gate <= 1.2 m; {:.2f}s between burst shots.".format(
                cfg.TARGET_FIRE_BURST_INTERVAL_SEC
            ), wraplength=340,
        ).pack(anchor="w", pady=(4, 0))

        geom_box = ttk.LabelFrame(side, text="Aim Geometry", padding=8)
        geom_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.geometry_text = tk.StringVar()
        ttk.Label(geom_box, textvariable=self.geometry_text, justify="left").pack(anchor="w")

        status_box = ttk.LabelFrame(side, text="Robot / Target Status", padding=8)
        status_box.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        self.status_text = tk.StringVar(value="READY TO START")
        ttk.Label(
            status_box, textvariable=self.status_text, justify="left",
            wraplength=340,
        ).pack(anchor="w", fill="x")

        aim_box = ttk.LabelFrame(side, text="Last Fire Solution", padding=8)
        aim_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.aim_text = tk.StringVar(value="No fire solution yet")
        ttk.Label(aim_box, textvariable=self.aim_text, justify="left", wraplength=340).pack(anchor="w")

        # ---------------- Page 2: 16 color x shape rules ----------------
        target_outer = ttk.Frame(self.target_page, padding=14)
        target_outer.pack(fill="both", expand=True)
        ttk.Label(
            target_outer,
            text=(
                "Tick ONLY the color/shape combinations that are legal targets. "
                "Unselected combinations are still detected and remembered, but will never fire. "
                "Round 2 also ignores unselected Round-1 firing hints."
            ),
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 12))

        self.target_vars = {}
        pretty_shape = {
            "SQUARE": "Square",
            "CIRCLE": "Circle",
            "RECT_HORIZONTAL": "Rectangle H",
            "RECT_VERTICAL": "Rectangle V",
        }
        for col_idx, shape in enumerate(cfg.TARGET_FILTER_SHAPES, start=1):
            ttk.Label(target_outer, text=pretty_shape[shape]).grid(
                row=1, column=col_idx, padx=10, pady=4, sticky="w"
            )
        for row_idx, color in enumerate(cfg.TARGET_FILTER_COLORS, start=2):
            ttk.Label(target_outer, text=color.title()).grid(
                row=row_idx, column=0, padx=(0, 12), pady=7, sticky="w"
            )
            for col_idx, shape in enumerate(cfg.TARGET_FILTER_SHAPES, start=1):
                key = (color, shape)
                var = tk.BooleanVar(value=True)
                self.target_vars[key] = var
                ttk.Checkbutton(
                    target_outer,
                    variable=var,
                    command=self._target_filter_changed,
                ).grid(row=row_idx, column=col_idx, padx=10, pady=7, sticky="w")

        target_buttons = ttk.Frame(target_outer)
        target_buttons.grid(row=7, column=0, columnspan=5, sticky="w", pady=(16, 8))
        ttk.Button(target_buttons, text="SELECT ALL 16", command=self._select_all_targets).pack(side="left", padx=(0, 8))
        ttk.Button(target_buttons, text="CLEAR ALL", command=self._clear_all_targets).pack(side="left")
        self.target_filter_text = tk.StringVar(value="16 / 16 target classes enabled")
        ttk.Label(target_outer, textvariable=self.target_filter_text).grid(
            row=8, column=0, columnspan=5, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            target_outer,
            text=(
                "Examples: Red + Square ON and Blue + Square OFF means a red square can fire, "
                "while a blue square is detection-only. Changes are applied live before the next shot."
            ),
            wraplength=900,
            justify="left",
        ).grid(row=9, column=0, columnspan=5, sticky="w", pady=(10, 0))

        # ---------------- Page 3: generic arena / field setup ----------------
        arena_outer = ttk.Frame(self.arena_page, padding=14)
        arena_outer.pack(fill="both", expand=True)
        arena_outer.columnconfigure(0, weight=1)
        arena_outer.columnconfigure(1, weight=2)
        arena_outer.rowconfigure(1, weight=1)

        ttk.Label(
            arena_outer,
            text=(
                "Generic arena geometry. The DFS has no special fake-exit coordinate: any sensed OPEN edge "
                "whose destination falls outside these bounds is rejected BEFORE translation. "
                "Apply before Round 1. Round 2 automatically restores the geometry saved in the Round-1 snapshot."
            ),
            wraplength=980, justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        arena_form = ttk.LabelFrame(arena_outer, text="Arena Profile", padding=12)
        arena_form.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        arena_form.columnconfigure(1, weight=1)

        active_profile = current_arena_profile()
        self.arena_name_var = tk.StringVar(value=str(active_profile["name"]))
        self.arena_width_var = tk.StringVar(value=str(active_profile["width_cells"]))
        self.arena_height_var = tk.StringVar(value=str(active_profile["height_cells"]))
        self.arena_xmin_var = tk.StringVar(value=str(active_profile["x_min"]))
        self.arena_ymin_var = tk.StringVar(value=str(active_profile["y_min"]))
        self.arena_startx_var = tk.StringVar(value=str(active_profile["start_cell"][0]))
        self.arena_starty_var = tk.StringVar(value=str(active_profile["start_cell"][1]))
        self.arena_boundary_var = tk.BooleanVar(value=bool(active_profile["boundary_guard"]))

        arena_fields = (
            ("Profile name", self.arena_name_var),
            ("Width (cells)", self.arena_width_var),
            ("Height (cells)", self.arena_height_var),
            ("X minimum", self.arena_xmin_var),
            ("Y minimum", self.arena_ymin_var),
            ("Start X", self.arena_startx_var),
            ("Start Y", self.arena_starty_var),
        )
        for row_i, (label, var) in enumerate(arena_fields):
            ttk.Label(arena_form, text=label + ":").grid(row=row_i, column=0, sticky="w", pady=4, padx=(0, 8))
            ttk.Entry(arena_form, textvariable=var, width=18).grid(row=row_i, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(
            arena_form, text="Boundary Guard (reject moves outside arena)",
            variable=self.arena_boundary_var, command=self._draw_arena_preview,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 4))

        arena_buttons = ttk.Frame(arena_form)
        arena_buttons.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Button(
            arena_buttons, text="APPLY TO ROUND 1", command=self._apply_arena_from_gui
        ).pack(side="left", padx=(0, 6))
        ttk.Button(arena_buttons, text="LOAD JSON", command=self._load_arena_json).pack(side="left", padx=(0, 6))
        ttk.Button(arena_buttons, text="SAVE JSON", command=self._save_arena_json).pack(side="left", padx=(0, 6))
        ttk.Button(arena_buttons, text="RESET DEFAULT", command=self._reset_arena_defaults).pack(side="left")

        self.arena_status_text = tk.StringVar(value="Arena settings ready. Apply before START.")
        ttk.Label(
            arena_form, textvariable=self.arena_status_text, wraplength=390, justify="left"
        ).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        arena_preview_box = ttk.LabelFrame(arena_outer, text="Arena Preview", padding=8)
        arena_preview_box.grid(row=1, column=1, sticky="nsew")
        arena_preview_box.rowconfigure(0, weight=1)
        arena_preview_box.columnconfigure(0, weight=1)
        self.arena_canvas = tk.Canvas(
            arena_preview_box, width=520, height=500, background="white", highlightthickness=0
        )
        self.arena_canvas.grid(row=0, column=0, sticky="nsew")
        self.arena_preview_text = tk.StringVar(value="")
        ttk.Label(
            arena_preview_box, textvariable=self.arena_preview_text, justify="left", wraplength=520
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # Redraw the field preview while typing; invalid intermediate input is simply ignored.
        for var in (
            self.arena_width_var, self.arena_height_var, self.arena_xmin_var, self.arena_ymin_var,
            self.arena_startx_var, self.arena_starty_var, self.arena_name_var,
        ):
            try:
                var.trace_add("write", lambda *_args: self._draw_arena_preview())
            except Exception:
                pass

        # ---------------- Page 4: speed / motion tuning ----------------
        speed_outer = ttk.Frame(self.speed_page, padding=14)
        speed_outer.pack(fill="both", expand=True)
        speed_outer.columnconfigure(0, weight=1)
        speed_outer.columnconfigure(1, weight=1)

        ttk.Label(
            speed_outer,
            text=(
                "Set mission motion limits here. Values are validated before applying. "
                "Speed changes are locked while a mission is running so one cell cannot change profile mid-run. "
                "The live panel keeps showing commanded and odometry-estimated speed."
            ),
            wraplength=980, justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        tune_box = ttk.LabelFrame(speed_outer, text="Motion Limits", padding=12)
        tune_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        tune_box.columnconfigure(1, weight=1)

        self.speed_explore_var = tk.StringVar(value=f"{cfg.DFS_EXPLORE_SPEED_MPS:.3f}")
        self.speed_known_var = tk.StringVar(value=f"{cfg.DFS_KNOWN_SPEED_MPS:.3f}")
        self.speed_explore_min_var = tk.StringVar(value=f"{cfg.DFS_EXPLORE_APPROACH_MIN_MPS:.3f}")
        self.speed_known_min_var = tk.StringVar(value=f"{cfg.DFS_KNOWN_APPROACH_MIN_MPS:.3f}")
        self.speed_target_fast_var = tk.StringVar(value=f"{cfg.TARGET_SIDE_SHIFT_SPEED_MPS:.3f}")
        self.speed_target_med_var = tk.StringVar(value=f"{cfg.TARGET_SIDE_SHIFT_MED_SPEED_MPS:.3f}")
        self.speed_target_slow_var = tk.StringVar(value=f"{cfg.TARGET_SIDE_SHIFT_SLOW_MPS:.3f}")
        self.speed_ir_var = tk.StringVar(value=f"{cfg.IR_SIMPLE_STRAFE_SPEED_MPS:.3f}")
        self.speed_sharp_var = tk.StringVar(value=f"{cfg.SHARP_SIDE_ESCAPE_SPEED_MPS:.3f}")
        self.speed_turn_var = tk.StringVar(value=f"{cfg.TURN_MAX_DPS:.1f}")
        self.speed_accel_var = tk.StringVar(value=f"{cfg.MOVE_FORWARD_ACCEL_LIMIT_MPS2:.2f}")
        self.speed_settle_var = tk.StringVar(value=f"{cfg.POST_MOVE_TILE_SETTLE_SEC:.3f}")

        speed_fields = (
            ("Explore cruise (m/s)", self.speed_explore_var),
            ("Known / Backtrack / Round2 (m/s)", self.speed_known_var),
            ("Explore approach min (m/s)", self.speed_explore_min_var),
            ("Known approach min (m/s)", self.speed_known_min_var),
            ("Target side shift FAST (m/s)", self.speed_target_fast_var),
            ("Target side shift MED (m/s)", self.speed_target_med_var),
            ("Target side shift SLOW (m/s)", self.speed_target_slow_var),
            ("IR side recovery base (m/s)", self.speed_ir_var),
            ("Sharp side escape (m/s)", self.speed_sharp_var),
            ("Turn max (deg/s)", self.speed_turn_var),
            ("Forward accel limit (m/s^2)", self.speed_accel_var),
            ("Post-cell settle (s)", self.speed_settle_var),
        )
        for ri, (label, var) in enumerate(speed_fields):
            ttk.Label(tune_box, text=label + ":").grid(row=ri, column=0, sticky="w", pady=3, padx=(0, 8))
            ttk.Entry(tune_box, textvariable=var, width=12).grid(row=ri, column=1, sticky="ew", pady=3)

        preset_row = ttk.Frame(tune_box)
        preset_row.grid(row=len(speed_fields), column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Label(preset_row, text="Preset:").pack(side="left", padx=(0, 6))
        ttk.Button(preset_row, text="SLIPPERY SAFE", command=lambda: self._load_speed_preset("SAFE")).pack(side="left", padx=(0, 5))
        ttk.Button(preset_row, text="BALANCED", command=lambda: self._load_speed_preset("BALANCED")).pack(side="left", padx=(0, 5))
        ttk.Button(preset_row, text="FAST", command=lambda: self._load_speed_preset("FAST")).pack(side="left")

        action_row = ttk.Frame(tune_box)
        action_row.grid(row=len(speed_fields)+1, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.speed_apply_btn = ttk.Button(action_row, text="APPLY SPEED SETTINGS", command=self._apply_speed_from_gui)
        self.speed_apply_btn.pack(side="left", padx=(0, 6))
        ttk.Button(action_row, text="RESTORE CURRENT", command=self._sync_speed_vars_from_runtime).pack(side="left")
        self.speed_status_text = tk.StringVar(value="Speed settings ready. Apply before START.")
        ttk.Label(tune_box, textvariable=self.speed_status_text, wraplength=430, justify="left").grid(
            row=len(speed_fields)+2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        live_box = ttk.LabelFrame(speed_outer, text="Live Chassis Speed", padding=12)
        live_box.grid(row=1, column=1, sticky="nsew")
        live_box.columnconfigure(0, weight=1)
        self.speed_live_text = tk.StringVar(value="Robot not moving / telemetry not connected yet")
        ttk.Label(
            live_box, textvariable=self.speed_live_text, justify="left",
            font=("TkFixedFont", 11), wraplength=480,
        ).grid(row=0, column=0, sticky="nw")
        ttk.Label(
            live_box,
            text=(
                "Commanded = velocity requested by this program.\n"
                "Measured = filtered estimate from chassis odometry callbacks.\n"
                "Forward/Right are projected into the current logical heading."
            ),
            wraplength=470, justify="left",
        ).grid(row=1, column=0, sticky="nw", pady=(12, 0))

        self._target_filter_changed()
        self._draw_arena_preview()
        self._refresh()

    def _arena_profile_from_vars(self):
        """Validate and return the profile currently typed in the Arena tab."""
        name = str(self.arena_name_var.get() or "gui_arena").strip() or "gui_arena"
        try:
            width = int(str(self.arena_width_var.get()).strip())
            height = int(str(self.arena_height_var.get()).strip())
            x_min = int(str(self.arena_xmin_var.get()).strip())
            y_min = int(str(self.arena_ymin_var.get()).strip())
            start_x = int(str(self.arena_startx_var.get()).strip())
            start_y = int(str(self.arena_starty_var.get()).strip())
        except Exception:
            raise ValueError("width/height/origin/start must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if width > 100 or height > 100:
            raise ValueError("width/height > 100 cells is rejected as likely input error")
        x_max = x_min + width - 1
        y_max = y_min + height - 1
        if not (x_min <= start_x <= x_max and y_min <= start_y <= y_max):
            raise ValueError(
                "start ({},{}) is outside x={}..{}, y={}..{}".format(
                    start_x, start_y, x_min, x_max, y_min, y_max
                )
            )
        return {
            "name": name,
            "width_cells": width,
            "height_cells": height,
            "x_min": x_min,
            "y_min": y_min,
            "start_cell": [start_x, start_y],
            "boundary_guard": bool(self.arena_boundary_var.get()),
        }

    def _sync_arena_vars_from_runtime(self):
        profile = current_arena_profile()
        self.arena_name_var.set(str(profile["name"]))
        self.arena_width_var.set(str(profile["width_cells"]))
        self.arena_height_var.set(str(profile["height_cells"]))
        self.arena_xmin_var.set(str(profile["x_min"]))
        self.arena_ymin_var.set(str(profile["y_min"]))
        self.arena_startx_var.set(str(profile["start_cell"][0]))
        self.arena_starty_var.set(str(profile["start_cell"][1]))
        self.arena_boundary_var.set(bool(profile["boundary_guard"]))
        self._draw_arena_preview()

    def _apply_arena_from_gui(self, quiet=False):
        if self.mission_started:
            if not quiet:
                self.arena_status_text.set("LOCKED: stop/restart before changing arena geometry.")
            return False
        try:
            profile = self._arena_profile_from_vars()
            apply_arena_profile(profile, source="GUI Arena tab")
            # New geometry means a fresh Round-1 logical frame.  Never keep an old
            # map loaded under different coordinates/bounds.
            self.explorer.reset_map_for_fresh_round1()
            self.round_var.set("ROUND1")
            self.explorer.set_mission_mode("ROUND1")
            self.memory_status_text.set(
                "Round-2 snapshot cleared from RAM; new arena requires a fresh Round 1"
            )
            self.route_preview_text.set("Route preview: not built")
            self._sync_arena_vars_from_runtime()
            self.status_override = None
            self.status_text.set("ARENA APPLIED - ready for fresh ROUND1")
            self.arena_status_text.set(
                "APPLIED: {}x{} | x={}..{} y={}..{} | start={} | boundary_guard={}".format(
                    cfg.GRID_WIDTH_CELLS, cfg.GRID_HEIGHT_CELLS, cfg.GRID_X_MIN, cfg.GRID_X_MAX,
                    cfg.GRID_Y_MIN, cfg.GRID_Y_MAX, cfg.ROOT_CELL, cfg.FIELD_BOUNDARY_GUARD_ENABLED
                )
            )
            return True
        except Exception as exc:
            msg = "ARENA INVALID: {}".format(exc)
            self.arena_status_text.set(msg)
            if not quiet:
                self.status_text.set(msg)
            return False

    def _load_arena_json(self):
        if self.mission_started:
            self.arena_status_text.set("LOCKED: cannot load arena JSON while mission is running")
            return False
        if filedialog is None:
            self.arena_status_text.set("File dialog unavailable")
            return False
        path = filedialog.askopenfilename(
            title="Load Arena Profile", initialdir=str(Path.cwd()),
            filetypes=(("Arena JSON", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return False
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("arena JSON must contain one object")
            merged = dict(cfg.DEFAULT_ARENA_PROFILE)
            merged.update(payload)
            start = merged.get("start_cell", [0, 0])
            self.arena_name_var.set(str(merged.get("name") or Path(path).stem))
            self.arena_width_var.set(str(merged.get("width_cells")))
            self.arena_height_var.set(str(merged.get("height_cells")))
            self.arena_xmin_var.set(str(merged.get("x_min", 0)))
            self.arena_ymin_var.set(str(merged.get("y_min", 0)))
            self.arena_startx_var.set(str(start[0]))
            self.arena_starty_var.set(str(start[1]))
            self.arena_boundary_var.set(bool(merged.get("boundary_guard", True)))
            self.arena_status_text.set("Loaded into editor: {} (press APPLY TO ROUND 1)".format(path))
            self._draw_arena_preview()
            return True
        except Exception as exc:
            self.arena_status_text.set("LOAD FAILED: {}: {}".format(type(exc).__name__, exc))
            return False

    def _save_arena_json(self):
        if filedialog is None:
            self.arena_status_text.set("File dialog unavailable")
            return False
        try:
            profile = self._arena_profile_from_vars()
        except Exception as exc:
            self.arena_status_text.set("SAVE BLOCKED - invalid arena: {}".format(exc))
            return False
        path = filedialog.asksaveasfilename(
            title="Save Arena Profile", initialdir=str(Path.cwd()),
            initialfile="arena_config.json", defaultextension=".json",
            filetypes=(("Arena JSON", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return False
        try:
            Path(path).write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.arena_status_text.set("Saved arena profile: {}".format(path))
            return True
        except Exception as exc:
            self.arena_status_text.set("SAVE FAILED: {}: {}".format(type(exc).__name__, exc))
            return False

    def _reset_arena_defaults(self):
        if self.mission_started:
            self.arena_status_text.set("LOCKED: cannot reset arena while mission is running")
            return False
        p = dict(cfg.DEFAULT_ARENA_PROFILE)
        start = p.get("start_cell", [0, 0])
        self.arena_name_var.set(str(p.get("name", "competition_default")))
        self.arena_width_var.set(str(p.get("width_cells", 6)))
        self.arena_height_var.set(str(p.get("height_cells", 6)))
        self.arena_xmin_var.set(str(p.get("x_min", 0)))
        self.arena_ymin_var.set(str(p.get("y_min", 0)))
        self.arena_startx_var.set(str(start[0]))
        self.arena_starty_var.set(str(start[1]))
        self.arena_boundary_var.set(bool(p.get("boundary_guard", True)))
        self.arena_status_text.set("Default loaded into editor; press APPLY TO ROUND 1")
        self._draw_arena_preview()
        return True

    def _draw_arena_preview(self):
        canvas = getattr(self, "arena_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        try:
            p = self._arena_profile_from_vars()
        except Exception as exc:
            self.arena_preview_text.set("Preview unavailable: {}".format(exc))
            return
        w = int(p["width_cells"]); h = int(p["height_cells"])
        xmin = int(p["x_min"]); ymin = int(p["y_min"])
        xmax = xmin + w - 1; ymax = ymin + h - 1
        sx, sy = int(p["start_cell"][0]), int(p["start_cell"][1])
        cw = max(360, int(canvas.winfo_width() or 520))
        ch = max(360, int(canvas.winfo_height() or 500))
        pad = 34.0
        cell_px = max(10.0, min((cw - 2*pad) / max(1, w), (ch - 2*pad) / max(1, h)))
        grid_w = cell_px * w; grid_h = cell_px * h
        ox = (cw - grid_w) / 2.0; oy = (ch - grid_h) / 2.0
        for yy in range(h):
            for xx in range(w):
                gx = xmin + xx; gy = ymin + yy
                x0 = ox + xx * cell_px
                # Logical +Y is North/up, so max Y is drawn at the top.
                y0 = oy + (ymax - gy) * cell_px
                canvas.create_rectangle(x0, y0, x0+cell_px, y0+cell_px, outline="#777777")
                if cell_px >= 34:
                    canvas.create_text(x0+cell_px/2, y0+cell_px/2, text="{},{}".format(gx, gy), fill="#777777")
                if (gx, gy) == (sx, sy):
                    margin = max(3.0, cell_px * 0.16)
                    canvas.create_oval(
                        x0+margin, y0+margin, x0+cell_px-margin, y0+cell_px-margin,
                        fill="#d9ecff", outline="#1f5f99", width=2,
                    )
                    canvas.create_text(x0+cell_px/2, y0+cell_px/2, text="START\nN↑", fill="#103b63")
        border_width = 5 if p["boundary_guard"] else 2
        border_dash = None if p["boundary_guard"] else (6, 4)
        canvas.create_rectangle(ox, oy, ox+grid_w, oy+grid_h, outline="#111111", width=border_width, dash=border_dash)
        self.arena_preview_text.set(
            "{} | {}x{} | bounds x={}..{}, y={}..{} | start=({}, {}) facing N | Boundary Guard={}".format(
                p["name"], w, h, xmin, xmax, ymin, ymax, sx, sy, "ON" if p["boundary_guard"] else "OFF"
            )
        )

    def _sync_speed_vars_from_runtime(self):
        self.speed_explore_var.set(f"{cfg.DFS_EXPLORE_SPEED_MPS:.3f}")
        self.speed_known_var.set(f"{cfg.DFS_KNOWN_SPEED_MPS:.3f}")
        self.speed_explore_min_var.set(f"{cfg.DFS_EXPLORE_APPROACH_MIN_MPS:.3f}")
        self.speed_known_min_var.set(f"{cfg.DFS_KNOWN_APPROACH_MIN_MPS:.3f}")
        self.speed_target_fast_var.set(f"{cfg.TARGET_SIDE_SHIFT_SPEED_MPS:.3f}")
        self.speed_target_med_var.set(f"{cfg.TARGET_SIDE_SHIFT_MED_SPEED_MPS:.3f}")
        self.speed_target_slow_var.set(f"{cfg.TARGET_SIDE_SHIFT_SLOW_MPS:.3f}")
        self.speed_ir_var.set(f"{cfg.IR_SIMPLE_STRAFE_SPEED_MPS:.3f}")
        self.speed_sharp_var.set(f"{cfg.SHARP_SIDE_ESCAPE_SPEED_MPS:.3f}")
        self.speed_turn_var.set(f"{cfg.TURN_MAX_DPS:.1f}")
        self.speed_accel_var.set(f"{cfg.MOVE_FORWARD_ACCEL_LIMIT_MPS2:.2f}")
        self.speed_settle_var.set(f"{cfg.POST_MOVE_TILE_SETTLE_SEC:.3f}")
        self.speed_status_text.set("Editor restored to currently active motion settings.")

    def _load_speed_preset(self, name):
        if self.mission_started:
            self.speed_status_text.set("LOCKED: stop the mission before changing speed preset.")
            return False
        presets = {
            "SAFE": {"explore":0.26,"known":0.40,"emin":0.09,"kmin":0.11,"tfast":0.155,"tmed":0.105,"tslow":0.055,"ir":0.075,"sharp":0.080,"turn":65.0,"accel":0.65,"settle":0.11},
            "BALANCED": {"explore":0.31,"known":0.48,"emin":0.10,"kmin":0.13,"tfast":0.185,"tmed":0.120,"tslow":0.065,"ir":0.085,"sharp":0.090,"turn":75.0,"accel":0.80,"settle":0.08},
            "FAST": {"explore":0.36,"known":0.56,"emin":0.12,"kmin":0.15,"tfast":0.215,"tmed":0.140,"tslow":0.075,"ir":0.095,"sharp":0.100,"turn":85.0,"accel":1.00,"settle":0.06},
        }
        p = presets.get(str(name).upper(), presets["BALANCED"])
        self.speed_explore_var.set(str(p["explore"])); self.speed_known_var.set(str(p["known"]))
        self.speed_explore_min_var.set(str(p["emin"])); self.speed_known_min_var.set(str(p["kmin"]))
        self.speed_target_fast_var.set(str(p["tfast"])); self.speed_target_med_var.set(str(p["tmed"]))
        self.speed_target_slow_var.set(str(p["tslow"])); self.speed_ir_var.set(str(p["ir"]))
        self.speed_sharp_var.set(str(p["sharp"])); self.speed_turn_var.set(str(p["turn"]))
        self.speed_accel_var.set(str(p["accel"])); self.speed_settle_var.set(str(p["settle"]))
        self.speed_status_text.set("{} preset loaded; press APPLY SPEED SETTINGS.".format(name))
        return True

    def _apply_speed_from_gui(self, quiet=False):
        # Update the shared config module.
        # Update the shared config module.
        # Update the shared config module.
        # Update the shared config module.
        # Update the shared config module.
        # Update the shared config module.
        if self.mission_started:
            if not quiet:
                self.speed_status_text.set("LOCKED: STOP mission before applying new speeds.")
            return False
        try:
            ex=float(self.speed_explore_var.get()); kn=float(self.speed_known_var.get())
            emin=float(self.speed_explore_min_var.get()); kmin=float(self.speed_known_min_var.get())
            tf=float(self.speed_target_fast_var.get()); tm=float(self.speed_target_med_var.get())
            ts=float(self.speed_target_slow_var.get()); ir=float(self.speed_ir_var.get())
            sh=float(self.speed_sharp_var.get()); turn=float(self.speed_turn_var.get())
            accel=float(self.speed_accel_var.get()); settle=float(self.speed_settle_var.get())
            vals=[ex,kn,emin,kmin,tf,tm,ts,ir,sh,turn,accel,settle]
            if not all(math.isfinite(v) for v in vals): raise ValueError("all values must be finite numbers")
            if not 0.08 <= ex <= 0.60: raise ValueError("Explore cruise must be 0.08..0.60 m/s")
            if not 0.08 <= kn <= 0.75: raise ValueError("Known/Backtrack/Round2 must be 0.08..0.75 m/s")
            if not 0.04 <= emin <= ex: raise ValueError("Explore approach min must be 0.04..Explore cruise")
            if not 0.04 <= kmin <= kn: raise ValueError("Known approach min must be 0.04..Known cruise")
            if not 0.05 <= tf <= 0.30: raise ValueError("Target FAST must be 0.05..0.30 m/s")
            if not 0.04 <= tm <= tf: raise ValueError("Target MED must be 0.04..Target FAST")
            if not 0.025 <= ts <= tm: raise ValueError("Target SLOW must be 0.025..Target MED")
            if not 0.04 <= ir <= 0.16: raise ValueError("IR side base must be 0.04..0.16 m/s")
            if not 0.04 <= sh <= 0.18: raise ValueError("Sharp escape must be 0.04..0.18 m/s")
            if not 35.0 <= turn <= 110.0: raise ValueError("Turn max must be 35..110 deg/s")
            if not 0.30 <= accel <= 1.60: raise ValueError("Accel limit must be 0.30..1.60 m/s^2")
            if not 0.02 <= settle <= 0.25: raise ValueError("Post-cell settle must be 0.02..0.25 s")
            cfg.DFS_EXPLORE_SPEED_MPS=ex; cfg.DFS_KNOWN_SPEED_MPS=kn
            cfg.DFS_EXPLORE_APPROACH_MIN_MPS=emin; cfg.DFS_KNOWN_APPROACH_MIN_MPS=kmin
            cfg.TARGET_SIDE_SHIFT_SPEED_MPS=tf; cfg.TARGET_SIDE_SHIFT_MED_SPEED_MPS=tm; cfg.TARGET_SIDE_SHIFT_SLOW_MPS=ts
            cfg.IR_SIMPLE_STRAFE_SPEED_MPS=ir
            cfg.IR_SIMPLE_STRAFE_FAST_MPS=min(0.20, ir*(0.110/0.085))
            cfg.IR_SIMPLE_STRAFE_SLOW_MPS=max(0.035, ir*(0.060/0.085))
            cfg.SHARP_SIDE_ESCAPE_SPEED_MPS=sh; cfg.TURN_MAX_DPS=turn
            cfg.MOVE_FORWARD_ACCEL_LIMIT_MPS2=accel; cfg.POST_MOVE_TILE_SETTLE_SEC=settle
            cfg.MAX_CELL_TIME_SEC=max(6.0,(cfg.CELL_LENGTH_M/max(0.05,cfg.DFS_EXPLORE_SPEED_MPS))*2.8)
            self.explorer.pid_turn.out_limit=abs(float(cfg.TURN_MAX_DPS))
            self._sync_speed_vars_from_runtime()
            self.speed_status_text.set("APPLIED: Explore={:.2f} | Known/R2={:.2f} m/s | Turn={:.0f} deg/s | Accel={:.2f}".format(ex,kn,turn,accel))
            return True
        except Exception as exc:
            msg="SPEED INVALID: {}".format(exc)
            self.speed_status_text.set(msg)
            if not quiet: self.status_text.set(msg)
            return False

    def _round_changed(self):
        if not self.mission_started:
            mode = self.round_var.get()
            if mode == "ROUND1" and self.explorer.round1_memory is not None:
                self.explorer.reset_map_for_fresh_round1()
                self.memory_status_text.set(
                    "Round-2 snapshot cleared from RAM; Round 1 will start a fresh map"
                )
                self.route_preview_text.set("Route preview: not built")
            self.explorer.set_mission_mode(mode)
            if mode == "ROUND2" and self.explorer.round1_memory is not None:
                self._preview_round2_route(show_fault=False)

    def _format_preview_status(self, plan):
        if not plan:
            return "Route preview: unavailable"
        route = plan.get("route_cells") or []
        anchors = plan.get("anchor_order") or []
        return (
            "Route preview: {} target-hint(s), {} firing cell(s), {} step(s), ~{:.2f} m\n"
            "Anchor order: {}\nPath: {}"
        ).format(
            int(plan.get("selected_hint_count", 0)), int(plan.get("anchor_count", 0)),
            int(plan.get("total_steps", 0)), float(plan.get("total_distance_m", 0.0)),
            " -> ".join(str(tuple(c)) for c in anchors) if anchors else "(none)",
            " -> ".join(str(tuple(c)) for c in route) if route else "(none)",
        )

    def _load_round1_snapshot(self):
        if self.mission_started:
            self.status_text.set("Cannot change Round-2 snapshot while mission is running")
            return False
        path = None
        if filedialog is not None:
            try:
                path = filedialog.askopenfilename(
                    title="Load Round-1 map + firing snapshot",
                    initialdir=str(cfg.MAP_DIR.resolve()),
                    initialfile=cfg.ROUND1_ATTACK_MEMORY_JSON.name,
                    filetypes=(("Round-1 snapshot", "*.json"), ("All files", "*.*")),
                )
            except Exception:
                path = None
            if not path:
                return False
        else:
            path = str(cfg.ROUND1_ATTACK_MEMORY_JSON)

        if not self.explorer.load_round1_attack_memory(path):
            self.memory_status_text.set("LOAD FAILED: {}".format(path))
            self.route_preview_text.set("Route preview: unavailable")
            return False

        self.round_var.set("ROUND2")
        self.explorer.set_mission_mode("ROUND2")
        self._sync_arena_vars_from_runtime()
        self.arena_status_text.set("Arena restored from loaded Round-1 snapshot (Round 2 authoritative geometry)")
        saved_pairs = []
        for row in (self.explorer.round1_memory or {}).get("selected_target_classes", []):
            if isinstance(row, dict):
                key = (str(row.get("color") or "").upper(), str(row.get("shape") or "").upper())
                if key in self.target_vars:
                    saved_pairs.append(key)
        if saved_pairs:
            selected_set = set(saved_pairs)
            for key, var in self.target_vars.items():
                var.set(key in selected_set)
            self._target_filter_changed()
        self.memory_status_text.set(
            "Loaded: {} | cells={} | firing hints={} | breadcrumbs={} | map_complete={}".format(
                self.explorer.round1_memory_path, len(self.explorer.visited),
                len(self.explorer.round2_hints), len(self.explorer.breadcrumb_snapshot()),
                self.explorer.map_complete
            )
        )
        self._preview_round2_route(show_fault=True)
        return True

    def _load_default_and_preview_round2(self):
        """One-click competition path: load standard Round-1 bundle then preview."""
        if self.mission_started:
            self.status_text.set("Cannot load Round-2 snapshot while mission is running")
            return False
        path = Path(cfg.ROUND1_ATTACK_MEMORY_JSON)
        if not self.explorer.load_round1_attack_memory(path):
            self.memory_status_text.set("LOAD FAILED: {}".format(path))
            self.route_preview_text.set("Route preview: unavailable")
            return False
        self.round_var.set("ROUND2")
        self.explorer.set_mission_mode("ROUND2")
        self._sync_arena_vars_from_runtime()
        self.arena_status_text.set("Arena restored from loaded Round-1 snapshot (Round 2 authoritative geometry)")

        saved_pairs = []
        for row in (self.explorer.round1_memory or {}).get("selected_target_classes", []):
            if isinstance(row, dict):
                key = (str(row.get("color") or "").upper(), str(row.get("shape") or "").upper())
                if key in self.target_vars:
                    saved_pairs.append(key)
        if saved_pairs:
            selected_set = set(saved_pairs)
            for key, var in self.target_vars.items():
                var.set(key in selected_set)
            self._target_filter_changed()

        self.memory_status_text.set(
            "Loaded DEFAULT: {} | cells={} | firing hints={} | map_complete={}".format(
                path, len(self.explorer.visited), len(self.explorer.round2_hints),
                self.explorer.map_complete,
            )
        )
        return self._preview_round2_route(show_fault=True)

    def _preview_round2_route(self, show_fault=True):
        if self.mission_started:
            return False
        if self.explorer.round1_memory is None:
            default_path = Path(self.explorer.round1_memory_path)
            if not default_path.exists() or not self.explorer.load_round1_attack_memory(default_path):
                if show_fault:
                    self.route_preview_text.set(
                        "Route preview: load maps/round1_attack_memory.json first"
                    )
                return False
            self._sync_arena_vars_from_runtime()
            self.arena_status_text.set("Arena restored from auto-loaded Round-1 snapshot")
            self.memory_status_text.set(
                "Loaded default: {} | cells={} | firing hints={} | breadcrumbs={}".format(
                    default_path, len(self.explorer.visited), len(self.explorer.round2_hints),
                    len(self.explorer.breadcrumb_snapshot())
                )
            )
        try:
            plan = self.explorer.build_round2_preview_plan()
            self.route_preview_text.set(self._format_preview_status(plan))
            return True
        except Exception as exc:
            if show_fault:
                self.route_preview_text.set(
                    "Route preview failed: {}: {}".format(type(exc).__name__, exc)
                )
            return False

    def _fire_changed(self):
        self.explorer.set_fire_mode(self.fire_var.get())

    def _burst_changed(self, event=None):
        self.explorer.set_fire_burst_count(self.burst_var.get())

    def _target_filter_changed(self):
        selected = [key for key, var in self.target_vars.items() if bool(var.get())]
        self.explorer.set_target_selection(selected)
        self.target_filter_text.set("{} / 16 target classes enabled".format(len(selected)))
        if (
            not self.mission_started and self.round_var.get() == "ROUND2"
            and self.explorer.round1_memory is not None
        ):
            self._preview_round2_route(show_fault=False)

    def _select_all_targets(self):
        for var in self.target_vars.values():
            var.set(True)
        self._target_filter_changed()

    def _clear_all_targets(self):
        for var in self.target_vars.values():
            var.set(False)
        self._target_filter_changed()

    def _start_mission(self):
        if self.mission_started:
            return

        requested_mode = self.round_var.get()
        # START consumes the currently visible speed editor values too.
        if not self._apply_speed_from_gui(quiet=True):
            self.status_text.set("MISSION NOT STARTED - fix Speed / Motion Tuning values")
            self.notebook.select(self.speed_page)
            return
        if requested_mode == "ROUND1":
            # START always consumes the values currently visible in Arena / Field Setup,
            # so forgetting to press APPLY cannot silently run an old geometry.
            if not self._apply_arena_from_gui(quiet=True):
                self.status_text.set("ROUND1 NOT STARTED - fix Arena / Field Setup values")
                self.notebook.select(self.arena_page)
                return
            requested_mode = "ROUND1"
        if requested_mode == "ROUND1" and self.explorer.round1_memory is not None:
            self.explorer.reset_map_for_fresh_round1()
        if requested_mode == "ROUND2":
            # Round 2 must have a frozen Round-1 bundle before the robot connects.
            # If the user did not press LOAD, transparently try the standard file.
            if self.explorer.round1_memory is None:
                if not self.explorer.load_round1_attack_memory(self.explorer.round1_memory_path):
                    self.status_text.set(
                        "ROUND2 NOT STARTED - load maps/round1_attack_memory.json first"
                    )
                    return
                self.memory_status_text.set(
                    "Loaded default: {} | cells={} | firing hints={}".format(
                        self.explorer.round1_memory_path, len(self.explorer.visited),
                        len(self.explorer.round2_hints)
                    )
                )
            if not self._preview_round2_route(show_fault=True):
                self.status_text.set("ROUND2 NOT STARTED - route preview could not be built")
                return

        self.mission_started = True
        self.mission_finished_announced = False
        self.status_override = None
        self.explorer.running = True
        self.explorer.set_mission_mode(requested_mode)
        self.explorer.set_fire_mode(self.fire_var.get())
        self.explorer.set_fire_burst_count(self.burst_var.get())
        self._target_filter_changed()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        try:
            self.speed_apply_btn.configure(state="disabled")
        except Exception:
            pass
        self.status_text.set(
            "START requested - {}{}".format(
                requested_mode,
                " using previewed shortest route" if requested_mode == "ROUND2" else ""
            )
        )
        self.mission_thread = threading.Thread(
            target=self._mission_worker, name="RoboMasterMission", daemon=False
        )
        self.mission_thread.start()

    def _mission_worker(self):
        try:
            if self.explorer.connect():
                self.explorer.run_selected_mission()
        except Exception as exc:
            self.explorer.fault(
                "GUI MISSION",
                "{}: {}".format(type(exc).__name__, exc),
                "safe stop + save current state",
            )
            self.explorer.enter_safe_pause("GUI mission exception")
        finally:
            # cleanup() is also the authoritative STOP+SAVE path: latest map,
            # target memory and the combined Round-1 snapshot are flushed before
            # the SDK connection is closed.
            self.explorer.cleanup()

    def _mission_finished_ui(self):
        try:
            self.mission_finished_announced = True
            self.stop_btn.configure(state="disabled")
            self.start_btn.configure(state="disabled")
            if self.explorer.mission_mode == "ROUND1":
                self.memory_status_text.set(
                    "Saved Round-1 snapshot: {}".format(cfg.ROUND1_ATTACK_MEMORY_JSON)
                )
            self.status_override = (
                "MISSION ENDED - robot stopped and autosaved. "
                "For Round 2 load maps/round1_attack_memory.json, Preview, then START."
            )
            self.status_text.set(self.status_override)
        except Exception:
            pass

    def _stop_mission(self):
        if not self.mission_started:
            return
        # Kill autonomous motion first, then persist the Round-1 bundle immediately.
        # cleanup() in the worker will save a second time before closing the SDK.
        self.explorer.running = False
        self.stop_btn.configure(state="disabled")
        self.status_override = (
            "STOP requested - robot stopping; saving map + targets + Round-2 bundle NOW..."
        )
        self.status_text.set(self.status_override)
        try:
            ok = self.explorer.save_manual_stop_checkpoint(reason="GUI_STOP")
            if self.explorer.mission_mode == "ROUND1":
                self.memory_status_text.set(
                    "STOP snapshot {}: {} | load THIS file for Round 2".format(
                        "SAVED" if ok else "PARTIAL", cfg.ROUND1_ATTACK_MEMORY_JSON
                    )
                )
            self.status_override = (
                "STOPPED + CHECKPOINT SAVED; cleanup is finishing safely"
                if ok else
                "STOPPED; checkpoint was partial - cleanup will retry save"
            )
            self.status_text.set(self.status_override)
        except Exception as exc:
            self.explorer.fault(
                "GUI STOP SAVE", "{}: {}".format(type(exc).__name__, exc),
                "worker cleanup will retry save",
            )
            self.status_override = "STOPPED; cleanup will retry automatic save"
            self.status_text.set(self.status_override)

    def _on_close(self):
        self.closing = True
        self._stop_mission()
        self.root.after(250, self.root.destroy)

    @staticmethod
    def _heading_arrow(heading):
        return {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}.get(int(heading) % 4, (0, -1))

    def _snapshot_map(self):
        try:
            visited = set(self.explorer.visited)
            edge_state = dict(self.explorer.edge_state)
            current = tuple(self.explorer.current)
            root = tuple(self.explorer.root)
            heading = int(self.explorer.heading)
            targets = [dict(t) for t in self.explorer.target_system.targets]
            hints = [dict(h) for h in self.explorer.round2_hints]
            preview_plan = dict(self.explorer.round2_preview_plan or {})
            breadcrumbs = self.explorer.breadcrumb_snapshot()
            return visited, edge_state, current, root, heading, targets, hints, preview_plan, breadcrumbs
        except Exception:
            return set(), {}, (0, 0), (0, 0), 0, [], [], {}, []

    def _draw_map(self):
        self.canvas.delete("all")
        visited, edge_state, current, root, heading, targets, hints, preview_plan, breadcrumbs = self._snapshot_map()
        cells = set(visited) | {current, root}
        for (cell, d), state in edge_state.items():
            c = tuple(cell)
            cells.add(c)
            if state == "OPEN":
                dx, dy = cfg.DIR_VEC[int(d) % 4]
                cells.add((c[0] + dx, c[1] + dy))
        for h in hints:
            c = h.get("cell")
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                cells.add((int(c[0]), int(c[1])))
        for c in preview_plan.get("route_cells", []) or []:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                cells.add((int(c[0]), int(c[1])))
        for b in breadcrumbs:
            for key in ("from", "to"):
                c = b.get(key) if isinstance(b, dict) else None
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    cells.add((int(c[0]), int(c[1])))
        if not cells:
            return

        min_x = min(c[0] for c in cells); max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells); max_y = max(c[1] for c in cells)
        cw = max(320, self.canvas.winfo_width())
        ch = max(320, self.canvas.winfo_height())
        pad = 36.0
        nx = max(1, max_x - min_x + 1); ny = max(1, max_y - min_y + 1)
        cell_px = max(28.0, min((cw - 2*pad) / nx, (ch - 2*pad) / ny))

        def origin(cell):
            x, y = cell
            return (
                pad + (x - min_x) * cell_px,
                pad + (max_y - y) * cell_px,
            )

        for c in cells:
            x0, y0 = origin(c)
            if c in visited:
                self.canvas.create_rectangle(
                    x0+2, y0+2, x0+cell_px-2, y0+cell_px-2,
                    fill="#f2f2f2", outline="",
                )
            label = "S" if c == root else "{},{}".format(c[0], c[1])
            self.canvas.create_text(x0+cell_px/2, y0+cell_px/2, text=label, fill="#777777")

        for (cell, d), state in edge_state.items():
            c = tuple(cell); d = int(d) % 4
            if state == "OPEN":
                continue
            x0, y0 = origin(c)
            x1, y1, x2, y2 = x0, y0, x0, y0
            if d == 0: x1,y1,x2,y2 = x0,y0,x0+cell_px,y0
            elif d == 1: x1,y1,x2,y2 = x0+cell_px,y0,x0+cell_px,y0+cell_px
            elif d == 2: x1,y1,x2,y2 = x0,y0+cell_px,x0+cell_px,y0+cell_px
            else: x1,y1,x2,y2 = x0,y0,x0,y0+cell_px
            if state in ("WALL", "BLOCKED"):
                self.canvas.create_line(x1,y1,x2,y2, fill="black", width=4)
            else:
                self.canvas.create_line(x1,y1,x2,y2, fill="#999999", width=2, dash=(5,4))

        # Chronological BREADCRUMB trail: orange = where the chassis ACTUALLY
        # traveled.  Draw it before the purple Round-2 preview so the planned
        # shortest route remains easy to distinguish.
        valid_breadcrumbs = []
        for b in breadcrumbs[-160:]:
            if not isinstance(b, dict):
                continue
            fr = b.get("from"); to = b.get("to")
            if not (isinstance(fr, (list, tuple)) and len(fr) >= 2 and isinstance(to, (list, tuple)) and len(to) >= 2):
                continue
            fr = (int(fr[0]), int(fr[1])); to = (int(to[0]), int(to[1]))
            valid_breadcrumbs.append((b, fr, to))
            fx, fy = origin(fr); tx, ty = origin(to)
            self.canvas.create_line(
                fx + cell_px/2, fy + cell_px/2,
                tx + cell_px/2, ty + cell_px/2,
                fill="#d27b00", width=2, arrow=tk.LAST,
            )
        # Number the most recent breadcrumbs only; numbering every revisit makes
        # a dense maze unreadable.  The complete order stays in JSON.
        for b, _fr, to in valid_breadcrumbs[-20:]:
            tx, ty = origin(to)
            seq = int(b.get("seq", 0) or 0)
            ox = ((seq % 3) - 1) * 6
            oy = (((seq // 3) % 3) - 1) * 6
            px, py = tx + cell_px/2 + ox, ty + cell_px/2 + oy
            self.canvas.create_oval(px-6, py-6, px+6, py+6, fill="#fff3df", outline="#d27b00")
            self.canvas.create_text(px, py, text=str(seq), fill="#8a4b00", font=("TkDefaultFont", 7))

        # Previewed Round-2 route (same planner used by execution).
        preview_route = [tuple(c) for c in (preview_plan.get("route_cells") or []) if len(c) >= 2]
        if len(preview_route) >= 2:
            pts = []
            for c in preview_route:
                x0, y0 = origin(c)
                pts.extend((x0 + cell_px/2, y0 + cell_px/2))
            self.canvas.create_line(
                *pts, fill="#7a3db8", width=3, dash=(8, 4), arrow=tk.LAST
            )
        for idx, c in enumerate(preview_plan.get("anchor_order") or [], start=1):
            c = tuple(c)
            x0, y0 = origin(c)
            ax, ay = x0 + cell_px*0.22, y0 + cell_px*0.22
            self.canvas.create_oval(
                ax-9, ay-9, ax+9, ay+9, fill="#ffffff", outline="#7a3db8", width=2
            )
            self.canvas.create_text(ax, ay, text=str(idx), fill="#7a3db8")

        target_colors = {"RED":"#d62728", "GREEN":"#2ca02c", "BLUE":"#1f77b4", "YELLOW":"#c7a600"}
        for t in targets:
            pos = t.get("estimated_grid_xy")
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            try:
                gx, gy = float(pos[0]), float(pos[1])
            except Exception:
                continue
            px = pad + (gx - min_x + 0.5) * cell_px
            py = pad + (max_y - gy + 0.5) * cell_px
            color = target_colors.get(str(t.get("color") or "").upper(), "#555555")
            self.canvas.create_oval(px-6, py-6, px+6, py+6, fill=color, outline="black")
            self.canvas.create_text(px+10, py-9, text=str(t.get("id") or "T"), anchor="w", fill=color)

        # Proven Round-1 firing anchors are shown as H markers during Round 2.
        for h in hints:
            c = h.get("cell")
            if not isinstance(c, (list, tuple)) or len(c) < 2:
                continue
            cell = (int(c[0]), int(c[1]))
            x0, y0 = origin(cell)
            px, py = x0 + cell_px*0.78, y0 + cell_px*0.22
            color = target_colors.get(str(h.get("color") or "").upper(), "#7a3db8")
            self.canvas.create_rectangle(px-5, py-5, px+5, py+5, fill=color, outline="black")
            self.canvas.create_text(px-7, py, text="H", anchor="e", fill=color)

        x0, y0 = origin(current)
        cx, cy = x0 + cell_px/2, y0 + cell_px/2
        dx, dy = self._heading_arrow(heading)
        self.canvas.create_oval(cx-9, cy-9, cx+9, cy+9, outline="#0057b7", width=3)
        self.canvas.create_line(cx,cy,cx+dx*cell_px*0.32,cy+dy*cell_px*0.32, fill="#0057b7", width=4, arrow=tk.LAST)

    def _refresh(self):
        if self.closing:
            return
        try:
            if (
                self.mission_started and self.explorer.cleanup_done
                and not self.mission_finished_announced
            ):
                self._mission_finished_ui()
            self._draw_map()
            tof = self.explorer.latest_tof(fresh=False)
            gp, gy = self.explorer.current_gimbal_relative()
            current = tuple(self.explorer.current)
            heading = cfg.DIR_NAMES[int(self.explorer.heading) % 4]
            status = self.explorer.target_system.status
            fire_mode = self.explorer.get_fire_mode()
            burst_count = self.explorer.get_fire_burst_count()
            fire_event = self.explorer.target_system.last_fire_event
            selected_count = len(self.explorer.get_target_selection())
            breadcrumb = self.explorer.breadcrumb_snapshot()
            speed = self.explorer.speed_snapshot()
            last_breadcrumb = breadcrumb[-1] if breadcrumb else None
            if last_breadcrumb:
                breadcrumb_last_text = "#{} {}->{} {} ({})".format(
                    last_breadcrumb.get("seq"), tuple(last_breadcrumb.get("from", [])),
                    tuple(last_breadcrumb.get("to", [])), last_breadcrumb.get("dir"),
                    last_breadcrumb.get("profile"),
                )
            else:
                breadcrumb_last_text = "none"
            live_status = (
                "Mode={}  Cell={}  Heading={}\nVisited={}  Breadcrumb={}  PoseTrusted={}\n"
                "Breadcrumb last={}\n"
                "ToF={} mm  Gimbal P/Y={}/{}\n"
                "Speed actual={:.2f} m/s  cmd={:.2f} m/s  [{}]\n"
                "Target={}\nFire={} x{}  Selected={}/16\nLast={}".format(
                    self.explorer.mission_mode, current, heading,
                    len(self.explorer.visited), len(breadcrumb), self.explorer.pose_trusted,
                    breadcrumb_last_text,
                    "NA" if tof is None else "{:.0f}".format(tof),
                    "NA" if gp is None else "{:+.1f}".format(gp),
                    "NA" if gy is None else "{:+.1f}".format(gy),
                    float(speed.get("actual_speed",0.0)), float(speed.get("command_speed",0.0)), speed.get("profile","IDLE"),
                    status, fire_mode, burst_count, selected_count, fire_event,
                )
            )
            if self.status_override is None:
                self.status_text.set(live_status)
            self.geometry_text.set(
                "Center -> ToF forward      : {:.1f} cm\n"
                "Center -> muzzle forward   : {:.1f} cm\n"
                "ToF -> muzzle forward      : {:.1f} cm\n"
                "Camera -> muzzle vertical  : {:+.1f} cm\n"
                "ToF -> muzzle vertical     : {:+.1f} cm\n"
                "Aim policy                  : CENTER -> physical muzzle LOS\n"
                "Arena                       : {}x{} start={} guard={}\n"
                "DFS explore speed           : {:.2f} m/s\n"
                "DFS known/backtrack speed   : {:.2f} m/s\n"
                "Round1 target time          : {:.0f} s\n"
                "Fire RAW ToF gate           : <= {:.0f} mm\n"
                "Round2 narrow replay        : +/- {:.0f} deg\n"
                "Round2 deadline guard       : {:.0f} s".format(
                    cfg.FIRE_TOF_FORWARD_FROM_CENTER_M*100.0,
                    cfg.FIRE_MUZZLE_FORWARD_FROM_CENTER_M*100.0,
                    cfg.FIRE_MUZZLE_AHEAD_OF_TOF_M*100.0,
                    cfg.FIRE_CAMERA_ABOVE_MUZZLE_M*100.0,
                    cfg.FIRE_TOF_ABOVE_MUZZLE_M*100.0,
                    cfg.GRID_WIDTH_CELLS, cfg.GRID_HEIGHT_CELLS, cfg.ROOT_CELL,
                    "ON" if cfg.FIELD_BOUNDARY_GUARD_ENABLED else "OFF",
                    cfg.DFS_EXPLORE_SPEED_MPS,
                    cfg.DFS_KNOWN_SPEED_MPS,
                    cfg.MAX_MISSION_SEC,
                    cfg.TARGET_FIRE_MAX_RANGE_MM,
                    cfg.ROUND2_NARROW_SWEEP_HALF_DEG,
                    cfg.ROUND2_HARD_LIMIT_SEC,
                )
            )
            self.speed_live_text.set(
                "ACTIVE PROFILE : {}\nCOMMAND LABEL  : {}\n\n"
                "Command Forward X : {:+.3f} m/s\nCommand Strafe Y : {:+.3f} m/s\n"
                "Command Total     : {:.3f} m/s\nCommand Yaw       : {:+.1f} deg/s\n\n"
                "Measured Forward  : {:+.3f} m/s\nMeasured Right    : {:+.3f} m/s\n"
                "Measured Total    : {:.3f} m/s\nOdom World X/Y    : {:+.3f} / {:+.3f} m/s\n\n"
                "Configured Explore: {:.3f} m/s\nConfigured Known/R2: {:.3f} m/s\n"
                "Target Shift F/M/S: {:.3f} / {:.3f} / {:.3f} m/s\n"
                "IR / Sharp side   : {:.3f} / {:.3f} m/s\nTurn max          : {:.1f} deg/s".format(
                    speed.get("profile","IDLE"), speed.get("command_label","-"),
                    float(speed.get("command_x",0.0)), float(speed.get("command_y",0.0)),
                    float(speed.get("command_speed",0.0)), float(speed.get("command_z",0.0)),
                    float(speed.get("actual_forward",0.0)), float(speed.get("actual_right",0.0)),
                    float(speed.get("actual_speed",0.0)), float(speed.get("actual_vx",0.0)), float(speed.get("actual_vy",0.0)),
                    cfg.DFS_EXPLORE_SPEED_MPS, cfg.DFS_KNOWN_SPEED_MPS, cfg.TARGET_SIDE_SHIFT_SPEED_MPS,
                    cfg.TARGET_SIDE_SHIFT_MED_SPEED_MPS, cfg.TARGET_SIDE_SHIFT_SLOW_MPS, cfg.IR_SIMPLE_STRAFE_SPEED_MPS,
                    cfg.SHARP_SIDE_ESCAPE_SPEED_MPS, cfg.TURN_MAX_DPS
                )
            )

            s = dict(self.explorer.target_system.last_aim_solution)
            if s:
                self.aim_text.set(
                    "Mode={}  ToF(raw)={:.0f} mm\n"
                    "Center range={:.0f} mm  Muzzle range={:.0f} mm\n"
                    "Camera P={:+.2f} deg\n"
                    "Muzzle correction={:+.2f} deg\n"
                    "Fire P={:+.2f} deg\n"
                    "ToF-Camera beam offset={:+.2f} deg".format(
                        s.get("fire_mode", "?"), float(s.get("tof_range_mm", 0.0)),
                        float(s.get("robot_center_to_target_planar_mm", 0.0)),
                        float(s.get("muzzle_to_target_planar_mm", 0.0)),
                        float(s.get("camera_lock_pitch_deg", 0.0)),
                        float(s.get("camera_muzzle_parallax_pitch_deg", 0.0)),
                        float(s.get("fire_pitch_deg", 0.0)),
                        float(s.get("tof_camera_parallax_deg", 0.0)),
                    )
                )
        except Exception:
            pass
        self.root.after(cfg.CONTROL_GUI_REFRESH_MS, self._refresh)

    def run(self):
        self.root.mainloop()
        if self.mission_thread is not None and self.mission_thread.is_alive():
            self.explorer.running = False
            try:
                self.explorer.safe_stop()
            except Exception:
                pass
            self.mission_thread.join(timeout=3.0)
        if not self.explorer.cleanup_done:
            self.explorer.cleanup()
