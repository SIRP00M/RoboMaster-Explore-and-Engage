#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thread-isolated Tkinter Mission Control GUI.
Displays live maze map, telemetry, target statuses, and operator decision dialogs.
"""

import math
import queue
import threading

from src.core.geometry import DIR_NAMES, DIR_VEC
from config.blaster import (
    TARGET_FIRE_COLORS,
    TARGET_FIRE_MAX_RANGE_MM,
    TARGET_FIRE_SHAPES,
)

class MissionControlGUI:
    """
    Thread-isolated Tkinter mission-control window.

    The RoboMaster/DFS code remains on the mission thread.  Tkinter owns its
    own GUI thread and receives immutable snapshots through a Queue, so the UI
    never iterates live DFS dictionaries while they are being modified.

    During normal exploration the window is read-only.  After the robot has
    returned to START and one or more EXIT_CANDIDATE edges exist, the operator
    can select a specific candidate, preview its shortest confirmed route,
    then explicitly continue through that edge or finish the mission.
    """

    POLL_MS = 100

    def __init__(self, explorer):
        self.explorer = explorer
        self.messages = queue.Queue(maxsize=80)
        self.ready_event = threading.Event()
        self.closed_event = threading.Event()
        self.decision_event = threading.Event()
        self.target_policy_event = threading.Event()

        self.available = False
        self.thread = None
        self.start_error = None

        self.decision = None
        self.selected_candidate_key = None
        self.target_policy = None

        # GUI-thread-only fields are initialized in _run().
        self.root = None
        self.canvas = None
        self.status_var = None
        self.mode_var = None
        self.telemetry_var = None
        self.target_runtime_var = None
        self.fire_runtime_var = None
        self.fire_policy_summary_var = None
        self.target_vars = {}
        self.sdk_enabled_var = None
        self.sdk_labels_var = None
        self.auto_fire_var = None
        self.notebook = None
        self.targets_tab = None
        self.candidate_list = None
        self.candidate_detail_var = None
        self.continue_button = None
        self.finish_button = None
        self.estop_button = None

        self.latest_snapshot = None
        self.candidate_options = []
        self.candidate_by_index = []
        self.selected_option = None
        self.marker_hits = []
        self.mission_complete = False

    # --------------------------------------------------------
    # Public / mission-thread API
    # --------------------------------------------------------

    def start(self, timeout=3.0):
        if self.thread is not None:
            return self.available

        self.thread = threading.Thread(
            target=self._run,
            name='RoboMasterMissionControlGUI',
            daemon=True,
        )
        self.thread.start()
        self.ready_event.wait(timeout=max(0.2, float(timeout)))

        if not self.available:
            if self.start_error:
                print(f'[GUI WARN] Mission Control unavailable: {self.start_error}')
            else:
                print('[GUI WARN] Mission Control did not become ready; using terminal fallback.')

        return self.available

    def _post(self, kind, payload=None):
        if not self.available and kind != 'close':
            return False

        item = (kind, payload)
        try:
            self.messages.put_nowait(item)
            return True
        except queue.Full:
            # Keep the newest state.  Dropping an old visual snapshot is fine;
            # decision messages are retried after freeing one slot.
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(item)
                return True
            except queue.Full:
                return False

    def post_snapshot(self, snapshot):
        return self._post('snapshot', snapshot)

    def post_mission_complete(self, snapshot=None):
        if snapshot is not None:
            self.post_snapshot(snapshot)
        return self._post('mission_complete', None)

    def request_exit_decision(self, options, snapshot=None):
        """
        Block the mission thread while the GUI stays responsive.

        Returns:
            ('continue', ((x, y), dir_index))
            ('finish', None)
            None  -> GUI unavailable/closed, caller should use terminal fallback
        """
        if not self.available or self.closed_event.is_set():
            return None

        self.decision = None
        self.selected_candidate_key = None
        self.decision_event.clear()

        payload = {
            'options': options,
            'snapshot': snapshot,
        }
        if not self._post('exit_decision', payload):
            return None

        while self.explorer.running and self.available:
            if self.decision_event.wait(0.10):
                break

        if not self.decision_event.is_set():
            return None

        return self.decision, self.selected_candidate_key

    def close(self):
        self._post('close', None)

    def wait_closed(self, timeout=None):
        return self.closed_event.wait(timeout=timeout)

    def request_target_fire_policy(self):
        """Wait until the operator explicitly arms a target-fire policy.

        The GUI is already visible before the robot moves.  This method blocks
        only the mission thread; Tk remains responsive on its own thread.
        """
        if not self.available or self.closed_event.is_set():
            return None

        while self.explorer.running and self.available:
            if self.target_policy_event.wait(0.10):
                break

        if not self.target_policy_event.is_set():
            return None
        return dict(self.target_policy or {})

    # --------------------------------------------------------
    # GUI thread
    # --------------------------------------------------------

    def _run(self):
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox

            self.tk = tk
            self.ttk = ttk
            self.messagebox = messagebox

            root = tk.Tk()
            self.root = root
            root.title('RoboMaster Maze Mission Control')
            root.geometry('1240x800')
            root.minsize(980, 660)
            root.configure(bg='#0d1117')

            style = ttk.Style(root)
            try:
                style.theme_use('clam')
            except Exception:
                pass

            style.configure('MC.TFrame', background='#0d1117')
            style.configure('Panel.TFrame', background='#161b22')
            style.configure(
                'MC.TLabel', background='#0d1117', foreground='#e6edf3',
                font=('Segoe UI', 10),
            )
            style.configure(
                'Title.TLabel', background='#0d1117', foreground='#f0f6fc',
                font=('Segoe UI Semibold', 16),
            )
            style.configure(
                'Status.TLabel', background='#161b22', foreground='#58a6ff',
                font=('Segoe UI Semibold', 11), padding=(10, 8),
            )
            style.configure(
                'Panel.TLabel', background='#161b22', foreground='#c9d1d9',
                font=('Segoe UI', 10),
            )
            style.configure(
                'Section.TLabel', background='#161b22', foreground='#f0f6fc',
                font=('Segoe UI Semibold', 11),
            )
            style.configure(
                'Hint.TLabel', background='#161b22', foreground='#8b949e',
                font=('Segoe UI', 9),
            )
            style.configure('Accent.TButton', font=('Segoe UI Semibold', 10), padding=(10, 8))
            style.configure('Danger.TButton', font=('Segoe UI Semibold', 10), padding=(10, 8))
            style.configure('Fire.TButton', font=('Segoe UI Semibold', 10), padding=(10, 8))
            style.configure('MC.TCheckbutton', background='#161b22', foreground='#e6edf3')
            style.map('MC.TCheckbutton', background=[('active', '#161b22')])
            style.configure('MC.TNotebook', background='#0d1117', borderwidth=0)
            style.configure('MC.TNotebook.Tab', padding=(14, 7), font=('Segoe UI Semibold', 10))

            root.columnconfigure(0, weight=1)
            root.rowconfigure(2, weight=1)

            header = ttk.Frame(root, style='MC.TFrame', padding=(16, 12, 16, 8))
            header.grid(row=0, column=0, sticky='ew')
            header.columnconfigure(1, weight=1)

            ttk.Label(
                header,
                text='RoboMaster Maze Mission Control',
                style='Title.TLabel',
            ).grid(row=0, column=0, sticky='w')

            self.mode_var = tk.StringVar(value='GUI READY - TARGETS NOT ARMED')
            ttk.Label(
                header,
                textvariable=self.mode_var,
                style='MC.TLabel',
                anchor='e',
            ).grid(row=0, column=1, sticky='e')

            self.status_var = tk.StringVar(
                value='Select the target types allowed to fire, then ARM before motion.'
            )
            ttk.Label(
                root,
                textvariable=self.status_var,
                style='Status.TLabel',
                anchor='w',
            ).grid(row=1, column=0, sticky='ew', padx=16, pady=(0, 8))

            self.notebook = ttk.Notebook(root, style='MC.TNotebook')
            self.notebook.grid(row=2, column=0, sticky='nsew', padx=16, pady=(0, 10))

            mission_tab = ttk.Frame(self.notebook, style='MC.TFrame')
            self.targets_tab = ttk.Frame(self.notebook, style='MC.TFrame')
            status_tab = ttk.Frame(self.notebook, style='MC.TFrame')
            self.notebook.add(mission_tab, text='MISSION / MAP')
            self.notebook.add(self.targets_tab, text='TARGET FIRE RULES')
            self.notebook.add(status_tab, text='LIVE STATUS')

            # ==============================================================
            # TAB 1: Mission / map
            # ==============================================================
            mission_tab.columnconfigure(0, weight=1)
            mission_tab.columnconfigure(1, weight=0)
            mission_tab.rowconfigure(0, weight=1)

            map_panel = ttk.Frame(mission_tab, style='Panel.TFrame', padding=8)
            map_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
            map_panel.columnconfigure(0, weight=1)
            map_panel.rowconfigure(1, weight=1)

            ttk.Label(
                map_panel, text='Live DFS Map', style='Section.TLabel'
            ).grid(row=0, column=0, sticky='w', padx=4, pady=(2, 8))

            self.canvas = tk.Canvas(
                map_panel,
                bg='#0b0f14',
                highlightthickness=1,
                highlightbackground='#30363d',
                bd=0,
            )
            self.canvas.grid(row=1, column=0, sticky='nsew')
            self.canvas.bind('<Configure>', lambda _e: self._draw_map())
            self.canvas.bind('<Button-1>', self._on_map_click)

            side = ttk.Frame(mission_tab, style='Panel.TFrame', padding=12, width=350)
            side.grid(row=0, column=1, sticky='ns')
            side.grid_propagate(False)
            side.columnconfigure(0, weight=1)

            ttk.Label(side, text='Exit Candidates', style='Section.TLabel').grid(
                row=0, column=0, sticky='w'
            )
            ttk.Label(
                side,
                text=(
                    'Available after the robot returns to START. Select an Exit '
                    'to preview the confirmed shortest route.'
                ),
                style='Panel.TLabel', wraplength=320, justify='left',
            ).grid(row=1, column=0, sticky='ew', pady=(4, 8))

            self.candidate_list = tk.Listbox(
                side, height=9, exportselection=False,
                bg='#0d1117', fg='#e6edf3',
                selectbackground='#1f6feb', selectforeground='white',
                highlightthickness=1, highlightbackground='#30363d',
                relief='flat', font=('Consolas', 10),
            )
            self.candidate_list.grid(row=2, column=0, sticky='ew')
            self.candidate_list.bind('<<ListboxSelect>>', self._on_list_select)

            self.candidate_detail_var = tk.StringVar(value='No Exit selection is active.')
            ttk.Label(
                side, textvariable=self.candidate_detail_var,
                style='Panel.TLabel', wraplength=320, justify='left',
            ).grid(row=3, column=0, sticky='ew', pady=(8, 8))

            self.continue_button = ttk.Button(
                side, text='CONTINUE VIA SELECTED EXIT', style='Accent.TButton',
                command=self._continue_selected, state='disabled',
            )
            self.continue_button.grid(row=4, column=0, sticky='ew', pady=(0, 6))

            self.finish_button = ttk.Button(
                side, text='FINISH MISSION AT START',
                command=self._finish_selected, state='disabled',
            )
            self.finish_button.grid(row=5, column=0, sticky='ew', pady=(0, 10))

            ttk.Separator(side, orient='horizontal').grid(
                row=6, column=0, sticky='ew', pady=(0, 10)
            )

            ttk.Label(side, text='Armed Fire Policy', style='Section.TLabel').grid(
                row=7, column=0, sticky='w'
            )
            self.fire_policy_summary_var = tk.StringVar(
                value='DISARMED\nNo target may fire until TARGET FIRE RULES is armed.'
            )
            ttk.Label(
                side, textvariable=self.fire_policy_summary_var,
                style='Panel.TLabel', wraplength=320, justify='left',
            ).grid(row=8, column=0, sticky='ew', pady=(4, 12))

            self.estop_button = ttk.Button(
                side, text='EMERGENCY STOP', style='Danger.TButton',
                command=self._emergency_stop,
            )
            self.estop_button.grid(row=9, column=0, sticky='ew')

            # ==============================================================
            # TAB 2: Target fire rules
            # ==============================================================
            self.targets_tab.columnconfigure(0, weight=1)
            target_outer = ttk.Frame(self.targets_tab, style='Panel.TFrame', padding=18)
            target_outer.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
            target_outer.columnconfigure(0, weight=1)

            ttk.Label(
                target_outer,
                text='Select targets that are ALLOWED to fire',
                style='Section.TLabel',
            ).grid(row=0, column=0, sticky='w')
            ttk.Label(
                target_outer,
                text=(
                    'Detection and mapping still record every confirmed target. '
                    'Only checked identities are allowed to trigger the blaster. '
                    'Firing mode is locked to INFRARED; no water shots are used. '
                    f'Final fire safety requires fresh ToF <= {TARGET_FIRE_MAX_RANGE_MM/10.0:.0f} cm '
                    '(2 tiles x 60 cm). Farther targets stay saved and are not fired.'
                ),
                style='Panel.TLabel', wraplength=900, justify='left',
            ).grid(row=1, column=0, sticky='ew', pady=(4, 12))

            matrix = ttk.Frame(target_outer, style='Panel.TFrame')
            matrix.grid(row=2, column=0, sticky='w')

            shape_titles = {
                'SQUARE': 'SQUARE',
                'RECT_VERTICAL': 'RECT V',
                'RECT_HORIZONTAL': 'RECT H',
                'CIRCLE': 'CIRCLE',
            }
            ttk.Label(matrix, text='COLOR', style='Section.TLabel').grid(
                row=0, column=0, sticky='w', padx=(0, 16), pady=(0, 6)
            )
            for col, shape in enumerate(TARGET_FIRE_SHAPES, start=1):
                ttk.Label(
                    matrix, text=shape_titles[shape], style='Section.TLabel'
                ).grid(row=0, column=col, padx=12, pady=(0, 6))

            for row, color in enumerate(TARGET_FIRE_COLORS, start=1):
                ttk.Label(matrix, text=color, style='Panel.TLabel').grid(
                    row=row, column=0, sticky='w', padx=(0, 16), pady=5
                )
                for col, shape in enumerate(TARGET_FIRE_SHAPES, start=1):
                    var = tk.BooleanVar(value=False)
                    self.target_vars[(color, shape)] = var
                    ttk.Checkbutton(
                        matrix,
                        variable=var,
                        style='MC.TCheckbutton',
                    ).grid(row=row, column=col, padx=20, pady=5)

            ttk.Separator(target_outer, orient='horizontal').grid(
                row=3, column=0, sticky='ew', pady=14
            )

            sdk_box = ttk.Frame(target_outer, style='Panel.TFrame')
            sdk_box.grid(row=4, column=0, sticky='ew')
            sdk_box.columnconfigure(1, weight=1)

            self.sdk_enabled_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                sdk_box,
                text='Allow RoboMaster SDK markers (red number / symbol markers)',
                variable=self.sdk_enabled_var,
                style='MC.TCheckbutton',
            ).grid(row=0, column=0, columnspan=2, sticky='w')

            ttk.Label(
                sdk_box,
                text='Optional SDK labels:',
                style='Panel.TLabel',
            ).grid(row=1, column=0, sticky='w', pady=(8, 0), padx=(24, 8))
            self.sdk_labels_var = tk.StringVar(value='')
            ttk.Entry(sdk_box, textvariable=self.sdk_labels_var).grid(
                row=1, column=1, sticky='ew', pady=(8, 0)
            )
            ttk.Label(
                sdk_box,
                text='Comma-separated. Leave blank = every SDK marker when enabled.',
                style='Hint.TLabel',
            ).grid(row=2, column=1, sticky='w', pady=(2, 0))

            self.auto_fire_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                target_outer,
                text='Auto-fire selected targets immediately after stable LOCK + center verification',
                variable=self.auto_fire_var,
                style='MC.TCheckbutton',
            ).grid(row=5, column=0, sticky='w', pady=(14, 4))

            ttk.Label(
                target_outer,
                text=(
                    'Safety rule: raw SEARCH / VERIFY detections never fire. '
                    'The target must first pass the existing temporal confirmation and final center gate.'
                ),
                style='Hint.TLabel', wraplength=900, justify='left',
            ).grid(row=6, column=0, sticky='w', pady=(0, 14))

            actions = ttk.Frame(target_outer, style='Panel.TFrame')
            actions.grid(row=7, column=0, sticky='ew')
            actions.columnconfigure(0, weight=1)
            actions.columnconfigure(1, weight=1)
            actions.columnconfigure(2, weight=1)
            actions.columnconfigure(3, weight=1)

            ttk.Button(
                actions, text='SELECT ALL', command=self._select_all_target_boxes
            ).grid(row=0, column=0, sticky='ew', padx=(0, 5))
            ttk.Button(
                actions, text='CLEAR ALL', command=self._clear_all_target_boxes
            ).grid(row=0, column=1, sticky='ew', padx=5)
            ttk.Button(
                actions, text='ARM SELECTED TARGETS', style='Accent.TButton',
                command=self._arm_selected_targets,
            ).grid(row=0, column=2, sticky='ew', padx=5)
            ttk.Button(
                actions, text='ARM FIRE ALL TARGETS', style='Fire.TButton',
                command=self._arm_all_targets,
            ).grid(row=0, column=3, sticky='ew', padx=(5, 0))

            self.target_policy_detail_var = tk.StringVar(
                value='DISARMED - mission will wait here before movement.'
            )
            ttk.Label(
                target_outer,
                textvariable=self.target_policy_detail_var,
                style='Status.TLabel', anchor='w',
            ).grid(row=8, column=0, sticky='ew', pady=(16, 0))

            # ==============================================================
            # TAB 3: Live status
            # ==============================================================
            status_tab.columnconfigure(0, weight=1)
            status_tab.rowconfigure(0, weight=1)
            status_panel = ttk.Frame(status_tab, style='Panel.TFrame', padding=18)
            status_panel.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
            status_panel.columnconfigure(0, weight=1)

            ttk.Label(status_panel, text='Robot / DFS', style='Section.TLabel').grid(
                row=0, column=0, sticky='w'
            )
            self.telemetry_var = tk.StringVar(value='No telemetry yet.')
            ttk.Label(
                status_panel, textvariable=self.telemetry_var,
                style='Panel.TLabel', wraplength=950, justify='left',
            ).grid(row=1, column=0, sticky='ew', pady=(4, 14))

            ttk.Separator(status_panel, orient='horizontal').grid(
                row=2, column=0, sticky='ew', pady=(0, 12)
            )
            ttk.Label(status_panel, text='Target Vision', style='Section.TLabel').grid(
                row=3, column=0, sticky='w'
            )
            self.target_runtime_var = tk.StringVar(value='Vision status unavailable.')
            ttk.Label(
                status_panel, textvariable=self.target_runtime_var,
                style='Panel.TLabel', wraplength=950, justify='left',
            ).grid(row=4, column=0, sticky='ew', pady=(4, 14))

            ttk.Separator(status_panel, orient='horizontal').grid(
                row=5, column=0, sticky='ew', pady=(0, 12)
            )
            ttk.Label(status_panel, text='Infrared Fire', style='Section.TLabel').grid(
                row=6, column=0, sticky='w'
            )
            self.fire_runtime_var = tk.StringVar(value='DISARMED')
            ttk.Label(
                status_panel, textvariable=self.fire_runtime_var,
                style='Panel.TLabel', wraplength=950, justify='left',
            ).grid(row=7, column=0, sticky='ew', pady=(4, 0))

            footer = ttk.Label(
                root,
                text=(
                    'Purple E# = EXIT_CANDIDATE   •   T# = locked target   •   '
                    'T#F = target already fired   •   Orange ? = pending glimpse   •   '
                    'Yellow triangle = robot'
                ),
                style='MC.TLabel', anchor='w',
            )
            footer.grid(row=3, column=0, sticky='ew', padx=16, pady=(0, 10))

            root.protocol('WM_DELETE_WINDOW', self._on_close)

            # The target rules page is intentionally the first page the operator
            # sees; mission motion will wait for an explicit ARM action.
            self.notebook.select(self.targets_tab)

            self.available = True
            self.ready_event.set()
            root.after(self.POLL_MS, self._process_messages)
            root.mainloop()

        except Exception as e:
            self.start_error = e
            self.available = False
            self.ready_event.set()
        finally:
            self.available = False
            if not self.decision_event.is_set():
                self.decision = 'finish'
                self.selected_candidate_key = None
                self.decision_event.set()
            if not self.target_policy_event.is_set():
                self.target_policy = None
                self.target_policy_event.set()

            try:
                self.canvas = None
                self.status_var = None
                self.mode_var = None
                self.telemetry_var = None
                self.target_runtime_var = None
                self.fire_runtime_var = None
                self.fire_policy_summary_var = None
                self.target_vars = {}
                self.sdk_enabled_var = None
                self.sdk_labels_var = None
                self.auto_fire_var = None
                self.notebook = None
                self.targets_tab = None
                self.candidate_list = None
                self.candidate_detail_var = None
                self.continue_button = None
                self.finish_button = None
                self.estop_button = None
                self.root = None
                import gc
                gc.collect()
            except Exception:
                pass

            self.closed_event.set()

    def _select_all_target_boxes(self):
        for var in self.target_vars.values():
            var.set(True)
        if self.sdk_enabled_var is not None:
            self.sdk_enabled_var.set(True)

    def _clear_all_target_boxes(self):
        for var in self.target_vars.values():
            var.set(False)
        if self.sdk_enabled_var is not None:
            self.sdk_enabled_var.set(False)
        if self.sdk_labels_var is not None:
            self.sdk_labels_var.set('')

    def _build_selected_target_policy(self, mode='selected'):
        selected = []
        for (color, shape), var in self.target_vars.items():
            try:
                checked = bool(var.get())
            except Exception:
                checked = False
            if checked:
                selected.append([color, shape])

        labels_raw = ''
        if self.sdk_labels_var is not None:
            try:
                labels_raw = str(self.sdk_labels_var.get())
            except Exception:
                labels_raw = ''
        sdk_labels = [
            item.strip() for item in labels_raw.split(',') if item.strip()
        ]

        sdk_enabled = False
        if self.sdk_enabled_var is not None:
            try:
                sdk_enabled = bool(self.sdk_enabled_var.get())
            except Exception:
                pass

        auto_fire = True
        if self.auto_fire_var is not None:
            try:
                auto_fire = bool(self.auto_fire_var.get())
            except Exception:
                pass

        return {
            'armed': True,
            'mode': str(mode),
            'fire_type': 'infrared',
            'auto_fire': bool(auto_fire),
            'selected_color_shapes': selected,
            'sdk_enabled': bool(sdk_enabled),
            'sdk_labels': sdk_labels,
        }

    @staticmethod
    def _policy_summary_text(policy):
        if not policy or not policy.get('armed'):
            return 'DISARMED'
        range_cm = float(policy.get('max_range_mm', TARGET_FIRE_MAX_RANGE_MM)) / 10.0
        if policy.get('mode') == 'all':
            return (
                'ARMED: ALL TARGETS\n'
                f'INFRARED only / one shot per confirmed target / range <= {range_cm:.0f} cm'
            )

        selected = list(policy.get('selected_color_shapes') or [])
        sdk_enabled = bool(policy.get('sdk_enabled'))
        sdk_labels = list(policy.get('sdk_labels') or [])
        parts = []
        if selected:
            parts.append(f'{len(selected)} color/shape identities')
        if sdk_enabled:
            parts.append('SDK=' + (','.join(sdk_labels) if sdk_labels else 'ALL'))
        if not parts:
            parts.append('NO TARGETS (safe no-fire policy)')
        return (
            'ARMED: ' + ' + '.join(parts)
            + '\nINFRARED only / auto-fire=' + str(bool(policy.get('auto_fire', True)))
            + f' / range <= {range_cm:.0f} cm'
        )

    def _apply_armed_policy(self, policy):
        self.target_policy = dict(policy)
        # Thread-safe explorer setter.  This also makes re-arming during a run
        # immediately update the gate rather than only changing GUI state.
        try:
            self.explorer.set_target_fire_policy(policy)
        except Exception as e:
            self.messagebox.showerror(
                'Fire policy error', f'Could not apply policy: {e}', parent=self.root
            )
            return

        summary = self._policy_summary_text(policy)
        if self.fire_policy_summary_var is not None:
            self.fire_policy_summary_var.set(summary)
        if hasattr(self, 'target_policy_detail_var') and self.target_policy_detail_var is not None:
            self.target_policy_detail_var.set(summary)
        self.mode_var.set('TARGET FIRE POLICY ARMED')
        self.status_var.set('Target fire policy armed. Mission may start / continue.')
        self.target_policy_event.set()

    def _arm_selected_targets(self):
        policy = self._build_selected_target_policy(mode='selected')
        selected_n = len(policy.get('selected_color_shapes') or [])
        sdk_on = bool(policy.get('sdk_enabled'))
        if selected_n == 0 and not sdk_on:
            ok = self.messagebox.askyesno(
                'Arm no-fire policy',
                'No targets are selected. Arm the mission with ALL FIRING BLOCKED?',
                parent=self.root,
            )
            if not ok:
                return
        self._apply_armed_policy(policy)

    def _arm_all_targets(self):
        ok = self.messagebox.askyesno(
            'Arm FIRE ALL targets',
            'Allow every confirmed color/shape target and every SDK marker to fire?\n\n'
            'This does NOT fire now. It arms the automatic INFRARED firing gate.',
            parent=self.root,
        )
        if not ok:
            return
        self._select_all_target_boxes()
        policy = self._build_selected_target_policy(mode='all')
        policy['sdk_enabled'] = True
        policy['sdk_labels'] = []
        policy['auto_fire'] = True
        if self.auto_fire_var is not None:
            self.auto_fire_var.set(True)
        self._apply_armed_policy(policy)

    def _process_messages(self):
        if not self.available or self.root is None:
            return

        processed = 0
        while processed < 30:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break

            processed += 1

            if kind == 'snapshot':
                self._apply_snapshot(payload)

            elif kind == 'exit_decision':
                snapshot = payload.get('snapshot') if isinstance(payload, dict) else None
                if snapshot is not None:
                    self._apply_snapshot(snapshot)
                options = payload.get('options', []) if isinstance(payload, dict) else []
                self._enter_exit_decision(options)

            elif kind == 'mission_complete':
                self.mission_complete = True
                self.mode_var.set('MISSION COMPLETE')
                self.status_var.set('Mission finished. Map remains available for inspection.')
                self.continue_button.configure(state='disabled')
                self.finish_button.configure(state='disabled')

            elif kind == 'close':
                try:
                    self.root.destroy()
                except Exception:
                    pass
                return

        if self.available and self.root is not None:
            self.root.after(self.POLL_MS, self._process_messages)

    def _apply_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return

        self.latest_snapshot = snapshot
        status = snapshot.get('status') or 'RUNNING'
        self.status_var.set(status)

        current = tuple(snapshot.get('current', (0, 0)))
        heading_idx = int(snapshot.get('heading', 0)) % 4
        heading = DIR_NAMES[heading_idx]
        visited_count = len(snapshot.get('visited', []))
        exit_count = len(snapshot.get('exit_candidates', []))
        target_count = len(snapshot.get('targets', []))
        fired_count = sum(
            1 for t in snapshot.get('targets', [])
            if t.get('fire_status') == 'FIRED_IR'
        )
        too_far_count = sum(
            1 for t in snapshot.get('targets', [])
            if t.get('fire_status') == 'WAITING_TOO_FAR'
        )
        glimpse_count = sum(
            1 for g in snapshot.get('target_glimpses', [])
            if not g.get('resolved_target_id')
        )
        pos = snapshot.get('position')
        pos_txt = 'n/a'
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                pos_txt = f'({float(pos[0]):+.2f}, {float(pos[1]):+.2f}) m'
            except Exception:
                pass

        if self.telemetry_var is not None:
            self.telemetry_var.set(
                f'Cell: {current}\n'
                f'Heading: {heading}\n'
                f'Visited: {visited_count}\n'
                f'Exit candidates: {exit_count}\n'
                f'Targets locked: {target_count}\n'
                f'Targets fired (IR): {fired_count}\n'
                f'Targets waiting >120cm: {too_far_count}\n'
                f'Pending glimpses: {glimpse_count}\n'
                f'Chassis odom XY: {pos_txt}'
            )

        if self.target_runtime_var is not None:
            self.target_runtime_var.set(str(snapshot.get('target_status', 'n/a')))

        policy = snapshot.get('target_fire_policy') or self.target_policy
        if policy:
            summary = self._policy_summary_text(policy)
            if self.fire_policy_summary_var is not None:
                self.fire_policy_summary_var.set(summary)

        if self.fire_runtime_var is not None:
            last_event = snapshot.get('last_fire_event', 'No fire event yet.')
            self.fire_runtime_var.set(
                f"{self._policy_summary_text(policy)}\n"
                f"Last event: {last_event}"
            )

        self._draw_map()

    def _enter_exit_decision(self, options):
        self.candidate_options = list(options or [])
        self.candidate_by_index = []
        self.selected_option = None
        self.candidate_list.delete(0, self.tk.END)

        for option in self.candidate_options:
            if not option.get('reachable', True):
                continue
            self.candidate_by_index.append(option)
            cid = option.get('id', '?')
            cell = tuple(option.get('cell', (0, 0)))
            direction = option.get('dir', '?')
            moves = option.get('moves')
            self.candidate_list.insert(
                self.tk.END,
                f'{cid:<3}  {cell!s:<10} -> {direction}   {moves} moves'
            )

        self.mode_var.set('WAITING FOR EXIT SELECTION')
        self.status_var.set(
            'Robot is safely at START. Select the EXIT_CANDIDATE you want to inspect.'
        )
        self.finish_button.configure(state='normal')
        self.continue_button.configure(state='disabled')

        if self.candidate_by_index:
            self.candidate_list.selection_set(0)
            self.candidate_list.activate(0)
            self._select_option(self.candidate_by_index[0])
        else:
            self.candidate_detail_var.set('No reachable EXIT_CANDIDATE.')

        self._draw_map()

    def _on_list_select(self, _event=None):
        selected = self.candidate_list.curselection()
        if not selected:
            return
        idx = int(selected[0])
        if 0 <= idx < len(self.candidate_by_index):
            self._select_option(self.candidate_by_index[idx])

    def _select_option(self, option):
        self.selected_option = option
        key = option.get('key')
        self.selected_candidate_key = key

        cid = option.get('id', '?')
        cell = tuple(option.get('cell', (0, 0)))
        direction = option.get('dir', '?')
        moves = option.get('moves', 0)
        distance_m = option.get('route_distance_m', 0.0)
        front = option.get('front_mm')
        reason = option.get('reason', 'unknown')
        path = option.get('path', [])
        front_txt = 'n/a' if front is None else f'{float(front):.0f} mm'

        self.candidate_detail_var.set(
            f'{cid}: source {cell} -> {direction}\n'
            f'Shortest confirmed route: {moves} cell moves\n'
            f'Approx. route to source: {distance_m:.2f} m\n'
            f'Front ToF when recorded: {front_txt}\n'
            f'Reason: {reason}\n'
            f'Path: {path}'
        )
        self.continue_button.configure(state='normal')

        # Mirror selection in listbox when the map marker was clicked.
        for i, item in enumerate(self.candidate_by_index):
            if item.get('key') == key:
                self.candidate_list.selection_clear(0, self.tk.END)
                self.candidate_list.selection_set(i)
                self.candidate_list.activate(i)
                self.candidate_list.see(i)
                break

        self._draw_map()

    def _continue_selected(self):
        if self.selected_option is None:
            return

        cid = self.selected_option.get('id', '?')
        cell = tuple(self.selected_option.get('cell', (0, 0)))
        direction = self.selected_option.get('dir', '?')

        ok = self.messagebox.askyesno(
            'Confirm EXIT traversal',
            f'Continue via {cid}: {cell} -> {direction}?\n\n'
            'The robot will first follow the displayed shortest confirmed route, '
            'then cross only this approved EXIT edge. Collision safety remains active.',
            parent=self.root,
        )
        if not ok:
            return

        self.decision = 'continue'
        self.selected_candidate_key = self.selected_option.get('key')
        self.continue_button.configure(state='disabled')
        self.finish_button.configure(state='disabled')
        self.mode_var.set('EXIT ROUTE APPROVED')
        self.status_var.set(f"Operator approved {cid}. Robot may leave START.")
        self.decision_event.set()

    def _finish_selected(self):
        ok = self.messagebox.askyesno(
            'Finish mission',
            'Finish the mission at START and do not traverse any deferred EXIT_CANDIDATE?',
            parent=self.root,
        )
        if not ok:
            return

        self.decision = 'finish'
        self.selected_candidate_key = None
        self.continue_button.configure(state='disabled')
        self.finish_button.configure(state='disabled')
        self.mode_var.set('FINISH SELECTED')
        self.status_var.set('Operator selected FINISH at START.')
        self.decision_event.set()

    def _emergency_stop(self):
        ok = self.messagebox.askyesno(
            'Emergency stop',
            'Stop the current mission?\n\nMotion loops will exit and the chassis will be stopped by cleanup.',
            parent=self.root,
        )
        if not ok:
            return

        self.explorer.running = False
        self.decision = 'finish'
        self.selected_candidate_key = None
        self.decision_event.set()
        if not self.target_policy_event.is_set():
            self.target_policy = None
            self.target_policy_event.set()
        self.mode_var.set('STOP REQUESTED')
        self.status_var.set('Emergency stop requested. Waiting for motion loop to stop...')
        self.continue_button.configure(state='disabled')
        self.finish_button.configure(state='disabled')

    def _on_close(self):
        if not self.mission_complete and self.explorer.running:
            ok = self.messagebox.askyesno(
                'Close Mission Control',
                'Closing Mission Control during a mission will request a safe stop. Continue?',
                parent=self.root,
            )
            if not ok:
                return
            self.explorer.running = False
            self.decision = 'finish'
            self.selected_candidate_key = None
            self.decision_event.set()
            if not self.target_policy_event.is_set():
                self.target_policy = None
                self.target_policy_event.set()

        try:
            self.root.destroy()
        except Exception:
            pass

    # --------------------------------------------------------
    # Live map drawing
    # --------------------------------------------------------

    def _on_map_click(self, event):
        if not self.candidate_by_index:
            return

        best = None
        best_d2 = None
        for x, y, option in self.marker_hits:
            d2 = (float(event.x) - x) ** 2 + (float(event.y) - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = option

        if best is not None and best_d2 is not None and best_d2 <= 28.0 ** 2:
            self._select_option(best)

    @staticmethod
    def _dir_vec_screen(direction):
        direction = int(direction) % 4
        if direction == 0:
            return (0, -1)
        if direction == 1:
            return (1, 0)
        if direction == 2:
            return (0, 1)
        return (-1, 0)

    def _draw_map(self):
        canvas = self.canvas
        snap = self.latest_snapshot
        if canvas is None:
            return

        canvas.delete('all')
        self.marker_hits = []

        if not isinstance(snap, dict):
            canvas.create_text(
                24, 24,
                anchor='nw',
                text='Waiting for map data...',
                fill='#8b949e',
                font=('Segoe UI', 12),
            )
            return

        cells = {tuple(c) for c in snap.get('cells', [])}
        if not cells:
            cells.add(tuple(snap.get('root', (0, 0))))

        root_cell = tuple(snap.get('root', (0, 0)))
        current = tuple(snap.get('current', root_cell))
        cells.add(root_cell)
        cells.add(current)

        exits = list(snap.get('exit_candidates', []))
        open_map = {
            tuple(item['cell']): set(int(d) for d in item.get('dirs', []))
            for item in snap.get('open_dirs', [])
        }
        blocked = {
            frozenset((tuple(edge[0]), tuple(edge[1])))
            for edge in snap.get('blocked_edges', [])
            if isinstance(edge, (list, tuple)) and len(edge) == 2
        }
        visited = {tuple(c) for c in snap.get('visited', [])}
        dead = {tuple(c) for c in snap.get('dead_end_cells', [])}

        candidate_keys = {
            (tuple(item.get('cell', (0, 0))), int(item.get('dir_index', 0)) % 4)
            for item in exits
        }

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        w = max(420, int(canvas.winfo_width()))
        h = max(420, int(canvas.winfo_height()))
        pad = 54
        cols = max(1, max_x - min_x + 1)
        rows = max(1, max_y - min_y + 1)
        cell_size = min(96.0, (w - 2 * pad) / cols, (h - 2 * pad) / rows)
        cell_size = max(38.0, cell_size)

        map_w = (max_x - min_x) * cell_size
        map_h = (max_y - min_y) * cell_size
        origin_x = (w - map_w) / 2.0
        origin_y = (h - map_h) / 2.0

        def center(cell):
            x, y = cell
            cx = origin_x + (x - min_x) * cell_size
            cy = origin_y + (max_y - y) * cell_size
            return cx, cy

        # Cells are drawn as a true edge-to-edge grid.  The previous GUI used
        # smaller node boxes plus graph-link lines between their centres, which
        # made the maze look like a graph instead of the physical 60 cm grid.
        # With half == cell_size / 2 every neighbouring cell touches exactly at
        # its shared boundary; OPEN edges are gaps in the wall, not connector
        # lines between nodes.
        half = cell_size * 0.50

        # Cells + wall segments.
        entrance_dir = int(snap.get('known_entrance_dir', 2)) % 4
        ingress_dir = int(snap.get('known_maze_ingress_dir', 0)) % 4

        for cell in sorted(cells, key=lambda p: (p[1], p[0])):
            cx, cy = center(cell)
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

            if cell == root_cell:
                fill = '#173d2a'
            elif cell == current:
                fill = '#5a4714'
            elif cell in dead:
                fill = '#4a2024'
            elif cell in visited:
                fill = '#172b42'
            else:
                fill = '#21262d'

            # Edge-to-edge tile.  A very thin neutral outline keeps individual
            # cells readable while preserving the continuous grid appearance.
            # Confirmed walls are drawn afterward with a much heavier stroke.
            canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=fill,
                outline='#30363d',
                width=1,
            )
            canvas.create_text(
                cx, cy + half - 12,
                text=f'({cell[0]},{cell[1]})',
                fill='#8b949e',
                font=('Consolas', max(8, int(cell_size * 0.11))),
            )

            opens = open_map.get(cell, set())
            for d in range(4):
                vx, vy = DIR_VEC[d]
                nb = (cell[0] + vx, cell[1] + vy)
                edge = frozenset((cell, nb))

                special_open = (
                    (cell, d) in candidate_keys
                    or (cell == root_cell and d == entrance_dir)
                    or (cell == root_cell and d == ingress_dir)
                )
                is_open = (d in opens and edge not in blocked) or special_open
                if is_open:
                    continue

                if d == 0:
                    coords = (x1, y1, x2, y1)
                elif d == 1:
                    coords = (x2, y1, x2, y2)
                elif d == 2:
                    coords = (x1, y2, x2, y2)
                else:
                    coords = (x1, y1, x1, y2)
                canvas.create_line(*coords, fill='#f0f6fc', width=max(2, int(cell_size * 0.055)))

        # Route overlay is intentionally drawn AFTER the grid cells/walls.
        # With edge-to-edge cells there is no inter-node gap anymore, so drawing
        # this underneath the cells would hide the route completely.
        # During normal autonomous motion this comes from the
        # explorer snapshot (return-home / navigation route). During EXIT
        # selection the locally selected candidate preview takes priority.
        selected_path = [tuple(c) for c in snap.get('route_preview', [])]
        selected_dir = None
        if self.selected_option is not None:
            selected_path = [tuple(c) for c in self.selected_option.get('path', [])]
            selected_dir = self.selected_option.get('dir_index')

        if len(selected_path) >= 2:
            pts = []
            for cell in selected_path:
                pts.extend(center(cell))
            canvas.create_line(
                *pts,
                fill='#58a6ff',
                width=7,
                capstyle='round',
                joinstyle='round',
            )

        if selected_path and selected_dir is not None:
            sx, sy = center(selected_path[-1])
            dx, dy = self._dir_vec_screen(selected_dir)
            ex = sx + dx * cell_size * 0.72
            ey = sy + dy * cell_size * 0.72
            canvas.create_line(
                sx, sy, ex, ey,
                fill='#58a6ff', width=7, arrow=self.tk.LAST,
                arrowshape=(12, 14, 6),
            )

        # Blocked edges as red X.
        for edge in blocked:
            pts = list(edge)
            if len(pts) != 2 or pts[0] not in cells or pts[1] not in cells:
                continue
            ax, ay = center(pts[0])
            bx, by = center(pts[1])
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            s = 8
            canvas.create_line(mx - s, my - s, mx + s, my + s, fill='#f85149', width=3)
            canvas.create_line(mx - s, my + s, mx + s, my - s, fill='#f85149', width=3)

        # Exit markers. Prefer the decision-option IDs when available.
        option_by_key = {
            option.get('key'): option for option in self.candidate_options
        }

        for idx, item in enumerate(exits, start=1):
            cell = tuple(item.get('cell', (0, 0)))
            d = int(item.get('dir_index', 0)) % 4
            key = (cell, d)
            option = option_by_key.get(key)
            cid = option.get('id') if option else f'E{idx}'

            cx, cy = center(cell)
            dx, dy = self._dir_vec_screen(d)
            mx = cx + dx * half
            my = cy + dy * half
            ox = cx + dx * cell_size * 0.55
            oy = cy + dy * cell_size * 0.55

            selected = (
                self.selected_option is not None
                and self.selected_option.get('key') == key
            )
            color = '#d2a8ff' if selected else '#a371f7'
            radius = 14 if selected else 11
            canvas.create_line(mx, my, ox, oy, fill=color, width=4, arrow=self.tk.LAST)
            canvas.create_oval(
                ox - radius, oy - radius, ox + radius, oy + radius,
                fill=color, outline='#ffffff', width=2,
            )
            canvas.create_text(
                ox, oy,
                text=str(cid),
                fill='#0d1117',
                font=('Segoe UI Semibold', 8),
            )
            if option is not None and option.get('reachable', True):
                self.marker_hits.append((ox, oy, option))

        # Locked target markers. Prefer the ranged grid estimate when it is
        # locally plausible; otherwise show a bearing marker near the source
        # cell so a bad/long ToF return cannot throw the GUI scale off.
        target_color_ui = {
            'RED': '#ff5f56',
            'YELLOW': '#f2cc60',
            'GREEN': '#3fb950',
            'BLUE': '#58a6ff',
        }

        for rec in snap.get('targets', []):
            try:
                source = tuple(rec.get('source_cell', root_cell))
                sx, sy = center(source)
                tx, ty = sx, sy

                est = rec.get('estimated_grid_xy')
                use_est = False
                if isinstance(est, (list, tuple)) and len(est) >= 2:
                    gx, gy = float(est[0]), float(est[1])
                    grid_dist = math.hypot(gx - source[0], gy - source[1])
                    if grid_dist <= 1.8:
                        tx, ty = center((gx, gy))
                        use_est = True

                if not use_est:
                    bearing = rec.get('bearing_deg_from_north')
                    if bearing is not None:
                        rad = math.radians(float(bearing))
                        tx = sx + math.sin(rad) * cell_size * 0.38
                        ty = sy - math.cos(rad) * cell_size * 0.38

                tid = str(rec.get('id', 'T?'))
                if rec.get('fire_status') == 'FIRED_IR':
                    tid += 'F'
                if rec.get('kind') == 'SDK_MARKER':
                    fill = '#d2a8ff'
                    r = max(8, int(cell_size * 0.10))
                    canvas.create_polygon(
                        tx, ty - r,
                        tx + r, ty,
                        tx, ty + r,
                        tx - r, ty,
                        fill=fill,
                        outline='#ffffff',
                        width=2,
                    )
                else:
                    fill = target_color_ui.get(rec.get('color'), '#ffffff')
                    r = max(8, int(cell_size * 0.10))
                    shape = rec.get('shape')
                    if shape == 'CIRCLE':
                        canvas.create_oval(
                            tx - r, ty - r, tx + r, ty + r,
                            fill=fill, outline='#ffffff', width=2,
                        )
                    elif shape == 'RECT_VERTICAL':
                        canvas.create_rectangle(
                            tx - r * 0.65, ty - r,
                            tx + r * 0.65, ty + r,
                            fill=fill, outline='#ffffff', width=2,
                        )
                    elif shape == 'RECT_HORIZONTAL':
                        canvas.create_rectangle(
                            tx - r, ty - r * 0.65,
                            tx + r, ty + r * 0.65,
                            fill=fill, outline='#ffffff', width=2,
                        )
                    else:
                        canvas.create_rectangle(
                            tx - r, ty - r, tx + r, ty + r,
                            fill=fill, outline='#ffffff', width=2,
                        )

                canvas.create_text(
                    tx,
                    ty,
                    text=tid,
                    fill='#0d1117',
                    font=('Segoe UI Semibold', 8),
                )
            except Exception:
                continue

        # Unresolved one/few-frame target glimpses.  These are NOT confirmed
        # targets; the orange G? marker only tells the operator that the return
        # pass has a direction worth re-checking.
        for rec in snap.get('target_glimpses', []):
            if rec.get('resolved_target_id'):
                continue
            try:
                source = tuple(rec.get('source_cell', root_cell))
                sx, sy = center(source)
                bearing = rec.get('bearing_deg_from_north')
                if bearing is None:
                    continue
                rad = math.radians(float(bearing))
                gx = sx + math.sin(rad) * cell_size * 0.30
                gy = sy - math.cos(rad) * cell_size * 0.30
                r = max(7, int(cell_size * 0.085))
                canvas.create_oval(
                    gx - r, gy - r, gx + r, gy + r,
                    fill='#d29922', outline='#ffffff', width=2,
                )
                canvas.create_text(
                    gx, gy,
                    text='?',
                    fill='#0d1117',
                    font=('Segoe UI Semibold', 9),
                )
            except Exception:
                continue

        # Root marker.
        rx, ry = center(root_cell)
        canvas.create_text(
            rx, ry - 2,
            text='S',
            fill='#7ee787',
            font=('Segoe UI Semibold', max(12, int(cell_size * 0.20))),
        )

        # Robot heading triangle.
        cx, cy = center(current)
        heading = int(snap.get('heading', 0)) % 4
        dx, dy = self._dir_vec_screen(heading)
        px, py = -dy, dx
        tip_x = cx + dx * cell_size * 0.25
        tip_y = cy + dy * cell_size * 0.25
        back_x = cx - dx * cell_size * 0.15
        back_y = cy - dy * cell_size * 0.15
        side = cell_size * 0.14
        points = (
            tip_x, tip_y,
            back_x + px * side, back_y + py * side,
            back_x - px * side, back_y - py * side,
        )
        canvas.create_polygon(
            points,
            fill='#f2cc60',
            outline='#ffffff',
            width=2,
        )

        # North indicator.
        canvas.create_text(26, 22, text='N', fill='#e6edf3', font=('Segoe UI Semibold', 12))
        canvas.create_line(26, 56, 26, 32, fill='#e6edf3', width=3, arrow=self.tk.LAST)

