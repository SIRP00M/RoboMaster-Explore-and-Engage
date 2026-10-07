"""DFS explorer state, map persistence, routing and Round-2 missions."""

from . import config as cfg
import json
import math
import threading
import time
from collections import deque
from .config import apply_arena_profile, atomic_write_text
from datetime import datetime
from pathlib import Path
from robomaster import robot
from .robot_control import RobotControlMixin, YawPID
from .sensors import (
    SensorMixin,
    SharedState,
    cell_in_field,
    direction_between,
    neighbor,
    tof_is_open_from_center,
    tof_topology_center_mm,
    wrap_deg,
)
from .vision import TargetVisionSubsystem


class DFSMapOnlyExplorer(RobotControlMixin, SensorMixin):
    """Explorer state, mapping and mission planning."""

    MOVE_ARRIVED = "ARRIVED"
    MOVE_BLOCKED_RETURNED = "BLOCKED_RETURNED"
    MOVE_POSE_UNCERTAIN = "POSE_UNCERTAIN"
    MOVE_STOPPED = "STOPPED"

    def __init__(self):
        self.ep_robot = robot.Robot()
        self.chassis = None
        self.gimbal = None
        self.sensor_adapter = None
        self.distance_sensor = None
        self.camera = None
        self.vision = None
        self.blaster = None

        self.state = SharedState()
        self.target_system = TargetVisionSubsystem(self)
        self.fire_mode_lock = threading.Lock()
        self.fire_mode = str(cfg.TARGET_FIRE_MODE_DEFAULT).upper()
        self.fire_burst_lock = threading.Lock()
        self.fire_burst_count = int(cfg.TARGET_FIRE_BURST_DEFAULT)
        self.target_filter_lock = threading.Lock()
        self.target_selection = set(cfg.TARGET_FILTER_ALL)
        self.mission_mode = "ROUND1"
        self.round1_memory = None
        self.round1_memory_path = Path(cfg.ROUND1_ATTACK_MEMORY_JSON)
        self.round2_hints = []
        self.round2_result = {}
        # GUI-only/pre-run planner cache.  This is computed entirely from the
        # frozen Round-1 snapshot, so Preview never needs a robot connection.
        self.round2_preview_plan = {}
        self.cleanup_done = False
        self.running = True
        self.connected = False
        self.pose_trusted = True
        self.safe_pause_reason = None

        self.position_origin_raw = None
        self.position_raw_latest = None
        # Preserve logical odometry continuity if the SDK position callback has
        # to be unsubscribed/re-subscribed during a Wi-Fi recovery.
        self.position_rebase_pending = False
        self.position_rebase_logical = None
        self.base_yaw_deg = None
        self.yaw_ref_deg = None
        self.gimbal_zero_pitch_raw = None
        self.gimbal_zero_yaw_raw = None

        self.heading = 0
        self.root = tuple(cfg.ROOT_CELL)
        self.current = self.root
        self.visited = set()
        self.parent = {}
        self.open_dirs = {}
        self.edge_state = {}      # (cell, dir) -> OPEN/WALL/UNKNOWN/BLOCKED/DEFERRED
        self.cell_scan_mm = {}
        # Learned physical distance for a traversed logical edge.  Normally this
        # is ~0.60 m, but a sensor-confirmed node may be reached a little early.
        # Remembering that distance makes the reverse/backtrack traversal use the
        # same physical anchor instead of blindly overshooting by 60 cm.
        self.edge_travel_m = {}   # ((cell), dir) -> metres
        # BREADCRUMB = chronological, confirmed cell-to-cell motion.  The graph
        # says where the robot CAN go; this trail says where it ACTUALLY went.
        self.breadcrumb_lock = threading.Lock()
        self.breadcrumb_trail = []
        self.breadcrumb_seq = 0
        self.last_move_distance_m = None
        self.last_stop_probe = None
        self.edge_failures = {}
        self.turn_failures = {}
        self.deferred_edges = set()
        self.unknown_rescan_counts = {}
        # Cells that look like the outside/fake-exit apron.  The legacy heuristic
        # is disabled when configured bounds are authoritative; boundary rejection
        # then happens before movement.
        self.fake_exit_candidates = set()
        self.fake_exit_rejected_edges = set()
        self.boundary_rejected_edges = set()
        self.map_complete = False
        self.map_created_at = datetime.now().isoformat(timespec="seconds")

        self.left_adc_hist = deque(maxlen=cfg.SHARP_FILTER_SAMPLES)
        self.right_adc_hist = deque(maxlen=cfg.SHARP_FILTER_SAMPLES)
        self.sharp_authority = None
        self.sharp_authority_since = 0.0

        self.yaw_hold_lock = threading.Lock()
        self.pid_straight = YawPID(
            cfg.STRAIGHT_YAW_KP, cfg.STRAIGHT_YAW_KI, cfg.STRAIGHT_YAW_KD,
            cfg.STRAIGHT_YAW_MAX_DPS, cfg.STRAIGHT_YAW_I_LIMIT,
            cfg.STRAIGHT_YAW_I_ZONE_DEG, cfg.STRAIGHT_YAW_D_ALPHA,
            cfg.STRAIGHT_YAW_DEADBAND_DEG,
        )
        self.pid_stationary = YawPID(
            cfg.STATIONARY_YAW_KP, cfg.STATIONARY_YAW_KI, cfg.STATIONARY_YAW_KD,
            cfg.STATIONARY_YAW_MAX_DPS, cfg.STATIONARY_YAW_I_LIMIT,
            cfg.STATIONARY_YAW_I_ZONE_DEG, cfg.STATIONARY_YAW_D_ALPHA,
            cfg.STATIONARY_YAW_DEADBAND_DEG,
        )
        self.pid_turn = YawPID(
            cfg.TURN_KP, cfg.TURN_KI, cfg.TURN_KD, cfg.TURN_MAX_DPS, cfg.TURN_I_LIMIT,
            cfg.TURN_I_ZONE_DEG, cfg.TURN_D_ALPHA, 0.0,
        )
        self.fault_log = deque(maxlen=240)
        self.mission_start_t = None
        self._mission_watchdog_warned = False
        # Set only after an actual 90/180 chassis turn.  The next edge may use
        # the angled IR whiskers to trim itself through a doorway/corner.
        self.ir_corner_trim_pending = False

        # Live chassis-speed telemetry for GUI. Commanded speed is updated after
        # an accepted drive_speed() call; measured speed is estimated from odometry.
        self.speed_telemetry_lock = threading.Lock()
        self.last_drive_command = {
            "x": 0.0, "y": 0.0, "z": 0.0, "label": "STOP", "t": time.monotonic()
        }
        self.odom_velocity = {
            "vx": 0.0, "vy": 0.0, "speed": 0.0, "t": time.monotonic()
        }
        self._speed_prev_position = None
        self.active_motion_profile_name = "IDLE"

    def _set_drive_command_telemetry(self, x, y, z, label):
        with self.speed_telemetry_lock:
            self.last_drive_command = {
                "x": float(x), "y": float(y), "z": float(z),
                "label": str(label or "DRIVE"), "t": time.monotonic(),
            }

    def speed_snapshot(self):
        """Return commanded + odometry-estimated speed for the live GUI."""
        with self.speed_telemetry_lock:
            cmd = dict(self.last_drive_command)
            odom = dict(self.odom_velocity)
        vx = float(odom.get("vx", 0.0)); vy = float(odom.get("vy", 0.0))
        d = int(self.heading) % 4
        fx, fy = cfg.DIR_VEC[d]
        rx, ry = cfg.DIR_VEC[(d + 1) % 4]
        return {
            "command_x": float(cmd.get("x", 0.0)),
            "command_y": float(cmd.get("y", 0.0)),
            "command_z": float(cmd.get("z", 0.0)),
            "command_speed": math.hypot(float(cmd.get("x", 0.0)), float(cmd.get("y", 0.0))),
            "command_label": str(cmd.get("label", "-")),
            "actual_vx": vx, "actual_vy": vy,
            "actual_forward": vx * fx + vy * fy,
            "actual_right": vx * rx + vy * ry,
            "actual_speed": float(odom.get("speed", 0.0)),
            "profile": str(self.active_motion_profile_name),
        }

    def set_fire_mode(self, mode):
        mode = str(mode or "INFRARED").upper().strip()
        if mode not in ("INFRARED", "WATER"):
            mode = "INFRARED"
        with self.fire_mode_lock:
            self.fire_mode = mode
        print("[FIRE MODE] {}".format(mode))
        return mode

    def get_fire_mode(self):
        with self.fire_mode_lock:
            return str(self.fire_mode)

    def set_fire_burst_count(self, count):
        try:
            count = int(count)
        except Exception:
            count = int(cfg.TARGET_FIRE_BURST_DEFAULT)
        if count not in cfg.TARGET_FIRE_BURST_OPTIONS:
            count = int(cfg.TARGET_FIRE_BURST_DEFAULT)
        with self.fire_burst_lock:
            self.fire_burst_count = count
        print("[FIRE BURST MODE] {} shot(s), interval={:.2f}s".format(
            count, float(cfg.TARGET_FIRE_BURST_INTERVAL_SEC)
        ))
        return count

    def get_fire_burst_count(self):
        with self.fire_burst_lock:
            return int(self.fire_burst_count)

    def set_mission_mode(self, mode):
        mode = str(mode or "ROUND1").upper().replace(" ", "")
        if mode in ("1", "ROUND1", "R1"):
            mode = "ROUND1"
        elif mode in ("2", "ROUND2", "R2"):
            mode = "ROUND2"
        else:
            mode = "ROUND1"
        self.mission_mode = mode
        print("[MISSION MODE] {}".format(mode))
        return mode

    def set_target_selection(self, pairs):
        cleaned = set()
        for item in pairs or ():
            try:
                color, shape = item
            except Exception:
                continue
            key = (str(color).upper(), str(shape).upper())
            if key in cfg.TARGET_FILTER_ALL:
                cleaned.add(key)
        with self.target_filter_lock:
            self.target_selection = cleaned
        print("[TARGET FILTER] enabled {}/16: {}".format(
            len(cleaned), ", ".join("{}/{}".format(c, sh) for c, sh in sorted(cleaned)) or "NONE"
        ))
        return set(cleaned)

    def get_target_selection(self):
        with self.target_filter_lock:
            return sorted(self.target_selection)

    def target_allowed(self, color, shape):
        key = (str(color or "").upper(), str(shape or "").upper())
        with self.target_filter_lock:
            return key in self.target_selection

    def reset_map_for_fresh_round1(self):
        """Clear any preloaded Round-2 snapshot before a brand-new Round 1."""
        if self.connected:
            return False
        self.heading = 0
        self.root = tuple(cfg.ROOT_CELL)
        self.current = self.root
        self.visited = set()
        self.parent = {}
        self.open_dirs = {}
        self.edge_state = {}
        self.cell_scan_mm = {}
        self.edge_travel_m = {}
        with self.breadcrumb_lock:
            self.breadcrumb_trail = []
            self.breadcrumb_seq = 0
        self.last_move_distance_m = None
        self.last_stop_probe = None
        self.edge_failures = {}
        self.turn_failures = {}
        self.deferred_edges = set()
        self.unknown_rescan_counts = {}
        self.fake_exit_candidates = set()
        self.fake_exit_rejected_edges = set()
        self.boundary_rejected_edges = set()
        self.map_complete = False
        self.map_created_at = datetime.now().isoformat(timespec="seconds")
        self.round1_memory = None
        self.round1_memory_path = Path(cfg.ROUND1_ATTACK_MEMORY_JSON)
        self.round2_hints = []
        self.round2_preview_plan = {}
        self.safe_pause_reason = None
        return True

    def breadcrumb_snapshot(self):
        """Thread-safe copy for autosave/GUI without blocking mission control."""
        with self.breadcrumb_lock:
            return [dict(row) for row in self.breadcrumb_trail]

    def _record_breadcrumb(self, source_cell, abs_dir, motion_profile, distance_m=None):
        """Record ONE confirmed cell-to-cell arrival, including backtracks.

        Only MOVE_ARRIVED is recorded.  Failed mid-edge excursions are deliberately
        excluded so the breadcrumb remains a trustworthy sequence of logical anchors.
        """
        source = tuple(source_cell)
        d = int(abs_dir) % 4
        dest = neighbor(source, d)
        try:
            dist = float(distance_m if distance_m is not None else self.last_move_distance_m)
            if not math.isfinite(dist):
                raise ValueError
        except Exception:
            learned = self.remembered_edge_distance(source, d)
            dist = float(learned if learned is not None else cfg.CELL_LENGTH_M)
        profile_name = str((motion_profile or {}).get("name") if isinstance(motion_profile, dict) else motion_profile or "EXPLORE").upper()
        if self.mission_mode == "ROUND2":
            move_kind = "ROUND2_SHORTEST"
        elif profile_name == "EXPLORE":
            move_kind = "EXPLORE_NEW_EDGE"
        elif "BACKTRACK" in profile_name:
            move_kind = "DFS_BACKTRACK"
        else:
            move_kind = "KNOWN_ROUTE_RELOCATION"
        with self.breadcrumb_lock:
            self.breadcrumb_seq += 1
            row = {
                "seq": int(self.breadcrumb_seq),
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "mission_mode": str(self.mission_mode),
                "kind": move_kind,
                "profile": profile_name,
                "from": [int(source[0]), int(source[1])],
                "to": [int(dest[0]), int(dest[1])],
                "dir": cfg.DIR_NAMES[d],
                "dir_index": d,
                "distance_m": round(float(dist), 4),
            }
            self.breadcrumb_trail.append(row)
        print(
            "[BREADCRUMB #{:03d}] {} -> {} dir={} profile={} d={:.3f}m mode={}".format(
                row["seq"], source, dest, row["dir"], profile_name, dist, self.mission_mode
            )
        )
        return row

    def save_breadcrumb(self):
        try:
            trail = self.breadcrumb_snapshot()
            payload = {
                "schema": "robomaster_breadcrumb_trail",
                "version": 1,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "count": len(trail),
                "root": [int(self.root[0]), int(self.root[1])],
                "current": [int(self.current[0]), int(self.current[1])],
                "trail": trail,
            }
            atomic_write_text(
                cfg.BREADCRUMB_LATEST_JSON,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            return True
        except Exception as exc:
            self.fault("BREADCRUMB SAVE", f"{type(exc).__name__}: {exc}", "keep trail in RAM/map bundle")
            return False

    def set_edge_state(self, cell, direction, state):
        """Store one sensed edge state while enforcing configured arena bounds.

        An OPEN ToF ray is only a geometric observation.  If that ray points from
        a legal cell to an out-of-field logical coordinate, it is semantically an
        OUTSIDE aperture and is stored as BLOCKED before DFS can translate.
        """
        direction = int(direction) % 4
        cell = tuple(cell)
        state = str(state)
        nb = neighbor(cell, direction)

        if (
            cfg.FIELD_BOUNDARY_GUARD_ENABLED
            and cell_in_field(cell)
            and not cell_in_field(nb)
            and state == "OPEN"
        ):
            state = "BLOCKED"
            key = (cell, direction)
            first = key not in self.boundary_rejected_edges
            self.boundary_rejected_edges.add(key)
            self.fake_exit_rejected_edges.add(key)  # retained for old GUI/snapshot compatibility
            if first:
                print(
                    "[FIELD BOUNDARY] {} -> {} dir={} sensor=OPEN but destination is "
                    "outside {}x{} -> BLOCKED (no translation)".format(
                        cell, nb, cfg.DIR_NAMES[direction], cfg.GRID_WIDTH_CELLS, cfg.GRID_HEIGHT_CELLS
                    )
                )

        self.edge_state[(cell, direction)] = state
        opposite = (direction + 2) % 4

        # Never create reciprocal pseudo-cells outside the legal map.
        if not cell_in_field(nb):
            return state

        existing = self.edge_state.get((nb, opposite))
        if state == "OPEN":
            if existing not in ("WALL", "BLOCKED"):
                self.edge_state[(nb, opposite)] = "OPEN"
        elif state in ("WALL", "BLOCKED"):
            if existing != "OPEN":
                self.edge_state[(nb, opposite)] = state
        return state

    def mark_traversed_open(self, cell, direction):
        """Physical traversal is definitive OPEN evidence, but never outside configured arena bounds."""
        direction = int(direction) % 4
        cell = tuple(cell)
        nb = neighbor(cell, direction)
        if cfg.FIELD_BOUNDARY_GUARD_ENABLED and not cell_in_field(nb):
            self.set_edge_state(cell, direction, "BLOCKED")
            return False
        self.edge_state[(cell, direction)] = "OPEN"
        self.edge_state[(nb, (direction + 2) % 4)] = "OPEN"
        return True

    def root_back_scan(self):
        if not cfg.ROOT_BACK_SCAN_ENABLED or self.base_yaw_deg is None:
            return None
        original_heading = self.heading
        original_target = self.desired_yaw_for_heading(original_heading)
        back_target = wrap_deg(original_target + 180.0)
        print("[ROOT BACK] temporary 180-degree chassis scan")

        turned = self.turn_closed_loop(back_target, cfg.TURN_TIMEOUT_180_SEC)
        if not turned:
            self.fault("ROOT BACK", "could not reach back heading", "restore runtime North")
        mm = None
        if turned:
            self.gimbal_front_down(force=True)
            mm = self.sample_fresh_tof()

        restored = False
        for attempt in range(cfg.ROOT_BACK_RESTORE_RETRIES + 1):
            if self.turn_closed_loop(
                original_target,
                cfg.TURN_TIMEOUT_180_SEC * (1.0 if attempt == 0 else cfg.TURN_RECOVERY_TIMEOUT_SCALE),
            ):
                restored = True
                break
            self.fault("ROOT BACK", f"restore attempt {attempt+1} failed", "retry")

        self.heading = original_heading
        self.yaw_ref_deg = original_target
        if restored:
            # turn_closed_loop already performs a fine settle, but verify the
            # mission cardinal once more before the first DFS translation.
            restored = self.align_heading_stationary(
                original_target, timeout_sec=cfg.STATIONARY_ALIGN_TIMEOUT_SEC,
                tolerance_deg=cfg.STATIONARY_SETTLE_TOL_DEG, settle_sec=0.12,
            )

        if not restored:
            snapped = self.recover_to_nearest_cardinal()
            if snapped != original_heading:
                self.enter_safe_pause("root BACK scan could not restore a trusted runtime heading")
                return mm

        self.pid_straight.reset()
        self.gimbal_front_down(force=True)
        return mm

    def commit_stop_probe_to_cell(self, cell, parent_cell, probe_rays):
        """Commit an already-performed stopped L/F/R probe as this cell's map.

        The chassis has already been accepted at `cell`, so repeating the same
        mechanical scan is unnecessary.  BACK is definitive because it is the
        edge physically traversed from parent_cell.
        """
        if not isinstance(probe_rays, dict):
            return False
        cell = tuple(cell)
        parent_cell = tuple(parent_cell) if parent_cell is not None else None
        ordered_open = []
        scan = {}
        valid = 0

        print(f"[NODE MAP] commit stopped Gimbal probe -> cell={cell} heading={cfg.DIR_NAMES[self.heading]}")
        for label, _yaw_deg, rel in cfg.SCAN_RELATIVE_ORDER:
            mm = probe_rays.get(label)
            scan[label] = mm
            abs_dir = (self.heading + rel) % 4
            if mm is None:
                state = "UNKNOWN"
            else:
                valid += 1
                state = "OPEN" if tof_is_open_from_center(mm) else "WALL"
            state = self.set_edge_state(cell, abs_dir, state)
            if state == "OPEN":
                ordered_open.append(abs_dir)
            center_mm = tof_topology_center_mm(mm)
            print(f"  [NODE MAP] {label:<5} ToF(raw)={mm} center={center_mm} -> {state} ({cfg.DIR_NAMES[abs_dir]})")

        if parent_cell is not None:
            back_dir = direction_between(cell, parent_cell)
            if back_dir is not None:
                self.mark_traversed_open(cell, back_dir)
                scan["BACK"] = "TRAVERSED"
                if back_dir not in ordered_open:
                    ordered_open.append(back_dir)

        seen = set()
        ordered_open = [d for d in ordered_open if not (d in seen or seen.add(d))]
        self.cell_scan_mm[cell] = scan
        self.open_dirs[cell] = ordered_open
        print("  [NODE MAP] OPEN:", [cfg.DIR_NAMES[d] for d in ordered_open])
        return valid >= cfg.FRONT_NODE_CAPTURE_MIN_VALID_RAYS

    def scan_cell(self, cell):
        """Exception-contained stationary topology scan."""
        last_exc = None
        for attempt in range(1, cfg.SCAN_EXCEPTION_RETRIES + 2):
            try:
                return self._scan_cell_impl(cell)
            except Exception as exc:
                last_exc = exc
                self.safe_stop()
                self.fault(
                    "SCAN UNEXPECTED",
                    f"cell={cell} attempt {attempt}: {type(exc).__name__}: {exc}",
                    "recover telemetry/gimbal and retry stationary scan",
                )
                self.recover_telemetry(
                    require_position=True, require_attitude=True, require_tof=True, require_gimbal=True,
                    reason=f"scan_cell {cell}",
                )
                self.recover_gimbal_front()
                time.sleep(0.10 * attempt)

        # A scan failure at a known node must not crash DFS. Preserve the parent
        # edge (if any) so the robot can still backtrack; leave all other bearings
        # UNKNOWN for later recovery.
        ordered_open = []
        cell = tuple(cell)
        p = self.parent.get(cell)
        if p is not None:
            back_dir = direction_between(cell, p)
            if back_dir is not None:
                self.mark_traversed_open(cell, back_dir)
                ordered_open.append(back_dir)
        for d in range(4):
            if (cell, d) not in self.edge_state:
                self.set_edge_state(cell, d, "UNKNOWN")
        self.open_dirs[cell] = ordered_open
        self.cell_scan_mm[cell] = {"ERROR": None}
        self.fault(
            "SCAN FALLBACK",
            f"cell={cell}: {type(last_exc).__name__ if last_exc else 'unknown'}",
            "continue DFS with UNKNOWN edges / parent backtrack",
        )
        return ordered_open

    def _confirm_fake_exit_candidate(self, cell, scan, ordered_open):
        """Strictly verify a newly-entered wide-open apron / fake exit.

        A normal 60-cm four-way intersection usually has nearby diagonal corner
        geometry, while an opening that leads outside the maze stays long on both
        +/-45-degree rays.  We only flag the cell; the DFS loop performs the
        physical retreat so scan_cell itself never translates the chassis.
        """
        if not cfg.FAKE_EXIT_GUARD_ENABLED or tuple(cell) == tuple(self.root):
            return False
        p = self.parent.get(tuple(cell))
        if p is None:
            return False

        # BACK is already a proven traversed OPEN edge.  Require all four
        # cardinal directions open before spending time on the two diagonal rays.
        back_dir = direction_between(tuple(cell), tuple(p))
        if back_dir is None:
            return False
        cardinal_open = all(
            self.edge_state.get((tuple(cell), d)) == "OPEN" for d in range(4)
        )
        if cfg.FAKE_EXIT_REQUIRE_ALL_CARDINAL_OPEN and not cardinal_open:
            return False

        diag_rows = []
        diag_ok = True
        print(f"[EXIT GUARD] cell={cell} four-way OPEN -> verify diagonals +/-45deg")
        for yaw in cfg.FAKE_EXIT_DIAG_YAWS_DEG:
            mm = self.scan_tof_at_yaw(yaw)
            center_mm = tof_topology_center_mm(mm)
            diag_rows.append((float(yaw), mm, center_mm))
            good = bool(center_mm is not None and center_mm >= cfg.FAKE_EXIT_DIAG_MIN_CENTER_MM)
            diag_ok = diag_ok and good
            print(
                "  [EXIT DIAG] yaw={:+.0f} raw={} center={} -> {}".format(
                    yaw, mm, center_mm, "WIDE" if good else "MAZE_GEOMETRY"
                )
            )

        scan["EXIT_DIAGONALS"] = [
            {"yaw_deg": y, "raw_mm": mm, "center_mm": cm} for y, mm, cm in diag_rows
        ]
        self.gimbal_front_down(force=True)
        if not diag_ok:
            print(f"[EXIT GUARD] cell={cell} rejected: diagonals do not look like outside apron")
            return False

        self.fake_exit_candidates.add(tuple(cell))
        print(
            f"[FAKE EXIT CANDIDATE] cell={cell} parent={p} -> DO NOT finish mission; "
            "DFS will retreat and block this edge"
        )
        return True

    def _reject_fake_exit_and_return(self, cell):
        """Return from a verified wide-open fake exit and blacklist its edge."""
        cell = tuple(cell)
        parent = self.parent.get(cell)
        if parent is None:
            return False
        parent = tuple(parent)
        back_dir = direction_between(cell, parent)
        out_dir = direction_between(parent, cell)
        if back_dir is None or out_dir is None:
            return False

        print(f"[FAKE EXIT RETURN] {cell} -> {parent} dir={cfg.DIR_NAMES[back_dir]}")
        if not self.turn_to_direction(back_dir):
            self.enter_safe_pause(f"fake-exit retreat turn failed at {cell}")
            return False
        result = self.move_one_cell(cell, back_dir, motion_profile="BACKTRACK_FAST")
        if result != self.MOVE_ARRIVED:
            self.enter_safe_pause(f"fake-exit retreat could not prove return {cell}->{parent}")
            return False

        self.current = parent
        # Semantic block: the aperture is physically traversable but intentionally
        # excluded from maze exploration / Round-2 shortest routing.
        self.edge_state[(parent, out_dir)] = "BLOCKED"
        self.edge_state[(cell, back_dir)] = "BLOCKED"
        self.fake_exit_rejected_edges.add((parent, out_dir))
        self.open_dirs[parent] = [d for d in self.open_dirs.get(parent, []) if d != out_dir]
        # Resolve all outside-cell bearings so they cannot keep map_complete false.
        for d in range(4):
            self.edge_state[(cell, d)] = "BLOCKED"
        self.open_dirs[cell] = []
        self.fake_exit_candidates.discard(cell)
        print(
            f"[FAKE EXIT BLOCKED] edge {parent}->{cell} ({cfg.DIR_NAMES[out_dir]}) excluded; "
            "continue exploring remaining maze"
        )
        if cfg.MAP_AUTOSAVE:
            self.save_map(final=False)
        return True

    def _scan_cell_impl(self, cell):
        print(f"\n[SCAN] cell={cell} heading={cfg.DIR_NAMES[self.heading]}")
        ordered_open = []
        scan = {}

        # IR at a node is informational only.  Never slide the chassis during a
        # topology scan; that would move the node anchor.
        l_low, r_low, l_raw, r_raw = self.read_ir_filtered()
        if l_low or r_low:
            print(f"[IR NODE] L={l_raw} R={r_raw} -> HOLD POSITION, scan only")
            self.safe_stop()

        for label, yaw_deg, rel in cfg.SCAN_RELATIVE_ORDER:
            mm = self.scan_tof_at_yaw(yaw_deg)
            abs_dir = (self.heading + rel) % 4
            scan[label] = mm
            if mm is None:
                state = "UNKNOWN"
            elif tof_is_open_from_center(mm):
                state = "OPEN"
            else:
                state = "WALL"
            state = self.set_edge_state(cell, abs_dir, state)
            if state == "OPEN":
                ordered_open.append(abs_dir)
            center_mm = tof_topology_center_mm(mm)
            print(f"  {label:<5} yaw={yaw_deg:+5.0f} ToF(raw)={mm} center={center_mm} -> {state} ({cfg.DIR_NAMES[abs_dir]})")

        # A transient gimbal/ToF miss must not make DFS immediately backtrack
        # from a cell that may still have an unexplored route.  Retry ONLY the
        # UNKNOWN bearings while the chassis stays parked on the node anchor.
        for recover_pass in range(1, cfg.UNKNOWN_EDGE_RESCAN_PASSES + 1):
            missing = [item for item in cfg.SCAN_RELATIVE_ORDER if scan.get(item[0]) is None]
            if not missing:
                break
            print(
                f"[SCAN RECOVER] pass {recover_pass}/{cfg.UNKNOWN_EDGE_RESCAN_PASSES} "
                f"retry UNKNOWN={[m[0] for m in missing]}"
            )
            self.safe_stop()
            time.sleep(cfg.UNKNOWN_EDGE_RESCAN_SETTLE_SEC)
            for label, yaw_deg, rel in missing:
                mm = self.scan_tof_at_yaw(yaw_deg)
                if mm is None:
                    continue
                scan[label] = mm
                abs_dir = (self.heading + rel) % 4
                state = "OPEN" if tof_is_open_from_center(mm) else "WALL"
                state = self.set_edge_state(cell, abs_dir, state)
                if state == "OPEN" and abs_dir not in ordered_open:
                    ordered_open.append(abs_dir)
                print(
                    f"  [SCAN RECOVER] {label:<5} yaw={yaw_deg:+5.0f} "
                    f"ToF={mm} -> {state} ({cfg.DIR_NAMES[abs_dir]})"
                )

        p = self.parent.get(cell)
        if p is not None:
            back_dir = direction_between(cell, p)
            if back_dir is not None:
                self.mark_traversed_open(cell, back_dir)
                if back_dir not in ordered_open:
                    ordered_open.append(back_dir)
        elif tuple(cell) == self.root and cfg.ROOT_BACK_SCAN_ENABLED and self.pose_trusted:
            mm = self.root_back_scan()
            scan["BACK"] = mm
            back_dir = (self.heading + cfg.REL_BACK) % 4
            if mm is None:
                state = self.set_edge_state(cell, back_dir, "UNKNOWN")
            elif tof_is_open_from_center(mm):
                state = self.set_edge_state(cell, back_dir, "OPEN")
                if state == "OPEN" and back_dir not in ordered_open:
                    ordered_open.append(back_dir)
            else:
                state = self.set_edge_state(cell, back_dir, "WALL")
            print(f"  BACK  chassis-180 ToF={mm} -> {self.edge_state.get((tuple(cell), back_dir))} ({cfg.DIR_NAMES[back_dir]})")

        self.cell_scan_mm[tuple(cell)] = scan
        # Deduplicate while preserving scan/parent order.
        seen = set()
        ordered_open = [d for d in ordered_open if not (d in seen or seen.add(d))]
        self.open_dirs[tuple(cell)] = ordered_open
        print("  OPEN:", [cfg.DIR_NAMES[d] for d in ordered_open])

        # Legacy geometry-based fake-exit guard is disabled when configured bounds are authoritative.
        # The authoritative boundary guard rejects any out-of-field opening before movement.
        try:
            self._confirm_fake_exit_candidate(cell, scan, ordered_open)
        except Exception as exc:
            self.fault(
                "EXIT GUARD", f"cell={cell}: {type(exc).__name__}: {exc}",
                "ignore exit classification and continue normal DFS",
            )

        # TARGET SERVICE ISOLATION:
        # Do not waste competition time searching/shooting in a verified outside
        # apron.  The DFS loop will retreat from this cell immediately.
        if tuple(cell) not in self.fake_exit_candidates:
            try:
                self.target_system.scan_cell(cell)
            except Exception as exc:
                self.fault(
                    "TARGET SCAN",
                    f"cell={cell}: {type(exc).__name__}: {exc}",
                    "skip targets at this node / continue DFS",
                )
        else:
            print(f"[FAKE EXIT] skip target service at outside candidate cell={cell}")

        self.gimbal_front_down(force=True)
        return ordered_open

    def mapped_cells(self):
        cells = set(self.visited)
        cells.update(self.open_dirs.keys())
        for (cell, d), state in self.edge_state.items():
            cells.add(cell)
            if state == "OPEN" and neighbor(cell, d) in self.visited:
                cells.add(neighbor(cell, d))
        return cells

    def build_map_payload(self, final=False):
        edge_rows = []
        for (cell, d), state in sorted(self.edge_state.items(), key=lambda x: (x[0][0][1], x[0][0][0], x[0][1])):
            edge_rows.append({
                "cell": [cell[0], cell[1]],
                "dir": cfg.DIR_NAMES[d],
                "dir_index": d,
                "state": state,
            })
        return {
            "schema": "robomaster_dfs_map_only",
            "version": 1,
            "created_at": self.map_created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "complete": bool(final and self.map_complete),
            "safe_pause_reason": self.safe_pause_reason,
            "root": [int(self.root[0]), int(self.root[1])],
            "current": [self.current[0], self.current[1]],
            "heading": cfg.DIR_NAMES[self.heading],
            "grid_tile_m": cfg.GRID_TILE_M,
            "cell_length_m": cfg.CELL_LENGTH_M,
            "tof_open_threshold_mm": cfg.TOF_OPEN_THRESHOLD_MM,
            "tof_open_threshold_reference": "robot_center_planar",
            "tof_forward_from_center_m": cfg.TOF_FORWARD_FROM_CENTER_M,
            "tof_raw_open_equivalent_mm_approx": max(
                0.0, cfg.TOF_OPEN_THRESHOLD_MM - cfg.TOF_FORWARD_FROM_CENTER_M*1000.0
            ),
            "visited": [[x, y] for x, y in sorted(self.visited)],
            "fake_exit_rejected_edges": [
                {"cell": [c[0], c[1]], "dir": cfg.DIR_NAMES[d], "dir_index": int(d)}
                for c, d in sorted(self.fake_exit_rejected_edges, key=lambda row: (row[0][1], row[0][0], row[1]))
            ],
            "field_bounds": {
                "profile_name": cfg.ARENA_PROFILE_NAME,
                "width_cells": cfg.GRID_WIDTH_CELLS,
                "height_cells": cfg.GRID_HEIGHT_CELLS,
                "x_min": cfg.GRID_X_MIN, "x_max": cfg.GRID_X_MAX,
                "y_min": cfg.GRID_Y_MIN, "y_max": cfg.GRID_Y_MAX,
                "boundary_guard": bool(cfg.FIELD_BOUNDARY_GUARD_ENABLED),
            },
            "boundary_rejected_edges": [
                {"cell": [c[0], c[1]], "dir": cfg.DIR_NAMES[d], "dir_index": int(d),
                 "outside_cell": list(neighbor(c, d))}
                for c, d in sorted(self.boundary_rejected_edges, key=lambda row: (row[0][1], row[0][0], row[1]))
            ],
            "parent": {
                f"{c[0]},{c[1]}": (None if p is None else [p[0], p[1]])
                for c, p in self.parent.items()
            },
            "open_dirs": {
                f"{c[0]},{c[1]}": [cfg.DIR_NAMES[d] for d in dirs]
                for c, dirs in self.open_dirs.items()
            },
            "edges": edge_rows,
            "cell_scan_mm": {
                f"{c[0]},{c[1]}": vals for c, vals in self.cell_scan_mm.items()
            },
            "edge_travel_m": [
                {
                    "cell": [c[0], c[1]],
                    "dir": cfg.DIR_NAMES[d],
                    "distance_m": round(float(dist), 4),
                }
                for (c, d), dist in sorted(
                    self.edge_travel_m.items(),
                    key=lambda item: (item[0][0][1], item[0][0][0], item[0][1]),
                )
            ],
            "breadcrumb_count": len(self.breadcrumb_snapshot()),
            "breadcrumb_trail": self.breadcrumb_snapshot(),
            "deferred_edges": [
                {"cell": [c[0], c[1]], "dir": cfg.DIR_NAMES[d]}
                for c, d in sorted(self.deferred_edges)
            ],
            "faults": list(self.fault_log),
        }

    def edge_symbol_state(self, cell, d):
        return self.edge_state.get((tuple(cell), d), "UNKNOWN")

    def render_ascii_map(self):
        cells = self.mapped_cells()
        if not cells:
            return "<empty map>\n"
        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)

        out = []
        for y in range(max_y, min_y - 1, -1):
            # North walls
            top = []
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                state = self.edge_symbol_state(cell, 0)
                top.append("+" + ("   " if state == "OPEN" else "---" if state in ("WALL", "BLOCKED") else " ? "))
            out.append("".join(top) + "+")

            mid = []
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                west = self.edge_symbol_state(cell, 3)
                mid.append(" " if west == "OPEN" else "|" if west in ("WALL", "BLOCKED") else "?")
                if cell == self.current:
                    glyph = "^>v<"[self.heading]
                    mid.append(f" {glyph} ")
                elif cell == self.root:
                    mid.append(" S ")
                elif cell in self.visited:
                    mid.append(" . ")
                else:
                    mid.append("   ")
            east = self.edge_symbol_state((max_x, y), 1)
            mid.append(" " if east == "OPEN" else "|" if east in ("WALL", "BLOCKED") else "?")
            out.append("".join(mid))

        bottom = []
        y = min_y
        for x in range(min_x, max_x + 1):
            state = self.edge_symbol_state((x, y), 2)
            bottom.append("+" + ("   " if state == "OPEN" else "---" if state in ("WALL", "BLOCKED") else " ? "))
        out.append("".join(bottom) + "+")
        return "\n".join(out) + "\n"

    def render_svg_map(self):
        cells = self.mapped_cells()
        if not cells:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="120"><text x="20" y="60">empty map</text></svg>\n'

        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)
        cell_px = 70
        pad = 30
        width = (max_x - min_x + 1) * cell_px + pad * 2
        height = (max_y - min_y + 1) * cell_px + pad * 2

        def xy(cell):
            x, y = cell
            sx = pad + (x - min_x) * cell_px
            sy = pad + (max_y - y) * cell_px
            return sx, sy

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<defs><marker id="crumbArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#d27b00"/></marker></defs>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g stroke-linecap="round" font-family="monospace">',
        ]

        for cell in cells:
            sx, sy = xy(cell)
            if cell in self.visited:
                lines.append(f'<rect x="{sx+3}" y="{sy+3}" width="{cell_px-6}" height="{cell_px-6}" fill="#f4f4f4" stroke="none"/>')
            for d in range(4):
                state = self.edge_symbol_state(cell, d)
                if state == "OPEN":
                    continue
                if d == 0:
                    x1, y1, x2, y2 = sx, sy, sx + cell_px, sy
                elif d == 1:
                    x1, y1, x2, y2 = sx + cell_px, sy, sx + cell_px, sy + cell_px
                elif d == 2:
                    x1, y1, x2, y2 = sx, sy + cell_px, sx + cell_px, sy + cell_px
                else:
                    x1, y1, x2, y2 = sx, sy, sx, sy + cell_px
                if state in ("WALL", "BLOCKED"):
                    dash = ""
                    stroke = "black"
                    sw = 4
                else:
                    dash = ' stroke-dasharray="5,5"'
                    stroke = "#999"
                    sw = 2
                lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{dash}/>' )

            label = "S" if cell == self.root else f"{cell[0]},{cell[1]}"
            lines.append(f'<text x="{sx+cell_px/2}" y="{sy+cell_px/2+5}" text-anchor="middle" font-size="12">{label}</text>')

        # Saved SVG also carries the physical breadcrumb history (orange arrows).
        for b in self.breadcrumb_snapshot()[-160:]:
            if not isinstance(b, dict):
                continue
            fr = b.get("from"); to = b.get("to")
            if not (isinstance(fr, (list, tuple)) and len(fr) >= 2 and isinstance(to, (list, tuple)) and len(to) >= 2):
                continue
            fr = (int(fr[0]), int(fr[1])); to = (int(to[0]), int(to[1]))
            fx, fy = xy(fr); tx, ty = xy(to)
            lines.append(
                f'<line x1="{fx+cell_px/2}" y1="{fy+cell_px/2}" x2="{tx+cell_px/2}" y2="{ty+cell_px/2}" '
                'stroke="#d27b00" stroke-width="2" marker-end="url(#crumbArrow)" opacity="0.75"/>'
            )

        sx, sy = xy(self.current)
        cx, cy = sx + cell_px/2, sy + cell_px/2
        arrow = {0: (0, -18), 1: (18, 0), 2: (0, 18), 3: (-18, 0)}[self.heading]
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="none" stroke="black" stroke-width="2"/>')
        lines.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+arrow[0]}" y2="{cy+arrow[1]}" stroke="black" stroke-width="3"/>')
        lines.append('</g></svg>')
        return "\n".join(lines) + "\n"

    def save_map(self, final=False):
        try:
            cfg.MAP_DIR.mkdir(parents=True, exist_ok=True)
            payload = self.build_map_payload(final=final)
            atomic_write_text(cfg.MAP_LATEST_JSON, json.dumps(payload, ensure_ascii=False, indent=2))
            # Keep a tiny standalone chronological trail too.  Round 2 does NOT
            # require this file because the same trail is embedded in its bundle.
            self.save_breadcrumb()
            # V20: during the live run JSON is the crash-safe checkpoint.  ASCII/SVG
            # rendering is deferred until final save so DFS does not repeatedly spend
            # competition time regenerating presentation files after every move.
            if final:
                atomic_write_text(cfg.MAP_LATEST_ASCII, self.render_ascii_map())
                atomic_write_text(cfg.MAP_LATEST_SVG, self.render_svg_map())
            print(f"[MAP SAVE] visited={len(self.visited)} complete={bool(final and self.map_complete)} -> {cfg.MAP_LATEST_JSON}")
            return True
        except Exception as exc:
            self.fault("MAP SAVE", f"{type(exc).__name__}: {exc}", "keep mission state in memory")
            return False

    def _build_round1_fire_hints(self):
        hints = []
        for target in self.target_system.targets:
            if not isinstance(target, dict):
                continue
            if not str(target.get("fire_status") or "").startswith("FIRED_"):
                continue
            anchor = target.get("fire_anchor")
            if not isinstance(anchor, dict):
                anchor = {
                    "cell": target.get("source_cell"),
                    "heading_index": cfg.DIR_NAMES.index(target.get("heading")) if target.get("heading") in cfg.DIR_NAMES else 0,
                    "heading": target.get("heading"),
                    "sector": target.get("scan_sector"),
                    "detected_sweep_yaw_deg": target.get("detected_sweep_yaw_deg"),
                    "lock_yaw_deg": target.get("lock_yaw_deg"),
                    "range_mm": target.get("fire_range_mm"),
                    "target_estimated_grid_xy": target.get("estimated_grid_xy"),
                    "color": target.get("color"),
                    "shape": target.get("shape"),
                    "source_target_id": target.get("id"),
                    "fired_at": target.get("fired_at"),
                }
            h = dict(anchor)
            h["color"] = str(h.get("color") or target.get("color") or "").upper()
            h["shape"] = str(h.get("shape") or target.get("shape") or "").upper()
            h["fire_status"] = target.get("fire_status")
            h["fire_mode"] = target.get("fire_mode")
            h["fire_times"] = target.get("fire_times")
            if isinstance(h.get("cell"), (list, tuple)) and len(h.get("cell")) >= 2:
                hints.append(h)
        return hints

    def save_round1_attack_memory(self):
        """Atomically freeze map + only PROVEN fired positions for Round 2."""
        try:
            hints = self._build_round1_fire_hints()
            payload = {
                "schema": "robomaster_round1_attack_memory",
                "version": 1,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "map_complete": bool(self.map_complete),
                "start_pose_rule": {"cell": [int(self.root[0]), int(self.root[1])], "heading": "N"},
                "map": self.build_map_payload(final=self.map_complete),
                "selected_target_classes": [
                    {"color": c, "shape": sh} for c, sh in self.get_target_selection()
                ],
                "fire_hint_count": len(hints),
                "fire_hints": hints,
                "breadcrumb_count": len(self.breadcrumb_snapshot()),
                # Full trail also lives inside map.breadcrumb_trail; count here
                # makes snapshot inspection obvious without parsing the map first.
            }
            atomic_write_text(
                cfg.ROUND1_ATTACK_MEMORY_JSON,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            print(
                "[ROUND1 MEMORY] map_cells={} fired_hints={} complete={} -> {}".format(
                    len(self.visited), len(hints), self.map_complete, cfg.ROUND1_ATTACK_MEMORY_JSON
                )
            )
            return True
        except Exception as exc:
            self.fault(
                "ROUND1 MEMORY", f"{type(exc).__name__}: {exc}",
                "keep latest_map/latest_targets as fallback",
            )
            return False

    def save_manual_stop_checkpoint(self, reason="manual_stop"):
        """STOP NOW + persist everything Round 2 needs.

        This is intentionally callable directly from the GUI before the mission
        worker reaches cleanup().  Cleanup still saves again as a second layer.
        Round 1 writes: latest_map.json + latest_targets.json + the self-contained
        round1_attack_memory.json bundle used by Round 2.
        """
        self.running = False
        try:
            self.safe_stop()
        except Exception as exc:
            self.fault("STOP CHECKPOINT", f"safe_stop: {type(exc).__name__}: {exc}", "continue saving")

        map_ok = False
        try:
            map_ok = bool(self.save_map(final=bool(self.map_complete)))
        except Exception as exc:
            self.fault("STOP CHECKPOINT", f"map: {type(exc).__name__}: {exc}", "continue saving")

        target_ok = True
        try:
            self.target_system.save_targets()
        except Exception as exc:
            target_ok = False
            self.fault("STOP CHECKPOINT", f"targets: {type(exc).__name__}: {exc}", "continue saving")

        memory_ok = True
        if self.mission_mode == "ROUND1":
            try:
                memory_ok = bool(self.save_round1_attack_memory())
            except Exception as exc:
                memory_ok = False
                self.fault("STOP CHECKPOINT", f"round1 memory: {type(exc).__name__}: {exc}", "cleanup will retry")

        print(
            "[STOP CHECKPOINT] reason={} map={} targets={} round1_bundle={} bundle_path={}".format(
                reason, map_ok, target_ok, memory_ok if self.mission_mode == "ROUND1" else "N/A",
                cfg.ROUND1_ATTACK_MEMORY_JSON,
            )
        )
        return bool(map_ok and target_ok and memory_ok)

    def load_round1_attack_memory(self, path=None):
        """Load one self-contained Round-1 competition snapshot.

        The file contains BOTH the learned map and proven firing anchors, so
        Round 2 never needs latest_map.json/latest_targets.json separately.
        Passing a path from the GUI also makes that path the active Round-2
        source used again when START MISSION is pressed.
        """
        memory_path = Path(path) if path is not None else Path(self.round1_memory_path)
        self.round1_memory_path = memory_path
        try:
            payload = json.loads(memory_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "robomaster_round1_attack_memory":
                raise ValueError("wrong round1 memory schema")
            mp = payload.get("map") or {}

            # Rehydrate the generic arena frame saved by Round 1 before parsing
            # cells.  This makes Round 2 portable across arbitrary WxH profiles.
            saved_bounds = mp.get("field_bounds") or {}
            saved_root = mp.get("root", payload.get("start_pose_rule", {}).get("cell", list(cfg.ROOT_CELL)))
            if isinstance(saved_bounds, dict) and saved_bounds.get("width_cells") and saved_bounds.get("height_cells"):
                apply_arena_profile({
                    "name": "round1_snapshot",
                    "width_cells": saved_bounds.get("width_cells"),
                    "height_cells": saved_bounds.get("height_cells"),
                    "x_min": saved_bounds.get("x_min", cfg.GRID_X_MIN),
                    "y_min": saved_bounds.get("y_min", cfg.GRID_Y_MIN),
                    "start_cell": saved_root,
                    "boundary_guard": bool(saved_bounds.get("boundary_guard", True)),
                }, source="round1 snapshot")

            visited = {
                tuple(int(v) for v in c[:2])
                for c in mp.get("visited", [])
                if len(c) >= 2 and cell_in_field(tuple(int(v) for v in c[:2]))
            }
            if not visited:
                raise ValueError("round1 memory has no in-field visited cells")

            edges = {}
            for row in mp.get("edges", []):
                cell = row.get("cell")
                if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                    continue
                d = row.get("dir_index")
                if d is None and row.get("dir") in cfg.DIR_NAMES:
                    d = cfg.DIR_NAMES.index(row.get("dir"))
                if d is None:
                    continue
                c = (int(cell[0]), int(cell[1]))
                if not cell_in_field(c):
                    continue
                dd = int(d) % 4
                state = str(row.get("state") or "UNKNOWN")
                if state == "OPEN" and not cell_in_field(neighbor(c, dd)):
                    state = "BLOCKED"
                edges[(c, dd)] = state

            open_dirs = {}
            for key, dirs in (mp.get("open_dirs") or {}).items():
                try:
                    xs, ys = key.split(",", 1)
                    cell = (int(xs), int(ys))
                except Exception:
                    continue
                out = []
                for d in dirs or []:
                    if d in cfg.DIR_NAMES:
                        out.append(cfg.DIR_NAMES.index(d))
                if cell_in_field(cell):
                    open_dirs[cell] = [d for d in out if cell_in_field(neighbor(cell, d))]

            edge_travel = {}
            for row in mp.get("edge_travel_m", []):
                cell = row.get("cell")
                dname = row.get("dir")
                if not isinstance(cell, (list, tuple)) or len(cell) < 2 or dname not in cfg.DIR_NAMES:
                    continue
                try:
                    c = (int(cell[0]), int(cell[1]))
                    d = cfg.DIR_NAMES.index(dname)
                    if cell_in_field(c) and cell_in_field(neighbor(c, d)):
                        edge_travel[(c, d)] = float(row.get("distance_m"))
                except Exception:
                    pass

            self.visited = visited
            self.edge_state = edges
            self.open_dirs = open_dirs
            self.edge_travel_m = edge_travel
            loaded_trail = []
            for row in mp.get("breadcrumb_trail", []) or []:
                if not isinstance(row, dict):
                    continue
                fr = row.get("from"); to = row.get("to")
                if not (isinstance(fr, (list, tuple)) and len(fr) >= 2 and isinstance(to, (list, tuple)) and len(to) >= 2):
                    continue
                loaded_trail.append(dict(row))
            with self.breadcrumb_lock:
                self.breadcrumb_trail = loaded_trail
                self.breadcrumb_seq = max(
                    [int(r.get("seq", 0) or 0) for r in loaded_trail] or [0]
                )
            self.boundary_rejected_edges = set()
            for row in mp.get("boundary_rejected_edges", []) or []:
                try:
                    c = tuple(int(v) for v in row.get("cell", [])[:2])
                    raw_d = row.get("dir_index")
                    if raw_d is None and row.get("dir") in cfg.DIR_NAMES:
                        raw_d = cfg.DIR_NAMES.index(row.get("dir"))
                    if raw_d is None:
                        continue
                    d = int(raw_d) % 4
                    if len(c) == 2 and cell_in_field(c):
                        self.boundary_rejected_edges.add((c, d))
                except Exception:
                    pass

            self.cell_scan_mm = {}
            for key, value in (mp.get("cell_scan_mm") or {}).items():
                try:
                    xs, ys = key.split(",", 1)
                    self.cell_scan_mm[(int(xs), int(ys))] = value
                except Exception:
                    pass
            try:
                candidate_root = (int(saved_root[0]), int(saved_root[1]))
            except Exception:
                candidate_root = tuple(cfg.ROOT_CELL)
            self.root = candidate_root if cell_in_field(candidate_root) else tuple(cfg.ROOT_CELL)
            self.current = self.root
            self.heading = 0
            self.map_complete = bool(payload.get("map_complete", mp.get("complete", False)))
            self.round1_memory = payload
            self.round2_hints = [dict(h) for h in payload.get("fire_hints", []) if isinstance(h, dict)]
            self.round2_preview_plan = {}
            print(
                "[ROUND2 LOAD] cells={} hints={} breadcrumbs={} map_complete={} <- {}".format(
                    len(self.visited), len(self.round2_hints), len(self.breadcrumb_snapshot()),
                    self.map_complete, memory_path
                )
            )
            return True
        except FileNotFoundError:
            self.fault(
                "ROUND2 LOAD", f"missing {memory_path}",
                "run ROUND1 first",
            )
        except Exception as exc:
            self.fault(
                "ROUND2 LOAD", f"{type(exc).__name__}: {exc}",
                "do not move on an unverified map",
            )
        return False

    def _round2_hint_selected(self, hint):
        return self.target_allowed(hint.get("color"), hint.get("shape"))

    def _round2_route_distance(self, a, b):
        route = self.find_visited_route(tuple(a), tuple(b))
        return (float("inf"), None) if not route else (float(len(route) - 1), route)

    def _round2_anchor_order(self, start, anchors):
        """Order unique firing cells by known-path distance.

        Uses exact Held-Karp open-path optimization for up to 11 unique firing
        cells; above that, a nearest-neighbour pass avoids exponential startup
        time.  Distances come from the learned OPEN-edge graph, never Euclidean
        shortcuts through walls.
        """
        start = tuple(start)
        anchors = list(dict.fromkeys(tuple(a) for a in anchors if tuple(a) != start))
        if not anchors:
            return []

        points = [start] + anchors
        dist = {}
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                if i == j:
                    dist[(i, j)] = 0.0
                elif (j, i) in dist:
                    dist[(i, j)] = dist[(j, i)]
                else:
                    d, _ = self._round2_route_distance(a, b)
                    dist[(i, j)] = d

        reachable = [i for i in range(1, len(points)) if math.isfinite(dist[(0, i)])]
        if len(reachable) != len(anchors):
            bad = [points[i] for i in range(1, len(points)) if i not in reachable]
            self.fault("ROUND2 PLAN", f"unreachable fire anchors={bad}", "skip unreachable anchors")
            anchors = [points[i] for i in reachable]
            points = [start] + anchors
            if not anchors:
                return []
            # Recompute compact distance table after dropping unreachable anchors.
            dist = {}
            for i, a in enumerate(points):
                for j, b in enumerate(points):
                    if i == j:
                        dist[(i, j)] = 0.0
                    elif (j, i) in dist:
                        dist[(i, j)] = dist[(j, i)]
                    else:
                        d, _ = self._round2_route_distance(a, b)
                        dist[(i, j)] = d

        n = len(anchors)
        if n <= cfg.ROUND2_EXACT_ORDER_MAX_ANCHORS:
            # dp[(mask,last)] = (distance, previous_last) ; last is 0..n-1.
            dp = {}
            for j in range(n):
                d = dist[(0, j + 1)]
                if math.isfinite(d):
                    dp[(1 << j, j)] = (d, None)
            for mask in range(1, 1 << n):
                for last in range(n):
                    state = dp.get((mask, last))
                    if state is None:
                        continue
                    base = state[0]
                    for nxt in range(n):
                        bit = 1 << nxt
                        if mask & bit:
                            continue
                        step = dist[(last + 1, nxt + 1)]
                        if not math.isfinite(step):
                            continue
                        nm = mask | bit
                        nd = base + step
                        old = dp.get((nm, nxt))
                        if old is None or nd < old[0]:
                            dp[(nm, nxt)] = (nd, last)
            full = (1 << n) - 1
            ends = [(v[0], last) for (mask, last), v in dp.items() if mask == full]
            if ends:
                _, last = min(ends)
                order_idx = []
                mask = full
                while last is not None:
                    order_idx.append(last)
                    prev = dp[(mask, last)][1]
                    mask &= ~(1 << last)
                    last = prev
                order = [anchors[i] for i in reversed(order_idx)]
                print("[ROUND2 PLAN] exact shortest anchor order={}".format(order))
                return order

        # Deterministic nearest-neighbour fallback for unusually many anchors.
        remaining = set(anchors)
        cur = start
        order = []
        while remaining:
            candidates = []
            for a in remaining:
                d, _ = self._round2_route_distance(cur, a)
                if math.isfinite(d):
                    candidates.append((d, a))
            if not candidates:
                break
            _, nxt = min(candidates, key=lambda x: (x[0], x[1][1], x[1][0]))
            order.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        print("[ROUND2 PLAN] nearest-neighbour anchor order={}".format(order))
        return order

    def _route_physical_distance_m(self, route):
        """Estimate known-route travel using learned edge distances when available."""
        if not route or len(route) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(route, route[1:]):
            d = direction_between(tuple(a), tuple(b))
            if d is None:
                continue
            dist = self.edge_travel_m.get((tuple(a), d))
            if dist is None:
                dist = self.edge_travel_m.get((tuple(b), (d + 2) % 4))
            try:
                total += float(dist) if dist is not None else float(cfg.CELL_LENGTH_M)
            except Exception:
                total += float(cfg.CELL_LENGTH_M)
        return total

    def build_round2_preview_plan(self):
        """Build the exact route that Round 2 intends to use, without moving.

        Target anchors are the proven firing cells saved by Round 1.  Anchor
        order is globally shortest on the known OPEN-edge graph for <=11 unique
        anchors (Held-Karp), with the existing deterministic fallback above that.
        """
        selected = [h for h in self.round2_hints if self._round2_hint_selected(h)]
        hints_by_cell = {}
        rejected = []
        for h in selected:
            cell = h.get("cell")
            if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                rejected.append(dict(h, preview_failure="invalid_cell"))
                continue
            c = (int(cell[0]), int(cell[1]))
            if c not in self.visited:
                rejected.append(dict(h, preview_failure="cell_not_in_map"))
                continue
            hints_by_cell.setdefault(c, []).append(h)

        start = tuple(self.current)
        order = self._round2_anchor_order(start, list(hints_by_cell.keys()))
        if start in hints_by_cell:
            order = [start] + order

        route_cells = [start]
        segments = []
        cursor = start
        reachable_anchors = []
        for anchor_cell in order:
            anchor_cell = tuple(anchor_cell)
            if cursor == anchor_cell:
                route = [cursor]
            else:
                route = self.find_visited_route(cursor, anchor_cell)
            if not route:
                for h in hints_by_cell.get(anchor_cell, []):
                    rejected.append(dict(h, preview_failure="route_unavailable"))
                continue
            route = [tuple(c) for c in route]
            if len(route) > 1:
                route_cells.extend(route[1:])
            reachable_anchors.append(anchor_cell)
            segments.append({
                "from": tuple(cursor),
                "to": anchor_cell,
                "route": route,
                "steps": max(0, len(route) - 1),
                "distance_m": self._route_physical_distance_m(route),
                "hint_count": len(hints_by_cell.get(anchor_cell, [])),
            })
            cursor = anchor_cell

        plan = {
            "source_path": str(self.round1_memory_path),
            "start": start,
            "selected_hint_count": len(selected),
            "anchor_count": len(reachable_anchors),
            "anchor_order": reachable_anchors,
            "route_cells": route_cells,
            "segments": segments,
            "total_steps": sum(int(seg["steps"]) for seg in segments),
            "total_distance_m": sum(float(seg["distance_m"]) for seg in segments),
            "rejected": rejected,
            # Internal execution cache.  Kept in memory only; never serialized.
            "hints_by_cell": hints_by_cell,
        }
        self.round2_preview_plan = plan
        print(
            "[ROUND2 PREVIEW] anchors={} hints={} steps={} distance~{:.2f}m route={}".format(
                plan["anchor_count"], plan["selected_hint_count"],
                plan["total_steps"], plan["total_distance_m"], plan["route_cells"]
            )
        )
        return plan

    def _save_round2_result(self, started_at, attempts, successes, failed_hints, reason):
        try:
            payload = {
                "schema": "robomaster_round2_attack_result",
                "version": 1,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "reason": str(reason),
                "attempts": int(attempts),
                "successes": int(successes),
                "failed_hints": failed_hints,
                "current_cell": list(self.current),
                "selected_target_classes": [
                    {"color": c, "shape": sh} for c, sh in self.get_target_selection()
                ],
            }
            self.round2_result = payload
            atomic_write_text(cfg.ROUND2_RESULT_JSON, json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            self.fault("ROUND2 SAVE", f"{type(exc).__name__}: {exc}", "continue cleanup")

    def run_round2_attack(self):
        started_iso = datetime.now().isoformat(timespec="seconds")
        t0 = time.monotonic()
        if not self.load_round1_attack_memory(self.round1_memory_path):
            self.safe_stop()
            self._save_round2_result(started_iso, 0, 0, [], "round1 memory unavailable")
            return False

        plan = self.build_round2_preview_plan()
        selected = [h for h in self.round2_hints if self._round2_hint_selected(h)]
        if not selected:
            print("[ROUND2] no proven Round-1 fired hints match the current 16-target selection")
            self.safe_stop()
            self._save_round2_result(started_iso, 0, 0, [], "no selected fired hints")
            return True

        # Use exactly the same grouped hints + globally-shortest anchor order that
        # the GUI preview displayed before START.
        hints_by_cell = dict(plan.get("hints_by_cell") or {})
        order = [tuple(c) for c in (plan.get("anchor_order") or [])]

        attempts = 0
        successes = 0
        failed = []
        print(
            "\n[ROUND2] START shortest attack: hints={} anchor_cells={} hard_limit={:.0f}s".format(
                len(selected), len(hints_by_cell), cfg.ROUND2_HARD_LIMIT_SEC
            )
        )

        for anchor in order:
            if not self.running or not self.pose_trusted:
                break
            if time.monotonic() - t0 >= cfg.ROUND2_HARD_LIMIT_SEC:
                print("[ROUND2 DEADLINE] stopping before 5-minute limit")
                break

            if tuple(self.current) != tuple(anchor):
                route = self.find_visited_route(self.current, anchor)
                print("[ROUND2 ROUTE] {} -> {} route={}".format(self.current, anchor, route))
                if not route or not self.navigate_known_route(route):
                    self.fault("ROUND2 ROUTE", f"cannot reach anchor {anchor}", "skip this anchor")
                    for h in hints_by_cell.get(anchor, []):
                        failed.append(dict(h, failure="route_unavailable"))
                    continue

            # Service the proven shots at this anchor.  Sort by heading then sector
            # to reduce unnecessary chassis turns.
            local_hints = sorted(
                hints_by_cell.get(anchor, []),
                key=lambda h: (int(h.get("heading_index", 0)) % 4, str(h.get("sector") or "")),
            )
            for hint in local_hints:
                if time.monotonic() - t0 >= cfg.ROUND2_HARD_LIMIT_SEC:
                    break
                heading_idx = int(hint.get("heading_index", 0)) % 4
                if self.heading != heading_idx:
                    if not self.turn_to_direction(heading_idx):
                        failed.append(dict(hint, failure="turn_failed"))
                        continue

                ok = False
                for hint_attempt in range(1, cfg.ROUND2_MAX_HINT_ATTEMPTS + 1):
                    if time.monotonic() - t0 >= cfg.ROUND2_HARD_LIMIT_SEC:
                        break
                    attempts += 1
                    print(
                        "[ROUND2 ATTACK] attempt {}/{} {} {} @ cell={} heading={}".format(
                            hint_attempt, cfg.ROUND2_MAX_HINT_ATTEMPTS,
                            hint.get("color"), hint.get("shape"), anchor, cfg.DIR_NAMES[self.heading]
                        )
                    )
                    if self.target_system.scan_round2_hint(hint):
                        ok = True
                        successes += 1
                        break
                if not ok:
                    failed.append(dict(hint, failure="target_not_reacquired_or_not_fired"))

        self.safe_stop()
        elapsed = time.monotonic() - t0
        if elapsed >= cfg.ROUND2_HARD_LIMIT_SEC:
            reason = "deadline_guard"
        elif not self.running:
            reason = "stopped"
        elif failed:
            reason = "completed_with_failed_hints"
        else:
            reason = "completed"
        print(
            "[ROUND2 DONE] success={}/{} attempts={} elapsed={:.1f}s failed={}".format(
                successes, len(selected), attempts, elapsed, len(failed)
            )
        )
        self._save_round2_result(started_iso, attempts, successes, failed, reason)
        return not failed and successes >= len(selected)

    def run_selected_mission(self):
        if self.mission_mode == "ROUND2":
            return self.run_round2_attack()
        result = self.run_dfs()
        self.save_round1_attack_memory()
        return result

    def edge_key(self, cell, direction):
        return (tuple(cell), int(direction) % 4)

    def edge_is_deferred(self, cell, direction):
        return self.edge_key(cell, direction) in self.deferred_edges

    def defer_edge(self, cell, direction, reason, blocked=False):
        key = self.edge_key(cell, direction)
        self.deferred_edges.add(key)
        self.set_edge_state(cell, direction, "BLOCKED" if blocked else "DEFERRED")
        self.fault("DFS EDGE", f"{cell}->{cfg.DIR_NAMES[direction]}: {reason}", "skip edge and continue DFS")
        if direction in self.open_dirs.get(tuple(cell), []):
            self.open_dirs[tuple(cell)] = [d for d in self.open_dirs[tuple(cell)] if d != direction]
        if cfg.MAP_AUTOSAVE:
            self.save_map(final=False)

    def increment_edge_failure(self, cell, direction, reason):
        key = self.edge_key(cell, direction)
        n = self.edge_failures.get(key, 0) + 1
        self.edge_failures[key] = n
        self.fault("EDGE RETRY", f"{cell}->{cfg.DIR_NAMES[direction]} failure {n}/{cfg.EDGE_MAX_FAILURES}: {reason}", "rescan/retry")
        if n >= cfg.EDGE_MAX_FAILURES:
            self.defer_edge(cell, direction, f"repeated motion failure: {reason}", blocked=True)
            return False
        # Force a fresh topology scan before retrying this cell.
        self.open_dirs.pop(tuple(cell), None)
        return True

    def global_limits_ok(self):
        if len(self.visited) > cfg.MAX_VISITED_CELLS:
            self.enter_safe_pause(f"runaway guard: visited cells exceeded {cfg.MAX_VISITED_CELLS}")
            return False
        if self.mission_start_t is not None and time.monotonic() - self.mission_start_t > cfg.MAX_MISSION_SEC:
            if cfg.MISSION_WATCHDOG_HARD_STOP:
                self.enter_safe_pause(f"mission watchdog exceeded {cfg.MAX_MISSION_SEC}s")
                return False
            if not self._mission_watchdog_warned:
                self._mission_watchdog_warned = True
                self.fault(
                    "MISSION WATCHDOG",
                    f"elapsed time exceeded {cfg.MAX_MISSION_SEC}s",
                    "autosave checkpoint and continue (hard stop disabled)",
                )
                self.save_map(final=False)
        return True

    def try_explore_edge(self, cell, direction, next_cell):
        key = self.edge_key(cell, direction)
        if self.edge_is_deferred(cell, direction):
            return False

        # Authoritative configured-boundary check.  This catches any perimeter
        # opening generically, without knowing an exit coordinate in advance and
        # before the chassis moves even one centimetre.
        if cfg.FIELD_BOUNDARY_GUARD_ENABLED and not cell_in_field(next_cell):
            self.set_edge_state(cell, direction, "BLOCKED")
            if direction in self.open_dirs.get(tuple(cell), []):
                self.open_dirs[tuple(cell)] = [d for d in self.open_dirs[tuple(cell)] if d != direction]
            self.fault(
                "FIELD BOUNDARY",
                f"reject {tuple(cell)}->{tuple(next_cell)} dir={cfg.DIR_NAMES[int(direction)%4]}",
                "outside configured arena; continue DFS inside field",
            )
            if cfg.MAP_AUTOSAVE:
                self.save_map(final=False)
            return False

        if not self.turn_to_direction(direction):
            n = self.turn_failures.get(key, 0) + 1
            self.turn_failures[key] = n
            self.fault("DFS TURN", f"{cell}->{cfg.DIR_NAMES[direction]} failed {n}/{cfg.TURN_EDGE_MAX_FAILURES}", "defer if repeated")
            if n >= cfg.TURN_EDGE_MAX_FAILURES:
                self.defer_edge(cell, direction, "repeated closed-loop turn failure", blocked=False)
            return False

        result = self.move_one_cell(cell, direction, motion_profile="EXPLORE")
        if result == self.MOVE_ARRIVED:
            self.mark_traversed_open(cell, direction)
            if self.last_move_distance_m is not None:
                self.remember_edge_distance(cell, direction, self.last_move_distance_m)
            self.edge_failures.pop(key, None)
            self.turn_failures.pop(key, None)
            self.current = next_cell
            self.visited.add(next_cell)
            self.parent.setdefault(next_cell, cell)
            if self.last_stop_probe is not None:
                # The stop-probe is already a stationary L/F/R scan at the
                # accepted node.  Commit it now so a dead-end is mapped before
                # DFS immediately chooses to backtrack.
                if not self.commit_stop_probe_to_cell(next_cell, cell, self.last_stop_probe):
                    self.fault(
                        "NODE MAP",
                        f"probe at {next_cell} had too little valid ToF data",
                        "normal scan_cell will retry",
                    )
                    self.open_dirs.pop(tuple(next_cell), None)
            return True

        if result == self.MOVE_BLOCKED_RETURNED:
            self.current = cell
            self.increment_edge_failure(cell, direction, "translation aborted but source pose recovered")
            return False

        if result == self.MOVE_POSE_UNCERTAIN:
            self.enter_safe_pause(f"pose uncertain while traversing {cell}->{next_cell}")
            return False

        return False

    def find_visited_route(self, start, goal, excluded_edges=None):
        """BFS over already-visited OPEN edges, used only as backtrack fallback."""
        start, goal = tuple(start), tuple(goal)
        excluded = set(excluded_edges or ())
        if start == goal:
            return [start]
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            for d in range(4):
                nb = neighbor(cur, d)
                key = (cur, d)
                rev = (nb, (d + 2) % 4)
                if key in excluded or rev in excluded:
                    continue
                if nb not in self.visited:
                    continue
                if self.edge_state.get(key) != "OPEN":
                    continue
                if nb in prev:
                    continue
                prev[nb] = cur
                if nb == goal:
                    q.clear()
                    break
                q.append(nb)
        if goal not in prev:
            return None
        route = []
        cur = goal
        while cur is not None:
            route.append(cur)
            cur = prev[cur]
        return list(reversed(route))

    def navigate_known_route(self, route):
        """Traverse a visited-cell route without changing DFS parent links."""
        if not route or tuple(route[0]) != tuple(self.current):
            return False
        for src, dst in zip(route, route[1:]):
            d = direction_between(src, dst)
            if d is None:
                return False
            if not self.turn_to_direction(d):
                return False
            result = self.move_one_cell(src, d, motion_profile="KNOWN_FAST")
            if result != self.MOVE_ARRIVED:
                if result == self.MOVE_BLOCKED_RETURNED:
                    self.current = src
                return False
            self.current = dst
            self.mark_traversed_open(src, d)
            if self.last_move_distance_m is not None:
                self.remember_edge_distance(src, d, self.last_move_distance_m)
        return True

    def find_fastest_known_route(self, start, goal):
        """Dijkstra over confirmed visited OPEN edges, including turn cost."""
        import heapq
        start = tuple(start); goal = tuple(goal)
        if start == goal:
            return [start]
        start_h = int(self.heading) % 4
        pq = [(0.0, start, start_h)]
        best = {(start, start_h): 0.0}
        prev = {}
        goal_state = None
        while pq:
            cost, cell, h = heapq.heappop(pq)
            state = (cell, h)
            if cost > best.get(state, float("inf")) + 1e-9:
                continue
            if cell == goal:
                goal_state = state
                break
            for d in range(4):
                if self.edge_state.get((cell, d)) != "OPEN":
                    continue
                nb = neighbor(cell, d)
                if nb not in self.visited:
                    continue
                edge_m = self.remembered_edge_distance(cell, d)
                if edge_m is None:
                    edge_m = cfg.CELL_LENGTH_M
                q = abs((d - h) % 4); q = min(q, 4-q)
                step = float(edge_m) / max(0.10, cfg.DFS_KNOWN_SPEED_MPS) + 0.55*float(q)
                ns = (nb, d)
                nc = cost + step
                if nc + 1e-9 < best.get(ns, float("inf")):
                    best[ns] = nc
                    prev[ns] = state
                    heapq.heappush(pq, (nc, nb, d))
        if goal_state is None:
            return None
        states=[]; cur=goal_state
        while True:
            states.append(cur)
            if cur == (start, start_h): break
            cur=prev.get(cur)
            if cur is None: return None
        states.reverse()
        route=[]
        for cell,_h in states:
            if not route or route[-1] != cell:
                route.append(cell)
        return route

    def _cell_has_unvisited_open_neighbor(self, cell):
        """Return True when a scanned visited cell still borders unexplored OPEN space."""
        c = tuple(cell)
        for d in self.open_dirs.get(c, []):
            if self.edge_is_deferred(c, d):
                continue
            if self.edge_state.get((c, d)) != "OPEN":
                continue
            if neighbor(c, d) not in self.visited:
                return True
        return False

    def _route_time_score(self, route):
        """Cheap travel-time estimate for a known route, including turn cost."""
        if not route or len(route) < 2:
            return 0.0
        h = int(self.heading) % 4
        score = 0.0
        for a, b in zip(route, route[1:]):
            d = direction_between(a, b)
            if d is None:
                return float("inf")
            q = abs((int(d) - h) % 4)
            q = min(q, 4 - q)
            # A known 60 cm edge is ~1.1-1.5 s in the field.  A 90 deg turn has
            # meaningful overhead, so equal-hop routes prefer fewer turns.
            edge_m = self.remembered_edge_distance(a, d)
            if edge_m is None or not math.isfinite(float(edge_m)):
                edge_m = cfg.CELL_LENGTH_M
            score += float(edge_m) / max(0.10, cfg.DFS_KNOWN_SPEED_MPS)
            score += 0.55 * float(q)
            h = int(d)
        return score

    def find_best_frontier_route(self):
        """Fastest known OPEN route from current cell to any remaining frontier.

        Discovery still uses the existing DFS edge policy.  Only the *return trip*
        changes: instead of blindly following DFS parent links, jump through the
        already-proven graph to the nearest useful visited cell.
        """
        start = tuple(self.current)
        best = None
        for c in list(self.visited):
            c = tuple(c)
            if c == start or not self._cell_has_unvisited_open_neighbor(c):
                continue
            route = self.find_fastest_known_route(start, c)
            if not route or len(route) < 2:
                continue
            score = self._route_time_score(route)
            key = (score, len(route), c)
            if best is None or key < best[0]:
                best = (key, route)
        return None if best is None else best[1]

    def shortcut_to_frontier(self):
        route = self.find_best_frontier_route()
        if not route:
            return False
        target = tuple(route[-1])
        print(
            f"\n[FRONTIER SHORTCUT] {tuple(self.current)} -> {target} "
            f"known_route={route} score={self._route_time_score(route):.2f}"
        )
        if not self.navigate_known_route(route):
            self.fault(
                "FRONTIER SHORTCUT",
                f"known route failed before frontier {target}",
                "fall back to classic DFS parent backtrack",
            )
            return False
        self.current = target
        return True

    def backtrack_to_parent(self, cell, parent):
        d = direction_between(cell, parent)
        if d is None:
            self.enter_safe_pause(f"invalid DFS parent relation {cell}->{parent}")
            return False

        for attempt in range(1, cfg.EDGE_MAX_FAILURES + 2):
            if not self.turn_to_direction(d):
                self.fault("BACKTRACK TURN", f"{cell}->{parent} attempt {attempt}", "retry")
                continue
            result = self.move_one_cell(cell, d, motion_profile="BACKTRACK_FAST")
            if result == self.MOVE_ARRIVED:
                self.current = parent
                self.mark_traversed_open(cell, d)
                if self.last_move_distance_m is not None:
                    # Keep the shorter of two close measurements only if they are
                    # reasonably consistent; otherwise retain the original edge
                    # anchor learned on exploration.
                    old_d = self.remembered_edge_distance(cell, d)
                    new_d = self.last_move_distance_m
                    if old_d is None or abs(old_d - new_d) <= 0.10:
                        self.remember_edge_distance(cell, d, (new_d if old_d is None else 0.5 * (old_d + new_d)))
                return True
            if result == self.MOVE_BLOCKED_RETURNED:
                self.current = cell
                self.fault("BACKTRACK MOVE", f"{cell}->{parent} attempt {attempt} blocked", "retry known parent edge")
                time.sleep(0.20)
                continue
            if result == self.MOVE_POSE_UNCERTAIN:
                break

        failed_dir = direction_between(cell, parent)
        excluded = set()
        if failed_dir is not None:
            excluded.add((tuple(cell), failed_dir))
            excluded.add((tuple(parent), (failed_dir + 2) % 4))
        route = self.find_visited_route(cell, parent, excluded_edges=excluded)
        if route and len(route) > 2:
            self.fault(
                "BACKTRACK REROUTE",
                f"direct parent edge unavailable; alternate visited route={route}",
                "navigate alternate OPEN path",
            )
            if self.navigate_known_route(route):
                self.current = parent
                return True

        self.enter_safe_pause(f"known DFS parent edge could not be traversed safely: {cell}->{parent}")
        return False

    def reconstruct_dfs_stack(self):
        """Rebuild the root->current DFS ancestry after a contained Python fault."""
        cur = tuple(self.current)
        chain = []
        seen = set()
        while cur is not None:
            if cur in seen:
                return None
            seen.add(cur)
            chain.append(cur)
            cur = self.parent.get(cur)
        chain.reverse()
        if not chain or chain[0] != self.root:
            return None
        return chain

    def run_dfs(self):
        """Mission-level exception containment with bounded in-place resume."""
        resume = False
        for attempt in range(cfg.DFS_EXCEPTION_RESTARTS + 1):
            try:
                return self._run_dfs_impl(resume=resume)
            except Exception as exc:
                self.safe_stop()
                self.fault(
                    "DFS UNEXPECTED",
                    f"attempt {attempt + 1}/{cfg.DFS_EXCEPTION_RESTARTS + 1}: {type(exc).__name__}: {exc}",
                    "recover node telemetry/cardinal and resume DFS state",
                )
                if attempt >= cfg.DFS_EXCEPTION_RESTARTS:
                    break
                try:
                    telemetry_ok = self.recover_telemetry(
                        require_position=True, require_attitude=True,
                        reason="DFS top-level exception",
                    )
                    cardinal = self.recover_to_nearest_cardinal() if telemetry_ok else None
                    if not telemetry_ok or cardinal is None:
                        break
                    stack = self.reconstruct_dfs_stack()
                    if stack is None:
                        break
                    self.save_map(final=False)
                    resume = True
                    self.fault(
                        "DFS RESUME",
                        f"recovered at logical cell={self.current} heading={cfg.DIR_NAMES[self.heading]}",
                        "resume existing stack/map",
                    )
                except Exception as recover_exc:
                    self.fault(
                        "DFS RESUME",
                        f"{type(recover_exc).__name__}: {recover_exc}",
                        "cannot prove resumable node state",
                    )
                    break

        self.enter_safe_pause("DFS encountered repeated unexpected exceptions and could not resume safely")
        return False

    def _run_dfs_impl(self, resume=False):
        if not resume:
            self.visited = {self.root}
            self.parent = {self.root: None}
            self.current = self.root
            self.map_complete = False
            stack = [self.root]
            print("\n[DFS] START MAP-ONLY")
        else:
            stack = self.reconstruct_dfs_stack()
            if not stack:
                raise ValueError("cannot reconstruct DFS ancestry for resume")
            print(f"\n[DFS] RESUME cell={self.current} stack={stack}")

        while self.running and self.pose_trusted and stack:
            if not self.global_limits_ok():
                break

            cell = stack[-1]
            self.current = cell

            if cell not in self.open_dirs:
                self.scan_cell(cell)
                if not self.pose_trusted:
                    break
                if cfg.MAP_AUTOSAVE:
                    self.save_map(final=False)

            # A verified wide-open exit candidate is NEVER a reason to finish
            # exploration.  Physically return to the parent, blacklist only that
            # aperture, pop the provisional outside cell, and continue DFS.
            if tuple(cell) in self.fake_exit_candidates:
                if self._reject_fake_exit_and_return(cell):
                    if stack and tuple(stack[-1]) == tuple(cell):
                        stack.pop()
                    if not stack or tuple(stack[-1]) != tuple(self.current):
                        rebuilt = self.reconstruct_dfs_stack()
                        stack = rebuilt if rebuilt else [tuple(self.current)]
                    continue
                break

            next_dir = None
            next_cell = None
            for d in self.open_dirs.get(cell, []):
                if self.edge_is_deferred(cell, d):
                    continue
                nb = neighbor(cell, d)
                if cfg.FIELD_BOUNDARY_GUARD_ENABLED and not cell_in_field(nb):
                    self.set_edge_state(cell, d, "BLOCKED")
                    continue
                if nb not in self.visited:
                    next_dir = d
                    next_cell = nb
                    break

            if next_cell is not None:
                print(f"\n[DFS] EXPLORE {cell} -> {next_cell} dir={cfg.DIR_NAMES[next_dir]}")
                if self.try_explore_edge(cell, next_dir, next_cell):
                    stack.append(next_cell)
                    if cfg.MAP_AUTOSAVE:
                        self.save_map(final=False)
                # On a recoverable failure we remain at `cell`; the next loop
                # rescans or selects another non-deferred edge.
                continue

            # Before backtracking, give unresolved UNKNOWN edges one bounded
            # stationary re-scan.  This catches a temporarily unhappy gimbal/ToF
            # instead of silently abandoning a physically open branch.
            unknown_here = [
                d for d in range(4)
                if self.edge_state.get((tuple(cell), d)) == "UNKNOWN"
            ]
            retry_count = self.unknown_rescan_counts.get(tuple(cell), 0)
            if unknown_here and retry_count < 1:
                self.unknown_rescan_counts[tuple(cell)] = retry_count + 1
                print(
                    f"[DFS UNKNOWN RECOVER] cell={cell} "
                    f"dirs={[cfg.DIR_NAMES[d] for d in unknown_here]} -> rescan before backtrack"
                )
                self.open_dirs.pop(tuple(cell), None)
                self.scan_cell(cell)
                if cfg.MAP_AUTOSAVE:
                    self.save_map(final=False)
                continue

            # V20.6: no local branch remains.  Do NOT automatically walk the
            # entire DFS parent chain.  The discovered maze is already a graph, so
            # route through confirmed OPEN visited cells to the nearest remaining
            # frontier (e.g. 22->32->33->34 instead of 22->21->20->30->31->32...).
            # This changes only relocation; new-edge discovery remains DFS-safe.
            if self.shortcut_to_frontier():
                rebuilt = self.reconstruct_dfs_stack()
                if rebuilt:
                    stack = rebuilt
                else:
                    # Parent links remain first-visit ancestry, so this should be
                    # rare.  Keep current logical node usable even if ancestry was
                    # damaged by a prior recovery.
                    stack = [tuple(self.current)]
                if cfg.MAP_AUTOSAVE:
                    self.save_map(final=False)
                continue

            # V20.7: no useful frontier remains anywhere.  The exploration work is
            # done; do NOT unwind the DFS parent chain edge-by-edge.  Go home over
            # the fastest confirmed OPEN route in the map.
            if tuple(cell) != tuple(self.root):
                home_route = self.find_fastest_known_route(cell, self.root)
                if home_route and len(home_route) >= 2:
                    print(
                        f"\n[FAST HOME] exploration frontier exhausted: {cell} -> {self.root} "
                        f"route={home_route}"
                    )
                    if self.navigate_known_route(home_route):
                        self.current = self.root
                        stack = []
                        break
                    self.fault(
                        "FAST HOME", "shortest known route failed",
                        "fall back to classic DFS parent backtrack",
                    )

            # Fallback only if shortest-home cannot be executed safely.
            parent = self.parent.get(cell)
            if parent is None:
                stack.pop()
                break

            print(f"\n[DFS] BACKTRACK {cell} -> {parent}")
            if not self.backtrack_to_parent(cell, parent):
                break
            stack.pop()
            if cfg.MAP_AUTOSAVE:
                self.save_map(final=False)

        self.safe_stop()

        if self.pose_trusted and not stack and self.current == self.root:
            unknown_count = sum(1 for s in self.edge_state.values() if s == "UNKNOWN")
            deferred_count = len(self.deferred_edges)
            self.map_complete = unknown_count == 0 and deferred_count == 0
            print("\n[DFS] FINISHED AT ROOT")
            print(f"[DFS] visited={len(self.visited)} unknown_edges={unknown_count} deferred_edges={deferred_count}")
            if not self.map_complete:
                print("[DFS] topology pass ended safely, but map is PARTIAL because some edges were unknown/deferred.")
        else:
            self.map_complete = False
            print("\n[DFS] stopped with a PARTIAL map.")

        self.save_map(final=self.map_complete)
        if self.mission_mode == "ROUND1":
            self.save_round1_attack_memory()
        print(self.render_ascii_map())
        return self.map_complete
