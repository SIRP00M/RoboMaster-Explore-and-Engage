"""Camera frames, lighting, color/shape detection and temporal tracking."""

from . import config as cfg
import math
import statistics
import threading
import time
from collections import deque
from .sensors import clamp
from .targeting import TargetingMixin

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


class TargetVisionSubsystem(TargetingMixin):
    """Camera processing and target-service state."""


    def __init__(self, owner):
        self.owner = owner
        self.running = False
        self.available = False
        self.stream_started = False
        self.preview_enabled = bool(cfg.TARGET_PREVIEW_ENABLED)
        self.thread = None
        self.lock = threading.Lock()
        self.roi_lock = threading.Lock()
        self.dynamic_roi = None
        self.latest_frame = None
        self.latest_frame_t = 0.0
        self.latest_detections = []
        self.latest_color_blobs = []
        self.history = deque(maxlen=160)
        self.frame_seq = 0
        self.foam_profile = None
        self.foam_profile_t = 0.0
        self.foam_coverage = 0.0
        self.foam_components = []
        self.scanned_cells = set()
        self.last_scan_t = {}
        self.targets = []
        self.target_seq = 0
        self.fired_target_ids = set()
        self.status = "VISION OFF"
        self.last_fire_event = "NONE"
        self.last_aim_solution = {}
        self._clahe = None
        if cv2 is not None:
            try:
                self._clahe = cv2.createCLAHE(
                    clipLimit=float(cfg.TARGET_CLAHE_CLIP_LIMIT),
                    tileGridSize=(int(cfg.TARGET_CLAHE_GRID), int(cfg.TARGET_CLAHE_GRID)),
                )
            except Exception:
                self._clahe = None

    def start(self):
        if not cfg.TARGET_VISION_ENABLED:
            self.status = "VISION DISABLED"
            return False
        if cv2 is None or np is None:
            self.owner.fault("VISION", "opencv-python or numpy unavailable", "continue DFS without targets")
            return False
        if self.owner.camera is None:
            self.owner.fault("VISION", "RoboMaster camera unavailable", "continue DFS without targets")
            return False
        try:
            try:
                from robomaster import camera as rm_camera
                resolution = getattr(rm_camera, "STREAM_360P", cfg.TARGET_CAMERA_RESOLUTION)
            except Exception:
                resolution = cfg.TARGET_CAMERA_RESOLUTION
            self.owner.camera.start_video_stream(display=False, resolution=resolution)
            self.stream_started = True
        except Exception as exc:
            self.owner.fault("VISION STREAM", f"{type(exc).__name__}: {exc}", "continue DFS without targets")
            return False

        self.running = True
        self.available = True
        self.status = "VISION READY"
        self.thread = threading.Thread(target=self._vision_loop, name="TargetVision", daemon=True)
        self.thread.start()
        print("[VISION] Lab-CLAHE + HSV/LAB target detector ready")
        print(
            f"[VISION] Foam-board top gate={'ON' if cfg.TARGET_FOAM_GATE_ENABLED else 'OFF'}; "
            f"real fire={'ON' if cfg.TARGET_REAL_FIRE_ENABLED else 'DRY'}; range<={cfg.TARGET_FIRE_MAX_RANGE_MM:.0f} mm"
        )
        return True

    def stop(self):
        self.running = False
        self.available = False
        self.stream_started = False
        self._stop_gimbal_velocity()
        if self.owner.camera is not None:
            try:
                self.owner.camera.stop_video_stream()
            except Exception:
                pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            if cv2 is not None:
                cv2.destroyWindow(cfg.TARGET_WINDOW_NAME)
        except Exception:
            pass

    def _vision_loop(self):
        print("[VISION] frame thread started")
        while self.owner.running and self.running and self.stream_started:
            try:
                frame = self.owner.camera.read_cv2_image(strategy="newest", timeout=0.6)
            except Exception as exc:
                if self.running:
                    self.owner.fault("VISION FRAME", f"{type(exc).__name__}: {exc}", "retry frame")
                time.sleep(0.08)
                continue
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            try:
                foam_profile, coverage, components = self._compute_foam_wall_profile(frame)
                detections, blobs = self._detect_targets(frame, foam_profile, coverage)
            except Exception as exc:
                self.owner.fault("VISION DETECT", f"{type(exc).__name__}: {exc}", "drop frame / keep DFS alive")
                time.sleep(0.02)
                continue

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_frame_t = now
                self.latest_detections = [dict(d) for d in detections]
                self.latest_color_blobs = [dict(d) for d in blobs]
                self.foam_profile = None if foam_profile is None else foam_profile.copy()
                self.foam_profile_t = now
                self.foam_coverage = float(coverage)
                self.foam_components = [dict(c) for c in components]
                self.frame_seq += 1
                self.history.append((now, self.frame_seq, [dict(d) for d in detections]))
                cutoff = now - cfg.TARGET_HISTORY_SEC
                while self.history and self.history[0][0] < cutoff:
                    self.history.popleft()

            if self.preview_enabled:
                try:
                    annotated = self._draw_overlay(frame, detections, foam_profile, coverage)
                    cv2.imshow(cfg.TARGET_WINDOW_NAME, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self.preview_enabled = False
                except Exception as exc:
                    self.owner.fault("VISION PREVIEW", f"{type(exc).__name__}: {exc}", "disable preview only")
                    self.preview_enabled = False

        print("[VISION] frame thread stopped")

    def _normalize_lighting(self, frame):
        if self._clahe is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_norm = self._clahe.apply(l_chan)
        return cv2.cvtColor(cv2.merge((l_norm, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _hue_distance(h1, h2):
        raw = abs(float(h1) - float(h2))
        return min(raw, 180.0 - raw)

    def _color_confidence(self, color_name, median_hsv, median_lab):
        h, s, v = median_hsv
        hue_score = max(0.0, 1.0 - self._hue_distance(h, cfg.TARGET_HUE_CENTERS[color_name]) / 30.0)
        saturation_score = clamp((s - 45.0) / 100.0, 0.0, 1.0)
        brightness_score = clamp((v - 25.0) / 90.0, 0.0, 1.0)
        _l, a, b = median_lab
        if color_name == "GREEN":
            lab_score = clamp((128.0 - a) / 45.0 + 0.25, 0.0, 1.0)
        elif color_name == "RED":
            lab_score = clamp((a - 128.0) / 45.0 + 0.25, 0.0, 1.0)
        elif color_name == "YELLOW":
            lab_score = clamp((b - 128.0) / 55.0 + 0.20, 0.0, 1.0)
        elif color_name == "BLUE":
            lab_score = clamp((128.0 - b) / 55.0 + 0.20, 0.0, 1.0)
        else:
            lab_score = 0.5
        return float(0.50*hue_score + 0.22*saturation_score + 0.08*brightness_score + 0.20*lab_score)

    @staticmethod
    def _inside_statistics(contour, hsv, lab):
        region = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, thickness=-1)
        ys, xs = np.where(region > 0)
        if len(xs) == 0:
            return (0.0, 0.0, 0.0), (0.0, 128.0, 128.0)
        hsv_pixels = hsv[ys, xs]
        lab_pixels = lab[ys, xs]
        return (
            tuple(float(v) for v in np.median(hsv_pixels, axis=0)),
            tuple(float(v) for v in np.median(lab_pixels, axis=0)),
        )

    def _active_roi(self):
        with self.roi_lock:
            if self.dynamic_roi is not None:
                return tuple(self.dynamic_roi)
        return tuple(cfg.TARGET_SEARCH_ROI)

    def _set_focus_roi(self, candidate):
        if not isinstance(candidate, dict):
            return
        try:
            cx, cy = [float(v) for v in candidate.get("center_norm", (0.5, 0.5))]
            _x, _y, bw, bh = [float(v) for v in candidate.get("bbox_norm", (0, 0, 0, 0))]
        except Exception:
            return
        base_x1, base_y1, base_x2, base_y2 = cfg.TARGET_SEARCH_ROI
        rw = clamp(max(cfg.TARGET_DYNAMIC_ROI_MIN_W, bw*cfg.TARGET_DYNAMIC_ROI_BBOX_SCALE), cfg.TARGET_DYNAMIC_ROI_MIN_W, cfg.TARGET_DYNAMIC_ROI_MAX_W)
        rh = clamp(max(cfg.TARGET_DYNAMIC_ROI_MIN_H, bh*cfg.TARGET_DYNAMIC_ROI_BBOX_SCALE), cfg.TARGET_DYNAMIC_ROI_MIN_H, cfg.TARGET_DYNAMIC_ROI_MAX_H)
        rw = min(rw, base_x2-base_x1)
        rh = min(rh, base_y2-base_y1)
        x1 = clamp(cx-rw*0.5, base_x1, base_x2-rw)
        y1 = clamp(cy-rh*0.5, base_y1, base_y2-rh)
        with self.roi_lock:
            self.dynamic_roi = (x1, y1, x1+rw, y1+rh)

    def _clear_focus_roi(self):
        with self.roi_lock:
            self.dynamic_roi = None

    def _roi_px(self, frame):
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self._active_roi()
        x1 = max(0, min(w-1, int(round(rx1*w))))
        x2 = max(x1+1, min(w, int(round(rx2*w))))
        y1 = max(0, min(h-1, int(round(ry1*h))))
        y2 = max(y1+1, min(h, int(round(ry2*h))))
        return x1, y1, x2, y2

    def _compute_foam_wall_profile(self, frame):
        if not cfg.TARGET_FOAM_GATE_ENABLED or frame is None:
            return None, 0.0, []
        frame_h, frame_w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.asarray(cfg.TARGET_FOAM_HSV_LOW, dtype=np.uint8),
            np.asarray(cfg.TARGET_FOAM_HSV_HIGH, dtype=np.uint8),
        )
        ok = max(1, int(cfg.TARGET_FOAM_OPEN_KERNEL)); ck = max(1, int(cfg.TARGET_FOAM_CLOSE_KERNEL))
        if ok % 2 == 0: ok += 1
        if ck % 2 == 0: ck += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ok,ok), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ck,ck), np.uint8))
        found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found)==2 else found[1]
        profile = np.full(frame_w, np.nan, dtype=np.float32)
        components = []
        min_bottom = frame_h * cfg.TARGET_FOAM_COMPONENT_MIN_BOTTOM_FRAC
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x,y,bw,bh = cv2.boundingRect(contour)
            if area < cfg.TARGET_FOAM_MIN_COMPONENT_AREA_PX or bw < cfg.TARGET_FOAM_MIN_COMPONENT_WIDTH_PX or bh < cfg.TARGET_FOAM_MIN_COMPONENT_HEIGHT_PX:
                continue
            if y+bh < min_bottom:
                continue
            components.append({"bbox_px":[int(x),int(y),int(bw),int(bh)], "area_px":area})
            local = np.zeros((bh,bw), dtype=np.uint8)
            shifted = contour.copy(); shifted[:,:,0] -= x; shifted[:,:,1] -= y
            cv2.drawContours(local, [shifted], -1, 255, thickness=-1)
            for lx in range(bw):
                ys = np.flatnonzero(local[:,lx])
                if ys.size:
                    px = x+lx; top_y = float(y+int(ys[0]))
                    if not math.isfinite(float(profile[px])) or top_y < float(profile[px]):
                        profile[px] = top_y
        valid = np.flatnonzero(np.isfinite(profile))
        max_gap = int(cfg.TARGET_FOAM_PROFILE_MAX_INTERP_GAP_PX)
        if valid.size >= 2 and max_gap > 0:
            for li,ri in zip(valid[:-1], valid[1:]):
                gap = int(ri-li-1)
                if 0 < gap <= max_gap:
                    profile[li:ri+1] = np.linspace(float(profile[li]), float(profile[ri]), int(ri-li+1), dtype=np.float32)
        radius = int(cfg.TARGET_FOAM_PROFILE_MEDIAN_RADIUS_PX)
        if radius > 0 and np.isfinite(profile).any():
            src = profile.copy(); smooth = profile.copy()
            for px in np.flatnonzero(np.isfinite(src)):
                vals = src[max(0,px-radius):min(frame_w,px+radius+1)]
                vals = vals[np.isfinite(vals)]
                if vals.size: smooth[px] = float(np.median(vals))
            profile = smooth
        rx1 = max(0,min(frame_w-1,int(cfg.TARGET_SEARCH_ROI[0]*frame_w)))
        rx2 = max(rx1+1,min(frame_w,int(cfg.TARGET_SEARCH_ROI[2]*frame_w)))
        rs = profile[rx1:rx2]
        coverage = float(np.count_nonzero(np.isfinite(rs))) / float(max(1,rs.size))
        return profile, coverage, components

    @staticmethod
    def _foam_wall_y(profile, x_px):
        if profile is None:
            return None
        x = int(round(float(x_px)))
        if x < 0 or x >= len(profile):
            return None
        r = int(cfg.TARGET_FOAM_PROFILE_SAMPLE_RADIUS_PX)
        vals = np.asarray(profile[max(0,x-r):min(len(profile),x+r+1)], dtype=float)
        vals = vals[np.isfinite(vals)]
        return None if vals.size == 0 else float(np.median(vals))

    def _foam_gate(self, frame_w, frame_h, cx, cy, bbox_px, profile, profile_t=None, coverage=None):
        if not cfg.TARGET_FOAM_GATE_ENABLED:
            return True, None, "disabled", 1.0
        if profile is None:
            if cfg.TARGET_FOAM_FAIL_CLOSED:
                return False, None, "profile-missing", 0.0
            return True, None, "profile-missing-fail-open", 1.0
        if coverage is not None and coverage < cfg.TARGET_FOAM_MIN_ROI_COVERAGE:
            if cfg.TARGET_FOAM_FAIL_CLOSED:
                return False, None, "profile-low-coverage", 0.0
            return True, None, "profile-low-coverage-fail-open", 1.0
        wall_y = self._foam_wall_y(profile, cx)
        if wall_y is None:
            if cfg.TARGET_FOAM_FAIL_CLOSED:
                return False, None, "local-profile-missing", 0.0
            return True, None, "local-profile-missing-fail-open", 1.0
        x,y,bw,bh = [float(v) for v in bbox_px]
        bh = max(1.0,bh)
        cutoff = wall_y + cfg.TARGET_FOAM_CENTER_MARGIN_PX
        bbox_bottom = y+bh
        below_h = max(0.0, bbox_bottom-max(y,cutoff))
        below_frac = clamp(below_h/bh,0.0,1.0)
        if float(cy) < cutoff:
            return False, wall_y, "centre-above-foam", below_frac
        if below_frac < cfg.TARGET_FOAM_MIN_BBOX_BELOW_FRAC:
            return False, wall_y, "bbox-cut-by-foam-edge", below_frac
        return True, wall_y, "foam-pass", below_frac

    def _classify_shape(self, contour):
        area = float(cv2.contourArea(contour)); perimeter = float(cv2.arcLength(contour, True))
        if area <= 1.0 or perimeter <= 1.0:
            return None
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area/hull_area if hull_area > 1e-6 else 0.0
        if solidity < cfg.TARGET_MIN_SOLIDITY:
            return None
        approx = cv2.approxPolyDP(contour, cfg.TARGET_POLY_EPS_FRAC*perimeter, True)
        x,y,w,h = cv2.boundingRect(contour)
        fill = area/float(max(1,w*h))
        circularity = 4.0*math.pi*area/max(1e-9, perimeter*perimeter)
        shape = None; quality = 0.0
        if len(approx)==4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1,2).astype(float); cosines=[]
            for i in range(4):
                v1=pts[(i-1)%4]-pts[i]; v2=pts[(i+1)%4]-pts[i]
                den=float(np.linalg.norm(v1)*np.linalg.norm(v2))
                cosines.append(1.0 if den<=1e-9 else abs(float(np.dot(v1,v2))/den))
            max_cos=max(cosines) if cosines else 1.0
            rect=cv2.minAreaRect(contour); rw,rh=rect[1]
            if rw>1.0 and rh>1.0 and fill>=cfg.TARGET_RECT_MIN_FILL and max_cos<=cfg.TARGET_RECT_MAX_CORNER_COS:
                aspect_rot=max(rw,rh)/max(1e-6,min(rw,rh))
                if cfg.TARGET_SQUARE_ASPECT_MIN <= aspect_rot <= cfg.TARGET_SQUARE_ASPECT_MAX:
                    shape="SQUARE"
                else:
                    axis=float(w)/max(1.0,float(h))
                    if axis>=cfg.TARGET_RECT_ASPECT_MIN: shape="RECT_HORIZONTAL"
                    elif (1.0/max(axis,1e-6))>=cfg.TARGET_RECT_ASPECT_MIN: shape="RECT_VERTICAL"
                quality=0.45*solidity+0.35*min(1.0,fill)+0.20*(1.0-min(1.0,max_cos))
        elif len(approx)>=5 and circularity>=cfg.TARGET_CIRCLE_MIN_CIRCULARITY:
            axis=float(w)/max(1.0,float(h))
            if cfg.TARGET_CIRCLE_ASPECT_MIN <= axis <= cfg.TARGET_CIRCLE_ASPECT_MAX:
                shape="CIRCLE"; quality=0.55*solidity+0.45*min(1.0,circularity)
        if shape is None:
            return None
        return {"shape":shape,"quality":float(quality),"solidity":solidity,"fill":fill,"circularity":circularity,"bbox_local":[x,y,w,h],"area":area}

    def _detect_targets(self, frame, foam_profile=None, foam_coverage=None):
        normalized = self._normalize_lighting(frame)
        frame_h,frame_w=frame.shape[:2]
        x1,y1,x2,y2=self._roi_px(frame)
        roi=normalized[y1:y2,x1:x2]
        if roi.size==0:
            return [],[]
        roi_h,roi_w=roi.shape[:2]; roi_area=float(max(1,roi_h*roi_w))
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV); lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)
        ok=max(1,int(cfg.TARGET_MORPH_OPEN_KERNEL)); ck=max(1,int(cfg.TARGET_MORPH_CLOSE_KERNEL))
        if ok%2==0: ok+=1
        if ck%2==0: ck+=1
        k_open=np.ones((ok,ok),np.uint8); k_close=np.ones((ck,ck),np.uint8)
        coverage = None if foam_coverage is None else float(foam_coverage)
        detections=[]; blobs=[]
        focus_active=self.dynamic_roi is not None
        for color_name,ranges in cfg.TARGET_HSV_RANGES.items():
            mask=np.zeros((roi_h,roi_w),dtype=np.uint8)
            for lo,hi in ranges:
                mask=cv2.bitwise_or(mask,cv2.inRange(hsv,np.asarray(lo,np.uint8),np.asarray(hi,np.uint8)))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k_open)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k_close)
            found=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours=found[0] if len(found)==2 else found[1]
            for contour in contours:
                area=float(cv2.contourArea(contour)); frac=area/roi_area
                min_frac=cfg.TARGET_MIN_AREA_FRAC_ROI*(0.55 if focus_active else 1.0)
                max_frac=0.62 if focus_active else cfg.TARGET_MAX_AREA_FRAC_ROI
                if area<cfg.TARGET_MIN_AREA_PX or frac<min_frac or frac>max_frac:
                    continue
                bx,by,bw,bh=cv2.boundingRect(contour)
                margin=2 if focus_active else cfg.TARGET_BORDER_MARGIN_PX
                # Explicit cut-border rejection from the friend's detector: a
                # contour clipped by the current ROI has untrustworthy shape.
                if bx<=margin or by<=margin or bx+bw>=roi_w-margin or by+bh>=roi_h-margin:
                    continue
                gx,gy=x1+bx,y1+by; cx=gx+bw*0.5; cy=gy+bh*0.5
                foam_ok,foam_y,foam_reason,below_frac=self._foam_gate(
                    frame_w,frame_h,cx,cy,[gx,gy,bw,bh],foam_profile,profile_t=time.monotonic(),coverage=coverage
                )
                if not foam_ok:
                    continue
                median_hsv,median_lab=self._inside_statistics(contour,hsv,lab)
                color_conf=self._color_confidence(color_name,median_hsv,median_lab)
                blob={
                    "kind":"COLOR_BLOB","color":color_name,
                    "center_norm":[cx/frame_w,cy/frame_h],
                    "bbox_norm":[gx/frame_w,gy/frame_h,bw/frame_w,bh/frame_h],
                    "bbox_px":[int(gx),int(gy),int(bw),int(bh)],
                    "area_frac_roi":frac,"color_confidence":color_conf,
                    "median_hsv":list(median_hsv),"median_lab":list(median_lab),
                    "foam_gate":foam_reason,"foam_wall_y_px":foam_y,"foam_bbox_below_frac":below_frac,
                }
                blobs.append(blob)
                classified=self._classify_shape(contour)
                if classified is None:
                    continue
                area_score=min(1.0,frac/0.025)
                score=clamp(0.58*color_conf+0.34*float(classified["quality"])+0.08*area_score,0.0,1.0)
                if score<cfg.TARGET_MIN_CONFIDENCE:
                    continue
                item=dict(blob)
                item.update({
                    "kind":"COLOR_SHAPE","shape":classified["shape"],"score":score,
                    "shape_confidence":float(classified["quality"]),"solidity":classified["solidity"],
                    "fill":classified["fill"],"circularity":classified["circularity"],
                })
                detections.append(item)
        detections.sort(key=lambda d:float(d.get("score",0.0)),reverse=True)
        blobs.sort(key=lambda d:float(d.get("color_confidence",0.0)),reverse=True)
        return detections,blobs

    def _fresh_frame_count(self, since_t):
        with self.lock:
            return sum(1 for ts,_seq,_d in self.history if ts>=float(since_t))

    def _snapshot_latest(self):
        with self.lock:
            return self.latest_frame_t, [dict(d) for d in self.latest_detections], [dict(b) for b in self.latest_color_blobs]

    @staticmethod
    def _center_distance(a,b):
        try:
            return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
        except Exception:
            return float("inf")

    def _best_candidate_since(self, since_t):
        with self.lock:
            rows=[(ts,[dict(d) for d in dets]) for ts,_seq,dets in self.history if ts>=float(since_t)]
        candidates=[]
        for _ts,dets in rows:
            candidates.extend(d for d in dets if float(d.get("score",0.0))>=cfg.TARGET_MIN_CONFIDENCE)
        if not candidates:
            return None
        return dict(max(candidates,key=lambda d:float(d.get("score",0.0))))

    def _quick_gate_candidate_since(self, since_t, min_frames=cfg.TARGET_SEARCH_QUICK_GATE_FRAMES):
        """Require the same color/shape candidate on distinct recent frames.

        This mirrors the linked branch's cheap multi-frame candidate gate so a
        one-frame colored reflection does not stop the gimbal and enter AIM.
        """
        with self.lock:
            rows=[(ts,[dict(d) for d in dets]) for ts,_seq,dets in self.history if ts>=float(since_t)]
        tracks=[]
        needed=max(1,int(min_frames))
        for _ts,dets in rows:
            next_tracks=[]; used=set()
            for d in dets:
                if float(d.get("score",0.0))<cfg.TARGET_MIN_CONFIDENCE:
                    continue
                best_i=None; best_dist=None
                for i,tr in enumerate(tracks):
                    if i in used: continue
                    if tr["color"]!=d.get("color") or tr["shape"]!=d.get("shape"):
                        continue
                    dist=self._center_distance(tr["center"],d.get("center_norm"))
                    if dist>cfg.TARGET_TRACK_MAX_JUMP_NORM:
                        continue
                    if best_dist is None or dist<best_dist:
                        best_i=i; best_dist=dist
                if best_i is None:
                    next_tracks.append({"color":d.get("color"),"shape":d.get("shape"),"count":1,"center":d.get("center_norm"),"best":dict(d)})
                else:
                    used.add(best_i); tr=tracks[best_i]
                    best=dict(d) if float(d.get("score",0.0))>=float(tr["best"].get("score",0.0)) else dict(tr["best"])
                    next_tracks.append({"color":d.get("color"),"shape":d.get("shape"),"count":int(tr["count"])+1,"center":d.get("center_norm"),"best":best})
            tracks=next_tracks
            qualified=[tr for tr in tracks if int(tr["count"])>=needed]
            if qualified:
                winner=max(qualified,key=lambda tr:(int(tr["count"]),float(tr["best"].get("score",0.0))))
                out=dict(winner["best"]); out["quick_gate_frames"]=int(winner["count"]); return out
        return None

    @staticmethod
    def _shape_compatible(expected_shape, observed_shape):
        """Allow perspective jitter among square/rectangle labels during AIM only.

        A planar quadrilateral can flip SQUARE <-> RECT_VERTICAL/HORIZONTAL as
        the gimbal moves.  Circle remains exact.  Color + spatial continuity
        still have to match, and the final fire gate still needs a stable target.
        """
        a=str(expected_shape or "")
        b=str(observed_shape or "")
        if a == b:
            return True
        quad={"SQUARE","RECT_VERTICAL","RECT_HORIZONTAL"}
        return a in quad and b in quad

    def _latest_match(self, expected, expected_center, allow_blob_fallback=True):
        _ts,dets,blobs=self._snapshot_latest()
        exact=[d for d in dets if d.get("color")==expected.get("color") and self._shape_compatible(expected.get("shape"), d.get("shape"))]
        if exact:
            best=min(exact,key=lambda d:self._center_distance(d.get("center_norm"),expected_center))
            if self._center_distance(best.get("center_norm"),expected_center)<=cfg.TARGET_TRACK_MAX_JUMP_NORM:
                return dict(best)
        if allow_blob_fallback:
            same=[b for b in blobs if b.get("color")==expected.get("color")]
            if same:
                best=min(same,key=lambda b:self._center_distance(b.get("center_norm"),expected_center))
                if self._center_distance(best.get("center_norm"),expected_center)<=cfg.TARGET_TRACK_MAX_JUMP_NORM:
                    pseudo=dict(expected)
                    pseudo.update(best)
                    pseudo["kind"]="COLOR_SHAPE"
                    pseudo["shape"]=expected.get("shape")
                    pseudo["score"]=max(float(expected.get("score",0.5)),float(best.get("color_confidence",0.5)))
                    pseudo["shape_fallback_track"]=True
                    return pseudo
        return None

    def _verify_exact(self, expected, since_t=None):
        start=time.monotonic() if since_t is None else float(since_t)
        deadline=time.monotonic()+cfg.TARGET_VERIFY_WINDOW_SEC
        last_seq=-1; hits=[]
        while self.owner.running and self.running and time.monotonic()<deadline and len(hits)<cfg.TARGET_VERIFY_FRAMES:
            with self.lock:
                seq=self.frame_seq; dets=[dict(d) for d in self.latest_detections]; frame_t=self.latest_frame_t
            if seq==last_seq or frame_t<start:
                time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
            last_seq=seq
            matches=[d for d in dets if d.get("color")==expected.get("color") and self._shape_compatible(expected.get("shape"), d.get("shape")) and float(d.get("score",0.0))>=cfg.TARGET_MIN_CONFIDENCE]
            if not matches:
                hits=[]
                time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
            ref=expected.get("center_norm",(0.5,0.5))
            best=min(matches,key=lambda d:self._center_distance(d.get("center_norm"),ref))
            if self._center_distance(best.get("center_norm"),ref)>cfg.TARGET_TRACK_MAX_JUMP_NORM:
                hits=[]; time.sleep(cfg.TARGET_AIM_POLL_SEC); continue
            hits.append(best); expected=dict(best)
            time.sleep(cfg.TARGET_AIM_POLL_SEC)
        if len(hits)<cfg.TARGET_VERIFY_FRAMES:
            return None
        scores=[float(d.get("score",0.0)) for d in hits]
        xs=[float(d["center_norm"][0]) for d in hits]; ys=[float(d["center_norm"][1]) for d in hits]
        areas=[float(d.get("area_frac_roi",0.0)) for d in hits]
        center_std=math.hypot(statistics.pstdev(xs) if len(xs)>1 else 0.0, statistics.pstdev(ys) if len(ys)>1 else 0.0)
        area_mean=statistics.fmean(areas) if areas else 0.0
        area_cv=(statistics.pstdev(areas)/max(area_mean,1e-6)) if len(areas)>1 else 0.0
        mean_score=statistics.fmean(scores)
        if mean_score<cfg.TARGET_SAVE_CONFIDENCE or center_std>cfg.TARGET_VERIFY_MAX_CENTER_STD or area_cv>cfg.TARGET_VERIFY_MAX_AREA_CV:
            return None
        best=dict(max(hits,key=lambda d:float(d.get("score",0.0))))
        best["confirm_frames"]=len(hits); best["temporal_score"]=mean_score; best["center_std"]=center_std; best["area_cv"]=area_cv
        return best

    @staticmethod
    def _sector_fire_cone(sector_name):
        """Return the V15 SOFT acquisition cone used to authorize SHOOT-INTENT."""
        name = str(sector_name or "").upper()
        for sector, _start, _end, _slide, lo, hi in cfg.TARGET_RADAR_SECTORS:
            if sector == name:
                return float(lo), float(hi)
        return None

    @staticmethod
    def _sector_preferred_fire_cone(sector_name):
        """Return the original narrow preferred acquisition cone for telemetry/UI."""
        name = str(sector_name or "").upper()
        cone = cfg.TARGET_PREFERRED_ACQUISITION_CONES.get(name)
        if cone is None:
            return None
        return float(cone[0]), float(cone[1])

    @staticmethod
    def _sector_good_fire_cone(sector_name):
        """Final physical-fire cone; search/track may remain much wider."""
        name = str(sector_name or "").upper()
        cone = cfg.TARGET_GOOD_FIRE_CONES.get(name)
        if cone is None:
            return None
        return float(cone[0]), float(cone[1])

    @staticmethod
    def _sector_aim_envelope(sector_name):
        name = str(sector_name or "").upper()
        if name == "FRONT":
            half = float(cfg.TARGET_FRONT_CROSSHAIR_AIM_HALF_DEG)
            return (
                max(-float(cfg.TARGET_AIM_YAW_LIMIT_DEG), -half),
                min(+float(cfg.TARGET_AIM_YAW_LIMIT_DEG), +half),
            )
        if name == "LEFT":
            return (
                max(-float(cfg.TARGET_AIM_YAW_LIMIT_DEG), -float(cfg.TARGET_SIDE_CROSSHAIR_AIM_OUTER_DEG)),
                -float(cfg.TARGET_SIDE_CROSSHAIR_AIM_INNER_DEG),
            )
        if name == "RIGHT":
            return (
                +float(cfg.TARGET_SIDE_CROSSHAIR_AIM_INNER_DEG),
                min(+float(cfg.TARGET_AIM_YAW_LIMIT_DEG), +float(cfg.TARGET_SIDE_CROSSHAIR_AIM_OUTER_DEG)),
            )
        cone = TargetVisionSubsystem._sector_fire_cone(name)
        if cone is None:
            return None
        axis = 0.5 * (float(cone[0]) + float(cone[1]))
        half = float(cfg.TARGET_CROSSHAIR_AIM_MAX_FROM_AXIS_DEG)
        return (
            max(-float(cfg.TARGET_AIM_YAW_LIMIT_DEG), axis - half),
            min(+float(cfg.TARGET_AIM_YAW_LIMIT_DEG), axis + half),
        )

    @staticmethod
    def _yaw_in_cone(yaw_deg, cone):
        if yaw_deg is None or cone is None:
            return False
        lo, hi = cone
        y = float(yaw_deg)
        return float(lo) <= y <= float(hi)

    def _draw_overlay(self,frame,detections,profile,coverage):
        out=frame.copy(); h,w=out.shape[:2]; x1,y1,x2,y2=self._roi_px(out)
        if profile is not None:
            pts=[]
            for px in range(max(0,x1),min(x2,len(profile))):
                wy=float(profile[px])
                if math.isfinite(wy):
                    pts.append((px,int(round(wy))))
                elif len(pts)>=2:
                    cv2.polylines(out,[np.asarray(pts,np.int32)],False,(255,255,0),2,cv2.LINE_AA); pts=[]
            if len(pts)>=2:
                cv2.polylines(out,[np.asarray(pts,np.int32)],False,(255,255,0),2,cv2.LINE_AA)
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),1)
        colors={"RED":(0,0,255),"YELLOW":(0,255,255),"GREEN":(0,220,0),"BLUE":(255,100,0)}
        for d in detections:
            x,y,bw,bh=d.get("bbox_px",(0,0,0,0)); c=colors.get(d.get("color"),(255,255,255))
            cv2.rectangle(out,(x,y),(x+bw,y+bh),c,2)
            cx=int((x+bw*0.5)); cy=int((y+bh*0.5)); cv2.circle(out,(cx,cy),3,c,-1)
            txt=f"{d.get('color')} {d.get('shape')} {float(d.get('score',0)):.2f}"
            cv2.putText(out,txt,(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1,cv2.LINE_AA)
        aim_x=int(round((0.5+cfg.TARGET_AIM_OFFSET_X)*w)); aim_y=int(round((0.5+cfg.TARGET_AIM_OFFSET_Y)*h))
        cv2.drawMarker(out,(aim_x,aim_y),(0,255,0),cv2.MARKER_CROSS,24,1)
        cv2.putText(out,f"FOAM {'OK' if coverage>=cfg.TARGET_FOAM_MIN_ROI_COVERAGE else 'FAIL-OPEN'} cov={coverage*100:.0f}% | {self.status[:70]}",(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
        return out
