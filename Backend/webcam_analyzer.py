"""
webcam_analyzer.py — Webcam Nervousness Detection Module (v2.1)
================================================================
Research basis:
  - Kuipers et al. (2023) "How nervous am I?" — Cognitive Emotion (37:1105-1115)
    Interview-specific nervousness AUs: AU4, AU7, AU20, AU1, AU14.
  - Soukupová & Čech (2016) CVWW — EAR for blink detection; PERCLOS threshold 0.20.
  - Tomashin et al. (2025) PLoS ONE — BPM + BRV as state-anxiety biomarkers (R=0.557).
  - PMC Multimodal Review (2025) — gaze direction + fixation length index sympathetic arousal.
  - Explainable Traits from Head Motion & AUs (IEEE/PMC 2023) — kinemes predict nervousness.
  - Research Square (2025) — EAR PERCLOS + variance, ~80% accuracy.
  - Schuller et al. (2011) IEEE TAC — voice 82–91% vs face 64–73%; fusion > unimodal.

Features extracted per frame:
  - EAR (Eye Aspect Ratio)       → PERCLOS, EAR variance (anxiety biomarker)
  - Blink rate + IBI             → BPM excess above calm baseline (12 bpm speaking)
  - Head pose proxy (yaw, pitch) → kineme variance, lean-away score
  - Mouth aspect ratio (MAR)     → lip tension / jaw movement
  - Facial asymmetry index       → stress amplifies micro-asymmetry

MediaPipe compatibility (v2.1):
  MediaPipe >= 0.10.0 removed mp.solutions (the legacy Solutions API).
  This module now uses the new mediapipe.tasks.python.vision API with a
  two-stage fallback so the server never crashes regardless of MP version:

    1. mediapipe.tasks FaceLandmarker  — new API (MP >= 0.10.0)
    2. mp.solutions.face_mesh.FaceMesh — legacy API (MP < 0.10.0)
    3. Neutral result                  — if both unavailable

_score_nervousness() v2.0 uses continuous piecewise formulas replacing the
original step-based scoring. Raw sub-scores all on 0–1 before weighted sum.

Return dict now includes raw time-series lists (ear_values, yaw_values,
pitch_values, inter_blink_intervals) so analyzer.py compute_facial_nervousness()
can apply its own formula independently.

Usage:
    analyzer = WebcamNervousnessAnalyzer()
    result   = analyzer.analyze_frames(frames_b64_list)

Install:
    pip install mediapipe opencv-python-headless numpy
"""

from __future__ import annotations

import base64
import math
import os
import urllib.request
import tempfile
from typing import List, Optional

import cv2
import numpy as np

# ── MediaPipe landmarks (468-point Face Mesh) ────────────────────────────────────────────
_LEFT_EYE    = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
_MOUTH_OUTER = [61, 291, 0, 17]   # left-corner, right-corner, top-center, bottom-center
_LEFT_CHEEK  = 234
_RIGHT_CHEEK = 454
_NOSE_TIP    = 1
_CHIN        = 152
_FOREHEAD    = 10

# ── MediaPipe API detection ─────────────────────────────────────────────────────
# Detect which MediaPipe API is available at import time so we pick the right
# backend once rather than on every frame.

_MP_BACKEND = "none"   # "tasks" | "solutions" | "none"
_mp_vision  = None
_mp_python  = None
_mp_solutions = None

try:
    import mediapipe as _mp_module
    try:
        # New Tasks API — available in MediaPipe >= 0.10.0
        from mediapipe.tasks import python as _mp_python        # noqa: F811
        from mediapipe.tasks.python import vision as _mp_vision  # noqa: F811
        _MP_BACKEND = "tasks"
    except (ImportError, AttributeError):
        # Legacy Solutions API — available in MediaPipe < 0.10.0
        if hasattr(_mp_module, "solutions"):
            _mp_solutions = _mp_module.solutions
            _MP_BACKEND = "solutions"
except ImportError:
    pass  # No mediapipe at all — will return neutral results

# ── FaceLandmarker model download (Tasks API only) ────────────────────────
# The Tasks API requires a .task model file; we download it once to a temp dir.
_TASK_MODEL_PATH: Optional[str] = None

def _get_task_model() -> Optional[str]:
    """
    Return path to face_landmarker.task, downloading it on first call.
    Uses the official MediaPipe model asset CDN.
    """
    global _TASK_MODEL_PATH
    if _TASK_MODEL_PATH and os.path.exists(_TASK_MODEL_PATH):
        return _TASK_MODEL_PATH

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".task", delete=False, prefix="mp_face_landmarker_"
        )
        urllib.request.urlretrieve(model_url, tmp.name)
        _TASK_MODEL_PATH = tmp.name
        return _TASK_MODEL_PATH
    except Exception:
        return None

# ── EAR / MAR helpers ────────────────────────────────────────────────────────
def _ear(landmarks, indices: list[int], w: int, h: int) -> float:
    """Eye Aspect Ratio — Soukupová & Čech 2016."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(landmarks, indices: list[int], w: int, h: int) -> float:
    """Mouth Aspect Ratio — vertical / horizontal opening."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    vertical   = math.dist(pts[2], pts[3])
    horizontal = math.dist(pts[0], pts[1])
    return vertical / (horizontal + 1e-6)


def _asymmetry(landmarks, w: int, h: int) -> float:
    """
    Simple facial asymmetry index.
    Compares left vs right eye EAR difference — stress amplifies micro-asymmetry.
    Ref: Arxiv 2310.20083 (Facial asymmetry as behaviometric index).
    """
    left  = _ear(landmarks, _LEFT_EYE,  w, h)
    right = _ear(landmarks, _RIGHT_EYE, w, h)
    return abs(left - right)


def _head_pose_proxy(landmarks) -> tuple[float, float]:
    """
    Lightweight head pose using nose-tip deviation from cheek midpoint (yaw)
    and nose-tip to chin ratio (pitch). Returns (yaw_proxy, pitch_proxy).
    No solvePnP — fast enough for real-time frame batches.
    """
    nose   = landmarks[_NOSE_TIP]
    chin   = landmarks[_CHIN]
    left_c = landmarks[_LEFT_CHEEK]
    right_c = landmarks[_RIGHT_CHEEK]
    top    = landmarks[_FOREHEAD]

    cheek_mid_x = (left_c.x + right_c.x) / 2.0
    yaw_proxy   = nose.x - cheek_mid_x                     # left/right tilt
    face_height = abs(top.y - chin.y) + 1e-6
    pitch_proxy = (nose.y - chin.y) / face_height           # forward/backward tilt
    return yaw_proxy, pitch_proxy


# ── Main analyzer ─────────────────────────────────────────────────────────────
class WebcamNervousnessAnalyzer:
    """
    Processes a batch of webcam frames (base64-encoded JPEGs) captured during
    an interview answer and returns a nervousness score + all sub-metrics.

    Recommended capture: 1 frame every 2 seconds, 15 frames total for a 30s answer.
    Frontend can sample at ~0.5 fps to keep payload small.

    v2.0 changes:
      - _score_nervousness() uses continuous piecewise formulas (not step bins).
      - Return dict now exposes raw time-series lists:
          ear_values, yaw_values, pitch_values, inter_blink_intervals
        so analyzer.py compute_facial_nervousness() can apply its own sub-scores.
      - PERCLOS added as an explicit reported metric.
      - Blink IBI (inter-blink-interval) list computed for BRV (Tomashin 2025).
      - EAR_BLINK_THRESHOLD lowered to 0.20 (Soukupová & Čech 2016 PERCLOS value).
    """

    EAR_BLINK_THRESHOLD  = 0.20     # Soukupová & Čech 2016 PERCLOS threshold
    EAR_CONSEC_FRAMES    = 2        # min consecutive closed frames = 1 blink
    DEFAULT_FPS          = 0.5      # default: frontend captures 1 frame every 2 s
    MIN_FRAMES_REQUIRED  = 3        # fewer frames → return neutral score
    CALM_BLINK_BPM       = 12.0     # speaking calm baseline (Tomashin 2025)
    ANXIOUS_BLINK_EXCESS = 20.0     # excess above baseline that maps to score 1.0

    # ── Facial baseline calibration ───────────────────────────────────────────
    # Population-mean facial values for a calm speaker (from Tomashin 2025,
    # Soukupová & Čech 2016, Research Square 2025).
    _POP_BLINK_RATE      = 15.0     # bpm, neutral reading aloud baseline
    _POP_EAR_VAR         = 0.0015   # EAR variance during calm reading
    _POP_PERCLOS         = 0.04     # fraction of closed-eye frames
    _POP_HEAD_INSTAB     = 0.008    # std(yaw)+std(pitch) for natural micro-movement
    _FACIAL_MAX_CORR     = 0.30     # max ±30% correction from facial baseline
    _FACIAL_BASELINE_MIN_FRAMES = 8  # minimum frames for a valid facial baseline

    def analyze_frames(self, frames_b64: List[str], fps: float = 0.0) -> dict:
        """
        Args:
            frames_b64: List of base64-encoded JPEG strings (no data URI prefix).
            fps: Actual capture rate in frames-per-second. Pass 0.0 (default)
                 to use DEFAULT_FPS (0.5 fps = 1 frame every 2 s). The frontend
                 should pass the real capture rate so blink BPM and PERCLOS
                 calculations remain accurate regardless of device performance.
        Returns:
            dict with nervousness_score, blink_rate_per_min, perclos,
            eye_stability, head_stability, gaze_aversion_rate,
            facial_asymmetry, interpretation, frames_analyzed,
            PLUS raw time-series lists for analyzer.py:
              ear_values, yaw_values, pitch_values, inter_blink_intervals.
        """
        # Resolve actual FPS — caller can override the default for accurate timing
        self._session_fps = fps if fps > 0.0 else self.DEFAULT_FPS

        if not frames_b64:
            return self._neutral_result(0, reason="no_frames")

        blink_count       = 0
        blink_counter     = 0
        in_blink          = False
        blink_frame_times: List[float] = []   # frame indices of each blink onset
        ear_values        = []
        mar_values        = []
        asymm_values      = []
        head_poses        = []   # list of (yaw_proxy, pitch_proxy)
        total_frames      = 0
        detected_frames   = 0

        # ── Select MediaPipe backend ────────────────────────────────────────────────
        # Supports both the new Tasks API (MP ≥ 0.10.0) and the legacy Solutions
        # API (MP < 0.10.0) so the same code works across all installed versions.
        if _MP_BACKEND == "none":
            return self._neutral_result(0, reason="mediapipe_unavailable")

        if _MP_BACKEND == "tasks":
            return self._analyze_frames_tasks(frames_b64)

        # ── Legacy Solutions API path (MP < 0.10.0) ────────────────────────
        mp_face = _mp_solutions.face_mesh
        with mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            for frame_idx, b64 in enumerate(frames_b64):
                # ── Decode frame ──────────────────────────────────────
                try:
                    img_bytes = base64.b64decode(b64)
                    arr   = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        # Count as a total frame but reset blink state to avoid
                        # carrying a partial blink counter across a gap.
                        total_frames += 1
                        blink_counter = 0
                        in_blink = False
                        continue
                except Exception:
                    total_frames += 1
                    blink_counter = 0
                    in_blink = False
                    continue

                h, w = frame.shape[:2]
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(rgb)
                total_frames += 1

                if not results.multi_face_landmarks:
                    # Face not detected → gaze aversion signal; also reset blink
                    # state so a partial blink before the gap isn't double-counted.
                    blink_counter = 0
                    in_blink = False
                    continue

                lm = results.multi_face_landmarks[0].landmark
                detected_frames += 1

                # ── EAR (blink detection + PERCLOS) ───────────────────────
                left_ear  = _ear(lm, _LEFT_EYE,  w, h)
                right_ear = _ear(lm, _RIGHT_EYE, w, h)
                avg_ear   = (left_ear + right_ear) / 2.0
                ear_values.append(avg_ear)

                if avg_ear < self.EAR_BLINK_THRESHOLD:
                    blink_counter += 1
                    in_blink = True
                else:
                    if in_blink and blink_counter >= self.EAR_CONSEC_FRAMES:
                        blink_count += 1
                        blink_frame_times.append(frame_idx / self._session_fps)
                    blink_counter = 0
                    in_blink      = False

                # ── MAR (mouth tension / jaw movement) ────────────────────
                mar_values.append(_mar(lm, _MOUTH_OUTER, w, h))

                # ── Facial asymmetry ───────────────────────────────────────
                asymm_values.append(_asymmetry(lm, w, h))

                # ── Head pose (yaw, pitch proxy) ──────────────────────────
                head_poses.append(_head_pose_proxy(lm))

        if total_frames < self.MIN_FRAMES_REQUIRED:
            return self._neutral_result(total_frames, reason="too_few_frames")

        return self._compute_result(
            total_frames, detected_frames, blink_count, blink_frame_times,
            ear_values, mar_values, asymm_values, head_poses,
        )

    # ── MediaPipe Tasks API path (MP ≥ 0.10.0) ─────────────────────────────────
    def _analyze_frames_tasks(self, frames_b64: List[str]) -> dict:
        """
        Frame-processing path for MediaPipe >= 0.10.0 (Tasks API).

        The new API replaced mp.solutions with mediapipe.tasks.python.vision.
        FaceLandmarker now takes an image object instead of a raw numpy RGB array,
        and returns FaceLandmarkerResult instead of the old SolutionOutputs.

        Landmark access differs from the legacy API:
          Legacy:  results.multi_face_landmarks[0].landmark[i].x
          Tasks:   results.face_landmarks[0][i].x
        Both use normalised [0, 1] coordinates, so all EAR/MAR/asymmetry
        helpers (_ear, _mar, _asymmetry, _head_pose_proxy) work unchanged.
        """
        model_path = _get_task_model()
        if model_path is None:
            return self._neutral_result(0, reason="task_model_download_failed")

        import mediapipe as mp_local
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision as mp_vis
        from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

        blink_count       = 0
        blink_counter     = 0
        in_blink          = False
        blink_frame_times: List[float] = []
        ear_values        = []
        mar_values        = []
        asymm_values      = []
        head_poses        = []
        total_frames      = 0
        detected_frames   = 0

        base_opts = mp_py.BaseOptions(model_asset_path=model_path)
        opts = mp_vis.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vis.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with mp_vis.FaceLandmarker.create_from_options(opts) as landmarker:
            for frame_idx, b64 in enumerate(frames_b64):
                # ── Decode frame ──────────────────────────────────────
                try:
                    img_bytes = base64.b64decode(b64)
                    arr   = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        # Count as a total frame but reset blink state to avoid
                        # carrying a partial blink counter across a gap.
                        total_frames += 1
                        blink_counter = 0
                        in_blink = False
                        continue
                except Exception:
                    total_frames += 1
                    blink_counter = 0
                    in_blink = False
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Tasks API uses mp.Image wrapper
                mp_image = mp_local.Image(
                    image_format=mp_local.ImageFormat.SRGB, data=rgb
                )
                result = landmarker.detect(mp_image)
                total_frames += 1

                if not result.face_landmarks:
                    # Face not detected → gaze aversion signal; also reset blink
                    # state so a partial blink before the gap isn't double-counted.
                    blink_counter = 0
                    in_blink = False
                    continue

                # Tasks API: result.face_landmarks[0] is a list of NormalizedLandmark
                # Each has .x, .y, .z — same normalised coords as the legacy API.
                lm = result.face_landmarks[0]
                detected_frames += 1

                # ── EAR (blink detection + PERCLOS) ───────────────────────
                left_ear  = _ear(lm, _LEFT_EYE,  w, h)
                right_ear = _ear(lm, _RIGHT_EYE, w, h)
                avg_ear   = (left_ear + right_ear) / 2.0
                ear_values.append(avg_ear)

                if avg_ear < self.EAR_BLINK_THRESHOLD:
                    blink_counter += 1
                    in_blink = True
                else:
                    if in_blink and blink_counter >= self.EAR_CONSEC_FRAMES:
                        blink_count += 1
                        blink_frame_times.append(frame_idx / self._session_fps)
                    blink_counter = 0
                    in_blink      = False

                # ── MAR / asymmetry / head pose ────────────────────────────────
                mar_values.append(_mar(lm, _MOUTH_OUTER, w, h))
                asymm_values.append(_asymmetry(lm, w, h))
                head_poses.append(_head_pose_proxy(lm))

        if total_frames < self.MIN_FRAMES_REQUIRED:
            return self._neutral_result(total_frames, reason="too_few_frames")

        return self._compute_result(
            total_frames, detected_frames, blink_count, blink_frame_times,
            ear_values, mar_values, asymm_values, head_poses,
        )

    # ── Shared result computation ─────────────────────────────────────────────
    def _compute_result(
        self,
        total_frames: int,
        detected_frames: int,
        blink_count: int,
        blink_frame_times: List[float],
        ear_values: list,
        mar_values: list,
        asymm_values: list,
        head_poses: list,
    ) -> dict:
        """Shared metric aggregation used by both the Tasks and Solutions paths."""
        duration_sec = total_frames / self._session_fps
        duration_min = max(duration_sec / 60.0, 0.01)
        blink_rate   = blink_count / duration_min

        inter_blink_intervals: List[float] = []
        if len(blink_frame_times) >= 2:
            inter_blink_intervals = [
                round(blink_frame_times[i] - blink_frame_times[i - 1], 3)
                for i in range(1, len(blink_frame_times))
            ]

        ear_mean     = float(np.mean(ear_values))   if ear_values   else 0.25
        ear_variance = float(np.var(ear_values))    if ear_values   else 0.0
        mar_mean     = float(np.mean(mar_values))   if mar_values   else 0.05
        asymm_mean   = float(np.mean(asymm_values)) if asymm_values else 0.0

        perclos = (
            sum(1 for e in ear_values if e < self.EAR_BLINK_THRESHOLD) /
            max(len(ear_values), 1)
        )

        yaw_values: List[float] = []
        pitch_values: List[float] = []
        head_instability = 0.0
        if len(head_poses) >= 2:
            yaw_arr   = np.array([p[0] for p in head_poses])
            pitch_arr = np.array([p[1] for p in head_poses])
            yaw_values   = [round(float(v), 5) for v in yaw_arr]
            pitch_values = [round(float(v), 5) for v in pitch_arr]
            head_instability = float(np.std(yaw_arr) + np.std(pitch_arr))

        gaze_aversion_rate = 1.0 - (detected_frames / max(total_frames, 1))

        nervousness_score = self._score_nervousness(
            blink_rate           = blink_rate,
            ear_variance         = ear_variance,
            perclos              = perclos,
            head_instability     = head_instability,
            gaze_aversion        = gaze_aversion_rate,
            asymm_mean           = asymm_mean,
            mar_mean             = mar_mean,
            inter_blink_intervals= inter_blink_intervals,
        )

        return {
            "nervousness_score":      nervousness_score,
            "nervousness_fraction":   round(nervousness_score / 100, 3),
            "blink_rate_per_min":     round(blink_rate, 1),
            "perclos":                round(perclos, 3),
            "eye_stability":          round(max(0.0, 1.0 - min(ear_variance * 600, 1.0)), 3),
            "head_stability":         round(max(0.0, 1.0 - min(head_instability * 60, 1.0)), 3),
            "gaze_aversion_rate":     round(gaze_aversion_rate, 3),
            "facial_asymmetry":       round(asymm_mean, 4),
            "frames_analyzed":        total_frames,
            "frames_with_face":       detected_frames,
            "interpretation":         self._interpret(nervousness_score),
            "coaching_note":          self._coaching_note(
                blink_rate, head_instability, gaze_aversion_rate
            ),
            "ear_values":             [round(float(v), 4) for v in ear_values],
            "yaw_values":             yaw_values,
            "pitch_values":           pitch_values,
            "inter_blink_intervals":  inter_blink_intervals,
        }

    # ── Scoring model v2.0 ───────────────────────────────────────────────────
    def _score_nervousness(
        self,
        blink_rate: float,
        ear_variance: float,
        perclos: float,
        head_instability: float,
        gaze_aversion: float,
        asymm_mean: float,
        mar_mean: float,
        inter_blink_intervals: Optional[List[float]] = None,
    ) -> int:
        """
        Weighted nervousness score 0–100 using continuous piecewise formulas.

        Formula (v2.0):
          Component              Weight  Research basis
          ─────────────────────  ──────  ──────────────────────────────────────
          Blink rate score         30%   Tomashin et al. PLoS ONE 2025
          BRV (IBI std dev)        10%   Tomashin et al. PLoS ONE 2025
          EAR: PERCLOS             12%   Soukupová & Čech CVWW 2016
          EAR: variance            13%   Research Square 2025
          Head instability         20%   IEEE/PMC 2023 kineme traits
          Gaze aversion            10%   PMC Multimodal Review 2025
          Asymmetry + MAR           5%   Arxiv 2310.20083

        All sub-scores normalised to 0–1 before weighting.
        Final = round(clamp(0–100, Σ(sub × weight × 100))).
        """

        # ── 1. Blink rate score (30%) ─────────────────────────────────────
        # Tomashin 2025: speaking baseline ~12 bpm; anxiety → 25–35 bpm.
        # Also score low-blink "frozen stare" as anxiety:
        #   <5 bpm → 0.80, 5–12 bpm → linear 0.10→0.00, 12–20 bpm → 0.00 (calm)
        #   20–32 bpm → linear 0.00→1.00, >32 bpm → 1.00
        if blink_rate < 5.0:
            blink_sc = 0.80
        elif blink_rate < self.CALM_BLINK_BPM:
            blink_sc = 0.10 * (self.CALM_BLINK_BPM - blink_rate) / (self.CALM_BLINK_BPM - 5.0)
        elif blink_rate <= 20.0:
            blink_sc = 0.0
        else:
            excess   = blink_rate - 20.0
            blink_sc = min(1.0, excess / self.ANXIOUS_BLINK_EXCESS)

        # ── 2. Blink Rate Variability (10%) ───────────────────────────────
        # BRV = std(IBI). Calm: 0.5–1.0 s; Anxious bursts: >2.0 s std.
        # Formula: clamp(0–1, (std − 0.5) / 2.5)
        brv_sc = 0.0
        if inter_blink_intervals and len(inter_blink_intervals) >= 2:
            import statistics as _st
            brv = _st.stdev(inter_blink_intervals)
            brv_sc = min(1.0, max(0.0, (brv - 0.5) / 2.5))

        # ── 3. PERCLOS (12%) ──────────────────────────────────────────────
        # Soukupová & Čech 2016: proportion of frames with EAR < 0.20.
        # 0 = no eye closure; 1 = all frames closed (extreme tension/fatigue)
        perclos_sc = min(1.0, perclos * 2.0)   # 50% PERCLOS → score 1.0

        # ── 4. EAR variance (13%) ─────────────────────────────────────────
        # Research Square 2025: continuous eye tremor captured in EAR variance.
        # ceiling: 0.005 → score 1.0
        ear_var_sc = min(1.0, ear_variance / 0.005)

        # ── 5. Head instability (20%) ─────────────────────────────────────
        # IEEE/PMC 2023: head kineme variance predicts interview anxiety traits.
        # head_instability = std(yaw) + std(pitch) in landmark normalised units.
        # 0.008 = mild movement; 0.015 = moderate; 0.025+ = high.
        if head_instability < 0.008:
            head_sc = 0.0
        elif head_instability < 0.015:
            head_sc = (head_instability - 0.008) / (0.015 - 0.008) * 0.4
        elif head_instability < 0.025:
            head_sc = 0.4 + (head_instability - 0.015) / (0.025 - 0.015) * 0.4
        else:
            head_sc = min(1.0, 0.8 + (head_instability - 0.025) / 0.025 * 0.2)

        # ── 6. Gaze aversion (10%) ────────────────────────────────────────
        # PMC Multimodal 2025: proportion of frames where face was undetected.
        # 0 = always looking at camera; 1 = never detected.
        gaze_sc = min(1.0, gaze_aversion * 1.5)   # 67% aversion → score 1.0

        # ── 7. Asymmetry + MAR (5%) ───────────────────────────────────────
        # Arxiv 2310.20083: facial asymmetry amplified under stress.
        # Tight lips (MAR < 0.02) = jaw tension.
        asymm_sc = min(1.0, asymm_mean / 0.06)   # 0.06 EAR diff = ceiling
        mar_sc   = 1.0 if mar_mean < 0.02 else 0.0

        # ── Weighted sum → 0–100 ──────────────────────────────────────────
        raw = (
            blink_sc    * 0.30 +
            brv_sc      * 0.10 +
            perclos_sc  * 0.12 +
            ear_var_sc  * 0.13 +
            head_sc     * 0.20 +
            gaze_sc     * 0.10 +
            (asymm_sc * 0.03 + mar_sc * 0.02)   # combined 5%
        )
        return min(100, round(raw * 100))

    # ── Interpretation helpers ────────────────────────────────────────────────
    def _interpret(self, score: int) -> str:
        if score >= 75:
            return "High nervousness detected — visible stress signals"
        if score >= 50:
            return "Moderate nervousness — some tension present"
        if score >= 25:
            return "Mild nervousness — mostly composed"
        return "Calm and composed"

    def _coaching_note(
        self, blink_rate: float, head_instability: float, gaze_aversion: float
    ) -> str:
        notes = []
        if blink_rate > 25:
            notes.append("Rapid blinking detected — try slow deep breaths before answering.")
        elif blink_rate < 8:
            notes.append("Reduced blinking noticed — you may be concentrating too hard; soften your gaze.")
        if head_instability > 0.015:
            notes.append("Frequent head movement — plant your feet and anchor your posture.")
        if gaze_aversion > 0.3:
            notes.append("Frequent gaze aversion — maintain soft eye contact with the camera.")
        return " ".join(notes) if notes else "Good visual composure throughout."

    # ── Pre-warm (call at server startup) ────────────────────────────────────
    def prewarm(self) -> None:
        """
        Pre-download the MediaPipe Tasks model at server startup so the first
        analyze_frames() call doesn't block a live request.
        Safe to call even if the Tasks API is not in use (no-op for Solutions path).
        """
        if _MP_BACKEND == "tasks":
            _get_task_model()

    # ── Facial baseline calibration ───────────────────────────────────────────

    def calibrate_facial_baseline(self, frames_b64: List[str], fps: float = 0.0) -> dict:
        """
        Capture the candidate's facial baseline from a short neutral reading.

        Runs the same frame-processing pipeline as analyze_frames() but returns
        a baseline dict rather than a scored result. The baseline records the
        candidate's natural blink rate, EAR variance, PERCLOS, and head
        instability so that interview frames can be normalised per-speaker.

        Parameters
        ----------
        frames_b64 : list of base64 JPEG strings captured during neutral reading
                     (recommended: 8–15 frames at 0.5 fps = 16–30 seconds)
        fps        : capture rate (0.0 → DEFAULT_FPS)

        Returns
        -------
        dict with keys:
            valid             : bool   — True if enough frames and face detected
            blink_rate_bpm    : float  — natural blink rate (bpm)
            ear_variance      : float  — natural EAR variance
            perclos           : float  — natural PERCLOS fraction
            head_instability  : float  — natural head movement std
            frames_analyzed   : int
            reason            : str    — empty if valid; failure reason otherwise

        Research: Kappen et al. (2024, Scientific Reports) §4.1 —
        within-speaker delta scores for facial features predict self-reported
        interview stress significantly better than absolute scores (r=0.63 vs 0.41).
        """
        raw = self.analyze_frames(frames_b64, fps=fps)
        n_frames     = raw.get("frames_with_face", 0)
        if n_frames < self._FACIAL_BASELINE_MIN_FRAMES:
            return {
                "valid": False,
                "reason": f"too_few_detected_frames ({n_frames} < {self._FACIAL_BASELINE_MIN_FRAMES})",
                "frames_analyzed": raw.get("frames_analyzed", 0),
                "blink_rate_bpm":   self._POP_BLINK_RATE,
                "ear_variance":     self._POP_EAR_VAR,
                "perclos":          self._POP_PERCLOS,
                "head_instability": self._POP_HEAD_INSTAB,
            }
        return {
            "valid":            True,
            "reason":           "",
            "frames_analyzed":  raw.get("frames_analyzed", 0),
            "blink_rate_bpm":   raw.get("blink_rate_per_min", self._POP_BLINK_RATE),
            "ear_variance":     1.0 - raw.get("eye_stability", 0.85),   # invert stability → variance proxy
            "perclos":          raw.get("perclos", self._POP_PERCLOS),
            "head_instability": 1.0 - raw.get("head_stability", 0.85),  # invert stability → instability proxy
        }

    def apply_facial_baseline_correction(
        self, result: dict, baseline: Optional[dict]
    ) -> dict:
        """
        Apply per-speaker delta normalisation to a webcam nervousness result.

        For each facial feature, computes the relative deviation of the
        candidate's baseline from the population mean and applies a
        proportional correction:

            correction = (baseline_val / pop_mean) - 1.0
            correction = clamp(correction, -_FACIAL_MAX_CORR, +_FACIAL_MAX_CORR)
            corrected_score = score × (1 - correction)

        The weighted sum in _score_nervousness() is then recomputed from
        corrected sub-scores. For efficiency, this is approximated by applying
        the correction directly to nervousness_score (scalar path) since the
        sub-scores are not individually re-exposed after scoring.

        Parameters
        ----------
        result   : dict — output of analyze_frames() or _compute_result()
        baseline : dict — output of calibrate_facial_baseline(); None = no-op

        Returns
        -------
        dict — same keys as result with nervousness_score adjusted and
               baseline_corrected=True added.
        """
        if not baseline or not baseline.get("valid"):
            result["baseline_corrected"] = False
            return result

        def _corr(raw_val: float, baseline_val: float, pop_mean: float) -> float:
            if pop_mean < 1e-6:
                return raw_val
            correction = (baseline_val / pop_mean) - 1.0
            correction = max(-self._FACIAL_MAX_CORR, min(self._FACIAL_MAX_CORR, correction))
            return max(0.0, min(1.0, raw_val * (1.0 - correction)))

        # Map baseline and result to comparable scales
        bl_blink   = baseline.get("blink_rate_bpm", self._POP_BLINK_RATE)
        bl_ear_var = baseline.get("ear_variance", self._POP_EAR_VAR)
        bl_perclos = baseline.get("perclos", self._POP_PERCLOS)
        bl_head    = baseline.get("head_instability", self._POP_HEAD_INSTAB)

        # Pop means for score components (0–1 normalised)
        # These mirror the scoring formula in _score_nervousness():
        #   blink excess: calm at rate 20; max at rate 32+
        #   ear_variance: ceiling 0.005
        #   perclos:      score = perclos × 2.0; pop≈0.08 (= score 0.16)
        #   head:         instability 0.008 → score 0.0; 0.015 → 0.4; 0.025 → 0.8
        raw_score_0_1 = result.get("nervousness_score", 20) / 100.0

        # Weighted correction across four facial sub-signals
        bl_blink_score = min(1.0, max(0.0, (bl_blink - 20.0) / 20.0)) if bl_blink > 20 else 0.0
        bl_ear_score   = min(1.0, bl_ear_var / 0.005)
        bl_perclos_sc  = min(1.0, bl_perclos * 2.0)
        bl_head_score  = min(1.0, max(0.0, (bl_head - 0.008) / 0.025))

        # Pop means of those sub-scores for a typical calm reader
        pop_blink_sc  = min(1.0, max(0.0, (self._POP_BLINK_RATE - 20.0) / 20.0))
        pop_ear_sc    = min(1.0, self._POP_EAR_VAR / 0.005)
        pop_perclos_sc = min(1.0, self._POP_PERCLOS * 2.0)
        pop_head_sc   = min(1.0, max(0.0, (self._POP_HEAD_INSTAB - 0.008) / 0.025))

        # Aggregate correction as weighted average of per-feature corrections
        # (weights mirror _score_nervousness() weights for these four signals)
        w_blink, w_ear_var, w_perclos, w_head = 0.30, 0.13, 0.12, 0.20
        w_total = w_blink + w_ear_var + w_perclos + w_head

        blink_c  = _corr(bl_blink_score,   bl_blink_score, max(pop_blink_sc, 1e-6))
        ear_c    = _corr(bl_ear_score,      bl_ear_score,   max(pop_ear_sc,   1e-6))
        percl_c  = _corr(bl_perclos_sc,     bl_perclos_sc,  max(pop_perclos_sc,1e-6))
        head_c   = _corr(bl_head_score,     bl_head_score,  max(pop_head_sc,  1e-6))

        # Average relative correction factor
        sum_bl   = (bl_blink_score  * w_blink + bl_ear_score * w_ear_var +
                    bl_perclos_sc   * w_perclos + bl_head_score * w_head)
        sum_pop  = (pop_blink_sc    * w_blink + pop_ear_sc   * w_ear_var +
                    pop_perclos_sc  * w_perclos + pop_head_sc  * w_head) + 1e-6
        scalar_correction = max(-self._FACIAL_MAX_CORR,
                                min(self._FACIAL_MAX_CORR, (sum_bl / sum_pop) - 1.0))

        corrected_0_1    = max(0.0, min(1.0, raw_score_0_1 * (1.0 - scalar_correction)))
        corrected_result = dict(result)
        corrected_result["nervousness_score"]      = min(100, round(corrected_0_1 * 100))
        corrected_result["nervousness_fraction"]   = round(corrected_0_1, 3)
        corrected_result["interpretation"]         = self._interpret(corrected_result["nervousness_score"])
        corrected_result["baseline_corrected"]     = True
        corrected_result["baseline_blink_rate"]    = bl_blink
        corrected_result["baseline_correction_pct"]= round(scalar_correction * 100, 1)
        return corrected_result

    # ── Fallback ──────────────────────────────────────────────────────────────
    def _neutral_result(self, frames: int, reason: str = "") -> dict:
        return {
            "nervousness_score":      20,
            "nervousness_fraction":   0.20,
            "blink_rate_per_min":     15.0,
            "perclos":                0.05,
            "eye_stability":          0.85,
            "head_stability":         0.85,
            "gaze_aversion_rate":     0.0,
            "facial_asymmetry":       0.01,
            "frames_analyzed":        frames,
            "frames_with_face":       frames,
            "interpretation":         "Insufficient data — neutral score assigned",
            "coaching_note":          "Enable webcam for visual nervousness tracking.",
            # Raw lists — empty defaults for analyzer.py (will use calm defaults)
            "ear_values":             [],
            "yaw_values":             [],
            "pitch_values":           [],
            "inter_blink_intervals":  [],
            "_reason":                reason,
        }