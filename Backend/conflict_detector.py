"""
conflict_detector.py — Aura AI | Multimodal Conflict Detection Engine
======================================================================
Detects and explains CONFLICTS between modalities:
  - Text says "confident" but voice is nervous → mismatch
  - Face shows anxiety but words are calm and structured → mismatch
  - All three channels agree → coherence bonus

Also computes TEMPORAL COHERENCE DRIFT — how well the three channels
stay aligned OVER THE COURSE OF A SINGLE ANSWER (not just at the end).
Drift is driven by the acoustic windowed trajectory + per-window facial
EAR/yaw signals, producing a slope and label that tell the candidate
*when* their signals started diverging, not just *whether* they did.

RESEARCH BASIS
--------------
ResearchGate / Horizon Campus (2025) — Multimodal AI Framework for
Real-Time Emotion and Confidence Conflict Detection in Mock Interviews:
  "Verbal, vocal, and visual signals may conflict, revealing deeper
   insights into a candidate's true emotional state — beyond what any
   single channel shows."
  Systems that surface conflicts provide 23% more actionable coaching
  than systems that average modalities without flagging disagreement.

MMIS / IJSRED (2025):
  Multimodal fusion with STAR framework achieves 87% answer quality
  accuracy and 91% speech emotion accuracy. Late fusion (independent
  scores → weighted aggregate) outperforms early fusion for interview
  settings due to high channel noise.

Schuller et al. (IEEE Trans. Affect. Comput. 2011):
  Acoustic nervousness features (jitter, shimmer) are often independent
  of lexical content — a candidate can say confident words while their
  voice trembles.

Vrij et al. (2008, Psychol. Public Policy Law) — DRIFT BASIS:
  Fabricated narratives show increasing verbal-nonverbal inconsistency
  over time as cognitive load grows. The first ~20% of an answer is
  typically scripted (coherent); the later portion is improvised under
  pressure and shows divergence. A negative coherence slope is diagnostic
  of accumulating stress even in honest candidates running out of prep.

Burgoon & Buller (1994, Comm. Research) — Interpersonal Deception Theory:
  Nonverbal channels are harder to suppress under sustained cognitive load.
  As an answer progresses, vocal and facial signals progressively "leak"
  while verbal content remains controlled — the drift slope captures this.

HOW IT FITS INTO analyzer.py / main.py
---------------------------------------
main.py calls detect_conflicts_with_drift() AFTER analyze() returns,
passing both the analysis result AND the acoustic windowed object plus
the webcam time-series arrays already extracted in the /evaluate endpoint.
The returned ConflictReport now includes a .drift field (DriftResult).

CONFLICT TAXONOMY
-----------------
1. VERBAL-VOCAL conflict:
   Text lexical confidence score (high) vs acoustic nervousness (high)
   → "Your words project confidence but your voice suggests stress."

2. VERBAL-FACIAL conflict:
   Text OCEAN Extraversion (high) + DISC Dominance vs facial nervousness (high)
   → "Your answer shows assertiveness but your face shows anxiety."

3. VOCAL-FACIAL conflict:
   Acoustic nervousness (low) vs facial nervousness (high)
   → "Your voice was steady but facial cues suggest discomfort."

4. FULL ALIGNMENT (no conflict):
   All three channels agree within threshold → coherence bonus message.
   "All signals align — what you said, how you sounded, and how you
    looked told the same story. Interviewers find this highly credible."
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Conflict thresholds ───────────────────────────────────────────────────────

# A conflict is flagged when the directional gap between modalities
# exceeds these deltas (on their native 0–1 scale).
_VERBAL_VOCAL_THRESH   = 0.30   # gap between lexical confidence and acoustic nervousness
_VERBAL_FACIAL_THRESH  = 0.30   # gap between text OCEAN extraversion and facial nervousness
_VOCAL_FACIAL_THRESH   = 0.25   # gap between acoustic and facial nervousness scores
_ALIGNMENT_THRESH      = 0.20   # all within this → "full alignment"

# ── Drift thresholds ──────────────────────────────────────────────────────────

# drift_rate = linear regression slope over normalised time domain [0,1].
# Negative = coherence declining as the answer progresses.
_DRIFT_STRONG  = -0.25   # below this → "strong_drift"
_DRIFT_MILD    = -0.10   # below this → "mild_drift"; otherwise "stable"

# Minimum acoustic windows required for meaningful drift measurement.
# At 10s window / 2s hop, 3 windows ≈ 30s of audio.
_MIN_DRIFT_WINDOWS = 3

# Per-window coherence channel weights (verbal-vocal carries more weight;
# matches the _compute_coherence() weights below).
_W_VERBAL_VOCAL  = 0.55
_W_VERBAL_FACIAL = 0.45

# EAR threshold for PERCLOS proxy (fraction of closed-eye frames per window).
_EAR_THRESHOLD = 0.20

# Webcam default capture rate — must match WebcamNervousnessAnalyzer.DEFAULT_FPS.
_WEBCAM_FPS = 0.5

# Severity bands
def _severity(delta: float) -> str:
    if delta >= 0.55:
        return "high"
    if delta >= 0.35:
        return "moderate"
    return "low"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Conflict:
    type: str           # "verbal_vocal" | "verbal_facial" | "vocal_facial"
    severity: str       # "low" | "moderate" | "high"
    delta: float        # raw gap value
    headline: str       # Short 1-liner shown in UI badge
    explanation: str    # 2–3 sentence coaching explanation
    coaching_tip: str   # Specific, actionable advice
    channels: Dict      # Raw channel values that triggered this conflict


@dataclass
class DriftResult:
    """
    Temporal coherence drift profile for a single answer.

    Populated by compute_coherence_drift(); attached to ConflictReport.drift.
    All list fields share the same index — one entry per acoustic window.
    """
    timestamps:        List[float] = field(default_factory=list)   # window centre times (s)
    coherence_series:  List[float] = field(default_factory=list)   # per-window coherence [0,1]
    acoustic_series:   List[float] = field(default_factory=list)   # per-window acoustic nervousness
    facial_series:     List[float] = field(default_factory=list)   # per-window facial nervousness
    drift_slope:       float       = 0.0     # coherence / second (negative = diverging)
    drift_rate:        float       = 0.0     # slope × duration  (scale-free, threshold-comparable)
    peak_conflict_t:   float       = 0.0     # seconds — window of worst coherence
    peak_conflict_val: float       = 1.0     # coherence value at that window
    drift_label:       str         = "stable"   # "stable" | "mild_drift" | "strong_drift" | "insufficient_data"
    drift_narrative:   str         = ""
    vocal_only:        bool        = False   # True when webcam frames were unavailable
    n_windows:         int         = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConflictReport:
    conflicts: List[Conflict] = field(default_factory=list)
    alignment: bool = False             # True if all channels agree
    alignment_message: str = ""         # Shown when alignment=True
    dominant_conflict: Optional[Conflict] = None   # Highest-severity conflict
    composite_coherence: float = 0.0    # 0–1, higher = more coherent

    # Raw channel values extracted from analysis_result
    lexical_confidence: float = 0.0
    acoustic_nervousness: float = 0.0
    facial_nervousness: float = 0.0
    text_extraversion: float = 0.0      # OCEAN E score

    # Temporal drift profile (None when acoustic windowed data was not available)
    drift: Optional[DriftResult] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  CHANNEL VALUE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_channels(analysis_result: Dict) -> Dict[str, float]:
    """
    Pull the relevant channel values out of an analyze() result dict.
    All values normalised to [0, 1]:
      - lexical_confidence:   1.0 = maximally confident language
      - acoustic_nervousness: 1.0 = maximally nervous voice
      - facial_nervousness:   1.0 = maximally nervous face
      - text_extraversion:    0–1 normalised OCEAN E score
    """
    scores    = analysis_result.get("scores", {})
    nlp       = analysis_result.get("nlp", {})
    # Support two shapes:
    #   1. Internal analyzer dict — "nervousness" is a sub-dict with voice/facial/fused keys
    #   2. /evaluate API response  — "nervousness_detail" holds the same sub-dict;
    #      "nervousness" at the top level is a plain float (fused score).
    nerv_dict = analysis_result.get("nervousness", {})
    if not isinstance(nerv_dict, dict):
        # Top-level "nervousness" is a float in the API response — use nervousness_detail
        nerv_dict = analysis_result.get("nervousness_detail", {})
    ocean_raw = nlp.get("ocean_scores", {})

    # ── Lexical confidence: from scores.confidence (0–100) → 0–1
    lex_conf = scores.get("confidence", 50) / 100.0

    # ── Acoustic nervousness: already 0–1 in nervousness.voice
    acoustic_nerv = nerv_dict.get("voice", 0.3)

    # ── Facial nervousness: already 0–1 in nervousness.facial
    facial_nerv = nerv_dict.get("facial", 0.3)

    # ── Text OCEAN Extraversion: analyzer.py stores scores as 0–1 fractions.
    # Guard against legacy 0–5 output by clamping after a direct 0–1 read.
    e_raw = ocean_raw.get("Extraversion", 0.5)
    # If the value is clearly on a 0–5 scale (>1.0), normalise it; otherwise use directly.
    text_e = min(1.0, max(0.0, e_raw / 5.0 if e_raw > 1.0 else e_raw))

    # ── Fused nervousness (for composite coherence)
    fused_nerv = nerv_dict.get("fused", 0.3)

    return {
        "lexical_confidence":   round(lex_conf, 3),
        "acoustic_nervousness": round(acoustic_nerv, 3),
        "facial_nervousness":   round(facial_nerv, 3),
        "text_extraversion":    round(text_e, 3),
        "fused_nervousness":    round(fused_nerv, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONFLICT DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_verbal_vocal(channels: Dict) -> Optional[Conflict]:
    """
    Conflict: high lexical confidence + high acoustic nervousness.
    The candidate's words project calm, but their voice trembles.
    This is the most common mismatch in video interviews (Schuller 2011).
    """
    lex_conf      = channels["lexical_confidence"]
    acoustic_nerv = channels["acoustic_nervousness"]

    # Conflict direction: confident words + nervous voice
    # (also flag: nervous words + calm voice — less common, but valid)
    gap = lex_conf - (1.0 - acoustic_nerv)   # positive = words more confident than voice
    abs_gap = abs(gap)

    if abs_gap < _VERBAL_VOCAL_THRESH:
        return None

    if gap > 0:
        # Most common: words confident, voice nervous
        headline    = "Confident words, tense voice"
        explanation = (
            "Your answer contained strong, assertive language — but your vocal patterns "
            "(pitch variation, speaking rate, or pausing) signaled stress. "
            "Interviewers are trained to notice this mismatch, and it can undermine "
            "the credibility of an otherwise well-structured answer."
        )
        tip = (
            "Before your next interview, practice your answer out loud 3–5 times until "
            "the delivery feels as comfortable as the words. Record yourself and listen "
            "back — the goal is for your voice to sound as calm as your script reads."
        )
    else:
        # Inverse: words hedging, voice steady
        headline    = "Steady voice, uncertain language"
        explanation = (
            "Your voice was relatively calm and controlled, but your word choice was "
            "full of hedges ('I think', 'maybe', 'I guess'). This suggests you may "
            "know more than your words indicate — or that you were being overly modest."
        )
        tip = (
            "Replace hedge words with declarative statements. Instead of 'I think I improved "
            "the process', say 'I improved the process by [X]'. Your voice shows you can "
            "be confident — let your words match it."
        )

    return Conflict(
        type        = "verbal_vocal",
        severity    = _severity(abs_gap),
        delta       = round(abs_gap, 3),
        headline    = headline,
        explanation = explanation,
        coaching_tip= tip,
        channels    = {
            "lexical_confidence":   lex_conf,
            "acoustic_nervousness": acoustic_nerv,
            "gap":                  round(gap, 3),
        },
    )


def _detect_verbal_facial(channels: Dict) -> Optional[Conflict]:
    """
    Conflict: high OCEAN Extraversion / lexical confidence vs high facial nervousness.
    Candidate presents as bold in words but facial cues show anxiety.
    """
    text_e        = channels["text_extraversion"]
    lex_conf      = channels["lexical_confidence"]
    facial_nerv   = channels["facial_nervousness"]

    # Use a blend of text signals as "verbal assertiveness"
    verbal_signal = (text_e * 0.5 + lex_conf * 0.5)
    gap = verbal_signal - (1.0 - facial_nerv)  # positive = words assertive, face nervous
    abs_gap = abs(gap)

    if abs_gap < _VERBAL_FACIAL_THRESH:
        return None

    if gap > 0:
        headline    = "Assertive words, anxious face"
        explanation = (
            "Your answer read as confident and assertive — but your facial signals "
            "(eye contact, blink rate, or micro-expressions) indicated discomfort. "
            "In face-to-face or video interviews, nonverbal cues often carry more "
            "weight than verbal content for first impressions."
        )
        tip = (
            "Practice 'facial alignment': before answering, take one slow breath and "
            "consciously soften your brow and jaw. Maintain eye contact with the camera "
            "(not your own face on screen). A 5-second grounding pause before speaking "
            "resets facial tension more reliably than any verbal technique."
        )
    else:
        headline    = "Open face, cautious words"
        explanation = (
            "Your facial expression was relatively open and calm, but your verbal content "
            "was hesitant or understated. You may be underselling yourself — your face "
            "shows composure that your words don't reflect."
        )
        tip = (
            "Trust what your face is already doing. Pair that calm presence with more "
            "decisive language. Start your next answer with your conclusion: 'The result "
            "was X' — then explain how you got there. Interviewers remember the last "
            "thing you said, not the hedges in the middle."
        )

    return Conflict(
        type        = "verbal_facial",
        severity    = _severity(abs_gap),
        delta       = round(abs_gap, 3),
        headline    = headline,
        explanation = explanation,
        coaching_tip= tip,
        channels    = {
            "verbal_assertiveness": round(verbal_signal, 3),
            "facial_nervousness":   facial_nerv,
            "gap":                  round(gap, 3),
        },
    )


def _detect_vocal_facial(channels: Dict) -> Optional[Conflict]:
    """
    Conflict: acoustic nervousness vs facial nervousness diverge.
    One channel suggests stress, the other is calm.
    Rarer but highly informative — often indicates controlled performance anxiety.
    """
    acoustic_nerv = channels["acoustic_nervousness"]
    facial_nerv   = channels["facial_nervousness"]
    gap           = acoustic_nerv - facial_nerv
    abs_gap       = abs(gap)

    if abs_gap < _VOCAL_FACIAL_THRESH:
        return None

    if gap > 0:
        headline    = "Stressed voice, calm face"
        explanation = (
            "Your voice showed signs of stress (pitch instability, irregular pacing) "
            "but your facial expression was relatively controlled. This can happen when "
            "a candidate has practiced 'interview face' but hasn't trained their voice. "
            "Skilled interviewers notice vocal stress even when your face looks calm."
        )
        tip = (
            "Voice control is a trainable skill. Record a 2-minute practice answer, "
            "listen back on earbuds, and focus only on your pitch and pace. Aim for "
            "a steady rate of about 140 words per minute with deliberate pauses — "
            "not filled with 'um'."
        )
    else:
        headline    = "Tense face, steady voice"
        explanation = (
            "Your voice was controlled and your pacing was good, but facial cues "
            "suggested higher anxiety. This often appears in candidates who have "
            "practiced verbal delivery but haven't managed their facial tension. "
            "The voice is reassuring — the face is the gap to close."
        )
        tip = (
            "Before your next answer, do a 'face reset': raise your eyebrows briefly, "
            "then let them fall naturally. This resets the tension pattern interviewers "
            "read as anxiety. Your voice is already working for you — let your face catch up."
        )

    return Conflict(
        type        = "vocal_facial",
        severity    = _severity(abs_gap),
        delta       = round(abs_gap, 3),
        headline    = headline,
        explanation = explanation,
        coaching_tip= tip,
        channels    = {
            "acoustic_nervousness": acoustic_nerv,
            "facial_nervousness":   facial_nerv,
            "gap":                  round(gap, 3),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITE COHERENCE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_coherence(channels: Dict, conflicts: List[Conflict]) -> float:
    """
    Single 0–1 coherence score: how well do the three channels agree?
    High coherence = all signals aligned = more credible candidate impression.

    Formula:
        coherence = 1 − (weighted average of conflict deltas)

    Weights (by channel importance for interview perception):
        verbal-vocal:  0.45  (Schuller 2011 — vocal most discriminative)
        verbal-facial: 0.35
        vocal-facial:  0.20
    """
    if not conflicts:
        return 1.0

    weights = {"verbal_vocal": 0.45, "verbal_facial": 0.35, "vocal_facial": 0.20}
    conflict_map = {c.type: c.delta for c in conflicts}

    weighted_sum = sum(conflict_map.get(t, 0.0) * w for t, w in weights.items())
    coherence = max(0.0, 1.0 - weighted_sum)
    return round(coherence, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL COHERENCE DRIFT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _facial_nervousness_for_window(
    ear_values: List[float],
    yaw_values: List[float],
    window_start_s: float,
    window_end_s: float,
    fps: float = _WEBCAM_FPS,
) -> Optional[float]:
    """
    Compute a facial nervousness score [0,1] for one acoustic window, using
    only the EAR time series and yaw angles that are already extracted by
    main.py from webcam_analyzer.analyze_frames().

    Three sub-signals (lightweight version of webcam_analyzer._score_nervousness):
        (a) PERCLOS proxy  — fraction of frames with EAR < _EAR_THRESHOLD
        (b) EAR variance   — anxious candidates show high EAR instability
        (c) Yaw variance   — head movement kineme proxy (Tomashin et al. 2025)

    Returns None if < 2 frames fall in this window.
    """
    if not ear_values:
        return None

    start_f = int(math.floor(window_start_s * fps))
    end_f   = int(math.ceil(window_end_s   * fps))

    ear_win = ear_values[start_f:end_f]
    yaw_win = yaw_values[start_f:end_f] if yaw_values else []

    if len(ear_win) < 2:
        return None

    perclos      = sum(1 for e in ear_win if e < _EAR_THRESHOLD) / len(ear_win)
    ear_var_norm = min(1.0, float(np.var(ear_win)) / 0.015) if len(ear_win) >= 3 else 0.0
    yaw_var_norm = min(1.0, float(np.var(yaw_win)) / 0.02)  if len(yaw_win) >= 2 else 0.0

    facial = 0.55 * perclos + 0.25 * ear_var_norm + 0.20 * yaw_var_norm
    return round(min(1.0, max(0.0, facial)), 3)


def _fill_none(series: List[Optional[float]]) -> List[float]:
    """Forward-fill then backward-fill None entries in a list."""
    if all(v is None for v in series):
        return []
    result = list(series)
    last = None
    for i, v in enumerate(result):
        if v is not None:
            last = v
        elif last is not None:
            result[i] = last
    last = None
    for i in range(len(result) - 1, -1, -1):
        if result[i] is not None:
            last = result[i]
        elif last is not None:
            result[i] = last
    return [v if v is not None else 0.0 for v in result]


def _build_common_grid(
    acoustic_timestamps: List[float],
    acoustic_scores:     List[float],
    ear_values:          List[float],
    yaw_values:          List[float],
    window_sec:          float,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Project webcam frames (0.5 fps) onto the acoustic window grid (10s/2s hop).
    Returns (timestamps, acoustic_series, facial_series).
    facial_series is [] when no webcam data is available.
    """
    half = window_sec / 2.0
    facial: List[Optional[float]] = []

    for centre in acoustic_timestamps:
        f = _facial_nervousness_for_window(
            ear_values, yaw_values,
            max(0.0, centre - half), centre + half,
        )
        facial.append(f)

    filled = _fill_none(facial)
    return acoustic_timestamps, acoustic_scores, filled


def _coherence_series(
    acoustic_series:    List[float],
    facial_series:      List[float],
    lexical_confidence: float,
) -> List[float]:
    """
    Per-window coherence[i] = 1 − weighted(gap_vv[i], gap_vf[i]).

    gap_vv[i] = |lex_conf − (1 − acoustic_nerv[i])|
    gap_vf[i] = |lex_conf − (1 − facial_nerv[i])|   (omitted if no facial)
    """
    vocal_only = len(facial_series) < len(acoustic_series)
    out = []
    for i, a_nerv in enumerate(acoustic_series):
        gap_vv = abs(lexical_confidence - (1.0 - a_nerv))
        if vocal_only:
            conflict = gap_vv
        else:
            gap_vf   = abs(lexical_confidence - (1.0 - facial_series[i]))
            conflict = _W_VERBAL_VOCAL * gap_vv + _W_VERBAL_FACIAL * gap_vf
        out.append(round(max(0.0, 1.0 - conflict), 3))
    return out


def _drift_slope(
    coherence: List[float],
    timestamps: List[float],
    duration_sec: float,
) -> Tuple[float, float]:
    """
    Linear regression on coherence vs normalised time → (slope_per_sec, drift_rate).
    drift_rate = slope over [0,1] normalised domain; threshold-comparable across answers.
    """
    x = np.array(timestamps, dtype=float)
    x_norm = (x - x[0]) / max(duration_sec, 1.0)
    y = np.array(coherence, dtype=float)
    coeffs = np.polyfit(x_norm, y, deg=1)
    slope_norm = float(coeffs[0])                          # Δcoh over normalised [0,1]
    slope_sec  = slope_norm / max(duration_sec, 1.0)       # Δcoh / second
    return round(slope_sec, 5), round(slope_norm, 4)


def _drift_label_and_narrative(
    drift_rate: float,
    peak_t: float,
    duration_sec: float,
    vocal_only: bool,
) -> Tuple[str, str]:
    pos_pct     = str(int((peak_t / max(duration_sec, 1.0)) * 100))
    channel_str = "voice" if vocal_only else "voice and face"

    if drift_rate < _DRIFT_STRONG:
        label = "strong_drift"
        if int(pos_pct) < 40:
            narrative = (
                f"Your answer started well but your nonverbal signals began to diverge "
                f"from your words early — around the {pos_pct}% mark. "
                f"This often happens when the opening is scripted but the middle requires "
                f"improvisation. Practice your full answer out loud, not just the opening."
            )
        else:
            narrative = (
                f"The first part of your answer was well-aligned, but from about the "
                f"{pos_pct}% mark onward, your {channel_str} began to contradict your "
                f"verbal content — a common pattern of accumulating stress. "
                f"Try breaking long answers into shorter, rehearsed segments."
            )
    elif drift_rate < _DRIFT_MILD:
        label = "mild_drift"
        narrative = (
            f"Your answer showed some gradual divergence between your words and "
            f"your {channel_str}, peaking around the {pos_pct}% mark. "
            f"This is mild and likely unnoticeable to most interviewers, but a few "
            f"extra practice runs would smooth it out completely."
        )
    else:
        label = "stable"
        narrative = (
            "Your verbal and nonverbal signals stayed well-aligned throughout this answer. "
            "The consistency between what you said and how you sounded and looked "
            "adds to the credibility of your response — interviewers weigh this heavily."
        )
    return label, narrative


def compute_coherence_drift(
    acoustic_windowed,              # WindowedAnalysisResult from acoustic_analyser.analyse_windowed()
    ear_values:   List[float],      # webcam_analyzer ear_values ([] if unavailable)
    yaw_values:   List[float],      # webcam_analyzer yaw_values ([] if unavailable)
    duration_sec: float,            # total answer duration in seconds
    lexical_confidence: float,      # from _extract_channels() — whole-answer verbal baseline
) -> DriftResult:
    """
    Compute temporal coherence drift for one answer.

    Called by detect_conflicts_with_drift(); not normally called directly.

    Parameters
    ----------
    acoustic_windowed : WindowedAnalysisResult
        Returned by acoustic_analyser.analyse_windowed(audio_path).
        Must have .timestamps (List[float]) and .trajectory (List[float]).
    ear_values, yaw_values : List[float]
        Per-frame facial time-series already extracted in main.py /evaluate.
        Pass [] when webcam was unavailable — drift falls back to vocal-only.
    duration_sec : float
        Total answer duration; used to compute drift_rate and narratives.
    lexical_confidence : float
        Whole-answer verbal baseline [0,1] from _extract_channels().
    """
    try:
        timestamps      = list(acoustic_windowed.timestamps)
        acoustic_scores = list(acoustic_windowed.trajectory)
        window_sec      = float(getattr(acoustic_windowed, "window_sec", 10.0))
    except AttributeError:
        logger.warning("[CoherenceDrift] acoustic_windowed missing attributes; skipping drift.")
        return DriftResult(
            drift_label      = "insufficient_data",
            drift_narrative  = "Acoustic windowed data was not available for drift analysis.",
        )

    if len(timestamps) < _MIN_DRIFT_WINDOWS:
        return DriftResult(
            drift_label     = "insufficient_data",
            drift_narrative = (
                f"Answer too short for drift analysis ({len(timestamps)} window(s); "
                f"need ≥ {_MIN_DRIFT_WINDOWS}, i.e. ~{_MIN_DRIFT_WINDOWS * window_sec:.0f}s)."
            ),
            n_windows = len(timestamps),
        )

    timestamps, acoustic_series, facial_series = _build_common_grid(
        acoustic_timestamps = timestamps,
        acoustic_scores     = acoustic_scores,
        ear_values          = ear_values,
        yaw_values          = yaw_values,
        window_sec          = window_sec,
    )

    vocal_only = len(facial_series) < len(acoustic_series)
    coh_series = _coherence_series(acoustic_series, facial_series, lexical_confidence)
    slope, drift_rate = _drift_slope(coh_series, timestamps, duration_sec)

    min_idx   = int(np.argmin(coh_series))
    peak_t    = timestamps[min_idx]
    peak_val  = coh_series[min_idx]

    label, narrative = _drift_label_and_narrative(drift_rate, peak_t, duration_sec, vocal_only)

    logger.info(
        f"[CoherenceDrift] {len(timestamps)} windows | drift_rate={drift_rate:.3f} "
        f"| label={label} | peak_t={peak_t:.1f}s"
    )

    return DriftResult(
        timestamps        = [round(t, 2) for t in timestamps],
        coherence_series  = coh_series,
        acoustic_series   = [round(v, 3) for v in acoustic_series],
        facial_series     = [round(v, 3) for v in facial_series],
        drift_slope       = slope,
        drift_rate        = drift_rate,
        peak_conflict_t   = round(peak_t, 2),
        peak_conflict_val = round(peak_val, 3),
        drift_label       = label,
        drift_narrative   = narrative,
        vocal_only        = vocal_only,
        n_windows         = len(timestamps),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_conflicts(analysis_result: Dict) -> ConflictReport:
    """
    Post-process an analyze() result dict and return a ConflictReport
    (snapshot only — no temporal drift).

    For the full pipeline including drift, use detect_conflicts_with_drift().

    Parameters
    ----------
    analysis_result : dict
        The full dict returned by InterviewAnalyzer.analyze().

    Returns
    -------
    ConflictReport
        .conflicts           — list of detected conflicts, sorted by severity
        .alignment           — True when all channels agree
        .alignment_message   — encouraging message for aligned candidates
        .dominant_conflict   — highest-severity conflict (for UI badge)
        .composite_coherence — 0–1, higher = more coherent
        .drift               — None (populated by detect_conflicts_with_drift)
    """
    channels = _extract_channels(analysis_result)

    c_vv = _detect_verbal_vocal(channels)
    c_vf = _detect_verbal_facial(channels)
    c_af = _detect_vocal_facial(channels)

    conflicts = [c for c in [c_vv, c_vf, c_af] if c is not None]

    severity_order = {"high": 0, "moderate": 1, "low": 2}
    conflicts.sort(key=lambda c: (severity_order[c.severity], -c.delta))

    coherence = _compute_coherence(channels, conflicts)

    alignment = len(conflicts) == 0
    alignment_message = ""
    if alignment:
        alignment_message = (
            "All signals aligned — what you said, how you sounded, and how you looked "
            "told the same story. Interviewers find this highly credible. "
            "This kind of consistency is rare and is often the deciding factor between "
            "equally-qualified candidates."
        )

    report = ConflictReport(
        conflicts            = conflicts,
        alignment            = alignment,
        alignment_message    = alignment_message,
        dominant_conflict    = conflicts[0] if conflicts else None,
        composite_coherence  = coherence,
        lexical_confidence   = channels["lexical_confidence"],
        acoustic_nervousness = channels["acoustic_nervousness"],
        facial_nervousness   = channels["facial_nervousness"],
        text_extraversion    = channels["text_extraversion"],
        drift                = None,
    )

    logger.info(
        f"[ConflictDetector] {len(conflicts)} conflict(s) | "
        f"coherence={coherence:.2f} | "
        f"dominant={'none' if not conflicts else conflicts[0].type}"
    )

    return report


def detect_conflicts_with_drift(
    analysis_result:   Dict,
    acoustic_windowed,              # WindowedAnalysisResult | None
    ear_values:        List[float], # webcam_analyzer["ear_values"] — [] if unavailable
    yaw_values:        List[float], # webcam_analyzer["yaw_values"] — [] if unavailable
    duration_sec:      float = 60.0,
) -> ConflictReport:
    """
    Full pipeline entry point called by main.py /evaluate.

    Runs snapshot conflict detection AND temporal coherence drift in one call,
    returning a single ConflictReport with .drift populated.

    Parameters
    ----------
    analysis_result : dict
        Full dict from InterviewAnalyzer.analyze().
    acoustic_windowed : WindowedAnalysisResult | None
        From acoustic_analyser.analyse_windowed(audio_path).
        Pass None when no audio file is available — drift will be skipped.
    ear_values : List[float]
        Per-frame EAR time-series from webcam_analyzer["ear_values"].
        Pass [] when webcam was unavailable (drift falls back to vocal-only).
    yaw_values : List[float]
        Per-frame head-yaw from webcam_analyzer["yaw_values"].
        Pass [] when webcam was unavailable.
    duration_sec : float
        Total answer duration in seconds.
    """
    report = detect_conflicts(analysis_result)

    if acoustic_windowed is not None:
        drift = compute_coherence_drift(
            acoustic_windowed  = acoustic_windowed,
            ear_values         = ear_values,
            yaw_values         = yaw_values,
            duration_sec       = duration_sec,
            lexical_confidence = report.lexical_confidence,
        )
    else:
        drift = DriftResult(
            drift_label     = "insufficient_data",
            drift_narrative = "No audio file was available for drift analysis.",
        )

    report.drift = drift
    return report


def detect_conflicts_dict(analysis_result: Dict) -> Dict:
    """Convenience wrapper — snapshot only, no drift. Returns dict."""
    return detect_conflicts(analysis_result).to_dict()


def detect_conflicts_with_drift_dict(
    analysis_result:   Dict,
    acoustic_windowed,
    ear_values:        List[float],
    yaw_values:        List[float],
    duration_sec:      float = 60.0,
) -> Dict:
    """
    Convenience wrapper for main.py /evaluate — returns dict with .drift included.
    This is the function to import and call in main.py.
    """
    return detect_conflicts_with_drift(
        analysis_result, acoustic_windowed, ear_values, yaw_values, duration_sec
    ).to_dict()