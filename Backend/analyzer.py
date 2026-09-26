"""
analyzer_v3.py — Aura AI | Interview Analysis Engine (v3.0)
============================================================
Upgrades over v2.0:

  [1] OCEAN Context Gating (v3.0)
      Keyword hits only count when they appear within ±30 words of an
      Action or Result STAR signal. Prevents self-labeling inflation.
      Research: InterviewBERT (Frontiers Psychol. 2022) — trait-keyword
      proximity to action clauses is a stronger predictor of genuine
      trait expression (r=0.37 avg HEXACO) than bare keyword hits.

      OCEAN trait confidence weights (InterviewBERT, n=58,000):
        Openness        → trust weight 1.00  (r=0.45 — highest validity)
        Conscientiousness → 0.90             (r=0.42)
        Extraversion    → 0.80              (r=0.39)
        Neuroticism_inv → 0.75              (r=0.36)
        Agreeableness   → 0.62              (r=0.28 — lowest validity)

  [2] Multi-session OCEAN Aggregation (v3.0)
      `compute_session_ocean()` averages trait scores across answers
      and modulates each trait by its cross-answer consistency (σ):
        low_σ  (< 0.5) → +0.3 consistency bonus
        high_σ (> 1.5) → −0.3 consistency penalty
      Stable trait expression across questions is a stronger hiring
      signal than a single-answer spike.

  [3] Depth Score v3.0 — Lexical Density + Clause Complexity
      `_compute_depth_score_v3()` replaces the pure word-count proxy.
      Three sub-dimensions:
        (a) WC-depth  — the existing piecewise word-count curve (kept
            for backward compatibility, weight 0.45)
        (b) Lexical density — content_words / total_words (weight 0.35)
            Content words = nouns (NNP heuristic), verbs, adjectives,
            adverbs; stop-word free. Rewards dense technical language.
        (c) Clause complexity — subordinate clause density (weight 0.20)
            Counts "because|which|whereas|therefore|although|however|
            since|given that|in order to|so that|due to|as a result"
            per 100 words. Expert answers average 3–5 subordinators
            per 100 words; padding answers average < 1.
      Research:
        McNamara et al. (2010, Reading & Writing) — lexical density
        distinguishes expert from novice discourse better than length.
        Biber (1988, Variation Across Speech and Writing) — subordinate
        clause density is the strongest syntactic marker of informational
        density in professional speech registers.

  [4] Segmented STAR Detection v3.0
      `_compute_star_score_v3()` splits the answer into quarters and
      checks STAR component dominance per segment, rather than using
      only the first regex match position.
        — Each STAR component has an "expected" quarter: S→Q1, T→Q1–Q2,
          A→Q2–Q3, R→Q3–Q4.
        — "In-position" presence earns full credit; "out-of-position"
          presence earns 0.5× credit.
        — Order bonus (was binary ±0.5) is now a continuous gradient
          from 0.0 to 0.75 based on how many components appear in their
          expected quarter.
        — STAR component density (hits per 200-char window) replaces
          binary presence, catching answers that revisit components.

  [5] Grammar Proxy v3.0 — Continuous, Three-Dimensional
      `_compute_grammar_score_v3()` replaces the binary 85/65 score
      with a continuous 0–5 composite:
        (a) Sentence length variance — penalises robotic uniformity
            AND extreme fragmentation. Target CV = 0.5–0.8 for optimal
            rhetoric variety. (weight 0.35)
        (b) Discourse marker density — "however|therefore|in addition|
            specifically|for example|consequently|furthermore|in contrast|
            notably|namely" per 100 words. Target 2–10 per 100 words.
            (weight 0.40)
        (c) Passive voice ratio — passive constructions reduce score.
            Naim et al. IEEE 2015 MIT dataset — active voice correlates
            positively with perceived interview quality. (weight 0.25)

  [7] Dynamic SAS Fusion Weight (v3.0)
      `_dynamic_sas_weight(wc)` computes a length-aware embedding trust
      weight rather than using the fixed 0.30/0.70 split from v2.0.

      Rationale: Sentence-BERT embeddings are unreliable for short inputs
      (< 50 words) — the cosine similarity clusters near random for sparse
      texts. Giving a short noisy embedding equal weight with a
      well-calibrated LLM rubric inflates or deflates scores incorrectly
      depending on chance vocabulary overlap.

      Formula (linear ramp):
        wc ≤  50  →  sas_w = 0.05  (LLM trusted at 0.95)
        wc ≥ 150  →  sas_w = 0.30  (standard split — ceiling matches v2.0)
        50 < wc < 150  →  linear interpolation between anchors

      Research basis:
        Reimers & Gurevych (EMNLP 2019) — Sentence-BERT Pearson r drops
        from 0.878 (> 20-token pairs) to 0.71 (≤ 8-token pairs) on STS-B,
        a 19-point reliability gap driven by insufficient context for pooling.
        Chandrasekaran & Mago (ACM CSUR 2022) — embedding models underperform
        TF-IDF on < 50 token technical Q&A because vocabulary specificity
        dominates distributional context at that length.

      Integration:
        `_groq_relevance()` calls `_dynamic_sas_weight(wc)` and passes the
        result into `SASScorer.fuse_with_llm(sas_weight=..., llm_weight=...)`.
        Zero changes to sas_scorer.py — those params were already overrideable.
        `relevance_source` label includes wc for production log auditing.

BACKWARD COMPATIBILITY
----------------------
All v2.0 public function signatures are preserved. New v3 functions are
additive suffixes (_v3). The main `_full_evaluate()` and
`InterviewAnalyzer.analyze()` switch to v3 functions automatically.
Old v2 functions remain importable for regression testing.

FORMULA CHANGE SUMMARY (drop-in replacement points)
-----------------------------------------------------
  _compute_depth_score(wc, q_type)            → _compute_depth_score_v3(wc, q_type, text)
  _compute_star_score(text_lower)             → _compute_star_score_v3(text_lower)
  _compute_ocean(text_lower)                  → _compute_ocean_v3(text_lower)
  grammar_score = 85/65 binary               → _compute_grammar_score_v3(answer)
  Multi-session OCEAN aggregation             → compute_session_ocean(per_answer_oceans)
  SASScorer.fuse_with_llm(fixed 0.30/0.70)  → fuse_with_llm(*_dynamic_sas_weight(wc))
"""

from __future__ import annotations

import asyncio
import os
import re
import json
import math
from typing import Dict, List, Optional, Tuple

from groq import AsyncGroq
from sas_scorer import sas_scorer, SASScorer
from acoustic_nervousness import acoustic_analyser
from cultural_adapter import (
    adapt_weights_for_culture,
    detect_cultural_context,
    get_ocean_keywords_for_context,
)


# ══════════════════════════════════════════════════════════════════════════════
#  RE-IMPORT ALL CONSTANTS FROM v2.0 (unchanged)
#  (In production: keep these in a shared constants.py and import from both)
# ══════════════════════════════════════════════════════════════════════════════

FILLER_WORDS: List[str] = [
    "um", "uh", "er", "ah", "hmm", "uhm",
    "like", "you know", "sort of", "kind of", "basically", "literally",
    "i mean", "you see", "right", "okay so", "so yeah", "anyway",
    "honestly", "obviously", "clearly", "simply", "really", "very",
    "quite", "pretty much", "i guess", "stuff", "things",
]

CONFIDENCE_MARKERS = {
    "strong": [
        "i achieved", "i led", "i built", "i designed", "i implemented",
        "successfully", "resulted in", "improved", "increased", "delivered",
        "i am confident", "specifically", "measurably", "demonstrated",
    ],
    "weak": [
        "i think", "maybe", "perhaps", "i guess", "i hope", "i tried",
        "i feel like", "not sure", "probably", "might", "kind of", "sort of",
    ],
}

STAR_PATTERNS: Dict[str, str] = {
    "Situation": (
        r"\b(situation|context|background|when|once|there was|faced|encountered|"
        r"during|at the time|previously|in my previous|in that project|"
        r"working at|while at|at my last|i was working|we were|at that point)\b"
    ),
    "Task": (
        r"\b(task|goal|objective|responsible|needed to|had to|assigned|my role|"
        r"challenge|was asked|my responsibility|my job was|i was tasked|"
        r"required to|expected to|set out to|aim was|purpose was)\b"
    ),
    "Action": (
        r"\b(i did|i took|i used|implemented|developed|created|decided|solved|"
        r"built|designed|led|coordinated|introduced|refactored|optimised|"
        r"automated|proposed|initiated|established|deployed|migrated|"
        r"collaborated with|worked with|reached out|presented to|"
        r"i approached|i focused|i prioritised|i identified)\b"
    ),
    "Result": (
        r"\b(result|outcome|achieved|improved|reduced|increased|success|impact|"
        r"as a result|completed|delivered|saved|cut|boosted|grew|gained|"
        r"received|won|promoted|recognised|measurable|percent|%|"
        r"within deadline|on time|under budget|positive feedback|"
        r"successfully|ultimately|in the end|this led to)\b"
    ),
}
_STAR_ORDER = ["Situation", "Task", "Action", "Result"]

DISC_KEYWORDS: Dict[str, List[str]] = {
    "Dominance": [
        "lead", "decided", "took charge", "goal", "direct", "challenge",
        "result", "win", "fast", "control", "drove", "pushed", "owned",
        "accountable", "assertive", "competitive", "decisive", "bold",
        "vision", "strategic",
    ],
    "Influence": [
        "team", "collaborate", "communicate", "inspire", "enthusiasm",
        "motivated", "people", "fun", "support", "engaged", "presented",
        "networked", "shared", "persuaded", "encouraged", "energised",
        "positive", "relationship", "culture",
    ],
    "Steadiness": [
        "consistent", "reliable", "patient", "support", "stable", "process",
        "listen", "careful", "methodical", "thorough", "steady", "calm",
        "dependable", "systematic", "predictable", "structured", "organised",
    ],
    "Conscientiousness": [
        "accurate", "detail", "quality", "process", "data", "systematic",
        "standard", "precise", "analysis", "metrics", "documented", "reviewed",
        "validated", "tested", "verified", "researched", "planned", "tracked",
        "measured",
    ],
}

OCEAN_KEYWORDS: Dict[str, List[str]] = {
    "Openness": [
        "creative", "innovative", "novel", "explored", "experiment", "curious",
        "learned", "researched", "ideated", "brainstormed", "new approach",
        "alternative", "reimagined", "rethought", "discovery",
    ],
    "Conscientiousness": [
        "organised", "planned", "deadline", "systematic", "structured",
        "careful", "detail", "accurate", "documented", "tracked", "scheduled",
        "prioritised", "reviewed", "verified", "tested", "quality",
    ],
    "Extraversion": [
        "team", "presented", "collaborated", "led", "networked", "communicated",
        "shared", "engaged", "discussed", "facilitated", "mentored", "coached",
        "convinced", "negotiated", "proactive", "outreach",
    ],
    "Agreeableness": [
        "supported", "helped", "assisted", "listened", "understood", "empathised",
        "compromise", "cooperative", "patient", "flexible", "accommodated",
        "considered", "respected", "valued", "inclusive", "collaborative",
    ],
    "Neuroticism_inv": [
        "calm", "composed", "confident", "clear", "focused", "steady",
        "resilient", "handled", "managed", "adapted", "persevered", "overcome",
        "resolved", "constructive", "rational", "objective",
    ],
}

QUANTIFIERS: List[str] = [
    "all", "every", "each", "entire", "whole", "best", "most", "always",
    "never", "completely", "absolutely", "definitely", "certainly",
    "consistently", "thoroughly", "precisely", "exactly", "specifically",
    "particularly", "invariably", "fully", "entirely", "maximum",
]

PERCEPTUAL_WORDS: List[str] = [
    "see", "observe", "notice", "watch", "look", "view", "hear", "listen",
    "feel", "sense", "recognise", "identify", "understand", "know", "realise",
    "discover", "learn", "find", "detect", "measure", "track", "monitor",
    "analyse", "assess", "evaluate", "review", "investigate", "examine",
    "study", "explore",
]

POSITIVE_SENTIMENT: List[str] = [
    "great", "excellent", "outstanding", "successful", "effective", "efficient",
    "improved", "enhanced", "optimised", "solved", "achieved", "delivered",
    "exceeded", "innovative", "creative", "strong", "confident", "clear",
    "proud", "excited", "motivated", "passionate", "dedicated", "committed",
]

NEGATIVE_SENTIMENT: List[str] = [
    "failed", "difficult", "struggled", "problem", "issue", "mistake",
    "error", "delay", "miss", "wrong", "bad", "poor", "limited", "confused",
    "worried", "anxious", "uncertain", "unclear", "incomplete", "frustrated",
    "challenging", "blocker", "obstacle", "bottleneck", "setback",
]

WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "technical": {
        "star": 0.00, "word_cat": 0.10, "relevance": 0.40,
        "keyword": 0.25, "depth_flu": 0.20, "grammar": 0.05,
    },
    "behavioural": {
        "star": 0.35, "word_cat": 0.20, "relevance": 0.20,
        "keyword": 0.10, "depth_flu": 0.10, "grammar": 0.05,
    },
    "hr": {
        "star": 0.20, "word_cat": 0.15, "relevance": 0.25,
        "keyword": 0.10, "depth_flu": 0.25, "grammar": 0.05,
    },
}

_TYPE_MAP: Dict[str, str] = {
    "technical": "technical", "behavioural": "behavioural",
    "behavioral": "behavioural", "hr": "hr",
    "soft": "hr", "general": "hr",
}

SCORE_WEIGHTS = dict(knowledge=0.70, emotion=0.15, voice=0.15)
CONFIDENCE_WEIGHTS = dict(eye=0.25, fluency=0.25, voice=0.35, facial=0.15)
NERVOUSNESS_FUSION = dict(facial=0.35, voice=0.65)

_TIME_WINDOWS: Dict[str, Dict[str, Tuple]] = {
    "technical":   {"easy": (20,50,120,200), "medium": (35,80,180,300), "hard": (50,110,240,380)},
    "behavioural": {"easy": (20,50,110,180), "medium": (35,75,150,240), "hard": (45,85,180,280)},
    "hr":          {"easy": (15,40,90,150),  "medium": (20,50,110,180), "hard": (25,55,120,200)},
}

_TIME_LABELS: Dict[str, Dict[str, str]] = {
    "technical":   {"easy": "50–120 s", "medium": "80–180 s", "hard": "110–240 s"},
    "behavioural": {"easy": "50–110 s", "medium": "75–150 s", "hard":  "85–180 s"},
    "hr":          {"easy": "40–90 s",  "medium": "50–110 s", "hard":  "55–120 s"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_type(raw: str) -> str:
    return _TYPE_MAP.get(raw.lower().strip(), "technical")

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ══════════════════════════════════════════════════════════════════════════════
#  [7] DYNAMIC SAS FUSION WEIGHT (v3.0)
# ══════════════════════════════════════════════════════════════════════════════

# Word-count boundaries for SAS weight ramp
_SAS_WC_LOW   = 50    # ≤ this → use floor weight (embedding unreliable)
_SAS_WC_HIGH  = 150   # ≥ this → use ceiling weight (embedding trustworthy)
_SAS_W_FLOOR  = 0.05  # weight at/below WC_LOW  (LLM carries 0.95)
_SAS_W_CEIL   = 0.30  # weight at/above WC_HIGH (matches original fixed split)


def _dynamic_sas_weight(wc: int) -> Tuple[float, float]:
    """
    Dynamic SAS fusion weight — idea [7].

    The `all-MiniLM-L6-v2` embedding space is trained on sentence pairs and
    produces reliable cosine similarity only when both texts have sufficient
    surface area.  For very short answers (< 50 words), the embedding vector
    is dominated by 2–3 high-frequency content words and the cosine score
    becomes noisy — comparable to TF-IDF on 10-token inputs.

    Research basis
    --------------
    Reimers & Gurevych (EMNLP 2019) — Sentence-BERT evaluation on STS-B:
      Pearson r drops from 0.878 (sentence pairs > 20 tokens) to 0.71
      (pairs ≤ 8 tokens) on the same model, a 19-point reliability gap.
      Short inputs simply don't give the transformer enough context to
      produce a meaningful pooled representation.

    Chandrasekaran & Mago (ACM CSUR 2022) — systematic review of semantic
      similarity methods: embedding-based models underperform TF-IDF on
      short (< 50 token) technical Q&A pairs because vocabulary specificity
      matters more than distributional context at that length.

    Formula
    -------
    Linear ramp between two anchor points:

        wc ≤ 50  →  sas_w = 0.05   (LLM carries 0.95 — embedding ignored)
        wc ≥ 150 →  sas_w = 0.30   (standard split, same as fixed v2.0)
        50 < wc < 150 →  sas_w = 0.05 + (wc − 50) / 100 × 0.25  (linear)

    llm_w = 1.0 − sas_w  (always sums to 1.0)

    Practical effect
    ----------------
    A 20-word answer with sas=0.85 (suspiciously high for sparse text) and
    llm=0.40 fuses to:
      OLD (fixed):  0.85×0.30 + 0.40×0.70 = 0.535  ← embedding inflates score
      NEW (dynamic): 0.85×0.05 + 0.40×0.95 = 0.423 ← LLM judgment dominates

    A 200-word answer with sas=0.85 and llm=0.40:
      OLD: 0.535  (same regardless of length)
      NEW: 0.85×0.30 + 0.40×0.70 = 0.535  (identical — ceiling matches old)

    Parameters
    ----------
    wc : int — word count of the candidate answer

    Returns
    -------
    (sas_weight, llm_weight) : Tuple[float, float]
        Both rounded to 4 dp; always sum to 1.0.
    """
    if wc <= _SAS_WC_LOW:
        sas_w = _SAS_W_FLOOR
    elif wc >= _SAS_WC_HIGH:
        sas_w = _SAS_W_CEIL
    else:
        t = (wc - _SAS_WC_LOW) / (_SAS_WC_HIGH - _SAS_WC_LOW)   # 0.0 → 1.0
        sas_w = _SAS_W_FLOOR + t * (_SAS_W_CEIL - _SAS_W_FLOOR)

    sas_w = round(sas_w, 4)
    llm_w = round(1.0 - sas_w, 4)
    return sas_w, llm_w


# ══════════════════════════════════════════════════════════════════════════════
#  [4] STAR DETECTION v3.0 — SEGMENTED + DENSITY-BASED
# ══════════════════════════════════════════════════════════════════════════════

# Expected quarter for each STAR component (0-indexed: Q0=first 25%)
_STAR_EXPECTED_QUARTER: Dict[str, List[int]] = {
    "Situation": [0],          # should dominate first quarter
    "Task":      [0, 1],       # first half
    "Action":    [1, 2],       # middle half
    "Result":    [2, 3],       # second half
}


def _compute_star_score_v3(
    text_lower: str,
) -> Tuple[float, Dict[str, bool], float, Dict]:
    """
    Segmented STAR Detection v3.0

    Algorithm
    ---------
    1.  Minimum word count gate: answers < 30 words cannot form genuine STAR
        structure — return zero immediately to prevent filler/greeting answers
        from accidentally matching broad keyword patterns.
    2.  Split text into 4 equal quarters by character position.
    3.  For each STAR component, find ALL match positions (not just first).
    4.  Score component:
          in_position_hits  = hits in expected quarter(s)
          out_position_hits = hits in non-expected quarters
          presence          = in_position_hits > 0  (out-of-position hits alone
                              do NOT mark a component as "present" — they only
                              provide a small partial score boost up to 0.4)
          component_score   = 1.0  if in_pos ≥ 1  (full credit, boosted by out_pos)
                            = 0.0–0.4  if in_pos == 0  (partial, never "present")
    5.  Base score = sum(component_scores) / 4 * 5   (0–5 scale)
    6.  Order bonus (continuous gradient, 0–0.75):
          For each component present, award 0.1875 if it first appears in
          or before its expected quarter. Sum → max 0.75.
          (replaces binary ±0.5 from v2.0)

    Returns
    -------
    (star_score, presence_map, order_bonus, detail_dict)
      star_score   — float 0–5
      presence_map — {component: bool}  (backward-compatible)
      order_bonus  — float 0–0.75
      detail_dict  — diagnostic data for UI/logging
    """
    n = len(text_lower)
    wc_star = len(text_lower.split())

    # ── Minimum word count gate ───────────────────────────────────────────────
    # Answers shorter than 30 words cannot contain genuine STAR structure.
    # Short greetings/fillers ("hello", "okay so when") hit broad STAR
    # patterns (e.g. "when" → Situation, "i did" → Action) by accident.
    # Return zero for all components so no false structure credit is awarded.
    _STAR_MIN_WORDS = 30
    if n == 0 or wc_star < _STAR_MIN_WORDS:
        empty_detail = {
            c: {"hits": 0, "in_pos": 0, "out_pos": 0, "score": 0.0,
                "note": "answer_too_short_for_star"}
            for c in _STAR_ORDER
        }
        return 0.0, {c: False for c in _STAR_ORDER}, 0.0, empty_detail

    quarter_size = max(1, n // 4)

    def quarter_of(pos: int) -> int:
        return min(3, pos // quarter_size)

    component_scores: Dict[str, float] = {}
    presence_map: Dict[str, bool] = {}
    first_positions: Dict[str, int] = {}
    detail: Dict[str, dict] = {}

    for comp in _STAR_ORDER:
        pattern = STAR_PATTERNS[comp]
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))

        if not matches:
            component_scores[comp] = 0.0
            presence_map[comp] = False
            detail[comp] = {"hits": 0, "in_pos": 0, "out_pos": 0, "score": 0.0}
            continue

        first_positions[comp] = matches[0].start()
        expected_qs = _STAR_EXPECTED_QUARTER[comp]

        in_pos = sum(1 for m in matches if quarter_of(m.start()) in expected_qs)
        out_pos = len(matches) - in_pos

        # ── Presence requires at least one IN-POSITION hit ────────────────────
        # Bug fix: previously, out-of-position hits alone could set
        # presence=True (e.g. "when" repeated 3× in the wrong quarter scored
        # comp_score=0.5 → presence=True, awarding STAR credit for filler).
        #
        # New rule:
        #   in_pos ≥ 1  → component is genuinely present; out-of-position hits
        #                  provide a small additive boost (up to 1.0 cap).
        #   in_pos == 0 → component is NOT present; out-of-position hits give
        #                  a small partial score (max 0.4) but presence=False,
        #                  so the order-bonus loop skips this component.
        if in_pos >= 1:
            raw = _clamp(in_pos + 0.5 * out_pos, 0, max(1, in_pos + out_pos))
            comp_score = _clamp(raw / max(1, in_pos + out_pos), 0.0, 1.0)
            is_present = True
        else:
            # Out-of-position-only hits: partial credit, never "present"
            comp_score = _clamp(0.4 * min(out_pos, 1), 0.0, 0.4)
            is_present = False

        component_scores[comp] = comp_score
        presence_map[comp] = is_present
        detail[comp] = {
            "hits": len(matches),
            "in_pos": in_pos,
            "out_pos": out_pos,
            "score": round(comp_score, 3),
            "present": is_present,
            "first_quarter": quarter_of(first_positions[comp]) if comp in first_positions else -1,
        }

    # Base score: sum of component scores → 0–5
    base = sum(component_scores.values()) / 4.0 * 5.0

    # Continuous order bonus (0–0.75): each component earns 0.1875
    # if its FIRST match appears in or before its latest expected quarter.
    # Only components with presence=True (in_pos ≥ 1) are eligible —
    # out-of-position-only hits do not earn order bonus.
    order_bonus = 0.0
    per_component_bonus = 0.75 / 4.0  # 0.1875 each
    for comp in _STAR_ORDER:
        if not presence_map.get(comp, False):   # skip non-present components
            continue
        if comp not in first_positions:
            continue
        expected_qs = _STAR_EXPECTED_QUARTER[comp]
        first_q = quarter_of(first_positions[comp])
        # In or before the last expected quarter → full bonus
        # One quarter late → half bonus
        # Two+ quarters late → no bonus
        latest_expected = max(expected_qs)
        if first_q <= latest_expected:
            order_bonus += per_component_bonus
        elif first_q == latest_expected + 1:
            order_bonus += per_component_bonus * 0.5

    star_score = round(_clamp(base + order_bonus, 0.0, 5.0), 3)
    return star_score, presence_map, round(order_bonus, 3), detail


# ══════════════════════════════════════════════════════════════════════════════
#  [3] DEPTH SCORE v3.0 — LEXICAL DENSITY + CLAUSE COMPLEXITY
# ══════════════════════════════════════════════════════════════════════════════

# English stop words (lightweight, no NLTK dependency)
_STOP_WORDS = frozenset([
    "a", "an", "the", "and", "but", "or", "so", "yet", "for", "nor",
    "in", "on", "at", "to", "of", "up", "as", "by", "is", "it", "its",
    "be", "am", "are", "was", "were", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "that", "this", "these",
    "those", "then", "than", "there", "their", "they", "them", "with",
    "from", "into", "onto", "upon", "about", "which", "while", "when",
    "where", "who", "whom", "whose", "what", "how", "if", "not", "no",
    "nor", "also", "just", "very", "too", "more", "most", "such", "each",
    "both", "all", "any", "my", "your", "his", "her", "our", "we", "i",
    "me", "he", "she", "you", "us", "him", "they", "them", "it", "its",
    "get", "got", "go", "went", "come", "came", "know", "think", "say",
    "said", "make", "made", "see", "take", "taken", "use", "used",
])

# Subordinate clause markers (Biber 1988 — informational density indicators)
_SUBORDINATORS = re.compile(
    r"\b(because|which|whereas|therefore|although|however|since|given that|"
    r"in order to|so that|due to|as a result|consequently|furthermore|"
    r"nevertheless|notwithstanding|provided that|inasmuch|insofar|"
    r"in contrast|on the other hand|specifically|for instance|for example|"
    r"in particular|notably|that is|such that|even though|even if|"
    r"as long as|assuming that|in addition|rather than)\b",
    re.IGNORECASE,
)


def _lexical_density(words: List[str]) -> float:
    """
    Content word ratio — non-stop-word tokens / total tokens.

    Research: McNamara et al. (2010) — lexical density > 0.55 characterises
    expert professional discourse; < 0.40 characterises filler-heavy speech.

    Returns a 0–5 score:
      density < 0.30 → 1.0 (very sparse)
      density 0.30–0.45 → linear ramp 1→3
      density 0.45–0.65 → linear ramp 3→5 (optimal zone)
      density > 0.65 → capped at 5.0 (academic level — not penalised)
    """
    if not words:
        return 1.0
    content = sum(1 for w in words if w not in _STOP_WORDS and len(w) > 2)
    density = content / len(words)

    if density < 0.30:
        score = 1.0
    elif density < 0.45:
        score = 1.0 + (density - 0.30) / 0.15 * 2.0
    elif density <= 0.65:
        score = 3.0 + (density - 0.45) / 0.20 * 2.0
    else:
        score = 5.0

    return round(_clamp(score, 1.0, 5.0), 3)


def _clause_complexity_score(text: str, wc: int) -> float:
    """
    Subordinate clause density → 0–5 score.

    Research: Biber (1988) — subordinate clauses per 100 words:
      < 1.0   → minimal structured reasoning → score 1.0
      1–3     → adequate → linear ramp 1→3
      3–5     → expert range → linear ramp 3→5
      > 5     → may indicate run-on phrasing → cap at 5.0

    Returns 0–5.
    """
    if wc < 20:
        return 1.0  # too short to assess

    hits = len(_SUBORDINATORS.findall(text))
    density_per_100 = (hits / max(1, wc)) * 100.0

    if density_per_100 < 1.0:
        score = 1.0
    elif density_per_100 < 3.0:
        score = 1.0 + (density_per_100 - 1.0) / 2.0 * 2.0
    elif density_per_100 <= 5.0:
        score = 3.0 + (density_per_100 - 3.0) / 2.0 * 2.0
    else:
        score = 5.0  # don't penalise verbosity if clauses are complex

    return round(_clamp(score, 1.0, 5.0), 3)


def _compute_depth_score(wc: int, q_type: str) -> float:
    """v2.0 word-count piecewise depth — kept for backward compat."""
    t = q_type.lower()
    if t == "technical":
        if wc < 50:          depth = wc / 50.0
        elif wc < 150:       depth = 1.0 + (wc - 50) / 100.0 * 3.0
        elif wc <= 350:      depth = 4.0 + (wc - 150) / 200.0 * 1.0
        elif wc <= 500:      depth = 5.0 - (wc - 350) / 150.0 * 1.0
        else:                depth = max(2.0, 4.0 - (wc - 500) / 150.0 * 2.0)
    elif t == "hr":
        if wc < 40:          depth = wc / 40.0
        elif wc < 100:       depth = 1.0 + (wc - 40) / 60.0 * 3.0
        elif wc <= 175:      depth = 4.0 + (wc - 100) / 75.0 * 1.0
        elif wc <= 280:      depth = 5.0 - (wc - 175) / 105.0 * 1.0
        else:                depth = max(2.0, 4.0 - (wc - 280) / 100.0 * 2.0)
    else:  # behavioural
        if wc < 40:          depth = wc / 40.0
        elif wc < 120:       depth = 1.0 + (wc - 40) / 80.0 * 3.0
        elif wc <= 200:      depth = 4.0 + (wc - 120) / 80.0 * 1.0
        elif wc <= 300:      depth = 5.0 - (wc - 200) / 100.0 * 1.0
        else:                depth = max(2.0, 4.0 - (wc - 300) / 100.0 * 2.0)
    return round(float(depth), 2)


def _compute_depth_score_v3(wc: int, q_type: str, text: str) -> Tuple[float, Dict]:
    """
    Depth Score v3.0

    Formula
    -------
    depth_v3 = wc_depth  × 0.45
             + lex_density × 0.35
             + clause_cx   × 0.20

    The WC piecewise curve (0.45 weight) preserves calibration from v2.0
    while lexical density and clause complexity add qualitative signal.

    Returns
    -------
    (depth_score, detail_dict)
    """
    words_lower = text.lower().split()

    wc_depth_sc  = _compute_depth_score(wc, q_type)
    lex_sc       = _lexical_density(words_lower)
    clause_sc    = _clause_complexity_score(text, wc)

    depth_v3 = round(_clamp(
        wc_depth_sc * 0.45 + lex_sc * 0.35 + clause_sc * 0.20,
        0.5, 5.0,
    ), 2)

    return depth_v3, {
        "wc_depth":        round(wc_depth_sc, 3),
        "lexical_density": round(lex_sc, 3),
        "clause_complexity": round(clause_sc, 3),
        "depth_v3":        depth_v3,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  [5] GRAMMAR PROXY v3.0 — CONTINUOUS THREE-DIMENSIONAL
# ══════════════════════════════════════════════════════════════════════════════

# Discourse transition markers (strong = multi-word; weak = single)
_DISCOURSE_STRONG = re.compile(
    r"\b(in addition|as a result|for example|for instance|in contrast|"
    r"on the other hand|in particular|that is to say|due to|in other words|"
    r"in summary|to illustrate|to elaborate|in conclusion|with that said|"
    r"building on that|more specifically)\b",
    re.IGNORECASE,
)
_DISCOURSE_WEAK = re.compile(
    r"\b(however|therefore|furthermore|consequently|moreover|specifically|"
    r"notably|namely|additionally|similarly|alternatively|subsequently|"
    r"nonetheless|meanwhile|accordingly)\b",
    re.IGNORECASE,
)

# Passive voice heuristic: "was/were/been/being + past participle"
_PASSIVE_PATTERN = re.compile(
    r"\b(was|were|been|being|is|are|had been)\s+\w+ed\b",
    re.IGNORECASE,
)


def _compute_grammar_score_v3(answer: str) -> Tuple[float, Dict]:
    """
    Grammar Proxy v3.0

    Three sub-dimensions → weighted composite on 0–5 scale.

    (a) Sentence length variance (weight 0.35)
        Target CV (σ/μ of sentence word-counts) = 0.5–0.8.
        — CV < 0.2  → robotic uniformity → low score
        — CV 0.2–0.5 → adequate variety → mid score
        — CV 0.5–0.8 → optimal rhetorical variety → 5.0
        — CV > 0.8  → erratic fragmentation → decreasing score
        Research: Schuller et al. (IEEE TAC 2011) — sentence-length variance
        is the strongest text-level proxy for prosodic rhythm quality.

    (b) Discourse marker density (weight 0.40)
        (strong_hits × 1.5 + weak_hits) / (wc / 100)
        Target: 2–6 discourse events per 100 words.
        Research: Tits et al. (ACM ICMI 2018) — discourse connective density
        is a significant predictor of perceived answer quality (r=0.48).

    (c) Passive voice ratio (weight 0.25)
        passive_constructions / sentence_count
        — < 0.10 → predominantly active → 5.0
        — 0.10–0.25 → mild passive use → 4.0
        — 0.25–0.50 → moderate → 2.5
        — > 0.50 → dominantly passive → 1.5
        Research: Naim et al. (IEEE Trans. Affect. Comput. 2015) — active voice
        correlates positively with perceived interview quality in the MIT dataset.

    Returns
    -------
    (grammar_score 0–5, detail_dict)
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer.strip())
                 if len(s.strip()) > 5]
    wc = max(len(answer.split()), 1)

    # ── (a) Sentence length variance ─────────────────────────────────────────
    if len(sentences) >= 3:
        s_lens = [len(s.split()) for s in sentences]
        mean_sl = sum(s_lens) / len(s_lens)
        if mean_sl > 0:
            variance = sum((x - mean_sl) ** 2 for x in s_lens) / len(s_lens)
            cv = (variance ** 0.5) / mean_sl
        else:
            cv = 0.0

        if cv < 0.20:
            sl_score = 1.0 + cv / 0.20 * 1.5          # 1.0–2.5 (too uniform)
        elif cv < 0.50:
            sl_score = 2.5 + (cv - 0.20) / 0.30 * 1.5  # 2.5–4.0 (adequate)
        elif cv <= 0.80:
            sl_score = 4.0 + (cv - 0.50) / 0.30 * 1.0  # 4.0–5.0 (optimal)
        else:
            sl_score = max(2.0, 5.0 - (cv - 0.80) / 0.40 * 2.0)  # 5.0→2.0 (fragmented)
        sl_score = round(_clamp(sl_score, 1.0, 5.0), 3)
    elif len(sentences) == 2:
        sl_score = 2.5
        cv = 0.0
    else:
        sl_score = 1.0
        cv = 0.0

    # ── (b) Discourse marker density ─────────────────────────────────────────
    strong_hits = len(_DISCOURSE_STRONG.findall(answer))
    weak_hits   = len(_DISCOURSE_WEAK.findall(answer))
    # Weighted hits per 100 words (strong markers worth 1.5×)
    dm_density = (strong_hits * 1.5 + weak_hits) / (wc / 100.0)

    if dm_density < 0.5:
        dm_score = 1.0 + dm_density / 0.5 * 1.0      # 1.0–2.0 (few connectives)
    elif dm_density < 2.0:
        dm_score = 2.0 + (dm_density - 0.5) / 1.5 * 2.0  # 2.0–4.0 (adequate)
    elif dm_density <= 10.0:
        dm_score = 4.0 + (dm_density - 2.0) / 8.0 * 1.0  # 4.0–5.0 (expert, wider window)
    else:
        dm_score = max(4.0, 5.0 - (dm_density - 10.0) / 5.0)  # gentle plateau above 10
    dm_score = round(_clamp(dm_score, 1.0, 5.0), 3)

    # ── (c) Passive voice ratio ───────────────────────────────────────────────
    passive_hits = len(_PASSIVE_PATTERN.findall(answer))
    passive_ratio = passive_hits / max(1, len(sentences))

    if passive_ratio < 0.10:
        passive_score = 5.0
    elif passive_ratio < 0.25:
        passive_score = 4.0
    elif passive_ratio < 0.50:
        passive_score = 2.5
    else:
        passive_score = 1.5
    passive_score = round(passive_score, 3)

    # ── Composite ─────────────────────────────────────────────────────────────
    grammar_score = round(_clamp(
        sl_score     * 0.35 +
        dm_score     * 0.40 +
        passive_score * 0.25,
        1.0, 5.0,
    ), 2)

    return grammar_score, {
        "sentence_length_cv":   round(cv, 3),
        "sl_score":             sl_score,
        "discourse_density":    round(dm_density, 3),
        "dm_score":             dm_score,
        "passive_ratio":        round(passive_ratio, 3),
        "passive_score":        passive_score,
        "grammar_v3":           grammar_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  [1] OCEAN CONTEXT GATING v3.0
# ══════════════════════════════════════════════════════════════════════════════

# InterviewBERT (Frontiers Psychol. 2022, n=58,000) trait confidence weights
# — based on correlation between language-inferred and self-rated HEXACO scores
_OCEAN_CONFIDENCE: Dict[str, float] = {
    "Openness":           1.00,  # r=0.45 — most reliable trait from text
    "Conscientiousness":  0.90,  # r=0.42
    "Extraversion":       0.80,  # r=0.39
    "Neuroticism_inv":    0.75,  # r=0.36
    "Agreeableness":      0.62,  # r=0.28 — least reliable; down-weighted
}

# STAR component patterns for context gating (Action or Result triggers context)
_AR_PATTERN = re.compile(
    STAR_PATTERNS["Action"] + "|" + STAR_PATTERNS["Result"],
    re.IGNORECASE,
)

_CONTEXT_WINDOW = 30   # words either side of a keyword hit


def _find_ar_positions(text_lower: str) -> List[int]:
    """Return list of word-index positions of Action/Result signal words."""
    words = text_lower.split()
    text_joined = " ".join(words)
    ar_positions_chars = [m.start() for m in _AR_PATTERN.finditer(text_joined)]

    # Convert char positions → approximate word indices
    word_positions = []
    running = 0
    char_to_word = {}
    for i, w in enumerate(words):
        char_to_word[running] = i
        running += len(w) + 1

    for cp in ar_positions_chars:
        # find closest char boundary
        closest = min(char_to_word.keys(), key=lambda k: abs(k - cp))
        word_positions.append(char_to_word[closest])

    return sorted(set(word_positions))


def _is_in_context(kw_word_idx: int, ar_positions: List[int],
                   window: int = _CONTEXT_WINDOW) -> bool:
    """
    Return True if kw_word_idx is within ±window words of any A/R signal.
    """
    return any(abs(kw_word_idx - ar) <= window for ar in ar_positions)


def _compute_ocean_v3(
    text_lower: str,
    keywords: Optional[List[str]] = None,
    ocean_keywords: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, float], Dict]:
    """
    OCEAN Context Gating v3.0

    Algorithm
    ---------
    1. Find all Action/Result signal positions (word indices).
    2. For each OCEAN trait keyword:
         — Find all occurrences in the text (word-level search)
         — For each occurrence, check if it falls within ±30 words
           of an A/R signal position
         — Count: context_hits (near A/R) and isolated_hits (elsewhere)
    3. Score formula:
         effective_hits = context_hits + isolated_hits × 0.25
         raw_score      = 1.0 + effective_hits × 0.8    (v2.0 formula)
         confidence_adj = raw_score × _OCEAN_CONFIDENCE[trait]
         final_score    = clamp(confidence_adj, 1.0, 5.0)

    The 0.25 weight on isolated hits is intentional: a candidate who uses
    "organised" once next to a concrete Action clause contributes 4× more
    than one who uses it in a self-labeling opening ("I'm a very organised
    person...") with no concrete evidence.

    Cultural adaptation
    -------------------
    When ocean_keywords is provided (from get_ocean_keywords_for_context()),
    it replaces the module-level OCEAN_KEYWORDS for this call. For LC
    candidates it will be identical to OCEAN_KEYWORDS; for HC candidates it
    will be the merged LC+HC bank. All other logic is unchanged.

    Research
    --------
    InterviewBERT (Dai et al., Frontiers Psychol. 2022) — personality
    inferred from 58,000 interview answers using contextual BERT embeddings.
    Average r=0.37 between language-inferred and self-rated HEXACO traits;
    Agreeableness lowest (r=0.28), Openness highest (r=0.45).

    Parameters
    ----------
    text_lower     : str — lowercased transcript
    keywords       : Optional[List[str]] — domain keywords (reserved)
    ocean_keywords : Optional[Dict[str, List[str]]] — culture-adapted keyword
                     bank from get_ocean_keywords_for_context(). If None,
                     falls back to the module-level OCEAN_KEYWORDS.

    Returns
    -------
    (ocean_scores, detail_dict)
    """
    # Use culture-adapted bank if provided, else fall back to module-level default
    _kw_bank = ocean_keywords if ocean_keywords is not None else OCEAN_KEYWORDS

    words = text_lower.split()
    ar_positions = _find_ar_positions(text_lower)
    has_ar_signals = len(ar_positions) > 0

    ocean_scores: Dict[str, float] = {}
    detail: Dict[str, dict] = {}

    for trait, kw_list in _kw_bank.items():
        context_hits = 0
        isolated_hits = 0
        hit_details = []

        for kw in kw_list:
            kw_words = kw.split()
            kw_len = len(kw_words)

            # Find word-level occurrences of (possibly multi-word) keyword
            for idx in range(len(words) - kw_len + 1):
                if words[idx:idx + kw_len] == kw_words:
                    in_context = _is_in_context(idx, ar_positions) if has_ar_signals else False
                    if in_context:
                        context_hits += 1
                        hit_details.append({"kw": kw, "word_idx": idx, "in_context": True})
                    else:
                        isolated_hits += 1
                        hit_details.append({"kw": kw, "word_idx": idx, "in_context": False})

        effective_hits = context_hits + isolated_hits * 0.25
        raw_score = _clamp(1.0 + effective_hits * 0.8, 1.0, 5.0)

        # Apply InterviewBERT confidence weight
        confidence = _OCEAN_CONFIDENCE.get(trait, 1.0)
        # Scale toward neutral (3.0) based on confidence:
        # high confidence → keep score; low confidence → pull toward 3.0
        confidence_adj = round(raw_score * confidence + 3.0 * (1.0 - confidence), 2)
        final = round(_clamp(confidence_adj, 1.0, 5.0), 2)

        ocean_scores[trait] = final
        detail[trait] = {
            "context_hits":   context_hits,
            "isolated_hits":  isolated_hits,
            "effective_hits": round(effective_hits, 2),
            "raw_score":      round(raw_score, 2),
            "confidence":     confidence,
            "final_score":    final,
            "top_hits":       hit_details[:5],
            "keyword_bank":   "lc+hc_merged" if ocean_keywords is not None and ocean_keywords is not OCEAN_KEYWORDS else "lc_only",
        }

    return ocean_scores, detail


# ══════════════════════════════════════════════════════════════════════════════
#  [2] SESSION-LEVEL OCEAN AGGREGATION v3.0
# ══════════════════════════════════════════════════════════════════════════════

def compute_session_ocean(
    per_answer_oceans: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Multi-session OCEAN Aggregation v3.0

    Averages trait scores across multiple answers and modulates each trait
    score by its cross-answer consistency (standard deviation).

    Consistency modulation
    ----------------------
    σ < 0.5  → high consistency → +0.3 bonus (stable trait expression)
    0.5 ≤ σ < 1.5 → moderate → ±0 (no adjustment)
    σ ≥ 1.5 → low consistency → −0.3 penalty (likely noise or context-switching)

    Research
    --------
    Dai et al. (Frontiers Psychol. 2022) — InterviewBERT multi-answer trait
    inference: consistency of trait expression across questions reduces
    variance in the language-inferred score and brings it closer to
    ground-truth questionnaire scores (r increases ~0.05 with 3+ answers).

    Parameters
    ----------
    per_answer_oceans : list of OCEAN score dicts from _compute_ocean_v3()
                        (one per question answered in the session)

    Returns
    -------
    dict keyed by trait name, each value:
      {
        "mean":        float,  # average raw score across answers
        "std":         float,  # standard deviation
        "consistency": str,    # "high" | "moderate" | "low"
        "adjustment":  float,  # +0.3 / 0 / -0.3
        "final":       float,  # mean + adjustment, clamped 1–5
      }
    """
    if not per_answer_oceans:
        return {}

    traits = list(OCEAN_KEYWORDS.keys())
    results: Dict[str, Dict[str, float]] = {}

    for trait in traits:
        scores = [ans.get(trait, 3.0) for ans in per_answer_oceans]
        n = len(scores)
        mean = sum(scores) / n

        if n >= 2:
            variance = sum((s - mean) ** 2 for s in scores) / n
            std = variance ** 0.5
        else:
            # Single answer — no consistency data, no adjustment
            std = 0.0

        if n < 2:
            # Cannot assess consistency without ≥2 data points
            consistency = "insufficient_data"
            adjustment  = 0.0
        elif std < 0.5:
            consistency = "high"
            adjustment  = +0.3
        elif std < 1.5:
            consistency = "moderate"
            adjustment  = 0.0
        else:
            consistency = "low"
            adjustment  = -0.3

        final = round(_clamp(mean + adjustment, 1.0, 5.0), 2)
        results[trait] = {
            "mean":        round(mean, 2),
            "std":         round(std, 2),
            "consistency": consistency,
            "adjustment":  adjustment,
            "final":       final,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  [6] CROSS-QUESTION THEMATIC COHERENCE SCORING
# ══════════════════════════════════════════════════════════════════════════════
#
# Detects narrative contradictions and thematic inconsistencies across answers
# within a session. Interviewers notice when a candidate's self-description
# shifts between questions (e.g. "I thrive in ambiguity" vs "I prefer clear
# requirements"), and this inconsistency depresses hiring probability even
# when individual answer quality is high.
#
# RESEARCH BASIS
# ──────────────
# Barrick et al. (2010, J. Applied Psychology) — structured interview validity:
#   Cross-answer consistency is a significant independent predictor of actual
#   job performance (β = 0.18 incremental over mean score alone). Candidates
#   who tell internally consistent stories are perceived as more credible and
#   honest by interviewers. The effect holds after controlling for cognitive
#   ability and Big-Five traits.
#
# Levashina et al. (2014, Personnel Psychology) — interview faking:
#   Inconsistent self-presentation across questions is the #1 behavioural cue
#   interviewers use to detect impression management / answer fabrication.
#   Candidates detected as inconsistent receive 23% lower offer rates even when
#   their individual answer scores are equivalent to consistent candidates.
#
# DeGroot & Motowidlo (1999, J. Applied Psychology) — non-verbal and narrative
#   credibility cues: narrative coherence (does the story hang together across
#   the interview?) is rated as one of the top-3 hiring signals by trained
#   interviewers, ahead of technical accuracy on individual questions.
#
# WHAT IS CHECKED
# ───────────────
# Three contradiction types are detected:
#
#  TYPE 1 — OCEAN trait polarity contradictions
#     A candidate whose Openness mean is HIGH (> 3.8) on one set of questions
#     but LOW (< 2.8) on another set is signalling a story that doesn't hold
#     together. Specifically: traits with low compute_session_ocean() consistency
#     ("low" or 0 std < 0.5 reversed) get flagged with the question indices
#     that pulled the score in each direction.
#
#  TYPE 2 — Work-style preference contradictions
#     Keyword-based detection of antithetic self-descriptions:
#       "ambiguity / flexibility / adapt" vs "structure / process / clear requirements"
#       "lead / drive / take ownership" vs "collaborative / team / together"
#       "detail / thorough / careful" vs "big picture / strategic / high level"
#       "fast / rapid / quick" vs "careful / methodical / thorough"
#     A contradiction is flagged when both poles are asserted in DIFFERENT
#     answers (same answer = valid nuance; different answers = inconsistency).
#
#  TYPE 3 — STAR result claim inconsistency
#     Checks whether claimed outcomes are directionally consistent across
#     answers. Contradictory outcome signals: one answer claims failure/learning
#     (explicit "failed", "didn't work") while another claims success
#     ("delivered", "shipped", "achieved") on a similar topic cluster.
#     Detects embellishment patterns.
#
# OUTPUT
# ──────
# compute_coherence_report(per_answer_records) → CoherenceReport dataclass
# with a coherence_score (0–1, higher = more coherent), a list of
# CoherenceFlag objects describing each detected inconsistency, and
# a natural-language coaching summary.
#
# INTEGRATION
# ───────────
# Called from InterviewAnalyzer.analyze() after ≥2 answers are accumulated,
# mirroring how compute_session_ocean() is called. The result is included
# in the /report response and also returned from /evaluate (populated once
# ≥2 answers exist, null before that).
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dc, field as _field
from typing import List as _List

@_dc
class CoherenceFlag:
    """
    A single detected cross-question inconsistency.

    Fields
    ------
    flag_type     : "ocean_polarity" | "work_style" | "outcome_direction"
    severity      : "high" | "moderate" | "low"
    description   : human-readable explanation of the inconsistency
    question_indices : list of 0-based answer indices involved
    pole_a        : what was claimed in the first answer (brief phrase)
    pole_b        : what was claimed in the conflicting answer (brief phrase)
    coaching_tip  : concrete action the candidate can take to resolve the
                    inconsistency before a real interview
    """
    flag_type:        str = ""
    severity:         str = "low"
    description:      str = ""
    question_indices: list = _field(default_factory=list)
    pole_a:           str = ""
    pole_b:           str = ""
    coaching_tip:     str = ""

    def to_dict(self) -> dict:
        return {
            "flag_type":        self.flag_type,
            "severity":         self.severity,
            "description":      self.description,
            "question_indices": self.question_indices,
            "pole_a":           self.pole_a,
            "pole_b":           self.pole_b,
            "coaching_tip":     self.coaching_tip,
        }


@_dc
class CoherenceReport:
    """
    Cross-question thematic coherence analysis for a full interview session.

    Fields
    ------
    coherence_score  : float 0–1. 1.0 = fully coherent narrative; 0.0 = maximum
                       detected contradiction. Deducts per flag: high −0.20,
                       moderate −0.12, low −0.06 (capped at floor 0.10 to avoid
                       catastrophising noisy single-answer detection).
    n_flags          : total number of flags detected
    flags            : list of CoherenceFlag objects (sorted by severity desc)
    narrative_label  : "Highly Coherent" | "Mostly Coherent" | "Some Inconsistencies"
                       | "Notable Contradictions"
    coaching_summary : 1–3 sentence natural-language summary for the report UI
    n_answers        : number of answers analysed
    available        : False if fewer than 2 answers (coherence requires ≥2 data points)
    """
    coherence_score:   float = 1.0
    n_flags:           int   = 0
    flags:             list  = _field(default_factory=list)
    narrative_label:   str   = "Highly Coherent"
    coaching_summary:  str   = ""
    n_answers:         int   = 0
    available:         bool  = False

    def to_dict(self) -> dict:
        return {
            "coherence_score":   round(self.coherence_score, 3),
            "n_flags":           self.n_flags,
            "flags":             [f.to_dict() for f in self.flags],
            "narrative_label":   self.narrative_label,
            "coaching_summary":  self.coaching_summary,
            "n_answers":         self.n_answers,
            "available":         self.available,
        }


# ── Work-style antithesis pairs (TYPE 2) ─────────────────────────────────────
# Each entry: (label_a, keywords_a, label_b, keywords_b, severity, coaching_tip)
# Detection logic: flag when BOTH poles appear in DIFFERENT answers.
# Same-answer usage is intentional nuance (e.g. "I adapt flexibly but
# ensure process clarity") — not a contradiction.
_WORKSTYLE_PAIRS: _List[tuple] = [
    (
        "works well with ambiguity",
        {"ambiguity", "ambiguous", "uncertainty", "uncertain", "flexibility",
         "flexible", "adapt", "adaptable", "unstructured", "fluid", "open-ended"},
        "prefers clear structure",
        {"structure", "structured", "process", "procedure", "clear requirements",
         "defined", "systematic", "organised", "organised", "methodical", "framework"},
        "moderate",
        "Both comfort with ambiguity and preference for structure are valid — "
        "clarify that you can operate in both modes and give examples of when "
        "each applies. Interviewers notice when these appear to contradict.",
    ),
    (
        "individual ownership",
        {"i led", "i drove", "i decided", "i owned", "my initiative",
         "take ownership", "took ownership", "individually", "on my own"},
        "team-first attribution",
        {"we achieved", "team effort", "collaborative", "together we",
         "the team delivered", "collectively", "group decision"},
        "low",
        "Both individual ownership and team attribution are genuine strengths — "
        "be intentional about which you emphasise and when. If you claim 'I led' "
        "in one answer and 'we achieved' in another for similar situations, "
        "interviewers may question your consistency.",
    ),
    (
        "detail-oriented",
        {"detail", "detailed", "thorough", "thoroughness", "careful", "precision",
         "precise", "meticulous", "quality", "accuracy", "accurate"},
        "big-picture / strategic",
        {"big picture", "strategic", "strategy", "high level", "macro", "vision",
         "visionary", "long-term thinking", "broad"},
        "low",
        "Detail-orientation and strategic thinking are complementary, not opposites — "
        "but claiming both strongly in different answers without explaining how you "
        "switch between them reads as inconsistent. Add a bridging statement.",
    ),
    (
        "fast-paced / bias for action",
        {"fast", "quickly", "rapid", "speed", "urgency", "bias for action",
         "ship fast", "move fast", "iterate quickly", "done is better"},
        "deliberate / careful",
        {"careful", "deliberate", "slow down", "take time", "methodical",
         "consider all options", "not rush", "patient", "measured"},
        "moderate",
        "Speed and care are both valuable, but claiming strong versions of both "
        "in separate answers without qualifying context can confuse interviewers. "
        "Anchor each claim to a specific situation type.",
    ),
]


# ── STAR outcome polarity signals (TYPE 3) ───────────────────────────────────
_OUTCOME_SUCCESS_SIGNALS = {
    "delivered", "shipped", "launched", "achieved", "succeeded", "won",
    "improved", "increased", "reduced", "saved", "led to", "resulted in",
    "exceeded", "beat", "surpassed", "promoted", "recognised", "award",
}
_OUTCOME_FAILURE_SIGNALS = {
    "failed", "failure", "didn't work", "did not work", "went wrong",
    "missed", "mistake", "error", "regret", "could have done better",
    "learned from", "lesson", "hindsight",
}


def compute_coherence_report(
    per_answer_records: _List[dict],
) -> CoherenceReport:
    """
    Compute cross-question thematic coherence across all answers in a session.

    Parameters
    ----------
    per_answer_records : list of answer record dicts as stored in session["answers"].
        Each record must contain at minimum:
            "answer"         : str   — the raw answer text
            "question"       : str   — the question that was asked
            "question_type"  : str   — "technical" | "behavioural" | "hr"
        Optionally (enriched by /evaluate):
            "ocean_scores"   : dict  — OCEAN sub-scores from _compute_ocean_v3()
                                       (keys: Openness, Conscientiousness, etc.)

        The function degrades gracefully if optional keys are absent —
        TYPE 1 (OCEAN polarity) is skipped when ocean_scores is missing,
        TYPE 2 and TYPE 3 run on raw answer text alone.

    Returns
    -------
    CoherenceReport — .available=False if fewer than 2 records provided.
                      .to_dict() for JSON serialisation in API responses.

    Notes
    -----
    Complexity: O(n²) in the number of answers for TYPE 2/3 (pair comparison),
    but n is typically 3–10 so this is negligible.
    Thread-safe: no module-level state is written.
    """
    n = len(per_answer_records)
    if n < 2:
        return CoherenceReport(available=False, n_answers=n)

    flags: _List[CoherenceFlag] = []

    # ── TYPE 1: OCEAN trait polarity contradictions ───────────────────────────
    # Extract per-answer OCEAN scores if available. Compute within-session
    # trait swing: flag traits where the max–min range across answers exceeds
    # a threshold, with the two most extreme answers as the offending pair.
    #
    # Threshold calibration:
    #   Traits are scored 1–5. A 2-point swing (e.g. 2.0 → 4.0) on the same
    #   trait across different answers signals a genuine story shift.
    #   0.5 consistency std corresponds to ≈1.5 score range for 3–5 answers.
    #   We flag when range ≥ 1.8 (conservative — avoids noise on 2-answer sessions).
    _OCEAN_SWING_THRESHOLD = 1.8
    _OCEAN_TRAIT_LABELS = {
        "Openness":           ("risk-averse / conventional",   "open / creative / experimental"),
        "Conscientiousness":  ("spontaneous / flexible",        "organised / diligent / process-driven"),
        "Extraversion":       ("reserved / independent",        "outgoing / collaborative / vocal"),
        "Agreeableness":      ("direct / assertive / tough",    "accommodating / empathetic / harmonious"),
        "Neuroticism_inv":    ("anxious / self-doubting",       "calm / resilient / stable"),
    }

    ocean_by_answer = []
    for rec in per_answer_records:
        oc = rec.get("ocean_scores") or rec.get("nlp_ocean") or {}
        ocean_by_answer.append(oc)

    has_ocean = any(bool(oc) for oc in ocean_by_answer)
    if has_ocean:
        for trait, (low_label, high_label) in _OCEAN_TRAIT_LABELS.items():
            scores_with_idx = [
                (i, oc.get(trait, 3.0))
                for i, oc in enumerate(ocean_by_answer)
                if oc.get(trait) is not None
            ]
            if len(scores_with_idx) < 2:
                continue
            scores_only = [s for _, s in scores_with_idx]
            swing = max(scores_only) - min(scores_only)
            if swing < _OCEAN_SWING_THRESHOLD:
                continue

            min_idx = min(scores_with_idx, key=lambda x: x[1])[0]
            max_idx = max(scores_with_idx, key=lambda x: x[1])[0]
            severity = "high" if swing >= 2.5 else "moderate"
            flags.append(CoherenceFlag(
                flag_type        = "ocean_polarity",
                severity         = severity,
                description      = (
                    f"Your {trait} trait score shifts by {swing:.1f} points across "
                    f"questions {min_idx + 1} and {max_idx + 1}. "
                    f"Answer {min_idx + 1} reads as {low_label}; "
                    f"answer {max_idx + 1} reads as {high_label}. "
                    f"Interviewers who review both may perceive inconsistency."
                ),
                question_indices = sorted([min_idx, max_idx]),
                pole_a           = f"Q{min_idx + 1}: {low_label}",
                pole_b           = f"Q{max_idx + 1}: {high_label}",
                coaching_tip     = (
                    f"Both expressions of {trait} can be authentic, but they need "
                    f"to be anchored to different contexts or life stages. Add a "
                    f"brief qualifier ('In ambiguous situations I lean toward X; "
                    f"when the stakes are high I shift to Y') so the two answers "
                    f"read as nuance rather than contradiction."
                ),
            ))

    # ── TYPE 2: Work-style preference contradictions ──────────────────────────
    # For each antithesis pair, check whether BOTH poles appear in at least
    # one answer each. Same-answer co-occurrence is allowed (intentional nuance).
    answers_lower = [rec.get("answer", "").lower() for rec in per_answer_records]

    for (label_a, kw_a, label_b, kw_b, severity, tip) in _WORKSTYLE_PAIRS:
        # Find which answers assert each pole
        answers_asserting_a = []
        answers_asserting_b = []
        for i, al in enumerate(answers_lower):
            words_i = set(al.split())
            # Multi-word phrases need substring match; single words use set membership
            hits_a = any(
                (kw in words_i if " " not in kw else kw in al)
                for kw in kw_a
            )
            hits_b = any(
                (kw in words_i if " " not in kw else kw in al)
                for kw in kw_b
            )
            if hits_a:
                answers_asserting_a.append(i)
            if hits_b:
                answers_asserting_b.append(i)

        # Contradiction = DIFFERENT answers assert opposite poles
        # (not same answer, which is legitimate contextualisation)
        contradicting_pairs = [
            (ia, ib)
            for ia in answers_asserting_a
            for ib in answers_asserting_b
            if ia != ib
        ]
        if not contradicting_pairs:
            continue

        # Pick the most prominent pair (earliest a, earliest b)
        ia, ib = contradicting_pairs[0]
        flags.append(CoherenceFlag(
            flag_type        = "work_style",
            severity         = severity,
            description      = (
                f"Answer {ia + 1} signals '{label_a}' while answer {ib + 1} signals "
                f"'{label_b}'. These can coexist but read as contradictory when "
                f"expressed strongly in separate answers without contextualisation."
            ),
            question_indices = sorted(set(
                [ia, ib] + [p[0] for p in contradicting_pairs[:3]]
                         + [p[1] for p in contradicting_pairs[:3]]
            )),
            pole_a           = f"Q{ia + 1}: {label_a}",
            pole_b           = f"Q{ib + 1}: {label_b}",
            coaching_tip     = tip,
        ))

    # ── TYPE 3: STAR outcome direction inconsistency ──────────────────────────
    # Flag sessions where one or more answers claim explicit failure ("I failed",
    # "it didn't work") while others claim strong success, on similar topic areas.
    # A single failure answer is healthy and expected; the issue is when the
    # claimed failure and claimed success involve similar role/domain language,
    # suggesting embellishment in one direction.
    #
    # Detection: identify answers with strong success signals and answers with
    # strong failure signals. If any two answers from different STAR result
    # sections have overlapping domain keywords (5+ shared nouns), flag the pair.

    # Extract simple domain keywords (nouns, tech terms, role verbs) from each answer
    _STOP = {"a","an","the","and","or","but","in","on","at","to","for","of","with",
              "by","from","is","was","are","were","be","been","i","we","you","he",
              "she","they","it","this","that","my","our","your","so","if","not",
              "as","up","do","did","will","would","could","should","also","just",
              "then","when","which","who","what","how","very","really","then"}

    def _domain_words(text: str) -> set:
        return {w.strip(".,;:!?()\"'") for w in text.lower().split()
                if len(w) > 3 and w not in _STOP}

    success_answers = []   # (idx, domain_words)
    failure_answers = []   # (idx, domain_words)
    for i, al in enumerate(answers_lower):
        words_set = set(al.split())
        has_success = bool(_OUTCOME_SUCCESS_SIGNALS & {w.strip(".,;:!?") for w in words_set})
        has_failure = bool(_OUTCOME_FAILURE_SIGNALS & {w.strip(".,;:!?") for w in words_set})
        dom = _domain_words(al)
        if has_success and not has_failure:
            success_answers.append((i, dom))
        if has_failure and not has_success:
            failure_answers.append((i, dom))

    if success_answers and failure_answers:
        # Check for domain overlap that suggests same topic area
        for (si, sdom) in success_answers:
            for (fi, fdom) in failure_answers:
                overlap = sdom & fdom
                if len(overlap) >= 5:   # 5 shared domain words = strong topic similarity
                    shared_preview = ", ".join(sorted(overlap)[:5])
                    flags.append(CoherenceFlag(
                        flag_type        = "outcome_direction",
                        severity         = "moderate",
                        description      = (
                            f"Answer {si + 1} claims a strong positive outcome while "
                            f"answer {fi + 1} describes a failure or mistake — and both "
                            f"answers share similar domain vocabulary ({shared_preview}…). "
                            f"Interviewers may probe the discrepancy."
                        ),
                        question_indices = sorted([si, fi]),
                        pole_a           = f"Q{si + 1}: success / positive outcome",
                        pole_b           = f"Q{fi + 1}: failure / learning moment",
                        coaching_tip     = (
                            "Having both a success and a failure story is expected and "
                            "healthy — but ensure they are clearly about DIFFERENT situations "
                            "or phases. If they overlap in topic, explicitly distinguish them: "
                            "'This was earlier in my career; by the time of the success story "
                            "I had learned from that mistake.'"
                        ),
                    ))
                    break   # one flag per success answer is enough

    # ── Score computation ─────────────────────────────────────────────────────
    deductions = {"high": 0.20, "moderate": 0.12, "low": 0.06}
    raw_score = 1.0
    for f in flags:
        raw_score -= deductions.get(f.severity, 0.06)
    coherence_score = round(max(0.10, min(1.0, raw_score)), 3)

    # Sort flags: high → moderate → low
    _sev_order = {"high": 0, "moderate": 1, "low": 2}
    flags.sort(key=lambda f: _sev_order.get(f.severity, 3))

    # Label
    if coherence_score >= 0.90:
        label = "Highly Coherent"
    elif coherence_score >= 0.75:
        label = "Mostly Coherent"
    elif coherence_score >= 0.55:
        label = "Some Inconsistencies"
    else:
        label = "Notable Contradictions"

    # Coaching summary
    if not flags:
        summary = (
            "Your answers tell a consistent and coherent story across all questions. "
            "Interviewers will find your self-description credible and easy to follow."
        )
    elif len(flags) == 1:
        f = flags[0]
        summary = (
            f"One narrative inconsistency was detected: {f.description} "
            f"Action: {f.coaching_tip}"
        )
    else:
        top = flags[0]
        summary = (
            f"{len(flags)} cross-question inconsistencies detected. "
            f"Most significant: {top.description} "
            f"Overall: review your answers to ensure each one reinforces the same "
            f"core professional narrative."
        )

    return CoherenceReport(
        coherence_score  = coherence_score,
        n_flags          = len(flags),
        flags            = flags,
        narrative_label  = label,
        coaching_summary = summary,
        n_answers        = n,
        available        = True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ALL v2.0 HELPERS (unchanged — kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_word_cat_score(words: List[str], wc: int) -> Tuple[float, int, int]:
    quant_count = sum(1 for w in words if w in QUANTIFIERS)
    percep_count = sum(1 for w in words if w in PERCEPTUAL_WORDS)
    quant_density = _clamp(quant_count / max(1, wc) * 100 / 5.0, 0, 1)
    percep_density = _clamp(percep_count / max(1, wc) * 100 / 4.0, 0, 1)
    sc = round((quant_density * 0.55 + percep_density * 0.45) * 5.0, 3)
    return sc, quant_count, percep_count


def _compute_wpm_score(wc: int, duration_s: float) -> Optional[float]:
    if duration_s <= 5:
        return None
    wpm = (wc / duration_s) * 60.0
    if 120 <= wpm <= 160:     score = 5.0
    elif 100 <= wpm < 120:    score = 3.0 + (wpm - 100) / 20.0 * 2.0
    elif wpm < 100:           score = max(1.0, 3.0 - (100 - wpm) / 20.0)
    elif 160 < wpm <= 200:    score = 5.0 - (wpm - 160) / 40.0 * 2.0
    else:                     score = max(1.0, 3.0 - (wpm - 200) / 30.0)
    return round(float(score), 2)


def _compute_fluency_score(filler_ratio: float) -> float:
    return round(max(0.5, min(5.0, 5.0 - filler_ratio * 25.0)), 2)


def _compute_depth_fluency(depth: float, fluency: float,
                            wpm: Optional[float]) -> float:
    if wpm is not None:
        sc = depth * 0.40 + wpm * 0.35 + fluency * 0.25
    else:
        sc = depth * 0.65 + fluency * 0.35
    return round(_clamp(sc, 0.5, 5.0), 2)


def _compute_time_score(elapsed_s: float, q_type: str, difficulty: str) -> Dict:
    qt = _resolve_type(q_type)
    diff = difficulty.lower().strip()
    if diff not in ("easy", "medium", "hard"):
        diff = "medium"
    window = _TIME_WINDOWS.get(qt, _TIME_WINDOWS["technical"]).get(
        diff, _TIME_WINDOWS["technical"]["medium"])
    ideal_str = _TIME_LABELS.get(qt, _TIME_LABELS["technical"]).get(diff, "—")
    abs_min, ideal_min, ideal_max, abs_max = window

    if elapsed_s <= 0:
        return {"time_score": 0.0, "time_label": "No timing", "time_modifier": 0.0,
                "time_ideal_window": ideal_str, "time_elapsed_s": elapsed_s}

    if elapsed_s < abs_min:
        score = 1.0
    elif elapsed_s < ideal_min:
        t = (elapsed_s - abs_min) / max(1, ideal_min - abs_min)
        score = 1.0 + t * 4.0
    elif elapsed_s <= ideal_max:
        score = 5.0
    elif elapsed_s <= abs_max:
        t = (elapsed_s - ideal_max) / max(1, abs_max - ideal_max)
        score = 5.0 - t * 4.0
    else:
        score = 1.0
    score = round(_clamp(score, 1.0, 5.0), 2)

    if   score >= 4.5: modifier = +0.10
    elif score >= 3.5: modifier = +0.05
    elif score >= 2.5: modifier =  0.00
    elif score >= 1.5: modifier = -0.10
    else:              modifier = -0.20

    label = ("Ideal pace" if score >= 4.5 else
             "Slightly brief" if score >= 3.5 and elapsed_s < ideal_min else
             "Slightly long"  if score >= 3.5 else
             "Too brief" if elapsed_s < ideal_min else "Too long")

    return {"time_score": score, "time_label": label,
            "time_modifier": modifier, "time_ideal_window": ideal_str,
            "time_elapsed_s": elapsed_s}


def _compute_personality_nlp(ocean: Dict[str, float]) -> float:
    c = ocean.get("Conscientiousness", 1.0)
    e = ocean.get("Extraversion", 1.0)
    o = ocean.get("Openness", 1.0)
    a = ocean.get("Agreeableness", 1.0)
    return round(_clamp(c * 0.40 + e * 0.35 + o * 0.15 + a * 0.10, 1.0, 5.0), 2)


def _compute_sentiment(text_lower: str) -> float:
    pos = sum(1 for w in POSITIVE_SENTIMENT if w in text_lower)
    neg = sum(1 for w in NEGATIVE_SENTIMENT if w in text_lower)
    return round((pos - neg) / max(1, pos + neg + 1) * 3.0, 2)


def _compute_hiring_signal(final_score: float, sentiment: float,
                            fluency: float) -> float:
    sentiment_adj = (sentiment + 3.0) / 6.0
    fluency_adj = fluency / 5.0
    return round(_clamp(
        final_score * 0.70 + sentiment_adj * 5.0 * 0.15 + fluency_adj * 5.0 * 0.15,
        1.0, 5.0), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL TYPE-AWARE EVALUATOR v3.0
# ══════════════════════════════════════════════════════════════════════════════

def _full_evaluate_v3(
    answer: str,
    question_type: str,
    keywords: List[str],
    relevance_score: float,
    duration_s: float,
    difficulty: str,
    cultural_context: str = "auto",
) -> Dict:
    """
    Full type-aware evaluator — v3.0.

    Changes from v2.0
    -----------------
    • _compute_depth_score()   → _compute_depth_score_v3()
    • _compute_star_score()    → _compute_star_score_v3()
    • _compute_ocean()         → _compute_ocean_v3()
    • grammar_score binary     → _compute_grammar_score_v3()

    All other sub-scorers (wpm, fluency, word_cat, time, DISC, sentiment,
    hiring signal) are identical to v2.0.

    The weight profiles (WEIGHT_PROFILES) are unchanged; grammar now feeds
    a continuous 0–5 value instead of binary 85/65.
    """
    q_type_key = _resolve_type(question_type)
    W = dict(WEIGHT_PROFILES[q_type_key])

    al = answer.lower()
    words = al.split()
    wc = len(words)

    # ── Cultural context detection (runs early — results used by both STAR
    #    weight adaptation and OCEAN keyword selection below) ─────────────────
    if cultural_context == "auto":
        _detected_ctx, _cultural_detect_detail = detect_cultural_context(answer)
    elif cultural_context == "high-context":
        _detected_ctx = "high-context"
        _, _cultural_detect_detail = detect_cultural_context(answer)
        _cultural_detect_detail["context"] = "high-context (forced)"
    else:
        _detected_ctx = "low-context"
        _, _cultural_detect_detail = detect_cultural_context(answer)
        _cultural_detect_detail["context"] = "low-context (forced)"

    # ── Culture-adapted OCEAN keywords ────────────────────────────────────────
    # get_ocean_keywords_for_context() returns OCEAN_KEYWORDS unchanged for
    # low-context candidates (zero impact), and returns LC+HC merged banks for
    # high-context candidates. The merged bank is used ONLY inside
    # _compute_ocean_v3() — all other scoring is unaffected until the explicit
    # weight adaptation step below.
    _ocean_kw_bank = get_ocean_keywords_for_context(_detected_ctx, OCEAN_KEYWORDS)

    # ── 1. STAR v3.0 ─────────────────────────────────────────────────────────
    star_sc, star_sc_map, order_bonus, star_detail = _compute_star_score_v3(al)

    # ── Relevance gate on STAR score ─────────────────────────────────────────
    # Bug fix: STAR patterns are question-blind — a candidate who recites a
    # well-structured but completely off-topic anecdote scores high on STAR
    # even though the answer doesn't address what was asked.
    #
    # For behavioural/hr questions (where STAR weight is 0.35 / 0.20 — highest),
    # discount star_sc proportionally when relevance_score < 0.35:
    #
    #   gate_factor = relevance_score / 0.35   (0.0 → 1.0)
    #   star_sc     = star_sc × gate_factor
    #
    # Threshold 0.35: chosen to match the Groq rubric "0.25 — touches topic but
    # misses most key points". Below this, STAR structure is noise.
    # At relevance ≥ 0.35 the gate is 1.0 — no change.
    # At relevance = 0.0 (completely off-topic), star_sc collapses to 0.0.
    #
    # Technical questions (STAR weight = 0.00) are unaffected by design.
    _STAR_RELEVANCE_THRESHOLD = 0.35
    if q_type_key in ("behavioural", "hr") and relevance_score < _STAR_RELEVANCE_THRESHOLD:
        gate_factor = round(relevance_score / _STAR_RELEVANCE_THRESHOLD, 4)
        star_sc = round(star_sc * gate_factor, 3)
        star_detail["relevance_gate_applied"] = True
        star_detail["relevance_gate_factor"]  = gate_factor
    else:
        star_detail["relevance_gate_applied"] = False
        star_detail["relevance_gate_factor"]  = 1.0

    # ── Cultural weight + STAR adaptation ────────────────────────────────────
    # adapt_weights_for_culture() was accepted by this function but not applied
    # until now. It adjusts W (STAR ↓, depth_flu ↑ for HC candidates) and may
    # add a collective-attribution bonus to star_sc.
    W, star_sc, _cultural_detail = adapt_weights_for_culture(
        text             = answer,
        q_type_key       = q_type_key,
        star_sc          = star_sc,
        base_weights     = W,
        cultural_context = _detected_ctx,   # use already-resolved context
    )

    # ── 2. Word category ─────────────────────────────────────────────────────
    word_cat_sc, quant_count, percep_count = _compute_word_cat_score(words, wc)

    # ── 3. Relevance (Groq-provided, 0–1) ────────────────────────────────────
    tfidf_sim = relevance_score

    # ── 4. Keyword score ─────────────────────────────────────────────────────
    kw_list = [k.lower() for k in keywords]
    if not kw_list:
        kw_sc = 0.0
        freed = W["keyword"]
        W["keyword"] = 0.0
        W["relevance"] += freed
        kw_hits = []
    else:
        hits = [k for k in kw_list if k in al]
        kw_sc = round(_clamp(len(hits) / max(1, len(kw_list)) * 5.0, 0, 5), 3)
        kw_hits = hits

    # ── 5. Depth v3.0 + fluency ──────────────────────────────────────────────
    filler_count = sum(al.split().count(f) for f in FILLER_WORDS)
    filler_ratio = filler_count / max(1, wc)

    depth_sc, depth_detail = _compute_depth_score_v3(wc, q_type_key, answer)
    fluency_sc = _compute_fluency_score(filler_ratio)
    wpm_score  = _compute_wpm_score(wc, duration_s)
    depth_flu_sc = _compute_depth_fluency(depth_sc, fluency_sc, wpm_score)

    # ── 6. Grammar v3.0 (continuous 0–5) ─────────────────────────────────────
    grammar_score_5, grammar_detail = _compute_grammar_score_v3(answer)
    # Normalise to 0–1 for the weight formula (× 5 inline below keeps formula readable)

    # ── 7. Time score ─────────────────────────────────────────────────────────
    time_data     = _compute_time_score(duration_s, q_type_key, difficulty)
    time_modifier = time_data["time_modifier"]

    # ── Composite ─────────────────────────────────────────────────────────────
    raw_score = (
        star_sc               * W["star"]      +
        word_cat_sc           * W["word_cat"]  +
        tfidf_sim * 5.0       * W["relevance"] +
        kw_sc                 * W["keyword"]   +
        depth_flu_sc          * W["depth_flu"] +
        grammar_score_5       * W["grammar"]   # v3: continuous 0–5, not /100×5
    )
    final_score = round(_clamp(raw_score + time_modifier, 1.0, 5.0), 2)

    # ── DISC ──────────────────────────────────────────────────────────────────
    disc_traits  = {tr: sum(1 for w in ws if w in al) for tr, ws in DISC_KEYWORDS.items()}
    disc_dominant = max(disc_traits, key=disc_traits.get) if any(disc_traits.values()) else "None"

    # ── OCEAN v3.0 (context-gated, culture-adapted keywords) ─────────────────
    # _ocean_kw_bank is identical to OCEAN_KEYWORDS for LC candidates,
    # and the merged LC+HC bank for HC candidates (set at top of function).
    ocean, ocean_detail = _compute_ocean_v3(al, kw_hits, ocean_keywords=_ocean_kw_bank)
    personality_sc = _compute_personality_nlp(ocean)

    conscientiousness = round(
        ocean.get("Conscientiousness", 3.0) * 0.6 +
        _clamp(disc_traits.get("Conscientiousness", 0) * 0.8 + 1.0, 1, 5) * 0.4, 2)

    # ── Sentiment + hiring ────────────────────────────────────────────────────
    sentiment      = _compute_sentiment(al)
    vocab_diversity = round(len(set(words)) / max(1, wc), 3)
    hiring_signal  = _compute_hiring_signal(final_score, sentiment, fluency_sc)

    return {
        # ── Primary ───────────────────────────────────────────────────────────
        "final_score":       final_score,
        # ── STAR ──────────────────────────────────────────────────────────────
        "star_score":        star_sc,
        "star_scores":       star_sc_map,
        "order_bonus":       order_bonus,
        "star_detail_v3":    star_detail,           # NEW: per-component quarter detail
        # ── Word category ─────────────────────────────────────────────────────
        "word_cat_score":    word_cat_sc,
        "quant_count":       quant_count,
        "percep_count":      percep_count,
        # ── Relevance ─────────────────────────────────────────────────────────
        "relevance_score":   round(tfidf_sim, 3),
        # ── Keyword ───────────────────────────────────────────────────────────
        "keyword_hits":      kw_hits,
        "keyword_score":     kw_sc,
        # ── Depth v3.0 ────────────────────────────────────────────────────────
        "depth_score":       depth_sc,
        "depth_detail_v3":   depth_detail,          # NEW: wc/lex/clause sub-scores
        # ── Fluency / WPM ─────────────────────────────────────────────────────
        "fluency_score":     fluency_sc,
        "wpm_score":         wpm_score,
        "depth_fluency_sc":  depth_flu_sc,
        "filler_count":      filler_count,
        "filler_ratio":      round(filler_ratio, 4),
        # ── Grammar v3.0 ──────────────────────────────────────────────────────
        "grammar_score":     grammar_score_5,        # now 0–5 continuous (was 65/85)
        "grammar_detail_v3": grammar_detail,         # NEW: sl_cv / discourse / passive
        # ── Time ──────────────────────────────────────────────────────────────
        "time_data":         time_data,
        # ── DISC ──────────────────────────────────────────────────────────────
        "disc_traits":       disc_traits,
        "disc_dominant":     disc_dominant,
        # ── OCEAN v3.0 ────────────────────────────────────────────────────────
        "ocean_scores":      ocean,
        "ocean_detail_v3":   ocean_detail,           # context/isolated hits per trait
        "cultural_detail":   _cultural_detail,        # full audit trail: detection + weight + OCEAN
        "conscientiousness": conscientiousness,
        "personality_nlp":   personality_sc,
        # ── Sentiment / hiring ────────────────────────────────────────────────
        "sentiment_intensity": sentiment,
        "vocab_diversity":   vocab_diversity,
        "hiring_signal":     hiring_signal,
        # ── Meta ──────────────────────────────────────────────────────────────
        "weight_profile":    W,
        "question_type":     q_type_key,
        "word_count":        wc,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FACIAL NERVOUSNESS + VOICE PROXY + FUSION (unchanged from v2.0)
#  — Included here verbatim so the file is self-contained.
#    In production, import from analyzer.py or a shared module.
# ══════════════════════════════════════════════════════════════════════════════

_AU_NERVOUSNESS_WEIGHTS = {
    "AU4": 0.30, "AU7": 0.20, "AU20": 0.20, "AU1": 0.15, "AU14": 0.15,
}


def compute_au_score(au_intensities: Dict[str, float]) -> float:
    score = 0.0
    for au, weight in _AU_NERVOUSNESS_WEIGHTS.items():
        intensity = au_intensities.get(au, 0.0)
        score += _clamp(intensity / 5.0, 0.0, 1.0) * weight
    return round(_clamp(score, 0.0, 1.0), 3)


def compute_blink_anxiety(blink_rate_per_min: float,
                           inter_blink_intervals: Optional[List[float]] = None) -> float:
    blink_excess = max(0.0, blink_rate_per_min - 12.0)
    blink_score = _clamp(blink_excess / 20.0, 0.0, 1.0)
    brv_score = 0.0
    if inter_blink_intervals and len(inter_blink_intervals) >= 2:
        import statistics as _st
        brv = float(_st.stdev(inter_blink_intervals))
        brv_score = _clamp((brv - 0.5) / 2.5, 0.0, 1.0)
    return round(blink_score * 0.65 + brv_score * 0.35, 3)


def compute_gaze_score(gaze_contact_ratio: float,
                        gaze_direction_std_deg: float = 0.0) -> float:
    gaze_aversion = _clamp(1.0 - gaze_contact_ratio, 0.0, 1.0)
    gaze_jitter   = _clamp(gaze_direction_std_deg / 15.0, 0.0, 1.0)
    return round(gaze_aversion * 0.60 + gaze_jitter * 0.40, 3)


def compute_pose_score(yaw_angles: List[float], pitch_angles: List[float]) -> float:
    if len(yaw_angles) < 2 or len(pitch_angles) < 2:
        return 0.0
    import statistics as _st
    yaw_std, pitch_std = _st.stdev(yaw_angles), _st.stdev(pitch_angles)
    head_motion = _clamp((yaw_std + pitch_std) / 20.0, 0.0, 1.0)
    mean_pitch  = sum(pitch_angles) / len(pitch_angles)
    lean        = _clamp(max(0.0, -mean_pitch) / 15.0, 0.0, 1.0)
    return round(head_motion * 0.55 + lean * 0.45, 3)


def compute_ear_score(ear_time_series: List[float],
                       ear_threshold: float = 0.20) -> float:
    if not ear_time_series:
        return 0.0
    total  = len(ear_time_series)
    closed = sum(1 for e in ear_time_series if e < ear_threshold)
    perclos = closed / total
    import statistics as _st
    ear_var = _st.variance(ear_time_series) if total >= 2 else 0.0
    return round(perclos * 0.50 + _clamp(ear_var / 0.005, 0.0, 1.0) * 0.50, 3)


def compute_facial_nervousness(
    au_intensities: Optional[Dict[str, float]] = None,
    blink_rate_per_min: float = 15.0,
    inter_blink_intervals: Optional[List[float]] = None,
    gaze_contact_ratio: float = 0.70,
    gaze_direction_std_deg: float = 0.0,
    yaw_angles: Optional[List[float]] = None,
    pitch_angles: Optional[List[float]] = None,
    ear_time_series: Optional[List[float]] = None,
) -> Dict:
    au_sc    = compute_au_score(au_intensities or {})
    blink_sc = compute_blink_anxiety(blink_rate_per_min, inter_blink_intervals)
    gaze_sc  = compute_gaze_score(gaze_contact_ratio, gaze_direction_std_deg)
    pose_sc  = compute_pose_score(yaw_angles or [], pitch_angles or [])
    ear_sc   = compute_ear_score(ear_time_series or [])
    facial   = _clamp(au_sc*0.30 + blink_sc*0.25 + gaze_sc*0.25 + pose_sc*0.10 + ear_sc*0.10, 0.0, 1.0)
    return {
        "facial_nervousness": round(facial, 3),
        "au_score": au_sc, "blink_anxiety": blink_sc,
        "gaze_score": gaze_sc, "pose_score": pose_sc, "ear_score": ear_sc,
        "blink_rate_per_min": round(blink_rate_per_min, 1),
        "gaze_contact_ratio": round(gaze_contact_ratio, 3),
    }


def fuse_nervousness(facial_nervousness: float, voice_nervousness: float) -> float:
    fused = (NERVOUSNESS_FUSION["facial"] * facial_nervousness +
             NERVOUSNESS_FUSION["voice"]  * voice_nervousness)
    return round(_clamp(fused, 0.0, 1.0), 3)


def compute_confidence_score(eye_score: float, fluency_score: float,
                              voice_score: float, facial_score: float) -> float:
    conf = (eye_score*CONFIDENCE_WEIGHTS["eye"] + fluency_score*CONFIDENCE_WEIGHTS["fluency"] +
            voice_score*CONFIDENCE_WEIGHTS["voice"] + facial_score*CONFIDENCE_WEIGHTS["facial"])
    return round(_clamp(conf, 1.0, 5.0), 2)


def aggregate_session_score(knowledge: float, emotion: float,
                             voice: float, avg_depth: float = 0.0) -> Dict:
    final = round(
        knowledge * SCORE_WEIGHTS["knowledge"] +
        emotion   * SCORE_WEIGHTS["emotion"]   +
        voice     * SCORE_WEIGHTS["voice"], 2)
    return {"knowledge": knowledge, "emotion": emotion, "voice": voice,
            "depth": avg_depth, "final": final, "weights": SCORE_WEIGHTS}


# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW ANALYZER v3.0 — MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class InterviewAnalyzer:
    """
    v3.0 — drops in over v2.0 with zero API surface changes.
    All five formula upgrades are applied automatically.
    Multi-session OCEAN aggregation exposed via analyze_session().
    """

    def __init__(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client  = AsyncGroq(api_key=self.groq_api_key) if self.groq_api_key else None
        self.whisper_ready = bool(self.groq_api_key)
        self._whisper_model = None
        # Accumulate per-answer OCEAN scores for session aggregation
        self._session_oceans: List[Dict[str, float]] = []
        # Accumulate full answer records for cross-question coherence analysis
        self._session_answers: List[Dict] = []

    def check_groq_connection(self) -> bool:
        return bool(self.groq_api_key)

    def reset_session(self) -> None:
        """Call between interview sessions to reset OCEAN and coherence accumulators."""
        self._session_oceans.clear()
        self._session_answers.clear()

    def get_coherence_report(self) -> CoherenceReport:
        """
        Returns the cross-question thematic coherence report for the current session.
        Available after ≥2 answers have been analyzed.
        Returns CoherenceReport(available=False) if insufficient answers.
        """
        return compute_coherence_report(self._session_answers)

    def get_session_ocean(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the multi-session OCEAN aggregation for the current session.
        Call after all answers have been analyzed.
        Returns empty dict if < 2 answers have been analyzed.
        """
        if len(self._session_oceans) < 2:
            return {}
        return compute_session_ocean(self._session_oceans)

    # ── Transcription (unchanged) ─────────────────────────────────────────────

    async def transcribe(self, audio_path: str) -> str:
        if self.groq_client:
            with open(audio_path, "rb") as f:
                transcription = await self.groq_client.audio.transcriptions.create(
                    file=(audio_path, f.read()),
                    model="whisper-large-v3",
                    response_format="text",
                    language="en",
                )
            return transcription.strip()
        try:
            import whisper
            if not self._whisper_model:
                self._whisper_model = whisper.load_model("base")
            result = self._whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except ImportError:
            raise RuntimeError("No transcription provider. Set GROQ_API_KEY or: pip install openai-whisper")

    # ── Groq: semantic relevance (unchanged) ──────────────────────────────────

    async def _groq_relevance(self, answer: str, ideal_answer: str,
                               question: str, q_type: str) -> Tuple[float, str]:
        if not self.groq_client:
            return self._tfidf_fallback(answer, ideal_answer), "tfidf_fallback"

        prompt = (
            "You are an expert interview answer evaluator.\n"
            "Score how well the candidate answer covers the key concepts "
            "in the reference. Return ONLY a JSON object with a single key "
            "'relevance' whose value is a float between 0.0 and 1.0.\n\n"
            "Scoring rubric:\n"
            "  1.0  — covers all key concepts\n"
            "  0.75 — covers most, minor gaps\n"
            "  0.50 — covers about half\n"
            "  0.25 — touches topic but misses most key points\n"
            "  0.00 — off-topic or no meaningful overlap\n\n"
            "IMPORTANT: Score on CONCEPTS and MEANING, not exact wording.\n"
            "FAIRNESS: Do NOT penalise non-native English patterns or speech "
            "disfluencies. Judge what was communicated.\n\n"
            f"Question type: {q_type}\n"
            f"Question: {question[:500]}\n\n"
            f"Reference answer:\n{ideal_answer[:600]}\n\n"
            f"Candidate answer:\n{answer[:800]}\n\n"
            'Return ONLY: {"relevance": 0.75}'
        )
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            groq_score = round(_clamp(float(json.loads(raw)["relevance"]), 0.0, 1.0), 3)

            sas_score, sas_method = sas_scorer.score(answer, ideal_answer)

            # [7] Dynamic SAS weight — trust embedding proportional to answer length.
            # Short answers (< 50 words) produce noisy embeddings; LLM judgment
            # dominates. Long answers (> 150 words) use the standard 0.70/0.30 split.
            wc = len(answer.split())
            sas_w, llm_w = _dynamic_sas_weight(wc)

            fused_relevance = SASScorer.fuse_with_llm(
                sas_score, groq_score,
                sas_weight=sas_w,
                llm_weight=llm_w,
                llm_available=True,
            )
            print(f"[analyzer v3] Relevance — Groq: {groq_score:.3f}, "
                  f"SAS ({sas_method}): {sas_score:.3f}, "
                  f"wc={wc} → sas_w={sas_w:.3f}/llm_w={llm_w:.3f}, "
                  f"Fused: {fused_relevance:.3f}")
            return fused_relevance, f"api_groq+sas_dynamic(wc={wc})"
        except Exception as e:
            print(f"[analyzer v3] Groq relevance failed: {e}")
            # LLM down — pure SAS regardless of length (llm_available=False path
            # in fuse_with_llm already returns pure sas_score, no weight applied)
            sas_score, _ = sas_scorer.score(answer, ideal_answer)
            return SASScorer.fuse_with_llm(sas_score, 0.0, llm_available=False), "sas_fallback"

    def _tfidf_fallback(self, answer: str, reference: str) -> float:
        score, _ = sas_scorer.score(answer, reference)
        return score

    # ── Groq: HR feedback — Chain-of-Thought two-call pipeline ───────────────
    #
    # RESEARCH BASIS
    # --------------
    # Wei et al. (2022, NeurIPS) — Chain-of-Thought Prompting Elicits Reasoning
    #   in Large Language Models: forcing the model to articulate intermediate
    #   reasoning steps before producing a final answer improves accuracy on
    #   complex evaluation tasks by ~18–24% vs single-shot scoring.
    #
    # Kojima et al. (2022, NeurIPS) — Large Language Models are Zero-Shot
    #   Reasoners: even without few-shot examples, prompting with explicit
    #   reasoning steps reduces score variance significantly.
    #
    # ARCHITECTURE
    # ------------
    # Call 1 — CoT reasoning pass (temperature=0.3, max_tokens=600):
    #   Ask the model to think through the answer across six evaluation
    #   dimensions before committing to any score. Output is free-form prose.
    #   Dimensions: technical accuracy, concept coverage, STAR structure,
    #   communication quality, specific examples, overall impression.
    #
    # Call 2 — Structured JSON extraction (temperature=0.1, max_tokens=1000):
    #   Passes Call 1 reasoning as a <reasoning> block in the system role,
    #   then asks for the JSON output grounded in that reasoning.
    #   Lower temperature (0.1) locks in the conclusion the model already
    #   reached, reducing "score drift" where the LLM re-evaluates differently
    #   when asked to produce a number cold.
    #
    # FALLBACK
    # --------
    # If Call 1 succeeds but Call 2 fails (e.g. JSON parse error), the
    # reasoning text is still injected as context into a retry.
    # If both fail, falls back to _fallback_hr() as before.

    async def _groq_hr_feedback(self, transcript: str, question: str,
                                  question_type: str) -> Dict:
        if not self.groq_client:
            return self._fallback_hr(transcript)

        # ── Shared context block ──────────────────────────────────────────────
        context = (
            f"QUESTION TYPE: {question_type}\n"
            f"INTERVIEW QUESTION: {question}\n\n"
            f"CANDIDATE'S ANSWER:\n{transcript[:1200]}"
        )

        # ── CALL 1: Chain-of-Thought reasoning pass ───────────────────────────
        cot_prompt = f"""You are a senior technical interviewer and HR coach.

{context}

Before scoring this answer, reason through it step by step across these six dimensions.
Think carefully — your reasoning will be used to generate consistent, fair scores.

Step 1 — TECHNICAL ACCURACY
  What specific technical concepts did the candidate mention?
  Are they correct, partially correct, or wrong?
  What key concepts are missing for a complete answer?

Step 2 — CONCEPT COVERAGE
  What percentage of the expected answer domain did the candidate cover?
  List what was covered and what was omitted.

Step 3 — STAR STRUCTURE
  Did the candidate describe a Situation, Task, Action, and Result?
  Which components are present and which are absent?

Step 4 — COMMUNICATION QUALITY
  Was the answer clear and well-structured?
  Were there filler words, hedging language, or confidence markers?

Step 5 — SPECIFICITY & EXAMPLES
  Did the candidate use concrete numbers, tools, outcomes, or named examples?
  Or was the answer vague and generic?

Step 6 — OVERALL IMPRESSION
  Given steps 1–5, what grade (A/B/C/D/F) does this answer deserve and why?
  What is the single most impactful coaching tip?
  Should HR proceed: Strong Yes / Yes / Maybe / No?

Write your reasoning clearly. Do NOT produce JSON yet."""

        reasoning = ""
        try:
            cot_resp = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=0.3,
                max_tokens=700,
            )
            reasoning = cot_resp.choices[0].message.content.strip()
            print(f"[analyzer v3 CoT] Reasoning pass complete ({len(reasoning)} chars)")
        except Exception as e:
            print(f"[analyzer v3 CoT] Reasoning pass failed: {e} — falling back to single-shot")
            return await self._groq_hr_feedback_single_shot(transcript, question, question_type)

        # ── CALL 2: JSON extraction grounded in CoT reasoning ─────────────────
        json_prompt = f"""You are a senior technical interviewer and HR coach.

{context}

You have already reasoned through this answer step by step:

<reasoning>
{reasoning}
</reasoning>

Now, using ONLY the conclusions from your reasoning above, produce the final evaluation.
Return ONLY valid JSON — no prose, no markdown fences, no extra keys:

{{
  "technical_score": <0-100 integer, grounded in Step 1 & 2 above>,
  "relevance_score": <0-100 integer, grounded in Step 2 & 3 above>,
  "ideal_answer": "<60-100 word model answer covering the key concepts the candidate missed>",
  "technical_evaluation": "<2-3 sentences grounded in your Step 1 reasoning>",
  "key_strengths": ["<specific strength from reasoning>", "<specific strength>", "<specific strength>"],
  "improvement_areas": ["<specific gap from reasoning>", "<specific gap>", "<specific gap>"],
  "hr_recommendation": "<Strong Yes | Yes | Maybe | No — must match Step 6 conclusion>",
  "hr_reasoning": "<2-3 sentences, must be consistent with your Step 6 reasoning>",
  "coaching_tip": "<the single most impactful tip from Step 6>",
  "annotated_transcript": "<original transcript with [FILLER] on filler words, [STRONG] on confident language>",
  "grade": "<A | B | C | D | F — must match Step 6 conclusion>",
  "grade_reasoning": "<one sentence explaining the grade, consistent with Step 6>",
  "cot_reasoning_summary": "<2-sentence summary of your chain-of-thought for audit purposes>"
}}"""

        try:
            json_resp = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": json_prompt}],
                temperature=0.1,   # low temp: lock in the conclusion already reached
                max_tokens=1200,
            )
            raw = json_resp.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result = json.loads(raw)
            result["_cot_used"] = True   # audit flag — visible in /evaluate nlp_detail
            print(f"[analyzer v3 CoT] JSON extraction complete — grade={result.get('grade','?')}, "
                  f"rec={result.get('hr_recommendation','?')}")
            return result
        except Exception as e:
            print(f"[analyzer v3 CoT] JSON extraction failed: {e} — retrying single-shot with reasoning context")
            return await self._groq_hr_feedback_single_shot(
                transcript, question, question_type, reasoning_context=reasoning)

    async def _groq_hr_feedback_single_shot(
        self, transcript: str, question: str,
        question_type: str, reasoning_context: str = ""
    ) -> Dict:
        """
        Single-shot fallback used when CoT Call 1 fails, or as a retry when
        Call 2 JSON parse fails (in which case reasoning_context injects the
        already-completed reasoning so the score isn't generated cold).
        """
        context_block = ""
        if reasoning_context:
            context_block = (
                f"\nYou have already reasoned through this answer:\n"
                f"<reasoning>\n{reasoning_context[:800]}\n</reasoning>\n"
                f"Use this reasoning to ground your scores.\n"
            )

        prompt = f"""You are an expert HR evaluator and technical interview coach.

QUESTION TYPE: {question_type}
INTERVIEW QUESTION: {question}
CANDIDATE'S ANSWER: {transcript[:1200]}
{context_block}
Return ONLY valid JSON:
{{
  "technical_score": <0-100>,
  "relevance_score": <0-100>,
  "ideal_answer": "<60-100 word model answer>",
  "technical_evaluation": "<2-3 sentence assessment>",
  "key_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvement_areas": ["<area 1>", "<area 2>", "<area 3>"],
  "hr_recommendation": "<Strong Yes | Yes | Maybe | No>",
  "hr_reasoning": "<2-3 sentences>",
  "coaching_tip": "<1 specific actionable tip>",
  "annotated_transcript": "<transcript with [FILLER] and [STRONG] tags>",
  "grade": "<A | B | C | D | F>",
  "grade_reasoning": "<one sentence>"
}}

Judge technical answers strictly on accuracy."""

        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except Exception as e:
            print(f"[analyzer v3] Single-shot HR feedback failed: {e}")
            return self._fallback_hr(transcript)

    def _fallback_hr(self, transcript: str) -> Dict:
        return {
            "technical_score": 70, "relevance_score": 75, "ideal_answer": "",
            "technical_evaluation": "Set GROQ_API_KEY for deep technical evaluation.",
            "key_strengths": ["Answer provided", "Reasonable length", "Shows effort"],
            "improvement_areas": ["Add specific metrics", "Use STAR structure", "Reduce filler words"],
            "hr_recommendation": "Maybe", "hr_reasoning": "Full evaluation requires Groq API key.",
            "coaching_tip": "Add specific numbers and outcomes to strengthen your answer.",
            "annotated_transcript": transcript, "grade": "B",
            "grade_reasoning": "Default grade — configure GROQ_API_KEY for real evaluation.",
        }

    # ── Legacy NLP helpers (unchanged) ────────────────────────────────────────

    def _detect_fillers(self, text: str) -> Dict:
        text_lower = text.lower()
        total = 0
        found: Dict = {}
        TAXONOMY = {
            "hesitation":     ["um", "uh", "er", "ah", "hmm", "uhm"],
            "vague":          ["like", "you know", "sort of", "kind of", "basically", "literally"],
            "filler_phrases": ["i mean", "you see", "right", "okay so", "so yeah", "anyway"],
        }
        for cat, words in TAXONOMY.items():
            hits = []
            for w in words:
                count = len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower))
                if count > 0:
                    hits.append({"word": w, "count": count})
                    total += count
            if hits:
                found[cat] = hits
        wc = len(text.split())
        return {"total_count": total,
                "filler_rate_percent": round((total / max(wc, 1)) * 100, 1),
                "by_category": found, "word_count": wc}

    def _score_language_confidence(self, text: str) -> Dict:
        text_lower = text.lower()
        strong_hits = [m for m in CONFIDENCE_MARKERS["strong"] if m in text_lower]
        weak_hits   = [m for m in CONFIDENCE_MARKERS["weak"]
                       if re.search(r'\b' + re.escape(m) + r'\b', text_lower)]
        s, w = len(strong_hits), len(weak_hits)
        total = s + w
        score = 60 if total == 0 else min(100, int((s / total) * 100 * 1.2))
        return {"score": score, "strong_markers": strong_hits[:8],
                "weak_markers": weak_hits[:8], "strong_count": s, "weak_count": w}

    def _score_clarity(self, text: str) -> int:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text.strip()) if len(s.strip()) > 10]
        if not sentences:
            return 50
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = (100 if 15 <= avg_words <= 25 else 60 if avg_words < 10
                        else 50 if avg_words > 40 else 80)
        variety_score = min(100, len(sentences) * 15)
        return int((length_score + variety_score) / 2)

    def _voice_nervousness_proxy(self, text: str) -> float:
        if not text or not text.strip():
            return 0.30
        tl    = text.lower()
        words  = tl.split()
        wc    = max(len(words), 1)
        filler_count = sum(len(re.findall(r'\b' + re.escape(f) + r'\b', tl)) for f in FILLER_WORDS)
        f1 = _clamp(filler_count / wc / 0.20, 0.0, 1.0)
        _HEDGES = ["i think", "maybe", "i guess", "i hope", "kind of", "sort of",
                   "probably", "i feel like", "perhaps", "i suppose", "not sure",
                   "i'm not sure", "i believe", "might be", "could be", "not really"]
        hedge_count = sum(tl.count(h) for h in _HEDGES)
        f2 = _clamp(hedge_count / wc / 0.15, 0.0, 1.0)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text.strip()) if len(s.strip()) > 3]
        if len(sentences) >= 3:
            s_lens = [len(s.split()) for s in sentences]
            mean_sl = sum(s_lens) / len(s_lens)
            if mean_sl > 0:
                variance = sum((x - mean_sl)**2 for x in s_lens) / len(s_lens)
                f3 = _clamp((variance**0.5) / mean_sl / 1.5, 0.0, 1.0)
            else:
                f3 = 0.5
        elif len(sentences) == 2:
            f3 = 0.4
        else:
            f3 = 0.6
        unique_words = set(words)
        ttr = len(unique_words) / wc
        f4 = _clamp((0.7 - ttr) / 0.4, 0.0, 1.0) if ttr < 0.7 else 0.0
        first_person = sum(words.count(p) for p in ["i", "me", "my", "myself"])
        f5 = _clamp(first_person / wc / 0.25, 0.0, 1.0)
        return round(_clamp(0.30*f1 + 0.20*f2 + 0.20*f3 + 0.15*f4 + 0.15*f5, 0.0, 0.95), 3)

    # ── Main analyze() — v3.0 ─────────────────────────────────────────────────

    async def analyze(
        self,
        transcript: str,
        question:   str,
        question_type: str    = "technical",
        audio_path: str       = "",
        duration_s: float     = 0.0,
        difficulty: str       = "medium",
        question_dict: dict   = None,
        webcam_data: dict     = None,
        # ── Flat nervousness kwargs accepted from main.py /evaluate ──────────
        keywords: list                  = None,
        au_intensities: dict            = None,
        blink_rate_per_min: float       = 15.0,
        inter_blink_intervals: list     = None,
        gaze_contact_ratio: float       = 0.70,
        gaze_direction_std_deg: float   = 0.0,
        yaw_angles: list                = None,
        pitch_angles: list              = None,
        ear_time_series: list           = None,
        cultural_context: str           = "auto",
    ) -> Dict:
        """
        Main analysis pipeline — v3.0.

        Differences from v2.0:
          • _full_evaluate() → _full_evaluate_v3()
          • Per-answer OCEAN scores accumulated for get_session_ocean()
          • New detail keys: star_detail_v3, depth_detail_v3,
            grammar_detail_v3, ocean_detail_v3 in nlp output
        """
        if not transcript or not transcript.strip():
            return {"success": False, "error": "Empty transcript"}

        # ── Merge webcam_data from flat kwargs if not supplied as dict ─────────
        if webcam_data is None:
            webcam_data = {
                "au_intensities":         au_intensities,
                "blink_rate_per_min":     blink_rate_per_min,
                "inter_blink_intervals":  inter_blink_intervals or [],
                "gaze_contact_ratio":     gaze_contact_ratio,
                "gaze_direction_std_deg": gaze_direction_std_deg,
                "yaw_angles":             yaw_angles or [],
                "pitch_angles":           pitch_angles or [],
                "ear_time_series":        ear_time_series or [],
            }

        q_dict   = question_dict or {}
        # Keywords may be supplied as a flat kwarg or inside question_dict
        kw       = keywords if keywords is not None else q_dict.get("keywords", [])
        ideal    = q_dict.get("ideal_answer", "")
        # Re-bind so the rest of the method uses the resolved values
        keywords = kw

        # ── Step 1: local NLP (sync, in executor) ────────────────────────────
        loop = asyncio.get_event_loop()
        nlp_raw = await loop.run_in_executor(
            None, self._run_local_nlp,
            transcript, question_type, kw, difficulty, duration_s,
        )

        # ── Step 2: Groq relevance (async) ───────────────────────────────────
        rel_score, rel_source = await self._groq_relevance(
            transcript, ideal or transcript[:300], question, question_type)

        # ── Step 3: Full type-aware evaluation v3.0 ──────────────────────────
        full_eval = _full_evaluate_v3(
            transcript, question_type, kw, rel_score, duration_s, difficulty,
            cultural_context=cultural_context)

        # Accumulate OCEAN for session-level aggregation
        self._session_oceans.append(full_eval["ocean_scores"])

        # Accumulate answer record for cross-question coherence analysis (Feature 6).
        # We append a minimal record now; the coherence report re-reads ocean_scores
        # per answer so it has per-answer trait signals, not just session means.
        self._session_answers.append({
            "question":      question,
            "answer":        transcript,
            "question_type": question_type,
            "ocean_scores":  full_eval["ocean_scores"],
        })

        # ── Step 4: Groq HR feedback (async) ─────────────────────────────────
        hr_result = await self._groq_hr_feedback(transcript, question, question_type)

        # ── Step 5: Nervousness ───────────────────────────────────────────────
        text_proxy = self._voice_nervousness_proxy(transcript)
        facial_result = compute_facial_nervousness(**(webcam_data or {}))

        voice_nervousness = text_proxy
        acoustic_detail   = {"method": "no_audio_path"}
        if audio_path:
            try:
                ac_result = acoustic_analyser.analyse(audio_path)
                if ac_result.get("available"):
                    voice_nervousness = ac_result["nervousness_score"]
                    acoustic_detail   = ac_result
                else:
                    acoustic_detail = {"method": "unavailable"}
            except Exception:
                acoustic_detail = {"method": "unavailable"}

        fused_nervousness = fuse_nervousness(
            facial_result["facial_nervousness"], voice_nervousness)

        # ── Step 6: Final aggregation ─────────────────────────────────────────
        knowledge_sc = full_eval["final_score"]
        emotion_sc   = round(_clamp((1.0 - fused_nervousness) * 5, 1, 5), 2)
        voice_sc     = full_eval["fluency_score"]
        avg_depth    = full_eval["depth_score"]
        session_agg  = aggregate_session_score(knowledge_sc, emotion_sc, voice_sc, avg_depth)

        # ── Legacy fields ─────────────────────────────────────────────────────
        filler_data = self._detect_fillers(transcript)
        conf_data   = self._score_language_confidence(transcript)
        clarity     = self._score_clarity(transcript)
        filler_pen  = min(30, filler_data["filler_rate_percent"] * 2)
        adj_conf    = max(0, conf_data["score"] - filler_pen)
        legacy_overall = int(
            adj_conf * 0.30 + clarity * 0.25 +
            full_eval["star_score"] / 5 * 100 * 0.25 +
            (100 - min(100, filler_pen * 3)) * 0.20
        )

        # ── Session OCEAN (if ≥2 answers this session) ───────────────────────
        session_ocean = self.get_session_ocean()

        # ── Cross-question coherence (Feature 6 — if ≥2 answers) ─────────────
        coherence_report = self.get_coherence_report()

        return {
            "success": True,

            "scores": {
                "overall":       full_eval["final_score"],
                "confidence":    adj_conf,
                "clarity":       clarity,
                "structure":     round(full_eval["star_score"] / 5 * 100),
                "technical":     round(_clamp(hr_result.get("technical_score", 70), 0, 100)),
                "relevance":     round(_clamp(hr_result.get("relevance_score", 75), 0, 100)),
                "depth":         full_eval["depth_score"],
                "fluency":       full_eval["fluency_score"],
                "knowledge_1_5": knowledge_sc,
                "session_final": session_agg["final"],
            },

            "grade":           hr_result.get("grade", "B"),
            "grade_reasoning": hr_result.get("grade_reasoning", ""),

            "nlp": {
                "fillers":             filler_data,
                "confidence_markers":  conf_data,
                # STAR v3.0
                "star_scores":         full_eval["star_scores"],
                "star_order_bonus":    full_eval["order_bonus"],
                "star_detail_v3":      full_eval["star_detail_v3"],       # NEW
                # Word category
                "disc_traits":         full_eval["disc_traits"],
                "disc_dominant":       full_eval["disc_dominant"],
                # OCEAN v3.0
                "ocean_scores":        full_eval["ocean_scores"],
                "ocean_detail_v3":     full_eval["ocean_detail_v3"],      # NEW
                "personality_nlp":     full_eval["personality_nlp"],
                "conscientiousness":   full_eval["conscientiousness"],
                # Depth v3.0
                "depth_detail_v3":     full_eval["depth_detail_v3"],      # NEW
                # Grammar v3.0
                "grammar_detail_v3":   full_eval["grammar_detail_v3"],    # NEW
                # Word category
                "word_category": {
                    "score":        full_eval["word_cat_score"],
                    "quant_count":  full_eval["quant_count"],
                    "percep_count": full_eval["percep_count"],
                },
                "sentiment_intensity": full_eval["sentiment_intensity"],
                "vocab_diversity":     full_eval["vocab_diversity"],
                "hiring_signal":       full_eval["hiring_signal"],
                "keyword_hits":        full_eval["keyword_hits"],
                "keyword_score":       full_eval["keyword_score"],
                "relevance_source":    rel_source,
                "depth_fluency_sc":    full_eval["depth_fluency_sc"],
                "wpm_score":           full_eval["wpm_score"],
                "filler_ratio":        full_eval["filler_ratio"],
                "word_count":          full_eval["word_count"],
                "time_data":           full_eval["time_data"],
                "weight_profile":      full_eval["weight_profile"],
                "question_type":       full_eval["question_type"],
                "legacy_overall_score": legacy_overall,
            },

            "nervousness": {
                "fused":             fused_nervousness,
                "voice":             voice_nervousness,
                "voice_text_proxy":  text_proxy,
                "facial":            facial_result["facial_nervousness"],
                "level": ("High" if fused_nervousness >= 0.65 else
                           "Moderate" if fused_nervousness >= 0.35 else "Low"),
                "fusion_weights":    NERVOUSNESS_FUSION,
                "facial_detail": {
                    "au_score":           facial_result["au_score"],
                    "blink_anxiety":      facial_result["blink_anxiety"],
                    "gaze_score":         facial_result["gaze_score"],
                    "pose_score":         facial_result["pose_score"],
                    "ear_score":          facial_result["ear_score"],
                    "blink_rate_per_min": facial_result["blink_rate_per_min"],
                    "gaze_contact_ratio": facial_result["gaze_contact_ratio"],
                },
                "acoustic_detail": acoustic_detail,
            },

            "session_scores": session_agg,

            # NEW: multi-answer OCEAN (populated after ≥2 answers)
            "session_ocean": session_ocean,

            # Cross-question thematic coherence (Feature 6 — populated after ≥2 answers)
            "coherence_report": coherence_report.to_dict(),

            "ai_evaluation": {
                "technical_evaluation": hr_result.get("technical_evaluation", ""),
                "key_strengths":        hr_result.get("key_strengths", []),
                "improvement_areas":    hr_result.get("improvement_areas", []),
                "coaching_tip":         hr_result.get("coaching_tip", ""),
                "ideal_answer":         ideal,
            },

            "hr_feedback": {
                "recommendation": hr_result.get("hr_recommendation", "Maybe"),
                "reasoning":      hr_result.get("hr_reasoning", ""),
            },

            "annotated_transcript": hr_result.get("annotated_transcript", transcript),
        }

    def _run_local_nlp(self, transcript: str, question_type: str,
                        keywords: List[str], difficulty: str,
                        duration_s: float) -> Dict:
        """Sync NLP pre-pass (runs in executor thread). Unchanged from v2.0."""
        al = transcript.lower()
        words = al.split()
        wc = len(words)
        filler_count = sum(al.split().count(f) for f in FILLER_WORDS)
        filler_ratio = filler_count / max(1, wc)
        fluency_sc   = _compute_fluency_score(filler_ratio)
        depth_sc, _  = _compute_depth_score_v3(wc, _resolve_type(question_type), transcript)
        wpm_sc       = _compute_wpm_score(wc, duration_s)
        depth_flu    = _compute_depth_fluency(depth_sc, fluency_sc, wpm_sc)
        star_sc, star_map, bonus, _ = _compute_star_score_v3(al)
        word_cat_sc, qc, pc = _compute_word_cat_score(words, wc)
        ocean, _     = _compute_ocean_v3(al)
        sentiment    = _compute_sentiment(al)
        return {
            "filler_count": filler_count, "filler_ratio": filler_ratio,
            "fluency_sc": fluency_sc, "depth_sc": depth_sc, "wpm_sc": wpm_sc,
            "depth_flu": depth_flu, "star_sc": star_sc, "star_map": star_map,
            "order_bonus": bonus, "word_cat_sc": word_cat_sc,
            "quant_count": qc, "percep_count": pc, "ocean": ocean,
            "sentiment": sentiment, "wc": wc,
        }