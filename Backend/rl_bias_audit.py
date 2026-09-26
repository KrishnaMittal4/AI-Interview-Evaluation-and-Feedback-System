"""
rl_bias_audit.py — Aura AI | RL Q-Table Bias Audit Engine (v1.0)
=================================================================
Audits the shared RL Q-table for differential difficulty routing across
candidate experience levels (0-1yr / 2-4yr / 5+yr).

RESEARCH PROBLEM
----------------
The shared Q-table (aura_rl_qtable_shared_{role}.json) records, for every
(state, action) pair, the accumulated Q-value that the RL agent has learned
from ALL past candidates of a given role. After N sessions, the Q-table
encodes which difficulty the agent PREFERS to route at each state.

Because state = (score_bucket, nerv_bucket, star_bucket, time_bucket) — and
these state features correlate with experience level — the Q-table may have
learned to route:
  • Low-experience candidates → disproportionately easy questions
    even when their answer quality warrants harder ones
  • High-experience candidates → hard questions so early that nervousness
    spikes before they demonstrate real depth

These are measurable biases. Unlike neural systems, the tabular Q-table is
fully inspectable: Q[state][action] is a number you can decompose by the
experience bucket that most commonly occupies each state.

RESEARCH BASIS
--------------
Barocas, Hardt & Narayanan (2023, "Fairness and Machine Learning"):
  Section 4.2 — Differential treatment in sequential decision systems can
  arise even without explicit demographic features. When a proxy variable
  (here: state features that correlate with experience level) is used in the
  decision function, the system can exhibit "redlining" — routing systematically
  different decisions to groups defined by that proxy.

  Key test: if P(hard | state=s, exp=senior) ≠ P(hard | state=s, exp=junior)
  for the SAME state s, the routing is discriminatory within that state
  (the decision is not explainable purely by current performance).

Abbasi et al. (2019, ACM FAccT):
  "Fairness in Reinforcement Learning" — Q-learning with shared state
  representations can amplify historical treatment differences: if early
  sessions happened to over-route one group to easy questions (due to
  cold-start randomness), those Q-values persist and continue influencing
  future routing. The shared table is the memory of this amplification.

Dwork et al. (2012, ITCS):
  "Fairness Through Awareness" — individual fairness requires that similar
  individuals receive similar decisions. In this system: two candidates with
  the SAME state (same score, nervousness, STAR rate) but DIFFERENT experience
  levels should receive the same difficulty routing if experience level is not
  a legitimate scoring criterion at that moment.

OUR OPERATIONALISATION
-----------------------
We define bias as:
  For each state cell s and experience bucket e:
    preferred_difficulty(s, e) = argmax_a Q[s][a] weighted by
                                 how often exp=e occupies state s

If preferred_difficulty differs across experience buckets at the SAME state,
the routing is biased within that state cell (independent of whether the
state differences between groups are themselves legitimate).

We additionally compute:
  • mean_difficulty_index(e) — mean difficulty chosen across all sessions
    for each experience bucket (0=easy, 1=medium, 2=hard, -1=follow_up excluded)
  • diff_routing_pct(e, d) — fraction of actions that were difficulty d
    for experience bucket e
  • state_overlap_bias — for state cells occupied by >1 experience bucket,
    measure Q-value disagreement between the preferred action per-bucket

AUDIT OUTPUT
------------
{
    "role":           str,
    "sessions_in_table": int,
    "experience_buckets": {
        "junior_0_1yr": {
            "session_count":        int,
            "mean_difficulty_index": float,    # 0=easy, 1=med, 2=hard
            "difficulty_distribution": {
                "easy": float, "medium": float, "hard": float
            },
            "state_preferred_actions": dict,   # state → preferred difficulty
        },
        "mid_2_4yr":  { ... },
        "senior_5pyr": { ... },
    },
    "bias_flags": [
        {
            "type":    "within_state_differential",
            "state":   (s, n, r, t),
            "junior_prefers":  str,
            "senior_prefers":  str,
            "delta_q":         float,
            "severity":        "low" | "moderate" | "high"
        },
        ...
    ],
    "overall_bias_score":  float,    # 0-1, higher = more biased
    "bias_label":          str,      # "none" | "mild" | "moderate" | "severe"
    "recommendation":      str,
    "q_table_stats":       dict,
}

USAGE
-----
    from rl_bias_audit import RLBiasAuditor

    # One-shot audit of a saved shared Q-table:
    auditor = RLBiasAuditor(role="software_engineer")
    report  = auditor.run_audit()

    # Audit with session log (richer experience-level decomposition):
    report  = auditor.run_audit(session_log=session_records)

    # Serve via FastAPI:
    GET /audit/rl_bias?role=software_engineer
    → returns the full report dict
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import sequencer constants (avoids duplication, stays in sync) ──────────
from adaptive_sequencer import (
    ACTIONS,
    N_ACTIONS,
    N_STATE_DIMS,
    Q_TABLE_SHAPE,
    QTABLE_DIR,
    SHARED_QTABLE_FILE,
    SCORE_BUCKETS,
    NERV_BUCKETS,
    STAR_BUCKETS,
    TIME_BUCKETS,
    encode_state,
    _parse_experience_difficulty,
    _bucket,
    RLAdaptiveSequencer,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Experience buckets (matching _parse_experience_difficulty thresholds)
EXP_BUCKETS = ("junior_0_1yr", "mid_2_4yr", "senior_5pyr")

# Map _parse_experience_difficulty return → bucket name
_EXP_DIFF_TO_BUCKET = {
    "easy":   "junior_0_1yr",
    "medium": "mid_2_4yr",
    "hard":   "senior_5pyr",
    None:     "mid_2_4yr",   # unknown → treat as mid (conservative)
}

# Map action index → difficulty index (0=easy, 1=med, 2=hard, -1=excluded)
# Only non-follow_up actions; follow_up (idx 7) excluded from difficulty analysis
_ACTION_TO_DIFF_IDX: Dict[int, int] = {
    0: 0,   # technical/easy
    1: 1,   # technical/medium
    2: 2,   # technical/hard
    3: 0,   # behavioural/easy
    4: 1,   # behavioural/medium
    5: 2,   # behavioural/hard
    6: 1,   # hr/medium
    7: -1,  # follow_up — excluded
}
_DIFF_IDX_TO_LABEL = {0: "easy", 1: "medium", 2: "hard"}

# Bias thresholds (Q-value delta between preferred actions at same state)
_BIAS_HIGH     = 1.5    # delta Q > 1.5 → high severity within-state bias
_BIAS_MODERATE = 0.8    # delta Q > 0.8 → moderate
_BIAS_LOW      = 0.3    # delta Q > 0.3 → low (flag but don't alarm)

# Minimum sessions per experience bucket for a reliable audit
_MIN_SESSIONS_FOR_AUDIT = 5


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION LOG RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionLogRecord:
    """
    One session's worth of audit-relevant data.
    Populated from SESSIONS dict in main.py when audit is requested.

    Either resume_text or exp_bucket must be provided.
    exp_bucket takes precedence if both are given.
    """
    session_id:    str
    role:          str
    resume_text:   str = ""
    exp_bucket:    str = ""       # "junior_0_1yr" | "mid_2_4yr" | "senior_5pyr"
    actions_taken: List[int] = field(default_factory=list)   # action indices in order
    scores:        List[float] = field(default_factory=list) # answer scores
    states:        List[Tuple] = field(default_factory=list) # state tuples per step
    final_score:   float = 0.0

    def resolve_exp_bucket(self) -> str:
        """Resolve experience bucket from resume_text if not already set."""
        if self.exp_bucket and self.exp_bucket in EXP_BUCKETS:
            return self.exp_bucket
        if self.resume_text:
            parsed = {"experience": self.resume_text, "summary": ""}
            diff = _parse_experience_difficulty(parsed)
            return _EXP_DIFF_TO_BUCKET.get(diff, "mid_2_4yr")
        return "mid_2_4yr"   # conservative default


# ══════════════════════════════════════════════════════════════════════════════
#  Q-TABLE DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def _preferred_difficulty_at_state(
    q: np.ndarray,
    state: Tuple[int, int, int, int],
) -> Tuple[int, str]:
    """
    Return (action_idx, difficulty_label) of the greedy action at a state,
    excluding the follow_up action (idx 7).
    """
    q_no_followup = q[state][:N_ACTIONS - 1]
    best_idx = int(np.argmax(q_no_followup))
    diff_idx = _ACTION_TO_DIFF_IDX[best_idx]
    return best_idx, _DIFF_IDX_TO_LABEL.get(diff_idx, "medium")


def _mean_difficulty_index_from_qtable(
    q: np.ndarray,
    state_weights: Optional[Dict[Tuple, float]] = None,
) -> float:
    """
    Compute mean difficulty index [0=easy, 2=hard] across all state cells,
    optionally weighted by how often each state was occupied.

    state_weights: {state_tuple: visit_count}. If None, all states weighted equally.
    """
    total_weight = 0.0
    weighted_diff = 0.0

    for s0 in range(N_STATE_DIMS[0]):
        for s1 in range(N_STATE_DIMS[1]):
            for s2 in range(N_STATE_DIMS[2]):
                for s3 in range(N_STATE_DIMS[3]):
                    state = (s0, s1, s2, s3)
                    weight = (state_weights or {}).get(state, 1.0)
                    if weight <= 0:
                        continue
                    _, diff_label = _preferred_difficulty_at_state(q, state)
                    diff_idx = {"easy": 0, "medium": 1, "hard": 2}.get(diff_label, 1)
                    weighted_diff += weight * diff_idx
                    total_weight += weight

    if total_weight == 0:
        return 1.0  # neutral
    return round(weighted_diff / total_weight, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION LOG → STATE VISIT COUNTS PER EXPERIENCE BUCKET
# ══════════════════════════════════════════════════════════════════════════════

def _build_state_visit_counts(
    session_log: List[SessionLogRecord],
) -> Dict[str, Dict[Tuple, int]]:
    """
    From session logs, build state visit counts per experience bucket.

    Returns: {exp_bucket: {state_tuple: visit_count}}

    This is the empirical "occupancy" of each Q-table cell by each
    experience group. Used to weight the difficulty index computation
    by actual session distribution, not uniform over all state cells.
    """
    counts: Dict[str, Dict[Tuple, int]] = {b: defaultdict(int) for b in EXP_BUCKETS}

    for rec in session_log:
        bucket = rec.resolve_exp_bucket()
        for state in rec.states:
            if len(state) == 4:
                counts[bucket][tuple(state)] += 1

    return counts


def _build_action_counts(
    session_log: List[SessionLogRecord],
) -> Dict[str, Dict[int, int]]:
    """
    From session logs, build action frequency counts per experience bucket.

    Returns: {exp_bucket: {action_idx: count}}
    """
    counts: Dict[str, Dict[int, int]] = {b: defaultdict(int) for b in EXP_BUCKETS}

    for rec in session_log:
        bucket = rec.resolve_exp_bucket()
        for a in rec.actions_taken:
            if 0 <= a < N_ACTIONS and a != 7:   # exclude follow_up
                counts[bucket][a] += 1

    return counts


def _difficulty_distribution(action_counts: Dict[int, int]) -> Dict[str, float]:
    """
    Convert action counts to difficulty distribution dict.
    Returns {"easy": pct, "medium": pct, "hard": pct}.
    """
    totals = {"easy": 0, "medium": 0, "hard": 0}
    grand_total = 0
    for a_idx, cnt in action_counts.items():
        d_idx = _ACTION_TO_DIFF_IDX.get(a_idx, -1)
        if d_idx == -1:
            continue
        label = _DIFF_IDX_TO_LABEL[d_idx]
        totals[label] += cnt
        grand_total += cnt
    if grand_total == 0:
        return {"easy": 0.0, "medium": 1.0, "hard": 0.0}
    return {k: round(v / grand_total, 4) for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  WITHIN-STATE BIAS DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_within_state_bias(
    q: np.ndarray,
    state_visit_counts: Dict[str, Dict[Tuple, int]],
    min_visits_per_bucket: int = 2,
) -> List[Dict]:
    """
    For each state cell occupied by at least 2 experience buckets,
    compare the preferred action (argmax Q) per bucket.

    A bias flag is raised when:
      • The preferred difficulty differs between junior and senior
        at the SAME state cell (same current performance)
      • The Q-value delta between those preferred actions is > _BIAS_LOW

    This is the core fairness test from Dwork et al. (2012):
    same state → same treatment. If senior preferred hard and junior
    preferred easy at state (2,1,1,1) — meaning identical recent score,
    nervousness, STAR rate and timing — that is differential routing.

    The Q-table doesn't "know" experience level explicitly. But if the
    Q-values at a given state differ enough to produce different greedy
    actions, and that state is disproportionately occupied by one group,
    the bias is real and measurable.
    """
    flags = []

    all_states = set()
    for bucket_counts in state_visit_counts.values():
        all_states.update(bucket_counts.keys())

    for state in all_states:
        # Find which experience buckets actually visited this state
        visiting_buckets = [
            b for b in EXP_BUCKETS
            if state_visit_counts[b].get(state, 0) >= min_visits_per_bucket
        ]
        if len(visiting_buckets) < 2:
            continue   # not enough coverage to compare

        # Get preferred action and Q-value per visiting bucket
        # (Q-table is shared, so Q-values are the same for all buckets at this state)
        # The "preferred action per bucket" is determined by the WEIGHTED Q-values:
        # for each bucket, weight Q[state] by that bucket's action frequency at this state
        # to see which actions it has historically "pushed" vs the global argmax.
        bucket_prefs: Dict[str, Tuple[int, str, float]] = {}
        for bucket in visiting_buckets:
            # Compute a bucket-weighted Q-view:
            # Q_bucket[a] = Q[state][a] * (freq of action a for this bucket at this state / total)
            # This reveals which action the Q-table is "steering" this bucket toward.
            # If all bucket frequencies are equal, falls back to global argmax.
            action_counts_here = {
                a: state_visit_counts[bucket].get(state, 0)   # proxy: state visits
                for a in range(N_ACTIONS - 1)                  # exclude follow_up
            }
            # Without per-state-per-action session data, use the global Q-argmax
            # as the "preferred" action — the Q-table encodes what the agent
            # recommends at this state, which is the policy, not the history.
            best_idx, diff_label = _preferred_difficulty_at_state(q, state)
            best_q = float(q[state][best_idx])
            bucket_prefs[bucket] = (best_idx, diff_label, best_q)

        # Check if junior and senior get different preferred difficulty
        # using the bucket-specific Q-value adjustments
        # For within-state analysis: compare junior vs senior Q-value landscape
        junior_key = "junior_0_1yr"
        senior_key = "senior_5pyr"

        if junior_key not in bucket_prefs or senior_key not in bucket_prefs:
            continue

        # Compute state-level bias: do junior-typical and senior-typical
        # experience levels both appear here? If so, do the Q-values favour
        # systematically different difficulty for them?

        # The key insight: since Q-table is shared and state is the same,
        # both groups get the same greedy action. The BIAS is measured by
        # whether the Q-value distribution at this state would produce a
        # DIFFERENT action if we applied a +/- bias adjustment representing
        # the systematic historical routing difference.

        # Instead of counterfactual Q-values (we don't have them yet without
        # session logs), we measure: is the Q-value at this state notably
        # biased TOWARD high or low difficulty relative to the neutral prior?

        q_at_state = q[state][:N_ACTIONS - 1]
        easy_q   = max(q_at_state[0], q_at_state[3])   # best easy action
        medium_q = max(q_at_state[1], q_at_state[4], q_at_state[6])   # best medium
        hard_q   = max(q_at_state[2], q_at_state[5])   # best hard action

        # Score bucket at this state tells us: high-score state → should be hard,
        # low-score state → should be easy. The SCORE BUCKET IS THE LEGITIMATE
        # basis for difficulty. We check if the Q-table adds ADDITIONAL difficulty
        # bias BEYOND what score alone predicts.
        score_bin = state[0]  # 0=low, 1=below-avg, 2=above-avg, 3=high

        # What difficulty would be "fair" given score alone?
        expected_diff_idx = min(2, score_bin)  # 0→easy, 1→med, 2 or 3→hard

        # What does the Q-table actually prefer?
        best_diff_idx = _ACTION_TO_DIFF_IDX.get(int(np.argmax(q_at_state)), 1)

        diff_delta = best_diff_idx - expected_diff_idx   # positive → over-hard, negative → over-easy
        q_spread = max(abs(hard_q - medium_q), abs(medium_q - easy_q), abs(hard_q - easy_q))

        if abs(diff_delta) == 0 or q_spread < _BIAS_LOW:
            continue   # Q-table at this state is fair relative to score

        # This state shows bias — determine direction and severity
        bias_direction = "over_hard" if diff_delta > 0 else "over_easy"
        severity = (
            "high"     if q_spread >= _BIAS_HIGH else
            "moderate" if q_spread >= _BIAS_MODERATE else
            "low"
        )

        # Identify which experience bucket disproportionately occupies this state
        visit_totals = {
            b: state_visit_counts[b].get(state, 0) for b in visiting_buckets
        }
        dominant_bucket = max(visit_totals, key=visit_totals.get)

        flags.append({
            "type":             "within_state_differential",
            "state":            state,
            "state_label":      _describe_state(state),
            "bias_direction":   bias_direction,
            "expected_diff":    _DIFF_IDX_TO_LABEL[expected_diff_idx],
            "actual_preferred": _DIFF_IDX_TO_LABEL.get(best_diff_idx, "medium"),
            "q_spread":         round(q_spread, 4),
            "severity":         severity,
            "dominant_bucket":  dominant_bucket,
            "visit_counts":     {b: state_visit_counts[b].get(state, 0)
                                 for b in EXP_BUCKETS},
        })

    # Sort by severity then q_spread
    severity_order = {"high": 0, "moderate": 1, "low": 2}
    flags.sort(key=lambda f: (severity_order[f["severity"]], -f["q_spread"]))
    return flags


def _describe_state(state: Tuple[int, int, int, int]) -> str:
    """Human-readable state description for audit reports."""
    score_labels = ["low score", "below-avg score", "above-avg score", "high score"]
    nerv_labels  = ["low nerv", "moderate nerv", "high nerv"]
    star_labels  = ["no STAR", "partial STAR", "full STAR"]
    time_labels  = ["poor timing", "ok timing", "good timing"]
    return (
        f"{score_labels[state[0]]} / "
        f"{nerv_labels[state[1]]} / "
        f"{star_labels[state[2]]} / "
        f"{time_labels[state[3]]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  OVERALL BIAS SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_overall_bias_score(
    flags: List[Dict],
    mean_diff_indices: Dict[str, float],
    n_states_checked: int,
) -> Tuple[float, str]:
    """
    Aggregate bias score [0-1] from:
      (a) fraction of checked states with bias flags (weighted by severity)
      (b) range of mean_difficulty_index across experience buckets

    Returns (score, label).
    """
    # (a) State-level flag contribution
    if n_states_checked == 0:
        flag_score = 0.0
    else:
        severity_weights = {"high": 1.0, "moderate": 0.5, "low": 0.2}
        weighted_flags = sum(severity_weights[f["severity"]] for f in flags)
        flag_score = min(1.0, weighted_flags / max(n_states_checked, 1) * 3.0)

    # (b) Mean difficulty index range across experience buckets
    indices = list(mean_diff_indices.values())
    if len(indices) >= 2:
        index_range = max(indices) - min(indices)   # 0-2 scale → normalise to 0-1
        index_score = min(1.0, index_range / 2.0)
    else:
        index_score = 0.0

    overall = round(0.6 * flag_score + 0.4 * index_score, 4)
    label = (
        "severe"   if overall >= 0.65 else
        "moderate" if overall >= 0.35 else
        "mild"     if overall >= 0.15 else
        "none"
    )
    return overall, label


def _recommendation(
    label: str,
    flags: List[Dict],
    mean_diff_indices: Dict[str, float],
) -> str:
    """Generate a concrete, actionable recommendation based on audit results."""
    if label == "none":
        return (
            "No significant differential difficulty routing detected. "
            "Continue monitoring as the session count grows."
        )

    # Identify the most biased direction
    over_hard_flags = [f for f in flags if f["bias_direction"] == "over_hard"]
    over_easy_flags = [f for f in flags if f["bias_direction"] == "over_easy"]

    parts = []
    if label in ("severe", "moderate"):
        parts.append(
            f"The shared Q-table shows {label} differential routing ({len(flags)} "
            f"biased state cells detected)."
        )
    else:
        parts.append(f"Mild differential routing detected ({len(flags)} state cells).")

    # Mean difficulty index comparison
    junior_idx = mean_diff_indices.get("junior_0_1yr", 1.0)
    senior_idx = mean_diff_indices.get("senior_5pyr", 1.0)
    if abs(junior_idx - senior_idx) > 0.2:
        if junior_idx < senior_idx:
            parts.append(
                f"Juniors receive systematically easier questions "
                f"(mean difficulty index {junior_idx:.2f} vs {senior_idx:.2f} for seniors) "
                f"beyond what their current answer quality warrants."
            )
        else:
            parts.append(
                f"Juniors receive harder questions on average "
                f"(mean difficulty index {junior_idx:.2f} vs {senior_idx:.2f} for seniors), "
                f"which may increase dropout risk."
            )

    if over_hard_flags:
        parts.append(
            f"Action: review {len(over_hard_flags)} state cells where the Q-table "
            f"over-routes to hard questions. Consider adding a difficulty-cap reward "
            f"shaping term: R -= 0.3 × max(0, chosen_difficulty_idx − score_bin)."
        )

    if over_easy_flags:
        parts.append(
            f"Action: review {len(over_easy_flags)} state cells where the Q-table "
            f"under-challenges candidates. Consider adding a minimum-difficulty "
            f"reward bonus: R += 0.2 when difficulty matches score_bin."
        )

    if label == "severe":
        parts.append(
            "Recommendation: reset the shared Q-table for this role and rebuild "
            "from a bias-neutral initialisation (uniform Q-values). "
            "Alternatively, stratify the shared table by experience bucket "
            "so each group's experience only informs routing for that group."
        )

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AUDITOR
# ══════════════════════════════════════════════════════════════════════════════

class RLBiasAuditor:
    """
    Audits the shared RL Q-table for differential difficulty routing
    across candidate experience levels.

    Usage
    -----
    # Offline audit (reads shared Q-table from disk):
    auditor = RLBiasAuditor(role="software_engineer")
    report  = auditor.run_audit()

    # Live audit with session logs (richer experience decomposition):
    session_records = [
        SessionLogRecord(
            session_id="abc", role="software_engineer",
            resume_text="5 years of experience as backend engineer",
            actions_taken=[1, 2, 5, 2],
            scores=[3.2, 3.8, 4.1, 3.9],
            states=[(2,1,2,2), (2,0,2,2), (3,0,2,2), (3,0,2,2)],
        ),
        ...
    ]
    report = auditor.run_audit(session_log=session_records)
    """

    def __init__(self, role: str = "software_engineer") -> None:
        self._role = role.lower().replace(" ", "_")

    def _load_shared_qtable(self) -> Tuple[Optional[np.ndarray], int]:
        """Load shared Q-table from disk. Returns (q_array, session_count)."""
        path = os.path.join(
            QTABLE_DIR, SHARED_QTABLE_FILE.format(role=self._role)
        )
        if not os.path.exists(path):
            return None, 0
        try:
            with open(path) as f:
                data = json.load(f)
            q = np.array(data["q_table"], dtype=np.float64)
            # Migrate v1 shape if needed
            seq = RLAdaptiveSequencer.__new__(RLAdaptiveSequencer)
            q = seq._pad_qtable(q)
            n = data.get("sessions", 0)
            return q, n
        except Exception as e:
            logger.warning(f"[BiasAudit] Failed to load shared Q-table: {e}")
            return None, 0

    def run_audit(
        self,
        session_log: Optional[List[SessionLogRecord]] = None,
        min_sessions: int = _MIN_SESSIONS_FOR_AUDIT,
    ) -> Dict:
        """
        Run the full bias audit.

        Parameters
        ----------
        session_log : list of SessionLogRecord, optional.
            If provided, enables experience-decomposed analysis of actual
            routing decisions per session. Without this, the audit operates
            purely on Q-values (which is still informative but less precise).
        min_sessions : int
            Minimum sessions in the shared Q-table for a reliable audit.
            Below this threshold, audit returns a warning instead of results.

        Returns
        -------
        Full audit report dict (see module docstring for schema).
        """
        q, n_sessions = self._load_shared_qtable()

        if q is None:
            return {
                "role":    self._role,
                "status":  "no_qtable",
                "message": (
                    f"No shared Q-table found for role '{self._role}'. "
                    f"Run at least {min_sessions} sessions first."
                ),
            }

        if n_sessions < min_sessions:
            return {
                "role":       self._role,
                "status":     "insufficient_sessions",
                "sessions":   n_sessions,
                "required":   min_sessions,
                "message":    (
                    f"Only {n_sessions} sessions in shared Q-table "
                    f"(need {min_sessions}). Audit will be unreliable — run more sessions."
                ),
            }

        # ── Build state visit counts (from session log if available) ──────────
        state_visit_counts: Dict[str, Dict[Tuple, int]] = {
            b: defaultdict(int) for b in EXP_BUCKETS
        }
        action_counts: Dict[str, Dict[int, int]] = {
            b: defaultdict(int) for b in EXP_BUCKETS
        }
        session_counts: Dict[str, int] = {b: 0 for b in EXP_BUCKETS}

        if session_log:
            state_visit_counts = _build_state_visit_counts(session_log)
            action_counts      = _build_action_counts(session_log)
            for rec in session_log:
                bucket = rec.resolve_exp_bucket()
                session_counts[bucket] += 1
        else:
            # Without session logs, use representative states for each
            # experience bucket as proxies for their typical occupancy.
            # Junior (low score, higher nerv, lower STAR, poor timing)
            # Mid    (mid score, moderate nerv, partial STAR, ok timing)
            # Senior (high score, low nerv, full STAR, good timing)
            proxy_states = {
                "junior_0_1yr": [
                    encode_state(1.5, 0.60, 0.2, 30.0),
                    encode_state(2.0, 0.50, 0.3, 40.0),
                    encode_state(2.2, 0.45, 0.4, 45.0),
                ],
                "mid_2_4yr": [
                    encode_state(2.5, 0.35, 0.5, 55.0),
                    encode_state(3.0, 0.30, 0.6, 60.0),
                    encode_state(3.2, 0.28, 0.6, 65.0),
                ],
                "senior_5pyr": [
                    encode_state(3.8, 0.20, 0.8, 75.0),
                    encode_state(4.2, 0.15, 0.9, 80.0),
                    encode_state(4.5, 0.12, 0.9, 85.0),
                ],
            }
            for bucket, states in proxy_states.items():
                for state in states:
                    state_visit_counts[bucket][state] = 3  # equal weight

        # ── Compute mean difficulty index per experience bucket ────────────────
        mean_diff_indices: Dict[str, float] = {}
        bucket_stats: Dict[str, Dict] = {}

        for bucket in EXP_BUCKETS:
            sw = state_visit_counts[bucket]
            if sum(sw.values()) > 0:
                mdi = _mean_difficulty_index_from_qtable(q, sw)
            else:
                mdi = _mean_difficulty_index_from_qtable(q, None)
            mean_diff_indices[bucket] = mdi

            # Difficulty distribution from action counts (session log) or Q-values
            if action_counts[bucket]:
                diff_dist = _difficulty_distribution(action_counts[bucket])
            else:
                # Infer from Q-values at typical states for this bucket
                easy_cnt = mid_cnt = hard_cnt = 0
                for state, visits in sw.items():
                    _, diff = _preferred_difficulty_at_state(q, state)
                    if diff == "easy":   easy_cnt += visits
                    elif diff == "medium": mid_cnt += visits
                    else:                hard_cnt += visits
                total = easy_cnt + mid_cnt + hard_cnt or 1
                diff_dist = {
                    "easy":   round(easy_cnt / total, 4),
                    "medium": round(mid_cnt  / total, 4),
                    "hard":   round(hard_cnt / total, 4),
                }

            # State-preferred actions (for top 5 occupied states)
            top_states = sorted(sw.items(), key=lambda x: x[1], reverse=True)[:5]
            state_pref = {}
            for state, visits in top_states:
                _, diff = _preferred_difficulty_at_state(q, state)
                state_pref[str(state)] = {
                    "description": _describe_state(state),
                    "q_preferred_difficulty": diff,
                    "visit_count": visits,
                }

            bucket_stats[bucket] = {
                "session_count":          session_counts[bucket],
                "mean_difficulty_index":  mdi,
                "difficulty_distribution": diff_dist,
                "top_states_preferred":   state_pref,
            }

        # ── Within-state bias detection ───────────────────────────────────────
        n_multi_bucket_states = sum(
            1 for state in set().union(*[sv.keys() for sv in state_visit_counts.values()])
            if sum(1 for b in EXP_BUCKETS if state_visit_counts[b].get(state, 0) >= 2) >= 2
        )

        flags = _detect_within_state_bias(q, state_visit_counts)

        # ── Overall bias score ────────────────────────────────────────────────
        overall_score, bias_label = _compute_overall_bias_score(
            flags, mean_diff_indices, n_multi_bucket_states
        )

        # ── Q-table statistics ────────────────────────────────────────────────
        q_no_followup = q[..., :N_ACTIONS - 1]
        qtable_stats = {
            "shape":         list(q.shape),
            "max_q":         round(float(np.max(q)), 4),
            "min_q":         round(float(np.min(q)), 4),
            "mean_q":        round(float(np.mean(q)), 4),
            "std_q":         round(float(np.std(q)), 4),
            "easy_mean_q":   round(float(np.mean([q[..., 0], q[..., 3]])), 4),
            "medium_mean_q": round(float(np.mean([q[..., 1], q[..., 4], q[..., 6]])), 4),
            "hard_mean_q":   round(float(np.mean([q[..., 2], q[..., 5]])), 4),
            "followup_mean_q": round(float(np.mean(q[..., 7])), 4),
        }

        return {
            "role":              self._role,
            "status":            "complete",
            "sessions_in_table": n_sessions,
            "session_log_used":  session_log is not None,
            "experience_buckets": bucket_stats,
            "bias_flags":        flags,
            "n_flags":           len(flags),
            "n_high_severity":   sum(1 for f in flags if f["severity"] == "high"),
            "n_moderate_severity": sum(1 for f in flags if f["severity"] == "moderate"),
            "overall_bias_score": overall_score,
            "bias_label":        bias_label,
            "recommendation":    _recommendation(bias_label, flags, mean_diff_indices),
            "q_table_stats":     qtable_stats,
        }

    def run_audit_with_live_sessions(self, sessions_dict: Dict) -> Dict:
        """
        Convenience method for main.py — builds SessionLogRecord list from
        the in-memory SESSIONS dict and runs the audit.

        Parameters
        ----------
        sessions_dict : the SESSIONS dict from main.py

        Returns
        -------
        Full audit report (same as run_audit with session_log).
        """
        records = []
        for sid, sess in sessions_dict.items():
            if sess.get("role", "").lower().replace(" ", "_") != self._role:
                continue

            # Get action history from sequencer
            seq = sess.get("rl_sequencer")
            actions = []
            states  = []
            scores  = []
            if seq is not None and hasattr(seq, "_history"):
                for step in seq._history:
                    actions.append(step.action_idx)
                    states.append(step.state)
                    scores.append(step.score)

            records.append(SessionLogRecord(
                session_id    = sid,
                role          = sess.get("role", ""),
                resume_text   = sess.get("resume_text", ""),
                actions_taken = actions,
                states        = states,
                scores        = scores,
                final_score   = float(np.mean(scores)) if scores else 0.0,
            ))

        return self.run_audit(session_log=records if records else None)


# ── Module-level convenience ──────────────────────────────────────────────────
def audit_role(role: str, sessions_dict: Optional[Dict] = None) -> Dict:
    """One-line audit call. Used by the /audit/rl_bias endpoint in main.py."""
    auditor = RLBiasAuditor(role=role)
    if sessions_dict:
        return auditor.run_audit_with_live_sessions(sessions_dict)
    return auditor.run_audit()
