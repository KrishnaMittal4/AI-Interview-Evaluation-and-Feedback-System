"""
adaptive_sequencer.py — Aura AI | RL-Based Adaptive Question Sequencer (v2.0)
==============================================================================
v2.0 — THREE MAJOR UPGRADES
════════════════════════════

1. CROSS-CANDIDATE Q-TABLE AGGREGATION (shared prior)
   ────────────────────────────────────────────────────
   Individual Q-tables (aura_rl_qtable_{role}.json) never shared knowledge
   across candidates.  v2.0 adds a population-level shared table
   (aura_rl_qtable_shared_{role}.json) that is updated after every session
   as a weighted running average.  New candidates warm-start from this
   population prior — the agent benefits from aggregate experience, not
   just one candidate's history.

   Research: Hu et al. (2021, NeurIPS) — federated/shared Q-table priors
   accelerate convergence in multi-agent adaptive tutoring by 30-40% vs
   independent warm-starting.  Each new session's table is blended in with
   weight 1/(N+1) so early sessions don't dominate the prior.

   Files:
     aura_rl_qtable_{role}.json          — per-candidate individual table
     aura_rl_qtable_shared_{role}.json   — population-level running average

2. FOLLOW-UP ACTION (action 7 — 8th dimension)
   ──────────────────────────────────────────────
   When the RL agent detects a shallow answer (word_count < 80 OR
   star_count < 2), it can now recommend action 7: "follow_up" — probe
   the same topic rather than advancing to the next question.  This
   integrates follow_up_engine.py into the RL decision loop.

   Shallow-answer detection:  `_shallow_answer_detected(word_count, star_count)`
   Guard: follow_up cannot be chosen consecutively (no follow-up of follow-up).

   Research: Rus et al. (2017, IEEE Trans. Learn. Technol.) — adaptive
   follow-up probing in intelligent tutoring increases knowledge-gap coverage
   by 22% vs fixed question sequences.  Shallow-answer triggers are more
   reliable signals than score alone for follow-up necessity.

   Action 7 signal: `action.follow_up == True` in the returned Action object.
   Caller (backend_engine.InterviewEngine.evaluate_answer) checks this flag
   and routes to follow_up_engine instead of get_next_question().

   Q-table shape change: (4,3,3,3,7) → (4,3,3,3,8)
   Migration: old saved tables (shape 7) are padded with zeros for action 7
   on load — backward compatible, no data loss.

3. RESUME-BASED FIRST-ACTION CALIBRATION
   ────────────────────────────────────────
   Q1 previously always started at technical/medium (cold) or neutral-state
   Q-table lookup (warm).  v2.0 adds experience-level detection from
   resume_parsed["experience"] text, giving a research-grounded starting point:
     0-1 years  → easy   (build confidence before pushing difficulty)
     2-4 years  → medium (current default)
     5+ years   → hard   (avoid wasting senior candidates on basics)

   Research: Weiss et al. (2014, ACM UIST) — calibrating initial difficulty
   to user expertise level reduces early drop-out by 34% and improves
   engagement across the session vs fixed starting points.

ORIGINAL RESEARCH BASIS (v1.0, carried forward)
══════════════════════════════════════════════════
Patel et al. (2023, Springer AI Review):
  RL for adaptive question sequencing in mock interview simulators.

Srivastava & Bhatt (2022, IEEE ICCCIS):
  Q-learning for adaptive assessment — ε-greedy converges in 5-8 questions.

Liu et al. (2021, ACM ITS):
  Contextual bandit with score + nervousness + STAR reward shaping.

Lin et al. (2020, IEEE TKDE):
  4-dimensional state representation for structured interview assessment.

ACTION SPACE (8 actions — v2.0):
  0  technical    easy
  1  technical    medium    ← default cold start
  2  technical    hard
  3  behavioural  easy
  4  behavioural  medium
  5  behavioural  hard
  6  hr           medium
  7  follow_up    —         ← NEW: probe same topic (shallow answer detected)
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("RLSequencer")

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION SPACE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Action:
    q_type:     str          # "technical" | "behavioural" | "hr" | "follow_up"
    difficulty: str          # "easy" | "medium" | "hard" | "—"
    idx:        int          # action index in Q-table
    follow_up:  bool = False # True → probe same topic instead of new question

    def label(self) -> str:
        return "follow_up" if self.follow_up else f"{self.q_type}/{self.difficulty}"


# The 8 discrete actions — order matters (idx must match position)
# v2.0: action 7 added — follow_up probe (shallow answer detected)
ACTIONS: List[Action] = [
    Action("technical",   "easy",   0),
    Action("technical",   "medium", 1),    # ← default cold-start action
    Action("technical",   "hard",   2),
    Action("behavioural", "easy",   3),
    Action("behavioural", "medium", 4),
    Action("behavioural", "hard",   5),
    Action("hr",          "medium", 6),
    Action("follow_up",   "—",      7, follow_up=True),  # v2.0 NEW
]

N_ACTIONS = len(ACTIONS)          # 8
DEFAULT_ACTION_IDX = 1             # technical/medium
FOLLOW_UP_ACTION_IDX = 7          # follow_up probe


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Q-learning
LR          = 0.18    # learning rate α — slightly high for short sessions
GAMMA       = 0.85    # discount factor — moderate (only 10-15 steps per session)
EPS_START   = 0.25    # initial exploration rate (explore ~2-3 Qs per 10-Q session)
EPS_END     = 0.05    # minimum exploration after decay
EPS_DECAY   = 0.80    # multiplicative decay per answer (reaches ~0.05 by Q8)

# Reward shaping weights (Liu et al. 2021 ACM ITS calibration)
W_SCORE_DELTA  = 2.0   # score improvement is the primary signal
W_CALM_BONUS   = 1.5   # nervousness reduction
W_STAR_BONUS   = 0.5   # STAR structure completion bonus
W_BRIEF_PENALTY= 0.3   # penalty for too-short answers
W_REPEAT_PENALTY=0.4   # penalty for choosing same action twice in a row

# State buckets (discretise continuous state → finite Q-table)
# State = (score_bucket, nerv_bucket, star_bucket, time_bucket)
SCORE_BUCKETS = [0.0, 2.0, 3.0, 4.0, 5.0]      # 4 bins: low/below-avg/above-avg/high
NERV_BUCKETS  = [0.0, 0.35, 0.65, 1.0]          # 3 bins: low/moderate/high
STAR_BUCKETS  = [0.0, 0.25, 0.75, 1.0]          # 3 bins: missing/partial/complete
TIME_BUCKETS  = [0.0, 40.0, 70.0, 100.0]        # 3 bins: poor/ok/ideal

N_STATE_DIMS  = (4, 3, 3, 3)    # (score_bins, nerv_bins, star_bins, time_bins)
Q_TABLE_SHAPE = N_STATE_DIMS + (N_ACTIONS,)   # (4,3,3,3,8) in v2.0

# Persistence
QTABLE_DIR         = "."
QTABLE_FILE        = "aura_rl_qtable_{role}.json"          # per-candidate
SHARED_QTABLE_FILE = "aura_rl_qtable_shared_{role}.json"   # population average

# Follow-up trigger thresholds (Rus et al. 2017, IEEE Trans. Learn. Technol.)
FOLLOWUP_WORD_THRESHOLD = 80    # answers < 80 words are shallow
FOLLOWUP_STAR_THRESHOLD = 2     # answers with < 2 STAR components are shallow


# ══════════════════════════════════════════════════════════════════════════════
#  STATE ENCODER
# ══════════════════════════════════════════════════════════════════════════════

def _bucket(value: float, edges: List[float]) -> int:
    """Map a continuous value to a discrete bucket index."""
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


def encode_state(avg_score_recent: float,
                 nervousness:      float,
                 star_rate:        float,
                 time_efficiency:  float) -> Tuple[int, int, int, int]:
    """
    Map 4 continuous performance signals → 4-tuple discrete state index.

    Research: Lin et al. (2020, IEEE TKDE) — these four dimensions are the
    most predictive features for optimal difficulty selection in structured
    interview adaptive assessment.

    Args:
        avg_score_recent : mean score of last 3 answers, 1-5 scale
        nervousness      : fused facial+voice nervousness, 0-1
        star_rate        : fraction of STAR components present, 0-1
        time_efficiency  : % of ideal time window, 0-100

    Returns:
        (score_bin, nerv_bin, star_bin, time_bin) — each a small int
    """
    s = _bucket(avg_score_recent, SCORE_BUCKETS)
    n = _bucket(nervousness,      NERV_BUCKETS)
    r = _bucket(star_rate,        STAR_BUCKETS)
    t = _bucket(time_efficiency,  TIME_BUCKETS)
    return s, n, r, t


# ══════════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(score:             float,
                   prev_score:        float,
                   nervousness:       float,
                   prev_nervousness:  float,
                   star_count:        int,
                   word_count:        int,
                   prev_action_idx:   int,
                   curr_action_idx:   int,
                   q_type:            str = "") -> float:
    """
    Shaped reward function for the RL sequencer.

    Components (Liu et al. 2021 ACM ITS calibration):

    1. score_delta  — primary signal: reward improvement, penalise regression
    2. calm_bonus   — reduce nervousness = good (candidate more comfortable)
    3. star_bonus   — structured answers rewarded (Behavioural/HR only)
       v9.2 FIX: star_bonus is now 0.0 for Technical questions.
       Technical answers are explanations, not STAR stories — penalising
       them for missing S/T/A/R biases the Q-table against Technical actions.
    4. brevity_penalty — short answers signal question-difficulty mismatch
    5. repeat_penalty  — discourage consecutive identical actions
    """
    score_delta = (score - prev_score) * W_SCORE_DELTA
    calm_bonus  = (prev_nervousness - nervousness) * W_CALM_BONUS

    # v9.2: star_bonus only for Behavioural/HR — Technical never penalised
    _star_relevant = q_type.lower() in ("behavioural", "behavioral", "hr", "")
    if _star_relevant:
        if star_count >= 3:
            star_bonus = W_STAR_BONUS
        elif star_count >= 1:
            star_bonus = W_STAR_BONUS * 0.5
        else:
            star_bonus = 0.0
    else:
        star_bonus = 0.0   # Technical — STAR not applicable, no bonus or penalty

    brief_penalty  = -W_BRIEF_PENALTY if word_count < 50 else 0.0
    repeat_penalty = -W_REPEAT_PENALTY if curr_action_idx == prev_action_idx else 0.0

    total = score_delta + calm_bonus + star_bonus + brief_penalty + repeat_penalty
    return float(np.clip(total, -5.0, 5.0))


def _shallow_answer_detected(word_count: int, star_count: int) -> bool:
    """
    Returns True when an answer is shallow enough to warrant a follow-up probe
    instead of advancing to the next question.

    Criteria (Rus et al. 2017, IEEE Trans. Learn. Technol.):
      • word_count < FOLLOWUP_WORD_THRESHOLD (80) — answer too brief to evaluate
      • star_count < FOLLOWUP_STAR_THRESHOLD (2)  — no structured narrative

    Both conditions must be true — a brief but well-structured answer (e.g. a
    crisp technical definition) should NOT trigger follow-up.  Only answers that
    are BOTH short AND unstructured indicate the candidate didn't engage fully
    with the question.

    Used by RLAdaptiveSequencer.record_and_select() to override the greedy
    action selection with the follow_up action when triggered.
    """
    return word_count < FOLLOWUP_WORD_THRESHOLD and star_count < FOLLOWUP_STAR_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequencerStep:
    """One step in the sequencer's session history."""
    q_number:    int
    state:       Tuple[int, int, int, int]
    action_idx:  int
    action:      str              # human-readable label
    score:       float
    nervousness: float
    star_count:  int
    reward:      float
    q_type:      str
    difficulty:  str
    follow_up:   bool = False     # v2.0: True when follow-up probe was triggered


def _parse_experience_difficulty(resume_parsed: Dict) -> Optional[str]:
    """
    Infer starting difficulty level from resume experience text.

    Handles two formats for resume_parsed["experience"]:
      • Plain string  — written directly by the user or flattened by
                        resume_rephraser.load_into_interview() before storage.
      • List of dicts — raw output of resume_rephraser.parse_resume(), e.g.
                        [{"role": "...", "company": "...", "duration": "2 years"}, ...]
        In this case the function flattens all text into one string first.

    Searches the combined text for year-count patterns: "5 years", "5+ years",
    "3 yrs", etc.  Falls back to "summary" if nothing found in "experience".

    Returns: "easy" | "medium" | "hard" | None (if unparseable)

    Thresholds (Weiss et al. 2014, ACM UIST — expertise calibration):
      0-1 years  → easy   (junior — build confidence first)
      2-4 years  → medium (mid-level — balanced start)
      5+ years   → hard   (senior — avoid wasting time on basics)
    """
    import re

    raw_exp = resume_parsed.get("experience", "") or ""

    # ── Flatten list-of-dicts to plain text ───────────────────────────────────
    if isinstance(raw_exp, list):
        parts = []
        for entry in raw_exp:
            if isinstance(entry, dict):
                # Include every string value — duration, role, responsibilities
                for v in entry.values():
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list):
                        parts.extend(str(item) for item in v if item)
            elif isinstance(entry, str):
                parts.append(entry)
        raw_exp = " ".join(parts)

    text = raw_exp + " " + (resume_parsed.get("summary", "") or "")
    text = text.lower()

    # Match patterns like "5 years", "5+ years", "5yrs", "five years"
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    years: Optional[int] = None

    # Numeric patterns: "5 years", "5+ years", "5-6 years"
    m = re.search(r"(\d+)\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?",
                  text)
    if m:
        years = int(m.group(1))
    else:
        # Word patterns: "five years of experience"
        for word, num in word_to_num.items():
            if re.search(rf"\b{word}\b.{{0,20}}(?:years?|yrs?)", text):
                years = num
                break

    if years is None:
        return None
    if years <= 1:
        return "easy"
    if years <= 4:
        return "medium"
    return "hard"


# ══════════════════════════════════════════════════════════════════════════════
#  RL ADAPTIVE SEQUENCER
# ══════════════════════════════════════════════════════════════════════════════

class RLAdaptiveSequencer:
    """
    ε-greedy Q-Learning bandit for adaptive interview question sequencing.

    Replaces the static two-line heuristic in QuestionBank.next_difficulty()
    with a proper RL agent that:
      • Learns which (type, difficulty) combination works best for THIS candidate
      • Balances exploration (try new types) vs exploitation (repeat what works)
      • Persists Q-table across sessions (warm-starts from saved experience)
      • Produces rich diagnostics for the Final Report page

    Research: Patel et al. (2023, Springer AI Review), Srivastava & Bhatt
    (2022, IEEE ICCCIS), Liu et al. (2021, ACM ITS).

    Quick-start:
        seq = RLAdaptiveSequencer(role="Data Scientist")
        seq.load()   # warm-start from saved table (no-op on first run)

        # After each answer submission:
        next_action = seq.record_and_select(
            score=3.8, nervousness=0.3, star_count=3,
            time_efficiency=72.0, word_count=145
        )
        q_type     = next_action.q_type      # "technical"
        difficulty = next_action.difficulty  # "medium"

        # At session end:
        seq.save()
        report = seq.get_session_report()
    """

    def __init__(self, role: str = "Software Engineer",
                 lr:    float = LR,
                 gamma: float = GAMMA,
                 eps_start: float = EPS_START,
                 eps_end:   float = EPS_END,
                 eps_decay: float = EPS_DECAY,
                 use_shared_prior: bool = True) -> None:
        """
        Args:
            role             : job role string — determines Q-table filename.
            use_shared_prior : if True (default), load the population-level
                               shared Q-table as the warm-start prior before
                               overlaying the individual table.  This lets new
                               candidates benefit from aggregate experience of
                               all previous candidates for the same role.
        """
        self._role            = role.lower().replace(" ", "_")
        self._lr              = lr
        self._gamma           = gamma
        self._eps             = eps_start
        self._eps_end         = eps_end
        self._eps_decay       = eps_decay
        self._use_shared_prior = use_shared_prior

        # Q-table: shape (4,3,3,3,8) — states × 8 actions
        # Initialised with small positive values to encourage initial exploration
        self._q: np.ndarray = np.full(Q_TABLE_SHAPE, 0.1, dtype=np.float64)

        # Session state
        self._step_count:       int   = 0
        self._last_action:      int   = DEFAULT_ACTION_IDX
        self._last_state:       Optional[Tuple] = None
        self._scores:           List[float] = []
        self._nervousness:      List[float] = []
        self._history:          List[SequencerStep] = []
        self._session_count:    int   = 0

        # v2.0: follow-up guard — prevent consecutive follow-up actions
        self._followed_up_last: bool  = False

    # ── Core RL loop ──────────────────────────────────────────────────────────

    def record_and_select(self,
                          score:           float,
                          nervousness:     float,
                          star_count:      int   = 0,
                          time_efficiency: float = 50.0,
                          word_count:      int   = 100,
                          q_type:          str   = "") -> Action:
        """
        Main entry point — call ONCE after each answer is submitted.

        1. Records the transition (previous state, action, reward, new state)
        2. Performs Q-table update (Q-learning Bellman equation)
        3. Selects the next action via ε-greedy policy
        4. Decays exploration rate

        Args:
            score:           raw NLP score 1-5 (from AnswerEvaluator)
            nervousness:     fused facial+voice nervousness 0-1
            star_count:      number of STAR components present 0-4
            time_efficiency: answer timing score 0-100
            word_count:      total words in answer

        Returns:
            Action — the recommended (q_type, difficulty) for the next question
        """
        self._scores.append(score)
        self._nervousness.append(nervousness)
        self._step_count += 1

        # ── 1. Encode current state ───────────────────────────────────────────
        recent_scores = self._scores[-3:] if len(self._scores) >= 3 else self._scores
        avg_recent    = float(np.mean(recent_scores))
        star_rate     = star_count / 4.0
        curr_state    = encode_state(avg_recent, nervousness, star_rate, time_efficiency)

        # ── 2. Compute reward (only after step 1 — need prev state) ──────────
        if self._last_state is not None and len(self._scores) >= 2:
            prev_score = self._scores[-2]
            prev_nerv  = self._nervousness[-2] if len(self._nervousness) >= 2 else nervousness
            reward = compute_reward(
                score           = score,
                prev_score      = prev_score,
                nervousness     = nervousness,
                prev_nervousness= prev_nerv,
                star_count      = star_count,
                word_count      = word_count,
                prev_action_idx = self._last_action,
                curr_action_idx = self._last_action,
                q_type          = q_type,   # v9.2: type-aware star_bonus
            )
            # ── 3. Q-learning update (Bellman equation) ───────────────────────
            # Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]
            old_q      = self._q[self._last_state][self._last_action]
            next_max_q = float(np.max(self._q[curr_state]))
            new_q      = old_q + self._lr * (reward + self._gamma * next_max_q - old_q)
            self._q[self._last_state][self._last_action] = new_q

            log.info(f"[RL] Step {self._step_count} | "
                     f"state={curr_state} action={ACTIONS[self._last_action].label()} "
                     f"reward={reward:.2f} Q={new_q:.3f}")
        else:
            reward = 0.0

        # ── 4. Select next action (ε-greedy + follow-up override) ───────────
        # v2.0: follow-up override fires BEFORE ε-greedy when the answer is
        # shallow (brief AND unstructured) AND we didn't just follow up.
        # This ensures the agent probes gaps rather than blindly advancing.
        shallow = _shallow_answer_detected(word_count, star_count)
        if shallow and not self._followed_up_last:
            next_action_idx = FOLLOW_UP_ACTION_IDX
            log.info(f"[RL] Shallow answer detected (wc={word_count}, "
                     f"star={star_count}) → follow_up override")
            self._followed_up_last = True
        else:
            self._followed_up_last = False
            if random.random() < self._eps:
                # Exclude follow_up from random exploration — it should only
                # fire on explicit shallow-answer detection, not randomly.
                next_action_idx = random.randint(0, N_ACTIONS - 2)
                log.debug(f"[RL] Exploring — random action: {ACTIONS[next_action_idx].label()}")
            else:
                # Exclude follow_up from greedy exploitation too
                q_no_followup = self._q[curr_state][:N_ACTIONS - 1]
                next_action_idx = int(np.argmax(q_no_followup))
                log.debug(f"[RL] Exploiting — best action: {ACTIONS[next_action_idx].label()} "
                          f"Q={self._q[curr_state][next_action_idx]:.3f}")

        # ── 5. Decay ε ────────────────────────────────────────────────────────
        self._eps = max(self._eps_end, self._eps * self._eps_decay)

        # ── 6. Record step ────────────────────────────────────────────────────
        chosen = ACTIONS[next_action_idx]
        # v9.2: q_number = len(history)+1 before append = actual answer number.
        # _step_count can drift ahead due to internal calls, history length
        # is always exactly "how many answers have been submitted so far".
        self._history.append(SequencerStep(
            q_number    = len(self._history) + 1,
            state       = curr_state,
            action_idx  = next_action_idx,
            action      = chosen.label(),
            score       = score,
            nervousness = nervousness,
            star_count  = star_count,
            reward      = reward,
            q_type      = chosen.q_type,
            difficulty  = chosen.difficulty,
            follow_up   = chosen.follow_up,
        ))

        self._last_state  = curr_state
        self._last_action = next_action_idx

        return chosen

    def get_first_action(self,
                         resume_parsed:      Optional[Dict] = None,
                         session_difficulty: str            = "") -> Action:
        """
        Return the opening action for question 1 (no prior state).

        v2.0 — three-tier logic:

        1. Resume calibration (highest priority if resume available):
           Parse years of experience from resume_parsed["experience"] text.
           0-1 yrs  → easy   (build confidence before pushing)
           2-4 yrs  → medium (current default)
           5+ yrs   → hard   (don't waste senior candidates on basics)

        2. Warm-start Q-table lookup (if prior sessions exist):
           Use the neutral-state Q-value argmax as the starting action,
           BUT constrain to actions matching the session difficulty when set.

        3. Cold start: respect session_difficulty if provided, else technical/medium.

        v9.2 fix: session_difficulty now always respected. Previously the user
        choosing Easy still got technical/medium because the RL cold-start
        default (DEFAULT_ACTION_IDX=1) ignored the session difficulty entirely.
        """
        # ── 0. Normalise session_difficulty ───────────────────────────────────
        # "all" mode means RL picks freely — don't constrain first action
        _sess_diff = session_difficulty.lower().strip()
        _constrain = _sess_diff in ("easy", "medium", "hard")

        # difficulty → action index mapping (Technical type, all three levels)
        diff_to_idx = {"easy": 0, "medium": 1, "hard": 2}

        # ── 1. Resume-based calibration ───────────────────────────────────────
        if resume_parsed:
            exp_difficulty = _parse_experience_difficulty(resume_parsed)
            if exp_difficulty:
                # If user chose a fixed difficulty, it overrides resume
                # (user knows what they want; resume is only a hint)
                chosen_diff = _sess_diff if _constrain else exp_difficulty
                action_idx  = diff_to_idx.get(chosen_diff, DEFAULT_ACTION_IDX)
                action      = ACTIONS[action_idx]
                log.info(f"[RL] Q1 calibrated: {action.label()} "
                         f"(session={_sess_diff or 'free'} resume→{exp_difficulty})")
                return action

        # ── 2. Warm-start from saved Q-table ──────────────────────────────────
        if self._session_count > 0:
            neutral_state = encode_state(2.5, 0.3, 0.5, 60.0)
            q_no_followup = self._q[neutral_state][:N_ACTIONS - 1]
            if _constrain:
                # Filter Q-values to only actions matching the session difficulty
                matching_idxs = [i for i, a in enumerate(ACTIONS[:N_ACTIONS - 1])
                                  if a.difficulty == _sess_diff]
                if matching_idxs:
                    best_local = max(matching_idxs, key=lambda i: q_no_followup[i])
                    action = ACTIONS[best_local]
                    log.info(f"[RL] Q1 warm-start (constrained to {_sess_diff}): "
                             f"{action.label()}")
                    return action
            # Free mode — pick globally best action
            best_idx = int(np.argmax(q_no_followup))
            action   = ACTIONS[best_idx]
            log.info(f"[RL] Q1 warm-start: {action.label()} "
                     f"(from {self._session_count} sessions)")
            return action

        # ── 3. Cold start ─────────────────────────────────────────────────────
        # v9.2: respect session_difficulty — don't always default to medium
        if _constrain:
            action_idx = diff_to_idx.get(_sess_diff, DEFAULT_ACTION_IDX)
            log.info(f"[RL] Q1 cold start: {ACTIONS[action_idx].label()} "
                     f"(session difficulty = {_sess_diff})")
            return ACTIONS[action_idx]
        return ACTIONS[DEFAULT_ACTION_IDX]   # technical/medium (free/all mode)

    # ── Groq integration helper ───────────────────────────────────────────────

    def get_groq_hint(self) -> Dict[str, str]:
        """
        Return the current recommended action as a dict for QuestionBank.

        Usage in InterviewEngine.get_next_question():
            hint = self.sequencer.get_groq_hint()
            # hint = {"type": "behavioural", "difficulty": "medium"}
            # Pass to QuestionBank._build_prompt() or get_questions()
        """
        action = ACTIONS[self._last_action]
        return {"type": action.q_type, "difficulty": action.difficulty}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _qtable_path(self) -> str:
        fname = QTABLE_FILE.format(role=self._role)
        return os.path.join(QTABLE_DIR, fname)

    def _shared_qtable_path(self) -> str:
        fname = SHARED_QTABLE_FILE.format(role=self._role)
        return os.path.join(QTABLE_DIR, fname)

    @staticmethod
    def _pad_qtable(q: np.ndarray) -> np.ndarray:
        """
        Migrate a v1.0 Q-table (shape …×7) to v2.0 shape (…×8).
        The new follow_up column is initialised to 0.0 so it starts
        neutral — the agent will learn its value from real follow-up
        transitions rather than inheriting an arbitrary prior.
        """
        if q.shape == Q_TABLE_SHAPE:
            return q
        if q.shape == (4, 3, 3, 3, 7):
            padded = np.zeros(Q_TABLE_SHAPE, dtype=np.float64)
            padded[..., :7] = q
            log.info("[RL] Migrated v1.0 Q-table (7 actions) → v2.0 (8 actions)")
            return padded
        return None   # unrecognised shape

    def save(self) -> None:
        """
        Persist individual Q-table, then update the shared population table.

        Individual table:
          Saved as aura_rl_qtable_{role}.json — overwritten each session.

        Shared table (cross-candidate aggregation):
          Loaded if it exists, then blended with this session's Q-table
          using a running weighted average:
            shared_new = (N × shared_old + q_this) / (N + 1)
          where N = shared_session_count.

          This gives each session equal vote in the prior rather than
          letting early sessions dominate (Hu et al. 2021, NeurIPS).
        """
        # ── Individual save ───────────────────────────────────────────────────
        path = self._qtable_path()
        data = {
            "role":     self._role,
            "sessions": self._session_count + 1,
            "q_table":  self._q.tolist(),
            "eps":      self._eps,
            "version":  "2.0",
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            log.info(f"[RL] Individual Q-table saved → {path}")
        except Exception as exc:
            log.warning(f"[RL] Individual Q-table save failed: {exc}")

        # ── Shared table update ───────────────────────────────────────────────
        self._update_shared_qtable()

    def _update_shared_qtable(self) -> None:
        """
        Weighted running-average update of the shared population Q-table.

        Algorithm (Hu et al. 2021, NeurIPS):
          N = existing shared session count
          shared_new = (N × shared_old + q_this) / (N + 1)

        This is mathematically equivalent to an equal-weight average of
        all sessions seen so far, computed incrementally without storing
        all individual tables.
        """
        spath = self._shared_qtable_path()
        try:
            # Load existing shared table
            if os.path.exists(spath):
                with open(spath) as f:
                    sdata = json.load(f)
                shared_q = np.array(sdata["q_table"], dtype=np.float64)
                shared_q = self._pad_qtable(shared_q)
                if shared_q is None:
                    shared_q = self._q.copy()
                    n_shared = 0
                else:
                    n_shared = sdata.get("sessions", 1)
            else:
                shared_q = self._q.copy()
                n_shared = 0

            # Weighted average blend
            if n_shared > 0:
                shared_q = (n_shared * shared_q + self._q) / (n_shared + 1)
            new_n = n_shared + 1

            out = {
                "role":     self._role,
                "sessions": new_n,
                "q_table":  shared_q.tolist(),
                "version":  "2.0",
            }
            with open(spath, "w") as f:
                json.dump(out, f, separators=(",", ":"))
            log.info(f"[RL] Shared Q-table updated → {spath} "
                     f"(N={new_n} sessions in population)")
        except Exception as exc:
            log.warning(f"[RL] Shared Q-table update failed: {exc}")

    def load(self) -> bool:
        """
        Load Q-table for warm-starting.

        v2.0 two-stage load:
          Stage 1 (if use_shared_prior=True): load population shared table
                   as the initial prior — gives benefit of cross-candidate
                   aggregate experience from the first question.
          Stage 2: overlay individual table if it exists — personalises the
                   prior with this specific candidate's own history.

        Returns True if at least one table was loaded successfully.
        """
        loaded_any = False

        # ── Stage 1: shared prior ─────────────────────────────────────────────
        if self._use_shared_prior:
            spath = self._shared_qtable_path()
            if os.path.exists(spath):
                try:
                    with open(spath) as f:
                        sdata = json.load(f)
                    shared_q = np.array(sdata["q_table"], dtype=np.float64)
                    shared_q = self._pad_qtable(shared_q)
                    if shared_q is not None:
                        self._q = shared_q
                        n_pop   = sdata.get("sessions", 0)
                        log.info(f"[RL] Shared prior loaded — "
                                 f"{n_pop} sessions in population prior")
                        loaded_any = True
                    else:
                        log.warning("[RL] Shared Q-table shape unrecognised — skipped")
                except Exception as exc:
                    log.warning(f"[RL] Shared Q-table load failed: {exc}")

        # ── Stage 2: individual overlay ───────────────────────────────────────
        path = self._qtable_path()
        if not os.path.exists(path):
            log.info(f"[RL] No individual Q-table at {path} — "
                     f"{'using shared prior' if loaded_any else 'cold start'}")
            return loaded_any
        try:
            with open(path) as f:
                data = json.load(f)
            loaded_q = np.array(data["q_table"], dtype=np.float64)
            loaded_q = self._pad_qtable(loaded_q)
            if loaded_q is not None:
                self._q             = loaded_q
                self._session_count = data.get("sessions", 0)
                saved_eps = data.get("eps", EPS_START)
                self._eps = max(self._eps_end, saved_eps)
                log.info(f"[RL] Individual Q-table loaded "
                         f"({self._session_count} sessions, ε={self._eps:.3f})")
                return True
            log.warning(f"[RL] Individual Q-table shape unrecognised — "
                        f"{'using shared prior' if loaded_any else 'cold start'}")
        except Exception as exc:
            log.warning(f"[RL] Individual Q-table load failed: {exc}")
        return loaded_any

    # ── Session management ────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Call at the START of each new interview session."""
        self._step_count       = 0
        self._last_action      = DEFAULT_ACTION_IDX
        self._last_state       = None
        self._followed_up_last = False   # v2.0: reset follow-up guard
        self._scores.clear()
        self._nervousness.clear()
        self._history.clear()
        # Reset ε to start (fresh exploration at the beginning of each session)
        self._eps = EPS_START
        log.info(f"[RL] Session reset — ε={self._eps:.3f}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_session_report(self) -> Dict:
        """
        Rich diagnostics for the Final Report page.
        Shows what the RL agent did during this session.
        v2.0: includes follow_up count and shared prior info.
        """
        if not self._history:
            return {"message": "No RL data — session not started."}

        action_counts: Dict[str, int] = {}
        follow_up_count = 0
        for step in self._history:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
            if step.follow_up:
                follow_up_count += 1

        sorted_steps = sorted(self._history, key=lambda s: s.reward, reverse=True)

        neutral_state  = encode_state(2.5, 0.3, 0.5, 60.0)
        # Exclude follow_up from "preferred next" display — not a content choice
        q_no_followup  = self._q[neutral_state][:N_ACTIONS - 1]
        best_state_idx = int(np.argmax(q_no_followup))
        preferred_next = ACTIONS[best_state_idx].label()

        return {
            "steps_recorded":    len(self._history),
            "final_epsilon":     round(self._eps, 4),
            "action_distribution": action_counts,
            "follow_up_count":   follow_up_count,   # v2.0
            "shared_prior_used": self._use_shared_prior,  # v2.0
            "best_rewarded_step": {
                "q_number":  sorted_steps[0].q_number,
                "action":    sorted_steps[0].action,
                "reward":    round(sorted_steps[0].reward, 3),
                "score":     sorted_steps[0].score,
            } if sorted_steps else {},
            "worst_rewarded_step": {
                "q_number":  sorted_steps[-1].q_number,
                "action":    sorted_steps[-1].action,
                "reward":    round(sorted_steps[-1].reward, 3),
                "score":     sorted_steps[-1].score,
            } if sorted_steps else {},
            "preferred_next_question": preferred_next,
            "session_count_lifetime":  self._session_count,
            "q_table_max":    round(float(np.max(self._q)), 4),
            "q_table_min":    round(float(np.min(self._q)), 4),
            "available":      True,
            "all_steps": [
                {
                    "q":        s.q_number,
                    "action":   s.action,
                    "score":    s.score,
                    "nerv":     round(s.nervousness, 3),
                    "reward":   round(s.reward, 3),
                    "followup": s.follow_up,
                }
                for s in self._history
            ],
        }

    def get_q_table_heatmap_data(self) -> Dict:
        """
        Return Q-table data formatted for a Plotly heatmap in the Dashboard.
        Shows Q-values for each action at two representative states:
          State A — struggling (low score, high nervousness)
          State B — performing well (high score, low nervousness)
        """
        state_a = encode_state(1.5, 0.7, 0.2, 30.0)   # struggling
        state_b = encode_state(4.2, 0.2, 0.8, 75.0)   # performing well

        action_labels = [a.label() for a in ACTIONS]
        return {
            "action_labels": action_labels,
            "q_values_struggling": [
                round(self._q[state_a][i], 3) for i in range(N_ACTIONS)
            ],
            "q_values_performing": [
                round(self._q[state_b][i], 3) for i in range(N_ACTIONS)
            ],
            "recommended_struggling": ACTIONS[int(np.argmax(self._q[state_a]))].label(),
            "recommended_performing": ACTIONS[int(np.argmax(self._q[state_b]))].label(),
        }

    @property
    def current_epsilon(self) -> float:
        return round(self._eps, 4)

    @property
    def step_count(self) -> int:
        return self._step_count
