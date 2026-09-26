"""
dialogic_feedback.py — Aura AI | Dialogic Feedback Engine
==========================================================
Implements ACM CSCW 2025 (Conversate) two-way feedback loop:
  - Candidate receives score → can challenge or clarify
  - System re-evaluates with the clarification, adjusts score if warranted
  - Keeps a bounded turn history (max 3 exchanges per question)

RESEARCH BASIS
--------------
Daryanto et al. (ACM CSCW 2025 — Conversate):
  One-way score delivery is a "core design failure". Dialogic feedback
  (back-and-forth where users can dispute the AI's judgment) promoted:
    - Personalized, continuous learning
    - Reduced feelings of being judged
    - Willingness to express genuine disagreement
  Key design principle: candidate must be able to EXPLAIN what they
  intended, and the system must re-evaluate with that context.

HOW IT FITS INTO analyzer.py
------------------------------
1. analyzer.py produces the initial `analysis_result` dict as before.
2. DialogicFeedbackEngine.open_dialogue(analysis_result, transcript, question)
   is called once — returns an initial coaching prompt.
3. On each candidate reply, call .advance(user_message) → AI response + 
   possibly a `score_revision` field if the score should be updated.
4. DialogicFeedbackEngine.close() returns the final agreed score + 
   full conversation log for the session report.

INTEGRATION POINTS
------------------
In main.py (FastAPI):
  POST /dialogue/open    → open_dialogue()
  POST /dialogue/turn    → advance()
  POST /dialogue/close   → close()

In App.jsx (React):
  After ScorePanel renders, show <DialoguePanel dialogueId={id}/>
  which calls the above endpoints.

GRACEFUL DEGRADATION
--------------------
If Groq API is unavailable, falls back to rule-based coaching prompts
derived purely from the score breakdown. No crashes, slightly less
personalised responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from groq import AsyncGroq
from dispute_corpus import dispute_corpus

logger = logging.getLogger(__name__)

# ── Optional SBERT (same lazy-load pattern as sas_scorer.py) ─────────────────
_sbert_model = None
_SBERT_OK    = False

def _load_sbert() -> bool:
    global _sbert_model, _SBERT_OK
    if _SBERT_OK:
        return True
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _sbert_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        _SBERT_OK    = True
    except Exception:
        _SBERT_OK = False
    return _SBERT_OK

# Stop-words for keyword novelty check
_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","i","we","you","he",
    "she","they","it","this","that","my","our","your","his","her","its",
    "so","if","not","as","up","do","did","will","would","could","should",
    "also","just","more","then","when","which","who","what","how","very",
    "really","like","basically","actually","just","yeah","okay","well",
}

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_TURNS       = 3          # Max candidate↔AI exchanges per question
_SCORE_DELTA_CAP = 0.20       # Max total score change per turn (0–1 scale → ×5 for 0–5)
_LLM_MODEL       = "llama-3.3-70b-versatile"

# ── Manipulation resistance constants (v2.0) ──────────────────────────────────

# LAYER 1 — Semantic Novelty Gate
# Minimum fraction of NEW content words the clarification must introduce.
# Below this → clarification is mostly a restatement → upward delta hard-capped.
_MIN_NOVELTY            = 0.20   # 20% of clarification words must be new vs original
_NOVELTY_SIMILARITY_CAP = 0.82   # SBERT cosine above this = "same content, restated"
_LOW_NOVELTY_DELTA_CAP  = 0.05   # Max upward shift (0–1 scale) when novelty is low

# LAYER 2 — Per-dimension cap
# Prevents one weak dimension from absorbing the full global cap.
_DIM_DELTA_CAP = 0.08    # Max shift per individual scoring dimension (0–1 scale)
# Scoring dimensions the Groq prompt uses:
_SCORE_DIMS    = ("relevance", "structure", "depth", "fluency", "confidence")

# LAYER 3 — Session cumulative upward cap
# Once the candidate has gained this many points (0–5 scale) across all turns,
# further upward revisions in this session are blocked.
_SESSION_UPWARD_CAP = 0.40   # 0–1 scale; ×5 = 2.0 raw points on the 0–5 scale

# ── Interrupt & Recover mechanic constants ────────────────────────────────────
# Research: Voss & Raz (2016, Never Split the Difference) — unexpected pivots
# in high-stakes conversations reveal composure and adaptability under pressure.
# Sian Beilock (2010, Choke) — time-pressured interrupts expose working-memory
# capacity; candidates who recover quickly score higher on real hiring panels.

_INTERRUPT_MIN_WORDS    = 40    # Don't interrupt before candidate has said this many words
_INTERRUPT_PROB         = 0.45  # 45% chance of generating an interrupt per answer session
_INTERRUPT_WINDOW_START = 0.30  # Interrupt can fire once candidate is 30% into their answer
_INTERRUPT_WINDOW_END   = 0.70  # …but no later than 70% through (so they can recover)

# Interrupt templates by type — Groq replaces these with context-aware versions
_INTERRUPT_TEMPLATES = {
    "evidence":   "Wait — can you give me a specific number or metric to back that up?",
    "outcome":    "Skip ahead — what was the actual outcome for the business?",
    "clarify":    "Hold on — when exactly did this happen, and what was your specific role?",
    "challenge":  "I'm going to push back on that — what would you say to someone who disagrees?",
    "compress":   "I need to stop you there — can you give me the one-sentence version of your key point?",
    "pivot":      "Let's shift — how does that experience apply to a fully remote team?",
}

# Groq client (reuses env var GROQ_API_KEY — same as analyzer.py)
import os
_groq_client: Optional[AsyncGroq] = None

def _get_groq() -> Optional[AsyncGroq]:
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY", "")
        if key:
            _groq_client = AsyncGroq(api_key=key)
    return _groq_client


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DialogueTurn:
    role: str          # "assistant" | "candidate"
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class DialogueSession:
    id: str
    question: str
    question_type: str
    original_transcript: str
    initial_score: float           # 0–5 knowledge score from analyzer
    current_score: float           # may change after clarifications
    score_breakdown: Dict          # scores dict from analyze() result
    ai_evaluation: Dict            # ai_evaluation dict from analyze() result
    turns: List[DialogueTurn] = field(default_factory=list)
    turn_count: int = 0
    closed: bool = False
    score_revised: bool = False
    final_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    # ── Dispute corpus fields ────────────────────────────────────────────────
    rubric_reasoning: str = ""         # RubricAgent CoT from agent_scores.rubric.reasoning
    last_candidate_message: str = ""   # most recent candidate turn (used for corpus on close)
    # ── Manipulation resistance tracking (v2.0) ──────────────────────────────
    cumulative_upward_delta: float = 0.0   # total upward score shift this session (0–1 scale)
    novelty_log: List[float] = field(default_factory=list)  # novelty score per turn

    # ── Interrupt & Recover fields ───────────────────────────────────────────────
    interrupt_fired: bool = False           # True once an interrupt has been triggered
    interrupt_type: str = ""               # which interrupt template was used
    interrupt_text: str = ""              # the actual interrupt question shown
    interrupt_response: str = ""          # candidate's recovery response
    recovery_score: float = -1.0          # 0–5; -1 means no interrupt happened
    interrupt_word_trigger: int = 0       # word count at which interrupt fired

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  COACHING PROMPT BUILDER  (rule-based fallback, no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

def _build_opening_prompt(session: DialogueSession) -> str:
    """
    Build the initial coaching prompt shown after the score.
    Derived from score breakdown so it's specific, not generic.
    Research: Conversate — specificity in feedback prevents the
    'too generic to be useful' failure mode (75% of candidates cite this).
    """
    score     = session.initial_score
    sb        = session.score_breakdown
    ai_eval   = session.ai_evaluation
    relevance = sb.get("relevance", 0)
    structure = sb.get("structure", 0)
    depth     = sb.get("depth", 3)

    # Identify the weakest dimension
    dims = {
        "relevance":  relevance,
        "structure":  structure,
        "confidence": sb.get("confidence", 0),
        "fluency":    sb.get("fluency", 3),
    }
    weakest_dim = min(dims, key=dims.get)

    # Score-calibrated opening sentence
    if score >= 4.0:
        opener = f"You scored **{score:.1f}/5** — that's a strong answer overall."
    elif score >= 3.0:
        opener = f"You scored **{score:.1f}/5** — solid, with room to sharpen a few areas."
    elif score >= 2.0:
        opener = f"You scored **{score:.1f}/5** — the foundation is there, but some key elements were thin."
    else:
        opener = f"You scored **{score:.1f}/5** — let's work through what happened."

    # Dimension-specific probe
    probes = {
        "relevance":  "Your answer may not have addressed what the question was really asking. What were you trying to convey?",
        "structure":  "The answer lacked a clear Situation → Action → Result flow. Walk me through the specific situation you had in mind.",
        "confidence": "Your language included quite a few hedges ('I think', 'maybe', 'I guess'). Were you genuinely uncertain, or just being modest?",
        "fluency":    "There were some pacing issues detected. Was this nerves, or was the question unclear?",
    }
    probe = probes[weakest_dim]

    # Coaching tip from HR evaluation
    tip = ai_eval.get("coaching_tip", "")
    tip_line = f"\n\n💡 **Quick tip:** {tip}" if tip else ""

    return (
        f"{opener}\n\n"
        f"**{probe}**\n\n"
        f"Tell me what you were trying to say — I'll factor that in.{tip_line}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MANIPULATION RESISTANCE HELPERS  (v2.0)
# ══════════════════════════════════════════════════════════════════════════════

def _content_words(text: str) -> set:
    """Lowercase alphabetic tokens longer than 2 chars, minus stop-words."""
    import re
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP}


def _cosine_simple(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity — no numpy dependency."""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_novelty(clarification: str, original_answer: str) -> Tuple[float, str]:
    """
    Measure how much NEW information the clarification contains relative to
    the original answer.

    Returns
    -------
    (novelty_score: float [0–1],  method: str)

    novelty_score = 1.0  → completely new content
    novelty_score = 0.0  → pure restatement of the original

    Two-tier (mirrors sas_scorer.py pattern):
      Tier 1 — SBERT cosine similarity:
          novelty = 1 − cosine(clarification, original)
          If cosine > _NOVELTY_SIMILARITY_CAP the texts are semantically
          near-identical — flag as low novelty regardless of word choice.

      Tier 2 — Keyword novelty (fallback):
          new_words  = content_words(clarification) − content_words(original)
          novelty    = |new_words| / max(|content_words(clarification)|, 1)
          Measures what fraction of the clarification's vocabulary is absent
          from the original answer.

    Research: Johnson & Sinatra (2013, Computers & Education) — automated
    feedback gaming requires restating existing content confidently; genuine
    corrections introduce domain-specific vocabulary not present in the
    original answer (vocabulary novelty r=0.61 with human-rated genuineness).
    """
    if not clarification.strip() or not original_answer.strip():
        return 0.0, "empty"

    # ── Tier 1: SBERT ─────────────────────────────────────────────────────────
    if _load_sbert() and _sbert_model is not None:
        try:
            emb_c = _sbert_model.encode(clarification[:1000], normalize_embeddings=True).tolist()
            emb_o = _sbert_model.encode(original_answer[:1000], normalize_embeddings=True).tolist()
            cosine = _cosine_simple(emb_c, emb_o)
            # High cosine = semantically same = low novelty
            if cosine > _NOVELTY_SIMILARITY_CAP:
                return round(1.0 - cosine, 3), "semantic_low"
            return round(1.0 - cosine, 3), "semantic"
        except Exception:
            pass  # fall through to keyword

    # ── Tier 2: Keyword novelty ────────────────────────────────────────────────
    clar_words = _content_words(clarification)
    orig_words = _content_words(original_answer)
    if not clar_words:
        return 0.0, "keyword_empty"
    new_words   = clar_words - orig_words
    novelty     = len(new_words) / len(clar_words)
    return round(novelty, 3), "keyword"


def _apply_novelty_gate(
    proposed_delta: float,        # raw delta from LLM (0–1 scale, can be negative)
    novelty_score: float,
    method: str,
) -> Tuple[float, bool]:
    """
    Apply the novelty gate to an upward delta.

    Downward revisions are NEVER gated — they represent genuine weaknesses
    the candidate revealed through poor clarification.

    Returns (gated_delta, was_capped: bool).
    """
    if proposed_delta <= 0:
        return proposed_delta, False   # downward — never blocked

    # If SBERT flagged near-identical content, cap hard regardless of fraction
    if method == "semantic_low":
        gated = min(proposed_delta, _LOW_NOVELTY_DELTA_CAP)
        return gated, (gated < proposed_delta)

    # If novelty is below minimum threshold, apply low-novelty cap
    if novelty_score < _MIN_NOVELTY:
        gated = min(proposed_delta, _LOW_NOVELTY_DELTA_CAP)
        return gated, (gated < proposed_delta)

    return proposed_delta, False


def _apply_dim_caps(dim_deltas: Dict[str, float]) -> Dict[str, float]:
    """
    Clamp each per-dimension delta to ±_DIM_DELTA_CAP.
    Returns a new dict with capped values.

    Research: Mislevy et al. (2003, ETS) evidence-centred design — each
    scoring dimension should be independently constrained so strong
    performance on one dimension cannot inflate another beyond its evidence.
    """
    return {
        dim: max(-_DIM_DELTA_CAP, min(_DIM_DELTA_CAP, delta))
        for dim, delta in dim_deltas.items()
    }


def _apply_session_cap(
    proposed_delta: float,         # already novelty-gated delta (0–1 scale)
    cumulative_upward: float,      # total upward gain so far this session
) -> Tuple[float, bool]:
    """
    Enforce the session-level upward cap.
    Returns (final_delta, session_cap_hit: bool).
    """
    if proposed_delta <= 0:
        return proposed_delta, False

    headroom = max(0.0, _SESSION_UPWARD_CAP - cumulative_upward)
    if headroom <= 0:
        return 0.0, True  # session cap exhausted

    final = min(proposed_delta, headroom)
    return final, (final < proposed_delta)


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE REVISION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

async def _groq_score_revision(
    session: DialogueSession,
    clarification: str,
) -> Tuple[float, str, Dict]:
    """
    Ask Groq whether the candidate's clarification warrants a score revision.
    Applies all three manipulation-resistance layers before returning.

    Returns
    -------
    (revised_score: float,  reasoning: str,  meta: Dict)

    meta keys:
      novelty_score      : float  — how novel the clarification was vs original
      novelty_method     : str    — "semantic" | "semantic_low" | "keyword"
      novelty_gated      : bool   — True if novelty cap reduced the delta
      session_cap_hit    : bool   — True if session upward cap was exhausted
      dim_deltas_raw     : dict   — per-dimension deltas before caps
      dim_deltas_capped  : dict   — per-dimension deltas after _DIM_DELTA_CAP
      final_delta        : float  — actual applied delta (0–1 scale)
    """
    groq = _get_groq()

    # ── LAYER 1: Semantic novelty check ──────────────────────────────────────
    # Run BEFORE calling Groq so we can warn the candidate and skip the
    # Groq call entirely when novelty is near-zero (saves latency + tokens).
    novelty_score, novelty_method = _semantic_novelty(
        clarification, session.original_transcript
    )
    session.novelty_log.append(novelty_score)

    # If the session upward cap is already exhausted, skip Groq immediately
    if session.cumulative_upward_delta >= _SESSION_UPWARD_CAP:
        logger.info(
            f"[DialogicFeedback] {session.id[:8]} session upward cap exhausted "
            f"(cumulative={session.cumulative_upward_delta:.3f})"
        )
        return session.current_score, "session_cap_exhausted", {
            "novelty_score":     novelty_score,
            "novelty_method":    novelty_method,
            "novelty_gated":     False,
            "session_cap_hit":   True,
            "dim_deltas_raw":    {},
            "dim_deltas_capped": {},
            "final_delta":       0.0,
        }

    if groq is None:
        return session.current_score, "groq_unavailable", {
            "novelty_score":     novelty_score,
            "novelty_method":    novelty_method,
            "novelty_gated":     False,
            "session_cap_hit":   False,
            "dim_deltas_raw":    {},
            "dim_deltas_capped": {},
            "final_delta":       0.0,
        }

    history_text = "\n".join(
        f"{'AI' if t.role == 'assistant' else 'CANDIDATE'}: {t.content}"
        for t in session.turns
    )

    # ── Groq prompt: asks for per-dimension deltas, not a monolithic score ────
    prompt = f"""You are an expert HR interviewer re-evaluating a candidate's answer.

ORIGINAL QUESTION: {session.question}
QUESTION TYPE: {session.question_type}

ORIGINAL ANSWER (transcript):
{session.original_transcript}

DIALOGUE SO FAR:
{history_text}

CANDIDATE'S LATEST CLARIFICATION:
{clarification}

CURRENT SCORE: {session.current_score:.2f}/5.0
NOVELTY OF CLARIFICATION: {novelty_score:.2f}/1.0
  (0 = pure restatement of original; 1 = entirely new information)

TASK:
Evaluate whether the clarification reveals genuinely NEW information that
changes your assessment. Consider:
  - Does the candidate reveal knowledge they HAVE but expressed poorly?
  - Or are they just restating the same points with more confidence?
  - Low novelty score ({novelty_score:.2f}) means the clarification is mostly
    a restatement — weight this heavily when deciding on upward revisions.

For each scoring dimension, provide a delta (change) in the range [-1.0, +1.0]:
  relevance  : did the clarification show the answer was more/less relevant?
  structure  : did the clarification reveal better/worse STAR organisation?
  depth      : did the clarification add/remove substantive technical depth?
  fluency    : did the clarification suggest the original fluency score was off?
  confidence : did the clarification reveal genuine/false confidence?

Per-dimension hard cap is ±{_DIM_DELTA_CAP} — do not exceed this per dimension.
Global total score change cap is ±{_SCORE_DELTA_CAP * 5:.2f} points on 0–5 scale.
Only assign non-zero deltas where the clarification DIRECTLY provides new evidence.

Respond ONLY with valid JSON (no markdown, no preamble):
{{
  "dim_deltas": {{
    "relevance":  <float in [-{_DIM_DELTA_CAP}, +{_DIM_DELTA_CAP}]>,
    "structure":  <float in [-{_DIM_DELTA_CAP}, +{_DIM_DELTA_CAP}]>,
    "depth":      <float in [-{_DIM_DELTA_CAP}, +{_DIM_DELTA_CAP}]>,
    "fluency":    <float in [-{_DIM_DELTA_CAP}, +{_DIM_DELTA_CAP}]>,
    "confidence": <float in [-{_DIM_DELTA_CAP}, +{_DIM_DELTA_CAP}]>
  }},
  "changed": <true|false>,
  "reasoning": "<one sentence — what specific new information justified this change, or why no change>"
}}"""

    try:
        resp = await groq.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250,
        )
        raw   = resp.choices[0].message.content.strip()
        raw   = raw.replace("```json", "").replace("```", "").strip()
        data  = json.loads(raw)
        dim_deltas_raw: Dict[str, float] = {
            d: float(data.get("dim_deltas", {}).get(d, 0.0))
            for d in _SCORE_DIMS
        }
    except Exception as e:
        logger.warning(f"[DialogicFeedback] Score revision LLM failed: {e}")
        return session.current_score, "revision_failed", {
            "novelty_score":     novelty_score,
            "novelty_method":    novelty_method,
            "novelty_gated":     False,
            "session_cap_hit":   False,
            "dim_deltas_raw":    {},
            "dim_deltas_capped": {},
            "final_delta":       0.0,
        }

    # ── LAYER 2: Per-dimension cap ────────────────────────────────────────────
    dim_deltas_capped = _apply_dim_caps(dim_deltas_raw)

    # Aggregate capped dimension deltas → overall delta (0–1 scale)
    # Each dimension contributes equally (weight = 1/5).
    raw_total_delta = sum(dim_deltas_capped.values()) / len(_SCORE_DIMS)

    # Still clamp to global per-turn cap as a final backstop
    sign = 1 if raw_total_delta >= 0 else -1
    if abs(raw_total_delta) > _SCORE_DELTA_CAP:
        raw_total_delta = sign * _SCORE_DELTA_CAP

    # ── LAYER 1 applied to aggregate: novelty gate on the total delta ─────────
    gated_delta, novelty_gated = _apply_novelty_gate(
        raw_total_delta, novelty_score, novelty_method
    )

    # ── LAYER 3: Session cumulative upward cap ────────────────────────────────
    final_delta, session_cap_hit = _apply_session_cap(
        gated_delta, session.cumulative_upward_delta
    )

    # Compute revised score (0–5 scale; final_delta is on 0–1 scale × 5)
    new_score = round(
        max(0.0, min(5.0, session.current_score + final_delta * 5.0)), 2
    )

    reasoning = data.get("reasoning", "")

    # Log what happened for transparency
    if novelty_gated or session_cap_hit:
        logger.info(
            f"[DialogicFeedback] {session.id[:8]} manipulation guard triggered | "
            f"novelty={novelty_score:.2f}({novelty_method}) "
            f"novelty_gated={novelty_gated} session_cap_hit={session_cap_hit} | "
            f"raw_delta={raw_total_delta:.3f} → final_delta={final_delta:.3f}"
        )

    return new_score, reasoning, {
        "novelty_score":     novelty_score,
        "novelty_method":    novelty_method,
        "novelty_gated":     novelty_gated,
        "session_cap_hit":   session_cap_hit,
        "dim_deltas_raw":    dim_deltas_raw,
        "dim_deltas_capped": dim_deltas_capped,
        "final_delta":       round(final_delta, 4),
    }


async def _groq_dialogue_response(
    session: DialogueSession,
    user_message: str,
    score_changed: bool,
    new_score: float,
    revision_reasoning: str,
    revision_meta: Optional[Dict] = None,
) -> str:
    """
    Generate the AI's next conversational turn.
    Acknowledges score revision if it happened, continues coaching.
    If a novelty guard fired, nudges the candidate to add genuinely new info
    rather than silently ignoring their message.
    """
    groq = _get_groq()
    if groq is None:
        return _fallback_response(session, score_changed, new_score, revision_meta)

    history_msgs = [
        {"role": "assistant" if t.role == "assistant" else "user", "content": t.content}
        for t in session.turns
    ]
    history_msgs.append({"role": "user", "content": user_message})

    score_note = ""
    if score_changed:
        direction = "up" if new_score > session.current_score else "down"
        score_note = (
            f"Note: Based on the clarification, revise the score {direction} "
            f"to {new_score:.1f}/5. Reason: {revision_reasoning}. "
            f"Briefly acknowledge this revision naturally in your response."
        )

    # Surface novelty guard to the candidate honestly and constructively
    novelty_note = ""
    meta = revision_meta or {}
    if meta.get("novelty_gated") or meta.get("session_cap_hit"):
        if meta.get("session_cap_hit"):
            novelty_note = (
                "The candidate has already received the maximum score adjustment "
                "for this session. Do NOT revise the score further upward. "
                "Gently redirect them toward practising a better answer rather "
                "than continuing to dispute this one."
            )
        elif meta.get("novelty_gated"):
            novelty_note = (
                f"The candidate's clarification was largely a restatement of "
                f"their original answer (novelty={meta.get('novelty_score', 0):.2f}/1.0). "
                f"Do NOT revise the score upward. Instead, acknowledge what they said, "
                f"but gently point out that they haven't added genuinely new information "
                f"and ask them to share a SPECIFIC example or detail they haven't mentioned yet."
            )

    turns_left = _MAX_TURNS - session.turn_count
    closing_note = ""
    if turns_left <= 1:
        closing_note = (
            "This is the last exchange. Wrap up with a concrete, actionable tip "
            "the candidate can use in their next attempt at this question."
        )

    system = f"""You are Aura, an AI interview coach having a constructive dialogue with a candidate.
Your goal: help them understand specifically what made their answer strong or weak.

INTERVIEW CONTEXT:
  Question: {session.question}
  Question type: {session.question_type}
  Original score: {session.initial_score:.1f}/5
  Current score: {new_score:.1f}/5

COACHING PRINCIPLES (ACM Conversate research):
  - Be specific, never generic
  - Reduce feelings of being judged — be warm, not harsh
  - Acknowledge genuine insights the candidate brings
  - Push back constructively if the clarification doesn't actually address the weakness
  - Never repeat feedback already given in this dialogue

{score_note}
{novelty_note}
{closing_note}

Keep your response concise (2–4 sentences max). No bullet lists. Conversational tone.
End with a single, specific follow-up question OR a concrete closing tip."""

    try:
        resp = await groq.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "system", "content": system}] + history_msgs,
            temperature=0.55,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[DialogicFeedback] Dialogue response LLM failed: {e}")
        return _fallback_response(session, score_changed, new_score, revision_meta)


def _fallback_response(
    session: DialogueSession,
    score_changed: bool,
    new_score: float,
    meta: Optional[Dict] = None,
) -> str:
    """Rule-based fallback when Groq is unavailable."""
    meta = meta or {}

    # Session cap hit — redirect constructively
    if meta.get("session_cap_hit"):
        return (
            "You've made good use of the dialogue rounds. The score won't adjust "
            "further in this session, but the coaching insight stands — "
            "practise surfacing your strongest points in the first 30 seconds "
            "so you don't need to clarify after the fact."
        )

    # Low novelty — ask for genuinely new information
    if meta.get("novelty_gated"):
        return (
            "That's helpful context, but I'm not seeing much new information beyond "
            "what you said originally. Can you give me a specific example, metric, "
            "or technical detail you haven't mentioned yet? That's what would shift the score."
        )

    if score_changed:
        delta = new_score - session.current_score
        direction = "revised up" if delta > 0 else "revised down"
        return (
            f"Thanks for clarifying — I've {direction} your score to {new_score:.1f}/5. "
            f"For your next attempt, try to make that point explicit in your answer "
            f"rather than implied, as interviewers rarely ask follow-up questions."
        )
    remaining = _MAX_TURNS - session.turn_count
    if remaining > 0:
        return (
            "That context is helpful. To strengthen the answer, try stating your "
            "specific action and its measurable impact upfront. "
            "Is there any other aspect of your answer you'd like to revisit?"
        )
    return (
        "Good conversation. The key takeaway: make your core argument explicit "
        "in the first 30 seconds of your answer — don't let the interviewer infer it."
    )



# ══════════════════════════════════════════════════════════════════════════════
#  INTERRUPT & RECOVER ENGINE  (v1.0)
#  Simulates a real interviewer cutting in mid-answer to test composure and
#  STAR-structure retention under pressure.
# ══════════════════════════════════════════════════════════════════════════════

import math as _math
import random as _random

def _should_interrupt(word_count: int, interrupt_already_fired: bool) -> bool:
    """
    Decide whether to fire an interrupt right now.
    Returns True if we should interrupt.

    Rules:
      - Only one interrupt per answer session (interrupt_already_fired guard)
      - Candidate must have spoken at least _INTERRUPT_MIN_WORDS words
      - Probabilistic: _INTERRUPT_PROB chance once the window opens
    """
    if interrupt_already_fired:
        return False
    if word_count < _INTERRUPT_MIN_WORDS:
        return False
    return _random.random() < _INTERRUPT_PROB


async def _generate_interrupt(
    question: str,
    partial_transcript: str,
    question_type: str,
) -> tuple:
    """
    Generate a context-aware interrupt question via Groq.
    Falls back to a random template if Groq is unavailable.

    Returns
    -------
    (interrupt_text: str, interrupt_type: str)
    """
    groq = _get_groq()

    # Pick interrupt type based on question_type heuristic
    if question_type in ("technical", "conceptual"):
        preferred_types = ["evidence", "clarify", "compress", "challenge"]
    elif question_type == "behavioral":
        preferred_types = ["outcome", "clarify", "pivot", "compress"]
    else:  # hr / situational
        preferred_types = ["challenge", "outcome", "pivot", "clarify"]

    chosen_type = _random.choice(preferred_types)
    fallback_text = _INTERRUPT_TEMPLATES[chosen_type]

    if groq is None:
        return fallback_text, chosen_type

    prompt = f"""You are a sharp, professional interviewer who has just interrupted a candidate mid-answer.

INTERVIEW QUESTION: {question}
QUESTION TYPE: {question_type}
CANDIDATE'S PARTIAL ANSWER SO FAR:
{partial_transcript}

INTERRUPT TYPE REQUESTED: {chosen_type}
({chosen_type} description: {fallback_text})

Generate ONE concise, realistic interviewer interrupt in the style of the type above.
The interrupt must:
- Be 1–2 sentences only
- Sound natural and conversational, not robotic
- Be directly grounded in what the candidate just said
- Create productive pressure — push for specificity, evidence, or pivoting

Respond with ONLY the interrupt text, no JSON, no preamble."""

    try:
        resp = await groq.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip().strip('"')
        return text, chosen_type
    except Exception as e:
        logger.warning(f"[Interrupt] Groq interrupt generation failed: {e}")
        return fallback_text, chosen_type


async def _score_recovery(
    question: str,
    interrupt_text: str,
    interrupt_type: str,
    original_partial: str,
    recovery_response: str,
    question_type: str,
) -> tuple:
    """
    Score how well the candidate recovered from the interrupt.

    Scoring rubric (0–5):
      5 — Pivots cleanly, directly addresses interrupt, maintains STAR thread
      4 — Addresses interrupt well, minor loss of structure
      3 — Partially answers the interrupt, notable structure drop
      2 — Confused or defensive response, largely ignores interrupt intent
      1 — Completely derailed, gives up on STAR structure
      0 — No response or hostile/dismissive reaction

    Returns
    -------
    (recovery_score: float,  recovery_feedback: str)
    """
    groq = _get_groq()

    if groq is None:
        # Heuristic fallback: word count proxy
        words = len(recovery_response.split())
        score = 2.0 if words < 15 else 3.0 if words < 40 else 3.8
        return score, "Recovery scored heuristically (Groq unavailable)."

    prompt = f"""You are an expert HR interview coach evaluating how well a candidate recovered from an interviewer interrupt.

ORIGINAL QUESTION: {question}
QUESTION TYPE: {question_type}

WHAT THE CANDIDATE WAS SAYING (partial answer before interrupt):
{original_partial}

INTERVIEWER INTERRUPT: "{interrupt_text}"
INTERRUPT TYPE: {interrupt_type}

CANDIDATE'S RECOVERY RESPONSE:
{recovery_response}

Evaluate the recovery on these dimensions:
1. Direct address — did they actually answer what the interrupt asked?
2. Composure — did they stay calm and professional?
3. STAR retention — did they maintain their answer structure after the pivot?
4. Conciseness — did they pivot efficiently without rambling?
5. Confidence — did they sound assured or did they become defensive?

Respond ONLY with valid JSON (no markdown):
{{
  "recovery_score": <float 0.0–5.0>,
  "direct_address": <float 0.0–1.0>,
  "composure": <float 0.0–1.0>,
  "star_retention": <float 0.0–1.0>,
  "conciseness": <float 0.0–1.0>,
  "confidence": <float 0.0–1.0>,
  "recovery_feedback": "<2 sentence specific feedback on what they did well and what to improve>"
}}"""

    try:
        resp = await groq.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        score = round(max(0.0, min(5.0, float(data.get("recovery_score", 3.0)))), 2)
        feedback = data.get("recovery_feedback", "Recovery assessed.")
        return score, feedback
    except Exception as e:
        logger.warning(f"[Interrupt] Recovery scoring failed: {e}")
        words = len(recovery_response.split())
        score = 2.0 if words < 15 else 3.5
        return score, "Recovery could not be fully scored — try adding more specific detail when responding to pivots."


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class DialogicFeedbackEngine:
    """
    Manages dialogic feedback sessions.
    One engine instance per application (singleton).
    Stores sessions in memory (replace with Redis for multi-worker deployments).

    Usage in main.py
    ----------------
    engine = DialogicFeedbackEngine()

    # After analyze():
    result = await engine.open_dialogue(analysis_result, transcript, question, q_type)
    # result = {"dialogue_id": str, "opening": str, "score": float, "turns_remaining": int}

    # On candidate reply:
    result = await engine.advance(dialogue_id, candidate_message)
    # result = {"response": str, "score": float, "score_revised": bool, "turns_remaining": int, "closed": bool}

    # Optional explicit close:
    result = engine.close(dialogue_id)
    # result = {"final_score": float, "score_revised": bool, "log": [...], "turns": int}
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, DialogueSession] = {}

    async def open_dialogue(
        self,
        analysis_result: Dict,
        transcript: str,
        question: str,
        question_type: str = "behavioral",
    ) -> Dict:
        """
        Called once immediately after analyze() delivers its result.
        Creates a DialogueSession and returns the first coaching message.

        Parameters
        ----------
        analysis_result : dict — the full dict returned by InterviewAnalyzer.analyze()
        transcript      : str  — candidate's raw answer
        question        : str  — the interview question
        question_type   : str  — "technical" | "behavioral" | "situational"

        Returns
        -------
        {
            "dialogue_id":     str,
            "opening":         str,   # First AI message shown to candidate
            "score":           float, # Initial 0–5 score
            "turns_remaining": int,
        }
        """
        scores    = analysis_result.get("scores", {})
        ai_eval   = analysis_result.get("ai_evaluation", {})
        initial   = float(scores.get("knowledge_1_5", 3.0))

        # Extract RubricAgent's CoT reasoning for dispute corpus annotation.
        # Present in multi-agent mode (agent_scores.rubric.reasoning);
        # falls back to Groq HR reasoning in single-agent mode.
        rubric_reasoning = (
            analysis_result.get("agent_scores", {})
                           .get("rubric", {})
                           .get("reasoning", "")
            or analysis_result.get("nlp", {}).get("rubric_reasoning", "")
            or ai_eval.get("technical_evaluation", "")
        )

        session = DialogueSession(
            id                  = str(uuid.uuid4()),
            question            = question,
            question_type       = question_type,
            original_transcript = transcript,
            initial_score       = initial,
            current_score       = initial,
            score_breakdown     = scores,
            ai_evaluation       = ai_eval,
            rubric_reasoning    = rubric_reasoning,
        )
        opening = _build_opening_prompt(session)

        # Log as first assistant turn
        session.turns.append(DialogueTurn(role="assistant", content=opening))
        self._sessions[session.id] = session

        logger.info(f"[DialogicFeedback] Session {session.id[:8]} opened (score={initial:.2f})")

        return {
            "dialogue_id":     session.id,
            "opening":         opening,
            "score":           initial,
            "turns_remaining": _MAX_TURNS,
        }

    async def advance(
        self,
        dialogue_id: str,
        candidate_message: str,
    ) -> Dict:
        """
        Process one candidate reply. Returns AI response + updated score.

        Returns
        -------
        {
            "response":          str,
            "score":             float,
            "score_revised":     bool,
            "revision_delta":    float,   # 0.0 if no revision
            "turns_remaining":   int,
            "closed":            bool,    # True when turn limit reached
            "guard_triggered":   bool,    # True if any manipulation layer fired
            "novelty_score":     float,   # clarification novelty (0–1)
            "session_cap_hit":   bool,    # True if session upward cap was exhausted
        }

        Raises
        ------
        KeyError  — dialogue_id not found
        ValueError — session already closed
        """
        if dialogue_id not in self._sessions:
            raise KeyError(f"Dialogue {dialogue_id} not found")

        session = self._sessions[dialogue_id]
        if session.closed:
            raise ValueError(f"Dialogue {dialogue_id} is already closed")

        # Log candidate message
        session.turns.append(DialogueTurn(role="candidate", content=candidate_message))
        session.turn_count += 1
        session.last_candidate_message = candidate_message   # for corpus on close

        # ── Score revision (now returns 3-tuple with meta) ────────────────────
        # NOTE: We can no longer run revision + response concurrently because
        # _groq_dialogue_response now needs revision_meta to surface guard info.
        # Sequential cost is ~2–4 s; acceptable given manipulation prevention value.
        # If latency is critical, a two-step approach is possible:
        #   1. Fire revision call
        #   2. Fire response call with meta attached
        # This is left as a future optimisation (requires asyncio.create_task).
        new_score, revision_reasoning, revision_meta = await _groq_score_revision(
            session, candidate_message
        )

        score_changed  = abs(new_score - session.current_score) >= 0.05
        guard_triggered = (
            revision_meta.get("novelty_gated", False)
            or revision_meta.get("session_cap_hit", False)
        )

        # Generate dialogue response — passes meta so guard messages are surfaced
        response = await _groq_dialogue_response(
            session, candidate_message,
            score_changed=score_changed,
            new_score=new_score,
            revision_reasoning=revision_reasoning,
            revision_meta=revision_meta,
        )

        # Append inline score note only when score actually changed
        # (guard blocks reduce the delta, so score_changed may be False
        #  even though Groq wanted to revise — this is intentional)
        if score_changed:
            direction = "up" if new_score > session.current_score else "down"
            score_note = (
                f"\n\n_(Score revised {direction} to **{new_score:.1f}/5** "
                f"based on your clarification.)_"
            )
            response = response + score_note

        # ── Update session state ──────────────────────────────────────────────
        prev_score = session.current_score
        if score_changed:
            delta_01 = (new_score - session.current_score) / 5.0
            session.current_score = new_score
            session.score_revised = True
            # Track cumulative upward movement for Layer 3
            if delta_01 > 0:
                session.cumulative_upward_delta = round(
                    session.cumulative_upward_delta + delta_01, 4
                )

        session.turns.append(DialogueTurn(role="assistant", content=response))

        turns_remaining = max(0, _MAX_TURNS - session.turn_count)
        if turns_remaining == 0:
            session.closed      = True
            session.final_score = session.current_score

        logger.info(
            f"[DialogicFeedback] {session.id[:8]} turn {session.turn_count} | "
            f"score {prev_score:.2f}→{session.current_score:.2f} | "
            f"guard={guard_triggered} | novelty={revision_meta.get('novelty_score', -1):.2f} | "
            f"cumulative_up={session.cumulative_upward_delta:.3f} | closed={session.closed}"
        )

        return {
            "response":          response,
            "score":             session.current_score,
            "score_revised":     score_changed,
            "revision_delta":    round(session.current_score - prev_score, 2),
            "turns_remaining":   turns_remaining,
            "closed":            session.closed,
            "guard_triggered":   guard_triggered,
            "novelty_score":     revision_meta.get("novelty_score", -1.0),
            "session_cap_hit":   revision_meta.get("session_cap_hit", False),
        }

    def close(self, dialogue_id: str) -> Dict:
        """
        Explicitly close a session (e.g., candidate clicks 'Done').
        Safe to call even if already closed.

        If the session produced a successful upward score revision, the dispute
        is recorded into the corpus as a few-shot training example for
        RubricAgent (see dispute_corpus.py for schema and retrieval logic).

        Returns
        -------
        {
            "final_score":   float,
            "initial_score": float,
            "score_revised": bool,
            "net_delta":     float,
            "turns":         int,
            "log":           list[{"role": str, "content": str, "ts": float}],
            "dispute_recorded": bool,   # True if corpus entry was created
        }
        """
        if dialogue_id not in self._sessions:
            raise KeyError(f"Dialogue {dialogue_id} not found")

        session = self._sessions[dialogue_id]
        session.closed     = True
        session.final_score = session.current_score

        log = [
            {"role": t.role, "content": t.content, "ts": t.timestamp}
            for t in session.turns
        ]

        # ── Dispute corpus recording ──────────────────────────────────────────
        # Record if: score was revised upward AND revision is above noise floor.
        # The candidate's clarification is synthesised from all their dialogue
        # turns (not just the last one) to give the corpus the full reasoning
        # chain, as Wei et al. (2022) show multi-step reasoning examples
        # transfer better than single-turn labels.
        dispute_recorded = False
        net_delta = round(session.final_score - session.initial_score, 2)

        if session.score_revised and net_delta > 0:
            # Concatenate all candidate turns as the clarification chain
            candidate_turns = [
                t.content for t in session.turns if t.role == "candidate"
            ]
            full_clarification = " | ".join(candidate_turns) if candidate_turns else ""

            corpus_record = dispute_corpus.record_dispute(
                question_type    = session.question_type,
                question         = session.question,
                original_answer  = session.original_transcript,
                clarification    = full_clarification,
                original_score   = session.initial_score,
                revised_score    = session.final_score,
                rubric_reasoning = session.rubric_reasoning,
                dialogue_turns   = session.turn_count,
            )
            dispute_recorded = corpus_record is not None
            if dispute_recorded:
                logger.info(
                    f"[DialogicFeedback] Dispute recorded for session "
                    f"{dialogue_id[:8]} | delta=+{net_delta:.2f}"
                )

        return {
            "final_score":             session.final_score,
            "initial_score":           session.initial_score,
            "score_revised":           session.score_revised,
            "net_delta":               net_delta,
            "turns":                   session.turn_count,
            "log":                     log,
            "dispute_recorded":        dispute_recorded,
            "cumulative_upward_delta": round(session.cumulative_upward_delta, 3),
            "novelty_log":             [round(n, 3) for n in session.novelty_log],
        }

    def get_session(self, dialogue_id: str) -> Optional[DialogueSession]:
        return self._sessions.get(dialogue_id)

    def purge_old(self, max_age_s: float = 3600.0) -> int:
        """Purge sessions older than max_age_s. Call from a periodic task."""
        now    = time.time()
        before = len(self._sessions)
        self._sessions = {
            k: v for k, v in self._sessions.items()
            if now - v.created_at < max_age_s
        }
        removed = before - len(self._sessions)
        if removed:
            logger.info(f"[DialogicFeedback] Purged {removed} old sessions")
        return removed

    # ── Interrupt & Recover API ───────────────────────────────────────────────

    async def check_interrupt(
        self,
        dialogue_id: str,
        partial_transcript: str,
    ) -> Dict:
        """
        Called by the frontend periodically while the candidate is answering
        (e.g. every 5 seconds of recording, or after each browser-STT chunk).

        If the interrupt conditions are met, fires an interrupt and stores it
        in the session. Returns a dict the frontend uses to display the interrupt.

        Parameters
        ----------
        dialogue_id         : str — active dialogue session id
        partial_transcript  : str — everything transcribed so far this answer

        Returns
        -------
        {
            "interrupt_fired": bool,
            "interrupt_text":  str,   # shown to candidate if fired
            "interrupt_type":  str,   # e.g. "evidence", "outcome"
            "word_trigger":    int,   # word count when it fired
        }
        """
        if dialogue_id not in self._sessions:
            raise KeyError(f"Dialogue {dialogue_id} not found")

        session = self._sessions[dialogue_id]
        word_count = len(partial_transcript.split())

        if not _should_interrupt(word_count, session.interrupt_fired):
            return {"interrupt_fired": False, "interrupt_text": "", "interrupt_type": "", "word_trigger": 0}

        # Generate the interrupt
        interrupt_text, interrupt_type = await _generate_interrupt(
            question=session.question,
            partial_transcript=partial_transcript,
            question_type=session.question_type,
        )

        # Store in session
        session.interrupt_fired       = True
        session.interrupt_type        = interrupt_type
        session.interrupt_text        = interrupt_text
        session.interrupt_word_trigger = word_count

        logger.info(
            f"[Interrupt] Session {dialogue_id[:8]} — interrupt fired at word {word_count} "
            f"type={interrupt_type}"
        )

        return {
            "interrupt_fired": True,
            "interrupt_text":  interrupt_text,
            "interrupt_type":  interrupt_type,
            "word_trigger":    word_count,
        }

    async def score_interrupt_recovery(
        self,
        dialogue_id: str,
        recovery_response: str,
        original_partial: str = "",
    ) -> Dict:
        """
        Called after the candidate has responded to the interrupt.
        Scores their recovery and stores the result in the session.

        Parameters
        ----------
        dialogue_id       : str — active dialogue session id
        recovery_response : str — candidate's recovery answer (full text after interrupt)
        original_partial  : str — what they were saying before the interrupt fired

        Returns
        -------
        {
            "recovery_score":    float,   # 0–5
            "recovery_feedback": str,
            "interrupt_type":    str,
            "interrupt_text":    str,
        }
        """
        if dialogue_id not in self._sessions:
            raise KeyError(f"Dialogue {dialogue_id} not found")

        session = self._sessions[dialogue_id]

        if not session.interrupt_fired:
            return {
                "recovery_score":    -1.0,
                "recovery_feedback": "No interrupt was fired for this session.",
                "interrupt_type":    "",
                "interrupt_text":    "",
            }

        score, feedback = await _score_recovery(
            question=session.question,
            interrupt_text=session.interrupt_text,
            interrupt_type=session.interrupt_type,
            original_partial=original_partial or session.original_transcript[:500],
            recovery_response=recovery_response,
            question_type=session.question_type,
        )

        session.recovery_score     = score
        session.interrupt_response = recovery_response

        logger.info(
            f"[Interrupt] Session {dialogue_id[:8]} — recovery scored {score:.2f}/5 "
            f"type={session.interrupt_type}"
        )

        return {
            "recovery_score":    score,
            "recovery_feedback": feedback,
            "interrupt_type":    session.interrupt_type,
            "interrupt_text":    session.interrupt_text,
        }

    def get_interrupt_summary(self, dialogue_id: str) -> Dict:
        """
        Returns the interrupt & recovery summary for inclusion in the session report.
        Safe to call even when no interrupt fired (returns recovery_score=-1).
        """
        if dialogue_id not in self._sessions:
            return {"interrupt_fired": False, "recovery_score": -1.0}

        session = self._sessions[dialogue_id]
        return {
            "interrupt_fired":        session.interrupt_fired,
            "interrupt_type":         session.interrupt_type,
            "interrupt_text":         session.interrupt_text,
            "interrupt_word_trigger": session.interrupt_word_trigger,
            "interrupt_response":     session.interrupt_response,
            "recovery_score":         session.recovery_score,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
dialogic_engine = DialogicFeedbackEngine()


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTE HELPERS  (paste into main.py)
# ══════════════════════════════════════════════════════════════════════════════
"""
PASTE THIS BLOCK INTO main.py (after importing dialogic_engine):

from dialogic_feedback import dialogic_engine

class DialogueOpenRequest(BaseModel):
    transcript:    str
    question:      str
    question_type: str = "behavioral"
    analysis_result: dict  # full analyze() output

class DialogueTurnRequest(BaseModel):
    dialogue_id: str
    message:     str

@app.post("/dialogue/open")
async def dialogue_open(req: DialogueOpenRequest):
    return await dialogic_engine.open_dialogue(
        req.analysis_result, req.transcript, req.question, req.question_type
    )

@app.post("/dialogue/turn")
async def dialogue_turn(req: DialogueTurnRequest):
    return await dialogic_engine.advance(req.dialogue_id, req.message)

@app.post("/dialogue/close")
async def dialogue_close(dialogue_id: str):
    return dialogic_engine.close(dialogue_id)
"""