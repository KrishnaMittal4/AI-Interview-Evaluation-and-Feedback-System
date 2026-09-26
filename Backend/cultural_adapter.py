"""
cultural_adapter.py — Aura AI | Cultural Communication Style Adapter (v1.0)
============================================================================
Addresses a systematic bias in STAR-based interview scoring:

PROBLEM
-------
STAR framework assumptions (direct first-person narrative, explicit quantified
results) reflect Western low-context communication norms. Candidates from
high-context cultures (East Asia, South Asia, MENA, Latin America) use:

  • Collective attribution  — "we achieved" rather than "I achieved"
  • Indirect framing        — context implied, not stated
  • Implicit results        — outcome inferred from narrative, not declared
  • Relational emphasis     — group harmony signals over individual achievement
  • Hedged certainty        — politeness norms produce hedge-heavy phrasing

The current _compute_star_score_v3() regex and WEIGHT_PROFILES["behavioural"]
penalise all of these patterns — not because the answer is weak, but because
it expresses competence in a culturally different register.

RESEARCH BASIS
--------------
Chua & Mazmanian (2022, ACM CHI) — Social class and communication style in
  asynchronous video interviews: candidates from high-context backgrounds
  receive systematically lower scores on "confidence" and "structure" rubrics
  despite equivalent underlying competence.

Hall (1976) — Beyond Culture: high-context vs low-context communication
  framework. High-context cultures encode meaning in relationships, tone, and
  shared context; low-context encode explicitly in words.

Kim et al. (2010, J Cross-Cultural Psychol.) — Collectivist cultures suppress
  first-person singular pronoun use as a politeness and humility norm; forced
  STAR scoring penalises this as "lack of ownership".

ACM CHI (2025) asynchronous interview study — rubric bias against indirect
  communicators persists even when evaluators are trained to ignore it.

SOLUTION IMPLEMENTED
--------------------
1. Auto-detect cultural context from the transcript ("low-context" default;
   "high-context" when collective/indirect/implicit signals detected).

2. Two adjustments applied when high-context mode is active:

   (a) STAR weight rebalancing
       behavioural: star 0.35 → 0.20; depth_flu 0.10 → 0.25
       hr:          star 0.20 → 0.10; depth_flu 0.25 → 0.35
       Rationale: depth and fluency score the *quality* of reasoning;
       STAR scores the *narrative structure* — a culturally loaded rubric.

   (b) Collective attribution bonus (±0.0–0.3 on final STAR score)
       "we" pronoun presence with action verbs earns partial STAR Action
       credit. "Our team achieved" is semantically equivalent to
       "I achieved" for competence inference purposes.

3. All adjustments are transparent — logged in `cultural_detail` dict
   returned alongside the patched weights, ready for paper reporting.

INTEGRATION
-----------
Drop-in: call adapt_weights_for_culture() before the composite formula in
_full_evaluate_v3() and apply the returned weight dict.

The `cultural_context` param defaults to "auto" — backward compatible with
all existing calls. Pass "low-context" to force the original scoring.

ABLATION CONDITIONS (for paper)
--------------------------------
Condition X: cultural_context="low-context"  — original STAR-biased scoring
Condition Y: cultural_context="auto"         — auto-detected adaptation
Condition Z: cultural_context="high-context" — forced high-context weights

OCEAN KEYWORD BIAS (this update)
---------------------------------
Problem: _compute_ocean_v3() uses a single global OCEAN_KEYWORDS dict built
on Western, low-context, individualist interview language. This creates a
second layer of bias on top of the STAR weight bias already addressed above.

Specifically (Kim et al. 2010, J Cross-Cultural Psychol.):

  Extraversion:    Keywords like "led", "convinced", "proactive", "outreach"
                   assume individual, assertive leadership. High-context
                   speakers who "facilitated group discussion", "brought people
                   together", or "ensured everyone was heard" express the same
                   construct in relational, collective terms — and score near
                   baseline on the LC keyword list.

  Agreeableness:   "compromise" and "accommodated" are explicitly Western
                   conflict-resolution terms. High-context equivalents
                   ("maintained harmony", "preserved the relationship",
                   "avoided confrontation", "found common ground") are
                   semantically equivalent but absent from the LC list.

  Conscientiousness: "organised", "deadline", "systematic" presuppose
                   explicit planning discourse. High-context conscientiousness
                   often appears as process narration ("step by step we
                   ensured", "carefully considered each aspect") — implicit
                   structure rather than named structure.

  Neuroticism_inv: "confident" and "assertive" are LC signals. HC stability
                   appears as "remained grounded", "kept perspective",
                   "stayed focused on what mattered" — composure described
                   through collective anchoring rather than individual
                   self-assertion.

  Openness:        Least biased — curiosity and creativity are expressed
                   fairly similarly across cultures. Minor additions for HC
                   (group ideation, building on others' ideas).

Solution: HC_OCEAN_KEYWORDS contains culture-equivalent keyword lists for each
trait. get_ocean_keywords_for_context() merges LC and HC lists when high-context
is detected, ensuring HC candidates are not penalised for using legitimate
alternative expressions of the same underlying traits.

Ablation extension:
  Condition X: cultural_context="low-context"  — LC OCEAN keywords only
  Condition Y: cultural_context="auto"         — merged LC+HC keywords
  Condition Z: cultural_context="high-context" — merged LC+HC keywords (forced)

  Report: does HC candidate OCEAN accuracy (vs self-rated) improve under Y/Z?
  Expected: Extraversion and Agreeableness show largest improvement.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

# Collective attribution — "we", "our team", "the team", "my colleagues"
_COLLECTIVE_PATTERN = re.compile(
    r"\b(we\b|our team|the team|my team|my colleagues|our group|"
    r"together we|as a team|collectively|our department|the department|"
    r"all of us|the group|we worked|we built|we achieved|we delivered|"
    r"we decided|we implemented|we developed)\b",
    re.IGNORECASE,
)

# Indirect result framing — implied outcome without explicit "I achieved X%"
_INDIRECT_RESULT = re.compile(
    r"\b(it helped|it allowed|this enabled|which led|which resulted|"
    r"everyone felt|the team felt|feedback was positive|things improved|"
    r"went well|worked out|turned out|the outcome was|it went|"
    r"by the end|at the end|overall|in the end it)\b",
    re.IGNORECASE,
)

# Humility hedging — politeness norms, NOT nervousness
_HUMILITY_HEDGE = re.compile(
    r"\b(humbly|respectfully|with all due respect|i was fortunate|"
    r"i was lucky|i had the opportunity|i was given|i was asked|"
    r"perhaps i|i may have|i believe i|i hope|i tried my best|"
    r"i did my part|i contributed|i supported)\b",
    re.IGNORECASE,
)

# Low-context strong ownership markers — direct first-person achievement claims
_LC_OWNERSHIP = re.compile(
    r"\b(i led|i built|i designed|i implemented|i achieved|i delivered|"
    r"i drove|i owned|i spearheaded|i created|i initiated|i launched|"
    r"i increased|i reduced|i saved|i cut|i improved by)\b",
    re.IGNORECASE,
)

# Collective action verbs that pair with "we" for STAR Action credit
_COLLECTIVE_ACTION_VERBS = re.compile(
    r"\b(we (built|designed|implemented|achieved|delivered|led|drove|created|"
    r"launched|improved|solved|fixed|completed|developed|released|"
    r"deployed|migrated|refactored|established|reduced|increased|"
    r"saved|optimised|automated|coordinated|shipped))\b",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT PROFILES — CULTURAL VARIANTS
#  Mirrors WEIGHT_PROFILES in analyzer.py but with high-context adjustments.
# ══════════════════════════════════════════════════════════════════════════════

# Original low-context weights (copied from analyzer.py for reference)
_LC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "technical":   {"star": 0.00, "word_cat": 0.10, "relevance": 0.40,
                    "keyword": 0.25, "depth_flu": 0.20, "grammar": 0.05},
    "behavioural": {"star": 0.35, "word_cat": 0.20, "relevance": 0.20,
                    "keyword": 0.10, "depth_flu": 0.10, "grammar": 0.05},
    "hr":          {"star": 0.20, "word_cat": 0.15, "relevance": 0.25,
                    "keyword": 0.10, "depth_flu": 0.25, "grammar": 0.05},
}

# High-context adjusted weights
# Key changes: STAR ↓, depth_flu ↑ (rewards reasoning quality over structure)
# word_cat also ↑ slightly for behavioural (collective/relational vocabulary)
_HC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "technical":   {"star": 0.00, "word_cat": 0.10, "relevance": 0.40,
                    "keyword": 0.25, "depth_flu": 0.20, "grammar": 0.05},
    "behavioural": {"star": 0.18, "word_cat": 0.22, "relevance": 0.20,
                    "keyword": 0.10, "depth_flu": 0.25, "grammar": 0.05},
    "hr":          {"star": 0.10, "word_cat": 0.17, "relevance": 0.25,
                    "keyword": 0.10, "depth_flu": 0.33, "grammar": 0.05},
}


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def detect_cultural_context(text: str) -> Tuple[str, Dict]:
    """
    Detect whether the answer exhibits high-context or low-context
    communication patterns.

    Parameters
    ----------
    text : str  — candidate's answer (raw, any case)

    Returns
    -------
    (context_label, detail)

    context_label : "high-context" | "low-context"
    detail : dict with signal counts and confidence score
    """
    tl = text.lower()
    wc = max(len(tl.split()), 1)

    collective_hits  = len(_COLLECTIVE_PATTERN.findall(text))
    indirect_hits    = len(_INDIRECT_RESULT.findall(text))
    humility_hits    = len(_HUMILITY_HEDGE.findall(text))
    lc_ownership     = len(_LC_OWNERSHIP.findall(text))
    coll_action_hits = len(_COLLECTIVE_ACTION_VERBS.findall(text))

    # First-person singular density (I / my / me / myself)
    first_person_sg = len(re.findall(r"\b(i|my|me|myself)\b", tl))
    first_person_pl = len(re.findall(r"\b(we|our|us|ourselves)\b", tl))
    fp_ratio = first_person_pl / max(first_person_sg + first_person_pl, 1)

    # Composite high-context score (0–1)
    # Weighted: collective language most diagnostic, then indirect results
    hc_score = (
        min(collective_hits / max(wc * 0.02, 1), 1.0) * 0.35 +
        min(indirect_hits   / max(wc * 0.015, 1), 1.0) * 0.25 +
        min(humility_hits   / max(wc * 0.01, 1), 1.0) * 0.15 +
        fp_ratio                                         * 0.25
    )
    # Low-context ownership markers subtract from score
    lc_signal = min(lc_ownership / max(wc * 0.02, 1), 1.0) * 0.30
    hc_score  = max(0.0, min(1.0, hc_score - lc_signal))

    # Threshold: 0.30 is enough to trigger adaptation
    # (conservative — we'd rather under-adapt than over-adapt)
    context = "high-context" if hc_score >= 0.30 else "low-context"

    detail = {
        "context":            context,
        "hc_score":           round(hc_score, 3),
        "collective_hits":    collective_hits,
        "collective_action":  coll_action_hits,
        "indirect_result":    indirect_hits,
        "humility_hedge":     humility_hits,
        "lc_ownership":       lc_ownership,
        "fp_plural_ratio":    round(fp_ratio, 3),
        "threshold":          0.30,
    }
    return context, detail


def _collective_star_bonus(text: str, star_sc: float) -> Tuple[float, str]:
    """
    Award partial STAR Action credit for collective action verbs.

    "We built a pipeline that reduced latency by 40%." contains
    Action + Result semantics — only the pronoun differs from
    "I built a pipeline that reduced latency by 40%."

    Bonus: +0.0 to +0.30 on the STAR score (never exceeds 5.0).
    The bonus is capped so it cannot push a weak answer above average.

    Returns (bonus_amount, reasoning_string)
    """
    hits = len(_COLLECTIVE_ACTION_VERBS.findall(text))
    if hits == 0:
        return 0.0, "no_collective_action_verbs"

    # Diminishing returns: first hit = +0.15, second = +0.10, rest = +0.05 each
    if hits == 1:
        bonus = 0.15
    elif hits == 2:
        bonus = 0.25
    else:
        bonus = 0.30

    # Cap: cannot push score above 4.5 (requires own narrative to reach 5.0)
    bonus = min(bonus, max(0.0, 4.5 - star_sc))
    return round(bonus, 3), f"collective_action_hits={hits}"


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def adapt_weights_for_culture(
    text: str,
    q_type_key: str,          # "technical" | "behavioural" | "hr"
    star_sc: float,           # raw STAR score from _compute_star_score_v3()
    base_weights: Dict,       # the weight dict already built in _full_evaluate_v3
    cultural_context: str = "auto",   # "auto" | "high-context" | "low-context"
) -> Tuple[Dict, float, Dict]:
    """
    Adapt scoring weights and STAR score for cultural communication style.

    Parameters
    ----------
    text             : candidate's raw answer
    q_type_key       : resolved question type
    star_sc          : raw STAR score (0–5)
    base_weights     : weight dict from WEIGHT_PROFILES (already keyword-adjusted)
    cultural_context : "auto" (default) | "high-context" | "low-context"

    Returns
    -------
    (adapted_weights, adapted_star_sc, cultural_detail)

    adapted_weights  : dict — may be identical to base_weights if low-context
    adapted_star_sc  : float — star_sc ± collective bonus
    cultural_detail  : dict — full audit trail for paper reporting
    """
    # ── 1. Resolve context ────────────────────────────────────────────────────
    if cultural_context == "auto":
        detected_ctx, detect_detail = detect_cultural_context(text)
    elif cultural_context == "high-context":
        detected_ctx = "high-context"
        _, detect_detail = detect_cultural_context(text)   # still compute for audit
        detect_detail["context"] = "high-context (forced)"
    else:
        # "low-context" or any unknown value — no adaptation
        detected_ctx = "low-context"
        _, detect_detail = detect_cultural_context(text)
        detect_detail["context"] = "low-context (forced)"

    # ── 2. Technical questions: no STAR adjustment needed ────────────────────
    if q_type_key == "technical":
        cultural_detail = {
            **detect_detail,
            "weights_adjusted":    False,
            "reason":              "technical questions unaffected by STAR bias",
            "star_bonus":          0.0,
            "star_bonus_reason":   "n/a",
            "adapted_star_sc":     star_sc,
        }
        return dict(base_weights), star_sc, cultural_detail

    # ── 3. Low-context: return originals unchanged ────────────────────────────
    if detected_ctx == "low-context":
        cultural_detail = {
            **detect_detail,
            "weights_adjusted": False,
            "reason":           "low-context detected — original weights used",
            "star_bonus":       0.0,
            "star_bonus_reason": "n/a",
            "adapted_star_sc":  star_sc,
        }
        return dict(base_weights), star_sc, cultural_detail

    # ── 4. High-context: apply adjusted weights ───────────────────────────────
    hc_w = dict(_HC_WEIGHTS[q_type_key])

    # Preserve any keyword-zero adjustment already applied (from _full_evaluate_v3)
    # If caller zeroed out keyword weight and moved it to relevance, keep that.
    if base_weights.get("keyword", 1.0) == 0.0:
        freed = hc_w["keyword"]
        hc_w["keyword"] = 0.0
        hc_w["relevance"] = round(hc_w["relevance"] + freed, 4)

    # ── 5. Collective attribution bonus on STAR ───────────────────────────────
    star_bonus, bonus_reason = _collective_star_bonus(text, star_sc)
    adapted_star = round(min(5.0, star_sc + star_bonus), 3)

    # ── 6. Build audit trail ──────────────────────────────────────────────────
    cultural_detail = {
        **detect_detail,
        "weights_adjusted":  True,
        "original_weights":  dict(base_weights),
        "adapted_weights":   hc_w,
        "weight_delta": {
            k: round(hc_w.get(k, 0) - base_weights.get(k, 0), 4)
            for k in hc_w
        },
        "star_bonus":        star_bonus,
        "star_bonus_reason": bonus_reason,
        "adapted_star_sc":   adapted_star,
        "reason": (
            f"high-context detected (hc_score={detect_detail['hc_score']:.2f}) — "
            f"STAR weight reduced, depth/fluency weight increased"
        ),
    }

    return hc_w, adapted_star, cultural_detail


# ══════════════════════════════════════════════════════════════════════════════
#  OCEAN KEYWORD BANKS — CULTURE-ADAPTED
# ══════════════════════════════════════════════════════════════════════════════
#
# DESIGN PRINCIPLES
# -----------------
# 1. Every HC keyword is a SEMANTIC EQUIVALENT of an existing LC keyword —
#    not a different trait, not a related concept, but the same construct
#    expressed in high-context register. Each entry is annotated with its
#    LC counterpart for paper reporting / ablation analysis.
#
# 2. Keywords are lowercase phrases (1–3 words), matching the format
#    expected by _compute_ocean_v3()'s word-level search loop.
#
# 3. HC keywords are ADDITIVE — they are merged with the LC bank, not
#    substituted. This ensures LC candidates are unaffected and HC
#    candidates gain coverage without losing LC keyword matches.
#
# 4. Multi-word phrases are intentional: "brought everyone together" is more
#    specific and less noisy than "together" alone, reducing false positives.
#
# RESEARCH BASIS
# --------------
# Kim et al. (2010) — first-person pronoun suppression in collectivist cultures.
# Earley & Ang (2003) — Cultural Intelligence: role of relational Extraversion.
# Markus & Kitayama (1991) — independent vs interdependent self-construal;
#   collectivist Conscientiousness is expressed through group-process narration.
# Triandis (1995) — Individualism & Collectivism: Agreeableness as harmony
#   maintenance rather than explicit compromise.
# Mesquita (2001) — emotional expression norms; HC Neuroticism_inv manifests
#   as equanimity and groundedness rather than confidence assertion.
# Hofstede (2001) — uncertainty avoidance and long-term orientation predict
#   HC Openness expression as iterative refinement over bold novelty claims.

HC_OCEAN_KEYWORDS: Dict[str, List[str]] = {

    # ── OPENNESS ──────────────────────────────────────────────────────────────
    # LC equivalents: creative, innovative, explored, experiment, brainstormed
    # HC pattern: curiosity expressed through collective ideation, building on
    # others, iterative improvement, cultural/contextual awareness.
    "Openness": [
        # Group ideation (≈ brainstormed, ideated)
        "we explored",
        "we considered different",
        "we looked at various",
        "built on each other",
        "discussed different approaches",
        "drew from different perspectives",
        "took ideas from the team",
        # Iterative refinement (≈ experiment, rethought)
        "refined over time",
        "adjusted our approach",
        "evolved our thinking",
        "revisited our assumptions",
        "reconsidered",
        "took a different path",
        "tried a new way",
        # Contextual/cultural awareness (no direct LC equiv — bonus signal)
        "adapted to the context",
        "understood the environment",
        "considered the broader picture",
        "sensitive to the situation",
        # Learning from others (≈ learned, researched)
        "learned from colleagues",
        "sought guidance",
        "asked for input",
        "gathered different views",
        "took on board feedback",
    ],

    # ── CONSCIENTIOUSNESS ─────────────────────────────────────────────────────
    # LC equivalents: organised, planned, deadline, systematic, documented,
    #   tracked, scheduled, prioritised, verified, quality
    # HC pattern: conscientiousness appears as careful collective process
    # narration — "step by step", "made sure each part", "nothing was missed"
    # rather than naming the planning behaviour explicitly.
    "Conscientiousness": [
        # Process narration (≈ systematic, structured, organised)
        "step by step",
        "one step at a time",
        "we made sure",
        "ensured each",
        "carefully considered",
        "nothing was overlooked",
        "covered every aspect",
        "went through each",
        "checked each",
        "made sure nothing",
        # Collective accountability (≈ tracked, documented, reviewed)
        "we kept track",
        "we monitored",
        "kept everyone informed",
        "updated the team",
        "made sure the team knew",
        "reported back",
        "followed up",
        # Thoroughness framed implicitly (≈ accurate, verified, tested)
        "double checked",
        "made sure it was right",
        "reviewed together",
        "validated with the team",
        "confirmed with",
        # Deadline/delivery in collective framing (≈ deadline, on time)
        "we delivered on time",
        "met the timeline",
        "the team finished",
        "completed as planned",
        "on schedule",
        "before the deadline",
    ],

    # ── EXTRAVERSION ──────────────────────────────────────────────────────────
    # LC equivalents: led, presented, networked, convinced, proactive, outreach,
    #   facilitated, mentored, coached, negotiated
    # HC pattern: leadership is relational and facilitative rather than
    # directive. Earley & Ang (2003) — culturally intelligent Extraversion
    # shows as "brought people together", "made sure all voices were heard".
    "Extraversion": [
        # Relational leadership (≈ led, facilitated)
        "brought the team together",
        "brought everyone together",
        "kept the team aligned",
        "made sure everyone was heard",
        "ensured all voices",
        "kept everyone on the same page",
        "helped the team stay focused",
        "kept the group together",
        "rallied the team",
        # Collective communication (≈ presented, communicated, shared)
        "we communicated",
        "the team discussed",
        "we shared our progress",
        "kept stakeholders informed",
        "presented our findings",
        "explained to the group",
        # Harmony-centred initiative (≈ proactive, outreach)
        "reached out to",
        "checked in with",
        "made sure to connect",
        "touched base",
        "stayed in contact",
        "maintained the relationship",
        # Collective mentoring (≈ mentored, coached)
        "supported the team",
        "helped colleagues",
        "guided the group",
        "assisted team members",
        "shared knowledge with",
        "passed on what i knew",
        # Collective negotiation (≈ negotiated, convinced)
        "found a solution together",
        "reached an agreement",
        "aligned on a decision",
        "came to a consensus",
        "worked through the disagreement",
        "found common ground",
    ],

    # ── AGREEABLENESS ─────────────────────────────────────────────────────────
    # LC equivalents: compromise, accommodated, cooperative, patient, flexible,
    #   respected, inclusive, collaborative
    # HC pattern: Triandis (1995) — HC Agreeableness is expressed as harmony
    # maintenance ("kept the peace", "avoided unnecessary conflict") rather
    # than explicit compromise framing ("we compromised on X").
    "Agreeableness": [
        # Harmony maintenance (≈ compromise, accommodated)
        "maintained harmony",
        "kept the peace",
        "preserved the relationship",
        "avoided unnecessary conflict",
        "prevented escalation",
        "kept things smooth",
        "de-escalated",
        # Relational sensitivity (≈ empathised, listened, understood)
        "was mindful of",
        "sensitive to their needs",
        "considered their perspective",
        "took their feelings into account",
        "respected their view",
        "acknowledged their concern",
        "gave them space",
        # Collective patience (≈ patient, flexible)
        "took the time",
        "gave it time",
        "waited for consensus",
        "let the process unfold",
        "did not rush",
        # Group-centred inclusion (≈ inclusive, collaborative, valued)
        "made sure everyone felt included",
        "ensured no one was left out",
        "brought in quieter voices",
        "made space for",
        "checked if everyone agreed",
        # Implicit cooperation (≈ cooperative, supported)
        "worked alongside",
        "contributed to the group",
        "pitched in",
        "did my part for the team",
        "helped where i could",
        "stepped in when needed",
    ],

    # ── NEUROTICISM_INV (emotional stability) ─────────────────────────────────
    # LC equivalents: calm, composed, confident, focused, resilient, persevered,
    #   resolved, rational, objective, steady
    # HC pattern: Mesquita (2001) — HC emotional stability appears as
    # equanimity and relational grounding rather than individual confidence
    # assertion. "Remained grounded" vs "I stayed calm"; "kept perspective"
    # vs "I was composed". Collective anchoring is also a stability signal.
    "Neuroticism_inv": [
        # Equanimity (≈ calm, composed, steady)
        "remained grounded",
        "kept my perspective",
        "stayed grounded",
        "kept a level head",
        "took it in my stride",
        "stayed centred",
        "maintained perspective",
        # Collective anchoring as stability source (no direct LC equiv)
        "leaned on the team",
        "we kept each other focused",
        "the team stayed calm",
        "we supported each other through",
        "together we stayed on track",
        # Implicit resilience (≈ persevered, overcome, adapted)
        "we pushed through",
        "continued despite",
        "kept going",
        "did not give up",
        "found another way",
        "worked around it",
        "adapted to the situation",
        # Measured response (≈ rational, objective, constructive)
        "thought it through",
        "took a step back",
        "considered carefully before",
        "did not react immediately",
        "approached it calmly",
        "looked at it objectively",
        "tried to understand before",
        # Resolution through group (≈ resolved, handled, managed)
        "we worked through it",
        "we resolved it together",
        "the team handled it",
        "we found a way forward",
        "we addressed it as a team",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — OCEAN KEYWORD SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_ocean_keywords_for_context(
    cultural_context: str,
    lc_keywords: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Return the OCEAN keyword bank appropriate for the detected cultural context.

    Parameters
    ----------
    cultural_context : str
        "high-context" | "low-context".
        Pass the first element of detect_cultural_context() return value,
        or the forced override from the caller.

    lc_keywords : Dict[str, List[str]]
        The existing low-context OCEAN_KEYWORDS dict from analyzer.py.
        This is passed in (not imported) to avoid circular imports.

    Returns
    -------
    Dict[str, List[str]]
        "low-context"  → lc_keywords unchanged (zero impact on existing scoring).
        "high-context" → merged dict: lc_keywords[trait] + HC_OCEAN_KEYWORDS[trait],
                         deduplicated, preserving LC order first.

    Merge strategy: additive, LC-first.
    Rationale: A HC candidate may still use some LC keywords (e.g. "led" appears
    in a sentence like "our team led the project"). Keeping LC keywords ensures
    those hits are captured. HC keywords are appended so they extend coverage
    without displacing existing scoring behaviour for LC candidates.

    The merged lists are used ONLY inside _compute_ocean_v3() — all other
    analyzer.py behaviour (weight profiles, STAR scoring, sentiment) is unaffected.
    """
    if cultural_context != "high-context":
        return lc_keywords   # LC path: zero change to existing behaviour

    merged: Dict[str, List[str]] = {}
    for trait, lc_kws in lc_keywords.items():
        hc_kws = HC_OCEAN_KEYWORDS.get(trait, [])
        # Deduplicate while preserving order (LC first)
        seen: set = set()
        combined: List[str] = []
        for kw in lc_kws + hc_kws:
            if kw not in seen:
                seen.add(kw)
                combined.append(kw)
        merged[trait] = combined

    return merged