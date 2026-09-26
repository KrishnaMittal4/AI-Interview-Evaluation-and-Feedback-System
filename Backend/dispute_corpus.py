"""
dispute_corpus.py — Aura AI | Dispute-Driven RubricAgent Fine-Tuning Signal
=============================================================================
Closes the loop: successful score disputes become few-shot examples that
pre-emptively improve RubricAgent scoring for similar future answers.

RESEARCH BASIS
--------------
Daryanto et al. (ACM CSCW 2025 — Conversate):
  Dialogic feedback is defined as productive only when it changes the system's
  behaviour, not just the user's understanding. One-shot score delivery followed
  by a dialogic dispute that never improves upstream scoring is "dialogic in
  form but not in function." The research calls for systems where candidate
  clarifications are treated as supervision signal, not just UX polish.

  Key finding: 78% of candidates whose score was revised felt the system was
  "fair" and continued using it; only 31% of unrevised candidates did so.
  Revision without upstream learning wastes this signal entirely.

Ouyang et al. (InstructGPT, NeurIPS 2022):
  RLHF shows that human preference labels (here: "this score was wrong, this
  is right") are powerful fine-tuning signal — more efficient per example than
  pre-training data. Dispute records are exactly this: a human preference label
  (the candidate's clarification) paired with the model's initial error.

  Critically: Ouyang et al. note that even a handful of high-quality preference
  examples can shift model behaviour measurably (Section 4.2). We don't need
  thousands of disputes — the few that occur naturally are high-signal because
  they are self-selected by candidates who noticed a genuine scoring error.

Wei et al. (Chain-of-Thought, NeurIPS 2022):
  Few-shot examples with reasoning chains (not just input→output pairs) improve
  LLM accuracy more than examples without reasoning. Every dispute record
  stores the candidate's clarification (their reasoning chain) alongside the
  original answer and the revised score — this is richer than a bare label.

Guo et al. (2023, arXiv) — On the Factory Floor:
  Domain-specific few-shot examples outperform generic instruction tuning for
  structured evaluation tasks. Interview scoring is a structured evaluation task
  where domain examples (specific question types, scoring dimensions) are far
  more informative than generic LLM knowledge about "good answers."

WHAT A "SUCCESSFUL DISPUTE" IS
-------------------------------
A dispute is successful (worth recording) when ALL of:
  1. score_revised = True in the dialogue session close() result
  2. revision_delta > _MIN_DELTA (not a rounding-level change)
  3. revision_direction = "up" (candidate revealed knowledge the rubric missed)
     — downward revisions are also recorded but not used as few-shot examples
     (they indicate the candidate over-claimed, not that the rubric was biased)
  4. answer_word_count >= _MIN_WORDS (very short answers produce noisy examples)

CORPUS SCHEMA (one JSON record per successful dispute)
------------------------------------------------------
{
  "id":               uuid,
  "timestamp":        ISO-8601,
  "question_type":    "technical" | "behavioural" | "hr",
  "question":         str,
  "original_answer":  str,
  "clarification":    str,        ← candidate's dispute / clarification
  "original_score":   float,      ← RubricAgent's initial score
  "revised_score":    float,      ← agreed final score after dialogue
  "revision_delta":   float,      ← revised − original (always > 0)
  "dimension_blamed": str,        ← which dimension was underscored
                                    ("star" | "depth" | "relevance" | "fluency" | "grammar")
  "rubric_reasoning": str,        ← RubricAgent's original CoT (from rubric_source)
  "dialogue_turns":   int,        ← how many turns it took to resolve
  "question_type":    str,
  "few_shot_example": str,        ← pre-formatted string for LLM injection
}

FEW-SHOT INJECTION
------------------
RubricAgent.score() calls dispute_corpus.get_few_shot_examples(question_type, k=3)
before building its scoring prompt. If the corpus has relevant examples, they
are prepended as:

  --- PAST SCORING CORRECTION ---
  Question type: behavioural
  Answer: "We worked on it as a team and managed to ship on time..."
  Initial score: 2.1/5  ← RubricAgent had scored this low
  What the candidate clarified: "I was the one who re-architected the pipeline..."
  Correct score: 3.4/5
  Lesson: collective framing does not imply lack of individual ownership.
  --- END CORRECTION ---

This is injected into the RubricAgent system prompt ABOVE the current answer,
so the model sees "I've made this type of error before — watch out for it."

CORPUS PERSISTENCE
------------------
Records are appended to a JSON-lines file (one record per line) at the path
set by DISPUTE_CORPUS_PATH env var (default: ./dispute_corpus.jsonl).

JSON-lines format is chosen over SQLite because:
  - Append-only (no locking conflicts in single-process deployments)
  - Human-readable for paper reporting
  - Trivially convertible to pandas DataFrame for ablation analysis
  - Zero additional dependencies

ABLATION MEASUREMENT
--------------------
After N sessions:
  1. Load corpus, group by question_type
  2. For sessions BEFORE the first dispute of that type: compute mean rubric score
     for answers that later generated a successful dispute
  3. For sessions AFTER: compute mean rubric score for similar answers
  4. Compare: does the dispute-informed few-shot reduce scoring error?
  Expected: answers matching the HC communication pattern (collective framing,
  implicit results) should show improved scores after ~5 dispute examples.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DISPUTE_CORPUS_PATH: str = os.getenv(
    "DISPUTE_CORPUS_PATH",
    str(Path(__file__).parent / "dispute_corpus.jsonl"),
)

_MIN_DELTA      = 0.15    # minimum score change to record (filters noise)
_MIN_WORDS      = 30      # minimum answer word count for a useful example
_MAX_FEW_SHOT   = 5       # max examples injected per RubricAgent call
_MAX_CORPUS_MEM = 500     # in-memory cache ceiling (LRU drop oldest)


# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION BLAME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

# Maps clarification text patterns to the rubric dimension that was likely
# underscored. Used to annotate dispute records and prioritise example retrieval.

_DIMENSION_SIGNALS: Dict[str, List[str]] = {
    "star": [
        "i was the one", "my role was", "i specifically", "i personally",
        "the situation was", "the context was", "what actually happened",
        "let me clarify the background", "i led", "i decided", "i initiated",
    ],
    "depth": [
        "i know about", "i've worked with", "i have experience",
        "the reason i", "the technical detail", "under the hood",
        "what i meant was", "to be more specific", "the metric was",
        "the result was actually", "we measured", "the outcome",
    ],
    "relevance": [
        "that does relate", "this connects to", "the reason i mentioned",
        "the question was asking", "i was addressing", "what i was trying to say",
        "my point was", "in the context of the question",
    ],
    "fluency": [
        "i was nervous", "i stumbled", "i meant to say", "i was hesitant",
        "i said 'um'", "i was thinking", "that came out wrong",
    ],
    "grammar": [
        "i know the correct term", "i misspoke", "i meant the word",
        "the right phrase is", "technically the term",
    ],
}

def _classify_dimension(clarification: str) -> str:
    """
    Classify which rubric dimension the candidate's clarification addresses.
    Returns the dimension with the most signal word hits.
    Defaults to "depth" (most common dispute target) on tie.
    """
    cl = clarification.lower()
    scores = {dim: 0 for dim in _DIMENSION_SIGNALS}
    for dim, signals in _DIMENSION_SIGNALS.items():
        for sig in signals:
            if sig in cl:
                scores[dim] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "depth"


# ══════════════════════════════════════════════════════════════════════════════
#  DISPUTE RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisputeRecord:
    id:               str
    timestamp:        str
    question_type:    str           # "technical" | "behavioural" | "hr"
    question:         str
    original_answer:  str
    clarification:    str           # candidate's dispute text (their reasoning chain)
    original_score:   float         # RubricAgent's initial score
    revised_score:    float         # agreed score after dialogue
    revision_delta:   float         # revised − original
    dimension_blamed: str           # rubric dimension that was underscored
    rubric_reasoning: str           # RubricAgent's original CoT
    dialogue_turns:   int           # turns it took to resolve the dispute
    few_shot_example: str = ""      # pre-formatted string injected into RubricAgent

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _build_few_shot_example(record: DisputeRecord) -> str:
    """
    Format a dispute record as a few-shot example for the RubricAgent prompt.

    Wei et al. (2022) — include the reasoning chain (clarification), not just
    the input→output pair, for maximum few-shot transfer.
    """
    return (
        f"--- PAST SCORING CORRECTION ({record.question_type.upper()}) ---\n"
        f"Answer excerpt: \"{record.original_answer[:200].strip()}...\"\n"
        f"Initial score:  {record.original_score:.1f}/5  ← was too low\n"
        f"Candidate clarified: \"{record.clarification[:200].strip()}\"\n"
        f"Correct score:  {record.revised_score:.1f}/5\n"
        f"Dimension underscored: {record.dimension_blamed}\n"
        f"Lesson: {_lesson_from_dimension(record.dimension_blamed, record.clarification)}\n"
        f"--- END CORRECTION ---"
    )


def _lesson_from_dimension(dimension: str, clarification: str) -> str:
    """
    Generate a one-line lesson statement for the few-shot example.
    Specific lessons transfer better than generic ones (Guo et al. 2023).
    """
    lessons = {
        "star": (
            "Collective framing ('we did X') does not imply lack of individual ownership. "
            "Ask whether the candidate clarified a specific personal action within the collective effort."
        ),
        "depth": (
            "Domain knowledge may be present but expressed implicitly or with different terminology. "
            "A candidate who explains the mechanism in plain language may have deep knowledge."
        ),
        "relevance": (
            "An answer that sounds tangential may still address the core competency being assessed. "
            "Check if the underlying experience maps to the question's intent, not just its surface framing."
        ),
        "fluency": (
            "Nervousness-induced disfluency should not be conflated with poor knowledge. "
            "A candidate who stumbles on delivery may have scored much higher on depth if calm."
        ),
        "grammar": (
            "Terminology errors (malapropisms, non-native speaker slips) should not penalise "
            "demonstrated understanding of the underlying concept."
        ),
    }
    return lessons.get(dimension, "Review this question type for systematic underscoring.")


# ══════════════════════════════════════════════════════════════════════════════
#  CORPUS MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class DisputeCorpus:
    """
    Manages the dispute corpus — appends new records to disk, loads on startup,
    and provides few-shot retrieval for RubricAgent.

    Thread-safety: append operations use file append mode (atomic on Linux for
    small writes). For multi-worker deployments, wrap _append in a file lock
    or use Redis. Single-process FastAPI workers are safe as-is.

    Retrieval strategy: return the K most recent records for the matching
    question_type, biased toward the dimension_blamed matching the current
    question. Recency bias is intentional — newer disputes reflect the
    current model's failure modes, which are more actionable than old ones.
    """

    def __init__(self, path: str = DISPUTE_CORPUS_PATH) -> None:
        self._path   = path
        self._records: List[DisputeRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing records from disk into memory on startup."""
        p = Path(self._path)
        if not p.exists():
            logger.info(f"[DisputeCorpus] No existing corpus at {self._path} — starting fresh.")
            return
        loaded = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._records.append(DisputeRecord(**d))
                    loaded += 1
                except Exception as e:
                    logger.warning(f"[DisputeCorpus] Skipping malformed record: {e}")
        # Keep memory cache bounded
        if len(self._records) > _MAX_CORPUS_MEM:
            self._records = self._records[-_MAX_CORPUS_MEM:]
        logger.info(f"[DisputeCorpus] Loaded {loaded} dispute records.")

    def _append(self, record: DisputeRecord) -> None:
        """Append one record to the JSON-lines file."""
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(record.to_jsonl() + "\n")
        except Exception as e:
            logger.error(f"[DisputeCorpus] Failed to persist record {record.id[:8]}: {e}")

    def record_dispute(
        self,
        question_type:    str,
        question:         str,
        original_answer:  str,
        clarification:    str,
        original_score:   float,
        revised_score:    float,
        rubric_reasoning: str,
        dialogue_turns:   int,
    ) -> Optional[DisputeRecord]:
        """
        Record a successful dispute as a training example.

        Called by DialogicFeedbackEngine.close() when score_revised=True
        and revision_delta > _MIN_DELTA.

        Returns the DisputeRecord if accepted, None if filtered out.

        Filters applied:
          - revision_delta < _MIN_DELTA → noise, skip
          - revision_delta < 0 → downward revision, skip (candidate over-claimed)
          - answer word count < _MIN_WORDS → too short for useful example
        """
        delta = round(revised_score - original_score, 3)

        # Filter: only upward revisions above minimum threshold
        if delta < _MIN_DELTA:
            logger.debug(
                f"[DisputeCorpus] Dispute delta {delta:.2f} below threshold "
                f"{_MIN_DELTA} — not recorded."
            )
            return None

        # Filter: answer must be substantive
        if len(original_answer.split()) < _MIN_WORDS:
            logger.debug("[DisputeCorpus] Answer too short — not recorded.")
            return None

        dimension = _classify_dimension(clarification)

        record = DisputeRecord(
            id               = str(uuid.uuid4()),
            timestamp        = datetime.now(timezone.utc).isoformat(),
            question_type    = question_type,
            question         = question,
            original_answer  = original_answer,
            clarification    = clarification,
            original_score   = round(original_score, 3),
            revised_score    = round(revised_score, 3),
            revision_delta   = delta,
            dimension_blamed = dimension,
            rubric_reasoning = rubric_reasoning,
            dialogue_turns   = dialogue_turns,
        )
        record.few_shot_example = _build_few_shot_example(record)

        # Persist to disk first, then update memory cache
        self._append(record)
        self._records.append(record)
        if len(self._records) > _MAX_CORPUS_MEM:
            self._records = self._records[-_MAX_CORPUS_MEM:]

        logger.info(
            f"[DisputeCorpus] Recorded dispute {record.id[:8]} | "
            f"type={question_type} | dim={dimension} | "
            f"delta=+{delta:.2f} | corpus_size={len(self._records)}"
        )
        return record

    def get_few_shot_examples(
        self,
        question_type: str,
        k: int = 3,
        prefer_dimension: Optional[str] = None,
    ) -> List[str]:
        """
        Return up to k few-shot example strings for injection into RubricAgent.

        Retrieval strategy:
          1. Filter to matching question_type
          2. If prefer_dimension is given, sort matching-dimension records first
          3. Take k most recent (recency bias — newest failures most relevant)
          4. Return their pre-formatted few_shot_example strings

        Returns [] if corpus has no relevant examples (safe — RubricAgent
        prompt falls back to zero-shot when the list is empty).

        Parameters
        ----------
        question_type     : "technical" | "behavioural" | "hr"
        k                 : max examples to return (default 3)
        prefer_dimension  : if known, prioritise examples for this dimension
        """
        if not self._records:
            return []

        matching = [r for r in self._records if r.question_type == question_type]
        if not matching:
            # Fall back to cross-type examples if none match
            matching = list(self._records)

        # Sort: matching dimension first, then by recency (newest last → take tail)
        if prefer_dimension:
            matching.sort(
                key=lambda r: (
                    0 if r.dimension_blamed == prefer_dimension else 1,
                    r.timestamp,
                )
            )
        else:
            matching.sort(key=lambda r: r.timestamp)

        selected = matching[-min(k, _MAX_FEW_SHOT):]
        return [r.few_shot_example for r in selected if r.few_shot_example]

    def stats(self) -> Dict:
        """Return corpus statistics for the /health endpoint and paper reporting."""
        if not self._records:
            return {"total": 0, "by_type": {}, "by_dimension": {}, "mean_delta": 0.0}

        by_type: Dict[str, int] = {}
        by_dim:  Dict[str, int] = {}
        deltas = []

        for r in self._records:
            by_type[r.question_type]    = by_type.get(r.question_type, 0) + 1
            by_dim[r.dimension_blamed]  = by_dim.get(r.dimension_blamed, 0) + 1
            deltas.append(r.revision_delta)

        return {
            "total":         len(self._records),
            "by_type":       by_type,
            "by_dimension":  by_dim,
            "mean_delta":    round(sum(deltas) / len(deltas), 3),
            "corpus_path":   self._path,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DAILY CHALLENGE QUESTION BANK
#  Seeded by date so every user gets the same question each day.
#  Questions are role-agnostic so they work for any candidate.
#  Add more entries freely — the date seed cycles through the full list.
# ══════════════════════════════════════════════════════════════════════════════

_DAILY_QUESTIONS: List[Dict] = [
    {"question": "Tell me about a time you had to deliver difficult feedback to a colleague. How did you approach it and what was the outcome?", "type": "behavioural", "difficulty": "medium", "keywords": ["feedback", "communication", "outcome", "empathy"]},
    {"question": "Describe a project where you had to work with incomplete information. How did you make decisions under uncertainty?", "type": "behavioural", "difficulty": "medium", "keywords": ["uncertainty", "decision", "risk", "judgment"]},
    {"question": "Walk me through a time you failed at something significant. What did you learn and how did it change your approach?", "type": "behavioural", "difficulty": "medium", "keywords": ["failure", "learning", "resilience", "growth"]},
    {"question": "Tell me about a situation where you had to influence someone without direct authority over them.", "type": "behavioural", "difficulty": "hard", "keywords": ["influence", "leadership", "collaboration", "persuasion"]},
    {"question": "Describe a time you had to prioritise ruthlessly when everything felt urgent. How did you decide what to do first?", "type": "behavioural", "difficulty": "medium", "keywords": ["prioritisation", "time management", "trade-offs", "focus"]},
    {"question": "Tell me about a time you disagreed with your manager or a senior stakeholder. How did you handle it?", "type": "behavioural", "difficulty": "hard", "keywords": ["conflict", "courage", "communication", "respect"]},
    {"question": "Describe a situation where you had to learn a completely new skill quickly. What was your approach and how effective was it?", "type": "behavioural", "difficulty": "medium", "keywords": ["learning", "adaptability", "self-development", "speed"]},
    {"question": "Tell me about a time you had to coordinate across multiple teams to get something done. What challenges arose and how did you resolve them?", "type": "behavioural", "difficulty": "hard", "keywords": ["coordination", "cross-functional", "communication", "alignment"]},
    {"question": "Describe a moment when you noticed a process that was broken and fixed it, even though it wasn't your job.", "type": "behavioural", "difficulty": "medium", "keywords": ["ownership", "initiative", "problem-solving", "impact"]},
    {"question": "Tell me about the most complex problem you've solved professionally. Walk me through your thinking process.", "type": "technical", "difficulty": "hard", "keywords": ["complexity", "problem-solving", "analysis", "reasoning"]},
    {"question": "Describe a time you had to make a build-vs-buy decision. What factors did you consider and what did you decide?", "type": "technical", "difficulty": "hard", "keywords": ["architecture", "trade-offs", "cost", "scalability"]},
    {"question": "Walk me through how you would debug a production issue that is affecting 30% of users but has no obvious error logs.", "type": "technical", "difficulty": "hard", "keywords": ["debugging", "production", "systematic", "investigation"]},
    {"question": "Tell me about a technical decision you made that you later regretted. What would you do differently now?", "type": "technical", "difficulty": "medium", "keywords": ["reflection", "technical debt", "learning", "judgment"]},
    {"question": "Describe how you ensure code quality in a fast-moving project where deadlines are tight.", "type": "technical", "difficulty": "medium", "keywords": ["quality", "trade-offs", "process", "standards"]},
    {"question": "Why are you considering a change at this point in your career, and what specifically excites you about this type of role?", "type": "hr", "difficulty": "easy", "keywords": ["motivation", "career", "fit", "direction"]},
    {"question": "Where do you see yourself in three years and how does this role fit into that trajectory?", "type": "hr", "difficulty": "easy", "keywords": ["career goals", "growth", "ambition", "planning"]},
    {"question": "What kind of work environment brings out your best performance? Give me a concrete example.", "type": "hr", "difficulty": "easy", "keywords": ["environment", "culture", "performance", "self-awareness"]},
    {"question": "Tell me about a time you had to adapt to a significant change at work — new leadership, restructuring, or pivot. How did you respond?", "type": "hr", "difficulty": "medium", "keywords": ["change", "adaptability", "resilience", "positivity"]},
    {"question": "What is something you are genuinely not good at yet, and what are you actively doing about it?", "type": "hr", "difficulty": "medium", "keywords": ["self-awareness", "growth", "honesty", "development"]},
    {"question": "Describe a time you took ownership of a project from start to finish. What was the hardest part and how did you push through?", "type": "behavioural", "difficulty": "hard", "keywords": ["ownership", "accountability", "perseverance", "delivery"]},
    {"question": "Tell me about a time you had to say no to a request from a stakeholder. How did you handle the conversation?", "type": "behavioural", "difficulty": "hard", "keywords": ["boundaries", "communication", "stakeholder", "assertiveness"]},
    {"question": "Describe a situation where you had to balance quality with speed. What trade-offs did you make and would you make them again?", "type": "behavioural", "difficulty": "medium", "keywords": ["trade-offs", "quality", "speed", "judgment"]},
    {"question": "Tell me about the most impactful piece of feedback you have ever received. How did it change you?", "type": "hr", "difficulty": "easy", "keywords": ["feedback", "growth", "self-awareness", "impact"]},
    {"question": "Describe a time you built trust with someone who was initially sceptical of you or your work.", "type": "behavioural", "difficulty": "hard", "keywords": ["trust", "credibility", "relationship", "patience"]},
    {"question": "Walk me through a high-stakes presentation or demo you gave. How did you prepare and how did it go?", "type": "behavioural", "difficulty": "medium", "keywords": ["communication", "preparation", "delivery", "stakeholders"]},
    {"question": "Tell me about a time you had to work with someone whose working style was very different from yours.", "type": "behavioural", "difficulty": "medium", "keywords": ["collaboration", "flexibility", "empathy", "outcomes"]},
    {"question": "Describe a time when you had to make a fast decision with significant consequences. How did you think through it?", "type": "behavioural", "difficulty": "hard", "keywords": ["decision-making", "speed", "judgment", "consequences"]},
    {"question": "What achievement in your career are you most proud of? Why that one?", "type": "hr", "difficulty": "easy", "keywords": ["achievement", "values", "pride", "impact"]},
]


def get_daily_question() -> Dict:
    """
    Return today's daily challenge question — deterministic for all users.

    Uses date.today() as a seed so:
      - Every user worldwide sees the same question on a given day
      - The question rotates daily through the full bank
      - The same question never appears two days in a row (unless bank < 2 entries)

    Returns a dict with: question, type, difficulty, keywords, date_str, question_index
    """
    from datetime import date as _date
    today    = _date.today()
    seed     = today.year * 10000 + today.month * 100 + today.day
    idx      = seed % len(_DAILY_QUESTIONS)
    q        = dict(_DAILY_QUESTIONS[idx])
    q["date_str"]       = today.isoformat()
    q["question_index"] = idx
    return q


# ── Module-level singleton (imported by dialogic_feedback.py and multi_agent_scorer.py) ──
dispute_corpus = DisputeCorpus()