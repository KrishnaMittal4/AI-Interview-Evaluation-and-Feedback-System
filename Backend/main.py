"""
AI Interview Analyzer — FastAPI Backend (Aura AI Edition)
=========================================================
Routes:
  Legacy (original analyzer UI):
    POST /analyze/text       — text answer → NLP + LLM analysis
    POST /analyze/audio      — audio file  → Whisper → NLP + LLM analysis

  Aura AI frontend routes:
    POST /session/start      — create session, generate first question via Groq
    POST /transcribe         — audio file  → Whisper transcript only
    POST /evaluate           — score answer (NLP + Groq), return DISC/STAR/RL hint
    POST /next_question      — RL-guided next question generation
    POST /report             — final session report + HR recommendation + resume gap analysis

  Utility:
    GET  /                   — health ping
    GET  /health             — detailed health check
    GET  /questions          — static question bank
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import queue

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from io import BytesIO

from analyzer import InterviewAnalyzer, compute_coherence_report
from webcam_analyzer import WebcamNervousnessAnalyzer
from acoustic_nervousness import acoustic_analyser, AcousticBaseline   # tiered analyser singleton
from unified_voice_pipeline import UnifiedVoicePipeline, TORCH_OK
from conflict_detector import detect_conflicts_dict
from dialogic_feedback import dialogic_engine
from multi_agent_scorer import question_ambiguity_tracker
from dispute_corpus import dispute_corpus, get_daily_question
from auth import (
    init_db, get_current_user, TokenData,
    create_user, authenticate_user, get_user_profile,
    save_interview, get_history, get_history_detail,
    get_user_stats_summary, get_daily_status, save_daily_completion,
    create_token,
    RegisterRequest, LoginRequest, SaveInterviewRequest, DailySubmitRequest,
)
from adaptive_sequencer import (
    RLAdaptiveSequencer,
    ACTIONS              as RL_ACTIONS_OBJ,
    encode_state         as rl_encode_state,
    compute_reward       as rl_compute_reward,
    get_syllabus,
    get_syllabus_weights,
    pick_topic_for_question,
    get_company_pack_info,
    list_company_packs,
    COMPANY_STYLE,
    DEFAULT_COMPANY_PACK,
)
from resume_rephraser_api import (
    extract_text_from_pdf,
    extract_text_from_docx,
    parse_resume,
    rephrase_resume,
    generate_questions as resume_generate_questions,
    score_resume,
    extract_resume_claims,
    analyze_gap,
    GROQ_OK as RESUME_GROQ_OK,
    PYPDF_OK,
    DOCX_OK,
)

_webcam_analyzer = WebcamNervousnessAnalyzer()

# ── Unified Voice Pipeline (CNN+BiLSTM) ───────────────────────────────────────
# Singleton shared by the startup trainer and the /voice/* endpoints.
# acoustic_analyser.set_unified() is called after training so all subsequent
# /evaluate calls automatically use the CNN+BiLSTM model.
_voice_pipeline = UnifiedVoicePipeline()


import asyncio as _asyncio
import threading as _threading
from contextlib import asynccontextmanager as _asynccontextmanager

# ── SSE training log queue ────────────────────────────────────────────────────
# Messages pushed here are streamed to the frontend via /voice/train-stream.
_training_log_queue: queue.Queue = queue.Queue(maxsize=500)
_training_active: bool = False
_training_metrics: dict = {}


def _classify_log_level(msg: str) -> str:
    """Classify a training log message for colour-coding on the frontend."""
    msg_lower = msg.lower()
    if any(k in msg_lower for k in ["error", "failed", "❌"]):
        return "error"
    if any(k in msg_lower for k in ["warning", "warn", "⚠"]):
        return "warning"
    if any(k in msg_lower for k in ["✅", "ready", "done", "complete", "saved"]):
        return "success"
    if any(k in msg_lower for k in ["epoch", "fold", "val=", "train=", "test="]):
        return "metric"
    if any(k in msg_lower for k in ["downloading", "loading", "kaggle", "crema", "tess"]):
        return "download"
    return "info"


def _background_pipeline_setup() -> None:
    """
    Train or load the CNN+BiLSTM voice model in a background thread.
    All progress messages are pushed to _training_log_queue so the
    /voice/train-stream SSE endpoint can relay them to the frontend.
    """
    global _training_active, _training_metrics
    import logging as _log
    _logger = _log.getLogger("voice_pipeline_startup")

    def _push(msg: str, level: str = "info", extra: dict = None) -> None:
        """Push a structured log event to the SSE queue."""
        payload = {"msg": msg, "level": level, "ts": __import__("time").time()}
        if extra:
            payload.update(extra)
        _logger.info(msg)
        try:
            _training_log_queue.put_nowait(payload)
        except queue.Full:
            pass  # Drop if queue is full — non-blocking

    _training_active = True
    _push("🚀 Background pipeline setup starting…", "start")

    try:
        metrics = _voice_pipeline.setup(
            force_retrain=False,
            max_per_dataset=3000,
            progress_cb=lambda m: _push(m, _classify_log_level(m)),
        )
        acoustic_analyser.set_unified(_voice_pipeline)
        _training_metrics = metrics

        model_type = metrics.get("model_type", "MLP")
        test_acc   = metrics.get("test_accuracy", "?")
        nerv_acc   = metrics.get("nervousness_binary_accuracy", "?")
        _push(
            f"✅ {model_type} ready — test_acc={test_acc}%  nervousness_acc={nerv_acc}%",
            "done",
            {"metrics": metrics},
        )
    except Exception as exc:
        _push(f"❌ Setup failed: {exc}", "error")
    finally:
        _training_active = False
        # Sentinel so SSE client knows the stream is over
        try:
            _training_log_queue.put_nowait({"msg": "__DONE__", "level": "sentinel"})
        except queue.Full:
            pass


@_asynccontextmanager
async def _lifespan(app_: "FastAPI"):  # type: ignore[name-defined]
    """
    FastAPI lifespan: kick off voice pipeline setup in a background thread
    so the server starts accepting requests immediately while training runs.
    Also pre-warms the MediaPipe Tasks model so the first webcam analysis
    request doesn't block while the model downloads.
    """
    # Pre-warm webcam model (no-op if Tasks API unavailable)
    _threading.Thread(
        target=_webcam_analyzer.prewarm, daemon=True, name="webcam_prewarm"
    ).start()

    t = _threading.Thread(target=_background_pipeline_setup, daemon=True, name="voice_setup")
    t.start()

    # Periodic task: purge dialogue sessions older than 1 hour every 30 minutes
    async def _purge_dialogue_sessions():
        while True:
            await _asyncio.sleep(1800)
            removed = dialogic_engine.purge_old(max_age_s=3600.0)
            if removed:
                import logging as _log
                _log.getLogger(__name__).info(
                    f"[lifespan] Purged {removed} expired dialogue session(s)"
                )

    purge_task = _asyncio.ensure_future(_purge_dialogue_sessions())

    yield

    # Graceful shutdown
    purge_task.cancel()
    t.join(timeout=5)


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aura AI — Interview Analyzer API",
    description="Whisper ASR · Groq LLaMA 3.3-70B · RL Adaptive Sequencer · NLP Pipeline",
    version="2.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # FIX 3: Added http://127.0.0.1:3000 and http://localhost:5174 (Vite fallback
    # port when 5173 is busy). Previously requests from these origins were blocked
    # and the frontend silently fell back to mock data — the backend appeared broken.
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = InterviewAnalyzer()

# ── In-memory session store ───────────────────────────────────────────────────
# Stores active session state: role, difficulty, question history, RL state
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ── DISC keyword map (from backend_engine.py) ─────────────────────────────────
DISC_KEYWORDS: Dict[str, List[str]] = {
    "Dominance":         ["lead","decided","took charge","goal","direct","challenge","result","win","fast","control","drove","pushed"],
    "Influence":         ["team","collaborate","communicate","inspire","enthusiasm","motivated","people","fun","support","engaged","presented"],
    "Steadiness":        ["consistent","reliable","patient","support","stable","process","listen","careful","methodical","thorough","steady"],
    "Conscientiousness": ["accurate","detail","quality","process","data","systematic","standard","precise","analysis","metrics","documented"],
}

# ── STAR patterns ─────────────────────────────────────────────────────────────
STAR_PATTERNS: Dict[str, str] = {
    "Situation": r"\b(situation|context|background|when|once|there was|faced|encountered|during|at the time)\b",
    "Task":      r"\b(task|goal|objective|responsible|needed to|had to|assigned|my role|challenge|was asked)\b",
    "Action":    r"\b(i did|i took|i used|implemented|developed|created|decided|solved|built|designed|led|coordinated)\b",
    "Result":    r"\b(result|outcome|achieved|improved|reduced|increased|success|impact|as a result|completed|delivered)\b",
}

# ── RL Action space — sourced from adaptive_sequencer.py ─────────────────────
# RL_ACTIONS_OBJ is the canonical list (imported above). Keep this alias so any
# legacy references inside the file still resolve without changes.
RL_ACTIONS = [
    {"type": a.q_type.capitalize() if a.q_type not in ("hr", "follow_up") else a.q_type.upper() if a.q_type == "hr" else a.q_type,
     "difficulty": a.difficulty,
     "action_idx": a.idx,
     "follow_up":  a.follow_up}
    for a in RL_ACTIONS_OBJ
]

# ── Groq model ────────────────────────────────────────────────────────────────
_GROQ_MODEL = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════════════════════
#  RL SEQUENCER FACTORY
#  Creates one RLAdaptiveSequencer per session and stores it in the session
#  dict under the key "rl_sequencer".  The sequencer handles:
#    • Q-table init with warm-start from shared prior (aura_rl_qtable_shared_{role}.json)
#    • Per-session overlay from individual table  (aura_rl_qtable_{role}.json)
#    • Bellman Q-update after every answer
#    • ε-greedy action selection with follow-up override
#    • Q-table persistence (save + shared-prior blend) at session end
# ══════════════════════════════════════════════════════════════════════════════

def _make_sequencer(role: str) -> RLAdaptiveSequencer:
    """Instantiate and warm-start a sequencer for the given role."""
    seq = RLAdaptiveSequencer(role=role, use_shared_prior=True)
    seq.load()          # loads shared prior then individual overlay (no-op on first run)
    seq.reset_session() # reset per-session counters, keep loaded Q-table
    return seq


def _get_sequencer(session: Dict) -> RLAdaptiveSequencer:
    """Return the sequencer stored in a session dict (create if missing)."""
    if "rl_sequencer" not in session:
        session["rl_sequencer"] = _make_sequencer(session.get("role", "Software Engineer"))
    return session["rl_sequencer"]




# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _score_disc(text: str) -> Dict[str, float]:
    """Score DISC dimensions via keyword matching."""
    text_lower = text.lower()
    scores = {}
    for dim, keywords in DISC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[dim] = round(min(10.0, hits * 1.5), 1)
    return scores


def _score_star(text: str) -> float:
    """Return STAR coverage fraction (0.0–1.0)."""
    text_lower = text.lower()
    covered = sum(
        1 for pattern in STAR_PATTERNS.values()
        if re.search(pattern, text_lower)
    )
    return round(covered / len(STAR_PATTERNS), 2)


def _count_fillers(text: str) -> int:
    filler_list = [
        "um","uh","er","ah","hmm","like","you know","sort of",
        "kind of","basically","literally","i mean","you see",
    ]
    text_lower = text.lower()
    return sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) for w in filler_list)


def _calc_wpm(text: str, duration_sec: float = 60.0) -> int:
    words = len(text.split())
    return int(words / max(duration_sec / 60, 0.1))



# ── No-coding-questions policy ────────────────────────────────────────────────
# This system has no code editor. All generated questions must be conceptual,
# architectural, or experience-based. Write-code / DSA tasks are forbidden.
_NO_CODE_SYSTEM = (
    "CRITICAL — NO CODE EDITOR: This interview platform has NO code editor, NO REPL, and NO whiteboard. "
    "You MUST NEVER generate any question that asks the candidate to write, type, implement, or produce code. "
    "\n\nABSOLUTELY FORBIDDEN question types:"
    "\n- 'Write a function that...'"
    "\n- 'Implement a...'"
    "\n- 'Code a solution for...'"
    "\n- 'Given an array / linked list / string, write...'"
    "\n- Any LeetCode / HackerRank / competitive-programming style problem"
    "\n- 'Write a SQL query that...'"
    "\n- 'Write a script / program / class that...'"
    "\n- Any question where the expected answer IS code"
    "\n\nALL technical questions MUST be spoken-answer format — one of:"
    "\n✓ Conceptual: 'How does X work?' / 'Explain Y'"
    "\n✓ Architectural: 'How would you design a system that...?' / 'What trade-offs would you consider?'"
    "\n✓ Experiential: 'Tell me about a time you dealt with X' / 'Walk me through how you debugged Y in production'"
    "\n✓ Decision-based: 'When would you choose X over Y and why?'"
    "\n\nGood examples: "
    "'Explain how consistent hashing works and when you would use it.', "
    "'Walk me through how you would debug a memory leak in a production service.', "
    "'What are the trade-offs between SQL and NoSQL for a high-write workload?', "
    "'How would you design a rate limiter for a public API — what components would you need?'"
)

async def _generate_question(
    groq_client,
    role: str,
    difficulty: str,
    q_type: str,
    avoid: List[str],
    is_follow_up:  bool = False,
    prev_question: str  = "",
    company_pack:  str  = DEFAULT_COMPANY_PACK,
    topic_hint:    Optional[Dict] = None,
) -> Dict:
    """
    Call Groq to generate a fresh interview question.

    v3.0 additions:
      • company_pack  — injects COMPANY_STYLE[pack]["prompt_style"] into the
                        system prompt so Groq shifts question style (depth, framing,
                        vocabulary) to match the target company archetype.
      • topic_hint    — dict from pick_topic_for_question(); if provided, the
                        prompt instructs Groq to target that specific syllabus topic
                        and one of its subtopics, ensuring full syllabus coverage
                        across a session rather than random topic drift.
    """
    avoid_block = "\n".join(f"- {q}" for q in avoid[-5:]) if avoid else "None"

    # ── Company pack style injection ──────────────────────────────────────────
    pack_info   = get_company_pack_info(company_pack)
    pack_style  = pack_info.get("prompt_style", "")
    system_msg  = _NO_CODE_SYSTEM
    if pack_style:
        system_msg = f"{_NO_CODE_SYSTEM}\n\n{pack_style}"

    # ── Topic hint injection ──────────────────────────────────────────────────
    topic_instruction = ""
    if topic_hint and topic_hint.get("topic_name"):
        subtopics = topic_hint.get("subtopics", [])
        sub_str   = ", ".join(subtopics[:3]) if subtopics else "any relevant subtopic"
        topic_instruction = (
            f"\nTarget topic: {topic_hint['topic_name']} "
            f"(subtopics to draw from: {sub_str})."
            f"\nREMINDER: Ask a spoken conceptual/architectural/experiential question about this topic. "
            f"Do NOT ask the candidate to write or produce any code."
        )

    if is_follow_up:
        prompt = (
            f"Generate ONE follow-up interview probe question for a {role} candidate.\n"
            f"The previous question was: {prev_question}\n"
            f"The candidate gave a shallow or incomplete answer. Probe deeper into the same topic.\n"
            f"Return ONLY valid JSON: "
            f'{{\"question\":\"...\",\"type\":\"{q_type}\",\"difficulty\":\"{difficulty}\",'
            f'\"keywords\":[\"...\"],\"ideal_answer\":\"...\",\"topic\":\"{topic_hint.get("topic_name","") if topic_hint else ""}\"}}'
        )
    else:
        prompt = (
            f"Generate ONE unique {difficulty} {q_type} interview question for a {role} candidate."
            f"{topic_instruction}\n"
            f"Do NOT repeat these questions:\n{avoid_block}\n"
            f"Return ONLY valid JSON: "
            f'{{\"question\":\"...\",\"type\":\"{q_type}\",\"difficulty\":\"{difficulty}\",'
            f'\"keywords\":[\"...\"],\"ideal_answer\":\"...\",\"topic\":\"{topic_hint.get("topic_name","") if topic_hint else ""}\"}}' 
        )

    response = await groq_client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


async def _generate_report(groq_client, role: str, answers: List[Dict]) -> Dict:
    """Generate final session HR report via Groq."""
    avg = sum(a.get("score", 3) for a in answers) / max(len(answers), 1)
    summary = "\n".join(
        f"Q{i+1}: score={a.get('score',3):.1f}, STAR={a.get('star_coverage',0.5):.0%}, "
        f"nervousness={a.get('nervousness',0.2):.0%}"
        for i, a in enumerate(answers)
    )

    prompt = f"""You are a senior HR evaluator. Review this mock interview session for a {role} candidate.

Session summary:
{summary}
Average score: {avg:.2f}/5

Return ONLY valid JSON:
{{
  "hr_recommendation": "<Strong Yes | Yes | Maybe | No>",
  "hr_reasoning": "<2-3 sentence hiring recommendation>",
  "top_strength": "<candidate's biggest strength>",
  "top_weakness": "<biggest area to improve>",
  "overall_coaching": "<one actionable improvement tip for the candidate>"
}}"""

    response = await groq_client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "Aura AI Interview Analyzer API running",
        "version": "2.0.0",
        "routes": [
            "/session/start", "/transcribe", "/evaluate", "/next_question", "/report",
            "/dialogue/open", "/dialogue/turn", "/dialogue/close",
            "/resume/parse", "/resume/rephrase", "/resume/score", "/resume/questions",
            "/resume/analyze",
            "/voice/status", "/voice/train",
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq_connected": analyzer.check_groq_connection(),
        "whisper_ready": analyzer.whisper_ready,
        "webcam_nervousness": True,
        "voice_model": {
            "tier1_ready":  acoustic_analyser.tier1_ready,
            "model_type":   acoustic_analyser.model_type,
            "torch_ok":     TORCH_OK,
            "test_accuracy": _voice_pipeline._metrics.get("test_accuracy", "not_trained"),
            "nervousness_accuracy": _voice_pipeline._metrics.get("nervousness_binary_accuracy", "not_trained"),
        },
        "features": {
            "baseline_calibration": True,    # Feature 5 — POST /baseline
            "coherence_scoring":    True,    # Feature 6 — in /evaluate + /report
        },
        "active_sessions": len(SESSIONS),
        "active_dialogues": len(dialogic_engine._sessions),
        "conflict_detector": "active",
        "model": _GROQ_MODEL,
        "resume_rephraser": {
            "groq_ok":  RESUME_GROQ_OK,
            "pdf_ok":   PYPDF_OK,
            "docx_ok":  DOCX_OK,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES  — /auth/*
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    """Register a new user. Returns { token, user }."""
    if len(req.username.strip()) < 3:
        raise HTTPException(status_code=422, detail="Username must be at least 3 characters.")
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    if "@" not in req.email:
        raise HTTPException(status_code=422, detail="Invalid email address.")
    try:
        user = create_user(req.username, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    token = create_token(user["id"], user["username"])
    return JSONResponse(content={
        "token": token,
        "user":  {"id": user["id"], "username": user["username"], "email": user["email"]},
    })


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """Authenticate with username/email + password. Returns { token, user }."""
    user = authenticate_user(req.username_or_email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")
    token = create_token(user["id"], user["username"])
    safe  = {k: v for k, v in user.items() if k != "password_hash"}
    return JSONResponse(content={"token": token, "user": safe})


@app.get("/auth/me")
async def auth_me(current_user: TokenData = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    profile = get_user_profile(current_user.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found.")
    return JSONResponse(content=profile)


# ══════════════════════════════════════════════════════════════════════════════
#  USER HISTORY ROUTES  — /user/*
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/user/history")
async def user_history(
    limit: int = 20, offset: int = 0,
    current_user: TokenData = Depends(get_current_user),
):
    """Paginated interview history for the authenticated user."""
    rows = get_history(current_user.user_id, limit=min(limit, 100), offset=offset)
    return JSONResponse(content={"history": rows, "total_fetched": len(rows)})


@app.get("/user/history/{record_id}")
async def user_history_detail(
    record_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Full detail for one interview history record."""
    record = get_history_detail(record_id, current_user.user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")
    return JSONResponse(content=record)


@app.get("/user/stats")
async def user_stats(current_user: TokenData = Depends(get_current_user)):
    """Aggregated stats + streak info for the authenticated user's dashboard."""
    stats = get_user_stats_summary(current_user.user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found.")
    return JSONResponse(content=stats)


# ══════════════════════════════════════════════════════════════════════════════
#  DAILY CHALLENGE ROUTES  — /daily/*
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/daily/question")
async def daily_question(current_user: TokenData = Depends(get_current_user)):
    """
    Return today's daily challenge question (same for all users, seeded by date)
    plus the user's current completion status and streak.

    Returns:
      question        : str
      type            : str   (behavioural | technical | hr)
      difficulty      : str
      keywords        : list
      date_str        : str   (ISO date, e.g. "2025-09-14")
      question_index  : int
      completed_today : bool
      daily_score     : float | null
      streak_current  : int
      streak_best     : int
    """
    q      = get_daily_question()
    status = get_daily_status(current_user.user_id)
    return JSONResponse(content={**q, **status})


@app.post("/daily/submit")
async def daily_submit(
    req: DailySubmitRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Submit a completed daily challenge answer.
    The score should come from the standard /evaluate endpoint —
    call /evaluate first, then pass the returned score here.

    Body (JSON):
      date_str : str    — ISO date of the challenge (must match today)
      question : str    — the question text
      score    : float  — score from /evaluate (0–5)
      answer   : str    — candidate's answer text

    Returns:
      new_streak    : int
      streak_best   : int
      streak_broken : bool   (True if streak reset)
      already_done  : bool   (True if duplicate submission)
      xp_earned     : int    (100 base + 50 bonus per streak day, caps at 500)
    """
    from datetime import date as _date
    if req.date_str != _date.today().isoformat():
        raise HTTPException(
            status_code=400,
            detail="date_str must match today's date. You cannot submit for past days.",
        )

    result = save_daily_completion(
        user_id  = current_user.user_id,
        date_str = req.date_str,
        question = req.question,
        score    = req.score,
        answer   = req.answer,
    )

    # XP: 100 base + 50 per streak day (capped at 500 total)
    xp_earned = min(500, 100 + (result["new_streak"] - 1) * 50) if not result["already_done"] else 0
    return JSONResponse(content={**result, "xp_earned": xp_earned})


@app.get("/daily/leaderboard")
async def daily_leaderboard(current_user: TokenData = Depends(get_current_user)):
    """
    Return today's top 10 daily challenge scores across all users.
    Anonymous — shows username and score only.

    Returns:
      date_str    : str
      leaderboard : list of { rank, username, score, streak }
    """
    from datetime import date as _date
    from auth import _db
    today_str = _date.today().isoformat()

    with _db() as conn:
        rows = conn.execute(
            """SELECT u.username, dc.score, u.streak_current
               FROM daily_completions dc
               JOIN users u ON u.id = dc.user_id
               WHERE dc.date_str = ?
               ORDER BY dc.score DESC
               LIMIT 10""",
            (today_str,),
        ).fetchall()

    board = [
        {"rank": i+1, "username": r["username"], "score": r["score"], "streak": r["streak_current"]}
        for i, r in enumerate(rows)
    ]
    return JSONResponse(content={"date_str": today_str, "leaderboard": board})

@app.get("/voice/status")
async def voice_status():
    """
    Current status of the CNN+BiLSTM voice nervousness model.

    Returns training metrics if available (test accuracy, nervousness binary
    accuracy, dataset sources, model type) so the frontend can display them.
    """
    metrics = _voice_pipeline._metrics or {}
    return JSONResponse(content={
        "tier1_ready":              acoustic_analyser.tier1_ready,
        "model_type":               metrics.get("model_type", "not_trained"),
        "torch_available":          TORCH_OK,
        "test_accuracy":            metrics.get("test_accuracy"),
        "val_accuracy":             metrics.get("val_accuracy"),
        "train_accuracy":           metrics.get("train_accuracy"),
        "cv_mean_accuracy":         metrics.get("cv_mean_accuracy"),
        "cv_std_accuracy":          metrics.get("cv_std_accuracy"),
        "nervousness_binary_accuracy": metrics.get("nervousness_binary_accuracy"),
        "nervousness_binary_f1":    metrics.get("nervousness_binary_f1"),
        "n_total":                  metrics.get("n_total"),
        "n_classes":                metrics.get("n_classes"),
        "class_names":              metrics.get("class_names"),
        "datasets":                 metrics.get("datasets"),
        "source":                   metrics.get("source"),
        "seq_len":                  metrics.get("seq_len"),
        "feature_size":             metrics.get("feature_size"),
        "per_class_accuracy":       metrics.get("per_class_accuracy"),
        "dataset_paths":            _voice_pipeline.get_dataset_paths(),
        "dataset_stats":            _voice_pipeline.get_dataset_stats(),
    })


@app.post("/voice/train")
async def voice_train(
    force_retrain: bool = Form(default=False),
    max_per_dataset: int = Form(default=3000),
):
    """
    Manually trigger CNN+BiLSTM training (or reload from disk).

    Body (form-data):
      force_retrain    — True to re-train even if cached weights exist (default False)
      max_per_dataset  — max WAV files to load per dataset, 500–7000 (default 3000)
                         Higher = better accuracy but longer training time.
                         3000 → ~10–15 min on CPU; 7000 → ~30 min on CPU.

    Returns training metrics when done.

    Note: This endpoint runs synchronously (blocks until training finishes).
    For a non-blocking version, use the background startup training which runs
    automatically when the server starts.
    """
    max_per_dataset = max(500, min(7000, max_per_dataset))
    progress_log: List[str] = []

    def _cb(msg: str) -> None:
        progress_log.append(msg)
        print(msg)

    try:
        metrics = await asyncio.to_thread(
            _voice_pipeline.setup,
            force_retrain,
            max_per_dataset,
            _cb,
        )
        # Hot-swap into the tiered acoustic analyser
        acoustic_analyser.set_unified(_voice_pipeline)
        return JSONResponse(content={
            "status":        "trained",
            "model_type":    metrics.get("model_type", "MLP"),
            "test_accuracy": metrics.get("test_accuracy"),
            "nervousness_binary_accuracy": metrics.get("nervousness_binary_accuracy"),
            "n_total":       metrics.get("n_total"),
            "progress_log":  progress_log[-30:],   # last 30 lines
            "metrics":       metrics,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(exc)}")


@app.get("/questions")
async def get_questions():
    """Static question bank (legacy UI compatibility)."""
    return {
        "behavioral": [
            "Tell me about a time you had to handle a difficult team member.",
            "Describe a situation where you failed and what you learned from it.",
            "Give an example of when you showed leadership under pressure.",
            "Tell me about your greatest professional achievement.",
            "How do you handle competing deadlines and priorities?",
        ],
        "technical": [
            # NOTE: All technical questions are conceptual/architectural only.
            # This system has no code editor — write-code questions are excluded.
            "Explain the difference between REST and GraphQL APIs and when you would choose each.",
            "What is the CAP theorem and how does it affect distributed system design?",
            "Walk me through how you would design a URL shortener at scale.",
            "Explain eventual consistency and a real scenario where you'd accept it.",
            "What are the trade-offs between SQL and NoSQL databases in a high-traffic system?",
        ],
        "situational": [
            "How would you handle a production outage affecting 10,000 users?",
            "If your manager asked you to implement something you disagreed with, what would you do?",
            "You have two days to complete a two-week project. What do you do?",
            "A client is angry about a bug in your code. How do you respond?",
            "Your team is behind schedule and morale is low. How do you lead?",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SYLLABUS & COMPANY PACK ROUTES  (v3.0)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/syllabus/{role}")
async def get_role_syllabus(role: str, company_pack: str = DEFAULT_COMPANY_PACK):
    """
    Return the full topic map and company-pack-adjusted weights for a role.

    Query params:
      company_pack — key from COMPANY_STYLE (default: no_pack)

    Returns:
      {
        role, company_pack_info,
        syllabus:         { topic: { weight, q_type, subtopics, resources } },
        adjusted_weights: { topic: normalised_weight_after_pack },
      }

    The frontend uses adjusted_weights to render the syllabus pie/bar chart so
    candidates can see how the company pack shifts focus before starting a session.
    """
    if company_pack not in COMPANY_STYLE:
        company_pack = DEFAULT_COMPANY_PACK

    syllabus  = get_syllabus(role)
    weights   = get_syllabus_weights(role, company_pack)
    pack_info = get_company_pack_info(company_pack)

    return JSONResponse(content={
        "role":             role,
        "company_pack_info": pack_info,
        "syllabus":          syllabus,
        "adjusted_weights":  weights,
    })


@app.get("/syllabus/packs")
async def get_company_packs():
    """
    Return the full list of available company style packs.

    Used by the frontend to populate the "Preparing for…" dropdown
    before the candidate starts a session.

    Returns:
      { packs: [ { key, display_name, description }, ... ] }
    """
    return JSONResponse(content={"packs": list_company_packs()})


# ══════════════════════════════════════════════════════════════════════════════
#  STUDY NOTES ROUTE  (v3.0)
#  Uses Groq (LLaMA 3.3-70B) — same key already used for question generation.
#  The browser calls POST /study/notes → FastAPI calls Groq → streams SSE back
#  so notes appear word-by-word instead of waiting for the full response.
# ══════════════════════════════════════════════════════════════════════════════

class StudyNotesRequest(BaseModel):
    role:         str
    company_pack: str      = "no_pack"
    depth:        str      = "standard"
    topics:       List[str] = []

_NOTES_PACK_EXTRA: Dict[str, str] = {
    "faang":      "Weight system design and leadership principles heavily. Use Amazon Leadership Principles language. Surface what separates good from great answers.",
    "startup":    "Weight ambiguity tolerance, speed, and culture fit. Use lean pragmatic examples. Avoid over-engineering.",
    "consulting": "Weight structured thinking, frameworks (MECE, STAR, issue trees), and stakeholder communication.",
    "fintech":    "Weight reliability, compliance awareness (GDPR, PCI-DSS), and data accuracy.",
    "healthtech": "Weight privacy (HIPAA), patient safety, and regulatory constraints.",
    "no_pack":    "",
}

_NOTES_PACK_LABELS: Dict[str, str] = {
    "no_pack": "Standard", "faang": "FAANG / Big Tech",
    "startup": "Startup",  "consulting": "Consulting",
    "fintech": "Fintech",  "healthtech": "Healthtech",
}

_NOTES_DEPTH_INSTR: Dict[str, str] = {
    "concise":  "Be concise — 2-3 bullet points per section. Only the most critical points.",
    "standard": "Be thorough — 4-6 bullet points per section with clear, memorable explanations.",
    "deep":     "Be comprehensive — 6-8 bullet points per section. Include nuance, edge cases, common interviewer follow-ups, and what separates average from exceptional answers.",
}

def _build_notes_prompt(role: str, pack: str, depth: str, topics: List[str]) -> str:
    """Build a prompt for ALL topics in one call (legacy / fallback path)."""
    pack_label  = _NOTES_PACK_LABELS.get(pack, "Standard")
    pack_extra  = _NOTES_PACK_EXTRA.get(pack, "")
    depth_instr = _NOTES_DEPTH_INSTR.get(depth, _NOTES_DEPTH_INSTR["standard"])
    topics_str  = ", ".join(topics) if topics else "all topics"
    pack_block  = f"\nCompany style: {pack_label}. {pack_extra}" if pack_extra else ""

    return f"""You are an expert interview coach preparing a {role} candidate for a job interview.

Generate structured, practical interview study notes covering these topics: {topics_str}.{pack_block}

{depth_instr}

CRITICAL — NO CODE QUESTIONS: This is a spoken interview system. Never include any question asking the candidate to write, implement, or produce code. All questions must be conceptual, architectural, or experiential — suitable for a verbal spoken answer.

For EACH topic produce exactly these five subsections:

## [Topic Name]

### Key concepts to know
- [concept name]: [one-sentence explanation of what it is and why it matters in interviews]

### Common interview questions
- [question text] *(conceptual / architectural / behavioural / situational)*

### Strong answer patterns
- [specific technique or structure that makes answers stand out for this topic]

### Common mistakes to avoid
- [pitfall]: [why candidates make this mistake and how to avoid it]

### Quick-recall tips
- [memorable mnemonic, acronym, or mental model a candidate can recall under pressure]

---

Be specific, actionable, and direct. Write as if coaching someone the night before their interview."""


def _build_single_topic_prompt(role: str, pack: str, depth: str, topic: str) -> str:
    """
    Build a tightly-scoped prompt for ONE topic only.
    Used by the parallel chunked generation path in /study/notes.
    Produces cleaner, more focused output than the all-topics prompt.
    """
    pack_label  = _NOTES_PACK_LABELS.get(pack, "Standard")
    pack_extra  = _NOTES_PACK_EXTRA.get(pack, "")
    depth_instr = _NOTES_DEPTH_INSTR.get(depth, _NOTES_DEPTH_INSTR["standard"])
    pack_block  = f"\nCompany style: {pack_label}. {pack_extra}" if pack_extra else ""

    return f"""You are an expert interview coach preparing a {role} candidate for a job interview.

Generate structured, practical study notes for EXACTLY ONE topic: {topic}.{pack_block}

{depth_instr}

CRITICAL — NO CODE QUESTIONS: This is a spoken interview system. Never include any question asking the candidate to write, implement, or produce code. All questions must be conceptual, architectural, or experiential.

Produce exactly these five subsections, no more, no less:

## {topic}

### Key concepts to know
- [concept name]: [one-sentence explanation — what it is and why it matters in interviews]

### Common interview questions
- [question text] *(conceptual / architectural / behavioural / situational)*

### Strong answer patterns
- [specific technique or structure that makes answers stand out]

### Common mistakes to avoid
- [pitfall]: [why candidates make this mistake and how to avoid it]

### Quick-recall tips
- [memorable mnemonic, acronym, or mental model for under-pressure recall]

Be specific, actionable, and direct. Output the markdown only — no preamble, no closing remarks."""


@app.post("/study/notes")
async def study_notes(req: StudyNotesRequest):
    """
    Generate role-specific interview study notes via Groq LLaMA 3.3-70B.

    v2: Parallel per-topic generation — each topic gets its own focused Groq call
    fired concurrently via asyncio.gather. Topics are streamed to the frontend
    one by one as they complete, so users see results immediately.

    Request body : { role, company_pack, depth, topics[] }
    Response     : text/event-stream  →  data: <chunk>\n\n  ...  data: [DONE]\n\n
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and restart the server."
            ),
        )

    topics = req.topics or list(get_syllabus(req.role).keys())
    _GROQ_NOTES_SYS = (
        "You are an expert interview coach. Generate structured, practical "
        "spoken-interview study notes. Never include write-code questions — "
        "all questions must be verbal/conceptual format only. "
        "Output ONLY the requested markdown — no preamble, no closing remarks."
    )

    async def _fetch_one_topic(topic: str) -> str:
        """
        Call Groq for a single topic and return the full markdown string.
        Retries once if the required ## heading is missing from the response.
        """
        prompt = _build_single_topic_prompt(req.role, req.company_pack, req.depth, topic)
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       "llama-3.3-70b-versatile",
            "max_tokens":  1200,
            "stream":      False,
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": _GROQ_NOTES_SYS},
                {"role": "user",   "content": prompt},
            ],
        }
        for attempt in range(2):
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Groq {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                # Validate: must contain a ## heading
                if "## " in text:
                    return text
                # Retry once with a correction prompt
                if attempt == 0:
                    payload["messages"].append({"role": "assistant", "content": text})
                    payload["messages"].append({
                        "role": "user",
                        "content": (
                            f"Your response is missing the required ## {topic} heading. "
                            "Please regenerate with the exact markdown structure requested."
                        ),
                    })
        return text  # return best effort after 2 attempts

    async def stream_parallel():
        """
        Fire all topic calls in parallel, then stream results to the client
        in original topic order as each task completes.
        """
        # Create all tasks simultaneously — Groq processes them in parallel
        tasks = [asyncio.create_task(_fetch_one_topic(t)) for t in topics]
        try:
            for i, task in enumerate(tasks):
                try:
                    topic_md = await task
                    # Stream line by line, escaping newlines for SSE framing
                    for line in topic_md.split("\n"):
                        safe = line.replace("\n", "\\n")
                        yield f"data: {safe}\\n\n"
                    # Separator between topics
                    yield "data: \\n\n"
                    yield "data: ---\\n\n"
                    yield "data: \\n\n"
                except Exception as exc:
                    topic_name = topics[i] if i < len(topics) else "unknown"
                    err_line = f"ERROR fetching {topic_name}: {str(exc)[:120]}"
                    yield f"data: {err_line}\\n\n"
        except Exception as exc:
            yield f"data: ERROR: {str(exc)[:200]}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_parallel(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STUDY NOTES — PDF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class NotesPdfRequest(BaseModel):
    role:       str
    pack_label: str       = "Standard"
    depth:      str       = "Standard"
    notes:      List[Dict]           # list of {title, subsections:[{heading, items:[str]}]}
    confidence: Dict[str, int] = {}  # topic title → 0-5 confidence rating from localStorage


@app.post("/study/notes/pdf")
async def study_notes_pdf(req: NotesPdfRequest):
    """
    Render the parsed study notes as a downloadable PDF.

    v2: now accepts optional `confidence` dict {topic: 0-5} from localStorage
    so the PDF can show per-topic confidence ratings alongside each section.

    Request body : { role, pack_label, depth, notes[], confidence? }
    Response     : application/pdf  (inline download)
    """
    try:
        from notes_pdf import generate_notes_pdf
        pdf_bytes = generate_notes_pdf(
            notes=req.notes,
            role=req.role,
            pack_label=req.pack_label,
            depth=req.depth,
            confidence=req.confidence or {},
        )
        filename = req.role.replace(" ", "_") + "_Interview_Notes.pdf"
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="notes_pdf module not found. Make sure notes_pdf.py is in the backend folder.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(exc)}")


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY ROUTES (original analyzer UI — kept for backwards compatibility)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze/text")
async def analyze_text(
    answer: str = Form(...),
    question: str = Form(...),
    question_type: str = Form(default="behavioral"),
):
    if not answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    start = time.time()
    result = await analyzer.analyze(transcript=answer, question=question, question_type=question_type)
    result["processing_time_ms"] = round((time.time() - start) * 1000)
    return JSONResponse(content=result)


@app.post("/analyze/audio")
async def analyze_audio(
    audio: UploadFile = File(...),
    question: str = Form(...),
    question_type: str = Form(default="behavioral"),
):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        start = time.time()
        transcript = await analyzer.transcribe(tmp_path)
        result = await analyzer.analyze(
            transcript=transcript,
            question=question,
            question_type=question_type,
            audio_path=tmp_path,    # ← ADD THIS — the same path used for Whisper
        )
        result["transcript"] = transcript
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  AURA AI ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. SESSION START ──────────────────────────────────────────────────────────
@app.post("/session/start")
async def session_start(
    role: str = Form(...),
    difficulty: str = Form(default="medium"),
    num_questions: int = Form(default=5),
    resume_text: str = Form(default=""),
    company_pack: str = Form(default=DEFAULT_COMPANY_PACK),   # v3.0: company style pack
):
    """
    Create a new interview session.
    - Instantiates RLAdaptiveSequencer (warm-starts from shared + individual Q-tables)
    - Calls sequencer.get_first_action() for resume-calibrated Q1 type/difficulty
    - Generates first question via Groq LLaMA 3.3-70B

    v3.0: accepts company_pack (faang | startup | consulting | fintech | healthtech | no_pack).
    The pack shifts question style and topic distribution for the entire session.

    Returns: session_id + first question + rl_first_action + syllabus + adjusted weights
    """
    # Validate company_pack — fall back silently to no_pack if unknown key
    if company_pack not in COMPANY_STYLE:
        company_pack = DEFAULT_COMPANY_PACK

    session_id = str(uuid.uuid4())

    # ── Instantiate and warm-start the real sequencer ─────────────────────────
    seq = _make_sequencer(role)

    # ── Resume-based first-action calibration via sequencer.get_first_action() ─
    # Wrap raw resume text into the dict shape that _parse_experience_difficulty expects
    resume_parsed: Dict[str, Any] = {}
    if resume_text.strip():
        resume_parsed = {"experience": resume_text, "summary": ""}

    first_action = seq.get_first_action(
        resume_parsed      = resume_parsed if resume_parsed else None,
        session_difficulty = difficulty,
    )
    first_diff = first_action.difficulty if first_action.difficulty != "—" else "medium"

    # ── First question type override ───────────────────────────────────────────
    # When the user picks easy / medium / hard the session opens with a
    # Behavioural or HR question (alternating randomly) so candidates always
    # get a warm-up that isn't purely technical.
    # "all" mode leaves the choice entirely to the RL Q-table.
    import random as _random
    if difficulty in ("easy", "medium", "hard"):
        first_type = _random.choice(["Behavioural", "HR"])
        # Mirror difficulty to the diff bucket the sequencer chose
        first_diff = {"easy": "easy", "medium": "medium", "hard": "hard"}.get(difficulty, "medium")
    else:
        # "all" → trust whatever the sequencer recommended
        first_type = "HR" if first_action.q_type == "hr" else first_action.q_type.capitalize()

    # ── v3.0: Pick topic from company-pack-adjusted syllabus distribution ─────
    topic_hint = pick_topic_for_question(role, company_pack)

    # Fallback questions per type (used when Groq is unavailable)
    # Fallback questions — Technical = conceptual/architectural only (no code editor)
    _FALLBACK_Q: Dict[str, str] = {
        "Technical":   f"Walk me through a complex system or architecture you designed as a {role}. What trade-offs did you consider?",
        "Behavioural": f"Tell me about a time you had to handle a difficult situation in a {role} context.",
        "HR":          f"Why do you want to work as a {role} and where do you see yourself in 5 years?",
    }

    # ── Generate first question via Groq ──────────────────────────────────────
    first_question = None
    if analyzer.groq_client:
        try:
            first_question = await _generate_question(
                analyzer.groq_client, role, first_diff, first_type, avoid=[],
                company_pack=company_pack,
                topic_hint=topic_hint,
            )
        except Exception as e:
            print(f"[session/start] Groq question gen failed: {e}")

    if not first_question:
        first_question = {
            "question":    _FALLBACK_Q.get(first_type, _FALLBACK_Q["Technical"]),
            "type":        first_type,
            "difficulty":  first_diff,
            "keywords":    ["experience", "problem-solving"],
            "ideal_answer":"Strong answer covers the context, specific actions taken, and measurable outcomes.",
            "topic":       topic_hint.get("topic_name", ""),
        }

    # ── Initialise session state ───────────────────────────────────────────────
    SESSIONS[session_id] = {
        "role":             role,
        "difficulty":       difficulty,
        "num_questions":    num_questions,
        "resume_text":      resume_text,
        "resume_parsed":    parse_resume(resume_text) if resume_text.strip() else {},
        "company_pack":     company_pack,                    # v3.0: stored for entire session
        "answers":          [],
        "questions_asked":  [first_question.get("question", "")],
        "current_question": first_question,
        "q_index":          0,
        "rl_sequencer":     seq,      # ← real RLAdaptiveSequencer instance
        "rl_hint":          None,     # populated after first evaluate()
        # Feature 5 — pre-session nervousness baseline calibration
        "acoustic_baseline": None,    # AcousticBaseline; set by POST /baseline
        "facial_baseline":   None,    # dict; set by POST /baseline
        "created_at":       time.time(),
    }

    return JSONResponse(content={
        "session_id":       session_id,
        "question":         first_question,
        "role":             role,
        "difficulty":       difficulty,
        "num_questions":    num_questions,
        "rl_first_action":  first_action.label(),              # e.g. "technical/medium"
        "company_pack":     get_company_pack_info(company_pack),  # v3.0: echo pack info
        "syllabus":         get_syllabus(role),                # v3.0: full topic map for UI
        "syllabus_weights": get_syllabus_weights(role, company_pack),  # v3.0: adjusted weights
    })


# ── 1b. PRE-SESSION NERVOUSNESS BASELINE ─────────────────────────────────────

@app.post("/baseline")
async def capture_baseline(
    session_id:   str   = Form(...),
    audio:        bytes = File(b""),
    audio_suffix: str   = Form(default=".webm"),
    frames:       str   = Form(default="[]"),   # JSON array of base64 JPEG strings
    fps:          float = Form(default=0.0),
):
    """
    Capture the candidate's personal acoustic and facial nervousness baseline
    before the interview begins.

    WHEN TO CALL
    ------------
    Call this endpoint ONCE after /session/start and BEFORE the first
    /evaluate. The frontend should:
      1. Ask the candidate to read a neutral sentence aloud for ~20–30 seconds
         while the webcam captures frames at the normal capture rate.
      2. POST the audio bytes and webcam frames here.
      3. Wait for the response (validation feedback) before starting Q1.

    The captured baseline is stored in the session and automatically applied
    to all subsequent /evaluate calls via delta normalisation:
        corrected_nervousness = raw_nervousness × (1 − correction_factor)
    where correction_factor reflects how far the candidate's natural speech
    deviates from population means (Kappen et al. 2024 §4.1).

    Body (multipart form-data)
    --------------------------
    session_id   : str   — active session ID from /session/start
    audio        : bytes — baseline recording (WebM/WAV/OGG, ≥15 s recommended)
    audio_suffix : str   — file extension hint (".webm" | ".wav" | ".ogg")
    frames       : str   — JSON array of base64 JPEG webcam frames (8–15 frames)
    fps          : float — webcam capture rate (0.0 = default 0.5 fps)

    Returns
    -------
    {
        "session_id"        : str,
        "acoustic_baseline" : { valid, duration_sec, method, f0_score, … },
        "facial_baseline"   : { valid, blink_rate_bpm, ear_variance, … },
        "calibrated"        : bool,   # True if at least acoustic OR facial is valid
        "warnings"          : list[str],   # user-friendly validation messages
    }

    Graceful degradation
    --------------------
    If audio is absent or too short, acoustic_baseline.valid = False and
    population means are used unchanged for that channel (existing behaviour).
    Same for webcam frames. Partial calibration (one channel valid) still
    provides partial normalisation benefit.
    """
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    warnings = []

    # ── Acoustic baseline ─────────────────────────────────────────────────────
    acoustic_bl: AcousticBaseline
    if audio:
        acoustic_bl = await asyncio.to_thread(
            acoustic_analyser._tier2.extract_baseline_bytes, audio, audio_suffix
        )
        if not acoustic_bl.valid:
            warnings.append(
                f"Acoustic baseline rejected: {acoustic_bl.method}. "
                "Please record at least 15 seconds of speech for accurate calibration."
            )
    else:
        acoustic_bl = AcousticBaseline(valid=False, method="no_audio_provided")
        warnings.append(
            "No audio provided for acoustic baseline. "
            "Voice nervousness scores will use population averages."
        )

    session["acoustic_baseline"] = acoustic_bl

    # ── Facial baseline ───────────────────────────────────────────────────────
    facial_bl: dict = {"valid": False, "reason": "no_frames_provided"}
    try:
        frame_list = json.loads(frames)
    except Exception:
        frame_list = []

    if frame_list:
        facial_bl = await asyncio.to_thread(
            _webcam_analyzer.calibrate_facial_baseline, frame_list, fps
        )
        if not facial_bl.get("valid"):
            warnings.append(
                f"Facial baseline insufficient: {facial_bl.get('reason', 'unknown')}. "
                "Ensure the webcam is active and your face is visible during calibration."
            )
    else:
        warnings.append(
            "No webcam frames provided for facial baseline. "
            "Facial nervousness scores will use population averages."
        )

    session["facial_baseline"] = facial_bl

    calibrated = acoustic_bl.valid or facial_bl.get("valid", False)

    return JSONResponse(content={
        "session_id":        session_id,
        "acoustic_baseline": acoustic_bl.to_dict(),
        "facial_baseline":   facial_bl,
        "calibrated":        calibrated,
        "warnings":          warnings,
    })


# ── 2. TRANSCRIBE ─────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    session_id: str = Form(default=""),
):
    """
    Audio file → Whisper transcript (Groq Whisper large-v3).

    Additionally runs the CNN+BiLSTM voice nervousness model on the same
    audio bytes (if the model is ready) and caches the result in the session
    under "last_voice_result".  The /evaluate endpoint reads this cache so
    the voice score uses real acoustic features rather than the text proxy.
    """
    suffix = ".webm"
    if audio.filename:
        ext = audio.filename.rsplit(".", 1)[-1]
        suffix = f".{ext}"

    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        transcript = await analyzer.transcribe(tmp_path)

        # ── Voice nervousness (CNN+BiLSTM or handcrafted fallback) ────────────
        voice_result: Dict = {}
        try:
            voice_result = await asyncio.to_thread(
                acoustic_analyser.analyse, tmp_path
            )
        except Exception as _ve:
            print(f"[transcribe] Voice model failed: {_ve}")

        # Cache in session so /evaluate can read it
        if session_id and session_id in SESSIONS:
            SESSIONS[session_id]["last_voice_result"] = voice_result

        return JSONResponse(content={
            "transcript":   transcript,
            "word_count":   len(transcript.split()),
            "voice_result": voice_result,   # includes nervousness_score, emotions, method
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── 3. EVALUATE ───────────────────────────────────────────────────────────────
@app.post("/evaluate")
async def evaluate_answer(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    q_index: int = Form(default=0),
    role: str = Form(default="Software Engineer"),
    difficulty: str = Form(default="medium"),
    answer_time_sec: float = Form(default=60.0),
    webcam_frames: str = Form(default=""),   # JSON array of base64 JPEG strings
    cultural_context: str = Form(default="auto"),  # "auto" | "high-context" | "low-context"
):
    """
    Score a candidate's answer using the full research-grounded pipeline.

    Scoring pipeline (v2.0):
      • analyzer.analyze() runs:
          - Groq HR feedback + ideal answer generation
          - Groq semantic relevance score (answer_evaluator.py v12.1)
          - Type-aware NLP composite (STAR/DISC/OCEAN/depth/fluency/word_cat)
          - Facial nervousness v2.0 (AU + blink + gaze + pose + EAR)
          - Voice nervousness proxy (filler rate + weak language)
          - Nervousness fusion: 0.65×voice + 0.35×facial (Schuller 2011)
          - Session aggregation: knowledge×0.70 + emotion×0.15 + voice×0.15

    Returns: full scoring dict + nervousness detail + rl_hint
    """
    if not answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")

    session = SESSIONS.get(session_id, {})

    # ── Determine question type ────────────────────────────────────────────────
    current_q  = session.get("current_question", {})
    q_type_raw = current_q.get("type", "Technical") if isinstance(current_q, dict) else "Technical"
    # "Technical" here means conceptual/architectural — no code editor, no write-code Qs
    q_type_map = {"Technical": "technical", "Behavioural": "behavioural", "HR": "hr", "Conceptual": "technical", "System Design": "technical"}
    q_type     = q_type_map.get(q_type_raw, "technical")

    keywords = current_q.get("keywords", []) if isinstance(current_q, dict) else []

    # ── Webcam analysis ────────────────────────────────────────────────────────
    webcam_result: dict = {}
    # Facial nervousness v2.0 sub-inputs (calm defaults used when no webcam)
    blink_rate_per_min:   float = 15.0
    gaze_contact_ratio:   float = 0.70
    ear_time_series:      list  = []
    yaw_angles:           list  = []
    pitch_angles:         list  = []
    inter_blink_intervals: list = []

    if webcam_frames.strip():
        try:
            frame_list = json.loads(webcam_frames)
            # FIX 5: json.loads("[]") succeeds and returns an empty list — skip
            # analysis in that case too (would return a meaningless neutral score).
            if isinstance(frame_list, list) and len(frame_list) > 0:
                webcam_result = _webcam_analyzer.analyze_frames(frame_list, fps=0.0)  # uses DEFAULT_FPS
                # Feature 5 — apply per-session facial baseline correction if available
                facial_baseline = session.get("facial_baseline") if session else None
                if facial_baseline and facial_baseline.get("valid"):
                    webcam_result = _webcam_analyzer.apply_facial_baseline_correction(
                        webcam_result, facial_baseline
                    )
                # Extract raw time-series exposed by webcam_analyzer v2.0
                blink_rate_per_min    = webcam_result.get("blink_rate_per_min", 15.0)
                gaze_contact_ratio    = max(
                    0.0, 1.0 - webcam_result.get("gaze_aversion_rate", 0.30)
                )
                ear_time_series       = webcam_result.get("ear_values", [])
                yaw_angles            = webcam_result.get("yaw_values", [])
                pitch_angles          = webcam_result.get("pitch_values", [])
                inter_blink_intervals = webcam_result.get("inter_blink_intervals", [])
        except Exception as e:
            print(f"[evaluate] Webcam analysis failed: {e}")

    start = time.time()

    # ── Read CNN+BiLSTM voice result cached by /transcribe ────────────────────
    # When the frontend records audio and calls /transcribe first, the voice
    # nervousness is already computed from real acoustics and stored here.
    # We pass the audio_path into analyzer.analyze() so it can also run
    # acoustic_analyser.analyse() as a secondary path if needed.
    cached_voice = session.get("last_voice_result", {}) if session else {}
    # Clear the cache so we don't reuse a stale result on the next question
    if session:
        session.pop("last_voice_result", None)

    # ── Full analysis — all nervousness handled inside analyzer.analyze() ──────
    analysis = await analyzer.analyze(
        transcript             = answer,
        question               = question,
        question_type          = q_type,
        keywords               = keywords,
        difficulty             = difficulty,
        duration_s             = answer_time_sec,
        # Facial nervousness v2.0 inputs
        au_intensities         = None,
        blink_rate_per_min     = blink_rate_per_min,
        inter_blink_intervals  = inter_blink_intervals,
        gaze_contact_ratio     = gaze_contact_ratio,
        gaze_direction_std_deg = 0.0,
        yaw_angles             = yaw_angles,
        pitch_angles           = pitch_angles,
        ear_time_series        = ear_time_series,
        cultural_context       = cultural_context,
    )

    # ── Override voice nervousness with CNN+BiLSTM score if available ─────────
    # The text-proxy inside analyzer.analyze() runs on transcript only; the
    # CNN+BiLSTM score comes from real acoustic features of the recording.
    # Fuse: 60% acoustic (CNN+BiLSTM) + 40% text proxy when both available.
    nervousness_d = analysis.get("nervousness", {})
    if cached_voice.get("available"):
        acoustic_score = float(cached_voice["nervousness_score"])
        # Feature 5 — apply per-session acoustic baseline correction if available
        acoustic_baseline = session.get("acoustic_baseline") if session else None
        if acoustic_baseline and getattr(acoustic_baseline, "valid", False):
            corrected = acoustic_analyser.analyse_with_baseline(
                cached_voice.get("_audio_path", ""),
                baseline=acoustic_baseline,
            )
            # If we have a corrected score use it; otherwise fall back to scalar correction
            if corrected.get("baseline_corrected") and corrected.get("nervousness_score"):
                acoustic_score = float(corrected["nervousness_score"])
            else:
                # Scalar fallback: apply correction factor directly on the cached score
                from acoustic_nervousness import (
                    _ACOUSTIC_WEIGHTS, _POPULATION_BASELINE, _BASELINE_MAX_CORRECTION, _clamp
                )
                bl = acoustic_baseline
                baseline_fused = sum(
                    getattr(bl, f, _POPULATION_BASELINE[f]) * w
                    for f, w in [
                        ("f0_score", _ACOUSTIC_WEIGHTS["f0"]),
                        ("mfcc_score", _ACOUSTIC_WEIGHTS["mfcc"]),
                        ("spectral_flux_score", _ACOUSTIC_WEIGHTS["spectral_flux"]),
                        ("energy_score", _ACOUSTIC_WEIGHTS["energy"]),
                        ("pause_score", _ACOUSTIC_WEIGHTS["pause"]),
                        ("rate_score", _ACOUSTIC_WEIGHTS["rate"]),
                    ]
                )
                correction = _clamp(
                    (baseline_fused / 0.35) - 1.0,
                    -_BASELINE_MAX_CORRECTION, _BASELINE_MAX_CORRECTION,
                )
                acoustic_score = round(_clamp(acoustic_score * (1.0 - correction)), 4)
            nervousness_d["baseline_corrected"] = True
            nervousness_d["baseline_method"]    = getattr(acoustic_baseline, "method", "unknown")

        text_score     = nervousness_d.get("voice", 0.3)
        fused_voice    = round(0.60 * acoustic_score + 0.40 * text_score, 3)
        nervousness_d["voice"]          = fused_voice
        nervousness_d["acoustic_cnn"]   = acoustic_score   # store raw for debug
        nervousness_d["voice_method"]   = cached_voice.get("method", "CNN+BiLSTM")
        nervousness_d["emotions_cnn"]   = cached_voice.get("emotions", {})
        nervousness_d["dominant_cnn"]   = cached_voice.get("dominant_emotion", "Neutral")
        nervousness_d["window_nervousness"] = cached_voice.get("window_nervousness", [])
        # Recompute fused nervousness with the updated voice score
        facial_score = nervousness_d.get("facial", 0.0)
        fusion_w     = nervousness_d.get("fusion_weights", {"voice": 0.65, "facial": 0.35})
        nervousness_d["fused"] = round(
            float(fusion_w.get("voice", 0.65)) * fused_voice +
            float(fusion_w.get("facial", 0.35)) * facial_score,
            3,
        )
        analysis["nervousness"] = nervousness_d

    # ── Extract scores ─────────────────────────────────────────────────────────
    scores         = analysis.get("scores", {})
    nervousness_d  = analysis.get("nervousness", {})
    session_scores = analysis.get("session_scores", {})
    nlp_detail     = analysis.get("nlp", {})

    # Primary 1–5 score from ScoreAggregator (knowledge×0.70 + emotion×0.15 + voice×0.15)
    # Bug fix: knowledge_score must NEVER fall back to scores["overall"] — that is the
    # session-aggregated final, not the raw NLP score. Use 3.0 as the neutral default instead.
    # Safe fused nervousness: recompute from components if "fused" key is missing/None
    _fused_raw = nervousness_d.get("fused")
    if _fused_raw is None:
        _v = float(nervousness_d.get("voice") or 0.3)
        _f = float(nervousness_d.get("facial") or 0.0)
        _fw = nervousness_d.get("fusion_weights", {"voice": 0.65, "facial": 0.35})
        _fused_raw = round(float(_fw.get("voice", 0.65)) * _v + float(_fw.get("facial", 0.35)) * _f, 3)
        nervousness_d["fused"] = _fused_raw
    fused_nervousness = float(_fused_raw)
    knowledge_score   = float(scores.get("knowledge_1_5") or 3.0)

    # Bug fix: if session_scores["final"] is missing, recompute from components rather than
    # silently collapsing to scores["overall"] (which equals knowledge — making Overall == Knowledge).
    if session_scores.get("final") is not None:
        overall_score = float(max(0.5, min(5.0, session_scores["final"])))
    else:
        _emotion_fallback = round(max(1.0, min(5.0, (1.0 - fused_nervousness) * 5.0)), 2)
        _voice_fallback   = scores.get("fluency", 3.0)
        overall_score = float(max(0.5, min(5.0, round(
            knowledge_score * 0.70 +
            _emotion_fallback * 0.15 +
            _voice_fallback   * 0.15,
            2,
        ))))

    # ── Legacy helpers (for session storage + RL) ──────────────────────────────
    disc       = _score_disc(answer)
    star_cov   = _score_star(answer)
    filler_cnt = nlp_detail.get("fillers", {}).get("total_count", _count_fillers(answer))
    wpm        = _calc_wpm(answer, answer_time_sec)

    # ── RL: Bellman update + next-action selection ────────────────────────────
    # Delegate entirely to RLAdaptiveSequencer:
    #   1. compute_reward()  — research-grounded reward function
    #   2. update()          — Bellman Q-update for previous (state, action) pair
    #   3. select_action()   — ε-greedy with follow-up override
    #   4. persist rl_hint   — so /next_question can read the recommended type/diff
    rl_hint = None
    if session:
        seq = _get_sequencer(session)
        word_count = len(answer.split())

        # Full reward from adaptive_sequencer.compute_reward()
        # NOTE: curr_action_idx is not yet known (we pick next action after this),
        # so we pass prev_action_idx for both — the sequencer's select_action()
        # will apply the repeat-penalty on the NEXT step when it updates the table.
        reward = rl_compute_reward(
            score            = overall_score,
            prev_score       = seq.last_score,
            nervousness      = fused_nervousness,
            prev_nervousness = seq.last_nervousness,
            star_count       = int(round(star_cov * 4)),   # fraction → count (0-4)
            word_count       = word_count,
            prev_action_idx  = seq.last_action_idx,
            curr_action_idx  = seq.last_action_idx,        # same — repeat penalty deferred
            q_type           = q_type,
        )

        # Bellman update on the Q-table for the just-completed (state, action)
        seq.update(reward=reward)

        # ε-greedy action selection for the NEXT question
        next_action = seq.select_action(
            score       = overall_score,
            nervousness = fused_nervousness,
            star_rate   = star_cov,
            time_eff    = min(100.0, (word_count / max(answer_time_sec, 1)) * 40),
        )

        # Persist hint in session so /next_question reads it without re-running RL
        session["rl_hint"] = {
            "action_idx":  next_action.idx,
            "type":        "HR" if next_action.q_type == "hr" else next_action.q_type.capitalize(),
            "difficulty":  next_action.difficulty,
            "follow_up":   next_action.follow_up,
            "label":       next_action.label(),
            "reward":      round(reward, 4),
            "epsilon":     round(seq.epsilon, 4),
        }
        rl_hint = session["rl_hint"]

    # ── Store answer in session ────────────────────────────────────────────────
    answer_record = {
        "question":           question,
        "answer":             answer,
        "score":              overall_score,
        "knowledge":          round(knowledge_score, 2),
        "star_coverage":      star_cov,
        "disc":               disc,
        "filler_count":       filler_cnt,
        "wpm":                wpm,
        "nervousness":        fused_nervousness,
        "voice_nervousness":  float(nervousness_d.get("voice") or 0.3),
        "facial_nervousness": float(nervousness_d.get("facial") or 0.0),
        "nervousness_level":  nervousness_d.get("level", "Low"),
        "webcam_nervousness": webcam_result,
        "coaching_tip":       analysis.get("ai_evaluation", {}).get("coaching_tip", ""),
        "answer_time_sec":    answer_time_sec,
        "q_index":            q_index,
        "depth_score":        scores.get("depth", 0.0),
        # ── Skill Gap fields ──────────────────────────────────────────────────
        "fluency_score":      scores.get("fluency", 0.0),
        "keyword_score":      round(len(nlp_detail.get("keyword_hits", [])) /
                              max(len(keywords), 1), 3) if nlp_detail.get("keyword_hits")
                              else scores.get("relevance", 50) / 100.0,
        "question_type":      q_type,
        "relevance_score":    scores.get("relevance", 50) / 100.0,
        "grammar_score":      scores.get("clarity", 50) / 100.0,
    }
    if session:
        session.setdefault("answers", []).append(answer_record)

    processing_ms = round((time.time() - start) * 1000)

    return JSONResponse(content={
        # ── Core answer record ─────────────────────────────────────────────────
        **answer_record,

        # ── Full scores breakdown ──────────────────────────────────────────────
        "scores": {
            "overall":       overall_score,
            "knowledge_1_5": knowledge_score,
            "session_final": session_scores.get("final"),
            "confidence":    scores.get("confidence"),
            "clarity":       scores.get("clarity"),
            "structure":     scores.get("structure"),
            "technical":     scores.get("technical"),
            "relevance":     scores.get("relevance"),
            "depth":         scores.get("depth"),
            "fluency":       scores.get("fluency"),
        },
        "session_scores": session_scores,

        # ── Grade ──────────────────────────────────────────────────────────────
        "grade":           analysis.get("grade", "B"),
        "grade_reasoning": analysis.get("grade_reasoning", ""),

        # ── Nervousness v2.0 detail ────────────────────────────────────────────
        "nervousness_detail": {
            "fused":              fused_nervousness,
            "voice":              nervousness_d.get("voice", 0.3),
            "facial":             nervousness_d.get("facial", 0.0),
            "level":              nervousness_d.get("level", "Low"),
            "fusion_weights":     nervousness_d.get("fusion_weights", {"voice": 0.65, "facial": 0.35}),
            "facial_detail":      nervousness_d.get("facial_detail", {}),
            # CNN+BiLSTM acoustic channel (populated when Whisper audio was uploaded)
            "acoustic_cnn":       nervousness_d.get("acoustic_cnn"),
            "voice_method":       nervousness_d.get("voice_method", "text_proxy"),
            "emotions_cnn":       nervousness_d.get("emotions_cnn", {}),
            "dominant_cnn":       nervousness_d.get("dominant_cnn"),
            "window_nervousness": nervousness_d.get("window_nervousness", []),
        },

        # ── NLP detail ─────────────────────────────────────────────────────────
        "nlp_detail": {
            "star_scores":      nlp_detail.get("star_scores", {}),
            "disc_dominant":    nlp_detail.get("disc_dominant"),
            "ocean_scores":     nlp_detail.get("ocean_scores", {}),
            "personality_nlp":  nlp_detail.get("personality_nlp"),
            "hiring_signal":    nlp_detail.get("hiring_signal"),
            "sentiment":        nlp_detail.get("sentiment_intensity"),
            "vocab_diversity":  nlp_detail.get("vocab_diversity"),
            "keyword_hits":     nlp_detail.get("keyword_hits", []),
            "word_category":    nlp_detail.get("word_category", {}),
            "time_data":        nlp_detail.get("time_data", {}),
            "weight_profile":   nlp_detail.get("weight_profile", {}),
            "question_type":    nlp_detail.get("question_type"),
            "word_count":       nlp_detail.get("word_count"),
            "depth_fluency_sc": nlp_detail.get("depth_fluency_sc"),
            "wpm_score":        nlp_detail.get("wpm_score"),
        },

        # ── AI evaluation ──────────────────────────────────────────────────────
        "annotated_transcript": analysis.get("annotated_transcript", answer),
        "key_strengths":        analysis.get("ai_evaluation", {}).get("key_strengths", []),
        "improvement_areas":    analysis.get("ai_evaluation", {}).get("improvement_areas", []),
        "hr_recommendation":    analysis.get("hr_feedback", {}).get("recommendation", "Maybe"),
        "technical_evaluation": analysis.get("ai_evaluation", {}).get("technical_evaluation", ""),
        "ideal_answer":         analysis.get("ai_evaluation", {}).get("ideal_answer", ""),

        # ── Webcam coaching ────────────────────────────────────────────────────
        "webcam_coaching":       webcam_result.get("coaching_note", ""),
        "webcam_interpretation": webcam_result.get("interpretation", ""),

        # ── Conflict detection ─────────────────────────────────────────────────
        # Build a conflict-detector-compatible dict from the internal analysis.
        # The internal analysis["nervousness"] is always a proper sub-dict here.
        "conflict_report": detect_conflicts_dict({
            **analysis,
            # Ensure nervousness sub-dict is always present for _extract_channels.
            # If upstream code somehow flattened nervousness to a float,
            # the explicit nervousness_detail key acts as a fallback.
            "nervousness": nervousness_d,
            "nervousness_detail": nervousness_d,
            "scores": {
                **analysis.get("scores", {}),
                "confidence": scores.get("confidence", 50),
            },
        }),

        # ── RL hint ────────────────────────────────────────────────────────────
        "rl_hint": rl_hint,

        "processing_time_ms": processing_ms,

        # ── Cultural adaptation detail ────────────────────────────────────────────
        "cultural_context":  analysis.get("cultural_context", "low-context"),
        "cultural_detail":   analysis.get("cultural_detail", {}),

        # ── Cross-question coherence (Feature 6) ──────────────────────────────
        # Populated from the 2nd answer onward; null on the first answer.
        "coherence_report": analysis.get("coherence_report", {"available": False}),
    })


# ── 3b. WEBCAM NERVOUSNESS ────────────────────────────────────────────────────
@app.post("/analyze/webcam")
async def analyze_webcam(
    frames: str = Form(...),        # JSON array of base64 JPEG strings
    session_id: str = Form(default=""),
    fps: float = Form(default=0.0), # actual capture rate; 0.0 = use default (0.5 fps)
):
    """
    Standalone webcam nervousness analysis.
    Accepts a JSON array of base64-encoded JPEG frames captured by the frontend
    (recommended: 1 frame every 2 seconds during the candidate's answer).

    Pass fps= if the frontend captures at a different rate than the default
    0.5 fps (1 frame / 2 s) so blink BPM and PERCLOS are calculated accurately.

    Returns:
        nervousness_score (0–100), blink_rate_per_min, eye_stability,
        head_stability, gaze_aversion_rate, facial_asymmetry,
        interpretation, coaching_note
    """
    try:
        frame_list = json.loads(frames)
    except Exception:
        raise HTTPException(status_code=400, detail="frames must be a JSON array of base64 strings")

    if not isinstance(frame_list, list):
        raise HTTPException(status_code=400, detail="frames must be a JSON array")

    result = _webcam_analyzer.analyze_frames(frame_list, fps=fps)
    return JSONResponse(content=result)


# ══════════════════════════════════════════════════════════════════════════════
#  DIALOGIC FEEDBACK ROUTES  (ACM CSCW 2025 — Conversate)
# ══════════════════════════════════════════════════════════════════════════════

class DialogueOpenRequest(BaseModel):
    transcript:      str
    question:        str
    question_type:   str = "behavioral"
    analysis_result: dict   # full /evaluate output


class DialogueTurnRequest(BaseModel):
    dialogue_id: str
    message:     str


@app.post("/dialogue/open")
async def dialogue_open(req: DialogueOpenRequest):
    """
    Open a dialogic feedback session after /evaluate returns its result.
    Returns the first AI coaching message and a dialogue_id for subsequent turns.

    Body (JSON):
      transcript      — candidate's raw answer text
      question        — the interview question that was asked
      question_type   — "technical" | "behavioral" | "situational"
      analysis_result — the full dict returned by /evaluate
    """
    return await dialogic_engine.open_dialogue(
        req.analysis_result, req.transcript, req.question, req.question_type
    )


@app.post("/dialogue/turn")
async def dialogue_turn(req: DialogueTurnRequest):
    """
    Submit one candidate reply and receive the AI's next coaching message.
    The score may be revised up or down (capped at ±1.0 point) if the
    clarification reveals the initial scoring was inaccurate.

    Body (JSON):
      dialogue_id — returned by /dialogue/open
      message     — candidate's reply text

    Returns:
      response, score, score_revised, revision_delta, turns_remaining, closed
    """
    try:
        return await dialogic_engine.advance(req.dialogue_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dialogue '{req.dialogue_id}' not found.")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/dialogue/close")
async def dialogue_close(dialogue_id: str = Form(...)):
    """
    Explicitly close a dialogue session (e.g. candidate clicks 'Done').
    Returns the final agreed score, revision history, and full turn log.

    Body (form-data):
      dialogue_id — the session to close
    """
    try:
        return dialogic_engine.close(dialogue_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dialogue '{dialogue_id}' not found.")


@app.get("/question_quality")
async def question_quality(
    min_obs:  int = 3,
    sort_by:  str = "mean_kappa",
):
    """
    Return the question ambiguity report — which interview questions are
    systematically ambiguous based on inter-agent κ aggregated across sessions.

    Questions are flagged when mean inter-agent κ (RubricAgent vs TraitAgent)
    falls below 0.50 across ≥ min_obs different candidate sessions. Low κ on
    a fixed pair of agents implicates the QUESTION, not the scorers.

    Only available when multi-agent mode is active (AgentOrchestrator.run()).
    In single-agent mode the tracker receives no observations and returns empty.

    Query parameters
    ----------------
    min_obs : int — minimum observations before classifying a question
              (default 3; set higher for more confident flagging)
    sort_by : str — "mean_kappa" (worst questions first) or
                    "n_observations" (most-seen first)

    Response
    --------
    {
        "total_questions_tracked": int,
        "total_observations":      int,
        "n_ambiguous":             int,
        "n_watch":                 int,
        "n_clear":                 int,
        "ambiguous": [
            {
                "question_text":  str,
                "question_type":  str,
                "mean_kappa":     float,   # lower = more ambiguous
                "n_observations": int,
                "kappa_std":      float,   # high std = inconsistent interpretation
                "label":          "ambiguous",
                "rewrite_tip":    str,     # concrete rewrite recommendation
            },
            ...
        ],
        "watch":  [...],   # borderline questions (κ 0.50–0.65)
        "clear":  [...],   # well-specified questions (κ ≥ 0.65)
    }
    """
    if sort_by not in ("mean_kappa", "n_observations"):
        raise HTTPException(
            status_code=400,
            detail="sort_by must be 'mean_kappa' or 'n_observations'",
        )
    return question_ambiguity_tracker.get_ambiguity_report(
        min_obs = min_obs,
        sort_by = sort_by,
    )


# ── 4. NEXT QUESTION ──────────────────────────────────────────────────────────
class NextQuestionRequest(BaseModel):
    session_id: str
    q_index: int = 0


@app.post("/next_question")
async def next_question(body: NextQuestionRequest):
    """
    Generate the next interview question using the RL agent's recommended action.

    Uses the rl_hint dict persisted by /evaluate (which holds the sequencer's
    output from select_action()).  This guarantees the Bellman update and the
    ε-greedy action selection happen exactly once — inside /evaluate — and
    /next_question simply executes the recommendation.

    Falls back to seq.select_action() if /evaluate hasn't been called yet
    (e.g. very first next_question call edge case).

    Returns: question object + rl_action metadata
    """
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    role  = session.get("role", "Software Engineer")
    avoid = session.get("questions_asked", [])

    # ── v3.0: read company pack stored at session start ───────────────────────
    company_pack = session.get("company_pack", DEFAULT_COMPANY_PACK)

    # ── Read action from persisted rl_hint (set by /evaluate) ─────────────────
    hint = session.get("rl_hint") or {}
    is_follow_up  = hint.get("follow_up", False)
    q_type        = hint.get("type", "Technical")
    q_diff        = hint.get("difficulty", "medium")

    # If rl_hint is not yet available (very first question path), ask the
    # sequencer directly — this is a safe fallback and never skips a Q-update.
    if not hint:
        seq         = _get_sequencer(session)
        next_action = seq.select_action(score=3.0, nervousness=0.3, star_rate=0.5, time_eff=65.0)
        is_follow_up = next_action.follow_up
        q_type       = "HR" if next_action.q_type == "hr" else next_action.q_type.capitalize()
        q_diff       = next_action.difficulty

    # follow_up probe: keep same type as current question, probe deeper
    prev_question = ""
    if is_follow_up and session.get("current_question"):
        prev_q        = session["current_question"]
        prev_question = prev_q.get("question", "") if isinstance(prev_q, dict) else str(prev_q)
        q_type        = (prev_q.get("type", "Technical") if isinstance(prev_q, dict) else "Technical")

    # When user locked difficulty (not "all"), respect it; RL only controls type
    session_diff = session.get("difficulty", "medium")
    if session_diff != "all":
        q_diff = session_diff

    # ── v3.0: pick topic from company-pack-adjusted syllabus distribution ─────
    topic_hint = pick_topic_for_question(role, company_pack)

    # ── Generate via Groq ──────────────────────────────────────────────────────
    question = None
    if analyzer.groq_client:
        try:
            question = await _generate_question(
                analyzer.groq_client,
                role, q_diff, q_type,
                avoid        = avoid,
                is_follow_up = is_follow_up,
                prev_question= prev_question,
                company_pack = company_pack,
                topic_hint   = topic_hint,
            )
        except Exception as e:
            print(f"[next_question] Groq failed: {e}")

    if not question:
        # Fallback — Technical = conceptual/architectural only (no code editor)
        _FB: Dict[str, str] = {
            "Technical":   f"Explain an architectural or design decision you made as a {role} and the reasoning behind it.",
            "Behavioural": f"Tell me about a time you demonstrated leadership or resilience in a {role} role.",
            "HR":          f"What motivates you to pursue a career as a {role} long-term?",
        }
        question = {
            "question":    _FB.get(q_type, _FB["Technical"]),
            "type":        q_type,
            "difficulty":  q_diff,
            "keywords":    ["challenge", "solution", "outcome"],
            "ideal_answer":"Cover context, your specific actions, and measurable result.",
            "topic":       topic_hint.get("topic_name", ""),
        }

    # ── Update session ─────────────────────────────────────────────────────────
    session["current_question"] = question
    session["q_index"]          = body.q_index
    session.setdefault("questions_asked", []).append(question.get("question", ""))
    # Clear consumed hint so stale data isn't reused
    session["rl_hint"] = None

    return JSONResponse(content={
        "question":   question,
        "rl_action":  hint or {},   # surface to frontend for debug/display
    })




# ── 5. REPORT ─────────────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    session_id: str
    answers: List[Dict[str, Any]] = []


@app.post("/report")
async def generate_report(body: ReportRequest):
    """
    Generate the final session report.
    - Aggregates scores, DISC, STAR across all answers
    - Calls Groq for HR recommendation + coaching summary
    - Runs resume ↔ interview gap analysis (analyze_gap)
    - Returns RL sequencer stats
    Returns: hr_recommendation, hr_reasoning, rl_report, aggregated metrics,
             resume_gap (GapReport dict — empty if no resume was provided)
    """
    session = SESSIONS.get(body.session_id, {})
    answers = body.answers or session.get("answers", [])

    if not answers:
        return JSONResponse(content={
            "hr_recommendation": "No Data",
            "hr_reasoning": "No answers were submitted in this session.",
            "rl_report": {},
            "avg_score": 0,
            "resume_gap": {},
        })

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    avg_score    = round(sum(a.get("score", 3) for a in answers) / len(answers), 2)
    avg_nerv     = round(sum(a.get("nervousness", 0.2) for a in answers) / len(answers), 2)
    avg_star     = round(sum(a.get("star_coverage", 0.5) for a in answers) / len(answers), 2)
    avg_filler   = round(sum(a.get("filler_count", 0) for a in answers) / len(answers), 1)
    avg_wpm      = round(sum(a.get("wpm", 120) for a in answers) / len(answers))

    # Aggregate DISC
    disc_agg: Dict[str, float] = {}
    for a in answers:
        for k, v in (a.get("disc") or {}).items():
            disc_agg[k] = disc_agg.get(k, 0) + v
    disc_avg = {k: round(v / len(answers), 1) for k, v in disc_agg.items()}

    # ── RL session report — pull stats from real sequencer + save Q-table ────
    seq             = session.get("rl_sequencer")
    follow_up_count = sum(1 for a in answers if a.get("is_follow_up", False))

    if seq:
        # Persist individual Q-table and blend into shared prior for future sessions
        try:
            seq.save()
        except Exception as e:
            print(f"[report] Q-table save failed: {e}")

        rl_report = {
            "total_steps":      seq.total_steps,
            "avg_reward":       round(seq.avg_reward, 4),
            "epsilon":          round(seq.epsilon, 4),
            "follow_up_count":  follow_up_count,
            "action_counts":    seq.action_histogram(),   # {label: count}
            "q_table_saved":    True,
        }
    else:
        # Sequencer missing (edge case) — compute stats manually
        avg_reward_fallback = round(
            sum((a.get("score", 3) / 5.0) - a.get("nervousness", 0.2) * 0.3 for a in answers)
            / len(answers), 3,
        )
        rl_report = {
            "total_steps":     len(answers),
            "avg_reward":      avg_reward_fallback,
            "epsilon":         0.05,
            "follow_up_count": follow_up_count,
            "action_counts":   {},
            "q_table_saved":   False,
        }

    # ── Resume ↔ Interview Gap Analysis ──────────────────────────────────────
    # Uses the parsed resume stored at session_start. Runs in a thread so the
    # Groq calls inside analyze_gap don't block the async event loop.
    resume_gap: Dict[str, Any] = {}
    resume_parsed = session.get("resume_parsed", {})
    role          = session.get("role", "Software Engineer")
    if resume_parsed and resume_parsed.get("experience"):
        try:
            resume_gap = await asyncio.to_thread(
                analyze_gap, resume_parsed, answers, role
            )
        except Exception as e:
            print(f"[report] resume gap analysis failed: {e}")

    # ── Groq HR recommendation ─────────────────────────────────────────────────
    hr_data: Dict[str, Any] = {}
    if analyzer.groq_client:
        try:
            hr_data = await _generate_report(analyzer.groq_client, role, answers)
        except Exception as e:
            print(f"[report] Groq HR report failed: {e}")

    if not hr_data:
        # Fallback rule-based recommendation
        if avg_score >= 4.0:
            rec, reasoning = "Strong Yes", f"Candidate consistently scored {avg_score}/5 with strong technical depth and composure."
        elif avg_score >= 3.2:
            rec, reasoning = "Yes", f"Candidate averaged {avg_score}/5 — solid performance with room to grow."
        elif avg_score >= 2.5:
            rec, reasoning = "Maybe", f"Average score of {avg_score}/5. Mixed performance — recommend a follow-up interview."
        else:
            rec, reasoning = "No", f"Score of {avg_score}/5 below threshold. Significant gaps in technical knowledge or communication."
        hr_data = {
            "hr_recommendation": rec,
            "hr_reasoning": reasoning,
            "top_strength": "Shows effort and engagement throughout the session.",
            "top_weakness": "Needs to improve answer structure and reduce filler words.",
            "overall_coaching": "Practice the STAR method and aim for 120–160 WPM with fewer hesitation words.",
        }

    return JSONResponse(content={
        **hr_data,
        "avg_score":         avg_score,
        "avg_nervousness":   avg_nerv,
        "avg_star_coverage": avg_star,
        "avg_filler_count":  avg_filler,
        "avg_wpm":           avg_wpm,
        "disc_avg":          disc_avg,
        "rl_report":         rl_report,
        "total_answers":     len(answers),
        "role":              role,
        "resume_gap":        resume_gap,   # ← GapReport dict; {} if no resume
        # Feature 6 — cross-question thematic coherence (full session)
        "coherence_report":  compute_coherence_report(answers).to_dict(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL GAP ANALYSIS ENDPOINT
#  POST /skill_gap
#
#  Computes longitudinal skill deltas between the first half and second half
#  of a session, and flags the two most improved and two most declined skills.
#  Research basis: IJCRT 2026 — competency-wise skill gap analysis produces
#  more actionable reports than global scores alone.
# ══════════════════════════════════════════════════════════════════════════════

class SkillGapRequest(BaseModel):
    session_id: str = ""
    answers:    List[Dict] = []

@app.post("/skill_gap")
async def skill_gap(body: SkillGapRequest):
    """
    Returns per-skill deltas comparing early vs late session performance,
    plus priority focus areas and trend sentences for the report page.
    """
    session = SESSIONS.get(body.session_id, {})
    answers = body.answers or session.get("answers", [])

    if len(answers) < 2:
        return JSONResponse(content={
            "available": False,
            "reason": "Need at least 2 answers for skill gap analysis.",
        })

    def _safe(val, default=0.0):
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    # ── Build per-answer skill vectors ────────────────────────────────────────
    # Each dimension is normalised to 0–1.
    def _skill_vec(a: Dict) -> Dict[str, float]:
        return {
            "Knowledge":   _safe(a.get("knowledge"), 3.0) / 5.0,
            "STAR":        _safe(a.get("star_coverage"), 0.5),
            "Composure":   1.0 - _safe(a.get("nervousness"), 0.3),
            "Depth":       _safe(a.get("depth_score"), 0.0) / 5.0,
            "Fluency":     _safe(a.get("fluency_score"), 0.0) / 5.0,
            "Keywords":    _safe(a.get("keyword_score"), 0.4),
            "Clarity":     _safe(a.get("score"), 3.0) / 5.0,
            "Grammar":     _safe(a.get("grammar_score"), 0.5),
        }

    vecs = [_skill_vec(a) for a in answers]
    skills = list(vecs[0].keys())
    n = len(vecs)

    # ── Split into first-half and second-half ─────────────────────────────────
    mid = max(1, n // 2)
    first_half  = vecs[:mid]
    second_half = vecs[mid:]

    def _avg(half: List[Dict], skill: str) -> float:
        return sum(v[skill] for v in half) / len(half)

    # ── Compute deltas ────────────────────────────────────────────────────────
    deltas = {}
    for s in skills:
        early = _avg(first_half, s)
        late  = _avg(second_half, s)
        deltas[s] = {
            "early":  round(early * 100, 1),
            "late":   round(late * 100, 1),
            "delta":  round((late - early) * 100, 1),
            "trend":  "up" if late > early + 0.03
                      else "down" if late < early - 0.03
                      else "stable",
        }

    # ── Session averages ──────────────────────────────────────────────────────
    session_avgs = {
        s: round(sum(v[s] for v in vecs) / n * 100, 1)
        for s in skills
    }

    # ── Priority focus: biggest drops first ───────────────────────────────────
    sorted_by_delta = sorted(deltas.items(), key=lambda x: x[1]["delta"])
    focus_areas     = [k for k, _ in sorted_by_delta[:2]]          # worst drops
    strengths       = [k for k, _ in sorted_by_delta[-2:]]         # best gains

    # ── Generate natural-language trend sentences ─────────────────────────────
    trend_sentences = []
    for skill, d in sorted(deltas.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)[:4]:
        delta_abs = abs(d["delta"])
        if delta_abs < 1.0:
            continue
        direction = "improved" if d["delta"] > 0 else "dropped"
        trend_sentences.append(
            f"{skill} {direction} {delta_abs:.0f}pp "
            f"({d['early']:.0f}% → {d['late']:.0f}%)"
        )

    # ── Question-type breakdown ───────────────────────────────────────────────
    by_type: Dict[str, List[float]] = {}
    for a in answers:
        qt = a.get("question_type", "unknown")
        by_type.setdefault(qt, []).append(_safe(a.get("knowledge"), 3.0) / 5.0)
    type_avgs = {
        qt: round(sum(scores) / len(scores) * 100, 1)
        for qt, scores in by_type.items()
    }

    return JSONResponse(content={
        "available":        True,
        "n_answers":        n,
        "skills":           deltas,
        "session_avgs":     session_avgs,
        "focus_areas":      focus_areas,
        "strengths":        strengths,
        "trend_sentences":  trend_sentences,
        "by_question_type": type_avgs,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  METACOGNITIVE PROMPT ENGINE
#  POST /metacognitive
#
#  After scoring, instead of showing the ideal answer, returns 3 tiered
#  reflective questions that invite the candidate to think about their own
#  thinking (self-regulated learning theory; Flavell 1979; Zimmermann 2002).
#
#  Research basis:
#    • Tian et al. (2024, arXiv) — multi-agent pipeline with metacognitive
#      prompts rated more helpful and empathetic than answer-provision alone.
#    • Zimmermann (2002) — self-regulated learners who reflect before seeing
#      the answer retain corrective feedback 40% better.
#    • Hattie & Timperley (2007) — "feed-forward" questions (what next?)
#      outperform "feed-back" statements (what was wrong) on task learning.
#
#  Three prompt tiers:
#    SURFACE   — "What would you add if you had 30 more seconds?"
#                Lowest cognitive load; gets the candidate unstuck.
#    DEEP      — "What principle or pattern were you trying to demonstrate?"
#                Connects answer to underlying knowledge structure.
#    TRANSFER  — "How would your answer change for a Senior vs Junior role?"
#                Transfers insight to a new context (highest SRL level).
#
#  Groq generates all three from the actual answer + question + scores.
#  A static fallback bank covers API-unavailable situations.
# ══════════════════════════════════════════════════════════════════════════════

class MetacognitiveRequest(BaseModel):
    question:          str   = ""
    transcript:        str   = ""
    question_type:     str   = "behavioural"
    score:             float = 3.0
    star_coverage:     float = 0.5
    improvement_areas: List[str] = []
    key_strengths:     List[str] = []
    session_id:        str   = ""


# ── Static fallback prompt banks (used when Groq is unavailable) ─────────────

_META_FALLBACK: Dict[str, Dict[str, str]] = {
    "behavioural": {
        "surface":  "If you could add one more sentence to your answer, what specific outcome or result would you include?",
        "deep":     "What personal value or working principle were you trying to demonstrate through that story?",
        "transfer": "How would you frame this experience differently if the interviewer was a technical lead rather than an HR manager?",
    },
    "technical": {
        "surface":  "Which part of your answer do you feel was least complete, and what would you add?",
        "deep":     "What trade-off did you consciously or unconsciously make in your answer — and would you make it again?",
        "transfer": "How would your approach change if the system had 100× the scale you described?",
    },
    "hr": {
        "surface":  "What example from your experience best supports the point you were making, but you didn't mention?",
        "deep":     "What does your answer reveal about how you define success in a role?",
        "transfer": "How would your answer change if you were applying for a leadership position rather than an individual contributor role?",
    },
}

_META_LOW_SCORE_EXTRA = {
    "surface":  "What would you say first if you had to restart this answer from scratch?",
    "deep":     "What part of the question do you think you misunderstood or underweighted?",
    "transfer": "If a colleague gave this answer in a mock interview, what one piece of advice would you give them?",
}

_META_HIGH_STAR_EXTRA = {
    "surface":  "Your STAR structure was strong — which element (Situation/Task/Action/Result) felt weakest to you?",
    "deep":     "The result you described was clear — how would you quantify it more precisely if you had the data?",
    "transfer": "How would you tell this same story to a non-technical stakeholder versus a technical interviewer?",
}


async def _generate_metacognitive_prompts(
    groq_client,
    question:          str,
    transcript:        str,
    question_type:     str,
    score:             float,
    star_coverage:     float,
    improvement_areas: List[str],
) -> Dict[str, str]:
    """
    Use Groq to generate 3 tiered metacognitive prompts personalised to the
    candidate's actual answer. Returns {"surface":…, "deep":…, "transfer":…}.
    """
    q_type_label = question_type.capitalize()
    improve_str  = ", ".join(improvement_areas[:3]) if improvement_areas else "none identified"
    star_pct     = round(star_coverage * 100)

    prompt = f"""You are an expert interview coach specialising in metacognitive feedback.

A candidate just answered this {q_type_label} interview question:
QUESTION: {question}

THEIR ANSWER (transcript):
{transcript[:600]}

SCORE: {score:.1f}/5 | STAR coverage: {star_pct}% | Improvement areas: {improve_str}

Generate exactly 3 metacognitive prompts — questions that help the candidate reflect on their OWN thinking, not questions that test more knowledge.

Rules:
- SURFACE: lowest cognitive load. Ask what they would simply ADD or change. Max 20 words.
- DEEP: ask about the underlying principle, value, or reasoning behind their answer. Max 25 words.
- TRANSFER: ask how the answer would change in a different context or seniority level. Max 25 words.
- Never reveal the ideal answer. Never say "you should have…"
- Make prompts specific to their actual answer, not generic.

Return ONLY valid JSON:
{{"surface":"…","deep":"…","transfer":"…"}}"""

    try:
        response = await groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You generate metacognitive coaching prompts. Return only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        # Validate keys
        for k in ("surface", "deep", "transfer"):
            if not isinstance(result.get(k), str) or len(result[k]) < 10:
                raise ValueError(f"Missing or short key: {k}")
        return result
    except Exception:
        # Graceful fallback — never return an error to the UI
        qt_key = question_type.lower()
        if qt_key not in _META_FALLBACK:
            qt_key = "behavioural"
        base = dict(_META_FALLBACK[qt_key])
        # Score-adaptive substitution
        if score < 2.5:
            base["surface"] = _META_LOW_SCORE_EXTRA["surface"]
            base["deep"]    = _META_LOW_SCORE_EXTRA["deep"]
        if star_coverage >= 0.75:
            base["surface"] = _META_HIGH_STAR_EXTRA["surface"]
        return base


@app.post("/metacognitive")
async def metacognitive_prompts(body: MetacognitiveRequest):
    """
    Generate 3 tiered metacognitive prompts for a scored answer.
    Replaces / augments the ideal-answer pattern with reflective questions.

    Tier definitions (Zimmermann 2002 self-regulated learning levels):
      SURFACE  — forethought: what would I add?
      DEEP     — performance: what was my reasoning?
      TRANSFER — self-reflection: how does this generalise?
    """
    qt = body.question_type.lower().strip()
    if qt not in _META_FALLBACK:
        qt = "behavioural"

    if analyzer.groq_client:
        prompts = await _generate_metacognitive_prompts(
            groq_client       = analyzer.groq_client,
            question          = body.question,
            transcript        = body.transcript,
            question_type     = qt,
            score             = body.score,
            star_coverage     = body.star_coverage,
            improvement_areas = body.improvement_areas,
        )
    else:
        # No Groq — use static bank with score-adaptive selection
        base = dict(_META_FALLBACK[qt])
        if body.score < 2.5:
            base["surface"] = _META_LOW_SCORE_EXTRA["surface"]
            base["deep"]    = _META_LOW_SCORE_EXTRA["deep"]
        if body.star_coverage >= 0.75:
            base["surface"] = _META_HIGH_STAR_EXTRA["surface"]
        prompts = base

    return JSONResponse(content={
        "surface":  prompts["surface"],
        "deep":     prompts["deep"],
        "transfer": prompts["transfer"],
        "score":    body.score,
        "tier_labels": {
            "surface":  "Surface — What would you add?",
            "deep":     "Deep — What was your reasoning?",
            "transfer": "Transfer — How does this generalise?",
        },
        "research_note": (
            "Metacognitive prompts operationalise Zimmermann (2002) self-regulated "
            "learning theory: surface (forethought), deep (performance monitoring), "
            "transfer (self-reflection). Rated more helpful than answer-provision "
            "alone (Tian et al., 2024)."
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME REPHRASER ROUTES
# ══════════════════════════════════════════════════════════════════════════════
#
#  POST /resume/parse       — extract structured sections from resume text/file
#  POST /resume/rephrase    — ATS-optimise an already-parsed resume
#  POST /resume/score       — VMock-style bullet scoring (rule + Groq)
#  POST /resume/questions   — generate interview questions from resume
#  POST /resume/analyze     — one-shot: parse + rephrase + score + questions
#
# All heavy work is offloaded to asyncio.to_thread so the async event loop
# is never blocked by synchronous Groq SDK calls.
# ══════════════════════════════════════════════════════════════════════════════


# ── /resume/parse ─────────────────────────────────────────────────────────────
@app.post("/resume/parse")
async def resume_parse(
    file: Optional[UploadFile] = File(default=None),
    text: str = Form(default=""),
):
    """
    Parse a resume into structured sections.

    Accepts either:
      - file: a PDF or DOCX upload   (multipart/form-data)
      - text: raw resume text string (multipart/form-data)

    Returns: { name, summary, skills, projects, experience, education,
               certifications, raw_text_length }
    """
    raw_text = text.strip()

    if file and not raw_text:
        file_bytes = await file.read()
        filename   = (file.filename or "").lower()
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract text from PDF. Install pypdf: pip install pypdf"
                )
        elif filename.endswith(".docx"):
            raw_text = extract_text_from_docx(file_bytes)
            if not raw_text:
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract text from DOCX. Install python-docx: pip install python-docx"
                )
        else:
            # Treat as plain text
            raw_text = file_bytes.decode("utf-8", errors="ignore")

    if not raw_text:
        raise HTTPException(status_code=400, detail="Provide resume text or upload a PDF/DOCX file.")

    parsed = await asyncio.to_thread(parse_resume, raw_text)
    parsed["raw_text_length"] = len(raw_text)
    return JSONResponse(content=parsed)


# ── /resume/rephrase ──────────────────────────────────────────────────────────
@app.post("/resume/rephrase")
async def resume_rephrase(
    parsed_json: str = Form(...),       # JSON string of the parsed resume dict
    target_role: str = Form(default=""),
):
    """
    ATS-optimise an already-parsed resume.

    Body (form-data):
      parsed_json  — JSON string (output of /resume/parse)
      target_role  — optional target job title

    Returns: rephrased resume dict with same structure as parsed_json
    """
    try:
        parsed = json.loads(parsed_json)
    except Exception:
        raise HTTPException(status_code=400, detail="parsed_json must be valid JSON.")

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="parsed_json must be a JSON object.")

    rephrased = await asyncio.to_thread(rephrase_resume, parsed, target_role)
    return JSONResponse(content=rephrased)


# ── /resume/score ─────────────────────────────────────────────────────────────
@app.post("/resume/score")
async def resume_score(
    parsed_json:   str = Form(...),
    rephrased_json: str = Form(default="{}"),
    target_role:   str = Form(default=""),
):
    """
    Score a resume on a 0-100 scale with per-bullet feedback.

    Body (form-data):
      parsed_json    — JSON string (output of /resume/parse)
      rephrased_json — JSON string (output of /resume/rephrase); falls back to parsed
      target_role    — optional target job title

    Returns: {
      overall, percentile_lo, percentile_hi, pct_label, pct_colour,
      section_scores, experience_bullets, project_bullets, bullet_count
    }
    """
    try:
        parsed = json.loads(parsed_json)
    except Exception:
        raise HTTPException(status_code=400, detail="parsed_json must be valid JSON.")

    try:
        rephrased = json.loads(rephrased_json) if rephrased_json.strip() not in ("", "{}") else {}
    except Exception:
        rephrased = {}

    score_data = await asyncio.to_thread(score_resume, parsed, rephrased, target_role)
    return JSONResponse(content=score_data)


# ── /resume/questions ─────────────────────────────────────────────────────────
@app.post("/resume/questions")
async def resume_questions(
    parsed_json:    str = Form(...),
    rephrased_json: str = Form(default="{}"),
    target_role:    str = Form(default=""),
    num_questions:  int = Form(default=10),
    difficulty:     str = Form(default="Medium"),   # Easy | Medium | Hard
):
    """
    Generate tailored interview questions from a candidate's resume.

    Body (form-data):
      parsed_json    — JSON string (output of /resume/parse)
      rephrased_json — JSON string (output of /resume/rephrase); optional
      target_role    — optional target job title
      num_questions  — how many questions to generate (5–20, default 10)
      difficulty     — Easy | Medium | Hard (default Medium)

    Returns: { questions: [ {question, type, target, difficulty,
                              ideal_keywords, ideal_answer}, ... ] }
    """
    try:
        parsed = json.loads(parsed_json)
    except Exception:
        raise HTTPException(status_code=400, detail="parsed_json must be valid JSON.")

    try:
        rephrased = json.loads(rephrased_json) if rephrased_json.strip() not in ("", "{}") else {}
    except Exception:
        rephrased = {}

    num_questions = max(1, min(20, num_questions))

    questions = await asyncio.to_thread(
        resume_generate_questions,
        parsed, rephrased, target_role, num_questions, difficulty,
    )
    return JSONResponse(content={"questions": questions})


# ── /resume/analyze ───────────────────────────────────────────────────────────
@app.post("/resume/analyze")
async def resume_analyze(
    file:          Optional[UploadFile] = File(default=None),
    text:          str = Form(default=""),
    target_role:   str = Form(default=""),
    num_questions: int = Form(default=10),
    difficulty:    str = Form(default="Medium"),
    auto_rephrase: bool = Form(default=True),
):
    """
    One-shot endpoint: parse → rephrase → score → generate questions.

    Accepts either a file upload (PDF/DOCX) or plain text.
    Runs parse first, then rephrase + score + questions concurrently.

    Returns: {
      parsed, rephrased, score_data, questions,
      processing_time_ms
    }
    """
    start = time.time()

    # ── 1. Extract raw text ────────────────────────────────────────────────────
    raw_text = text.strip()

    if file and not raw_text:
        file_bytes = await file.read()
        filename   = (file.filename or "").lower()
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                raise HTTPException(
                    status_code=422,
                    detail="PDF text extraction failed. Install pypdf: pip install pypdf"
                )
        elif filename.endswith(".docx"):
            raw_text = extract_text_from_docx(file_bytes)
            if not raw_text:
                raise HTTPException(
                    status_code=422,
                    detail="DOCX extraction failed. Install python-docx: pip install python-docx"
                )
        else:
            raw_text = file_bytes.decode("utf-8", errors="ignore")

    if not raw_text:
        raise HTTPException(status_code=400, detail="Provide resume text or upload a PDF/DOCX file.")

    # ── 2. Parse ───────────────────────────────────────────────────────────────
    parsed = await asyncio.to_thread(parse_resume, raw_text)

    # ── 3. Rephrase + score + questions in parallel ────────────────────────────
    # FIX 4: asyncio.sleep(0, result=...) is only available in Python ≥ 3.12.
    # On Python 3.10/3.11 (the most common deployment targets) this crashes with
    # TypeError, breaking every /resume/analyze call when auto_rephrase=False.
    # Use an explicit async lambda wrapper instead — works on Python 3.8+.
    async def _identity(v):
        return v

    rephrase_task  = asyncio.to_thread(rephrase_resume, parsed, target_role) if auto_rephrase else _identity(parsed)
    questions_task = asyncio.to_thread(
        resume_generate_questions,
        parsed, {}, target_role, max(1, min(20, num_questions)), difficulty,
    )

    rephrased, questions = await asyncio.gather(rephrase_task, questions_task)

    # Score uses rephrased content
    score_data = await asyncio.to_thread(score_resume, parsed, rephrased, target_role)

    return JSONResponse(content={
        "parsed":               parsed,
        "rephrased":            rephrased,
        "score_data":           score_data,
        "questions":            questions,
        "processing_time_ms":   round((time.time() - start) * 1000),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SSE TRAINING STREAM ENDPOINTS
#  Integrated from training_sse_endpoint.py
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/voice/train-stream")
async def voice_train_stream():
    """
    SSE endpoint — streams real-time training logs to the frontend.

    Usage (browser):
        const es = new EventSource("http://localhost:8000/voice/train-stream");
        es.onmessage = e => console.log(JSON.parse(e.data));

    Each event is a JSON object:
        { msg: string, level: "info"|"metric"|"error"|"success"|"download"|"sentinel", ts: float }

    The stream closes when it receives level="sentinel" (training done/failed).
    """
    async def _generate():
        # Drain any stale messages left from a previous connection
        while not _training_log_queue.empty():
            try:
                _training_log_queue.get_nowait()
            except queue.Empty:
                break

        # If training already finished and metrics exist, send them immediately
        if not _training_active and _training_metrics:
            try:
                yield f"data: {json.dumps({'msg': 'Model already trained — metrics loaded from disk.', 'level': 'success', 'metrics': _training_metrics})}\n\n"
                yield f"data: {json.dumps({'msg': '__DONE__', 'level': 'sentinel'})}\n\n"
            except (GeneratorExit, Exception):
                pass
            return

        # If not currently training, start it
        if not _training_active:
            import threading
            t = threading.Thread(
                target=_background_pipeline_setup,
                daemon=True,
                name="voice_setup_sse",
            )
            t.start()

        # Stream log events — exit cleanly when the client disconnects.
        # The queue.get() is offloaded to a thread so the event loop stays
        # responsive and asyncio.CancelledError can be raised between yields.
        timeout_counter = 0
        MAX_IDLE_SECS = 1800  # 30 min hard timeout
        try:
            while timeout_counter < MAX_IDLE_SECS:
                try:
                    event = await asyncio.to_thread(_training_log_queue.get, True, 1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    timeout_counter = 0  # reset on activity
                    if event.get("level") == "sentinel":
                        break
                except queue.Empty:
                    timeout_counter += 1
                    # Keep-alive ping every 15s so the browser doesn't disconnect
                    if timeout_counter % 15 == 0:
                        yield f"data: {json.dumps({'msg': '⏳ Training in progress…', 'level': 'ping'})}\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            # Client navigated away or closed the tab — stop silently.
            # The training thread continues unaffected in the background.
            pass
        except Exception:
            # Any other send failure (broken pipe, etc.) — exit cleanly.
            pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.post("/voice/retrain-stream")
async def voice_retrain_stream(
    force_retrain: bool = Form(default=True),
    max_per_dataset: int = Form(default=3000),
):
    """
    Trigger a fresh retrain AND immediately redirect the caller to the SSE stream.
    Accepts form-data: force_retrain (bool), max_per_dataset (int 500–7000).
    Returns 200 JSON confirming the background thread was launched.
    """
    global _training_active
    if _training_active:
        return JSONResponse({"status": "already_training"}, status_code=202)

    import threading

    def _retrain():
        global _training_active, _training_metrics
        _training_active = True

        def _push(msg: str, level: str = "info", extra: dict = None):
            payload = {"msg": msg, "level": level, "ts": __import__("time").time()}
            if extra:
                payload.update(extra)
            try:
                _training_log_queue.put_nowait(payload)
            except queue.Full:
                pass

        _push(f"🔄 Retraining with max_per_dataset={max_per_dataset}…", "start")
        try:
            capped = max(500, min(7000, max_per_dataset))
            metrics = _voice_pipeline.setup(
                force_retrain=force_retrain,
                max_per_dataset=capped,
                progress_cb=lambda m: _push(m, _classify_log_level(m)),
            )
            acoustic_analyser.set_unified(_voice_pipeline)
            _training_metrics = metrics
            _push(
                f"✅ Retrain complete — test_acc={metrics.get('test_accuracy')}%",
                "done",
                {"metrics": metrics},
            )
        except Exception as exc:
            _push(f"❌ Retrain failed: {exc}", "error")
        finally:
            _training_active = False
            try:
                _training_log_queue.put_nowait({"msg": "__DONE__", "level": "sentinel"})
            except queue.Full:
                pass

    t = threading.Thread(target=_retrain, daemon=True, name="voice_retrain")
    t.start()

    return JSONResponse({
        "status": "training_started",
        "stream_url": "/voice/train-stream",
        "message": "Connect to /voice/train-stream to watch real-time logs",
    })

# ══════════════════════════════════════════════════════════════════════════════
#  HR ROUND ROUTES
#  Requires: hr_round.py in same directory
# ══════════════════════════════════════════════════════════════════════════════

# ── HR Question Bank + Evaluator ──────────────────────────────────────────────
# Embedded directly here so there is NO external file dependency.
# hr_round.py imports streamlit at the top level which crashes the FastAPI
# process (ImportError) and caused every /hr/questions call to return 503.

HR_QUESTIONS = [
    # ── PART 1 ───────────────────────────────────────────────────────────────
    {"id":1,"part":"Part 1: The Frequent Golden Key","question":"Tell me about yourself.",
     "key":"Elevator Pitch – pull the string to win","method":"Elevator Pitch",
     "focus":"Education → Key skills/strength → Projects → Career goal/mission statement",
     "why":"Sets the tone for the entire interview; interviewers judge clarity and confidence.",
     "tip":"Keep it 90 seconds. End with a forward-looking mission statement."},
    {"id":2,"part":"Part 1: The Frequent Golden Key","question":"What do you know about our organization?",
     "key":"Research company overview – recent developments/news/values and mission","method":"4W Formula",
     "focus":"4W Formula: Who they are – what they do – why they are different – what are their principles and values",
     "why":"Tests preparation and genuine interest in the company.",
     "tip":"Mention a specific recent news item or product to stand out."},
    {"id":3,"part":"Part 1: The Frequent Golden Key","question":"Why did you choose Computer Science?",
     "key":"Explain interest in technology, problem-solving, and innovation","method":"Passion + Skills alignment",
     "focus":"Logical thinking | Problem solving | Team collaboration | Adaptability | Creative thinking",
     "why":"Checks authentic motivation and fit with technical roles.",
     "tip":"Tie your choice back to a specific experience or moment of curiosity."},
    {"id":4,"part":"Part 1: The Frequent Golden Key","question":"What are your strengths?",
     "key":"Mention 2–3 strengths with concrete examples","method":"Strength + Example",
     "focus":"Logical thinking | Problem solving | Team collaboration | Adaptability | Creative thinking",
     "why":"Evaluates self-awareness and ability to articulate value.",
     "tip":"Use real project examples. Never just list adjectives."},
    {"id":5,"part":"Part 1: The Frequent Golden Key","question":"What is your weakness?",
     "key":"Mention a genuine but improvable weakness and show how you are improving it","method":"WAAI Framework",
     "focus":"Weakness → Awareness → Action → Improvement",
     "why":"Tests honesty and growth mindset.",
     "tip":"Never say 'I'm a perfectionist.' Choose a real weakness that does not directly impact core job duties, then explain your corrective action."},
    {"id":6,"part":"Part 1: The Frequent Golden Key","question":"Why do you want to join our company?",
     "key":"Show research about the organization, its culture, and growth opportunities","method":"CRLC Framework",
     "focus":"Company fit + Role alignment + Learning opportunity + Contribution",
     "why":"Assesses motivation beyond salary and genuine cultural alignment.",
     "tip":"Personalise — generic answers are immediately spotted by experienced interviewers."},
    {"id":7,"part":"Part 1: The Frequent Golden Key","question":"Explain your final year project.",
     "key":"Structure: Objective → Technology → Role → Result","method":"COER Method",
     "focus":"COER: Context → Objective → Approach → Result",
     "why":"Tests technical communication and ownership of work.",
     "tip":"Mention your specific contribution clearly — not just what the team did."},
    {"id":8,"part":"Part 1: The Frequent Golden Key","question":"Where do you see yourself in five years?",
     "key":"Show growth mindset and long-term commitment to learning","method":"Career Vision",
     "focus":"Career progression + Expertise building + Value creation",
     "why":"Checks ambition, retention likelihood, and goal clarity.",
     "tip":"Link your personal growth trajectory to value you will create for the company."},
    {"id":9,"part":"Part 1: The Frequent Golden Key","question":"Why should we hire you?",
     "key":"Combine skills + attitude + learning ability","method":"SAV Framework",
     "focus":"Skills match + Right attitude + Value addition",
     "why":"The core pitch — your final chance to consolidate a strong impression.",
     "tip":"Summarise your top 2–3 differentiators confidently. This is your sales pitch — own it."},
    {"id":10,"part":"Part 1: The Frequent Golden Key","question":"What motivates you?",
     "key":"Purpose + Growth + Achievement","method":"PGA Framework",
     "focus":"Solving complex problems | Learning new technologies | Building useful solutions",
     "why":"Reveals drive, passion, and cultural fit.",
     "tip":"Align your motivators with what the role and company offer."},
    {"id":11,"part":"Part 1: The Frequent Golden Key","question":"Do you have any questions for us?",
     "key":"Ask for learning intent and interest","method":"Curiosity Signals",
     "focus":"Show curiosity about contribution, not just benefits",
     "why":"Tests engagement level and whether you are genuinely interested.",
     "tip":"Prepare 2–3 thoughtful questions about team culture, growth paths, or current challenges."},
    # ── PART 2 ───────────────────────────────────────────────────────────────
    {"id":12,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Tell me about a time when you faced a difficult problem.",
     "key":"Recruiters test problem-solving ability","method":"SOAR Method",
     "focus":"Situation → Challenge/Opportunity → Action → Result",
     "why":"Tests analytical thinking and resilience under challenges.",
     "tip":"Highlight what you learned from the experience, not just the outcome."},
    {"id":13,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Describe a situation where you had a conflict with a team member.",
     "key":"Tests communication and emotional intelligence","method":"DCUO Method",
     "focus":"Disagreement → Communication → Understanding → Outcome",
     "why":"Interviewers assess how you handle people, not just problems.",
     "tip":"Never blame the other person. Focus on maturity, communication, and the positive outcome."},
    {"id":14,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Tell me about a time when you worked under pressure or tight deadlines.",
     "key":"Shows time management and resilience","method":"SOAR Method",
     "focus":"High pressure + smart planning + focused execution + successful delivery",
     "why":"Reveals composure, prioritisation skills, and ability to deliver under constraints.",
     "tip":"Highlight calmness, prioritisation, and delivery under pressure — that's what recruiters truly test."},
    {"id":15,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Describe a situation where you took initiative.",
     "key":"Recruiters want proactive employees","method":"SOAR Method",
     "focus":"Identified a gap → Took ownership → Implemented solution → Achieved impact",
     "why":"Tests proactiveness and ownership mindset.",
     "tip":"Emphasise proactiveness and ownership — companies hire people who act, not just react."},
    {"id":16,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Tell me about a failure or mistake you made.",
     "key":"Evaluates accountability and learning mindset","method":"MOLI Method",
     "focus":"Mistake → Ownership → Learning → Improvement",
     "why":"Checks honesty, maturity, and the ability to grow from setbacks.",
     "tip":"Never justify or blame others. Focus on accountability and growth — that's what impresses recruiters."},
    {"id":17,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Describe a time when you helped a teammate.",
     "key":"Shows collaboration and leadership potential","method":"SOAR Method",
     "focus":"Situation → Support → Action → Result",
     "why":"Companies value team players over solo performers.",
     "tip":"Demonstrate team spirit, empathy, and collaborative instinct."},
    {"id":18,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Tell me about a time you had to learn something quickly.",
     "key":"Important for roles where continuous learning is required","method":"SOAR Method",
     "focus":"Situation → Learning need → Action → Result",
     "why":"Recruiters assess adaptability and learning agility in dynamic roles.",
     "tip":"Highlight the speed, method of learning, and how you applied it practically."},
    {"id":19,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Describe a situation where you had multiple tasks to manage.",
     "key":"Tests prioritisation and productivity","method":"SOAR Method",
     "focus":"Situation → Multiple tasks → Prioritisation → Execution → Result",
     "why":"Key skills recruiters look for in high-performance roles.",
     "tip":"Emphasise prioritisation, organisation, and time management — not just that you were busy."},
    {"id":20,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Tell me about a time when your idea improved a project or process.",
     "key":"Tests innovation and critical thinking","method":"SOAR Method",
     "focus":"Problem → Innovative idea → Action → Measurable impact",
     "why":"Companies need people who actively improve things, not just execute.",
     "tip":"Always mention the measurable impact (time saved, efficiency improved, quality enhanced)."},
    {"id":21,"part":"Part 2: Situation-Based HR/Managerial Questions","question":"Describe a situation where you had to adapt to change.",
     "key":"Evaluates flexibility and adaptability","method":"SOAR Method",
     "focus":"Disruption → Mindset shift → Smart action → Achievement",
     "why":"Recruiters assess flexibility, resilience, and positive mindset in dynamic environments.",
     "tip":"Emphasise your willingness to embrace change — not just that you survived it."},
]
HR_OK = True


def _heuristic_eval(answer: str) -> dict:
    words = len(answer.split())
    score = 2 if words < 10 else 4 if words < 30 else 6 if words < 60 else 7 if words < 100 else 8
    verdicts = {2: "Poor", 4: "Needs Improvement", 6: "Average", 7: "Good", 8: "Good"}
    return {
        "score": score, "verdict": verdicts.get(score, "Average"),
        "strengths": ["Answer provided"],
        "improvements": ["Add specific examples", "Follow the suggested framework"],
        "ideal_structure": "Use the recommended method with concrete examples.",
    }


def _evaluate_with_groq(question: dict, answer: str) -> dict:
    """Call Groq LLM to evaluate an HR answer; falls back to heuristic."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return _heuristic_eval(answer)
    prompt = (
        f"You are an expert HR interview coach evaluating a campus placement candidate.\n\n"
        f"QUESTION: {question['question']}\n"
        f"FRAMEWORK: {question['focus']}\n"
        f"METHOD: {question['method']}\n"
        f"CANDIDATE'S ANSWER: {answer}\n\n"
        f"Evaluate strictly. Respond ONLY with valid JSON — no preamble, no markdown:\n"
        f'{{"score":<int 1-10>,"verdict":"<Excellent/Good/Average/Needs Improvement/Poor>",'
        f'"strengths":["..."],"improvements":["..."],"ideal_structure":"..."}}'
    )
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(f"Groq HR eval failed: {exc}")
        return _heuristic_eval(answer)


@app.get("/hr/questions")
async def hr_questions(filter: str = "all"):
    """Return HR question bank filtered by part or preset."""
    if not HR_OK or not HR_QUESTIONS:
        raise HTTPException(status_code=503, detail="hr_round.py not found — place it next to main.py")

    if filter == "part1":
        qs = [q for q in HR_QUESTIONS if q["part"].startswith("Part 1")]
    elif filter == "part2":
        qs = [q for q in HR_QUESTIONS if q["part"].startswith("Part 2")]
    elif filter == "quick":
        top_ids = {1, 4, 5, 6, 9, 12, 13, 14, 16, 21}
        qs = [q for q in HR_QUESTIONS if q["id"] in top_ids]
    else:
        qs = HR_QUESTIONS

    return JSONResponse({"questions": qs, "total": len(qs)})


@app.post("/hr/evaluate")
async def hr_evaluate(
    question_id: int = Form(...),
    answer: str = Form(...),
):
    """Evaluate a single HR answer with Groq LLM (falls back to heuristic)."""
    q = next((q for q in HR_QUESTIONS if q["id"] == question_id), None)
    if not q:
        raise HTTPException(status_code=404, detail=f"Question id={question_id} not found")

    result = await asyncio.to_thread(_evaluate_with_groq, q, answer)
    return JSONResponse({"question": q, "eval": result})


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON / BENCHMARK ROUTES
#  Requires: model_comparison.py in same directory
# ══════════════════════════════════════════════════════════════════════════════

try:
    from model_comparison import (
        run_benchmark,
        score_keyword_match,
        score_tfidf,
        score_bm25,
        score_sbert,
        score_aura,
    )
    MC_OK = True
except ImportError:
    MC_OK = False


@app.post("/benchmark/run")
async def benchmark_run(max_entries: int = Form(default=40)):
    """
    Run the full 5-scorer benchmark on the embedded / Kaggle HR dataset.
    Returns Pearson r, MAE, consistency, coverage, and per-category averages.
    """
    if not MC_OK:
        raise HTTPException(status_code=503, detail="model_comparison.py not found — place it next to main.py")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    capped = max(10, min(200, max_entries))

    result = await asyncio.to_thread(
        run_benchmark,
        groq_api_key=groq_key,
        max_entries=capped,
    )
    return JSONResponse(result)


@app.post("/benchmark/live-score")
async def benchmark_live_score(
    question: str = Form(default=""),
    answer: str = Form(...),
    ideal: str = Form(...),
):
    """
    Score one answer against an ideal using all 5 scorers simultaneously.
    Returns a dict of {scorer_name: score_0_to_100}.
    """
    if not MC_OK:
        raise HTTPException(status_code=503, detail="model_comparison.py not found — place it next to main.py")

    groq_key = os.environ.get("GROQ_API_KEY", "")

    # Run all 5 scorers concurrently
    kw, tfidf, bm25, sbert, aura = await asyncio.gather(
        asyncio.to_thread(score_keyword_match, answer, ideal, []),
        asyncio.to_thread(score_tfidf, answer, ideal),
        asyncio.to_thread(score_bm25, answer, ideal),
        asyncio.to_thread(score_sbert, answer, ideal),
        asyncio.to_thread(score_aura, answer, ideal, [], groq_api_key=groq_key),
    )

    return JSONResponse({
        "scores": {
            "Keyword Match": round(kw, 1),
            "TF-IDF":        round(tfidf, 1),
            "BM25":          round(bm25, 1),
            "SBERT":         round(sbert, 1),
            "Aura AI":       round(aura, 1),
        },
        "question": question,
    })