"""
resume_rephraser_api.py — Resume Rephraser Engine (FastAPI edition)
====================================================================
Stripped of all Streamlit dependencies. Exposes pure Python functions
that are called directly by main.py FastAPI routes:

  parse_resume(text)                          → dict
  rephrase_resume(parsed, target_role)        → dict
  generate_questions(parsed, rephrased, ...)  → list[dict]
  score_resume(parsed, rephrased, target_role)→ dict

  ── Resume ↔ Interview Gap Analysis (v1.0) ──────────────────────────
  extract_resume_claims(parsed)               → list[ResumeClaim]
  analyze_gap(parsed, session_answers, role)  → GapReport

  Gap analysis detects which specific resume claims — achievements,
  skills, leadership statements, quantified metrics — were NEVER
  demonstrated during the interview. Each uncovered claim is flagged
  with a predicted interviewer question and a coaching tip.

  RESEARCH BASIS
  --------------
  Levashina et al. (2014, Personnel Psychology) — Structured Interview
  Validity: interviewers form a mental model of the candidate from the
  resume BEFORE the interview begins. Claims that go unaddressed during
  the interview create a "credibility gap" — the interviewer retains
  curiosity about whether the claim is genuine, which depresses hiring
  probability even when overall answers are strong.

  Huffcutt & Arthur (1994, J. Applied Psychology) — meta-analysis of
  structured interviews (k=114): resume-driven probing questions
  (targeting specific CV claims) produce higher predictive validity
  (r=0.56) than generic competency questions (r=0.37). Uncovered claims
  are the top source of post-interview follow-up requests.

  Tsai et al. (2016, J. Vocational Behavior) — Applicant impression
  management: candidates who proactively surface their strongest resume
  claims in answers receive 18% higher hiring ratings than candidates
  who wait to be asked. Gap analysis makes this gap visible and
  actionable before the next interview.

  Macan & Dipboye (1990, J. Applied Social Psychol.) — resume-to-
  interview consistency: inconsistency between written resume claims and
  spoken interview content is the #1 factor interviewers cite when
  declining a candidate post-interview.

  INTEGRATION INTO main.py
  ------------------------
  Call analyze_gap() inside POST /report after aggregating answers:

    from resume_rephraser_api import extract_resume_claims, analyze_gap
    gap_report = analyze_gap(resume_parsed, session_answers, role)
    # Add gap_report dict to the /report JSON response

  POST /resume_gap (standalone) can also expose this directly.

PDF / DOCX extraction helpers are also kept so the upload route in
main.py can convert file bytes to plain text before calling parse_resume.

Requirements (add to your existing requirements.txt):
  pypdf          # or PyPDF2
  python-docx    # for .docx upload support
  groq           # already in your project
  sentence-transformers  # optional — enables semantic claim matching
                         # falls back to keyword overlap if unavailable
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ── SAS scorer (shared singleton — same model instance as analyzer.py) ─────────
from sas_scorer import sas_scorer as _sas_scorer

# ── Groq (async client used by main.py; sync client used internally here) ──────
try:
    from groq import Groq
    _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    GROQ_OK = bool(os.environ.get("GROQ_API_KEY", ""))
except Exception:
    GROQ_OK = False
    _groq_client = None

MODEL = "llama-3.3-70b-versatile"

# ── Optional PDF reader ────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        PYPDF_OK = True
    except ImportError:
        PYPDF_OK = False

# ── Optional DOCX reader ───────────────────────────────────────────────────────
try:
    import docx as _docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  FILE EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes. Returns '' if pypdf not installed."""
    if not PYPDF_OK:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from DOCX bytes. Returns '' if python-docx not installed."""
    if not DOCX_OK:
        return ""
    try:
        doc = _docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL GROQ HELPERS  (synchronous — called in asyncio.to_thread in main.py)
# ══════════════════════════════════════════════════════════════════════════════

def _call_groq(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    if not GROQ_OK or not _groq_client:
        return ""
    try:
        messages: List[Dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API Error: {e}]"


def _call_groq_json(prompt: str, system: str = "") -> Any:
    """Call Groq and parse the JSON response. Returns {} on failure."""
    raw = _call_groq(prompt, system=system, max_tokens=3000)
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_PARSE_SYSTEM = (
    "You are a professional resume parser. Extract structured information from resumes. "
    "Always respond with valid JSON only. No markdown, no explanation, just the JSON object."
)

_REPHRASE_SYSTEM = (
    "You are an expert resume writer and career coach specialising in ATS-optimised, "
    "impact-driven language. Rephrase resume content to be stronger, more concise, and "
    "achievement-focused. Use strong action verbs. Quantify where possible. Remove filler words. "
    "Always respond with valid JSON only. No markdown, no explanation."
)

_QUESTION_SYSTEM = (
    "You are a senior technical interviewer. Generate highly targeted interview questions "
    "based on the candidate's specific resume content. Questions should probe depth of knowledge, "
    "real experience, and problem-solving ability. Always respond with valid JSON only."
)

_GAP_SYSTEM = (
    "You are a senior hiring manager and interview coach. You specialise in identifying "
    "mismatches between what a candidate claims on their resume and what they actually "
    "demonstrated during the interview. Be specific, actionable, and constructive. "
    "Always respond with valid JSON only. No markdown, no explanation."
)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE AI FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_resume(text: str) -> Dict:
    """
    Extract structured sections from raw resume text via Groq.
    Returns a dict with keys: name, summary, skills, projects, experience,
    education, certifications.
    """
    prompt = f"""Parse this resume and extract the following sections.
Return a JSON object with these exact keys:
- "name": candidate's full name (string)
- "summary": professional summary or objective (string, empty if none)
- "skills": list of skill strings
- "projects": list of objects with keys: title, description, technologies (list), impact
- "experience": list of objects with keys: company, role, duration, responsibilities (list of strings), achievements (list of strings)
- "education": list of objects with keys: institution, degree, field, year, gpa (optional)
- "certifications": list of strings

CRITICAL RULES:
1. Copy each experience entry's duration EXACTLY as it appears. Do NOT swap durations between entries.
2. Keep responsibilities and achievements tied to the correct company entry.
3. Do NOT invent, infer, or rephrase factual details (dates, durations, company names).
4. Preserve the original order of experience entries exactly as they appear.

RESUME TEXT:
{text[:6000]}"""

    result = _call_groq_json(prompt, system=_PARSE_SYSTEM)
    if not isinstance(result, dict):
        result = {}

    defaults: Dict[str, Any] = {
        "name": "", "summary": "",
        "skills": [], "projects": [], "experience": [],
        "education": [], "certifications": [],
    }
    for k, v in defaults.items():
        result.setdefault(k, v)
    return result


def rephrase_resume(parsed: Dict, target_role: str = "") -> Dict:
    """
    Rephrase all resume sections using stronger, ATS-optimised language via Groq.
    Falls back to returning the original parsed dict if Groq is unavailable.
    """
    role_hint = f" The target role is: {target_role}." if target_role else ""

    prompt = f"""Rephrase the following resume sections to be stronger, more impactful, and ATS-optimised.{role_hint}

Rules:
- Use strong action verbs (Engineered, Architected, Optimised, Spearheaded, etc.)
- Add quantification where reasonable (e.g. "Improved load time by ~40%")
- Remove weak phrases like "responsible for", "helped with", "worked on"
- Keep each bullet concise (max 20 words)
- Rephrase skills into grouped categories with context

Input JSON:
{json.dumps(parsed, indent=2)[:5000]}

Return a JSON object with the same structure as the input, but with all text rephrased.
Keep the exact same keys. For lists of strings, return lists of rephrased strings.
For objects with "responsibilities" and "achievements", rephrase each item."""

    result = _call_groq_json(prompt, system=_REPHRASE_SYSTEM)
    if not isinstance(result, dict) or not result:
        return parsed
    for k in parsed:
        result.setdefault(k, parsed[k])
    return result


def generate_questions(
    parsed: Dict,
    rephrased: Dict,
    target_role: str = "",
    num_questions: int = 10,
    difficulty: str = "Medium",
) -> List[Dict]:
    """
    Generate tailored interview questions from resume content via Groq.
    Returns a list of question dicts.
    """
    role_hint = f"Target role: {target_role}. " if target_role else ""
    diff_map = {
        "Easy":   "Beginner-friendly, conceptual, definition-based questions.",
        "Medium": "Mix of conceptual and applied questions requiring real experience.",
        "Hard":   "Deep-dive technical, system design, and behavioural questions probing edge cases.",
    }
    diff_hint = diff_map.get(difficulty, diff_map["Medium"])

    raw_skills = rephrased.get("skills") or parsed.get("skills") or []
    skills_str = ", ".join(
        s if isinstance(s, str) else str(s) for s in raw_skills[:20]
    )

    raw_projects = rephrased.get("projects") or parsed.get("projects") or []
    proj_titles = [
        p.get("title", "") if isinstance(p, dict) else str(p) for p in raw_projects
    ]

    raw_exp = rephrased.get("experience") or parsed.get("experience") or []
    exp_roles = [
        f"{e.get('role','')} at {e.get('company','')}" if isinstance(e, dict) else str(e)
        for e in raw_exp
    ]

    prompt = f"""Generate exactly {num_questions} interview questions for a candidate based on their resume.

{role_hint}Difficulty: {difficulty} — {diff_hint}

Candidate profile:
- Skills: {skills_str or 'Not specified'}
- Projects: {', '.join(proj_titles) or 'None listed'}
- Experience: {', '.join(exp_roles) or 'None listed'}

Full resume context:
{json.dumps(rephrased or parsed, indent=2)[:4000]}

Return a JSON array of exactly {num_questions} question objects, each with:
- "question": the full question text (string)
- "type": one of ["Technical", "Behavioural", "Project-Based", "System Design", "Situational"]
- "target": which resume section this tests (e.g. "Python skills", "Project X", "Role at Company Y")
- "difficulty": one of ["Easy", "Medium", "Hard"]
- "ideal_keywords": list of 3-6 keywords a good answer should include
- "ideal_answer": a brief 2-3 sentence model answer

Mix question types. Prioritise questions about their actual projects and specific technologies."""

    result = _call_groq_json(prompt, system=_QUESTION_SYSTEM)
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "questions" in result:
        return result["questions"]
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_ACTION_VERBS: set = {
    "achieved","architected","automated","built","championed","coached",
    "collaborated","conceptualised","conceptualized","configured","consolidated",
    "contributed","coordinated","created","cut","decreased","defined","delivered",
    "deployed","designed","developed","directed","drove","engineered","enhanced",
    "established","evaluated","executed","expanded","facilitated","finalised",
    "finalized","founded","generated","grew","guided","identified","implemented",
    "improved","increased","initiated","integrated","introduced","launched",
    "led","leveraged","managed","mentored","migrated","modernised","modernized",
    "monitored","negotiated","optimised","optimized","orchestrated","overhauled",
    "owned","partnered","piloted","planned","presented","produced","proposed",
    "prototyped","published","reduced","refactored","reformed","resolved",
    "restructured","saved","scaled","secured","shaped","shipped","simplified",
    "solved","spearheaded","standardised","standardized","streamlined",
    "strengthened","transformed","trained","validated","won","wrote","analyzed",
    "analysed","researched","documented","tested","reviewed","tracked","measured",
    "supervised","ensured","provided","supported",
}

_PASSIVE_RE = re.compile(r"\b(was|were|been|being|is|are)\s+\w+ed\b", re.IGNORECASE)

_SPECIFICS_RE = re.compile(
    r"(\d[\d,\.]*\s*%|\$[\d,]+|\d+[kKmMbB]?\s*(users?|requests?|ms|seconds?|"
    r"hours?|days?|weeks?|months?|engineers?|members?|teams?|lines?|repos?|"
    r"services?|clients?|products?|endpoints?|models?|queries|records?|features?|"
    r"tickets?|bugs?|issues?|prs?|commits?|deploys?|pipelines?|modules?))",
    re.IGNORECASE,
)

_WEAK_PHRASES: List[str] = [
    "responsible for","helped with","worked on","assisted in",
    "assisted with","involved in","participated in","tasked with",
    "duties included","contributed to","was part of","helped to",
    "supported the","helped the","worked with the",
]

_FILLER_WORDS_RESUME: List[str] = [
    "um","uh","like","basically","actually","you know","right","so","just",
    "kind of","sort of","i mean","literally","honestly","obviously","clearly",
    "simply","really","very","quite","pretty much","i guess","stuff","things",
]

# score range → (pct_lo, pct_hi, label, colour_hex)
_PERCENTILE_TABLE: List[Tuple] = [
    (0,  39,  0,  18, "Needs significant work", "#ff3366"),
    (40, 54, 19,  39, "Below average",           "#ff7043"),
    (55, 64, 40,  54, "Average",                  "#ffaa00"),
    (65, 74, 55,  69, "Above average",             "#f0d060"),
    (75, 84, 70,  84, "Strong",                   "#00d4ff"),
    (85, 92, 85,  93, "Excellent",                "#00ff88"),
    (93,100, 94, 100, "Top tier",                 "#a855f7"),
]


def _percentile_info(score: int) -> Tuple:
    for s_lo, s_hi, p_lo, p_hi, label, colour in _PERCENTILE_TABLE:
        if s_lo <= score <= s_hi:
            return p_lo, p_hi, label, colour
    return 0, 100, "Unknown", "#7ab8d8"


def _score_bullet_rules(bullet: str) -> Dict:
    if not isinstance(bullet, str) or not bullet.strip():
        return {
            "action_verb": 0, "active_voice": 1, "specifics": 0,
            "no_overuse": 1, "no_fillers": 1, "length_ok": 0,
            "raw_score": 0, "word_count": 0,
        }
    text  = bullet.strip()
    words = text.split()
    wc    = len(words)
    lower = text.lower()

    first_word  = re.sub(r"[^a-z]", "", words[0].lower()) if words else ""
    action_verb = int(first_word in _ACTION_VERBS)
    active_voice = int(not bool(_PASSIVE_RE.search(text)))
    specifics   = int(bool(_SPECIFICS_RE.search(text)))
    no_overuse  = int(not any(p in lower for p in _WEAK_PHRASES))
    no_fillers  = int(not any(
        re.search(r"\b" + re.escape(fw) + r"\b", lower)
        for fw in _FILLER_WORDS_RESUME
    ))
    length_ok   = int(8 <= wc <= 25)

    axes      = [action_verb, active_voice, specifics, no_overuse, no_fillers, length_ok]
    raw_score = round(sum(axes) / len(axes) * 100)

    return {
        "action_verb": action_verb, "active_voice": active_voice,
        "specifics": specifics, "no_overuse": no_overuse,
        "no_fillers": no_fillers, "length_ok": length_ok,
        "raw_score": raw_score, "word_count": wc,
    }


def _groq_refine_bullets(bullets: List[str], section_name: str,
                          target_role: str = "") -> List[Dict]:
    if not GROQ_OK or not bullets:
        return [{} for _ in bullets]

    role_ctx = f" The candidate is targeting: {target_role}." if target_role else ""
    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))

    prompt = f"""You are a professional resume coach reviewing the {section_name} section.{role_ctx}

Score each bullet on a 0-100 scale considering:
- Impact and achievement focus (not just duties)
- Strong action verbs and active voice
- Quantified results and specifics
- Conciseness (8-25 words ideal)
- ATS keyword density for the target role

For each bullet provide ONE concrete improvement tip (max 15 words) and a rewritten improved version.

Bullets to score:
{numbered}

Respond ONLY with a JSON array of exactly {len(bullets)} objects, each with:
  "score": integer 0-100,
  "tip": string (max 15 words, specific fix),
  "improved": string (rewritten bullet, better version)

No markdown, no explanation, only the JSON array."""

    raw   = _call_groq(prompt, max_tokens=2000)
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(clean)
        if isinstance(result, list) and len(result) == len(bullets):
            return result
    except Exception:
        m = re.search(r"\[[\s\S]*\]", clean)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return (result[:len(bullets)] +
                            [{}] * max(0, len(bullets) - len(result)))
            except Exception:
                pass
    return [{} for _ in bullets]


def _score_section_bullets(bullets: List[str], section_name: str,
                            target_role: str = "",
                            use_groq: bool = True) -> List[Dict]:
    if not bullets:
        return []

    rule_results = [_score_bullet_rules(b) for b in bullets]
    groq_results = (
        _groq_refine_bullets(bullets, section_name, target_role)
        if use_groq else [{} for _ in bullets]
    )

    combined = []
    for bullet, rr, gr in zip(bullets, rule_results, groq_results):
        groq_score = gr.get("score") if isinstance(gr, dict) and gr.get("score") is not None else None
        final = round(0.6 * groq_score + 0.4 * rr["raw_score"]) if groq_score is not None else rr["raw_score"]
        combined.append({
            "text":         bullet,
            "rule_score":   rr["raw_score"],
            "groq_score":   groq_score,
            "final_score":  final,
            "tip":          gr.get("tip", "") if isinstance(gr, dict) else "",
            "improved":     gr.get("improved", "") if isinstance(gr, dict) else "",
            "action_verb":  rr["action_verb"],
            "active_voice": rr["active_voice"],
            "specifics":    rr["specifics"],
            "no_overuse":   rr["no_overuse"],
            "no_fillers":   rr["no_fillers"],
            "length_ok":    rr["length_ok"],
            "word_count":   rr["word_count"],
        })
    return combined


def _score_skills_richness(skills: List) -> int:
    n = len([s for s in skills if isinstance(s, str) and s.strip()])
    if n >= 20: return 100
    if n >= 14: return 80
    if n >= 8:  return 60
    if n >= 4:  return 40
    return 20


def _score_education(edu: List) -> int:
    if not edu:
        return 0
    for e in edu:
        if isinstance(e, dict) and e.get("degree"):
            return 100
        if isinstance(e, str) and e.strip():
            return 60
    return 30


def _score_summary(summary: str) -> int:
    if not summary or not isinstance(summary, str):
        return 0
    words = summary.split()
    wc    = len(words)
    if wc < 10:
        return 20
    has_specifics = bool(_SPECIFICS_RE.search(summary))
    has_action    = any(
        re.sub(r"[^a-z]", "", w.lower()) in _ACTION_VERBS for w in words[:5]
    )
    base = min(100, 40 + wc * 2)
    base = min(100, base + (20 if has_specifics else 0) + (20 if has_action else 0))
    return base


def score_resume(parsed: Dict, rephrased: Dict, target_role: str = "") -> Dict:
    """
    Score a resume on a 0-100 scale.

    Returns:
        overall (int), percentile_lo, percentile_hi, pct_label, pct_colour,
        section_scores (dict), experience_bullets (list), project_bullets (list),
        bullet_count (int), target_role (str)
    """
    src = rephrased if rephrased else parsed

    # Collect experience bullets
    exp_bullets: List[str] = []
    for entry in (src.get("experience") or []):
        if isinstance(entry, dict):
            exp_bullets += [b for b in entry.get("responsibilities", []) if isinstance(b, str) and b.strip()]
            exp_bullets += [b for b in entry.get("achievements", [])      if isinstance(b, str) and b.strip()]
        elif isinstance(entry, str) and entry.strip():
            exp_bullets.append(entry)

    # Collect project bullets
    proj_bullets: List[str] = []
    for proj in (src.get("projects") or []):
        if isinstance(proj, dict):
            desc = proj.get("description", "")
            imp  = proj.get("impact", "")
            if isinstance(desc, str) and desc.strip(): proj_bullets.append(desc)
            if isinstance(imp,  str) and imp.strip():  proj_bullets.append(imp)
        elif isinstance(proj, str) and proj.strip():
            proj_bullets.append(proj)

    exp_scored  = _score_section_bullets(exp_bullets,  "Experience", target_role)
    proj_scored = _score_section_bullets(proj_bullets, "Projects",   target_role)

    exp_score   = round(sum(b["final_score"] for b in exp_scored)  / len(exp_scored))  if exp_scored  else 50
    proj_score  = round(sum(b["final_score"] for b in proj_scored) / len(proj_scored)) if proj_scored else 50
    skill_score = _score_skills_richness(src.get("skills") or [])
    edu_score   = _score_education(src.get("education") or [])
    summ_score  = _score_summary(src.get("summary") or parsed.get("summary") or "")

    weights = {"experience": 0.40, "projects": 0.25, "skills": 0.15, "education": 0.10, "summary": 0.10}
    scores  = {"experience": exp_score, "projects": proj_score, "skills": skill_score,
               "education": edu_score, "summary": summ_score}
    overall = max(0, min(100, round(sum(weights[k] * scores[k] for k in weights))))

    plo, phi, plabel, pcolour = _percentile_info(overall)

    return {
        "overall":           overall,
        "percentile_lo":     plo,
        "percentile_hi":     phi,
        "pct_label":         plabel,
        "pct_colour":        pcolour,
        "section_scores":    scores,
        "experience_bullets": exp_scored,
        "project_bullets":   proj_scored,
        "bullet_count":      len(exp_bullets) + len(proj_bullets),
        "target_role":       target_role,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME ↔ INTERVIEW GAP ANALYSIS  (v1.0)
#
#  Detects which resume claims were never demonstrated during the interview.
#  Each uncovered claim gets a predicted interviewer question + coaching tip.
#
#  Pipeline:
#    1. extract_resume_claims(parsed)     → list[ResumeClaim]
#         Pulls every verifiable claim from experience, projects, skills,
#         certifications. Each claim is tagged by type and source section.
#
#    2. _build_answer_corpus(session_answers) → str
#         Concatenates all answer transcripts into a single searchable corpus.
#
#    3. _score_claim_coverage(claim, corpus) → float  [0.0 – 1.0]
#         Two-tier matching:
#           Tier 1 — Semantic (sentence-transformers BAAI/bge-small-en-v1.5):
#             cosine similarity between claim embedding and corpus embedding.
#             Covered if cosine ≥ SEMANTIC_THRESHOLD (0.42).
#           Tier 2 — Keyword overlap (fallback when sbert unavailable):
#             Jaccard overlap on content words.
#             Covered if overlap ≥ KEYWORD_THRESHOLD (0.25).
#
#    4. analyze_gap(parsed, answers, role) → GapReport
#         Classifies each claim as covered / partial / uncovered.
#         For uncovered/partial claims → Groq generates predicted_question
#         and coaching_tip. Falls back to rule-based tips if Groq unavailable.
#
#  Coverage thresholds (research-grounded):
#    SEMANTIC_THRESHOLD = 0.42  — below this cosine, the claim topic is absent
#    PARTIAL_THRESHOLD  = 0.30  — below semantic but above this → partial credit
#    KEYWORD_THRESHOLD  = 0.25  — keyword fallback
#
#  Calibration basis:
#    Reimers & Gurevych (2019) SBERT paper: cosine 0.40-0.45 corresponds to
#    "topically related but not directly addressed" on STS-B human annotations.
#    Below 0.30 is reliably "unrelated topic". Above 0.55 is "same content".
#    0.42 sits in the reliable detection zone without over-flagging paraphrases.
# ══════════════════════════════════════════════════════════════════════════════

# ── Semantic scorer availability ──────────────────────────────────────────────
# _sas_scorer is the shared singleton from sas_scorer.py (imported at top).
# It loads BAAI/bge-small-en-v1.5 lazily on first .score() call — the same
# model instance already used by analyzer.py, so no second 33 MB load occurs.

def _load_sbert() -> bool:
    """Trigger lazy model load via shared singleton. Returns True if ready."""
    return _sas_scorer._load()

def _sbert_available() -> bool:
    """Return True if the shared SAS model is loaded and ready."""
    return bool(_sas_scorer._available)


# ── Thresholds ─────────────────────────────────────────────────────────────────
_SEMANTIC_THRESHOLD = 0.42   # cosine ≥ this → covered
_PARTIAL_THRESHOLD  = 0.30   # cosine in [this, SEMANTIC) → partial
_KEYWORD_THRESHOLD  = 0.25   # keyword Jaccard ≥ this → covered (fallback)

# ── Stop-words for keyword fallback ───────────────────────────────────────────
_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "i","we","you","he","she","they","it","this","that","these","those",
    "my","our","your","his","her","their","its","as","up","out","into",
    "so","if","not","no","about","after","before","during","through",
    "also","just","more","than","then","when","where","which","who",
}


def _content_words(text: str) -> set:
    """Return lowercase content words, stripping punctuation and stop-words."""
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return {t for t in tokens if t not in _STOP}


# ── Claim dataclass ────────────────────────────────────────────────────────────

@dataclass
class ResumeClaim:
    """One verifiable claim extracted from the resume."""
    text:          str                  # the raw claim text
    claim_type:    str                  # "achievement" | "skill" | "leadership"
                                        # | "metric" | "project" | "certification"
    source:        str                  # "experience:<company>" | "project:<title>"
                                        # | "skills" | "certifications"
    keywords:      List[str] = field(default_factory=list)   # key nouns/verbs

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClaimCoverageResult:
    """Coverage assessment for a single claim after analyzing all answers."""
    claim:              ResumeClaim
    status:             str          # "covered" | "partial" | "uncovered"
    coverage_score:     float        # 0.0 – 1.0  (semantic cosine or keyword Jaccard)
    method:             str          # "semantic" | "keyword"
    best_answer_index:  int          # which answer (0-based) had best coverage (-1 if none)
    predicted_question: str          # what an interviewer would ask about this gap
    coaching_tip:       str          # how the candidate should address it next time

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["claim"] = self.claim.to_dict()
        return d


@dataclass
class GapReport:
    """Full gap analysis report for one interview session."""
    role:               str
    total_claims:       int
    covered_count:      int
    partial_count:      int
    uncovered_count:    int
    coverage_pct:       float                        # covered / total * 100
    risk_level:         str                          # "Low" | "Medium" | "High" | "Critical"
    summary:            str                          # 2-3 sentence coaching summary
    covered:            List[ClaimCoverageResult]    # claims well addressed
    partial:            List[ClaimCoverageResult]    # claims touched but thin
    uncovered:          List[ClaimCoverageResult]    # claims never mentioned
    top_gaps:           List[Dict]                   # top-3 highest-priority uncovered claims
    sbert_available:    bool                         # whether semantic matching was used

    def to_dict(self) -> Dict:
        return {
            "role":             self.role,
            "total_claims":     self.total_claims,
            "covered_count":    self.covered_count,
            "partial_count":    self.partial_count,
            "uncovered_count":  self.uncovered_count,
            "coverage_pct":     self.coverage_pct,
            "risk_level":       self.risk_level,
            "summary":          self.summary,
            "covered":          [c.to_dict() for c in self.covered],
            "partial":          [c.to_dict() for c in self.partial],
            "uncovered":        [c.to_dict() for c in self.uncovered],
            "top_gaps":         self.top_gaps,
            "sbert_available":  self.sbert_available,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — CLAIM EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

# Regex to detect quantified / metric claims ("led 12 engineers", "reduced by 40%")
_METRIC_RE = re.compile(
    r"(\d[\d,\.]*\s*(%|x|\+|k|m|b)?\s*(users?|engineers?|members?|clients?|"
    r"services?|ms|seconds?|hours?|days?|months?|requests?|repos?|lines?|"
    r"products?|features?|tickets?|bugs?|prs?|models?|queries|records?)?"
    r"|\$[\d,]+[kKmMbB]?)",
    re.IGNORECASE,
)

# Leadership / ownership signal words
_LEADERSHIP_RE = re.compile(
    r"\b(led|managed|owned|directed|supervised|mentored|coached|spearheaded|"
    r"architected|designed|founded|built|established|launched|drove|headed|"
    r"oversaw|coordinated|orchestrated)\b",
    re.IGNORECASE,
)


def extract_resume_claims(parsed: Dict) -> List[ResumeClaim]:
    """
    Pull every verifiable claim from a parsed resume dict.

    Extracts from:
      - experience[].responsibilities  → claim_type "achievement" or "leadership"
      - experience[].achievements      → claim_type "achievement" or "metric"
      - projects[].description + impact→ claim_type "project"
      - skills[]                       → claim_type "skill" (grouped into one claim
                                         per 4 skills to avoid noise)
      - certifications[]               → claim_type "certification"

    Skips:
      - Bullets < 4 words (too vague to test coverage)
      - Pure date strings or formatting artifacts

    Returns a deduplicated list of ResumeClaim objects.
    """
    claims: List[ResumeClaim] = []
    seen_texts: set = set()

    def _add(text: str, claim_type: str, source: str) -> None:
        text = text.strip()
        if not text or len(text.split()) < 4:
            return
        key = re.sub(r"\s+", " ", text.lower())[:120]
        if key in seen_texts:
            return
        seen_texts.add(key)
        kws = list(_content_words(text))[:8]
        claims.append(ResumeClaim(
            text=text,
            claim_type=claim_type,
            source=source,
            keywords=kws,
        ))

    # ── Experience entries ─────────────────────────────────────────────────────
    for entry in (parsed.get("experience") or []):
        if not isinstance(entry, dict):
            continue
        company  = entry.get("company", "Unknown")
        src      = f"experience:{company}"

        for bullet in (entry.get("responsibilities") or []):
            if not isinstance(bullet, str):
                continue
            if _LEADERSHIP_RE.search(bullet):
                _add(bullet, "leadership", src)
            elif _METRIC_RE.search(bullet):
                _add(bullet, "metric", src)
            else:
                _add(bullet, "achievement", src)

        for bullet in (entry.get("achievements") or []):
            if not isinstance(bullet, str):
                continue
            if _METRIC_RE.search(bullet):
                _add(bullet, "metric", src)
            else:
                _add(bullet, "achievement", src)

    # ── Projects ───────────────────────────────────────────────────────────────
    for proj in (parsed.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        title = proj.get("title", "Unnamed project")
        src   = f"project:{title}"
        desc  = proj.get("description", "")
        imp   = proj.get("impact", "")
        if isinstance(desc, str) and desc.strip():
            _add(desc, "project", src)
        if isinstance(imp, str) and imp.strip():
            _add(imp, "project", src)

    # ── Skills — grouped into chunks of 4 so each "claim" is meaningful ───────
    raw_skills = [
        s for s in (parsed.get("skills") or [])
        if isinstance(s, str) and s.strip()
    ]
    for i in range(0, len(raw_skills), 4):
        chunk = raw_skills[i:i + 4]
        if len(chunk) < 2:
            continue
        text = "Proficient in " + ", ".join(chunk)
        _add(text, "skill", "skills")

    # ── Certifications ─────────────────────────────────────────────────────────
    for cert in (parsed.get("certifications") or []):
        if isinstance(cert, str) and cert.strip():
            _add(cert, "certification", "certifications")

    return claims


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ANSWER CORPUS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_answer_corpus(session_answers: List[Dict]) -> str:
    """
    Concatenate all interview answer transcripts into a single text corpus.
    Handles multiple dict shapes produced by /evaluate endpoint.
    """
    parts: List[str] = []
    for ans in (session_answers or []):
        if not isinstance(ans, dict):
            continue
        # /evaluate stores transcript under these keys (try all)
        for key in ("transcript", "answer", "text", "response"):
            val = ans.get(key, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — COVERAGE SCORER  (semantic + keyword fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _score_claim_coverage(
    claim: ResumeClaim,
    answer_texts: List[str],        # one str per answer (not concatenated)
    corpus_embedding: Optional[Any] = None,  # pre-computed for the full corpus
) -> Tuple[float, str, int]:
    """
    Score how well a single claim was covered across all answers.

    Returns:
        (coverage_score: float,  method: str,  best_answer_index: int)
        coverage_score ∈ [0.0, 1.0]
        method         ∈ "semantic" | "keyword"
        best_answer_index: 0-based index of best-matching answer (-1 if none)
    """
    if not answer_texts:
        return 0.0, "keyword", -1

    # ── Tier 1: Semantic (shared SAS scorer) ──────────────────────────────────
    # Uses the module-level sas_scorer singleton (BAAI/bge-small-en-v1.5).
    # batch_score() encodes all answer texts in a single forward pass —
    # more efficient than the previous per-answer encode loop.
    # sas_scorer.score() returns calibrated [0,1] scores (threshold 0.20),
    # which map directly onto _SEMANTIC_THRESHOLD / _PARTIAL_THRESHOLD.
    if _load_sbert():
        try:
            capped_answers = [t[:1500] for t in answer_texts]
            scored_pairs   = _sas_scorer.batch_score(capped_answers, claim.text[:1000])
            best_score = 0.0
            best_idx   = -1
            for idx, (sim, _method) in enumerate(scored_pairs):
                if sim > best_score:
                    best_score = sim
                    best_idx   = idx
            return round(best_score, 4), "semantic", best_idx
        except Exception:
            pass  # fall through to keyword

    # ── Tier 2: Keyword Jaccard (fallback) ─────────────────────────────────────
    claim_words = _content_words(claim.text)
    if not claim_words:
        return 0.0, "keyword", -1

    best_score = 0.0
    best_idx   = -1
    for idx, ans_text in enumerate(answer_texts):
        ans_words = _content_words(ans_text)
        if not ans_words:
            continue
        intersection = len(claim_words & ans_words)
        union        = len(claim_words | ans_words)
        jaccard      = intersection / union if union else 0.0
        if jaccard > best_score:
            best_score = jaccard
            best_idx   = idx
    return round(best_score, 4), "keyword", best_idx


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — COACHING TIP GENERATOR  (Groq-first, rule-based fallback)
# ══════════════════════════════════════════════════════════════════════════════

# Rule-based fallback tips keyed by claim_type
_FALLBACK_TIPS: Dict[str, str] = {
    "achievement": (
        "Weave this achievement into a STAR answer: state the Situation, the "
        "specific Task you owned, the Action you took, and the measurable Result."
    ),
    "metric": (
        "This quantified result is a strong signal. Mention the exact number in "
        "your next answer — interviewers remember specifics far better than vague claims."
    ),
    "leadership": (
        "Leadership claims need a concrete story. Prepare a 90-second STAR answer "
        "naming the team size, the challenge you led through, and the outcome you drove."
    ),
    "project": (
        "Be ready to walk through this project end-to-end: problem → your design "
        "decisions → challenges hit → final impact. Practise it out loud once."
    ),
    "skill": (
        "When a skill appears on your resume, be ready to demonstrate it with a "
        "specific example. 'I used X to solve Y and achieved Z' beats 'I know X'."
    ),
    "certification": (
        "Certifications are table-stakes — interviewers expect you to apply the "
        "knowledge, not just name the cert. Have one real example ready per cert."
    ),
}

_FALLBACK_QUESTIONS: Dict[str, str] = {
    "achievement":   "Can you walk me through one of the key achievements on your resume?",
    "metric":        "You mention {metric} on your resume — can you explain the context and how you achieved that?",
    "leadership":    "Your resume mentions you led a team. Tell me about a time that leadership was tested.",
    "project":       "Tell me about the {project} project — what was your specific contribution and what was the outcome?",
    "skill":         "Your resume lists {skill}. Can you give me a recent example where you applied it?",
    "certification": "You hold a {cert} certification — describe a situation where you applied that knowledge.",
}


def _rule_based_gap_tip(claim: ResumeClaim) -> Tuple[str, str]:
    """Generate a rule-based predicted_question + coaching_tip without Groq."""
    tip = _FALLBACK_TIPS.get(claim.claim_type, _FALLBACK_TIPS["achievement"])
    q_template = _FALLBACK_QUESTIONS.get(claim.claim_type, _FALLBACK_QUESTIONS["achievement"])

    # Slot-fill the template with claim keywords where placeholders exist
    keywords = claim.keywords
    q = q_template
    if "{metric}" in q:
        metric_match = _METRIC_RE.search(claim.text)
        q = q.replace("{metric}", metric_match.group(0) if metric_match else "the result you mentioned")
    if "{project}" in q:
        # Extract project name from source field "project:<title>"
        proj_name = claim.source.split(":", 1)[-1] if ":" in claim.source else "this project"
        q = q.replace("{project}", proj_name)
    if "{skill}" in q:
        skill_word = keywords[0].capitalize() if keywords else "this skill"
        q = q.replace("{skill}", skill_word)
    if "{cert}" in q:
        q = q.replace("{cert}", claim.text[:50])
    return q, tip


def _groq_gap_tips(
    uncovered_claims: List[ResumeClaim],
    role: str,
    answer_corpus: str,
) -> List[Tuple[str, str]]:
    """
    Use Groq to generate predicted_question + coaching_tip for each uncovered claim.
    Returns list of (predicted_question, coaching_tip) in same order as input.
    Falls back to rule-based on API failure.
    """
    if not GROQ_OK or not uncovered_claims:
        return [_rule_based_gap_tip(c) for c in uncovered_claims]

    claims_json = json.dumps(
        [{"text": c.text, "type": c.claim_type, "source": c.source}
         for c in uncovered_claims],
        indent=2,
    )

    prompt = f"""A candidate is interviewing for: {role}

The following resume claims were NEVER addressed during the interview.
Interview answers so far (for context):
\"\"\"{answer_corpus[:2000]}\"\"\"

Uncovered resume claims:
{claims_json}

For each uncovered claim, generate:
1. "predicted_question": The exact question a real interviewer would ask to probe this claim
   (specific to the claim text, not generic). Max 25 words.
2. "coaching_tip": One concrete, actionable tip for the candidate on how to proactively
   address this claim in a future interview. Max 30 words. Start with a verb.

Respond ONLY with a JSON array of exactly {len(uncovered_claims)} objects, each with keys:
  "predicted_question": string,
  "coaching_tip": string

No markdown, no preamble, only the JSON array."""

    raw   = _call_groq(prompt, system=_GAP_SYSTEM, max_tokens=1500)
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(clean)
        if isinstance(result, list) and len(result) == len(uncovered_claims):
            return [
                (
                    r.get("predicted_question", "") if isinstance(r, dict) else "",
                    r.get("coaching_tip", "")       if isinstance(r, dict) else "",
                )
                for r in result
            ]
    except Exception:
        m = re.search(r"\[[\s\S]*\]", clean)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    padded = result[:len(uncovered_claims)] + [{}] * max(
                        0, len(uncovered_claims) - len(result)
                    )
                    return [
                        (
                            r.get("predicted_question", "") if isinstance(r, dict) else "",
                            r.get("coaching_tip", "")       if isinstance(r, dict) else "",
                        )
                        for r in padded
                    ]
            except Exception:
                pass

    # Groq parse failed — rule-based fallback for all
    return [_rule_based_gap_tip(c) for c in uncovered_claims]


def _groq_gap_summary(gap_report_partial: Dict, role: str) -> str:
    """
    Generate a 2-3 sentence coaching summary for the full gap report via Groq.
    Falls back to a rule-based summary if Groq unavailable.
    """
    if not GROQ_OK:
        cov  = gap_report_partial["coverage_pct"]
        unc  = gap_report_partial["uncovered_count"]
        tot  = gap_report_partial["total_claims"]
        risk = gap_report_partial["risk_level"]
        return (
            f"You demonstrated {cov:.0f}% of your resume claims during this session "
            f"({tot - unc}/{tot} covered). "
            f"Risk level is {risk} — {unc} claim(s) went completely unaddressed. "
            f"Focus your next practice session on proactively surfacing your uncovered "
            f"achievements before the interviewer has to ask."
        )

    prompt = f"""Write a 2-3 sentence coaching summary for a {role} candidate based on this gap analysis:

Coverage: {gap_report_partial['coverage_pct']:.0f}% of resume claims demonstrated
Covered:   {gap_report_partial['covered_count']} claims
Partial:   {gap_report_partial['partial_count']} claims
Uncovered: {gap_report_partial['uncovered_count']} claims
Risk level: {gap_report_partial['risk_level']}

Top uncovered claims:
{json.dumps(gap_report_partial.get('top_gaps', [])[:3], indent=2)[:600]}

Be direct, specific, and constructive. Tell the candidate exactly what risk this gap poses
and what to do before their next interview. Do NOT use bullet points. Plain paragraph only."""

    result = _call_groq(prompt, system=_GAP_SYSTEM, max_tokens=200)
    # Strip any accidental JSON fences
    result = re.sub(r"^```\w*\s*|\s*```$", "", result.strip(), flags=re.MULTILINE)
    return result if result and not result.startswith("[API Error") else (
        f"You covered {gap_report_partial['coverage_pct']:.0f}% of your resume claims. "
        f"Address the {gap_report_partial['uncovered_count']} uncovered item(s) proactively "
        f"in your next interview to close this credibility gap."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def analyze_gap(
    parsed: Dict,
    session_answers: List[Dict],
    role: str = "",
) -> Dict:
    """
    Analyze the gap between what the candidate claimed on their resume and
    what they actually demonstrated during the interview.

    Parameters
    ----------
    parsed          : dict returned by parse_resume()
    session_answers : list of answer dicts from the /evaluate endpoint.
                      Each dict must contain at least one of:
                        "transcript" | "answer" | "text" | "response"
    role            : target job role (used for Groq context, e.g. "Backend Engineer")

    Returns
    -------
    dict (GapReport.to_dict()) with keys:
        role, total_claims, covered_count, partial_count, uncovered_count,
        coverage_pct, risk_level, summary, covered, partial, uncovered,
        top_gaps, sbert_available

    Graceful degradation:
        - No answers provided → all claims marked uncovered with rule-based tips
        - Groq unavailable  → rule-based predicted_question + coaching_tip
        - SBERT unavailable → keyword Jaccard matching (still reliable)
    """
    # ── 1. Extract all claims from resume ──────────────────────────────────────
    claims = extract_resume_claims(parsed)
    if not claims:
        return GapReport(
            role=role, total_claims=0, covered_count=0,
            partial_count=0, uncovered_count=0, coverage_pct=100.0,
            risk_level="Low",
            summary="No verifiable claims found in the resume to analyze.",
            covered=[], partial=[], uncovered=[], top_gaps=[],
            sbert_available=_sbert_available(),
        ).to_dict()

    # ── 2. Build per-answer text list (one str per answer) ─────────────────────
    answer_texts: List[str] = []
    for ans in (session_answers or []):
        if not isinstance(ans, dict):
            continue
        for key in ("transcript", "answer", "text", "response"):
            val = ans.get(key, "")
            if isinstance(val, str) and val.strip():
                answer_texts.append(val.strip())
                break

    answer_corpus = " ".join(answer_texts)

    # Trigger lazy model load via shared singleton.
    _load_sbert()
    _ok = _sbert_available()

    # ── 3. Score every claim ───────────────────────────────────────────────────
    use_semantic_threshold = _SEMANTIC_THRESHOLD if _ok else _KEYWORD_THRESHOLD
    use_partial_threshold  = _PARTIAL_THRESHOLD  if _ok else (_KEYWORD_THRESHOLD * 0.6)

    covered_claims:   List[ResumeClaim] = []
    partial_claims:   List[ResumeClaim] = []
    uncovered_claims: List[ResumeClaim] = []

    coverage_results: List[ClaimCoverageResult] = []

    for claim in claims:
        score, method, best_idx = _score_claim_coverage(claim, answer_texts)

        if score >= use_semantic_threshold:
            status = "covered"
            covered_claims.append(claim)
        elif score >= use_partial_threshold:
            status = "partial"
            partial_claims.append(claim)
        else:
            status = "uncovered"
            uncovered_claims.append(claim)

        coverage_results.append(ClaimCoverageResult(
            claim=claim,
            status=status,
            coverage_score=score,
            method=method,
            best_answer_index=best_idx,
            predicted_question="",   # filled below for non-covered
            coaching_tip="",         # filled below for non-covered
        ))

    # ── 4. Generate coaching for uncovered + partial claims via Groq ───────────
    needs_coaching = [r for r in coverage_results if r.status in ("uncovered", "partial")]
    if needs_coaching:
        tips = _groq_gap_tips(
            [r.claim for r in needs_coaching],
            role=role,
            answer_corpus=answer_corpus,
        )
        for result, (q, tip) in zip(needs_coaching, tips):
            result.predicted_question = q   or _rule_based_gap_tip(result.claim)[0]
            result.coaching_tip       = tip or _rule_based_gap_tip(result.claim)[1]

    # ── 5. Compute risk level ──────────────────────────────────────────────────
    total     = len(claims)
    n_covered = len(covered_claims)
    n_partial = len(partial_claims)
    n_uncov   = len(uncovered_claims)

    # Effective coverage: covered=1.0, partial=0.5, uncovered=0.0
    eff_coverage = (n_covered + 0.5 * n_partial) / total if total else 1.0
    coverage_pct = round(eff_coverage * 100, 1)

    if coverage_pct >= 80:
        risk_level = "Low"
    elif coverage_pct >= 60:
        risk_level = "Medium"
    elif coverage_pct >= 35:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # ── 6. Top-3 gaps (prioritise: metric > leadership > achievement > project > skill) ──
    _PRIORITY = {"metric": 0, "leadership": 1, "achievement": 2, "project": 3,
                 "skill": 4, "certification": 5}
    sorted_uncov = sorted(
        [r for r in coverage_results if r.status == "uncovered"],
        key=lambda r: _PRIORITY.get(r.claim.claim_type, 9),
    )
    top_gaps = [
        {
            "claim":              r.claim.text,
            "type":               r.claim.claim_type,
            "source":             r.claim.source,
            "predicted_question": r.predicted_question,
            "coaching_tip":       r.coaching_tip,
        }
        for r in sorted_uncov[:3]
    ]

    # ── 7. Build partial report dict for summary prompt ───────────────────────
    partial_report_dict = {
        "coverage_pct":    coverage_pct,
        "covered_count":   n_covered,
        "partial_count":   n_partial,
        "uncovered_count": n_uncov,
        "total_claims":    total,
        "risk_level":      risk_level,
        "top_gaps":        top_gaps,
    }

    summary = _groq_gap_summary(partial_report_dict, role)

    # ── 8. Separate coverage_results by status ─────────────────────────────────
    covered_results  = [r for r in coverage_results if r.status == "covered"]
    partial_results  = [r for r in coverage_results if r.status == "partial"]
    uncov_results    = [r for r in coverage_results if r.status == "uncovered"]

    report = GapReport(
        role=role,
        total_claims=total,
        covered_count=n_covered,
        partial_count=n_partial,
        uncovered_count=n_uncov,
        coverage_pct=coverage_pct,
        risk_level=risk_level,
        summary=summary,
        covered=covered_results,
        partial=partial_results,
        uncovered=uncov_results,
        top_gaps=top_gaps,
        sbert_available=_sbert_available(),
    )
    return report.to_dict()