"""
culture_fit_engine.py — Aura AI | Company Culture Fit Predictor (v2.0)
=======================================================================
Feature 05: Company Culture Fit Predictor
  • Matches the candidate's communication style (extracted from session answers)
    to a company's known interview culture using Groq LLM scoring.
  • Computes a per-company fit percentage and a ranked fit table.
    Profiles are stored in COMPANY_CULTURE_PROFILES below (20 major companies).
  • Output: "Your communication style is a 72% fit for Google's interview culture
    but only a 41% fit for McKinsey's. McKinsey interviewers prefer structured
    frameworks — here's how to adjust."

HOW IT WORKS (v2.0 — Groq backend)
────────────────────────────────────
1.  _build_candidate_style_doc(answers)
      Concatenates all session answers into a single style document.
      Extracts tone markers: directness, storytelling, data-first, structured
      frameworks, humility signals.

2.  _score_batch_with_groq(candidate_doc, companies, groq_api_key)
      Sends ONE Groq call with all company profiles in a single prompt.
      The model returns a JSON object: { "Company": score_0_to_100, ... }
      Much faster than one call per company.

3.  compute_all_fits(answers, groq_api_key)
      Scores candidate against all loaded company profiles via Groq.
      Falls back to keyword-overlap heuristic if Groq is unavailable.
      Sorts by fit descending.  Returns list of CompanyFitResult dataclasses.

4.  render_culture_fit_section(answers, target_co, groq_api_key, accent_color)
      Renders the fit report: top 3 matches + bottom 3 mismatches + adjustment
      tips for the target company (if set in session state).

INTEGRATION in app.py
─────────────────────
After page_report() builds the session report dict, call:

    from culture_fit_engine import render_culture_fit_section
    render_culture_fit_section(
        answers      = report.get("answers", []),
        target_co    = st.session_state.get("target_company", ""),
        groq_api_key = os.environ.get("GROQ_API_KEY", ""),
        accent_color = "#00e5ff",
    )

SESSION STATE KEYS
──────────────────
  culture_fit_results   list[CompanyFitResult] — cached after first compute
  culture_fit_target    str  — company name used for adjustment tips
  culture_fit_ts        float — unix timestamp of last compute (for cache TTL)

FALLBACK
────────
  If Groq is unavailable or the API key is missing, falls back to a
  keyword-overlap heuristic scoring.  Scores are labelled "[Heuristic]" in UI.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import streamlit as st

# ── Groq (primary scoring backend) ───────────────────────────────────────────
try:
    from groq import Groq as _Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# SBERT/TF-IDF no longer used — kept as False stubs for any legacy references
SBERT_OK = False
TFIDF_OK = False

_GROQ_MODEL        = "llama-3.3-70b-versatile"
_GROQ_SCORING_MODEL = "llama-3.3-70b-versatile"  # used for batch fit scoring
_CACHE_TTL         = 3600 * 6   # 6 hours

# ══════════════════════════════════════════════════════════════════════════════
#  COMPANY CULTURE PROFILES
#  Each entry describes "what interviewers at Company X value" — the text that
#  gets embedded and compared against the candidate's answer style.
# ══════════════════════════════════════════════════════════════════════════════

COMPANY_CULTURE_PROFILES: Dict[str, Dict] = {
    "Google": {
        "style_descriptor": (
            "Interviewers at Google value structured analytical thinking, data-driven decisions, "
            "and the ability to break down ambiguous problems into clear sub-problems. "
            "They appreciate Googleyness: intellectual humility, collaborative spirit, comfort with ambiguity, "
            "and a bias toward impact at scale.  Answers should be concise yet thorough, backed by metrics, "
            "and demonstrate systems-level thinking.  STAR format is appreciated but not mandatory — "
            "the focus is on reasoning process and quantifiable outcomes."
        ),
        "key_values":   ["data-driven", "systems thinking", "intellectual humility", "scalable impact", "ambiguity comfort"],
        "avoid":        ["vague storytelling without metrics", "credit-hoarding language", "over-engineered answers"],
        "tip":          "Lead with the problem structure, back every claim with a number, and show how your solution scales.",
        "fit_color":    "#4285F4",
    },
    "Meta": {
        "style_descriptor": (
            "Meta interviewers prize boldness and directness.  Candidates are expected to move fast, "
            "challenge assumptions, and show a bias for action even when data is incomplete.  "
            "Product sense is critical — frame answers around user impact and business growth.  "
            "Authenticity and conviction matter more than polish.  Show that you have strong opinions "
            "but hold them loosely when presented with counter-evidence."
        ),
        "key_values":   ["move fast", "bold decisions", "user impact", "product sense", "conviction"],
        "avoid":        ["excessive hedging", "over-process focus", "analysis paralysis"],
        "tip":          "Be direct, show your reasoning quickly, and always anchor back to user and business impact.",
        "fit_color":    "#0668E1",
    },
    "Amazon": {
        "style_descriptor": (
            "Amazon interviews are almost entirely structured around the 14 Leadership Principles.  "
            "Interviewers expect precise STAR-format answers (Situation, Task, Action, Result) "
            "with strong ownership language ('I decided', 'I drove', 'I was accountable').  "
            "Customer obsession is the top signal — every answer should connect back to the customer.  "
            "Frugality and long-term thinking are prized; short-term wins at the expense of the customer "
            "are viewed negatively.  Quantify results wherever possible."
        ),
        "key_values":   ["ownership", "customer obsession", "frugality", "STAR structure", "long-term thinking"],
        "avoid":        ["passive voice ('we did')", "unquantified results", "ignoring customer angle"],
        "tip":          "Use 'I' not 'we', quantify every result, and explicitly map your story to an LP.",
        "fit_color":    "#FF9900",
    },
    "Microsoft": {
        "style_descriptor": (
            "Microsoft values growth mindset — the willingness to learn, fail, and improve.  "
            "Interviewers look for collaborative candidates who can work across diverse teams and geographies.  "
            "Technical depth matters but so does empathy and communication.  "
            "Answers should reflect curiosity, adaptability, and a genuine desire to help customers and colleagues succeed.  "
            "Being prescriptive about solutions while remaining open to feedback is the ideal balance."
        ),
        "key_values":   ["growth mindset", "collaboration", "empathy", "adaptability", "technical depth"],
        "avoid":        ["fixed mindset language", "lone-wolf narratives", "dismissiveness toward feedback"],
        "tip":          "Show how you grew from a challenge, emphasise collaboration, and use empathetic framing.",
        "fit_color":    "#00A4EF",
    },
    "Apple": {
        "style_descriptor": (
            "Apple interviewers look for people who obsess over craft and quality.  "
            "Answers should demonstrate extreme attention to detail, a user-centric design philosophy, "
            "and a high bar for what 'done' means.  Candidates who cut corners are red flags.  "
            "Directness and brevity are valued — Apple engineers communicate precisely.  "
            "Show pride in your best work and be prepared to defend design decisions rigorously."
        ),
        "key_values":   ["craftsmanship", "quality bar", "design thinking", "precision", "user focus"],
        "avoid":        ["vague 'good enough' answers", "sloppy communication", "shipping for speed at cost of quality"],
        "tip":          "Highlight the quality bar you set, walk through specific design trade-offs, and show user empathy.",
        "fit_color":    "#999999",
    },
    "McKinsey": {
        "style_descriptor": (
            "McKinsey interviewers expect highly structured, framework-driven answers.  "
            "The MECE principle (Mutually Exclusive, Collectively Exhaustive) should underpin every answer.  "
            "Candidates should synthesise top-down (pyramid principle), state the conclusion first, "
            "then support with evidence.  Analytical rigour, hypothesis-first thinking, and clear "
            "communication to senior stakeholders are prized.  Storytelling is secondary to structure."
        ),
        "key_values":   ["MECE structure", "top-down synthesis", "analytical rigour", "hypothesis-first", "executive communication"],
        "avoid":        ["bottom-up rambling", "unstructured storytelling", "missing quantification"],
        "tip":          "State your conclusion first (pyramid principle), structure with MECE buckets, and drive to a clear recommendation.",
        "fit_color":    "#003189",
    },
    "Goldman Sachs": {
        "style_descriptor": (
            "Goldman Sachs interviewers value precision, commercial acumen, and composure under pressure.  "
            "Candidates should demonstrate deep market knowledge, quantitative fluency, and the ability "
            "to communicate complex financial concepts clearly.  Teamwork and client-service orientation "
            "are critical.  Answers should be succinct, numbers-first, and show awareness of macro context."
        ),
        "key_values":   ["commercial acumen", "quantitative fluency", "composure", "client focus", "market awareness"],
        "avoid":        ["vague financial reasoning", "inability to handle pushback", "excessive narrative over numbers"],
        "tip":          "Lead with numbers, show market awareness, and demonstrate calm under intellectual pressure.",
        "fit_color":    "#6AADE4",
    },
    "Netflix": {
        "style_descriptor": (
            "Netflix prizes freedom and responsibility.  Interviewers look for exceptional individuals "
            "who exercise good judgment without needing detailed rules or process guardrails.  "
            "Radical candour — giving and receiving direct feedback — is expected.  "
            "Candidates should show they have acted with context not control, driven results, "
            "and prioritised the company's long-term success over short-term comfort.  "
            "Avoid corporate-speak; authenticity is key."
        ),
        "key_values":   ["freedom and responsibility", "radical candour", "good judgment", "context not control", "authenticity"],
        "avoid":        ["rule-following language", "avoiding conflict", "corporate jargon"],
        "tip":          "Be radically honest, show you make judgment calls without being told, and demonstrate long-term thinking.",
        "fit_color":    "#E50914",
    },
    "Stripe": {
        "style_descriptor": (
            "Stripe interviewers look for intellectual curiosity, first-principles thinking, and extreme ownership.  "
            "The culture values thoughtful, written communication — candidates who can structure their "
            "thinking clearly in prose are advantaged.  Show deep craftsmanship, a bias toward simplicity, "
            "and the ability to zoom between the macro vision and the micro detail.  "
            "Being 'directionally bold and precisely humble' is the ideal posture."
        ),
        "key_values":   ["first principles", "intellectual curiosity", "written clarity", "craftsmanship", "ownership"],
        "avoid":        ["sloppy reasoning", "over-engineered solutions", "buzzword-heavy answers"],
        "tip":          "Think out loud from first principles, show elegant simplicity in your solution, and own the outcome.",
        "fit_color":    "#635BFF",
    },
    "Uber": {
        "style_descriptor": (
            "Uber interviewers value hustle, data fluency, and the ability to thrive in ambiguity.  "
            "Candidates should show they can build and iterate fast, make decisions with incomplete data, "
            "and balance short-term execution with long-term strategy.  Marketplace thinking — "
            "understanding supply, demand, and network effects — is prized for product and ops roles.  "
            "Show grit and resilience in the face of setbacks."
        ),
        "key_values":   ["hustle", "data fluency", "ambiguity tolerance", "marketplace thinking", "resilience"],
        "avoid":        ["slow methodical approach without speed", "ignoring data", "giving up at first obstacle"],
        "tip":          "Show how you shipped fast, used data to decide, and kept going when things got hard.",
        "fit_color":    "#000000",
    },
    "Airbnb": {
        "style_descriptor": (
            "Airbnb interviewers look for candidates who embody the 'belong anywhere' mission.  "
            "Empathy and community-first thinking are paramount.  Answers should reflect a deep "
            "understanding of the host-guest relationship and the importance of trust in two-sided markets.  "
            "Creativity, design sensibility, and cross-cultural awareness are valued.  "
            "Show that you care about people, not just products."
        ),
        "key_values":   ["empathy", "community", "design sensibility", "trust", "mission alignment"],
        "avoid":        ["purely business-metric framing without human angle", "lack of creativity", "ignoring culture"],
        "tip":          "Centre your answers on the human experience, show empathy for both sides of the marketplace.",
        "fit_color":    "#FF5A5F",
    },
    "Infosys": {
        "style_descriptor": (
            "Infosys interviewers value reliability, process adherence, and client-service orientation.  "
            "Candidates should demonstrate strong foundational technical skills, the ability to work "
            "in large collaborative teams, and a commitment to continuous learning.  "
            "Communication clarity and professionalism are key.  Show respect for processes and deadlines."
        ),
        "key_values":   ["reliability", "client service", "teamwork", "continuous learning", "process discipline"],
        "avoid":        ["lone-wolf narratives", "dismissing process", "vague skill claims"],
        "tip":          "Emphasise reliability, client impact, and your commitment to learning and process excellence.",
        "fit_color":    "#007CC3",
    },
    "TCS": {
        "style_descriptor": (
            "TCS interviewers look for adaptability, a service mindset, and strong communication skills.  "
            "The culture values lifelong learning and the ability to upskill rapidly across domains.  "
            "Answers should reflect a team-first orientation, customer satisfaction, and pride in delivery.  "
            "Candidates who show a global mindset and multicultural awareness stand out."
        ),
        "key_values":   ["adaptability", "service mindset", "lifelong learning", "delivery pride", "global mindset"],
        "avoid":        ["narrow specialisation without breadth", "customer-distant framing", "rigid thinking"],
        "tip":          "Show breadth of learning, service mindset, and pride in reliable delivery for the client.",
        "fit_color":    "#00A1E0",
    },
    "Wipro": {
        "style_descriptor": (
            "Wipro interviewers value integrity, a solutions-first mindset, and collaborative execution.  "
            "The culture emphasises respect for diversity, customer-centricity, and the ability to "
            "navigate change in a fast-moving technology landscape.  Strong communication and "
            "a proactive attitude toward problem-solving are essential."
        ),
        "key_values":   ["integrity", "solutions-first", "customer centricity", "diversity respect", "proactivity"],
        "avoid":        ["reactive waiting for instructions", "vague answers", "ignoring ethical dimensions"],
        "tip":          "Highlight proactive problem-solving, integrity, and customer impact in all your answers.",
        "fit_color":    "#9B59B6",
    },
    "Deloitte": {
        "style_descriptor": (
            "Deloitte interviewers look for structured thinking, professional presence, and the "
            "ability to manage multiple client engagements simultaneously.  "
            "Leadership potential, teamwork, and a consulting mindset (issue → hypothesis → analysis → recommendation) "
            "are valued.  Candidates should show commercial awareness and the ability to build trust with senior clients."
        ),
        "key_values":   ["structured thinking", "professional presence", "client trust", "commercial awareness", "leadership potential"],
        "avoid":        ["unstructured brainstorming", "poor executive communication", "single-task focus only"],
        "tip":          "Use the consulting issue → hypothesis → analysis → recommendation structure, and demonstrate client empathy.",
        "fit_color":    "#86BC25",
    },
    "Flipkart": {
        "style_descriptor": (
            "Flipkart interviewers prize speed, frugality, and deep customer empathy for the Indian market.  "
            "Candidates should show they can build for scale in a resource-constrained environment, "
            "make quick data-driven decisions, and understand the nuances of Tier 2/3 Indian consumers.  "
            "Ownership and entrepreneurial drive are critical success markers."
        ),
        "key_values":   ["speed", "frugality", "Indian market empathy", "scale thinking", "entrepreneurial drive"],
        "avoid":        ["over-engineered solutions", "Western market-only framing", "slow decision making"],
        "tip":          "Show you can ship fast, build cheap-but-robust systems, and deeply understand the Indian consumer.",
        "fit_color":    "#F6A623",
    },
    "Salesforce": {
        "style_descriptor": (
            "Salesforce interviewers value the Ohana culture — family, trust, and giving back.  "
            "Candidates should demonstrate customer success orientation, equality as a core value, "
            "and a collaborative, low-ego working style.  "
            "Show pride in helping customers achieve their goals, and enthusiasm for the CRM mission."
        ),
        "key_values":   ["Ohana culture", "customer success", "equality", "collaboration", "low ego"],
        "avoid":        ["credit-grabbing", "ignoring the social impact angle", "purely technical framing"],
        "tip":          "Lead with customer success, show collaborative spirit, and connect your work to the Ohana mission.",
        "fit_color":    "#00A1E0",
    },
    "Adobe": {
        "style_descriptor": (
            "Adobe interviewers look for creative problem-solving, a passion for empowering creators, "
            "and the ability to balance business impact with design excellence.  "
            "Candidates should show customer empathy, cross-functional collaboration skills, "
            "and a genuine appreciation for great creative work.  Data literacy is increasingly valued."
        ),
        "key_values":   ["creative problem solving", "creator empathy", "design excellence", "cross-functional collaboration", "data literacy"],
        "avoid":        ["purely metrics-driven framing without creativity", "ignoring design sensibility", "siloed thinking"],
        "tip":          "Show your appreciation for creative craft, balance business metrics with design quality, and show cross-team empathy.",
        "fit_color":    "#FF0000",
    },
    "Accenture": {
        "style_descriptor": (
            "Accenture interviewers value versatility, a learning mindset, and the ability to deliver "
            "transformation at enterprise scale.  "
            "Candidates should show strong client communication, the ability to bridge business and technology, "
            "and a structured approach to problem-solving.  "
            "Innovation and digital fluency are increasingly important signals."
        ),
        "key_values":   ["versatility", "learning mindset", "enterprise scale", "business-tech bridge", "digital fluency"],
        "avoid":        ["narrow specialist framing", "poor client communication", "resistance to change"],
        "tip":          "Show adaptability, bridge business and tech, and demonstrate how you deliver change at scale.",
        "fit_color":    "#A100FF",
    },
    "SpaceX": {
        "style_descriptor": (
            "SpaceX interviewers prize extreme ownership, first-principles engineering, and an obsession "
            "with making humanity multi-planetary.  "
            "Candidates must show they can work under intense pressure, move faster than anyone else, "
            "and never accept 'it can't be done' as an answer.  "
            "Frugality, mission alignment, and willingness to do whatever it takes are paramount."
        ),
        "key_values":   ["extreme ownership", "first-principles engineering", "mission obsession", "speed", "frugality"],
        "avoid":        ["comfort with the status quo", "passing the buck", "bloated solutions"],
        "tip":          "Show you question every assumption from first principles, own outcomes completely, and can sprint at extreme pace.",
        "fit_color":    "#005288",
    },
}

_COMPANY_NAMES = list(COMPANY_CULTURE_PROFILES.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompanyFitResult:
    company:    str
    fit_pct:    float          # 0–100
    fit_label:  str            # "Strong Fit" | "Good Fit" | "Moderate Fit" | "Low Fit"
    fit_color:  str            # CSS hex
    bar_color:  str            # profile brand color
    tip:        str            # adjustment suggestion
    key_values: List[str]
    method:     str = "Groq"  # "Groq" | "Heuristic"


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ BATCH FIT SCORER
#  Sends ONE LLM call scoring the candidate against ALL companies at once.
#  Returns dict: { company_name: score_0_to_100, ... }
# ══════════════════════════════════════════════════════════════════════════════

def _score_batch_with_groq(
    candidate_doc: str,
    groq_api_key:  str,
) -> Dict[str, float]:
    """
    Ask Groq to rate the candidate's communication style against every company
    profile in a single API call.

    Prompt design:
      • All company descriptors are listed compactly (key values + avoid).
      • The model returns ONLY a JSON object: {"CompanyName": score, ...}
        where score is 0–100 (higher = better fit).
      • Temperature 0.1 for deterministic scoring.
      • Strips markdown fences before JSON parse.

    Returns empty dict on any failure (caller falls back to heuristic).
    """
    if not groq_api_key or not GROQ_OK:
        return {}

    # Build compact company profiles string to stay within token budget
    profiles_lines = []
    for name, prof in COMPANY_CULTURE_PROFILES.items():
        kv   = ", ".join(prof.get("key_values", []))
        avoid = "; ".join(prof.get("avoid", []))
        profiles_lines.append(
            f'  "{name}": values=[{kv}], avoid=[{avoid}]'
        )
    profiles_str = "\n".join(profiles_lines)

    candidate_snippet = candidate_doc[:1200]   # trim to save tokens

    prompt = f"""You are an expert interview coach and organisational psychologist.

Your job: rate how well a candidate's communication style fits each company's interview culture.

CANDIDATE COMMUNICATION STYLE (extracted from their actual interview answers):
\"\"\"{candidate_snippet}\"\"\"

COMPANY CULTURE PROFILES:
{profiles_str}

INSTRUCTIONS:
- For each company, give a fit score from 0 to 100 based purely on communication style alignment.
- 80-100 = the candidate naturally communicates the way this company's interviewers love.
- 50-79  = reasonable fit with some style adjustments needed.
- 20-49  = noticeable style mismatch; significant changes required.
- 0-19   = very poor fit; the candidate's style contradicts what the company values.
- Differentiate meaningfully — do not cluster all scores in the same range.
- Respond with ONLY valid JSON. No explanation, no markdown, no extra keys.

FORMAT (return exactly this, filling in the numbers):
{{{", ".join(f'"{n}": 0' for n in COMPANY_CULTURE_PROFILES)}}}"""

    try:
        client = _Groq(api_key=groq_api_key)
        resp   = client.chat.completions.create(
            model       = _GROQ_SCORING_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 400,
            temperature = 0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$",       "", raw, flags=re.MULTILINE)
        raw = raw.strip()
        parsed = json.loads(raw)
        # Validate: keep only companies we know, clamp to [0, 100]
        result = {}
        for name in COMPANY_CULTURE_PROFILES:
            if name in parsed:
                try:
                    result[name] = float(max(0, min(100, parsed[name])))
                except (TypeError, ValueError):
                    pass
        return result
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD HEURISTIC FALLBACK
#  Used when Groq is unavailable.  Simple overlap scoring, no ML required.
# ══════════════════════════════════════════════════════════════════════════════

def _score_heuristic(candidate_doc: str) -> Dict[str, float]:
    """
    Simple keyword-overlap scorer.
    For each company, count how many of its key_values appear in the
    candidate document.  Normalise to 0-100 with reasonable spread.
    Returns dict: { company_name: score, ... }
    """
    doc_lower = candidate_doc.lower()
    scores: Dict[str, float] = {}
    for name, prof in COMPANY_CULTURE_PROFILES.items():
        kv   = prof.get("key_values", [])
        desc = prof.get("style_descriptor", "")
        avoid = prof.get("avoid", [])

        # Keyword hits
        hits  = sum(1 for k in kv   if k.lower() in doc_lower)
        misses= sum(1 for a in avoid if any(w in doc_lower for w in a.lower().split()[:2]))

        # Descriptor word overlap (rough)
        desc_words   = set(re.findall(r"\b\w{5,}\b", desc.lower()))
        cand_words   = set(re.findall(r"\b\w{5,}\b", doc_lower))
        overlap      = len(desc_words & cand_words) / max(len(desc_words), 1)

        raw  = (hits / max(len(kv), 1)) * 0.5 + overlap * 0.6 - (misses * 0.08)
        pct  = max(10.0, min(90.0, raw * 100))
        scores[name] = round(pct, 1)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  CANDIDATE STYLE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def _build_candidate_style_doc(answers: List[Dict]) -> str:
    """
    Concatenate all answer texts and inject style markers so the embedding
    captures communication style, not just topic keywords.
    """
    if not answers:
        return ""

    texts = []
    for a in answers:
        t = (a.get("answer") or a.get("transcript") or a.get("answer_text") or "").strip()
        if t:
            texts.append(t)

    full_text = " ".join(texts)

    # Style signal extraction
    word_count  = len(full_text.split())
    sentences   = re.split(r"[.!?]+", full_text)
    avg_sent_l  = word_count / max(len(sentences), 1)

    data_words  = len(re.findall(r"\b\d+[%x]?\b|\bmetric|result|data|percent|increase|decrease\b", full_text, re.I))
    star_words  = len(re.findall(r"\bsituation|task|action|result|led|drove|achieved\b", full_text, re.I))
    frame_words = len(re.findall(r"\bfirst|second|third|framework|structure|MECE|bucket|pillar\b", full_text, re.I))
    humble_words= len(re.findall(r"\blearned|mistake|could have|feedback|improved|challenge\b", full_text, re.I))
    direct_words= len(re.findall(r"\bI decided|I drove|I built|I led|I owned|I implemented\b", full_text, re.I))
    collab_words= len(re.findall(r"\bwe|team|together|collaborated|cross-functional|stakeholder\b", full_text, re.I))
    creative_w  = len(re.findall(r"\bdesign|creative|innovate|imagine|prototype|craft\b", full_text, re.I))

    # Build enriched style document
    style_parts = [full_text]

    if data_words > 2:
        style_parts.append("data-driven quantified metrics-focused results-oriented")
    if star_words > 3:
        style_parts.append("structured storytelling STAR format situation task action result")
    if frame_words > 2:
        style_parts.append("structured frameworks MECE first principles top-down synthesis")
    if humble_words > 2:
        style_parts.append("intellectual humility learning growth mindset feedback-receptive")
    if direct_words > 2:
        style_parts.append("ownership direct accountability taking charge driving outcomes")
    if collab_words > 4:
        style_parts.append("collaborative team-oriented cross-functional stakeholder empathy")
    if creative_w > 1:
        style_parts.append("creative design-thinking craft craftsmanship")
    if avg_sent_l < 15:
        style_parts.append("concise direct precise communication")
    else:
        style_parts.append("narrative storytelling elaborate explanations")

    return " ".join(style_parts)


# ══════════════════════════════════════════════════════════════════════════════
#  FIT SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _fit_label_color(pct: float) -> Tuple[str, str]:
    if pct >= 70:
        return "Strong Fit",  "#00ff88"
    if pct >= 55:
        return "Good Fit",    "#a5f3fc"
    if pct >= 40:
        return "Moderate Fit","#fbbf24"
    return "Low Fit",         "#ff3366"


def compute_all_fits(
    answers:      List[Dict],
    groq_api_key: str = "",
) -> List[CompanyFitResult]:
    """
    Score the candidate against all company profiles using Groq.
    Falls back to keyword heuristic if Groq unavailable.
    Returns list of CompanyFitResult sorted by fit_pct descending.
    Caches result for 6 hours in session state.
    """
    # Cache check
    cached    = st.session_state.get("culture_fit_results")
    cached_ts = st.session_state.get("culture_fit_ts", 0)
    if cached and (time.time() - cached_ts) < _CACHE_TTL:
        return cached

    candidate_doc = _build_candidate_style_doc(answers)
    if not candidate_doc.strip():
        return []

    # ── Primary: Groq batch scoring ──────────────────────────────────────────
    raw_scores = _score_batch_with_groq(candidate_doc, groq_api_key)
    method = "Groq"

    # ── Fallback: keyword heuristic ──────────────────────────────────────────
    if not raw_scores:
        raw_scores = _score_heuristic(candidate_doc)
        method = "Heuristic"

    results: List[CompanyFitResult] = []
    for name, prof in COMPANY_CULTURE_PROFILES.items():
        pct = round(float(raw_scores.get(name, 50.0)), 1)
        lbl, col = _fit_label_color(pct)
        results.append(CompanyFitResult(
            company    = name,
            fit_pct    = pct,
            fit_label  = lbl,
            fit_color  = col,
            bar_color  = prof.get("fit_color", "#00d4ff"),
            tip        = prof.get("tip", ""),
            key_values = prof.get("key_values", []),
            method     = method,
        ))

    results.sort(key=lambda x: x.fit_pct, reverse=True)

    st.session_state["culture_fit_results"] = results
    st.session_state["culture_fit_ts"]      = time.time()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ADJUSTMENT TIP GENERATOR (Groq-powered)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_adjustment_tip(
    candidate_doc: str,
    company:       str,
    profile:       Dict,
    fit_pct:       float,
    groq_api_key:  str,
) -> str:
    """
    Use Groq to generate a personalised 2-3 sentence adjustment tip for the
    target company based on the candidate's actual answer style.
    Falls back to the static profile tip.
    """
    static_tip = profile.get("tip", "")
    if not groq_api_key or not GROQ_OK:
        return static_tip

    cache_key = f"_culture_tip_{company}"
    if st.session_state.get(cache_key):
        return st.session_state[cache_key]

    key_vals  = ", ".join(profile.get("key_values", []))
    avoid     = "; ".join(profile.get("avoid", []))
    style_ctx = candidate_doc[:800]   # trim for token budget

    prompt = f"""You are an expert interview coach.

Candidate communication style (extracted from their interview answers):
\"\"\"{style_ctx}\"\"\"

Target company: {company}
Company culture fit score: {fit_pct:.0f}%
What {company} interviewers value: {key_vals}
What to avoid: {avoid}

Write exactly 2-3 sentences of highly specific, actionable advice explaining:
1. What the candidate's current style is getting right or wrong for {company}
2. One concrete change they should make to boost their fit score

Be direct and specific. No generic platitudes. Reference the actual candidate style above."""

    try:
        from groq import Groq as _G
        client = _G(api_key=groq_api_key)
        resp = client.chat.completions.create(
            model    = _GROQ_MODEL,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 150,
            temperature = 0.4,
        )
        tip = resp.choices[0].message.content.strip()
        st.session_state[cache_key] = tip
        return tip
    except Exception:
        return static_tip


# ══════════════════════════════════════════════════════════════════════════════
#  UI RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_culture_fit_section(
    answers:      List[Dict],
    target_co:    str  = "",
    groq_api_key: str  = "",
    accent_color: str  = "#00e5ff",
) -> None:
    """
    Full culture fit section for the Final Report page.

    Renders:
    1. Section header + methodology badge
    2. Target company card (if set) — large fit ring + personalised tip
    3. Top 3 strongest fits
    4. Bottom 3 weakest fits (with adjustment tips)
    5. Full ranked table (collapsible expander)
    """
    if not answers:
        st.info("Complete an interview session to see your culture fit analysis.")
        return

    with st.spinner("◈ Computing culture fit scores via Groq…"):
        fit_results = compute_all_fits(answers, groq_api_key)

    if not fit_results:
        st.warning("Culture fit analysis unavailable — no answer data found.")
        return

    method_badge = f"{fit_results[0].method if fit_results else 'Groq'} · LLM style analysis"

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="margin:2rem 0 1.2rem;">
  <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.4rem;">
    <div style="width:3px;height:28px;background:linear-gradient(180deg,{accent_color},{accent_color}55);
      border-radius:2px;flex-shrink:0;"></div>
    <div>
      <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;
        color:#e8f4ff;letter-spacing:.12em;text-transform:uppercase;">
        Company Culture Fit Predictor
      </div>
      <div style="font-size:.65rem;color:rgba(148,185,220,.55);
        font-family:'JetBrains Mono',monospace;margin-top:.2rem;">
        Communication style ↔ interview culture · Groq LLM · {method_badge}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Target company card ───────────────────────────────────────────────────
    target_clean = (target_co or "").strip()
    target_result: Optional[CompanyFitResult] = None
    if target_clean and target_clean not in ("No specific company", "Other", ""):
        target_result = next((r for r in fit_results if r.company == target_clean), None)

    if target_result:
        candidate_doc = _build_candidate_style_doc(answers)
        profile       = COMPANY_CULTURE_PROFILES.get(target_clean, {})
        adj_tip       = _generate_adjustment_tip(
            candidate_doc, target_clean, profile, target_result.fit_pct, groq_api_key
        )
        pct   = target_result.fit_pct
        ring_col = target_result.fit_color
        brand_col= target_result.bar_color
        deg   = int(pct / 100 * 360)
        kv_html = " ".join(
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.55rem;'
            f'padding:2px 8px;background:rgba(255,255,255,.05);'
            f'border:1px solid rgba(255,255,255,.1);border-radius:10px;'
            f'color:rgba(200,220,240,.7);">{v}</span>'
            for v in target_result.key_values[:5]
        )
        st.markdown(f"""
<div style="background:rgba(0,6,22,.95);border:1px solid {brand_col}33;
  border-top:2px solid {brand_col};border-radius:14px;
  padding:1.3rem 1.5rem;margin-bottom:1.2rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:50px;
    background:radial-gradient(ellipse 60% 100% at 50% 0%,{brand_col}12,transparent);
    pointer-events:none;"></div>
  <div style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;">
    <!-- Fit ring -->
    <div style="position:relative;width:88px;height:88px;flex-shrink:0;">
      <svg width="88" height="88" viewBox="0 0 88 88" style="transform:rotate(-90deg);">
        <circle cx="44" cy="44" r="36" fill="none"
          stroke="rgba(255,255,255,.07)" stroke-width="8"/>
        <circle cx="44" cy="44" r="36" fill="none"
          stroke="{ring_col}" stroke-width="8"
          stroke-dasharray="{int(pct/100*226.2)} 226.2"
          stroke-linecap="round"/>
      </svg>
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
        text-align:center;">
        <div style="font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:800;
          color:{ring_col};">{pct:.0f}%</div>
        <div style="font-size:.42rem;color:rgba(180,210,230,.5);
          font-family:'JetBrains Mono',monospace;margin-top:1px;">FIT</div>
      </div>
    </div>
    <!-- Company info -->
    <div style="flex:1;min-width:180px;">
      <div style="font-family:'Orbitron',monospace;font-size:.85rem;font-weight:700;
        color:#e8f4ff;margin-bottom:.25rem;">{target_clean}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
        color:{ring_col};font-weight:600;margin-bottom:.5rem;">
        {target_result.fit_label}
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:.65rem;">
        {kv_html}
      </div>
      <div style="font-family:'Inter',sans-serif;font-size:.78rem;
        color:rgba(200,220,240,.8);line-height:1.65;
        border-left:2px solid {brand_col};padding-left:.75rem;">
        💡 {adj_tip}
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Top 3 + Bottom 3 grid ─────────────────────────────────────────────────
    top3 = fit_results[:3]
    bot3 = fit_results[-3:][::-1]   # worst first in display

    def _mini_card(r: CompanyFitResult, show_tip: bool = False) -> str:
        bar_w  = int(r.fit_pct)
        kv_h   = " · ".join(r.key_values[:3])
        tip_h  = (
            f'<div style="font-size:.62rem;color:rgba(200,220,240,.65);'
            f'font-family:\'Inter\',sans-serif;line-height:1.6;margin-top:.5rem;'
            f'border-left:1.5px solid {r.bar_color};padding-left:.5rem;">'
            f'{r.tip}</div>'
            if show_tip else ""
        )
        return f"""
<div style="background:rgba(0,8,30,.7);border:1px solid rgba(255,255,255,.07);
  border-left:2px solid {r.bar_color};border-radius:10px;
  padding:.8rem 1rem;margin-bottom:.5rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
    <div style="font-family:'Inter',sans-serif;font-size:.82rem;
      font-weight:600;color:#e8f4ff;">{r.company}</div>
    <div style="font-family:'Orbitron',monospace;font-size:.88rem;
      font-weight:700;color:{r.fit_color};">{r.fit_pct:.0f}%</div>
  </div>
  <div style="background:rgba(255,255,255,.06);border-radius:3px;height:5px;
    overflow:hidden;margin-bottom:.4rem;">
    <div style="width:{bar_w}%;height:100%;background:{r.bar_color};
      border-radius:3px;box-shadow:0 0 8px {r.bar_color}66;"></div>
  </div>
  <div style="font-size:.6rem;color:rgba(180,210,230,.45);
    font-family:'JetBrains Mono',monospace;">{kv_h}</div>
  <div style="font-size:.58rem;color:{r.fit_color};
    font-family:'JetBrains Mono',monospace;margin-top:.2rem;font-weight:600;">
    {r.fit_label}
  </div>
  {tip_h}
</div>"""

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
            'color:#00ff88;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.6rem;">'
            '▲ STRONGEST FITS</div>',
            unsafe_allow_html=True,
        )
        for r in top3:
            st.markdown(_mini_card(r, show_tip=False), unsafe_allow_html=True)

    with col_bot:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
            'color:#ff3366;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.6rem;">'
            '▼ STYLE ADJUSTMENTS NEEDED</div>',
            unsafe_allow_html=True,
        )
        for r in bot3:
            st.markdown(_mini_card(r, show_tip=True), unsafe_allow_html=True)

    # ── Full ranked table (collapsible) ───────────────────────────────────────
    with st.expander("◈ Full company fit ranking  (" + str(len(fit_results)) + " companies)", expanded=False):
        rows_html = ""
        for i, r in enumerate(fit_results, 1):
            bg   = "rgba(0,255,136,.05)" if r.fit_pct >= 70 else ("rgba(255,255,255,.02)" if i % 2 == 0 else "transparent")
            rows_html += f"""
<tr style="background:{bg};">
  <td style="padding:6px 8px;font-family:'JetBrains Mono',monospace;
    font-size:.6rem;color:rgba(180,210,230,.4);text-align:center;">{i}</td>
  <td style="padding:6px 12px;font-family:'Inter',sans-serif;
    font-size:.78rem;color:#e8f4ff;font-weight:500;">{r.company}</td>
  <td style="padding:6px 12px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;">
        <div style="width:{int(r.fit_pct)}%;height:100%;background:{r.bar_color};border-radius:2px;"></div>
      </div>
      <span style="font-family:'Orbitron',monospace;font-size:.72rem;
        font-weight:700;color:{r.fit_color};min-width:36px;text-align:right;">
        {r.fit_pct:.0f}%
      </span>
    </div>
  </td>
  <td style="padding:6px 8px;font-family:'JetBrains Mono',monospace;
    font-size:.58rem;color:{r.fit_color};">{r.fit_label}</td>
</tr>"""

        st.markdown(f"""
<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;
  background:rgba(0,6,22,.85);border-radius:10px;overflow:hidden;">
  <thead>
    <tr style="background:rgba(0,212,255,.05);border-bottom:1px solid rgba(0,212,255,.1);">
      <th style="padding:8px;font-family:'JetBrains Mono',monospace;font-size:.58rem;
        color:rgba(0,212,255,.6);text-align:center;">#</th>
      <th style="padding:8px 12px;font-family:'JetBrains Mono',monospace;font-size:.58rem;
        color:rgba(0,212,255,.6);text-align:left;">Company</th>
      <th style="padding:8px 12px;font-family:'JetBrains Mono',monospace;font-size:.58rem;
        color:rgba(0,212,255,.6);text-align:left;">Fit Score</th>
      <th style="padding:8px;font-family:'JetBrains Mono',monospace;font-size:.58rem;
        color:rgba(0,212,255,.6);">Label</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>""", unsafe_allow_html=True)

    # ── Method footnote ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.5rem;'
        f'color:rgba(148,185,220,.3);margin-top:.5rem;text-align:center;">'
        f'Scores computed via {method_badge} · higher = better communication style alignment'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS  (add to app.py DEFAULTS dict)
# ══════════════════════════════════════════════════════════════════════════════

CULTURE_FIT_DEFAULTS: Dict = {
    "culture_fit_results": [],
    "culture_fit_ts":      0.0,
}