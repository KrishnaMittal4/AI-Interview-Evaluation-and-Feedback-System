"""
adaptive_sequencer.py — Aura AI | RL-Based Adaptive Question Sequencer (v3.0)
==============================================================================
v3.0 — COMPANY STYLE PACKS + ROLE SYLLABUS
════════════════════════════════════════════

4. ROLE SYLLABUS (new)
   ─────────────────────
   ROLE_SYLLABUS maps all 15 roles to a structured topic dict:
     { topic_name: { weight, q_type, subtopics, resources } }
   Weights express how heavily each topic is tested in real interviews
   (Huffcutt & Arthur 1994, Personnel Psychology — content validity).

   get_syllabus(role) → topic map
   get_syllabus_weights(role, company_pack) → normalised weight dict

5. COMPANY STYLE PACKS (new)
   ──────────────────────────
   COMPANY_STYLE defines 6 archetypes (faang, startup, consulting, fintech,
   healthtech, no_pack).  Each pack carries:
     • q_type_weights   — multipliers per question type (technical/behavioural/hr)
     • topic_overrides  — per-topic multipliers (substring match, case-insensitive)
     • prompt_style     — extra instruction appended to the Groq system prompt

   Weight application via get_syllabus_weights(role, pack):
     1. Start from ROLE_SYLLABUS base weights
     2. Apply q_type_weights multiplier for each topic's q_type
     3. Apply topic_overrides for any matching topic name substring
     4. Re-normalise so weights sum to 1.0

   pick_topic_for_question(role, pack) samples a topic for the next question
   using the adjusted distribution — used by _generate_question() in main.py.

   Research: Huffcutt et al. (2001, Personnel Psychology) — interview content
   validity improves when question distribution matches actual job task analysis
   of the target company archetype. Campion et al. (1997, Personnel Psychology)
   — situational and past-behaviour question weights should reflect the job's
   actual performance criteria, not a generic distribution.

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
  0  conceptual   easy        ← "Technical" = conceptual/architectural, NEVER write-code
  1  conceptual   medium      ← default cold start
  2  conceptual   hard
  3  behavioural  easy
  4  behavioural  medium
  5  behavioural  hard
  6  hr           medium
  7  follow_up    —           ← probe same topic (shallow answer detected)

NOTE — NO CODE EDITOR POLICY:
  q_type "technical" / "conceptual" means concept, architecture, and experience
  questions only. Write-code / DSA / LeetCode-style questions are NEVER generated.
  Examples of allowed "technical" questions:
    "Explain how a load balancer distributes traffic."
    "What are the trade-offs between SQL and NoSQL?"
    "Walk me through debugging a memory leak in production."
  Examples of FORBIDDEN questions (no code editor in this system):
    "Write a function to reverse a linked list."
    "Implement a binary search algorithm."
    "Given an array, find two numbers that sum to target." 
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
#  ROLE SYLLABUS  —  topic map for each role
#  Structure: role → { topic: { weight, subtopics, resources } }
#  Weights sum to 1.0 per role and express how heavily each topic is tested
#  in real interviews (research: Huffcutt & Arthur 1994, J. Applied Psych.).
# ══════════════════════════════════════════════════════════════════════════════

ROLE_SYLLABUS: Dict[str, Dict] = {
    "Software Engineer": {
        "Data Structures & Algorithms": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": [
                "Trade-offs between data structures (array vs linked list vs hash map vs tree)",
                "When to use a heap, trie, or graph — conceptual reasoning",
                "Time & space complexity analysis and real-world implications",
                "How dynamic programming concepts apply to system optimisation decisions",
            ],
            "resources": ["CLRS Ch. 3-6 (conceptual chapters)", "system-design-primer DS section"],
        },
        "System Design": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Scalability", "Caching & CDN", "Database sharding", "Message queues", "Load balancing"],
            "resources": ["Designing Data-Intensive Applications", "system-design-primer"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["STAR storytelling", "Conflict resolution", "Cross-team collaboration", "Ownership & accountability"],
            "resources": ["Your analyzer.py STAR coaching"],
        },
        "CS Fundamentals": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["OS & processes", "Networking basics", "Time & space complexity", "Concurrency"],
            "resources": ["Operating Systems: Three Easy Pieces"],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "5-year vision", "Team & culture preferences"],
            "resources": [],
        },
    },
    "Frontend Developer": {
        "JavaScript & TypeScript": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Event loop & async", "Closures & scope", "TypeScript generics", "ES2022+ features"],
            "resources": ["You Don't Know JS", "TypeScript Handbook"],
        },
        "React & Component Architecture": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["State management", "Hooks deep-dive", "Rendering optimisation", "Component design patterns"],
            "resources": ["React docs", "patterns.dev"],
        },
        "CSS & Web Performance": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["CSS Grid & Flexbox", "Critical rendering path", "Core Web Vitals", "Accessibility (WCAG)"],
            "resources": ["web.dev", "MDN"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Design feedback handling", "Cross-functional collaboration", "Deadline pressure", "Code review culture"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Preferred team structure", "Remote vs on-site preferences"],
            "resources": [],
        },
    },
    "Backend Developer": {
        "API Design & Protocols": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["REST vs GraphQL vs gRPC", "Authentication & JWT", "Rate limiting", "API versioning"],
            "resources": ["REST API Design Rulebook"],
        },
        "Databases": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["SQL vs NoSQL trade-offs", "Indexing strategy", "Transactions & ACID", "Query optimisation"],
            "resources": ["Use The Index, Luke", "Designing Data-Intensive Applications"],
        },
        "System Design": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Microservices vs monolith", "Event-driven architecture", "Caching layers", "Message brokers"],
            "resources": ["system-design-primer"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Incident response", "On-call culture", "Technical debt decisions", "Mentoring juniors"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Tech stack preferences", "Team collaboration style"],
            "resources": [],
        },
    },
    "Data Scientist": {
        "Statistics & Probability": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Hypothesis testing", "Bayesian inference", "A/B testing design", "Distributions & CLT"],
            "resources": ["Statistics for Data Scientists", "Think Stats"],
        },
        "Machine Learning": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Bias-variance trade-off", "Regularisation", "Model evaluation metrics", "Ensemble methods"],
            "resources": ["Hands-On ML (Géron)", "scikit-learn docs"],
        },
        "Data Engineering & SQL": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["SQL window functions", "Data pipeline design", "Feature engineering", "Data quality & cleaning"],
            "resources": ["Mode SQL Tutorial"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Communicating results to non-technical stakeholders", "Ambiguous problem definition", "Cross-team data projects"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Research vs product trade-off preference", "Career motivation", "Impact measurement"],
            "resources": [],
        },
    },
    "Product Manager": {
        "Product Strategy & Vision": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Market sizing", "Competitive analysis", "Roadmap prioritisation (RICE/ICE)", "OKR setting"],
            "resources": ["Inspired (Cagan)", "Shape Up"],
        },
        "Execution & Delivery": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Sprint planning", "Stakeholder alignment", "Trade-off decisions", "Launch planning"],
            "resources": ["The Lean Startup"],
        },
        "Metrics & Analytics": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["North Star metric", "Funnel analysis", "A/B test interpretation", "Retention vs acquisition"],
            "resources": ["Reforge Growth Series"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Influencing without authority", "Handling engineer pushback", "Saying no to stakeholders", "Failed product decisions"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["PM philosophy", "Career motivation", "Preferred product area"],
            "resources": [],
        },
    },
    "DevOps Engineer": {
        "CI/CD & Automation": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Pipeline design", "Blue-green & canary deployments", "GitOps", "Build optimisation"],
            "resources": ["The Phoenix Project", "Continuous Delivery (Humble & Farley)"],
        },
        "Infrastructure & Cloud": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["IaC (Terraform)", "Kubernetes concepts", "Cloud cost optimisation", "Multi-region HA"],
            "resources": ["AWS Well-Architected Framework"],
        },
        "Observability & Incident Response": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Monitoring & alerting", "SLO/SLA/SLI", "Post-mortem culture", "RCA process"],
            "resources": ["Site Reliability Engineering (Google)"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["On-call war stories", "Convincing dev teams to adopt DevOps practices", "Prioritising reliability vs speed"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.05,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Cloud platform preference"],
            "resources": [],
        },
    },
    "Machine Learning Engineer": {
        "ML Systems Design": {
            "weight": 0.30,
            "q_type": "technical",
            "subtopics": ["Feature store architecture", "Online vs batch inference", "Model serving", "Training pipeline design"],
            "resources": ["Designing ML Systems (Chip Huyen)"],
        },
        "Deep Learning Fundamentals": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Transformer architecture", "Attention mechanisms", "Regularisation & optimisers", "Distributed training"],
            "resources": ["fast.ai", "Deep Learning (Goodfellow)"],
        },
        "MLOps & Production": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Model monitoring & drift detection", "A/B testing models", "CI/CD for ML", "Data versioning"],
            "resources": ["MLflow docs", "Evidently AI docs"],
        },
        "Behavioural": {
            "weight": 0.20,
            "q_type": "behavioural",
            "subtopics": ["Research-to-production transitions", "Communicating model uncertainty", "Ambiguous requirements", "Failed experiments"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Research vs engineering balance preference", "Career motivation"],
            "resources": [],
        },
    },
    "Full Stack Developer": {
        "Frontend Fundamentals": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["React state management", "Browser rendering pipeline", "Accessibility basics", "Performance optimisation"],
            "resources": ["patterns.dev"],
        },
        "Backend & APIs": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["REST API design", "Authentication flows", "Database ORM vs raw SQL", "Background jobs"],
            "resources": ["REST API Design Rulebook"],
        },
        "System Design": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Monolith to microservices journey", "Caching strategies", "Deployment architecture", "Scalability trade-offs"],
            "resources": ["system-design-primer"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Wearing many hats", "Prioritising across stack", "Working with designers", "Technical debt balance"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.15,
            "q_type": "hr",
            "subtopics": ["Stack preferences", "Career motivation", "Team size preferences"],
            "resources": [],
        },
    },
    "Data Engineer": {
        "Data Pipeline & ETL": {
            "weight": 0.30,
            "q_type": "technical",
            "subtopics": ["Batch vs streaming", "Airflow DAG design", "Data quality checks", "Incremental loading"],
            "resources": ["Fundamentals of Data Engineering (Reis & Housley)"],
        },
        "Data Warehousing": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Star vs snowflake schema", "Partitioning & clustering", "dbt models", "Slowly changing dimensions"],
            "resources": ["The Data Warehouse Toolkit (Kimball)"],
        },
        "Distributed Systems": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["Spark architecture", "Kafka concepts", "CAP theorem applied to pipelines", "Data lakehouse architecture"],
            "resources": ["Designing Data-Intensive Applications"],
        },
        "Behavioural": {
            "weight": 0.20,
            "q_type": "behavioural",
            "subtopics": ["Data quality incident handling", "Collaborating with analysts", "Balancing pipeline speed vs reliability"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Cloud data platform preferences"],
            "resources": [],
        },
    },
    "QA Engineer": {
        "Test Strategy & Planning": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Test pyramid", "Risk-based testing", "Test coverage metrics", "Shift-left testing"],
            "resources": ["Agile Testing (Crispin & Gregory)"],
        },
        "Automation Frameworks": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Selenium vs Playwright", "API test automation", "CI integration", "Test data management"],
            "resources": ["Playwright docs"],
        },
        "Performance & Security Testing": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["Load testing concepts", "k6 / JMeter basics", "OWASP top 10 awareness", "Penetration test scoping"],
            "resources": ["k6 docs"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Advocating for quality in fast-moving teams", "Handling 'ship it anyway' pressure", "Bug triage under deadline"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["QA philosophy", "Career motivation", "Dev-QA relationship style"],
            "resources": [],
        },
    },
    "System Designer": {
        "Distributed System Fundamentals": {
            "weight": 0.30,
            "q_type": "technical",
            "subtopics": ["CAP theorem", "Consistency models", "Consensus algorithms (Raft/Paxos)", "Distributed transactions"],
            "resources": ["Designing Data-Intensive Applications", "system-design-primer"],
        },
        "Large-Scale Architecture": {
            "weight": 0.30,
            "q_type": "technical",
            "subtopics": ["URL shortener design", "News feed system", "Rate limiter design", "Notification service"],
            "resources": ["System Design Interview (Alex Xu)"],
        },
        "Scalability & Reliability": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["Horizontal vs vertical scaling", "Database replication", "Circuit breaker pattern", "Chaos engineering concepts"],
            "resources": ["Site Reliability Engineering (Google)"],
        },
        "Behavioural": {
            "weight": 0.15,
            "q_type": "behavioural",
            "subtopics": ["Architecture decisions under ambiguity", "Communicating trade-offs to stakeholders", "Legacy system migration"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Preferred system scale", "Build vs buy philosophy"],
            "resources": [],
        },
    },
    "Mobile Developer": {
        "Platform Fundamentals": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Activity/Fragment lifecycle (Android) or UIViewController lifecycle (iOS)", "Memory management", "Background task handling", "Push notification architecture"],
            "resources": ["Android Developer docs", "Apple Developer docs"],
        },
        "UI & Performance": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Jetpack Compose / SwiftUI declarative patterns", "Rendering pipeline", "App size optimisation", "Smooth animation (60fps)"],
            "resources": ["Android performance docs", "Human Interface Guidelines"],
        },
        "Cross-Platform": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["React Native vs Flutter trade-offs", "Native module bridges", "Code sharing strategies", "Platform-specific UX expectations"],
            "resources": ["Flutter docs", "React Native docs"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["App store rejection handling", "Coordinating with backend teams", "Delivering under tight release cycles"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.15,
            "q_type": "hr",
            "subtopics": ["Platform preference", "Career motivation", "Consumer vs enterprise app preference"],
            "resources": [],
        },
    },
    "Cloud Architect": {
        "Cloud Platform Expertise": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["AWS/Azure/GCP service selection rationale", "Managed vs self-hosted trade-offs", "Multi-cloud strategies", "Cloud cost governance"],
            "resources": ["AWS Well-Architected Framework", "Azure Architecture Center"],
        },
        "Security & Compliance": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["IAM design", "Network security (VPC, security groups)", "Data encryption at rest & in transit", "SOC2 / GDPR cloud implications"],
            "resources": ["NIST Cybersecurity Framework"],
        },
        "High Availability & DR": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["RTO/RPO definition", "Multi-region failover", "Backup & restore strategies", "Chaos engineering"],
            "resources": ["Site Reliability Engineering (Google)"],
        },
        "Behavioural": {
            "weight": 0.20,
            "q_type": "behavioural",
            "subtopics": ["Migrating on-prem to cloud", "Convincing cost-conscious stakeholders", "Post-outage leadership"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Cloud certification philosophy", "Platform agnosticism vs specialisation"],
            "resources": [],
        },
    },
    "Cybersecurity Analyst": {
        "Threat Detection & Response": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["SIEM alert triage", "Incident response playbooks", "Log analysis", "Malware analysis concepts"],
            "resources": ["SANS Reading Room", "MITRE ATT&CK Framework"],
        },
        "Vulnerability Management": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["CVE scoring & prioritisation", "Penetration testing methodology", "Patch management", "Attack surface reduction"],
            "resources": ["OWASP Testing Guide"],
        },
        "Security Architecture": {
            "weight": 0.20,
            "q_type": "technical",
            "subtopics": ["Zero trust model", "Network segmentation", "Identity & access governance", "Cloud security posture management"],
            "resources": ["NIST SP 800-207"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Communicating risk to non-technical leadership", "Security vs developer velocity tension", "Incident communication under pressure"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Career motivation", "Red vs blue team preference", "Continuous learning approach"],
            "resources": [],
        },
    },
    "Scrum Master": {
        "Agile & Scrum Framework": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Scrum ceremonies deep-dive", "Definition of Done vs Acceptance Criteria", "Velocity & capacity planning", "Scrum vs Kanban trade-offs"],
            "resources": ["Scrum Guide 2020", "Kanban: Successful Evolutionary Change"],
        },
        "Team Coaching & Facilitation": {
            "weight": 0.25,
            "q_type": "technical",
            "subtopics": ["Dysfunctional team patterns", "Retrospective techniques", "Conflict facilitation", "Self-organising team enablement"],
            "resources": ["Coaching Agile Teams (Adkins)"],
        },
        "Metrics & Continuous Improvement": {
            "weight": 0.15,
            "q_type": "technical",
            "subtopics": ["Cycle time & lead time", "Sprint burndown interpretation", "Flow efficiency", "OKR alignment with sprints"],
            "resources": ["Actionable Agile Metrics (Vacanti)"],
        },
        "Behavioural": {
            "weight": 0.25,
            "q_type": "behavioural",
            "subtopics": ["Removing blockers", "PO-developer tension mediation", "Introducing agile to resistant teams", "Managing stakeholder expectations"],
            "resources": [],
        },
        "HR & Culture Fit": {
            "weight": 0.10,
            "q_type": "hr",
            "subtopics": ["Servant leadership philosophy", "Career motivation", "Preferred team size & domain"],
            "resources": [],
        },
    },
}

# ── Fallback: any role not in ROLE_SYLLABUS gets a generic syllabus ───────────
_GENERIC_SYLLABUS: Dict[str, Dict] = {
    "Core Technical Skills": {
        "weight": 0.30, "q_type": "technical",
        "subtopics": ["Domain-specific tools & frameworks", "Problem-solving methodology", "Architecture & design decisions"],
        "resources": [],
    },
    "Behavioural": {
        "weight": 0.35, "q_type": "behavioural",
        "subtopics": ["STAR storytelling", "Conflict resolution", "Leadership & ownership"],
        "resources": [],
    },
    "HR & Culture Fit": {
        "weight": 0.15, "q_type": "hr",
        "subtopics": ["Career motivation", "Team preferences", "5-year vision"],
        "resources": [],
    },
    "Communication & Collaboration": {
        "weight": 0.20, "q_type": "behavioural",
        "subtopics": ["Cross-functional work", "Presenting ideas", "Stakeholder management"],
        "resources": [],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  COMPANY STYLE PACKS
#  ─────────────────────────────────────────────────────────────────────────────
#  Each pack is a dict of weight MULTIPLIERS applied on top of ROLE_SYLLABUS
#  topic weights.  The key must match a q_type or a topic name substring.
#
#  Structure:
#    "pack_name": {
#        "display_name":    human-readable name shown in the UI
#        "description":     one-line description for the UI
#        "prompt_style":    appended to the Groq system prompt to shift question style
#        "q_type_weights":  multipliers by q_type ("technical", "behavioural", "hr")
#        "topic_overrides": { topic_name_substring: weight_multiplier }
#                           Overrides applied AFTER q_type_weights; substring match
#                           (case-insensitive) so partial names work across roles.
#    }
#
#  Weight application (get_syllabus_weights()):
#    1. Start from ROLE_SYLLABUS base weights
#    2. Multiply each topic weight by its q_type_weights multiplier
#    3. Apply topic_overrides for any matching substring
#    4. Re-normalise so weights still sum to 1.0
#
#  Research basis:
#    Huffcutt et al. (2001, Personnel Psychology) — interview content validity
#    improves significantly when question distribution matches the actual job
#    task analysis of the target company/role archetype. Company-specific
#    archetypes (FAANG systems-heavy, startup ambiguity-heavy, consulting
#    communication-heavy) represent genuinely different task distributions.
#
#    Campion et al. (1997, Personnel Psychology) — situational questions
#    (startup ambiguity, rapid change) and past-behaviour questions (FAANG
#    leadership principles) should be weighted according to the job's actual
#    performance criteria — not a generic distribution.
# ══════════════════════════════════════════════════════════════════════════════

COMPANY_STYLE: Dict[str, Dict] = {

    "faang": {
        "display_name": "FAANG / Big Tech",
        "description":  "Google, Meta, Amazon, Apple, Netflix — systems-heavy, leadership principles, bar-raising standards",
        "prompt_style": (
            "This question is for a FAANG-style interview. Expect rigorous conceptual depth. "
            "System design questions should probe scalability to billions of users, "
            "fault tolerance, and CAP theorem trade-offs — ask the candidate to reason through "
            "architecture decisions verbally, not write code. "
            "Data structures & algorithms questions must test conceptual understanding: "
            "'Explain how a hash map handles collisions', 'When would you use a heap over a BST?' "
            "— NEVER ask the candidate to write or implement code. "
            "Behavioural questions should follow the Amazon Leadership Principles or Google's "
            "structured behavioural rubric — 'Tell me about a time...' with a very high bar "
            "for measurable impact, ownership, and specifics: exact numbers, timelines, personal contribution."
        ),
        "q_type_weights": {
            "technical":   1.6,   # system design & algorithms weighted heavily
            "behavioural": 1.2,   # leadership principles matter a lot
            "hr":          0.4,   # HR culture-fit de-emphasised vs structured Qs
        },
        "topic_overrides": {
            "system design":               2.0,   # flagship FAANG topic
            "data structures & algorithms": 1.8,  # LC-style conceptual depth
            "cs fundamentals":             1.4,
            "ml systems design":           1.8,   # for MLE roles
            "distributed system":          2.0,   # for system designer roles
            "large-scale architecture":    2.0,
            "hr & culture fit":            0.3,   # FAANG rarely asks generic HR Qs
        },
    },

    "startup": {
        "display_name": "Startup / Scale-up",
        "description":  "Early-stage to Series C startups — ambiguity tolerance, speed-quality trade-offs, generalist ability",
        "prompt_style": (
            "This question is for a startup-style interview. Prioritise questions that reveal "
            "how the candidate handles ambiguity, moves fast with limited resources, and "
            "makes pragmatic trade-offs. Ask about times they wore multiple hats, shipped "
            "imperfect things under pressure, or changed direction rapidly. Avoid questions "
            "that assume large teams, established processes, or unlimited engineering time."
        ),
        "q_type_weights": {
            "technical":   0.9,   # breadth over depth
            "behavioural": 1.5,   # startup culture fit is 50% of the decision
            "hr":          1.2,   # mission alignment & motivation matter a lot
        },
        "topic_overrides": {
            "behavioural":                 1.6,
            "hr & culture fit":            1.5,
            "execution & delivery":        1.8,   # PM roles
            "ci/cd & automation":          1.3,   # DevOps roles
            "communication & collaboration": 1.4,
            "data structures & algorithms": 0.5,  # less LC-grind culture
            "distributed system":          0.6,   # over-engineering not valued
            "large-scale architecture":    0.5,
        },
    },

    "consulting": {
        "display_name": "Consulting / Professional Services",
        "description":  "McKinsey, Deloitte, Accenture — structured thinking, client communication, frameworks",
        "prompt_style": (
            "This question is for a consulting-style interview. Prioritise questions that test "
            "structured thinking, ability to communicate complex ideas simply, stakeholder "
            "management, and the ability to work across industries. Behavioural questions should "
            "probe client-facing skills, difficult feedback delivery, and navigating ambiguous briefs. "
            "Avoid deep technical implementation questions — focus on architecture decisions, "
            "trade-off reasoning, and how the candidate translates technical concepts to executives."
        ),
        "q_type_weights": {
            "technical":   0.7,
            "behavioural": 1.6,
            "hr":          1.3,
        },
        "topic_overrides": {
            "communication & collaboration": 2.0,
            "stakeholder":                   1.8,
            "product strategy":              1.4,
            "behavioural":                   1.6,
            "hr & culture fit":              1.4,
            "team coaching":                 1.5,   # Scrum Master
            "data structures & algorithms":  0.3,
            "system design":                 0.5,
        },
    },

    "fintech": {
        "display_name": "Fintech / Financial Services",
        "description":  "Banks, payment platforms, trading firms — reliability, compliance, security, accuracy",
        "prompt_style": (
            "This question is for a fintech or financial services interview. Emphasise reliability, "
            "data accuracy, and compliance awareness. System design questions should probe fault "
            "tolerance, idempotency, and auditability. Behavioural questions should explore how the "
            "candidate balances speed-to-market with risk, handles regulatory constraints, and "
            "communicates technical risk to compliance or legal teams."
        ),
        "q_type_weights": {
            "technical":   1.3,
            "behavioural": 1.1,
            "hr":          0.8,
        },
        "topic_overrides": {
            "security":                    1.8,
            "databases":                   1.5,   # transactions & ACID critical
            "api design":                  1.4,   # payment APIs, idempotency
            "observability":               1.5,   # fintech needs deep monitoring
            "compliance":                  1.8,
            "vulnerability":               1.5,
            "threat detection":            1.5,
            "distributed system":          1.3,
            "hr & culture fit":            0.7,
        },
    },

    "healthtech": {
        "display_name": "Healthtech / MedTech",
        "description":  "Healthcare software, medical devices, EHR — privacy, compliance (HIPAA), patient safety",
        "prompt_style": (
            "This question is for a healthtech or medical technology interview. "
            "Emphasise data privacy (HIPAA), patient safety, and the stakes of failure in "
            "a healthcare context. Technical questions should probe how the candidate handles "
            "sensitive data, audit logging, and system reliability when lives may depend on it. "
            "Behavioural questions should explore how they navigate regulatory constraints, "
            "communicate risk, and balance feature velocity with patient safety."
        ),
        "q_type_weights": {
            "technical":   1.2,
            "behavioural": 1.2,
            "hr":          0.9,
        },
        "topic_overrides": {
            "security":                    1.8,
            "compliance":                  1.8,
            "high availability":           1.6,
            "databases":                   1.4,   # PHI storage requirements
            "behavioural":                 1.3,
            "communication & collaboration": 1.4,
            "data structures & algorithms": 0.6,
        },
    },

    "no_pack": {
        "display_name": "Standard (no company pack)",
        "description":  "Balanced preparation — no company-specific style applied",
        "prompt_style": "",
        "q_type_weights": {"technical": 1.0, "behavioural": 1.0, "hr": 1.0},
        "topic_overrides": {},
    },
}

# Convenience alias — default when no pack is specified
DEFAULT_COMPANY_PACK = "no_pack"


# ══════════════════════════════════════════════════════════════════════════════
#  SYLLABUS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_syllabus(role: str) -> Dict[str, Dict]:
    """Return the topic map for a role, falling back to the generic syllabus."""
    return ROLE_SYLLABUS.get(role, _GENERIC_SYLLABUS)


def get_syllabus_weights(role: str, company_pack: str = DEFAULT_COMPANY_PACK) -> Dict[str, float]:
    """
    Return normalised topic weights for (role, company_pack).

    Steps:
      1. Fetch base weights from ROLE_SYLLABUS (or generic fallback).
      2. Multiply each topic's weight by the pack's q_type_weights multiplier
         for that topic's q_type.
      3. Apply topic_overrides: for each (substring, multiplier) in the pack,
         multiply any topic whose name contains the substring (case-insensitive).
      4. Re-normalise so weights sum to 1.0.

    Returns:
        { topic_name: normalised_weight, ... }
    """
    syllabus = get_syllabus(role)
    pack     = COMPANY_STYLE.get(company_pack, COMPANY_STYLE[DEFAULT_COMPANY_PACK])

    qt_mults     = pack.get("q_type_weights", {})
    topic_ovr    = pack.get("topic_overrides", {})

    adjusted: Dict[str, float] = {}
    for topic, meta in syllabus.items():
        base   = meta.get("weight", 0.0)
        q_type = meta.get("q_type", "technical")

        # Step 2 — q_type multiplier
        w = base * qt_mults.get(q_type, 1.0)

        # Step 3 — topic_overrides (substring match, case-insensitive)
        topic_lower = topic.lower()
        for substr, mult in topic_ovr.items():
            if substr.lower() in topic_lower:
                w *= mult
                break  # first match wins — overrides are mutually exclusive per topic

        adjusted[topic] = max(w, 0.0)   # clamp to non-negative

    # Step 4 — normalise
    total = sum(adjusted.values())
    if total == 0:
        # Degenerate case: all weights zeroed by overrides — fall back to uniform
        n = len(adjusted)
        return {t: 1.0 / n for t in adjusted} if n else {}

    return {t: w / total for t, w in adjusted.items()}


def pick_topic_for_question(role: str, company_pack: str = DEFAULT_COMPANY_PACK) -> Dict:
    """
    Sample a topic for the next question, weighted by the (role, company_pack)
    distribution.  Returns the topic's metadata dict with 'topic_name' added.

    Used by _generate_question() in main.py to steer Groq toward underrepresented
    topics according to the company pack's adjusted distribution.
    """
    syllabus = get_syllabus(role)
    weights  = get_syllabus_weights(role, company_pack)
    if not weights:
        return {"topic_name": "General", "q_type": "technical", "subtopics": [], "resources": []}

    topics  = list(weights.keys())
    w_vals  = [weights[t] for t in topics]

    chosen_topic = random.choices(topics, weights=w_vals, k=1)[0]
    meta = dict(syllabus.get(chosen_topic, {}))
    meta["topic_name"] = chosen_topic
    return meta


def get_company_pack_info(pack_key: str = DEFAULT_COMPANY_PACK) -> Dict:
    """Return display metadata for a company pack (for the UI)."""
    pack = COMPANY_STYLE.get(pack_key, COMPANY_STYLE[DEFAULT_COMPANY_PACK])
    return {
        "key":          pack_key,
        "display_name": pack.get("display_name", pack_key),
        "description":  pack.get("description", ""),
        "prompt_style": pack.get("prompt_style", ""),
    }


def list_company_packs() -> List[Dict]:
    """Return all available company packs as a list (for /syllabus/packs endpoint)."""
    return [
        {
            "key":          k,
            "display_name": v.get("display_name", k),
            "description":  v.get("description", ""),
        }
        for k, v in COMPANY_STYLE.items()
    ]

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION SPACE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Action:
    q_type:     str          # "technical"(=conceptual) | "behavioural" | "hr" | "follow_up"
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
W_SCORE_DELTA   = 2.0   # score improvement is the primary signal
W_CALM_BONUS    = 1.5   # nervousness reduction
W_STAR_BONUS    = 0.5   # STAR structure completion bonus
W_BRIEF_PENALTY = 0.3   # penalty for too-short answers
W_REPEAT_PENALTY= 0.4   # penalty for choosing same action twice in a row
W_STRESS_PENALTY= 0.4   # penalty for difficulty-induced nervousness SPIKE
                         # fires when nervousness rises faster than score improves,
                         # indicating the chosen difficulty was too aggressive.
                         # Research: Csikszentmihalyi (1990) flow theory — optimal
                         # challenge keeps arousal in the flow channel, not above it.
                         # Yang et al. (2021, IEEE TKDE) — in adaptive testing, inter-
                         # item stress spikes predict early dropout and score suppression
                         # independent of the candidate's actual ability level.

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
                   q_type:            str   = "",
                   difficulty_idx:    int   = 1) -> float:
    """
    Shaped reward function for the RL sequencer.

    Components (Liu et al. 2021 ACM ITS calibration):

    1. score_delta     — primary signal: reward improvement, penalise regression
    2. calm_bonus      — reduce nervousness = good (candidate more comfortable)
    3. star_bonus      — structured answers rewarded (Behavioural/HR only)
       Technical (conceptual) answers are explanations, not STAR stories —
       star_bonus is always 0.0 for technical actions, preventing Q-table bias.
    4. brevity_penalty — short answers signal question-difficulty mismatch
    5. repeat_penalty  — discourage consecutive identical actions
    6. stress_penalty  — NEW: penalise difficulty-induced nervousness spike.
       Fires when nervousness RISES between questions AND the chosen difficulty
       was aggressive (medium or hard). The logic:
         - nervousness_delta = nervousness − prev_nervousness (positive = rising)
         - Only applies when delta > 0 (nervousness worsened)
         - Scales with difficulty_idx (0=easy, 1=med, 2=hard):
             easy   → no penalty (candidate chose easy, spike not caused by difficulty)
             medium → 0.5× penalty (moderate contribution)
             hard   → 1.0× penalty (most likely cause of spike)
         - Does NOT double-count with calm_bonus: calm_bonus already rewards
           reduction; stress_penalty specifically penalises SPIKES caused by
           over-aggressive difficulty selection. They are complementary.

       Research:
         Csikszentmihalyi (1990) — Flow Theory: optimal challenge keeps arousal
         in the "flow channel" between boredom and anxiety. Questions that spike
         nervousness above the candidate's current capacity move them out of flow.
         Yang et al. (2021, IEEE TKDE) — in adaptive testing, inter-item anxiety
         spikes predict score suppression on subsequent items independent of ability.
         Kappen et al. (2024, Scientific Reports) — accumulated stress over a speech
         session compounds; early spikes raise the baseline for all later questions.

    Parameters
    ----------
    score, prev_score        : answer scores on 1-5 scale
    nervousness, prev_nervousness : fused facial+voice nervousness 0-1
    star_count               : STAR components present 0-4
    word_count               : answer word count
    prev_action_idx          : action index of the previous question
    curr_action_idx          : action index being evaluated for repeat penalty
    q_type                   : question type string (for star_bonus gating)
    difficulty_idx           : 0=easy, 1=medium, 2=hard — the difficulty level
                               of the action whose reward is being computed.
                               Used to scale stress_penalty by how aggressive
                               the chosen difficulty was.
    """
    score_delta = (score - prev_score) * W_SCORE_DELTA
    calm_bonus  = (prev_nervousness - nervousness) * W_CALM_BONUS

    # star_bonus only for Behavioural/HR — Technical(conceptual) never penalised
    _star_relevant = q_type.lower() in ("behavioural", "behavioral", "hr", "")
    if _star_relevant:
        if star_count >= 3:
            star_bonus = W_STAR_BONUS
        elif star_count >= 1:
            star_bonus = W_STAR_BONUS * 0.5
        else:
            star_bonus = 0.0
    else:
        star_bonus = 0.0   # Technical/Conceptual — STAR not applicable

    brief_penalty  = -W_BRIEF_PENALTY  if word_count < 50 else 0.0
    repeat_penalty = -W_REPEAT_PENALTY if curr_action_idx == prev_action_idx else 0.0

    # ── Stress penalty (new) ──────────────────────────────────────────────────
    # Only fires when nervousness rose (delta > 0) on this answer.
    # Scales by difficulty_idx so hard questions bear the full penalty when
    # they cause a spike, while easy questions bear none (spike is not their fault).
    nervousness_delta = nervousness - prev_nervousness   # positive = worse
    if nervousness_delta > 0.0:
        difficulty_scale = difficulty_idx / 2.0         # 0.0 / 0.5 / 1.0
        stress_penalty   = -W_STRESS_PENALTY * nervousness_delta * difficulty_scale
    else:
        stress_penalty = 0.0   # nervousness dropped or held — no penalty

    total = (score_delta + calm_bonus + star_bonus
             + brief_penalty + repeat_penalty + stress_penalty)
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
    q_number:      int
    state:         Tuple[int, int, int, int]
    action_idx:    int
    action:        str              # human-readable label
    score:         float
    nervousness:   float
    star_count:    int
    reward:        float
    q_type:        str
    difficulty:    str
    follow_up:     bool  = False    # True when follow-up probe was triggered
    stress_penalty: float = 0.0    # stress_penalty component of reward (≤ 0)
                                   # non-zero only when nervousness spiked AND
                                   # difficulty was medium or hard. Stored for
                                   # audit reporting and ablation analysis.


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

            # Derive difficulty_idx from the last action so stress_penalty
            # correctly scales by how aggressive the difficulty was.
            # 0=easy, 1=medium, 2=hard; follow_up treated as medium (neutral).
            _last_diff = ACTIONS[self._last_action].difficulty
            _diff_idx  = {"easy": 0, "medium": 1, "hard": 2}.get(_last_diff, 1)

            reward = compute_reward(
                score            = score,
                prev_score       = prev_score,
                nervousness      = nervousness,
                prev_nervousness = prev_nerv,
                star_count       = star_count,
                word_count       = word_count,
                prev_action_idx  = self._last_action,
                curr_action_idx  = self._last_action,
                q_type           = q_type,
                difficulty_idx   = _diff_idx,
            )

            # Recompute stress_penalty separately so it can be stored in the
            # SequencerStep for audit logging and ablation analysis.
            _nerv_delta = nervousness - prev_nerv
            _stress_penalty = (
                -W_STRESS_PENALTY * _nerv_delta * (_diff_idx / 2.0)
                if _nerv_delta > 0.0 else 0.0
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
            reward         = 0.0
            _stress_penalty = 0.0

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
            q_number      = len(self._history) + 1,
            state         = curr_state,
            action_idx    = next_action_idx,
            action        = chosen.label(),
            score         = score,
            nervousness   = nervousness,
            star_count    = star_count,
            reward        = reward,
            q_type        = chosen.q_type,
            difficulty    = chosen.difficulty,
            follow_up     = chosen.follow_up,
            stress_penalty= round(_stress_penalty, 4),
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

        stress_steps = [s for s in self._history if s.stress_penalty < -0.01]

        return {
            "steps_recorded":    len(self._history),
            "final_epsilon":     round(self._eps, 4),
            "action_distribution": action_counts,
            "follow_up_count":   follow_up_count,
            "shared_prior_used": self._use_shared_prior,
            "stress_penalty_total": round(sum(s.stress_penalty for s in self._history), 4),
            "stress_spike_count":   len(stress_steps),    # questions that triggered penalty
            "stress_spike_steps":   [                     # which questions caused spikes
                {
                    "q":             s.q_number,
                    "action":        s.action,
                    "nervousness":   round(s.nervousness, 3),
                    "stress_penalty": round(s.stress_penalty, 4),
                }
                for s in stress_steps
            ],
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
                    "q":            s.q_number,
                    "action":       s.action,
                    "score":        s.score,
                    "nerv":         round(s.nervousness, 3),
                    "reward":       round(s.reward, 3),
                    "stress_penalty": round(s.stress_penalty, 4),
                    "followup":     s.follow_up,
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

    # ── Split API: update() + select_action() ─────────────────────────────────
    # main.py calls these separately for cleaner separation of concerns.
    # record_and_select() remains available as the combined single-call API.

    def update(self, reward: float) -> None:
        """
        Apply one Bellman Q-update for the last (state, action) pair.

        Called by main.py AFTER computing reward externally via compute_reward().
        Updates Q(last_state, last_action) using the standard Q-learning equation:
            Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',·) − Q(s,a)]

        NOTE: Only call this once per answer — calling record_and_select() AND
        update() for the same answer would double-update the Q-table.
        """
        if self._last_state is None:
            # No previous state yet (first answer) — nothing to update
            return
        # Encode current state from the most recent scores/nervousness
        recent_scores = self._scores[-3:] if len(self._scores) >= 3 else self._scores
        avg_recent    = float(np.mean(recent_scores)) if recent_scores else 2.5
        nerv          = self._nervousness[-1] if self._nervousness else 0.3
        star_rate     = 0.0   # not available here — use neutral
        curr_state    = encode_state(avg_recent, nerv, star_rate, 50.0)

        old_q      = self._q[self._last_state][self._last_action]
        next_max_q = float(np.max(self._q[curr_state]))
        new_q      = old_q + self._lr * (reward + self._gamma * next_max_q - old_q)
        self._q[self._last_state][self._last_action] = new_q
        log.debug(f"[RL] update() — Q({self._last_state},{self._last_action}): "
                  f"{old_q:.3f} → {new_q:.3f} (reward={reward:.3f})")

    def select_action(self,
                      score:       float = 3.0,
                      nervousness: float = 0.3,
                      star_rate:   float = 0.5,
                      time_eff:    float = 50.0,
                      word_count:  int   = 100,
                      q_type:      str   = "") -> Action:
        """
        ε-greedy action selection for the NEXT question without Q-table update.

        Called by main.py after update() to pick the next (type, difficulty).
        Records the new state, decays ε, appends to history, and returns the
        chosen Action object.

        Unlike record_and_select(), this method does NOT perform the Bellman
        update — that is main.py's responsibility via update().

        Args:
            score       : most recent answer score 1–5
            nervousness : fused nervousness 0–1
            star_rate   : STAR coverage fraction 0–1
            time_eff    : timing score 0–100
            word_count  : answer word count (for shallow-answer detection)
            q_type      : question type of the answer just given (for history)
        """
        self._scores.append(score)
        self._nervousness.append(nervousness)
        self._step_count += 1

        curr_state = encode_state(score, nervousness, star_rate, time_eff)

        # ── Follow-up override ────────────────────────────────────────────────
        star_count_int = int(round(star_rate * 4))
        shallow = _shallow_answer_detected(word_count, star_count_int)
        if shallow and not self._followed_up_last:
            next_action_idx = FOLLOW_UP_ACTION_IDX
            log.info(f"[RL] select_action: shallow answer (wc={word_count}, "
                     f"star_count≈{star_count_int}) → follow_up override")
            self._followed_up_last = True
        else:
            self._followed_up_last = False
            if random.random() < self._eps:
                next_action_idx = random.randint(0, N_ACTIONS - 2)
                log.debug(f"[RL] Exploring → {ACTIONS[next_action_idx].label()}")
            else:
                q_no_followup   = self._q[curr_state][:N_ACTIONS - 1]
                next_action_idx = int(np.argmax(q_no_followup))
                log.debug(f"[RL] Exploiting → {ACTIONS[next_action_idx].label()} "
                          f"Q={self._q[curr_state][next_action_idx]:.3f}")

        # Decay ε
        self._eps = max(self._eps_end, self._eps * self._eps_decay)

        chosen = ACTIONS[next_action_idx]
        self._history.append(SequencerStep(
            q_number    = len(self._history) + 1,
            state       = curr_state,
            action_idx  = next_action_idx,
            action      = chosen.label(),
            score       = score,
            nervousness = nervousness,
            star_count  = star_count_int,
            reward      = 0.0,   # reward was applied separately via update()
            q_type      = chosen.q_type,
            difficulty  = chosen.difficulty,
            follow_up   = chosen.follow_up,
        ))

        self._last_state  = curr_state
        self._last_action = next_action_idx
        return chosen

    @property
    def epsilon(self) -> float:
        """Alias for current_epsilon — used by main.py."""
        return round(self._eps, 4)

    @property
    def current_epsilon(self) -> float:
        return round(self._eps, 4)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_score(self) -> float:
        """Most recent answer score (1-5). Returns neutral 3.0 before first answer."""
        return self._scores[-1] if self._scores else 3.0

    @property
    def last_nervousness(self) -> float:
        """Most recent fused nervousness (0-1). Returns neutral 0.3 before first answer."""
        return self._nervousness[-1] if self._nervousness else 0.3

    @property
    def last_action_idx(self) -> int:
        """Index of the last action taken by the sequencer."""
        return self._last_action