# TruthLens 🔍

> A multi-agent global news fact-checking system for the American public.

TruthLens ingests news from around the world, extracts claims, and classifies them as
**empirically verifiable** (TRUE / FALSE / UNVERIFIED) or **politically contested**
(transparency report showing what all credible sources say — no verdict imposed).

Built with Python, LangGraph, FastAPI, PostgreSQL + pgvector, Gemini, and Groq.
100% open source. Zero paid infrastructure required.

---

## Architecture

```
NewsIngestionAgent       → fetches RSS / NewsAPI from global outlets
ClaimExtractorAgent      → extracts checkable claims from articles        ← IMPLEMENTED (v0.1)
ClaimClassifierAgent     → empirical vs. contested/political
FactCheckerAgent         → queries PolitiFact, Snopes, ClaimBuster, web
EvidenceRankerAgent      → dynamic source weighting + consensus scoring
VerdictAgent             → final verdict + confidence score
SummaryAgent             → formats output for dashboard
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + custom graph abstractions |
| LLM (primary) | Google Gemini 1.5 Pro |
| LLM (fast) | Groq / Llama 3.3 70b |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy (async) |
| API | FastAPI |
| Frontend | Next.js (coming soon) |
| Logging | Async fire-and-forget → PostgreSQL |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/truthlens
cd truthlens
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY and GROQ_API_KEY

# 3. Start PostgreSQL
docker-compose up -d db

# 4. Run migrations
python scripts/migrate.py

# 5. Run the pipeline (on-demand)
python -m api.main

# 6. Or run the scheduler
python scripts/scheduler.py
```

## Accuracy Philosophy

TruthLens does not claim to be a neutral arbiter of political truth.
- **Empirical claims** (statistics, events, scientific consensus) → verdict with confidence score
- **Political claims** → transparency report showing multiple credible source perspectives

> Accuracy target: >90% on empirically verifiable claims against ground truth datasets.

## Contributing

PRs welcome. See CONTRIBUTING.md.
