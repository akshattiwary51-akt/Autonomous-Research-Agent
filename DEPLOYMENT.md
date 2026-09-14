# Deployment Guide

This project is a **CLI tool**, not a web server -- there's no HTTP port to expose.
"Deploying" it means: run it somewhere other than your dev machine, with state that
survives restarts, and ideally without needing to sit at a keyboard for each run. This doc
covers that.

## Before deploying: production-readiness checklist

1. **Switch off the in-memory checkpointer.** `CHECKPOINT_BACKEND=memory` (the default)
   loses all state the moment the process exits -- fine for local testing, wrong for
   anything you walk away from. Set:
   ```
   CHECKPOINT_BACKEND=sqlite
   CHECKPOINT_DB_PATH=/data/checkpoints.sqlite
   ```
2. **Never commit `.env`.** Inject secrets (`OPENAI_API_KEY` etc.) via your deploy
   platform's secret manager / environment variable injection, not a file baked into an
   image or repo.
3. **Decide interactive vs. non-interactive.** The CLI supports both:
   ```
   python -m app                           # interactive prompt (needs a TTY)
   python -m app "your research question"   # non-interactive, scriptable
   python -m app "..." --output report.md --quiet   # for cron/automation
   ```
   If `ENABLE_HITL=true` and you run non-interactively with no TTY attached, any tool-call
   approval prompt safely **rejects by default** rather than crashing -- so don't enable
   HITL for unattended automated runs unless you also wire a real approval mechanism (see
   "Embedding in another service" below).
4. **Pick your LLM provider bounds.** `MAX_ITERATIONS` / `MAX_TOOL_CALLS` directly control
   API spend per run -- set them deliberately for a shared/production deployment, not just
   the local-dev defaults.

## Option A -- Docker (recommended)

```bash
cd project03_research_agent
docker build -t research-agent .
docker run --rm -it \
  --env-file .env \
  -v research-agent-data:/data \
  research-agent "What are the limitations of multimodal RAG?"
```

Or with `docker-compose` (also handles the persistent volume for you):

```bash
docker compose run --rm research-agent "your research question"
```

To run interactively (prompts for the question, and can show HITL approval prompts):

```bash
docker compose run --rm research-agent
```

**Note**: I don't have Docker available in the environment I built this in, so the
`Dockerfile`/`docker-compose.yml` are written to standard practice but not build-tested
here -- please do that first build yourself and let me know if anything needs adjusting.

## Option B -- Bare server (VM / VPS / on-prem box)

```bash
git clone <repo> && cd project03_research_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in secrets, set CHECKPOINT_BACKEND=sqlite
python -m app "your research question" --output report.md --quiet
```

For recurring/scheduled runs, a plain cron entry works since the CLI is fully scriptable:

```cron
0 6 * * * cd /opt/research-agent && .venv/bin/python -m app "$(cat question.txt)" \
  --output /var/reports/report-$(date +\%F).md --quiet >> /var/log/research-agent.log 2>&1
```

## Option C -- Cloud (batch/job-style, not a persistent server)

Since there's no HTTP server, this fits **job/task** compute better than a long-running
service:
- **AWS**: Fargate/ECS scheduled task, or a Lambda container image (mind Lambda's execution
  time limit against `MAX_ITERATIONS` -- a slow research run could exceed it).
- **GCP**: Cloud Run Jobs, or a scheduled Cloud Function (container-based, same time-limit
  caveat).
- **Any VM provider** (DigitalOcean, Linode, etc.): just Option B above, optionally with
  the cron pattern.

Use the SQLite checkpoint file on attached/persistent storage (an EBS volume, a Cloud Run
Jobs mounted volume, etc.) if you want research runs to survive a job restart mid-way.

## Embedding in another service

The HTTP API above already covers the most common case. If you want tighter, in-process
integration instead of an HTTP round-trip, import `run_research_state` directly (this is
exactly what `app/api/jobs.py` does):

```python
from app.__main__ import build_default_components, run_research_state
from app.config import get_settings

settings = get_settings()
components = build_default_components(settings)
final_state = run_research_state(
    "your research question", settings,
    on_update=my_progress_callback,       # e.g. push to a websocket
    approve_tool_call=my_approval_logic,  # e.g. a real human-approval queue, if HITL is on
    thread_id="my-unique-job-id",         # for checkpointing/resume tracking
    **components,
)
report = final_state["final_report"]
# final_state also has: status, iteration, evidence, contradictions,
# research_gaps, retrieved_papers, termination_reason, etc.
```

(`run_research` — no `_state` suffix — is the CLI's thinner convenience wrapper that returns
just the report string; use `run_research_state` when you need the full picture.)

This is the same dependency-injection pattern used throughout the project and in its test
suite -- neither function touches stdin/stdout directly except through the callbacks you
provide.

## What's intentionally NOT included

- No multi-tenant secret/key management -- one process, one `.env`/environment, one
  provider key.
- No autoscaling or queueing -- each invocation is a single synchronous research run.
- The HTTP API's job store is in-memory and single-process (see `app/api/jobs.py`) -- fine
  for one server instance, not a substitute for a real job queue (Celery/RQ/etc.) if you
  need multi-node scaling. Jobs are also lost on process restart (the LangGraph checkpoint
  itself can survive via `CHECKPOINT_BACKEND=sqlite`, but the API's job/progress tracking
  does not, currently).

## HTTP API (optional)

The project is CLI/library-first, but a thin FastAPI wrapper is included for callers that
want it over HTTP instead of a CLI invocation:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Research runs are asynchronous (they take multiple LLM calls, and can pause indefinitely
under HITL) so the API uses a job/poll pattern rather than a single blocking request:

```bash
# Start a job -- returns immediately with a job_id
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the limitations of multimodal RAG?", "max_iterations": 6}'

# Poll for status/result
curl http://localhost:8000/research/<job_id>

# If ENABLE_HITL=true and the job is "awaiting_approval":
curl -X POST http://localhost:8000/research/<job_id>/approve \
  -H "Content-Type: application/json" -d '{"approve": true}'
```

`GET /research/{job_id}` returns `status` (`queued` | `running` | `awaiting_approval` |
`done` | `error` | `aborted`), the `report` once done, iteration/evidence/gap counts, and a
`progress` list of the same safe, non-chain-of-thought events the CLI prints (Step 6/18) --
never raw internal reasoning.

Run it in Docker the same way — override the `Dockerfile`'s `ENTRYPOINT` for the API
container, or add a second service to `docker-compose.yml` running
`uvicorn app.api.main:app --host 0.0.0.0 --port 8000` with the port published.
