# Project 03: Autonomous Research Agent

An autonomous academic research agent that investigates a research question through a
**ReAct-style Reason → Act → Observe → Reason loop**, dynamically choosing academic search
tools, evaluating evidence quality, reformulating failed queries, detecting contradictions,
and synthesizing a literature-review-style report with evidence-backed research gaps.

This is deliberately **not** a static `Retrieve → Generate` RAG pipeline. The agent decides,
turn by turn, whether it has enough evidence, what to search for next, and when to stop.

```
python -m app
```

---

## 1. Project Overview

Given a research question, the agent:

1. Optionally decomposes it into specific sub-questions.
2. Reasons about what evidence it still needs.
3. Dynamically selects a tool (ArXiv, Semantic Scholar, or Crossref) via LLM function-calling
   — never hardcoded `if/elif` routing.
4. Executes the search and evaluates the results into **structured evidence** (claim,
   relevance, confidence, source) rather than dumping raw abstracts into context.
5. Detects contradictions between papers.
6. Reformulates its query if a search was unproductive, with a dedicated reflection step
   diagnosing *why* and proposing a genuinely different strategy.
7. Repeats, bounded by hard safety limits, until it decides it has enough evidence.
8. Synthesizes a full academic-report-style write-up: findings, comparative analysis,
   contradictions, limitations, evidence-backed research gaps, and references — every
   citation traceable to an actually-retrieved paper.

## 2. Architecture

```
                         USER
                           |
                           v
                     Research Planner  (dynamic question decomposition)
                           |
                           v
                      Agent Node  (Reason: pick a tool, or declare done)
                           |
                           v
                    Conditional Router
                    /       |         \
                   /        |          \
               Tool      Reflection   Synthesis
               Node        Node          Node
                |            |             |
        +-------+-------+    |       Final Report
        |       |       |    |     (findings, gaps,
      ArXiv  Semantic Crossref|      contradictions,
              Scholar        |         references)
        |       |       |    |
        +-------+-------+    |
                |            |
          Evidence Node      |
       (structured claims,   |
      contradiction check)   |
                |            |
                +----> Agent Node <--------+
                     (iterative loop, bounded)
```

Implemented with **LangGraph** (`app/graph/build_graph.py`) as a `StateGraph` with six
nodes and one conditional router — see §4 for the exact edge list.

## 3. ReAct Loop

Each iteration is one **Reason → Act → Observe** cycle:

- **Reason** (`agent` node, `app/agent.py::decide_action`): the LLM is given the current
  `ResearchState` (as safe structured text, never raw internals) and the tool registry's
  schemas, and picks one of: call a specific tool with specific arguments, or declare the
  evidence sufficient (`finish_research`). Hard safety bounds are checked *before* the LLM
  is even consulted (see §7).
- **Act** (`tool` node, `app/agent.py::execute_tool_call`): executes the chosen tool, or
  short-circuits with a recorded (not crashed) failure if the tool is unknown or the exact
  same call was already made this run.
- **Observe** (`evidence_eval` node, `app/graph/evidence_node.py`): turns the raw papers
  from that search into structured `EvidenceItem`s and checks for contradictions against
  evidence collected so far — this is what the *next* Reason step actually sees, not raw
  abstracts.

The agent never exposes raw chain-of-thought. Every internal decision is captured as safe,
structured metadata (`AgentDecision`, `scratchpad` entries) — tool name, query, a short
rationale string, evidence counts — logged via `app/logging_utils.py` in exactly this shape:

```
ITERATION 2
Tool: arxiv_search
Query: "multimodal RAG scientific document understanding"
Results: 5
Useful: 4
Status: ok
```

## 4. LangGraph State Graph

```
START -> planner -> agent -> [router] -> tool -> evidence_eval -> agent   (loop)
                                      \-> reflection ---------------> agent   (loop)
                                      \-> synthesis -> END
```

- `planner`: dynamic sub-question decomposition (never a hardcoded template — see §8).
- `agent`: reasons, stores its decision in `state["pending_action"]`.
- Router (`app/graph/router.py::route_after_agent`) reads `pending_action` and
  `consecutive_zero_yield_count` to pick `tool`, `reflection`, or `synthesis` —
  deterministically, no LLM call.
- `tool` → `evidence_eval` → back to `agent`: the "Tool Node → Evidence Store → Agent"
  loop from the target architecture.
- `reflection` → `agent`: strategy-change loop (§7).
- `synthesis` → `END`: terminal node, always reached exactly once per run.

Termination is guaranteed by two independent mechanisms even in the worst case (see §7):
the iteration/tool-call hard bounds, and reflection-cycle detection.

## 5. Tool Architecture

`app/tools/base.py` defines `ResearchTool` (an ABC with `.run()` and `.parameters_schema()`)
and `ToolRegistry`, which maps `name -> implementation` and exposes every tool as an
OpenAI/LangChain function-calling schema via `.schemas()`. The agent is handed this list and
picks dynamically — there is no hardcoded topic-based routing anywhere in the codebase.

Three tools are registered by default (`app/tools/registry.py::build_default_registry`):

| Tool | Source | Auth | Notes |
|---|---|---|---|
| `arxiv_search` | ArXiv Atom API | none | preprints, CS/ML/physics/math |
| `semantic_scholar_search` | Semantic Scholar Graph API | optional API key | broader coverage, richer metadata |
| `crossref_search` | Crossref REST API | none | peer-reviewed works, DOIs |

All three share: request timeouts, `tenacity`-based retry with exponential backoff on
transient network errors, explicit `429`/`5xx`/malformed-JSON/malformed-XML handling, and a
hard rule to **never fabricate a `paper_id`** — entries missing required fields are skipped,
not padded with invented data. Every tool schema also carries two optional reasoning
fields (`sub_question`, `rationale`) via `with_reasoning_fields()` so the agent can record
*why* it chose a query, without those fields ever reaching the tool's actual `.run()` call.

## 6. State Management

`app/state.py::ResearchState` is a typed `TypedDict` — never bare chat history. Fields that
multiple graph nodes append to (`search_history`, `evidence`, `contradictions`,
`tool_call_hashes`, `research_gaps`, `scratchpad`, ...) use LangGraph's
`Annotated[list, operator.add]` reducer so updates accumulate correctly across nodes; scalar
fields (`iteration`, `status`, `final_report`) use ordinary last-write-wins semantics. This
split is verified directly against a real compiled `StateGraph` in `tests/test_state.py`,
not assumed.

Supporting dataclasses (`SearchRecord`, `EvidenceItem`, `Contradiction`, `ResearchGap`,
`ToolCallRecord`) are always stored in state as plain dicts (`.as_dict()`), keeping the
whole state JSON-serializable for checkpointing.

## 7. Infinite-Loop Prevention

Three independent safeguards, all implemented in `app/safety/loop_guard.py` and
`app/agent.py`, and all exercised with worst-case adversarial tests (an LLM double that
*never* wants to stop, and one that repeats the *same* call forever):

- **A. Hard iteration bound** — `should_force_synthesis()` checks `iteration >=
  max_iterations` *before* the LLM is consulted at all.
- **B. Tool-call hashing** — `hash_tool_call()` normalizes (lowercase, whitespace-collapsed)
  `(tool, args)` into a SHA-256 hash; an exact repeat is detected and skipped without ever
  hitting the network.
- **C. Zero-yield reflection** — after `ZERO_YIELD_REFLECTION_THRESHOLD` consecutive
  searches with no useful results, the router forces a `reflection` pass instead of another
  search. The reflection node (`app/llm/reflection.py`, `app/graph/reflection_node.py`) makes
  a dedicated LLM call to diagnose *why* recent searches failed and propose a genuinely
  different strategy — this guidance is surfaced back to the agent's very next prompt.
- **Cycle detection** — `reflection_count` accumulates across the run; once it reaches
  `MAX_REFLECTION_ATTEMPTS`, the router gives up into `synthesis` instead of oscillating
  between `agent` and `reflection` forever, even if the LLM keeps ignoring guidance.
- **Independent tool-call ceiling** — `MAX_TOOL_CALLS` is checked separately from
  `MAX_ITERATIONS`, so a generous iteration budget can't bypass a tighter tool-call budget.

A worst-case adversarial test (`tests/test_reflection.py`) proves an LLM that never
converges and never produces useful evidence still terminates well before `max_iterations`.

## 8. Query Refinement

When a search underperforms, the reflection node doesn't just retry with reworded text — it
makes a dedicated LLM call (`OpenAIReflectionAdvisor`) that:

1. Diagnoses *why* recent searches likely failed (grounded only in the actual search
   history, never invented reasons).
2. Proposes a concretely different next step: different terminology, a different tool, a
   narrower/broader scope.

That guidance (`state["reflection_guidance"]`) is shown to the agent's next reasoning turn
with an explicit system-prompt instruction to follow it. Every tool call the agent makes can
also carry a `rationale` string explaining the query choice, recorded in `search_history` and
`scratchpad` — so refinement reasoning is traceable in state, not lost inside an unlogged LLM
call. Duplicate identical queries are hard-blocked (§7.B) regardless.

## 9. Academic Gap Discovery

`app/report/gap_discovery.py` / `app/report/openai_gap_discovery.py`: gaps are proposed only
from actual collected evidence — underexplored areas, unresolved contradictions, missing
evaluations, dataset/methodology limitations — never a generic "more research is needed."
Every `ResearchGap.supporting_evidence_paper_ids` entry is validated against real, retrieved
`paper_id`s; a gap citing zero real papers is dropped entirely, and a gap citing a mix of
real and invented ids keeps only the real ones. `ResearchGap.is_inference` is always `True`
by construction (this system only ever infers gaps from patterns across evidence, never
harvests a paper's own explicit "this is a gap" statement) and the report always labels it
as such: *"Inferred by the research agent from patterns across the evidence — not a claim
directly stated by any single source."*

## 10. Installation

```bash
git clone <this-repo>
cd project03_research_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --break-system-packages   # or omit the flag in a venv
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Note on `langgraph-checkpoint-sqlite`**: this project pins `langgraph==0.2.60` together
with `langgraph-checkpoint==2.1.2` and `langgraph-checkpoint-sqlite==2.0.11`. Installing the
*latest* `langgraph-checkpoint-sqlite` will pull a newer, incompatible `langgraph-checkpoint`
and break imports — always install from `requirements.txt` as pinned, don't upgrade these
three packages independently.

## 11. Environment Setup

Copy `.env.example` to `.env` and fill in at least `OPENAI_API_KEY`:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Key variables (see `.env.example` for the full list with descriptions):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(required for live runs)* | LLM calls |
| `MODEL_NAME` | `gpt-4o-mini` | model for all reasoning/evaluation/synthesis calls |
| `MAX_ITERATIONS` | `6` | hard ReAct loop bound |
| `MAX_TOOL_CALLS` | `12` | independent tool-call bound |
| `ZERO_YIELD_REFLECTION_THRESHOLD` | `2` | unproductive searches before reflection |
| `MAX_REFLECTION_ATTEMPTS` | `2` | reflection attempts before giving up gracefully |
| `CHECKPOINT_BACKEND` | `memory` | `memory` or `sqlite` |
| `ENABLE_HITL` | `false` | pause before every tool call for human approval |

Missing `OPENAI_API_KEY` never crashes the process — every LLM-backed component degrades to
a safe fallback (empty decomposition, an `AgentDecision(action="error")` that routes
straight to a clearly-labeled error report, etc.), verified in `tests/test_llm_openai_clients.py`.

## 12. Usage

```bash
python -m app
```

```
Research question:
> What are the limitations of multimodal RAG for scientific document understanding?

[Planner] Decomposing research question
  Decomposed into 4 sub-question(s):
    - What multimodal RAG approaches currently exist?
    - How do they retrieve information from scientific documents?
    - What limitations have been reported?
    - What datasets and benchmarks are used?
[Agent] Selecting next action
[Tool] Executing search
  [arxiv_search] query='multimodal RAG scientific documents' -> 5 result(s), status=ok
[Evidence] Evaluating retrieved papers
  Extracted 3 evidence item(s) from 5 paper(s) (2 high/medium relevance, 0 contradiction(s) found)
[Agent] Selecting next action
[Synthesis] Generating research report
  Report ready (enough evidence collected).

======================================================================
FINAL REPORT
======================================================================
# Research Question
...
# Research Gaps
### No large-scale benchmark exists for figure/table extraction accuracy
 *(Inferred by the research agent from patterns across the evidence — not
 a claim directly stated by any single source.)*
...
```

With `ENABLE_HITL=true`, the run instead pauses before every tool call and prompts:

```
[HITL] Approval required before calling 'arxiv_search' with args {'query': '...'}
Approve this tool call? [y/N]:
```

### Optional HTTP API

The project is CLI-first, but it also includes a thin FastAPI job API. See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for deployment details and the request/poll/approval
examples. Start it locally with:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

## 13. Testing

```bash
pytest                    # full offline suite; external API calls are mocked
pytest tests/test_tools.py -v     # run a single file
```

All external HTTP (ArXiv, Semantic Scholar, Crossref) is mocked. All LLM calls in tests use
either a scripted fake implementing the relevant interface (`ReasoningClient`,
`QueryDecomposer`, `EvidenceEvaluator`, `ReflectionAdvisor`, `GapDiscoverer`,
`NarrativeWriter`) or exercise the real `OpenAI*` class's graceful-failure path with no API
key configured. Nothing in the suite makes a live network call.

Coverage highlights: tool registration/parsing/failure handling, duplicate-call detection,
max-iteration and max-tool-call enforcement, zero-yield reflection and cycle detection, query
refinement metadata, LangGraph state-merge semantics (against a real compiled graph, not a
mock), full synthesis + gap-extraction + citation-fabrication guards, checkpointing
persistence across a simulated process restart, and HITL pause/resume/reject flows.

## 14. Example Research Query

> "What are the limitations of multimodal RAG for scientific document understanding?"

A worked example (with fake but representative evidence) is in the Phase 10 development
notes; running `python -m app` with a real `OPENAI_API_KEY` and this question is the
intended way to see a live end-to-end run against actual ArXiv/Semantic Scholar/Crossref
results.

## 15. Limitations

- **Abstract-only by default.** Evidence extraction normally reads paper titles/abstracts as
  returned by each API. Set `ENABLE_FULLTEXT_FETCH=true` to attempt PDF retrieval when a
  paper exposes a usable PDF URL; extraction remains bounded by `FULLTEXT_MAX_CHARS` and
  cannot guarantee access to every paper's full methodology or results sections.
- **Bounded by design.** A run that hits `MAX_ITERATIONS` may terminate with an
  incomplete picture rather than fully answering the question; this is intentional
  (Step 11) but means thoroughness trades off against the configured bounds.
- **Contradiction detection is LLM-judged, not verified.** Flagged contradictions are the
  model's read of two claims, not a formally checked logical conflict.
- **Single-model synthesis.** All LLM-backed steps currently use one configured
  `MODEL_NAME`; there's no per-step model selection (e.g., a cheaper model for
  decomposition vs. a stronger one for synthesis).
- **No live-network verification in this development environment** — this sandbox's egress
  does not reach `arxiv.org`/`semanticscholar.org`/`crossref.org`/`api.openai.com`, so every
  external call in the test suite is mocked. The implementation is standards-compliant
  against each API's public documentation, but a first live run in your own environment is
  the real end-to-end check.
- **Semantic Scholar rate limits are strict without an API key.** Set
  `SEMANTIC_SCHOLAR_API_KEY` for anything beyond light use.

## 16. Future Improvements

- Per-step model configuration (cheap/fast model for decomposition and reflection, a
  stronger model for synthesis).
- More robust full-text ingestion for papers whose PDFs are unavailable, scanned, or poorly
  structured.
- A production checkpoint backend (Postgres) — `app/checkpointing.py` is already a single
  swappable factory function, so this is additive, not a rewrite.
- Mandatory (not just optional) HITL gating for large-scale multi-tool fan-out, per Step 14's
  original framing, once real usage patterns clarify what "large-scale" should mean here.
- Evidence-relevance-driven zero-yield detection (currently based on raw result counts, a
  deliberate Phase 8 scope decision — see the evidence-node docstring) once the extra LLM
  cost of relevance-gating every zero-yield check is worth it for a given deployment.
- Richer cycle detection (e.g. semantic similarity between recent queries) beyond exact-hash
  duplicate detection and reflection-attempt counting.
