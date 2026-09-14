"""CLI entry point (Step 21): `python -m app`.

Prompts for a research question, runs the autonomous ReAct research agent,
and prints high-level progress as it works — never raw chain-of-thought,
only the safe structured events already established in
app.logging_utils/app.state (Step 6, Step 18).
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from app.checkpointing import build_checkpointer
from app.config import Settings, get_settings
from app.evidence.evaluator import ContradictionDetector, EvidenceEvaluator
from app.evidence.openai_evaluator import OpenAIContradictionDetector, OpenAIEvidenceEvaluator
from app.fulltext.fetcher import FullTextFetcher, NoOpFullTextFetcher
from app.fulltext.pdf_fetcher import HttpPdfFullTextFetcher
from app.graph.build_graph import build_research_graph
from app.llm.base import ReasoningClient
from app.llm.decomposition import QueryDecomposer
from app.llm.openai_client import OpenAIReasoningClient
from app.llm.openai_decomposer import OpenAIQueryDecomposer
from app.llm.openai_reflection import OpenAIReflectionAdvisor
from app.llm.reflection import ReflectionAdvisor
from app.report.gap_discovery import GapDiscoverer
from app.report.narrative import NarrativeWriter
from app.report.openai_gap_discovery import OpenAIGapDiscoverer
from app.report.openai_narrative import OpenAINarrativeWriter
from app.state import create_initial_state
from app.tools.base import ToolRegistry
from app.tools.registry import build_default_registry

NODE_LABELS = {
    "planner": "[Planner] Decomposing research question",
    "agent": "[Agent] Selecting next action",
    "tool": "[Tool] Executing search",
    "evidence_eval": "[Evidence] Evaluating retrieved papers",
    "reflection": "[Reflection] Refining search strategy",
    "synthesis": "[Synthesis] Generating research report",
}


def print_update(node_name: str, update: dict[str, Any]) -> None:
    """Prints one node's safe, structured update. Only ever reads fields
    that are explicitly non-CoT metadata (queries, counts, diagnoses) —
    never anything resembling a raw model reasoning trace."""
    print(NODE_LABELS.get(node_name, f"[{node_name}]"))

    if node_name == "planner":
        sub_questions = update.get("sub_questions") or []
        if sub_questions:
            print(f"  Decomposed into {len(sub_questions)} sub-question(s):")
            for sq in sub_questions:
                print(f"    - {sq}")
        else:
            print("  No decomposition available; reasoning over the raw question.")

    elif node_name == "tool":
        for rec in update.get("search_history") or []:
            status = "ok" if rec.get("success") else f"failed ({rec.get('error')})"
            print(
                f"  [{rec.get('tool')}] query={rec.get('query')!r} -> "
                f"{rec.get('result_count')} result(s), status={status}"
            )

    elif node_name == "evidence_eval":
        for note in update.get("scratchpad") or []:
            if note.get("stage") != "evidence_evaluation":
                continue
            fulltext_note = ""
            if note.get("full_text_fetched"):
                fulltext_note = f", {note['full_text_fetched']} with full text"
            print(
                f"  Extracted {note['evidence_extracted']} evidence item(s) from "
                f"{note['papers_evaluated']} paper(s){fulltext_note} "
                f"({note['high_or_medium_relevance']} high/medium relevance, "
                f"{note['contradictions_found']} contradiction(s) found)"
            )

    elif node_name == "reflection":
        for note in update.get("scratchpad") or []:
            if note.get("stage") != "reflection":
                continue
            print(f"  Diagnosis: {note.get('diagnosis')}")
            print(f"  New strategy: {note.get('suggested_strategy_change')}")

    elif node_name == "synthesis":
        print(f"  Report ready ({update.get('termination_reason', 'complete')}).")


def build_default_components(
    settings: Settings,
) -> dict[str, Any]:
    """Builds the real OpenAI-backed components for a live run. Kept
    separate from `run_research` so tests can inject fakes instead
    (mirroring every other phase's dependency-injection pattern)."""
    return {
        "registry": build_default_registry(),
        "reasoning_client": OpenAIReasoningClient(settings),
        "decomposer": OpenAIQueryDecomposer(settings),
        "evidence_evaluator": OpenAIEvidenceEvaluator(settings),
        "contradiction_detector": OpenAIContradictionDetector(settings),
        "reflection_advisor": OpenAIReflectionAdvisor(settings),
        "gap_discoverer": OpenAIGapDiscoverer(settings),
        "narrative_writer": OpenAINarrativeWriter(settings),
        "fulltext_fetcher": (
            HttpPdfFullTextFetcher(settings) if settings.enable_fulltext_fetch
            else NoOpFullTextFetcher()
        ),
    }


def run_research(
    research_question: str,
    settings: Settings,
    registry: ToolRegistry,
    reasoning_client: ReasoningClient,
    decomposer: QueryDecomposer,
    evidence_evaluator: EvidenceEvaluator,
    contradiction_detector: ContradictionDetector,
    reflection_advisor: ReflectionAdvisor,
    gap_discoverer: GapDiscoverer,
    narrative_writer: NarrativeWriter,
    fulltext_fetcher: FullTextFetcher | None = None,
    on_update: Callable[[str, dict[str, Any]], None] = print_update,
    approve_tool_call: Callable[[dict[str, Any]], bool] | None = None,
    thread_id: str | None = None,
) -> str:
    """Runs one full research loop and returns just the final report text.

    This is the CLI-facing convenience wrapper around `run_research_state`
    (below), which returns the complete final state — use that directly
    if you need iteration counts, evidence, gaps, etc. (e.g. an HTTP API
    layer), not just the report string.
    """
    final_state = run_research_state(
        research_question, settings, registry, reasoning_client, decomposer,
        evidence_evaluator, contradiction_detector, reflection_advisor,
        gap_discoverer, narrative_writer, fulltext_fetcher=fulltext_fetcher,
        on_update=on_update, approve_tool_call=approve_tool_call, thread_id=thread_id,
    )
    return final_state.get("final_report") or "(No report was generated.)"


def run_research_state(
    research_question: str,
    settings: Settings,
    registry: ToolRegistry,
    reasoning_client: ReasoningClient,
    decomposer: QueryDecomposer,
    evidence_evaluator: EvidenceEvaluator,
    contradiction_detector: ContradictionDetector,
    reflection_advisor: ReflectionAdvisor,
    gap_discoverer: GapDiscoverer,
    narrative_writer: NarrativeWriter,
    fulltext_fetcher: FullTextFetcher | None = None,
    on_update: Callable[[str, dict[str, Any]], None] = print_update,
    approve_tool_call: Callable[[dict[str, Any]], bool] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Runs one full research loop and returns the COMPLETE final
    `ResearchState` dict (report, iteration count, evidence, gaps,
    contradictions, status, etc.) — the full picture, for callers that
    need more than just the report text (e.g. an API layer).

    `approve_tool_call` is consulted whenever HITL is enabled and the graph
    pauses before a tool execution; defaults to an interactive `input()`
    prompt if not provided. Only relevant when `settings.enable_hitl` is
    True (see app.checkpointing / build_research_graph's `enable_hitl`).

    `thread_id` identifies this run for checkpointing/resume purposes;
    defaults to a hash of the question (CLI convenience) — callers that
    need to track/resume many concurrent runs (e.g. an API layer issuing
    one job per request) should pass an explicit unique id instead.
    """
    checkpointer = build_checkpointer(settings)
    graph = build_research_graph(
        registry, reasoning_client,
        decomposer=decomposer,
        evidence_evaluator=evidence_evaluator,
        contradiction_detector=contradiction_detector,
        reflection_advisor=reflection_advisor,
        gap_discoverer=gap_discoverer,
        narrative_writer=narrative_writer,
        fulltext_fetcher=fulltext_fetcher,
        checkpointer=checkpointer,
        enable_hitl=settings.enable_hitl,
    )

    initial_state = create_initial_state(research_question, max_iterations=settings.max_iterations)
    config = {
        "configurable": {"thread_id": thread_id or f"cli-{abs(hash(research_question))}"},
        "recursion_limit": 100,
    }

    resume_input: Any = initial_state
    while True:
        for event in graph.stream(resume_input, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                if node_name == "__interrupt__":
                    continue  # LangGraph's own pause marker, not a real node
                on_update(node_name, update)

        snapshot = graph.get_state(config)
        if not snapshot.next:
            break  # graph reached END

        # Paused for HITL approval.
        pending = snapshot.values.get("pending_action") or {}
        approved = (approve_tool_call or _default_approve_tool_call)(pending)
        if not approved:
            return {
                **snapshot.values,
                "status": "aborted",
                "final_report": (
                    "(Run stopped before completion: tool call "
                    f"{pending.get('tool_name')!r} was not approved by the user.)"
                ),
            }
        resume_input = None  # resume with no new input, same thread

    return graph.get_state(config).values


def _default_approve_tool_call(pending_action: dict[str, Any]) -> bool:
    print(
        f"\n[HITL] Approval required before calling "
        f"{pending_action.get('tool_name')!r} with args {pending_action.get('tool_args')!r}"
    )
    try:
        answer = input("Approve this tool call? [y/N]: ").strip().lower()
    except EOFError:
        # No interactive stdin available (e.g. running under cron, in a
        # container with no TTY attached, or piped input already
        # consumed). Fail safe: reject rather than silently approving an
        # unattended external API call.
        print(
            "No interactive input available to approve this tool call; "
            "rejecting by default. Run with ENABLE_HITL=false for "
            "unattended execution, or provide an approve_tool_call "
            "callback when embedding this project."
        )
        return False
    return answer == "y"


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Autonomous academic research agent (ReAct + LangGraph).",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Research question. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Write the final report to this file instead of printing it "
             "(progress lines still print to stdout unless --quiet).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-node progress output; print only the final report.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if not args.quiet:
        print("=" * 70)
        print("Autonomous Academic Research Agent")
        print("=" * 70)

    settings = get_settings()
    if not settings.has_llm_credentials:
        print(
            "\nWarning: OPENAI_API_KEY is not set. LLM-backed reasoning steps "
            "will fail gracefully and the run will terminate quickly with an "
            "error report. Set OPENAI_API_KEY in your environment or .env "
            "file for a real run.\n"
        )

    if args.question:
        research_question = args.question.strip()
    else:
        try:
            research_question = input("\nResearch question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted: no research question provided and no interactive "
                  "input available. Pass one as an argument instead: "
                  "python -m app \"your question\"")
            sys.exit(1)

    if not research_question:
        print("No question provided. Exiting.")
        sys.exit(1)

    components = build_default_components(settings)
    on_update = (lambda node_name, update: None) if args.quiet else print_update

    if not args.quiet:
        print()
    try:
        report = run_research(research_question, settings, on_update=on_update, **components)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)

    if args.output:
        from pathlib import Path
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}")
    else:
        if not args.quiet:
            print("\n" + "=" * 70)
            print("FINAL REPORT")
            print("=" * 70 + "\n")
        print(report)


if __name__ == "__main__":
    main()
