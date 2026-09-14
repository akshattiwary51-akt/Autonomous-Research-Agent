"""In-memory research job store + background execution.

A research run can take a while (multiple sequential LLM calls) and, with
HITL enabled, can pause indefinitely waiting for a human -- neither fits a
single synchronous HTTP request/response. So the API layer uses a
job/poll pattern instead: `POST /research` starts a background thread
running `run_research_state` and returns immediately with a `job_id`;
`GET /research/{job_id}` polls status; `POST /research/{job_id}/approve`
resumes a paused (HITL) run.

This is intentionally a simple in-memory store -- fine for a single-process
deployment (see DEPLOYMENT.md), not a substitute for a real job queue
(Celery/RQ/etc.) if you need multi-process/multi-node scaling.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from app.__main__ import build_default_components, run_research_state
from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="research-job")


def _summarize_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    """Same safe-fields-only principle as app.__main__.print_update, but
    producing a small dict for API responses instead of printing to
    stdout -- never the raw update (which may contain large evidence text
    or internal-only fields)."""
    summary: dict[str, Any] = {"node": node_name}

    if node_name == "planner":
        summary["sub_questions"] = update.get("sub_questions") or []
    elif node_name == "tool":
        summary["search_history"] = [
            {
                "tool": rec.get("tool"), "query": rec.get("query"),
                "result_count": rec.get("result_count"), "success": rec.get("success"),
            }
            for rec in (update.get("search_history") or [])
        ]
    elif node_name == "evidence_eval":
        notes = [n for n in (update.get("scratchpad") or []) if n.get("stage") == "evidence_evaluation"]
        if notes:
            summary["evidence_evaluation"] = notes[0]
    elif node_name == "reflection":
        notes = [n for n in (update.get("scratchpad") or []) if n.get("stage") == "reflection"]
        if notes:
            summary["reflection"] = {
                "diagnosis": notes[0].get("diagnosis"),
                "suggested_strategy_change": notes[0].get("suggested_strategy_change"),
            }
    elif node_name == "synthesis":
        summary["termination_reason"] = update.get("termination_reason")

    return summary


@dataclass
class ResearchJob:
    job_id: str
    question: str
    status: str = "queued"
    final_state: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    pending_approval: Optional[dict[str, Any]] = None
    progress: list[dict[str, Any]] = field(default_factory=list)

    _approval_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _approval_result: Optional[bool] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_update(self, node_name: str, update: dict[str, Any]) -> None:
        with self._lock:
            self.progress.append(_summarize_update(node_name, update))

    def approve_tool_call(self, pending_action: dict[str, Any]) -> bool:
        """Called from the background thread when the graph pauses for
        HITL. Blocks that thread (not the API server's event loop, since
        the whole run happens in a ThreadPoolExecutor worker) until
        `submit_approval` is called from an HTTP request."""
        with self._lock:
            self.pending_approval = pending_action
            self.status = "awaiting_approval"
            self._approval_event.clear()
        self._approval_event.wait()
        with self._lock:
            self.status = "running"
            self.pending_approval = None
            result = bool(self._approval_result)
        return result

    def submit_approval(self, approved: bool) -> None:
        with self._lock:
            self._approval_result = approved
        self._approval_event.set()


class JobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def create(
        self,
        question: str,
        settings: Optional[Settings] = None,
        max_iterations: Optional[int] = None,
        components: Optional[dict[str, Any]] = None,
    ) -> ResearchJob:
        job = ResearchJob(job_id=str(uuid.uuid4()), question=question)
        with self._lock:
            self._jobs[job.job_id] = job

        effective_settings = settings or get_settings()
        if max_iterations is not None:
            effective_settings = effective_settings.model_copy(
                update={"max_iterations": max_iterations}
            )
        effective_components = components or build_default_components(effective_settings)

        _executor.submit(self._run, job, effective_settings, effective_components)
        return job

    def _run(self, job: ResearchJob, settings: Settings, components: dict[str, Any]) -> None:
        job.status = "running"
        try:
            final_state = run_research_state(
                job.question, settings,
                on_update=job.on_update,
                approve_tool_call=job.approve_tool_call,
                thread_id=job.job_id,
                **components,
            )
            job.final_state = final_state
            job.status = final_state.get("status") or "done"
        except Exception as exc:  # noqa: BLE001 - a job must never silently
            # hang the executor; report the failure via job status instead.
            logger.warning("Research job %s failed with an unexpected error: %s", job.job_id, exc)
            job.error = str(exc)
            job.status = "error"

    def get(self, job_id: str) -> Optional[ResearchJob]:
        with self._lock:
            return self._jobs.get(job_id)


# Module-level singleton store, used by app/api/main.py. Tests construct
# their own `JobStore()` instances instead of touching this directly, so
# test runs never share state with each other or with a real server.
default_job_store = JobStore()
