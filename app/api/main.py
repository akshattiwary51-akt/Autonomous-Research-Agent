"""HTTP API for the research agent (optional -- the project is
CLI/library-first; this is a thin wrapper for callers that want it over
HTTP).

Run with: uvicorn app.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /research                    start a research job, returns immediately
  GET  /research/{job_id}            poll job status/progress/report
  POST /research/{job_id}/approve     approve or reject a paused HITL tool call
  GET  /health                        liveness check
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.api.jobs import JobStore, ResearchJob, default_job_store
from app.api.models import ApprovalRequest, JobStatusResponse, ResearchRequest

app = FastAPI(
    title="Autonomous Research Agent API",
    description=(
        "HTTP wrapper around the ReAct research agent. Research runs are "
        "asynchronous: POST to start one, then poll GET for status/result."
    ),
    version="1.0.0",
)


def _job_to_response(job: ResearchJob) -> JobStatusResponse:
    final_state = job.final_state or {}
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,  # type: ignore[arg-type]
        question=job.question,
        report=final_state.get("final_report"),
        iteration=final_state.get("iteration", 0),
        max_iterations=final_state.get("max_iterations", 0),
        papers_retrieved=len(final_state.get("retrieved_papers") or []),
        evidence_count=len(final_state.get("evidence") or []),
        contradictions_count=len(final_state.get("contradictions") or []),
        research_gaps_count=len(final_state.get("research_gaps") or []),
        termination_reason=final_state.get("termination_reason"),
        pending_approval=job.pending_approval,
        error=job.error,
        progress=job.progress,
    )


def get_job_store() -> JobStore:
    """FastAPI dependency -- overridden in tests via
    `app.dependency_overrides[get_job_store]` to inject an isolated store
    with fake components instead of the real OpenAI-backed default."""
    return default_job_store


@app.post("/research", response_model=JobStatusResponse, status_code=202)
def start_research(req: ResearchRequest, store: JobStore = Depends(get_job_store)) -> JobStatusResponse:
    job = store.create(req.question, max_iterations=req.max_iterations)
    return _job_to_response(job)


@app.get("/research/{job_id}", response_model=JobStatusResponse)
def get_research(job_id: str, store: JobStore = Depends(get_job_store)) -> JobStatusResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_to_response(job)


@app.post("/research/{job_id}/approve", response_model=JobStatusResponse)
def approve_research(
    job_id: str, req: ApprovalRequest, store: JobStore = Depends(get_job_store)
) -> JobStatusResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "awaiting_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not awaiting approval (current status: {job.status!r}).",
        )
    job.submit_approval(req.approve)
    return _job_to_response(job)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
