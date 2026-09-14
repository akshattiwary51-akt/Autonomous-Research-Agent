"""Request/response schemas for the HTTP API layer."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "awaiting_approval", "done", "error", "aborted"]


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The research question to investigate.")
    max_iterations: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Override the configured MAX_ITERATIONS for this run only.",
    )


class ApprovalRequest(BaseModel):
    approve: bool = Field(..., description="Whether to approve the pending tool call.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    question: str
    report: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 0
    papers_retrieved: int = 0
    evidence_count: int = 0
    contradictions_count: int = 0
    research_gaps_count: int = 0
    termination_reason: Optional[str] = None
    pending_approval: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    progress: list[dict[str, Any]] = Field(default_factory=list)
