"""Prototype local research API; NOT an authenticated robot control service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import Field

from vla_forge.belief import (
    BudgetedRepairPlanner, DiagnosticProbe, FailureBelief, PlannerSettings,
    Recommendation, RepairOption,
)
from vla_forge.evaluation import (
    ReleaseDecision, ReleaseThresholds, evaluate_release, summarize_pairs,
)
from vla_forge.models import PairOutcome, RepairCase, StrictModel
from vla_forge.storage import CaseConflictError, CaseRepository


class DiagnosticRequest(StrictModel):
    belief: FailureBelief
    options: list[RepairOption]
    probes: list[DiagnosticProbe] = Field(default_factory=list)
    settings: PlannerSettings
    remaining_budget: float | None = Field(default=None, ge=0)


class EventRequest(StrictModel):
    event_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    payload: dict[str, Any]


class ReleaseRequest(StrictModel):
    target: list[PairOutcome] = Field(min_length=1)
    retention: list[PairOutcome] = Field(min_length=1)
    p95_latency_ms: float = Field(ge=0)
    thresholds: ReleaseThresholds = Field(default_factory=ReleaseThresholds)


def create_app(database_path: str | Path) -> FastAPI:
    app = FastAPI(
        title="VLA-Forge Research API", version="0.1.0",
        description="Research-only repair evidence API, not a robot control service.",
    )
    repository = CaseRepository(database_path)
    app.state.repository = repository

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "component": "research_api"}

    @app.post("/v1/cases", status_code=201)
    def create_case(case: RepairCase) -> dict[str, str]:
        try:
            digest = repository.insert_case(case)
        except CaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"case_id": case.failure.case_id, "sha256": digest}

    @app.get("/v1/cases", response_model=list[RepairCase])
    def list_cases(limit: int = 100) -> list[RepairCase]:
        try:
            return repository.list_cases(limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/cases/{case_id}", response_model=RepairCase)
    def get_case(case_id: str) -> RepairCase:
        record = repository.get_case(case_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        return record

    @app.post("/v1/cases/{case_id}/events", status_code=201)
    def append_event(case_id: str, event: EventRequest) -> dict[str, str]:
        if repository.get_case(case_id) is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        try:
            digest = repository.append_event(event.event_id, case_id, event.kind, event.payload)
        except CaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"event_id": event.event_id, "sha256": digest}

    @app.get("/v1/cases/{case_id}/events")
    def list_events(case_id: str) -> list[dict[str, object]]:
        if repository.get_case(case_id) is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        return repository.list_events(case_id)

    @app.post("/v1/diagnosis/recommend", response_model=Recommendation)
    def recommend(request: DiagnosticRequest) -> Recommendation:
        try:
            planner = BudgetedRepairPlanner(
                request.options, request.probes, request.settings
            )
            return planner.recommend(request.belief, request.remaining_budget)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/releases/evaluate", response_model=ReleaseDecision)
    def release(request: ReleaseRequest) -> ReleaseDecision:
        try:
            target = summarize_pairs(request.target)
            retention = summarize_pairs(request.retention)
            return evaluate_release(
                target=target, retention=retention,
                p95_latency_ms=request.p95_latency_ms,
                thresholds=request.thresholds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app(os.environ.get("VLA_FORGE_DB_PATH", "data/vla_forge.sqlite"))
