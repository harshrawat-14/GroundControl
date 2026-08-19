"""POST /eval/run and GET /eval/run/{id} — M3 offline evaluation harness API.

Per LLD 2.7: `POST /eval/run {dataset_id, config} -> {eval_run_id}` kicks off
a benchmark run over a golden dataset synchronously (the golden set is
small; no queue is needed here, unlike ingestion) and persists one
`EvalRun` row plus one `EvalResultRow` per sample. `GET /eval/run/{id}`
returns the run's aggregate jsonb plus the per-sample rows.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import EvalResultRow, EvalRun
from app.db.session import get_session
from app.evaluation.composite import CompositeEvaluator
from app.evaluation.golden_dataset import (
    load_golden_dataset,
    passed,
    resolve_dataset_path,
    run_benchmark,
)

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRunConfig(BaseModel):
    faithfulness_threshold: float | None = None
    groundedness_threshold: float | None = None


class EvalRunRequest(BaseModel):
    dataset_id: str | None = Field(default=None, description="Registered dataset id, defaults to golden_qa")
    name: str | None = None
    config: EvalRunConfig = Field(default_factory=EvalRunConfig)


class EvalRunCreatedResponse(BaseModel):
    eval_run_id: str


class EvalResultItem(BaseModel):
    question: str
    answer: str
    metrics: dict
    passed: bool


class EvalRunDetailResponse(BaseModel):
    eval_run_id: str
    name: str
    dataset: str
    aggregate: dict
    results: list[EvalResultItem]


@router.post("/run", response_model=EvalRunCreatedResponse)
async def start_eval_run(
    req: EvalRunRequest,
    session: AsyncSession = Depends(get_session),
) -> EvalRunCreatedResponse:
    settings = get_settings()
    try:
        dataset_path = resolve_dataset_path(req.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    samples = load_golden_dataset(dataset_path)
    if not samples:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="dataset has no samples")

    evaluator = CompositeEvaluator()
    per_sample, aggregate = await run_benchmark(evaluator, samples)

    faithfulness_threshold = req.config.faithfulness_threshold or settings.faithfulness_threshold
    groundedness_threshold = req.config.groundedness_threshold or settings.groundedness_threshold

    eval_run = EvalRun(
        name=req.name or f"golden-run-{uuid.uuid4().hex[:8]}",
        dataset=req.dataset_id or "golden_qa",
        aggregate=aggregate,
    )
    session.add(eval_run)
    await session.flush()

    for sample_score in per_sample:
        session.add(
            EvalResultRow(
                eval_run_id=eval_run.id,
                question=sample_score.question,
                answer=sample_score.answer,
                metrics=sample_score.result.model_dump(),
                passed=passed(sample_score.result, faithfulness_threshold, groundedness_threshold),
            )
        )

    await session.commit()
    return EvalRunCreatedResponse(eval_run_id=str(eval_run.id))


@router.get("/run/{eval_run_id}", response_model=EvalRunDetailResponse)
async def get_eval_run(
    eval_run_id: str,
    session: AsyncSession = Depends(get_session),
) -> EvalRunDetailResponse:
    try:
        run_uuid = uuid.UUID(eval_run_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="eval run not found") from exc

    eval_run = await session.get(EvalRun, run_uuid)
    if eval_run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="eval run not found")

    result_rows = (
        (await session.execute(select(EvalResultRow).where(EvalResultRow.eval_run_id == run_uuid)))
        .scalars()
        .all()
    )

    return EvalRunDetailResponse(
        eval_run_id=str(eval_run.id),
        name=eval_run.name,
        dataset=eval_run.dataset,
        aggregate=eval_run.aggregate,
        results=[
            EvalResultItem(question=r.question, answer=r.answer, metrics=r.metrics, passed=r.passed)
            for r in result_rows
        ],
    )
