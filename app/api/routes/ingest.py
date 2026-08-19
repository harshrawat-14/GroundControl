from __future__ import annotations

from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(prefix="/ingest", tags=["ingest"])

_STATUS_MAP = {
    JobStatus.deferred: "queued",
    JobStatus.queued: "queued",
    JobStatus.in_progress: "running",
    JobStatus.complete: "complete",
    JobStatus.not_found: "not_found",
}


class IngestRequest(BaseModel):
    source_type: str
    config: dict[str, Any]


async def _get_pool():
    settings = get_settings()
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


@router.post("")
async def create_ingest_job(req: IngestRequest) -> dict[str, str]:
    if req.source_type not in {"pdf", "web"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"unsupported source_type: {req.source_type}")

    pool = await _get_pool()
    try:
        job = await pool.enqueue_job("run_ingest_job", req.source_type, req.config)
    finally:
        await pool.close()

    if job is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to enqueue ingest job")
    return {"job_id": job.job_id}


@router.get("/{job_id}")
async def get_ingest_job(job_id: str) -> dict[str, Any]:
    pool = await _get_pool()
    try:
        job = Job(job_id, pool)
        job_status = await job.status()
        if job_status == JobStatus.not_found:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")

        response: dict[str, Any] = {
            "job_id": job_id,
            "status": _STATUS_MAP.get(job_status, str(job_status)),
        }
        if job_status == JobStatus.complete:
            result_info = await job.result_info()
            if result_info is not None:
                response["result"] = result_info.result
                response["success"] = result_info.success
        return response
    finally:
        await pool.close()
