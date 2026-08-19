from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("")
async def create_ingest_job() -> dict[str, str]:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="wired in M1")


@router.get("/{job_id}")
async def get_ingest_job(job_id: str) -> dict[str, str]:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="wired in M1")
