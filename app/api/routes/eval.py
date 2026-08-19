from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run")
async def start_eval_run() -> dict[str, str]:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="wired in M3")


@router.get("/run/{eval_run_id}")
async def get_eval_run(eval_run_id: str) -> dict[str, str]:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="wired in M3")
