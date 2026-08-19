from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def run_query(req: QueryRequest) -> QueryResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail="wired in M1")
