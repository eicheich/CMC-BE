import httpx
from fastapi import APIRouter

from app.schemas import PreviewRequest, PreviewResponse
from app.verification import verify


router = APIRouter(prefix="/api/v1/verifications")


@router.post("/preview", response_model=PreviewResponse)
async def preview(request: PreviewRequest) -> PreviewResponse:
    async with httpx.AsyncClient(timeout=8.0) as client:
        return await verify(request.citation, client, request.style)
