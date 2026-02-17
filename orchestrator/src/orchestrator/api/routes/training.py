import base64
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.engine import get_session
from orchestrator.db.repositories import TrainingJobRepository
from orchestrator.schemas.training import CreateTrainingJobRequest, TrainingJobResponse

router = APIRouter(prefix="/api/v1/training", tags=["training"])

# Redis instance will be set by the app lifespan
_redis: Redis | None = None


def set_redis(redis: Redis) -> None:
    global _redis
    _redis = redis


def _get_repo(session: AsyncSession = Depends(get_session)) -> TrainingJobRepository:
    return TrainingJobRepository(session)


@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(
    request: CreateTrainingJobRequest,
    repo: TrainingJobRepository = Depends(_get_repo),
):
    job = await repo.create(
        num_rounds=request.num_rounds,
        min_devices=request.min_devices,
        learning_rate=request.learning_rate,
        config=request.config,
    )
    return job


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    status: str | None = None,
    repo: TrainingJobRepository = Depends(_get_repo),
):
    return await repo.list_all(status=status)


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: uuid.UUID,
    repo: TrainingJobRepository = Depends(_get_repo),
):
    job = await repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.post("/jobs/{job_id}/stop")
async def stop_training_job(
    job_id: uuid.UUID,
    repo: TrainingJobRepository = Depends(_get_repo),
):
    job = await repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Job is {job.status}, cannot stop")

    await repo.update(job_id, status="stopped")

    if _redis:
        await _redis.set(f"training:{job_id}:stop", "1")

    return {"status": "stopped", "job_id": str(job_id)}


@router.get("/jobs/{job_id}/model")
async def download_model(job_id: uuid.UUID):
    if not _redis:
        raise HTTPException(status_code=503, detail="Redis not available")

    encoded = await _redis.get(f"model:{job_id}:global")
    if not encoded:
        raise HTTPException(status_code=404, detail="Model not found")

    model_bytes = base64.b64decode(encoded)
    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=model-{job_id}.bin"},
    )
