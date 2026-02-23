import base64
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.engine import get_session
from orchestrator.db.repositories import ModelRepository, TrainingJobRepository
from orchestrator.schemas.training import CreateTrainingJobRequest, TrainingJobResponse

router = APIRouter(prefix="/api/v1/training", tags=["training"])

# Redis instance will be set by the app lifespan
_redis: Redis | None = None


def set_redis(redis: Redis) -> None:
    global _redis
    _redis = redis


def _get_repo(session: AsyncSession = Depends(get_session)) -> TrainingJobRepository:
    return TrainingJobRepository(session)


def _job_response(job, model=None) -> TrainingJobResponse:
    """Build a TrainingJobResponse with model info."""
    data = TrainingJobResponse.model_validate(job, from_attributes=True)
    if model:
        data.model_name = model.name
        data.architecture = model.architecture
    return data


@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(
    request: CreateTrainingJobRequest,
    session: AsyncSession = Depends(get_session),
):
    model_repo = ModelRepository(session)
    job_repo = TrainingJobRepository(session)

    if request.model_id:
        # Verify model exists
        model = await model_repo.get(request.model_id)
        if not model:
            raise HTTPException(status_code=400, detail="Model not found")
        model_id = model.id
    else:
        # Create a default MNIST model (backward compat)
        model = await model_repo.create(
            name="MNIST (auto)",
            architecture="mnist",
        )
        model_id = model.id

    job = await job_repo.create(
        num_rounds=request.num_rounds,
        min_devices=request.min_devices,
        learning_rate=request.learning_rate,
        config=request.config,
        model_id=model_id,
    )

    return _job_response(job, model)


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    status: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    job_repo = TrainingJobRepository(session)
    model_repo = ModelRepository(session)
    jobs = await job_repo.list_all(status=status)
    results = []
    for job in jobs:
        model = await model_repo.get(job.model_id) if job.model_id else None
        results.append(_job_response(job, model))
    return results


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    job_repo = TrainingJobRepository(session)
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    model = None
    if job.model_id:
        model_repo = ModelRepository(session)
        model = await model_repo.get(job.model_id)
    return _job_response(job, model)


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


@router.post("/jobs/{job_id}/retry")
async def retry_training_job(
    job_id: uuid.UUID,
    repo: TrainingJobRepository = Depends(_get_repo),
):
    job = await repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status != "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job.status}, only failed jobs can be retried",
        )
    await repo.update(job_id, status="running")
    return {
        "status": "running",
        "job_id": str(job_id),
        "resume_from_round": job.current_round + 1,
    }


@router.get("/jobs/{job_id}/model")
async def download_model(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    if not _redis:
        raise HTTPException(status_code=503, detail="Redis not available")

    # Try to find model_id from the job
    job_repo = TrainingJobRepository(session)
    job = await job_repo.get(job_id)
    model_key_id = str(job.model_id) if job and job.model_id else str(job_id)

    encoded = await _redis.get(f"model:{model_key_id}:global")
    if not encoded:
        # Fallback: try with job_id directly (backward compat)
        encoded = await _redis.get(f"model:{job_id}:global")
    if not encoded:
        raise HTTPException(status_code=404, detail="Model not found")

    model_bytes = base64.b64decode(encoded)
    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=model-{job_id}.bin"},
    )
