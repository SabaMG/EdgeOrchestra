import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.engine import get_session
from orchestrator.db.repositories import ModelRepository
from orchestrator.schemas.model import ArchitectureResponse, CreateModelRequest, ModelResponse
from orchestrator.services.model_registry import get_architecture, list_architectures

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def _get_repo(session: AsyncSession = Depends(get_session)) -> ModelRepository:
    return ModelRepository(session)


@router.get("/architectures", response_model=list[ArchitectureResponse])
async def get_architectures():
    return [
        ArchitectureResponse(
            key=a.key,
            name=a.name,
            input_shape=list(a.input_shape),
            num_classes=a.num_classes,
            layer_names=a.layer_names,
        )
        for a in list_architectures()
    ]


@router.post("", response_model=ModelResponse, status_code=201)
async def create_model(
    request: CreateModelRequest,
    repo: ModelRepository = Depends(_get_repo),
):
    try:
        get_architecture(request.architecture)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown architecture: {request.architecture}")

    if request.parent_model_id:
        parent = await repo.get(request.parent_model_id)
        if not parent:
            raise HTTPException(status_code=400, detail="Parent model not found")

    model = await repo.create(
        name=request.name,
        architecture=request.architecture,
        parent_model_id=request.parent_model_id,
    )
    return model


@router.get("", response_model=list[ModelResponse])
async def list_models(
    architecture: str | None = None,
    status: str | None = None,
    repo: ModelRepository = Depends(_get_repo),
):
    models = await repo.list_all(architecture=architecture)
    if status:
        models = [m for m in models if m.status == status]
    return models


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: uuid.UUID,
    repo: ModelRepository = Depends(_get_repo),
):
    model = await repo.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/{model_id}")
async def delete_model(
    model_id: uuid.UUID,
    repo: ModelRepository = Depends(_get_repo),
):
    model = await repo.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.status == "training":
        raise HTTPException(status_code=400, detail="Cannot delete model while training")

    deleted = await repo.delete(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "model_id": str(model_id)}
