import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CreateModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    architecture: str = Field(..., min_length=1, max_length=50)
    parent_model_id: uuid.UUID | None = None


class ModelResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    name: str
    architecture: str
    version: int
    status: str
    parent_model_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime


class ArchitectureResponse(BaseModel):
    key: str
    name: str
    input_shape: list[int]
    num_classes: int
    layer_names: list[str]
