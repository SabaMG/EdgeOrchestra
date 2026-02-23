import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CreateTrainingJobRequest(BaseModel):
    num_rounds: int = Field(5, ge=1, le=1000)
    min_devices: int = Field(1, ge=1, le=100)
    learning_rate: float = Field(0.01, gt=0.0, le=10.0)
    config: dict | None = None
    model_id: uuid.UUID | None = None


class TrainingJobResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    status: str
    num_rounds: int
    current_round: int
    min_devices: int
    learning_rate: float
    round_metrics: dict | None = None
    config: dict | None = None
    model_id: uuid.UUID | None = None
    model_name: str | None = None
    architecture: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
