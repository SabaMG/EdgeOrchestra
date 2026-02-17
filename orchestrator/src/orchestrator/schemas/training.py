import uuid
from datetime import datetime

from pydantic import BaseModel


class CreateTrainingJobRequest(BaseModel):
    num_rounds: int = 5
    min_devices: int = 1
    learning_rate: float = 0.01
    config: dict | None = None


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
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
