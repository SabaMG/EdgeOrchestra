import uuid
from datetime import datetime

from pydantic import BaseModel


class DeviceCapabilitiesSchema(BaseModel):
    chip: str | None = None
    memory_bytes: int | None = None
    cpu_cores: int | None = None
    gpu_cores: int | None = None
    neural_engine_cores: int | None = None


class DeviceMetricsSchema(BaseModel):
    cpu_usage: float | None = None
    memory_usage: float | None = None
    thermal_pressure: float | None = None
    battery_level: float | None = None
    battery_state: str | None = None


class DeviceResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    name: str
    device_model: str
    os_version: str
    chip: str | None = None
    memory_bytes: int | None = None
    cpu_cores: int | None = None
    gpu_cores: int | None = None
    neural_engine_cores: int | None = None
    battery_level: float | None = None
    battery_state: str | None = None
    status: str
    metrics: dict | None = None
    registered_at: datetime
    last_seen_at: datetime
