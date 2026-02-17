import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    device_model: Mapped[str] = mapped_column(String(255), nullable=False)
    os_version: Mapped[str] = mapped_column(String(50), nullable=False)
    chip: Mapped[str | None] = mapped_column(String(100))
    memory_bytes: Mapped[int | None] = mapped_column(BigInteger)
    cpu_cores: Mapped[int | None] = mapped_column(Integer)
    gpu_cores: Mapped[int | None] = mapped_column(Integer)
    neural_engine_cores: Mapped[int | None] = mapped_column(Integer)

    battery_level: Mapped[float | None] = mapped_column(Float)
    battery_state: Mapped[str | None] = mapped_column(String(20))

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="online")
    metrics: Mapped[dict | None] = mapped_column(JSON)

    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    notes: Mapped[str | None] = mapped_column(Text)
