import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.engine import get_session
from orchestrator.db.repositories import DeviceRepository
from orchestrator.schemas.device import DeviceResponse

router = APIRouter(prefix="/api/v1/devices", tags=["devices"])


def _get_repo(session: AsyncSession = Depends(get_session)) -> DeviceRepository:
    return DeviceRepository(session)


@router.get("", response_model=list[DeviceResponse])
async def list_devices(
    status: str | None = None,
    repo: DeviceRepository = Depends(_get_repo),
):
    devices = await repo.list_all(status=status)
    return devices


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: uuid.UUID,
    repo: DeviceRepository = Depends(_get_repo),
):
    device = await repo.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@router.delete("/{device_id}", status_code=204)
async def delete_device(
    device_id: uuid.UUID,
    repo: DeviceRepository = Depends(_get_repo),
):
    deleted = await repo.delete(device_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Device not found")


@router.get("/{device_id}/metrics")
async def get_device_metrics(
    device_id: uuid.UUID,
    repo: DeviceRepository = Depends(_get_repo),
):
    device = await repo.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return {
        "device_id": str(device.id),
        "metrics": device.metrics,
        "battery_level": device.battery_level,
        "battery_state": device.battery_state,
        "last_seen_at": device.last_seen_at.isoformat() if device.last_seen_at else None,
    }
