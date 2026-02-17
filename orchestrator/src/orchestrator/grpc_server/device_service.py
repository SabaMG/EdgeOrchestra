import uuid

import grpc
import structlog

from orchestrator.db.engine import async_session
from orchestrator.services.device_manager import DeviceManager

logger = structlog.get_logger()


class DeviceRegistryServicer:
    """gRPC service for device registration and management."""

    async def Register(self, request, context):
        from orchestrator.generated import common_pb2, device_pb2

        async with async_session() as session:
            manager = DeviceManager(session)

            caps = request.capabilities
            initial = request.initial_metrics
            battery_state_map = {
                0: None,
                1: "charging",
                2: "discharging",
                3: "full",
                4: "not_charging",
            }

            metrics = None
            battery_level = None
            battery_state = None
            if initial and initial.battery:
                battery_level = initial.battery.level
                battery_state = battery_state_map.get(initial.battery.state)
                metrics = {
                    "cpu_usage": initial.cpu_usage,
                    "memory_usage": initial.memory_usage,
                    "thermal_pressure": initial.thermal_pressure,
                }

            device = await manager.register_device(
                name=request.name,
                device_model=request.device_model,
                os_version=request.os_version,
                chip=caps.chip if caps else None,
                memory_bytes=caps.memory_bytes if caps else None,
                cpu_cores=caps.cpu_cores if caps else None,
                gpu_cores=caps.gpu_cores if caps else None,
                neural_engine_cores=caps.neural_engine_cores if caps else None,
                battery_level=battery_level,
                battery_state=battery_state,
                metrics=metrics,
            )

            return device_pb2.RegisterResponse(
                device_id=common_pb2.DeviceId(value=str(device.id)),
                device=_device_to_proto(device, common_pb2),
            )

    async def Unregister(self, request, context):
        from orchestrator.generated import device_pb2

        async with async_session() as session:
            manager = DeviceManager(session)
            device_id = uuid.UUID(request.device_id.value)
            deleted = await manager.unregister_device(device_id)
            if not deleted:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Device not found")
            return device_pb2.UnregisterResponse()

    async def ListDevices(self, request, context):
        from orchestrator.db.repositories import DeviceRepository
        from orchestrator.generated import common_pb2, device_pb2

        status_map = {
            0: None,
            1: "online",
            2: "offline",
            3: "training",
            4: "error",
        }
        status_filter = status_map.get(request.status_filter)

        async with async_session() as session:
            repo = DeviceRepository(session)
            devices = await repo.list_all(status=status_filter)
            return device_pb2.ListDevicesResponse(
                devices=[_device_to_proto(d, common_pb2) for d in devices]
            )

    async def GetDevice(self, request, context):
        from orchestrator.db.repositories import DeviceRepository
        from orchestrator.generated import common_pb2, device_pb2

        async with async_session() as session:
            repo = DeviceRepository(session)
            device = await repo.get(uuid.UUID(request.device_id.value))
            if not device:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Device not found")
                return device_pb2.GetDeviceResponse()
            return device_pb2.GetDeviceResponse(
                device=_device_to_proto(device, common_pb2)
            )


def _device_to_proto(device, common_pb2):
    from google.protobuf.timestamp_pb2 import Timestamp

    status_map = {
        "online": common_pb2.DEVICE_STATUS_ONLINE,
        "offline": common_pb2.DEVICE_STATUS_OFFLINE,
        "training": common_pb2.DEVICE_STATUS_TRAINING,
        "error": common_pb2.DEVICE_STATUS_ERROR,
    }

    registered_at = Timestamp()
    if device.registered_at:
        registered_at.FromDatetime(device.registered_at)
    last_seen_at = Timestamp()
    if device.last_seen_at:
        last_seen_at.FromDatetime(device.last_seen_at)

    return common_pb2.DeviceInfo(
        id=common_pb2.DeviceId(value=str(device.id)),
        name=device.name,
        device_model=device.device_model,
        os_version=device.os_version,
        capabilities=common_pb2.DeviceCapabilities(
            chip=device.chip or "",
            memory_bytes=device.memory_bytes or 0,
            cpu_cores=device.cpu_cores or 0,
            gpu_cores=device.gpu_cores or 0,
            neural_engine_cores=device.neural_engine_cores or 0,
        ),
        status=status_map.get(device.status, common_pb2.DEVICE_STATUS_UNSPECIFIED),
        registered_at=registered_at,
        last_seen_at=last_seen_at,
    )
