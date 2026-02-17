import uuid

import structlog

from orchestrator.db.engine import async_session
from orchestrator.services.heartbeat_monitor import HeartbeatMonitor

logger = structlog.get_logger()


class HeartbeatServiceServicer:
    """Bidirectional streaming heartbeat service."""

    def __init__(self, heartbeat_monitor: HeartbeatMonitor) -> None:
        self.monitor = heartbeat_monitor

    async def Heartbeat(self, request_iterator, context):
        from orchestrator.generated import heartbeat_pb2

        async for request in request_iterator:
            device_id = uuid.UUID(request.device_id.value)

            metrics = {}
            battery_level = None
            battery_state = None
            if request.HasField("metrics"):
                metrics = {
                    "cpu_usage": request.metrics.cpu_usage,
                    "memory_usage": request.metrics.memory_usage,
                    "thermal_pressure": request.metrics.thermal_pressure,
                }
                if request.metrics.HasField("battery"):
                    battery_level = request.metrics.battery.level
                    battery_state_map = {
                        0: None,
                        1: "charging",
                        2: "discharging",
                        3: "full",
                        4: "not_charging",
                    }
                    battery_state = battery_state_map.get(request.metrics.battery.state)

            async with async_session() as session:
                await self.monitor.process_heartbeat(
                    session, device_id, metrics,
                    battery_level=battery_level,
                    battery_state=battery_state,
                )

            command = await self.monitor.get_pending_command(str(device_id))

            if command:
                cmd_map = {
                    "update_interval": heartbeat_pb2.HEARTBEAT_COMMAND_UPDATE_INTERVAL,
                    "start_training": heartbeat_pb2.HEARTBEAT_COMMAND_START_TRAINING,
                    "stop_training": heartbeat_pb2.HEARTBEAT_COMMAND_STOP_TRAINING,
                    "shutdown": heartbeat_pb2.HEARTBEAT_COMMAND_SHUTDOWN,
                }
                yield heartbeat_pb2.HeartbeatResponse(
                    command=cmd_map.get(
                        command.get("type", ""),
                        heartbeat_pb2.HEARTBEAT_COMMAND_ACK,
                    ),
                    ack_sequence=request.sequence,
                    parameters=command.get("parameters", {}),
                )
            else:
                yield heartbeat_pb2.HeartbeatResponse(
                    command=heartbeat_pb2.HEARTBEAT_COMMAND_ACK,
                    ack_sequence=request.sequence,
                )
