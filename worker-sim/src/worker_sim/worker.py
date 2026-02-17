import asyncio
import logging

import grpc

from orchestrator.generated import common_pb2, device_pb2, device_pb2_grpc
from orchestrator.generated import heartbeat_pb2, heartbeat_pb2_grpc

from worker_sim.device_profile import DeviceProfile
from worker_sim.metrics import MetricsSimulator

logger = logging.getLogger(__name__)


class SimulatedWorker:
    def __init__(
        self,
        target: str,
        profile: DeviceProfile,
        heartbeat_interval: float = 5.0,
    ) -> None:
        self.target = target
        self.profile = profile
        self.heartbeat_interval = heartbeat_interval
        self.metrics_sim = MetricsSimulator(profile)

        self.device_id: str | None = None
        self.running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._channel: grpc.aio.Channel | None = None
        self._sequence = 0

    async def start(self) -> None:
        self._channel = grpc.aio.insecure_channel(self.target)
        await self._register()
        self.running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"[{self.profile.name}] Started (id={self.device_id})")

    async def stop(self) -> None:
        self.running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self._unregister()
        if self._channel:
            await self._channel.close()
        logger.info(f"[{self.profile.name}] Stopped")

    async def _register(self) -> None:
        stub = device_pb2_grpc.DeviceRegistryStub(self._channel)
        initial_metrics = self.metrics_sim.tick()

        response = await stub.Register(
            device_pb2.RegisterRequest(
                name=self.profile.name,
                device_model=self.profile.device_model,
                os_version=self.profile.os_version,
                capabilities=common_pb2.DeviceCapabilities(
                    chip=self.profile.chip,
                    memory_bytes=self.profile.memory_bytes,
                    cpu_cores=self.profile.cpu_cores,
                    gpu_cores=self.profile.gpu_cores,
                    neural_engine_cores=self.profile.neural_engine_cores,
                    supported_frameworks=self.profile.supported_frameworks,
                ),
                initial_metrics=initial_metrics,
            )
        )
        self.device_id = response.device_id.value
        logger.info(f"[{self.profile.name}] Registered as {self.device_id}")

    async def _unregister(self) -> None:
        if not self.device_id or not self._channel:
            return
        try:
            stub = device_pb2_grpc.DeviceRegistryStub(self._channel)
            await stub.Unregister(
                device_pb2.UnregisterRequest(
                    device_id=common_pb2.DeviceId(value=self.device_id)
                )
            )
            logger.info(f"[{self.profile.name}] Unregistered")
        except grpc.aio.AioRpcError as e:
            logger.warning(f"[{self.profile.name}] Unregister failed: {e.code()}")

    async def _heartbeat_loop(self) -> None:
        stub = heartbeat_pb2_grpc.HeartbeatServiceStub(self._channel)
        queue: asyncio.Queue = asyncio.Queue()

        async def request_iterator():
            while self.running:
                msg = await queue.get()
                if msg is None:
                    return
                yield msg

        # Start the bidi stream
        stream = stub.Heartbeat(request_iterator())

        # Producer: push heartbeats at interval
        producer = asyncio.create_task(self._heartbeat_producer(queue))

        # Consumer: read responses
        try:
            async for response in stream:
                await self._handle_command(response)
        except grpc.aio.AioRpcError as e:
            if self.running:
                logger.error(f"[{self.profile.name}] Heartbeat stream error: {e.code()}")
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await queue.put(None)  # Signal producer to stop
            producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    async def _heartbeat_producer(self, queue: asyncio.Queue) -> None:
        try:
            while self.running:
                self._sequence += 1
                metrics = self.metrics_sim.tick()
                msg = heartbeat_pb2.HeartbeatRequest(
                    device_id=common_pb2.DeviceId(value=self.device_id),
                    metrics=metrics,
                    sequence=self._sequence,
                )
                await queue.put(msg)
                logger.debug(
                    f"[{self.profile.name}] Heartbeat #{self._sequence} "
                    f"cpu={metrics.cpu_usage:.1%} bat={metrics.battery.level:.0%}"
                )
                await asyncio.sleep(self.heartbeat_interval)
        except asyncio.CancelledError:
            pass

    async def _handle_command(self, response: heartbeat_pb2.HeartbeatResponse) -> None:
        cmd = response.command
        if cmd == heartbeat_pb2.HEARTBEAT_COMMAND_ACK:
            return
        elif cmd == heartbeat_pb2.HEARTBEAT_COMMAND_UPDATE_INTERVAL:
            new_interval = float(response.parameters.get("interval_seconds", "5"))
            logger.info(f"[{self.profile.name}] Interval updated to {new_interval}s")
            self.heartbeat_interval = new_interval
        elif cmd == heartbeat_pb2.HEARTBEAT_COMMAND_START_TRAINING:
            logger.info(f"[{self.profile.name}] Training started")
            self.metrics_sim.start_training()
        elif cmd == heartbeat_pb2.HEARTBEAT_COMMAND_STOP_TRAINING:
            logger.info(f"[{self.profile.name}] Training stopped")
            self.metrics_sim.stop_training()
        elif cmd == heartbeat_pb2.HEARTBEAT_COMMAND_SHUTDOWN:
            logger.info(f"[{self.profile.name}] Shutdown command received")
            self.running = False
