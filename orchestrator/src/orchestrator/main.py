import asyncio
import logging
import signal

import structlog
import uvicorn
from redis.asyncio import Redis

from orchestrator.config import settings

_log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.log_format == "console"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(_log_level),
)

logger = structlog.get_logger()


async def main() -> None:
    logger.info(
        "edgeorchestra_starting",
        api_port=settings.api_port,
        grpc_port=settings.grpc_port,
    )

    # Import generated protobuf modules
    from orchestrator.generated import (
        device_pb2,
        device_pb2_grpc,
        heartbeat_pb2,
        heartbeat_pb2_grpc,
        model_pb2,
        model_pb2_grpc,
    )

    # Redis
    redis = Redis.from_url(settings.redis_url, decode_responses=True)

    # Services
    from orchestrator.services.heartbeat_monitor import HeartbeatMonitor

    heartbeat_monitor = HeartbeatMonitor(redis)

    # gRPC services
    from orchestrator.grpc_server.device_service import DeviceRegistryServicer
    from orchestrator.grpc_server.heartbeat_service import HeartbeatServiceServicer
    from orchestrator.grpc_server.model_service import ModelServiceServicer
    from orchestrator.grpc_server.server import create_grpc_server

    device_service = DeviceRegistryServicer()
    heartbeat_service = HeartbeatServiceServicer(heartbeat_monitor)
    model_service = ModelServiceServicer(redis)

    # Training coordinator
    from orchestrator.services.training_coordinator import TrainingCoordinator

    training_coordinator = TrainingCoordinator(redis, heartbeat_monitor)

    grpc_server = await create_grpc_server(
        device_service,
        heartbeat_service,
        model_service,
        device_pb2,
        heartbeat_pb2,
        model_pb2,
        device_pb2_grpc,
        heartbeat_pb2_grpc,
        model_pb2_grpc,
    )

    # mDNS
    from orchestrator.discovery.mdns import MDNSDiscovery

    mdns = MDNSDiscovery()

    # FastAPI
    from orchestrator.api.app import create_app

    app = create_app()

    # Share Redis with training routes
    from orchestrator.api.routes.training import set_redis

    set_redis(redis)
    uvicorn_config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    # Shutdown event
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Start all services
    await grpc_server.start()
    logger.info("grpc_server_started", port=settings.grpc_port)

    await mdns.register()

    tasks = [
        asyncio.create_task(uvicorn_server.serve(), name="uvicorn"),
        asyncio.create_task(heartbeat_monitor.run_stale_device_checker(), name="heartbeat"),
        asyncio.create_task(training_coordinator.run(), name="training_coordinator"),
        asyncio.create_task(shutdown_event.wait(), name="shutdown"),
    ]

    # Wait for shutdown signal
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Graceful shutdown
    logger.info("shutting_down")

    uvicorn_server.should_exit = True
    await mdns.unregister()
    await grpc_server.stop(grace=5)
    await redis.aclose()

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.info("shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
