import asyncio
import base64
import json
import uuid
from datetime import datetime, timezone

import structlog
from redis.asyncio import Redis

from orchestrator.config import settings
from orchestrator.db.engine import async_session
from orchestrator.db.repositories import DeviceRepository, TrainingJobRepository
from orchestrator.services.fed_avg import (
    MODEL_SIZE,
    aggregate_gradients,
    apply_gradients,
    create_initial_model,
)
from orchestrator.services.heartbeat_monitor import HeartbeatMonitor

logger = structlog.get_logger()


class TrainingCoordinator:
    def __init__(self, redis: Redis, heartbeat_monitor: HeartbeatMonitor) -> None:
        self.redis = redis
        self.heartbeat_monitor = heartbeat_monitor

    async def start_job(self, job_id: str, num_rounds: int, learning_rate: float, min_devices: int) -> None:
        # Store initial global model in Redis
        initial_model = create_initial_model()
        encoded = base64.b64encode(initial_model).decode()
        await self.redis.set(f"model:{job_id}:global", encoded)

        meta = json.dumps({
            "model_id": job_id,
            "name": f"fedavg-{job_id[:8]}",
            "version": "0",
            "framework": "numpy",
            "size_bytes": len(initial_model),
        })
        await self.redis.set(f"model:{job_id}:meta", meta)

        logger.info("training_job_starting", job_id=job_id, rounds=num_rounds)

        asyncio.create_task(
            self._run_training_loop(job_id, num_rounds, learning_rate, min_devices)
        )

    async def _run_training_loop(
        self, job_id: str, num_rounds: int, learning_rate: float, min_devices: int
    ) -> None:
        try:
            all_round_metrics = []

            for round_num in range(1, num_rounds + 1):
                # Check for stop signal
                stop_flag = await self.redis.get(f"training:{job_id}:stop")
                if stop_flag:
                    logger.info("training_job_stopped", job_id=job_id, round=round_num)
                    async with async_session() as session:
                        repo = TrainingJobRepository(session)
                        await repo.update(uuid.UUID(job_id), status="stopped")
                    return

                # Update current round in DB
                async with async_session() as session:
                    repo = TrainingJobRepository(session)
                    await repo.update(uuid.UUID(job_id), current_round=round_num)

                # Get online devices
                async with async_session() as session:
                    device_repo = DeviceRepository(session)
                    devices = await device_repo.list_all(status="online")

                if len(devices) < min_devices:
                    logger.warning(
                        "not_enough_devices",
                        job_id=job_id,
                        round=round_num,
                        online=len(devices),
                        required=min_devices,
                    )
                    await asyncio.sleep(10)
                    continue

                # Send START_TRAINING command to all online devices
                for device in devices:
                    await self.heartbeat_monitor.queue_command(
                        str(device.id),
                        {
                            "type": "start_training",
                            "parameters": {
                                "job_id": job_id,
                                "model_id": job_id,
                                "round": str(round_num),
                            },
                        },
                    )

                logger.info(
                    "training_round_started",
                    job_id=job_id,
                    round=round_num,
                    devices=len(devices),
                )

                # Wait for gradients
                gradients_key = f"gradients:{job_id}:{round_num}"
                collected = await self._wait_for_gradients(
                    gradients_key, len(devices), timeout=settings.training_round_timeout_seconds
                )

                if not collected:
                    logger.warning("no_gradients_received", job_id=job_id, round=round_num)
                    continue

                # Aggregate
                gradient_data = []
                round_device_metrics = []
                for entry_raw in collected:
                    entry = json.loads(entry_raw)
                    grad_bytes = base64.b64decode(entry["gradients"])
                    gradient_data.append((grad_bytes, entry["num_samples"]))
                    round_device_metrics.append(entry.get("metrics", {}))

                averaged_grads = aggregate_gradients(gradient_data)

                # Apply to global model
                encoded_model = await self.redis.get(f"model:{job_id}:global")
                current_model = base64.b64decode(encoded_model)
                new_model = apply_gradients(current_model, averaged_grads, learning_rate)
                await self.redis.set(
                    f"model:{job_id}:global", base64.b64encode(new_model).decode()
                )

                # Compute round summary
                avg_loss = 0.0
                avg_accuracy = 0.0
                if round_device_metrics:
                    avg_loss = sum(m.get("loss", 0) for m in round_device_metrics) / len(round_device_metrics)
                    avg_accuracy = sum(m.get("accuracy", 0) for m in round_device_metrics) / len(round_device_metrics)

                round_info = {
                    "round": round_num,
                    "participants": len(collected),
                    "avg_loss": round(avg_loss, 4),
                    "avg_accuracy": round(avg_accuracy, 4),
                }
                all_round_metrics.append(round_info)

                # Update DB
                async with async_session() as session:
                    repo = TrainingJobRepository(session)
                    await repo.update(
                        uuid.UUID(job_id),
                        round_metrics={"rounds": all_round_metrics},
                    )

                logger.info(
                    "training_round_completed",
                    job_id=job_id,
                    round=round_num,
                    participants=len(collected),
                    avg_loss=round(avg_loss, 4),
                    avg_accuracy=round(avg_accuracy, 4),
                )

                # Clean up gradients for this round
                await self.redis.delete(gradients_key)

            # Job complete
            async with async_session() as session:
                repo = TrainingJobRepository(session)
                await repo.update(
                    uuid.UUID(job_id),
                    status="completed",
                    completed_at=datetime.now(timezone.utc),
                    round_metrics={"rounds": all_round_metrics},
                )

            logger.info("training_job_completed", job_id=job_id)

        except Exception:
            logger.exception("training_job_failed", job_id=job_id)
            async with async_session() as session:
                repo = TrainingJobRepository(session)
                await repo.update(uuid.UUID(job_id), status="failed")

    async def _wait_for_gradients(
        self, key: str, expected: int, timeout: int = 60
    ) -> list[str]:
        elapsed = 0
        poll_interval = 2
        while elapsed < timeout:
            count = await self.redis.llen(key)
            if count >= expected:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Collect whatever we have
        entries = await self.redis.lrange(key, 0, -1)
        return entries

    async def stop_job(self, job_id: str) -> None:
        await self.redis.set(f"training:{job_id}:stop", "1")
        logger.info("training_job_stop_requested", job_id=job_id)

    async def run(self) -> None:
        """Background loop that picks up pending training jobs."""
        logger.info("training_coordinator_started")
        while True:
            try:
                async with async_session() as session:
                    repo = TrainingJobRepository(session)
                    pending_jobs = await repo.list_all(status="pending")

                for job in pending_jobs:
                    async with async_session() as session:
                        repo = TrainingJobRepository(session)
                        await repo.update(job.id, status="running")
                    await self.start_job(
                        str(job.id),
                        job.num_rounds,
                        job.learning_rate,
                        job.min_devices,
                    )
            except Exception:
                logger.exception("training_coordinator_error")

            await asyncio.sleep(5)
