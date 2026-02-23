import asyncio
import base64
import json
import math
import time
import uuid
from datetime import datetime, timezone

import structlog
from redis.asyncio import Redis

from orchestrator.observability.metrics import (
    TRAINING_JOBS_ACTIVE,
    TRAINING_ROUND_DURATION,
    TRAINING_ROUNDS_TOTAL,
)

from orchestrator.config import settings
from orchestrator.db.engine import async_session
from orchestrator.db.repositories import DeviceRepository, ModelRepository, TrainingJobRepository
from orchestrator.services.coreml_model import (
    create_updatable_mlmodel,
    create_updatable_mlmodel_for_architecture,
    extract_weights,
    inject_weights,
    set_learning_rate,
)
from orchestrator.services.fed_avg import (
    aggregate_gradients,
    apply_gradients,
)
from orchestrator.services.device_scheduler import SchedulerConfig, select_devices
from orchestrator.services.heartbeat_monitor import HeartbeatMonitor
from orchestrator.services.model_registry import ARCHITECTURES, get_architecture
from orchestrator.services.server_evaluator import ServerEvaluator

logger = structlog.get_logger()


class TrainingCoordinator:
    def __init__(self, redis: Redis, heartbeat_monitor: HeartbeatMonitor) -> None:
        self.redis = redis
        self.heartbeat_monitor = heartbeat_monitor
        self._active_jobs: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}

    async def start_job(
        self, job_id: str, num_rounds: int, learning_rate: float, min_devices: int,
        model_id: str | None = None, job_config: dict | None = None,
    ) -> None:
        # Determine model_id: use provided one or default to job_id (backward compat)
        effective_model_id = model_id or job_id

        model_key = f"model:{effective_model_id}:global"

        # If model already exists in Redis (from a previous job or explicit creation), reuse it
        if not await self.redis.exists(model_key):
            # Determine architecture from DB model if model_id was provided
            arch = ARCHITECTURES["mnist"]  # default
            if model_id:
                async with async_session() as session:
                    model_repo = ModelRepository(session)
                    db_model = await model_repo.get(uuid.UUID(model_id))
                    if db_model:
                        arch = get_architecture(db_model.architecture)

            initial_model = create_updatable_mlmodel_for_architecture(arch)
            encoded = base64.b64encode(initial_model).decode()
            await self.redis.set(model_key, encoded)

            meta = json.dumps({
                "model_id": effective_model_id,
                "name": f"fedavg-{effective_model_id[:8]}",
                "version": "0",
                "framework": "coreml",
                "size_bytes": len(initial_model),
            })
            await self.redis.set(f"model:{effective_model_id}:meta", meta)

        logger.info("training_job_starting", job_id=job_id, model_id=effective_model_id, rounds=num_rounds)

        self._active_jobs.add(job_id)
        TRAINING_JOBS_ACTIVE.inc()
        self._tasks[job_id] = asyncio.create_task(
            self._run_training_loop(
                job_id, num_rounds, learning_rate, min_devices,
                model_id=effective_model_id, job_config=job_config,
            )
        )

    async def resume_job(self, job: object) -> None:
        """Resume a running job from its last checkpoint (current_round in DB)."""
        job_id = str(job.id)
        model_id = str(job.model_id) if getattr(job, "model_id", None) else job_id

        # Ensure model exists in Redis (may have been lost in crash)
        model_key = f"model:{model_id}:global"
        if not await self.redis.exists(model_key):
            logger.warning("model_missing_in_redis_recreating", job_id=job_id, model_id=model_id)

            # Determine architecture from DB model
            arch = ARCHITECTURES["mnist"]
            if model_id != job_id:
                async with async_session() as session:
                    model_repo = ModelRepository(session)
                    db_model = await model_repo.get(uuid.UUID(model_id))
                    if db_model:
                        arch = get_architecture(db_model.architecture)

            initial_model = create_updatable_mlmodel_for_architecture(arch)
            encoded = base64.b64encode(initial_model).decode()
            await self.redis.set(model_key, encoded)
            meta = json.dumps({
                "model_id": model_id,
                "name": f"fedavg-{model_id[:8]}",
                "version": "0",
                "framework": "coreml",
                "size_bytes": len(initial_model),
            })
            await self.redis.set(f"model:{model_id}:meta", meta)

        # Restore metrics from DB
        existing_metrics = []
        if job.round_metrics and "rounds" in job.round_metrics:
            existing_metrics = job.round_metrics["rounds"]

        # Resume from next round after the last completed one
        resume_from = job.current_round + 1

        logger.info(
            "resuming_job_from_checkpoint",
            job_id=job_id,
            model_id=model_id,
            resume_from_round=resume_from,
            total_rounds=job.num_rounds,
        )

        self._active_jobs.add(job_id)
        TRAINING_JOBS_ACTIVE.inc()
        job_config = getattr(job, "config", None)
        self._tasks[job_id] = asyncio.create_task(
            self._run_training_loop(
                job_id, job.num_rounds, job.learning_rate, job.min_devices,
                model_id=model_id,
                start_round=resume_from,
                existing_metrics=existing_metrics,
                job_config=job_config,
            )
        )

    async def _run_training_loop(
        self, job_id: str, num_rounds: int, learning_rate: float, min_devices: int,
        model_id: str | None = None,
        start_round: int = 1,
        existing_metrics: list[dict] | None = None,
        job_config: dict | None = None,
    ) -> None:
        effective_model_id = model_id or job_id
        max_device_wait_retries = 30  # Max retries waiting for devices per round
        max_round_retries = 2  # Retry a round up to 2 times before skipping
        dispatched_device_ids: list[str] = []

        # Resolve architecture for this model
        arch_key = "mnist"
        if effective_model_id != job_id:
            try:
                async with async_session() as session:
                    model_repo = ModelRepository(session)
                    db_model = await model_repo.get(uuid.UUID(effective_model_id))
                    if db_model:
                        arch_key = db_model.architecture
            except Exception:
                pass

        try:
            all_round_metrics = list(existing_metrics) if existing_metrics else []

            for round_num in range(start_round, num_rounds + 1):
                round_start = time.perf_counter()
                # Check for stop signal
                stop_flag = await self.redis.get(f"training:{job_id}:stop")
                if stop_flag:
                    logger.info("training_job_stopped", job_id=job_id, round=round_num)
                    async with async_session() as session:
                        repo = TrainingJobRepository(session)
                        await repo.update(uuid.UUID(job_id), status="stopped")
                    await self._cleanup_redis_keys(job_id, model_id=effective_model_id)
                    return

                # Update current round in DB
                async with async_session() as session:
                    repo = TrainingJobRepository(session)
                    await repo.update(uuid.UUID(job_id), current_round=round_num)

                # Wait for enough online devices with exponential backoff
                sched_cfg = SchedulerConfig.from_job_config(job_config)
                devices = []
                for attempt in range(max_device_wait_retries):
                    # Check stop signal during wait
                    stop_flag = await self.redis.get(f"training:{job_id}:stop")
                    if stop_flag:
                        logger.info("training_job_stopped_while_waiting", job_id=job_id)
                        async with async_session() as session:
                            repo = TrainingJobRepository(session)
                            await repo.update(uuid.UUID(job_id), status="stopped")
                        await self._cleanup_redis_keys(job_id, model_id=effective_model_id)
                        return

                    async with async_session() as session:
                        device_repo = DeviceRepository(session)
                        all_online = await device_repo.list_all(status="online")

                    selected = select_devices(all_online, sched_cfg, min_devices)
                    if selected is not None:
                        devices = selected
                        break

                    wait_time = min(10 * (2 ** min(attempt, 4)), 120)  # 10s, 20s, 40s, ... max 120s
                    logger.warning(
                        "not_enough_devices",
                        job_id=job_id,
                        round=round_num,
                        online=len(all_online),
                        required=min_devices,
                        retry=attempt + 1,
                        wait_seconds=wait_time,
                        scheduler_enabled=sched_cfg.enabled,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "device_wait_exhausted",
                        job_id=job_id,
                        round=round_num,
                        online=len(devices),
                        required=min_devices,
                    )
                    async with async_session() as session:
                        repo = TrainingJobRepository(session)
                        await repo.update(uuid.UUID(job_id), status="failed")
                    await self._cleanup_redis_keys(job_id, model_id=effective_model_id, keep_model=True)
                    return

                # Mark selected devices as "training"
                dispatched_device_ids = [str(d.id) for d in devices]
                async with async_session() as session:
                    device_repo = DeviceRepository(session)
                    for device in devices:
                        await device_repo.update(device.id, status="training")

                # Cosine decay learning rate schedule
                lr_min = learning_rate * 0.01
                lr_max = learning_rate
                cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * round_num / num_rounds))
                encoded_model = await self.redis.get(f"model:{effective_model_id}:global")
                current_model_bytes = base64.b64decode(encoded_model)
                updated_model_bytes = set_learning_rate(current_model_bytes, cosine_lr)
                await self.redis.set(
                    f"model:{effective_model_id}:global", base64.b64encode(updated_model_bytes).decode()
                )

                # Round retry loop
                round_completed = False
                for retry in range(max_round_retries + 1):
                    # Send START_TRAINING command to selected devices
                    for i, device in enumerate(devices):
                        await self.heartbeat_monitor.queue_command(
                            str(device.id),
                            {
                                "type": "start_training",
                                "parameters": {
                                    "job_id": job_id,
                                    "model_id": effective_model_id,
                                    "round": str(round_num),
                                    "partition_index": str(i),
                                    "partition_total": str(len(devices)),
                                    "architecture": arch_key,
                                },
                            },
                        )

                    logger.info(
                        "training_round_started",
                        job_id=job_id,
                        round=round_num,
                        devices=len(devices),
                        retry=retry if retry > 0 else None,
                    )

                    # Wait for gradients
                    gradients_key = f"gradients:{effective_model_id}:{round_num}"
                    collected = await self._wait_for_gradients(
                        gradients_key, len(devices), timeout=settings.training_round_timeout_seconds
                    )

                    if collected:
                        round_completed = True
                        break

                    if retry < max_round_retries:
                        logger.warning(
                            "round_no_gradients_retrying",
                            job_id=job_id,
                            round=round_num,
                            retry=retry + 1,
                        )
                        await self.redis.delete(gradients_key)
                        continue

                    # All retries exhausted -- skip this round
                    logger.error(
                        "round_skipped_after_retries",
                        job_id=job_id,
                        round=round_num,
                        retries=max_round_retries,
                    )
                    await self.redis.delete(gradients_key)
                    all_round_metrics.append({
                        "round": round_num,
                        "participants": 0,
                        "avg_loss": None,
                        "avg_accuracy": None,
                        "skipped": True,
                        "retries": max_round_retries,
                    })
                    async with async_session() as session:
                        repo = TrainingJobRepository(session)
                        await repo.update(
                            uuid.UUID(job_id),
                            round_metrics={"rounds": all_round_metrics},
                        )

                if not round_completed:
                    await self._restore_device_statuses(dispatched_device_ids)
                    continue

                # Aggregate weight deltas
                gradients_key = f"gradients:{effective_model_id}:{round_num}"
                gradient_data = []
                round_device_metrics = []
                for entry_raw in collected:
                    entry = json.loads(entry_raw)
                    grad_bytes = base64.b64decode(entry["gradients"])
                    num_samples = entry.get("num_samples", 0)
                    if num_samples <= 0 or not grad_bytes:
                        logger.warning(
                            "skipping_invalid_gradient",
                            job_id=job_id,
                            device_id=entry.get("device_id"),
                        )
                        continue
                    gradient_data.append((grad_bytes, num_samples))
                    device_metric = entry.get("metrics", {})
                    device_metric["device_id"] = entry.get("device_id", "unknown")
                    device_metric["num_samples"] = num_samples
                    round_device_metrics.append(device_metric)

                if not gradient_data:
                    logger.error("all_gradients_invalid", job_id=job_id, round=round_num)
                    await self.redis.delete(gradients_key)
                    continue

                averaged_grads = aggregate_gradients(gradient_data)

                # Apply to global model: extract weights, apply deltas, rebuild .mlmodel
                encoded_model = await self.redis.get(f"model:{effective_model_id}:global")
                current_model_bytes = base64.b64decode(encoded_model)
                current_weights = extract_weights(current_model_bytes)
                new_weights = apply_gradients(current_weights, averaged_grads, learning_rate)
                new_model_bytes = inject_weights(current_model_bytes, new_weights)

                await self.redis.set(
                    f"model:{effective_model_id}:global", base64.b64encode(new_model_bytes).decode()
                )

                # Update model metadata version
                meta_raw = await self.redis.get(f"model:{effective_model_id}:meta")
                if meta_raw:
                    meta = json.loads(meta_raw)
                    meta["version"] = str(round_num)
                    meta["size_bytes"] = len(new_model_bytes)
                    await self.redis.set(f"model:{effective_model_id}:meta", json.dumps(meta))

                # Update model version in DB if model_id differs from job_id
                if effective_model_id != job_id:
                    async with async_session() as session:
                        model_repo = ModelRepository(session)
                        await model_repo.update(uuid.UUID(effective_model_id), version=round_num)

                # Server-side evaluation on held-out test set
                evaluator = ServerEvaluator.get_instance()
                eval_loss, eval_accuracy = evaluator.evaluate(new_weights, architecture=arch_key)

                round_info = {
                    "round": round_num,
                    "participants": len(gradient_data),
                    "dispatched": len(devices),
                    "avg_loss": round(eval_loss, 4),
                    "avg_accuracy": round(eval_accuracy, 4),
                    "device_metrics": round_device_metrics,
                }
                all_round_metrics.append(round_info)

                # Update DB
                async with async_session() as session:
                    repo = TrainingJobRepository(session)
                    await repo.update(
                        uuid.UUID(job_id),
                        round_metrics={"rounds": all_round_metrics},
                    )

                # Store latest metrics in Redis for heartbeat responses
                await self.redis.set(
                    "training:latest_metrics",
                    json.dumps({
                        "server_accuracy": round(eval_accuracy, 4),
                        "server_loss": round(eval_loss, 4),
                        "round": round_num,
                        "job_id": job_id,
                    }),
                )

                TRAINING_ROUNDS_TOTAL.inc()
                TRAINING_ROUND_DURATION.observe(time.perf_counter() - round_start)

                logger.info(
                    "training_round_completed",
                    job_id=job_id,
                    round=round_num,
                    participants=len(gradient_data),
                    avg_loss=round(eval_loss, 4),
                    avg_accuracy=round(eval_accuracy, 4),
                )

                # Restore device statuses and clean up gradients for this round
                await self._restore_device_statuses(dispatched_device_ids)
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
                # Update model status to trained
                if effective_model_id != job_id:
                    model_repo = ModelRepository(session)
                    await model_repo.update(uuid.UUID(effective_model_id), status="trained")

            logger.info("training_job_completed", job_id=job_id)

        except Exception:
            logger.exception("training_job_failed", job_id=job_id)
            async with async_session() as session:
                repo = TrainingJobRepository(session)
                await repo.update(uuid.UUID(job_id), status="failed")
            # Only clean stop flag, preserve model for potential resume
            await self._cleanup_redis_keys(job_id, model_id=effective_model_id, keep_model=True)
        finally:
            await self._restore_device_statuses(dispatched_device_ids)
            self._active_jobs.discard(job_id)
            self._tasks.pop(job_id, None)
            TRAINING_JOBS_ACTIVE.dec()

    async def _restore_device_statuses(self, device_ids: list[str]) -> None:
        if not device_ids:
            return
        try:
            async with async_session() as session:
                repo = DeviceRepository(session)
                for did in device_ids:
                    current = await repo.get(uuid.UUID(did))
                    if current and current.status == "training":
                        await repo.update(current.id, status="online")
        except Exception:
            logger.exception("restore_device_statuses_failed")

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

    async def _cleanup_redis_keys(
        self, job_id: str, model_id: str | None = None, keep_model: bool = False,
    ) -> None:
        """Clean up Redis state for a job that stopped or failed."""
        effective_model_id = model_id or job_id
        try:
            await self.redis.delete(f"training:{job_id}:stop")
            if not keep_model:
                await self.redis.delete(f"model:{effective_model_id}:global")
                await self.redis.delete(f"model:{effective_model_id}:meta")
            # Always clean up any leftover gradient keys
            cursor = b"0"
            while cursor:
                cursor, keys = await self.redis.scan(
                    cursor=cursor, match=f"gradients:{effective_model_id}:*", count=100
                )
                if keys:
                    await self.redis.delete(*keys)
        except Exception:
            logger.exception("cleanup_redis_failed", job_id=job_id)

    async def stop_job(self, job_id: str) -> None:
        await self.redis.set(f"training:{job_id}:stop", "1")
        logger.info("training_job_stop_requested", job_id=job_id)

    async def run(self) -> None:
        """Background loop that picks up pending training jobs."""
        logger.info("training_coordinator_started")

        # Resume any jobs that were running before a crash/restart
        try:
            async with async_session() as session:
                repo = TrainingJobRepository(session)
                running_jobs = await repo.list_all(status="running")

            for job in running_jobs:
                if str(job.id) not in self._active_jobs:
                    logger.info("resuming_interrupted_job", job_id=str(job.id), round=job.current_round)
                    await self.resume_job(job)
        except Exception:
            logger.exception("resume_running_jobs_error")

        while True:
            try:
                async with async_session() as session:
                    repo = TrainingJobRepository(session)
                    pending_jobs = await repo.list_all(status="pending")
                    running_jobs = await repo.list_all(status="running")

                for job in pending_jobs:
                    async with async_session() as session:
                        repo = TrainingJobRepository(session)
                        await repo.update(job.id, status="running")
                    model_id = str(job.model_id) if getattr(job, "model_id", None) else None
                    await self.start_job(
                        str(job.id),
                        job.num_rounds,
                        job.learning_rate,
                        job.min_devices,
                        model_id=model_id,
                        job_config=getattr(job, "config", None),
                    )

                # Pick up running jobs not yet tracked (e.g. from retry API)
                for job in running_jobs:
                    if str(job.id) not in self._active_jobs:
                        logger.info("resuming_interrupted_job", job_id=str(job.id), round=job.current_round)
                        await self.resume_job(job)

            except Exception:
                logger.exception("training_coordinator_error")

            await asyncio.sleep(5)
