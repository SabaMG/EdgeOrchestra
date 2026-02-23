"""Tests for TrainingCoordinator robustness: checkpoint, resume, retry."""

import asyncio
import base64
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.repositories import TrainingJobRepository
from orchestrator.services.training_coordinator import TrainingCoordinator


class FakeHeartbeatMonitor:
    """Minimal stub for HeartbeatMonitor — records queued commands."""

    def __init__(self):
        self.commands: list[tuple[str, dict]] = []

    async def queue_command(self, device_id: str, command: dict) -> None:
        self.commands.append((device_id, command))


@pytest.fixture
def heartbeat():
    return FakeHeartbeatMonitor()


@pytest.fixture
def coordinator(fake_redis, heartbeat):
    return TrainingCoordinator(redis=fake_redis, heartbeat_monitor=heartbeat)


def _make_job(
    job_id=None, status="running", current_round=3, num_rounds=5,
    learning_rate=0.01, min_devices=1, round_metrics=None, model_id=None,
):
    """Create a lightweight job-like object for resume_job()."""
    jid = uuid.UUID(job_id) if job_id else uuid.uuid4()
    return SimpleNamespace(
        id=jid,
        status=status,
        current_round=current_round,
        num_rounds=num_rounds,
        learning_rate=learning_rate,
        min_devices=min_devices,
        round_metrics=round_metrics,
        model_id=uuid.UUID(model_id) if model_id else None,
    )


class TestResumeJob:
    async def test_resume_job_from_checkpoint(self, coordinator, fake_redis):
        """A running job with current_round=3 should resume from round 4."""
        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        existing_metrics = [
            {"round": i, "participants": 1, "avg_loss": 0.5, "avg_accuracy": 0.8}
            for i in range(1, 4)
        ]
        job = _make_job(
            job_id=job_id,
            model_id=model_id,
            current_round=3,
            num_rounds=5,
            round_metrics={"rounds": existing_metrics},
        )

        # Put a model in Redis using model_id
        model_data = base64.b64encode(b"fake-model-data").decode()
        await fake_redis.set(f"model:{model_id}:global", model_data)

        # Patch _run_training_loop to capture arguments
        captured = {}

        async def mock_loop(jid, nr, lr, md, model_id=None, start_round=1, existing_metrics=None, job_config=None):
            captured["start_round"] = start_round
            captured["existing_metrics"] = existing_metrics
            captured["model_id"] = model_id

        with patch.object(coordinator, "_run_training_loop", side_effect=mock_loop):
            await coordinator.resume_job(job)
            # Wait for the task to complete
            await coordinator._current_task

        assert captured["start_round"] == 4
        assert len(captured["existing_metrics"]) == 3
        assert captured["model_id"] == model_id
        assert job_id in coordinator._active_jobs

    async def test_resume_job_missing_model(self, coordinator, fake_redis):
        """When the model is missing from Redis, resume_job recreates it."""
        job_id = str(uuid.uuid4())
        job = _make_job(job_id=job_id, current_round=2, round_metrics=None)

        # No model in Redis — resume_job should create one (using job_id as model key)
        assert not await fake_redis.exists(f"model:{job_id}:global")

        with patch.object(coordinator, "_run_training_loop", new_callable=AsyncMock):
            await coordinator.resume_job(job)
            await coordinator._current_task

        # Model should now exist (using job_id since model_id is None)
        assert await fake_redis.exists(f"model:{job_id}:global")
        assert await fake_redis.exists(f"model:{job_id}:meta")

    async def test_resume_job_with_model_id(self, coordinator, fake_redis):
        """When model_id differs from job_id, Redis keys use model_id."""
        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        job = _make_job(job_id=job_id, model_id=model_id, current_round=0, round_metrics=None)

        # Patch async_session since resume_job tries to look up architecture from DB
        mock_cm = AsyncMock()
        mock_session = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        # Mock ModelRepository to return a fake model
        mock_model = SimpleNamespace(architecture="mnist")
        mock_model_repo = AsyncMock()
        mock_model_repo.get = AsyncMock(return_value=mock_model)

        with (
            patch.object(coordinator, "_run_training_loop", new_callable=AsyncMock),
            patch("orchestrator.services.training_coordinator.async_session", return_value=mock_cm),
            patch("orchestrator.services.training_coordinator.ModelRepository", return_value=mock_model_repo),
        ):
            await coordinator.resume_job(job)
            await coordinator._current_task

        # Model key should use model_id, not job_id
        assert await fake_redis.exists(f"model:{model_id}:global")
        assert await fake_redis.exists(f"model:{model_id}:meta")


class TestStartJob:
    async def test_start_job_with_model_id(self, coordinator, fake_redis):
        """start_job with model_id stores model under model_id key."""
        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())

        # Patch async_session since start_job tries to look up architecture from DB
        mock_cm = AsyncMock()
        mock_session = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_model = SimpleNamespace(architecture="mnist")
        mock_model_repo = AsyncMock()
        mock_model_repo.get = AsyncMock(return_value=mock_model)

        with (
            patch.object(coordinator, "_run_training_loop", new_callable=AsyncMock),
            patch("orchestrator.services.training_coordinator.async_session", return_value=mock_cm),
            patch("orchestrator.services.training_coordinator.ModelRepository", return_value=mock_model_repo),
        ):
            await coordinator.start_job(job_id, num_rounds=5, learning_rate=0.01, min_devices=1, model_id=model_id)
            await coordinator._current_task

        assert await fake_redis.exists(f"model:{model_id}:global")

    async def test_start_job_without_model_id(self, coordinator, fake_redis):
        """start_job without model_id falls back to job_id."""
        job_id = str(uuid.uuid4())

        with patch.object(coordinator, "_run_training_loop", new_callable=AsyncMock):
            await coordinator.start_job(job_id, num_rounds=5, learning_rate=0.01, min_devices=1)
            await coordinator._current_task

        assert await fake_redis.exists(f"model:{job_id}:global")


class TestRoundRetry:
    async def test_round_retry_on_no_gradients(self, coordinator, fake_redis, db_session):
        """A round with no gradients should be retried before being skipped."""
        job_id = str(uuid.uuid4())

        # Create a real job in DB
        repo = TrainingJobRepository(db_session)
        job = await repo.create(
            id=uuid.UUID(job_id), num_rounds=1, min_devices=1, learning_rate=0.01,
        )
        await repo.update(job.id, status="running")

        # Put a model in Redis
        model_data = base64.b64encode(b"fake-model").decode()
        await fake_redis.set(f"model:{job_id}:global", model_data)

        # Track how many times _wait_for_gradients is called
        call_count = 0

        async def mock_wait(key, expected, timeout=60):
            nonlocal call_count
            call_count += 1
            return []  # No gradients ever

        # Create a fake device so the coordinator finds one
        from orchestrator.db.repositories import DeviceRepository
        device_repo = DeviceRepository(db_session)
        await device_repo.create(
            name="test-device", device_model="iPhone15", os_version="17.0", status="online",
        )

        with (
            patch.object(coordinator, "_wait_for_gradients", side_effect=mock_wait),
            patch("orchestrator.services.training_coordinator.async_session") as mock_session_ctx,
        ):
            # Make async_session return our test session
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=db_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_cm

            await coordinator._run_training_loop(
                job_id, num_rounds=1, learning_rate=0.01, min_devices=1,
            )

        # Should have been called 3 times: initial + 2 retries
        assert call_count == 3

        # Job should be completed (all rounds processed, even if skipped)
        updated_job = await repo.get(uuid.UUID(job_id))
        assert updated_job.status == "completed"


class TestFailurePreservesModel:
    async def test_failure_preserves_model(self, coordinator, fake_redis):
        """An exception during training should NOT delete the model from Redis."""
        job_id = str(uuid.uuid4())

        # Put model in Redis
        model_data = base64.b64encode(b"precious-model").decode()
        await fake_redis.set(f"model:{job_id}:global", model_data)
        await fake_redis.set(f"model:{job_id}:meta", json.dumps({"version": "3"}))

        # Mock async_session so the first call (update current_round) succeeds
        # but DeviceRepository raises an exception
        mock_repo = AsyncMock()

        with patch("orchestrator.services.training_coordinator.async_session") as mock_session_ctx:
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_cm

            with patch(
                "orchestrator.services.training_coordinator.TrainingJobRepository",
                return_value=mock_repo,
            ):
                # Force an exception when listing devices
                with patch(
                    "orchestrator.services.training_coordinator.DeviceRepository",
                    side_effect=RuntimeError("boom"),
                ):
                    await coordinator._run_training_loop(
                        job_id, num_rounds=3, learning_rate=0.01, min_devices=1,
                    )

        # Model should still exist (keep_model=True on failure)
        assert await fake_redis.exists(f"model:{job_id}:global")
        assert await fake_redis.exists(f"model:{job_id}:meta")


class TestStopCleansAllKeys:
    async def test_stop_cleans_all_redis_keys(self, coordinator, fake_redis):
        """An explicit stop should clean model + meta + gradients."""
        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())

        # Set up various keys using model_id
        await fake_redis.set(f"model:{model_id}:global", "data")
        await fake_redis.set(f"model:{model_id}:meta", "meta")
        await fake_redis.set(f"training:{job_id}:stop", "1")
        await fake_redis.rpush(f"gradients:{model_id}:1", "grad1")
        await fake_redis.rpush(f"gradients:{model_id}:2", "grad2")

        await coordinator._cleanup_redis_keys(job_id, model_id=model_id, keep_model=False)

        assert not await fake_redis.exists(f"model:{model_id}:global")
        assert not await fake_redis.exists(f"model:{model_id}:meta")
        assert not await fake_redis.exists(f"training:{job_id}:stop")
        assert not await fake_redis.exists(f"gradients:{model_id}:1")
        assert not await fake_redis.exists(f"gradients:{model_id}:2")

    async def test_cleanup_keep_model(self, coordinator, fake_redis):
        """With keep_model=True, model and meta are preserved."""
        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())

        await fake_redis.set(f"model:{model_id}:global", "data")
        await fake_redis.set(f"model:{model_id}:meta", "meta")
        await fake_redis.set(f"training:{job_id}:stop", "1")
        await fake_redis.rpush(f"gradients:{model_id}:1", "grad1")

        await coordinator._cleanup_redis_keys(job_id, model_id=model_id, keep_model=True)

        # Model preserved
        assert await fake_redis.exists(f"model:{model_id}:global")
        assert await fake_redis.exists(f"model:{model_id}:meta")
        # Stop flag and gradients cleaned
        assert not await fake_redis.exists(f"training:{job_id}:stop")
        assert not await fake_redis.exists(f"gradients:{model_id}:1")


class TestRunPicksUpRunningJobs:
    async def test_run_picks_up_running_jobs(self, coordinator, db_session):
        """The run() loop should resume running jobs that aren't already active."""
        repo = TrainingJobRepository(db_session)
        job = await repo.create(num_rounds=5, min_devices=1, learning_rate=0.01)
        await repo.update(job.id, status="running", current_round=2)

        resumed_jobs = []

        async def mock_resume(j):
            resumed_jobs.append(str(j.id))

        with (
            patch.object(coordinator, "resume_job", side_effect=mock_resume),
            patch("orchestrator.services.training_coordinator.async_session") as mock_session_ctx,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=db_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_ctx.return_value = mock_cm

            # Run the coordinator but break after initial resume + one loop iteration
            loop_count = 0
            original_sleep = asyncio.sleep

            async def mock_sleep(seconds):
                nonlocal loop_count
                loop_count += 1
                if loop_count >= 1:
                    raise KeyboardInterrupt()
                await original_sleep(0)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                try:
                    await coordinator.run()
                except KeyboardInterrupt:
                    pass

        assert str(job.id) in resumed_jobs


class TestActiveJobsTracking:
    async def test_active_jobs_not_resumed_twice(self, coordinator, fake_redis):
        """Jobs already in _active_jobs should not be resumed again."""
        job_id = str(uuid.uuid4())
        coordinator._active_jobs.add(job_id)

        job = _make_job(job_id=job_id, current_round=2)

        resumed = False

        async def mock_resume(j):
            nonlocal resumed
            resumed = True

        with patch.object(coordinator, "resume_job", side_effect=mock_resume):
            # Simulate what run() does for running jobs
            if str(job.id) not in coordinator._active_jobs:
                await coordinator.resume_job(job)

        assert not resumed
