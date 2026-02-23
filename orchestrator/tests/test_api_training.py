"""Tests for the /api/v1/training REST routes."""

import uuid

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.repositories import TrainingJobRepository


class TestTrainingAPI:
    async def test_create_job(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/api/v1/training/jobs",
            json={"num_rounds": 10, "min_devices": 2, "learning_rate": 0.01},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["num_rounds"] == 10
        assert data["current_round"] == 0
        assert "id" in data

    async def test_create_job_validation(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/api/v1/training/jobs",
            json={"num_rounds": 0},
        )
        assert resp.status_code == 422

    async def test_list_jobs(self, client: httpx.AsyncClient):
        await client.post("/api/v1/training/jobs", json={"num_rounds": 5})
        await client.post("/api/v1/training/jobs", json={"num_rounds": 10})

        resp = await client.get("/api/v1/training/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_get_job(self, client: httpx.AsyncClient):
        create_resp = await client.post(
            "/api/v1/training/jobs", json={"num_rounds": 5}
        )
        job_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

    async def test_get_job_not_found(self, client: httpx.AsyncClient):
        resp = await client.get(f"/api/v1/training/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_stop_job(
        self, client: httpx.AsyncClient, fake_redis, app
    ):
        # Inject fake redis into the training module
        from orchestrator.api.routes import training as training_mod

        training_mod._redis = fake_redis

        create_resp = await client.post(
            "/api/v1/training/jobs", json={"num_rounds": 5}
        )
        job_id = create_resp.json()["id"]

        resp = await client.post(f"/api/v1/training/jobs/{job_id}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

        # Verify Redis stop flag was set
        stop_flag = await fake_redis.get(f"training:{job_id}:stop")
        assert stop_flag == b"1"

        # Cleanup
        training_mod._redis = None

    async def test_stop_completed_job(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = TrainingJobRepository(db_session)
        job = await repo.create(num_rounds=5)
        await repo.update(job.id, status="completed")

        resp = await client.post(f"/api/v1/training/jobs/{job.id}/stop")
        assert resp.status_code == 400

    async def test_retry_failed_job(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = TrainingJobRepository(db_session)
        job = await repo.create(num_rounds=10, min_devices=1, learning_rate=0.01)
        await repo.update(job.id, status="failed", current_round=5)

        resp = await client.post(f"/api/v1/training/jobs/{job.id}/retry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["job_id"] == str(job.id)
        assert data["resume_from_round"] == 6

        # Verify DB was updated
        updated = await repo.get(job.id)
        assert updated.status == "running"

    async def test_retry_non_failed_job_returns_400(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = TrainingJobRepository(db_session)
        job = await repo.create(num_rounds=5)
        # Job is in "pending" status by default

        resp = await client.post(f"/api/v1/training/jobs/{job.id}/retry")
        assert resp.status_code == 400
        assert "pending" in resp.json()["detail"]

    async def test_retry_nonexistent_job(self, client: httpx.AsyncClient):
        resp = await client.post(f"/api/v1/training/jobs/{uuid.uuid4()}/retry")
        assert resp.status_code == 404
