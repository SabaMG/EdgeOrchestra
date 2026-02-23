"""Tests for the /api/v1/models REST routes."""

import uuid

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.repositories import ModelRepository, TrainingJobRepository


class TestModelsAPI:
    async def test_create_model(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/api/v1/models",
            json={"name": "My MNIST", "architecture": "mnist"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My MNIST"
        assert data["architecture"] == "mnist"
        assert data["version"] == 0
        assert data["status"] == "initial"

    async def test_list_models(self, client: httpx.AsyncClient):
        await client.post("/api/v1/models", json={"name": "M1", "architecture": "mnist"})
        await client.post("/api/v1/models", json={"name": "M2", "architecture": "cifar10"})

        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_models_filter_architecture(self, client: httpx.AsyncClient):
        await client.post("/api/v1/models", json={"name": "M1", "architecture": "mnist"})
        await client.post("/api/v1/models", json={"name": "M2", "architecture": "cifar10"})

        resp = await client.get("/api/v1/models?architecture=mnist")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) == 1
        assert models[0]["architecture"] == "mnist"

    async def test_get_model(self, client: httpx.AsyncClient):
        create_resp = await client.post(
            "/api/v1/models", json={"name": "Test", "architecture": "mnist"}
        )
        model_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/models/{model_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == model_id

    async def test_get_model_not_found(self, client: httpx.AsyncClient):
        resp = await client.get(f"/api/v1/models/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_delete_model(self, client: httpx.AsyncClient):
        create_resp = await client.post(
            "/api/v1/models", json={"name": "ToDelete", "architecture": "mnist"}
        )
        model_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/v1/models/{model_id}")
        assert resp.status_code == 200

        # Verify it's gone
        resp = await client.get(f"/api/v1/models/{model_id}")
        assert resp.status_code == 404

    async def test_list_architectures(self, client: httpx.AsyncClient):
        resp = await client.get("/api/v1/models/architectures")
        assert resp.status_code == 200
        archs = resp.json()
        keys = [a["key"] for a in archs]
        assert "mnist" in keys
        assert "cifar10" in keys

    async def test_create_model_unknown_architecture(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/api/v1/models",
            json={"name": "Bad", "architecture": "unknown"},
        )
        assert resp.status_code == 400

    async def test_create_job_with_model_id(self, client: httpx.AsyncClient):
        # Create a model first
        model_resp = await client.post(
            "/api/v1/models", json={"name": "Job Model", "architecture": "cifar10"}
        )
        model_id = model_resp.json()["id"]

        # Create a job with that model
        job_resp = await client.post(
            "/api/v1/training/jobs",
            json={"num_rounds": 5, "model_id": model_id},
        )
        assert job_resp.status_code == 201
        data = job_resp.json()
        assert data["model_id"] == model_id
        assert data["model_name"] == "Job Model"
        assert data["architecture"] == "cifar10"

    async def test_create_job_without_model_id(self, client: httpx.AsyncClient):
        job_resp = await client.post(
            "/api/v1/training/jobs",
            json={"num_rounds": 5},
        )
        assert job_resp.status_code == 201
        data = job_resp.json()
        # Should have auto-created a MNIST model
        assert data["model_id"] is not None
        assert data["model_name"] == "MNIST (auto)"
        assert data["architecture"] == "mnist"
