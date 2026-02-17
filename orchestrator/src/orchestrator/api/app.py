from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from orchestrator.api.routes import devices, health
from orchestrator.db.engine import engine

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI starting up")
    yield
    await engine.dispose()
    logger.info("FastAPI shut down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EdgeOrchestra",
        version="0.1.0",
        description="Federated Learning Edge Orchestrator",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(devices.router)
    return app
