from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from orchestrator.api.routes import devices, health, training
from orchestrator.dashboard.router import get_static_files_app, router as dashboard_router
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
    app.include_router(training.router)
    app.include_router(dashboard_router)
    app.mount("/dashboard/static", get_static_files_app(), name="dashboard-static")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/dashboard")

    return app
