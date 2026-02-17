from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.engine import get_session
from orchestrator.db.repositories import DeviceRepository, TrainingJobRepository

_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(_dir / "templates"))

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _time_ago(dt: datetime | None) -> str:
    if not dt:
        return "-"
    now = datetime.now(timezone.utc)
    dt_aware = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    delta = (now - dt_aware).total_seconds()
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


templates.env.globals["time_ago"] = _time_ago


# --- Pages ---

@router.get("", response_class=HTMLResponse)
async def dashboard_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Partials ---

@router.get("/partials/health", response_class=HTMLResponse)
async def health_partial(request: Request):
    return templates.TemplateResponse("partials/health.html", {"request": request, "healthy": True})


@router.get("/partials/devices", response_class=HTMLResponse)
async def devices_partial(request: Request, session: AsyncSession = Depends(get_session)):
    repo = DeviceRepository(session)
    devices = await repo.list_all()
    return templates.TemplateResponse(
        "partials/devices_table.html", {"request": request, "devices": devices}
    )


@router.get("/partials/jobs", response_class=HTMLResponse)
async def jobs_partial(request: Request, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    jobs = await repo.list_all()
    return templates.TemplateResponse(
        "partials/jobs_table.html", {"request": request, "jobs": jobs}
    )


@router.get("/partials/job/{job_id}", response_class=HTMLResponse)
async def job_detail_partial(request: Request, job_id: str, session: AsyncSession = Depends(get_session)):
    import uuid

    repo = TrainingJobRepository(session)
    job = await repo.get(uuid.UUID(job_id))
    if not job:
        return HTMLResponse("<p>Job not found.</p>", status_code=404)

    rounds = []
    if job.round_metrics and "rounds" in job.round_metrics:
        rounds = job.round_metrics["rounds"]

    return templates.TemplateResponse(
        "partials/job_detail.html", {"request": request, "job": job, "rounds": rounds}
    )


@router.get("/partials/job/{job_id}/chart-data")
async def job_chart_data(job_id: str, session: AsyncSession = Depends(get_session)):
    import uuid

    repo = TrainingJobRepository(session)
    job = await repo.get(uuid.UUID(job_id))
    if not job or not job.round_metrics:
        return {"labels": [], "loss": [], "accuracy": []}

    rounds = job.round_metrics.get("rounds", [])
    return {
        "labels": [f"R{r['round']}" for r in rounds],
        "loss": [r.get("avg_loss", 0) for r in rounds],
        "accuracy": [r.get("avg_accuracy", 0) for r in rounds],
    }


# --- Actions ---

@router.post("/actions/jobs/create", response_class=HTMLResponse)
async def create_job_action(
    request: Request,
    num_rounds: int = Form(5),
    min_devices: int = Form(1),
    learning_rate: float = Form(0.01),
    session: AsyncSession = Depends(get_session),
):
    repo = TrainingJobRepository(session)
    await repo.create(
        num_rounds=num_rounds,
        min_devices=min_devices,
        learning_rate=learning_rate,
    )
    jobs = await repo.list_all()
    return templates.TemplateResponse(
        "partials/jobs_table.html", {"request": request, "jobs": jobs}
    )


@router.post("/actions/jobs/{job_id}/stop", response_class=HTMLResponse)
async def stop_job_action(request: Request, job_id: str, session: AsyncSession = Depends(get_session)):
    import uuid

    from orchestrator.api.routes.training import _redis

    repo = TrainingJobRepository(session)
    job = await repo.update(uuid.UUID(job_id), status="stopped")
    if not job:
        return HTMLResponse("<tr><td colspan='5'>Job not found</td></tr>", status_code=404)

    if _redis:
        await _redis.set(f"training:{job_id}:stop", "1")

    return templates.TemplateResponse(
        "partials/job_row.html", {"request": request, "job": job}
    )


def get_static_files_app():
    return StaticFiles(directory=str(_dir / "static"))
