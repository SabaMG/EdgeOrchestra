import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.api.routes import training as _training_module
from orchestrator.db.engine import get_session
from orchestrator.db.repositories import DeviceRepository, ModelRepository, TrainingJobRepository

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


async def _attach_model_info(jobs: list, model_repo: ModelRepository) -> None:
    """Attach model_name and architecture to job objects for display."""
    for job in jobs:
        if job.model_id:
            model = await model_repo.get(job.model_id)
            job.model_name = model.name if model else None
            job.architecture = model.architecture if model else None
        else:
            job.model_name = None
            job.architecture = None


def _get_redis():
    """Get the live Redis reference from the training module."""
    return _training_module._redis


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
    all_devices = await repo.list_all()
    # Show online and training devices (exclude offline/error)
    devices = [d for d in all_devices if d.status in ("online", "training")]
    return templates.TemplateResponse(
        "partials/devices_table.html", {"request": request, "devices": devices}
    )


@router.get("/partials/jobs", response_class=HTMLResponse)
async def jobs_partial(request: Request, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    model_repo = ModelRepository(session)
    all_jobs = await repo.list_all()
    shown = all_jobs[:10]
    await _attach_model_info(shown, model_repo)
    return templates.TemplateResponse(
        "partials/jobs_table.html",
        {"request": request, "jobs": shown, "total_jobs": len(all_jobs)},
    )


@router.get("/partials/model-options", response_class=HTMLResponse)
async def model_options_partial(request: Request, session: AsyncSession = Depends(get_session)):
    model_repo = ModelRepository(session)
    models = await model_repo.list_all()
    options = "".join(
        f'<option value="{m.id}">{m.name} ({m.architecture})</option>' for m in models
    )
    if not options:
        return HTMLResponse('<label>Existing Model<select name="model_id"><option value="">No models available</option></select></label>')
    return HTMLResponse(f'<label>Existing Model<select name="model_id">{options}</select></label>')


@router.get("/partials/job/{job_id}", response_class=HTMLResponse)
async def job_detail_partial(request: Request, job_id: str, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    job = await repo.get(uuid.UUID(job_id))
    if not job:
        return HTMLResponse("<p>Job not found.</p>", status_code=404)

    model_repo = ModelRepository(session)
    await _attach_model_info([job], model_repo)

    rounds = []
    if job.round_metrics and "rounds" in job.round_metrics:
        rounds = job.round_metrics["rounds"]

    return templates.TemplateResponse(
        "partials/job_detail.html", {"request": request, "job": job, "rounds": rounds}
    )


@router.get("/partials/job/{job_id}/info", response_class=HTMLResponse)
async def job_info_partial(request: Request, job_id: str, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    job = await repo.get(uuid.UUID(job_id))
    if not job:
        return HTMLResponse("<p>Job not found.</p>", status_code=404)

    model_repo = ModelRepository(session)
    await _attach_model_info([job], model_repo)

    return templates.TemplateResponse(
        "partials/job_info.html", {"request": request, "job": job}
    )


@router.get("/partials/job/{job_id}/rounds", response_class=HTMLResponse)
async def job_rounds_partial(request: Request, job_id: str, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    job = await repo.get(uuid.UUID(job_id))
    if not job:
        return HTMLResponse("", status_code=404)

    rounds = []
    if job.round_metrics and "rounds" in job.round_metrics:
        rounds = job.round_metrics["rounds"]

    return templates.TemplateResponse(
        "partials/job_rounds.html", {"request": request, "rounds": rounds}
    )


@router.get("/partials/job/{job_id}/chart-data")
async def job_chart_data(job_id: str, session: AsyncSession = Depends(get_session)):
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
    model_source: str = Form("new_mnist"),
    model_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    repo = TrainingJobRepository(session)
    model_repo = ModelRepository(session)

    if model_source == "existing" and model_id:
        resolved_model_id = uuid.UUID(model_id)
    else:
        # Determine architecture from model_source
        arch = "mnist"
        if model_source == "new_cifar10":
            arch = "cifar10"
        model = await model_repo.create(
            name=f"{arch.upper()} (auto)",
            architecture=arch,
        )
        resolved_model_id = model.id

    await repo.create(
        num_rounds=num_rounds,
        min_devices=min_devices,
        learning_rate=learning_rate,
        model_id=resolved_model_id,
    )
    all_jobs = await repo.list_all()
    shown = all_jobs[:10]
    await _attach_model_info(shown, model_repo)
    return templates.TemplateResponse(
        "partials/jobs_table.html", {"request": request, "jobs": shown, "total_jobs": len(all_jobs)}
    )


@router.post("/actions/jobs/{job_id}/stop", response_class=HTMLResponse)
async def stop_job_action(request: Request, job_id: str, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    model_repo = ModelRepository(session)
    job = await repo.update(uuid.UUID(job_id), status="stopped")
    if not job:
        return HTMLResponse("<p>Job not found</p>", status_code=404)

    redis = _get_redis()
    if redis:
        await redis.set(f"training:{job_id}:stop", "1")

    # Re-render the full jobs list
    all_jobs = await repo.list_all()
    shown = all_jobs[:10]
    await _attach_model_info(shown, model_repo)
    return templates.TemplateResponse(
        "partials/jobs_table.html",
        {"request": request, "jobs": shown, "total_jobs": len(all_jobs)},
    )


@router.post("/actions/jobs/clear", response_class=HTMLResponse)
async def clear_finished_jobs(request: Request, session: AsyncSession = Depends(get_session)):
    repo = TrainingJobRepository(session)
    model_repo = ModelRepository(session)
    redis = _get_redis()

    all_jobs = await repo.list_all()
    terminal = ("completed", "stopped", "failed")
    model_ids_to_delete: list[uuid.UUID] = []
    for job in all_jobs:
        if job.status in terminal:
            # Clean up Redis data
            if redis:
                if job.model_id:
                    await redis.delete(f"model:{job.model_id}:global")
                    await redis.delete(f"model:{job.model_id}:meta")
                await redis.delete(f"training:{job.id}:stop")
            if job.model_id:
                model_ids_to_delete.append(job.model_id)
            # Delete the job first (it holds the FK reference)
            await repo.delete(job.id)
    # Now clean up orphaned models (no more FK references)
    for mid in model_ids_to_delete:
        await model_repo.delete(mid)

    remaining = await repo.list_all()
    shown = remaining[:10]
    await _attach_model_info(shown, model_repo)
    return templates.TemplateResponse(
        "partials/jobs_table.html",
        {"request": request, "jobs": shown, "total_jobs": len(remaining)},
    )


def get_static_files_app():
    return StaticFiles(directory=str(_dir / "static"))
