FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY orchestrator/pyproject.toml orchestrator/pyproject.toml
COPY pyproject.toml pyproject.toml

RUN cd orchestrator && uv pip install --system -e ".[dev]"

COPY orchestrator/ orchestrator/
COPY protos/ protos/

WORKDIR /app/orchestrator

CMD ["python", "-m", "orchestrator.main"]
