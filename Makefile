.PHONY: proto-gen dev-up dev-down lint test migrate validate

PROTO_DIR := protos
OUT_DIR := orchestrator/src/orchestrator/generated

proto-gen:
	@rm -f $(OUT_DIR)/*_pb2*.py $(OUT_DIR)/*_pb2*.pyi
	python -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(OUT_DIR) \
		--grpc_python_out=$(OUT_DIR) \
		--pyi_out=$(OUT_DIR) \
		$(PROTO_DIR)/edgeorchestra/v1/*.proto
	@# Move generated files from nested package dir to flat generated/
	@mv $(OUT_DIR)/edgeorchestra/v1/*_pb2*.py $(OUT_DIR)/
	@mv $(OUT_DIR)/edgeorchestra/v1/*_pb2*.pyi $(OUT_DIR)/ 2>/dev/null || true
	@rm -rf $(OUT_DIR)/edgeorchestra
	@# Fix imports in generated files to use relative imports
	@find $(OUT_DIR) -name '*_pb2_grpc.py' -exec sed -i '' \
		's/from edgeorchestra\.v1 import/from . import/g' {} +
	@find $(OUT_DIR) -name '*_pb2.py' -exec sed -i '' \
		's/from edgeorchestra\.v1 import/from . import/g' {} +
	@echo "Protobuf generation complete."

dev-up:
	docker compose up -d postgres redis
	@echo "Waiting for services..."
	@sleep 2
	docker compose ps

dev-down:
	docker compose down

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

migrate:
	cd orchestrator && uv run alembic upgrade head

migrate-new:
	cd orchestrator && uv run alembic revision --autogenerate -m "$(msg)"

run:
	cd orchestrator && uv run python -m orchestrator.main

validate:
	docker compose up -d postgres redis
	@sleep 3
	cd orchestrator && EO_LOG_FORMAT=console uv run python -m orchestrator.main
