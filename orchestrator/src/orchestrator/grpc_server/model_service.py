import base64
import json

import grpc
import structlog
from redis.asyncio import Redis

from orchestrator.observability.metrics import GRADIENT_SUBMISSIONS_TOTAL

logger = structlog.get_logger()

CHUNK_SIZE = 32 * 1024  # 32KB


class ModelServiceServicer:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def UploadModel(self, request_iterator, context):
        from orchestrator.generated import model_pb2

        model_id = None
        metadata = None
        chunks = []

        async for request in request_iterator:
            if request.HasField("metadata"):
                metadata = request.metadata
                model_id = metadata.model_id
            elif request.chunk:
                chunks.append(request.chunk)

        if not model_id or not metadata:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Missing model metadata")
            return model_pb2.UploadModelResponse()

        model_bytes = b"".join(chunks)
        encoded = base64.b64encode(model_bytes).decode()
        await self.redis.set(f"model:{model_id}:global", encoded)

        meta_dict = {
            "model_id": metadata.model_id,
            "name": metadata.name,
            "version": metadata.version,
            "framework": metadata.framework,
            "size_bytes": len(model_bytes),
        }
        await self.redis.set(f"model:{model_id}:meta", json.dumps(meta_dict))

        logger.info("model_uploaded", model_id=model_id, size=len(model_bytes))
        return model_pb2.UploadModelResponse(
            model_id=model_id,
            metadata=metadata,
        )

    async def DownloadModel(self, request, context):
        from orchestrator.generated import model_pb2

        model_id = request.model_id
        encoded = await self.redis.get(f"model:{model_id}:global")

        if not encoded:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model {model_id} not found")
            return

        model_bytes = base64.b64decode(encoded)
        meta_raw = await self.redis.get(f"model:{model_id}:meta")
        meta_dict = json.loads(meta_raw) if meta_raw else {}

        # Send metadata first
        yield model_pb2.DownloadModelChunk(
            metadata=model_pb2.ModelMetadata(
                model_id=model_id,
                name=meta_dict.get("name", ""),
                version=meta_dict.get("version", ""),
                framework=meta_dict.get("framework", ""),
                size_bytes=len(model_bytes),
            )
        )

        # Send data in chunks
        for i in range(0, len(model_bytes), CHUNK_SIZE):
            yield model_pb2.DownloadModelChunk(
                chunk=model_bytes[i : i + CHUNK_SIZE]
            )

        logger.debug("model_downloaded", model_id=model_id, device_id=request.device_id.value)

    async def SubmitGradients(self, request, context):
        from orchestrator.generated import model_pb2

        device_id = request.device_id.value
        model_id = request.model_id
        training_round = request.training_round

        # Validate inputs
        if not device_id or not model_id or not training_round:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Missing device_id, model_id, or training_round")
            return model_pb2.SubmitGradientsResponse(accepted=False)

        if not request.gradients:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty gradients payload")
            return model_pb2.SubmitGradientsResponse(accepted=False)

        if request.num_samples <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("num_samples must be > 0")
            return model_pb2.SubmitGradientsResponse(accepted=False)

        # Validate gradient binary format (must have at least a layer count header)
        if len(request.gradients) < 4:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Gradient data too small to be valid")
            return model_pb2.SubmitGradientsResponse(accepted=False)

        # Decompress gradients (float16+lz4 → float32, or passthrough for legacy)
        from orchestrator.services.gradient_codec import decompress_gradients
        gradients_bytes = decompress_gradients(request.gradients)

        # Store gradient entry as base64-encoded JSON
        entry = json.dumps({
            "device_id": device_id,
            "gradients": base64.b64encode(gradients_bytes).decode(),
            "num_samples": request.num_samples,
            "metrics": dict(request.metrics),
        })
        await self.redis.rpush(f"gradients:{model_id}:{training_round}", entry)
        GRADIENT_SUBMISSIONS_TOTAL.inc()

        logger.info(
            "gradients_received",
            device_id=device_id,
            model_id=model_id,
            round=training_round,
            num_samples=request.num_samples,
        )

        return model_pb2.SubmitGradientsResponse(accepted=True)
