import grpc
import structlog

logger = structlog.get_logger()


class ModelServiceServicer:
    """Stub model service for Phase 3."""

    async def UploadModel(self, request_iterator, context):
        from orchestrator.generated import model_pb2

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Model upload not yet implemented")
        return model_pb2.UploadModelResponse()

    async def DownloadModel(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Model download not yet implemented")

    async def SubmitGradients(self, request, context):
        from orchestrator.generated import model_pb2

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Gradient submission not yet implemented")
        return model_pb2.SubmitGradientsResponse()
