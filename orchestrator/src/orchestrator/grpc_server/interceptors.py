import time

import grpc
import grpc.aio
import structlog

from orchestrator.observability.metrics import GRPC_REQUEST_DURATION, GRPC_REQUESTS_TOTAL

logger = structlog.get_logger()


class LoggingMetricsInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        handler = await continuation(handler_call_details)
        if handler is None:
            return handler

        # Wrap all handler types
        if handler.unary_unary:
            original = handler.unary_unary

            async def wrapped_unary_unary(request, context):
                start = time.perf_counter()
                try:
                    response = await original(request, context)
                    status = "OK"
                    return response
                except Exception as exc:
                    status = "ERROR"
                    raise
                finally:
                    duration = time.perf_counter() - start
                    GRPC_REQUEST_DURATION.labels(method=method, status=status).observe(duration)
                    GRPC_REQUESTS_TOTAL.labels(method=method, status=status).inc()
                    logger.info("grpc_request", method=method, status=status, duration_ms=round(duration * 1000, 1))

            return grpc.unary_unary_rpc_method_handler(
                wrapped_unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.unary_stream:
            original = handler.unary_stream

            async def wrapped_unary_stream(request, context):
                start = time.perf_counter()
                status = "OK"
                try:
                    async for response in original(request, context):
                        yield response
                except Exception:
                    status = "ERROR"
                    raise
                finally:
                    duration = time.perf_counter() - start
                    GRPC_REQUEST_DURATION.labels(method=method, status=status).observe(duration)
                    GRPC_REQUESTS_TOTAL.labels(method=method, status=status).inc()
                    logger.info("grpc_stream_ended", method=method, status=status, duration_ms=round(duration * 1000, 1))

            return grpc.unary_stream_rpc_method_handler(
                wrapped_unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.stream_unary:
            original = handler.stream_unary

            async def wrapped_stream_unary(request_iterator, context):
                start = time.perf_counter()
                try:
                    response = await original(request_iterator, context)
                    status = "OK"
                    return response
                except Exception:
                    status = "ERROR"
                    raise
                finally:
                    duration = time.perf_counter() - start
                    GRPC_REQUEST_DURATION.labels(method=method, status=status).observe(duration)
                    GRPC_REQUESTS_TOTAL.labels(method=method, status=status).inc()
                    logger.info("grpc_stream_ended", method=method, status=status, duration_ms=round(duration * 1000, 1))

            return grpc.stream_unary_rpc_method_handler(
                wrapped_stream_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.stream_stream:
            original = handler.stream_stream

            async def wrapped_stream_stream(request_iterator, context):
                start = time.perf_counter()
                status = "OK"
                try:
                    async for response in original(request_iterator, context):
                        yield response
                except Exception:
                    status = "ERROR"
                    raise
                finally:
                    duration = time.perf_counter() - start
                    GRPC_REQUEST_DURATION.labels(method=method, status=status).observe(duration)
                    GRPC_REQUESTS_TOTAL.labels(method=method, status=status).inc()
                    logger.info("grpc_stream_ended", method=method, status=status, duration_ms=round(duration * 1000, 1))

            return grpc.stream_stream_rpc_method_handler(
                wrapped_stream_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        return handler


class ApiKeyInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, expected_key: str) -> None:
        self._expected_key = expected_key

    async def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        if metadata.get("x-api-key") != self._expected_key:
            return grpc.unary_unary_rpc_method_handler(self._abort)
        return await continuation(handler_call_details)

    @staticmethod
    async def _abort(request, context):
        await context.abort(
            grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing API key"
        )
