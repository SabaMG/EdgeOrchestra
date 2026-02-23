import grpc
from grpc_reflection.v1alpha import reflection
import structlog

from orchestrator.config import settings

logger = structlog.get_logger()


async def create_grpc_server(
    device_service,
    heartbeat_service,
    model_service,
    device_pb2,
    heartbeat_pb2,
    model_pb2,
    device_pb2_grpc,
    heartbeat_pb2_grpc,
    model_pb2_grpc,
) -> grpc.aio.Server:
    from orchestrator.grpc_server.interceptors import LoggingMetricsInterceptor

    interceptors = [LoggingMetricsInterceptor()]
    if settings.api_key:
        from orchestrator.grpc_server.interceptors import ApiKeyInterceptor

        interceptors.append(ApiKeyInterceptor(settings.api_key))

    server = grpc.aio.server(interceptors=interceptors)

    device_pb2_grpc.add_DeviceRegistryServicer_to_server(device_service, server)
    heartbeat_pb2_grpc.add_HeartbeatServiceServicer_to_server(heartbeat_service, server)
    model_pb2_grpc.add_ModelServiceServicer_to_server(model_service, server)

    service_names = (
        device_pb2.DESCRIPTOR.services_by_name["DeviceRegistry"].full_name,
        heartbeat_pb2.DESCRIPTOR.services_by_name["HeartbeatService"].full_name,
        model_pb2.DESCRIPTOR.services_by_name["ModelService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"

    if settings.tls_enabled:
        with open(settings.tls_ca_cert, "rb") as f:
            ca_cert = f.read()
        with open(settings.tls_server_cert, "rb") as f:
            server_cert = f.read()
        with open(settings.tls_server_key, "rb") as f:
            server_key = f.read()
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True,
        )
        server.add_secure_port(listen_addr, credentials)
        logger.info("grpc_server_configured", address=listen_addr, tls=True)
    else:
        server.add_insecure_port(listen_addr)
        logger.info("grpc_server_configured", address=listen_addr, tls=False)

    return server
