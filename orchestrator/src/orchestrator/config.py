from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "EO_"}

    # Database
    database_url: str = (
        "postgresql+asyncpg://edgeorchestra:edgeorchestra@localhost:5432/edgeorchestra"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051

    # mDNS
    mdns_enabled: bool = True
    mdns_service_name: str = "EdgeOrchestra"

    # Heartbeat
    heartbeat_interval_seconds: int = 1
    heartbeat_timeout_multiplier: int = 5

    # Training
    training_round_timeout_seconds: int = 180

    # Security - TLS
    tls_enabled: bool = False
    tls_ca_cert: str = "certs/ca.crt"
    tls_server_cert: str = "certs/server.crt"
    tls_server_key: str = "certs/server.key"

    # Security - API Key
    api_key: str = ""

    # Logging
    log_level: str = "INFO"
    log_format: str = "console"


settings = Settings()
