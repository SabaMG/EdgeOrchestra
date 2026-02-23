from prometheus_client import Counter, Gauge, Histogram

# HTTP
HTTP_REQUEST_DURATION = Histogram(
    "eo_http_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["method", "path", "status_code"],
)
HTTP_REQUESTS_TOTAL = Counter(
    "eo_http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

# gRPC
GRPC_REQUEST_DURATION = Histogram(
    "eo_grpc_request_duration_seconds",
    "Duration of gRPC requests in seconds",
    ["method", "status"],
)
GRPC_REQUESTS_TOTAL = Counter(
    "eo_grpc_requests_total",
    "Total number of gRPC requests",
    ["method", "status"],
)

# Business
TRAINING_JOBS_ACTIVE = Gauge(
    "eo_training_jobs_active",
    "Number of currently active training jobs",
)
TRAINING_ROUNDS_TOTAL = Counter(
    "eo_training_rounds_total",
    "Total number of completed training rounds",
)
TRAINING_ROUND_DURATION = Histogram(
    "eo_training_round_duration_seconds",
    "Duration of training rounds in seconds",
    buckets=(5, 15, 30, 60, 120, 180, 300, 600),
)
DEVICES_BY_STATUS = Gauge(
    "eo_devices",
    "Number of devices by status",
    ["status"],
)
GRADIENT_SUBMISSIONS_TOTAL = Counter(
    "eo_gradient_submissions_total",
    "Total number of gradient submissions received",
)
HEARTBEATS_TOTAL = Counter(
    "eo_heartbeats_total",
    "Total number of heartbeats processed",
)
