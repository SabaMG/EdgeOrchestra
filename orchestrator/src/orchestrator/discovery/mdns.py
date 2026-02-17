import socket

import structlog
from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from orchestrator.config import settings

logger = structlog.get_logger()

SERVICE_TYPE = "_edgeorchestra._tcp.local."


class MDNSDiscovery:
    def __init__(self) -> None:
        self.zeroconf: AsyncZeroconf | None = None
        self.service_info: ServiceInfo | None = None

    def _get_local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    async def register(self) -> None:
        if not settings.mdns_enabled:
            logger.info("mdns_disabled")
            return

        local_ip = self._get_local_ip()
        self.service_info = ServiceInfo(
            SERVICE_TYPE,
            f"{settings.mdns_service_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=settings.grpc_port,
            properties={
                "api_port": str(settings.api_port),
                "grpc_port": str(settings.grpc_port),
                "version": "0.1.0",
            },
            server=f"{settings.mdns_service_name}.local.",
        )

        self.zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self.zeroconf.async_register_service(self.service_info)
        logger.info("mdns_registered", service=SERVICE_TYPE, ip=local_ip)

    async def unregister(self) -> None:
        if self.zeroconf and self.service_info:
            await self.zeroconf.async_unregister_service(self.service_info)
            await self.zeroconf.async_close()
            logger.info("mdns_unregistered")
