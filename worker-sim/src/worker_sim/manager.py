import asyncio
import logging
import signal

from worker_sim.device_profile import get_profile
from worker_sim.worker import SimulatedWorker

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(
        self,
        target: str,
        profile_name: str,
        count: int,
        heartbeat_interval: float,
    ) -> None:
        self.target = target
        self.profile_name = profile_name
        self.count = count
        self.heartbeat_interval = heartbeat_interval
        self.workers: list[SimulatedWorker] = []
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        # Create and start workers
        for i in range(self.count):
            profile = get_profile(self.profile_name, i)
            worker = SimulatedWorker(
                target=self.target,
                profile=profile,
                heartbeat_interval=self.heartbeat_interval,
            )
            self.workers.append(worker)

        logger.info(f"Starting {self.count} simulated worker(s)...")

        for worker in self.workers:
            await worker.start()
            await asyncio.sleep(0.5)  # Stagger registrations

        logger.info(f"All {self.count} worker(s) running. Press Ctrl+C to stop.")

        await self._shutdown_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("Shutting down workers...")
        for worker in self.workers:
            try:
                await asyncio.wait_for(worker.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{worker.profile.name}] Stop timed out")
        logger.info(f"All {len(self.workers)} worker(s) stopped.")


def run_workers(
    target: str,
    profile_name: str,
    count: int,
    heartbeat_interval: float,
) -> None:
    manager = WorkerManager(target, profile_name, count, heartbeat_interval)
    asyncio.run(manager.run())
