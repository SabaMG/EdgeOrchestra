import random
import time

from google.protobuf.timestamp_pb2 import Timestamp

from orchestrator.generated import common_pb2

from worker_sim.device_profile import DeviceProfile


class MetricsSimulator:
    def __init__(self, profile: DeviceProfile, seed: int | None = None) -> None:
        self.profile = profile
        self.rng = random.Random(seed)
        self.is_training = False

        self.cpu_usage = self.rng.uniform(0.05, 0.15)
        self.memory_usage = self.rng.uniform(0.3, 0.5)
        self.thermal_pressure = self.rng.uniform(0.1, 0.2)
        self.battery_level = self.rng.uniform(0.6, 1.0)
        self.battery_charging = self.rng.random() < 0.3

    def start_training(self) -> None:
        self.is_training = True
        self.cpu_usage = max(self.cpu_usage, 0.6)

    def stop_training(self) -> None:
        self.is_training = False

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def tick(self) -> common_pb2.DeviceMetrics:
        # CPU random walk
        if self.is_training:
            self.cpu_usage += self.rng.gauss(0.02, 0.05)
            self.cpu_usage = self._clamp(self.cpu_usage, 0.55, 0.95)
        else:
            self.cpu_usage += self.rng.gauss(0, 0.03)
            self.cpu_usage = self._clamp(self.cpu_usage, 0.02, 0.4)

        # Memory
        if self.is_training:
            self.memory_usage += self.rng.gauss(0.01, 0.02)
            self.memory_usage = self._clamp(self.memory_usage, 0.5, 0.9)
        else:
            self.memory_usage += self.rng.gauss(0, 0.02)
            self.memory_usage = self._clamp(self.memory_usage, 0.2, 0.6)

        # Thermal correlates with CPU
        self.thermal_pressure = self._clamp(
            self.cpu_usage * 0.7 + self.rng.gauss(0, 0.05)
        )

        # Battery
        if self.battery_charging:
            self.battery_level += self.rng.uniform(0.005, 0.02)
            if self.battery_level >= 1.0:
                self.battery_level = 1.0
                self.battery_charging = False
        else:
            drain = 0.01 if self.is_training else 0.003
            self.battery_level -= self.rng.uniform(drain * 0.5, drain * 1.5)
            if self.battery_level <= 0.15 and self.rng.random() < 0.3:
                self.battery_charging = True

        self.battery_level = self._clamp(self.battery_level, 0.05, 1.0)

        if self.battery_charging:
            battery_state = common_pb2.BATTERY_STATE_CHARGING
        elif self.battery_level >= 0.99:
            battery_state = common_pb2.BATTERY_STATE_FULL
        else:
            battery_state = common_pb2.BATTERY_STATE_DISCHARGING

        now = Timestamp()
        now.FromSeconds(int(time.time()))

        return common_pb2.DeviceMetrics(
            cpu_usage=self.cpu_usage,
            memory_usage=self.memory_usage,
            thermal_pressure=self.thermal_pressure,
            battery=common_pb2.BatteryInfo(
                level=self.battery_level,
                state=battery_state,
                is_low_power_mode=self.battery_level < 0.2,
            ),
            collected_at=now,
        )
