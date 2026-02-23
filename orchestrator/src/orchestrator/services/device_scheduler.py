"""Device scheduler: eligibility filtering, scoring, and selection for training rounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.db.models import Device

_DEFAULT_WEIGHTS = {
    "battery": 0.35,
    "thermal": 0.25,
    "cpu_load": 0.20,
    "memory_load": 0.10,
    "hardware": 0.10,
}


@dataclass
class SchedulerConfig:
    enabled: bool = False
    target_devices: int | None = None
    min_battery: float = 0.20
    allow_low_power_mode: bool = False
    max_thermal_pressure: float = 0.70
    max_cpu_usage: float = 0.90
    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    @classmethod
    def from_job_config(cls, config: dict | None) -> SchedulerConfig:
        if not config:
            return cls()
        sched = config.get("scheduler", {})
        if not sched:
            return cls()
        weights = {**_DEFAULT_WEIGHTS, **sched.get("weights", {})}
        return cls(
            enabled=sched.get("enabled", False),
            target_devices=sched.get("target_devices"),
            min_battery=sched.get("min_battery", 0.20),
            allow_low_power_mode=sched.get("allow_low_power_mode", False),
            max_thermal_pressure=sched.get("max_thermal_pressure", 0.70),
            max_cpu_usage=sched.get("max_cpu_usage", 0.90),
            weights=weights,
        )


def _get_metric(device: Device, key: str, default: float | None = None) -> float | None:
    metrics = getattr(device, "metrics", None) or {}
    return metrics.get(key, default)


def _is_eligible(device: Device, cfg: SchedulerConfig) -> bool:
    battery = getattr(device, "battery_level", None)
    if battery is not None and battery < cfg.min_battery:
        return False

    is_lpm = _get_metric(device, "is_low_power_mode")
    if is_lpm and not cfg.allow_low_power_mode:
        return False

    thermal = _get_metric(device, "thermal_pressure")
    if thermal is not None and thermal > cfg.max_thermal_pressure:
        return False

    cpu = _get_metric(device, "cpu_usage")
    if cpu is not None and cpu > cfg.max_cpu_usage:
        return False

    return True


def _score_device(device: Device, cfg: SchedulerConfig, pool_max_ne: int, pool_max_mem: int) -> float:
    w = cfg.weights

    # Battery sub-score
    battery = getattr(device, "battery_level", None)
    if battery is not None:
        battery_state = getattr(device, "battery_state", None)
        bonus = 0.15 if battery_state in ("charging", "full") else 0.0
        battery_score = min(battery + bonus, 1.0)
    else:
        battery_score = 0.5

    # Thermal sub-score
    thermal = _get_metric(device, "thermal_pressure")
    thermal_score = (1.0 - thermal) if thermal is not None else 0.5

    # CPU load sub-score
    cpu = _get_metric(device, "cpu_usage")
    cpu_score = (1.0 - cpu) if cpu is not None else 0.5

    # Memory load sub-score
    mem = _get_metric(device, "memory_usage")
    mem_score = (1.0 - mem) if mem is not None else 0.5

    # Hardware sub-score (normalized within pool)
    ne_cores = getattr(device, "neural_engine_cores", None) or 0
    mem_bytes = getattr(device, "memory_bytes", None) or 0
    ne_norm = (ne_cores / pool_max_ne) if pool_max_ne > 0 else 0.5
    mem_norm = (mem_bytes / pool_max_mem) if pool_max_mem > 0 else 0.5
    hw_score = (ne_norm + mem_norm) / 2.0

    total = (
        w.get("battery", 0) * battery_score
        + w.get("thermal", 0) * thermal_score
        + w.get("cpu_load", 0) * cpu_score
        + w.get("memory_load", 0) * mem_score
        + w.get("hardware", 0) * hw_score
    )
    return total


def select_devices(
    devices: list[Device], cfg: SchedulerConfig, min_devices: int,
) -> list[Device] | None:
    if not cfg.enabled:
        return devices

    eligible = [d for d in devices if _is_eligible(d, cfg)]
    if len(eligible) < min_devices:
        return None

    # Compute pool maximums for hardware normalization
    pool_max_ne = max((getattr(d, "neural_engine_cores", None) or 0 for d in eligible), default=0)
    pool_max_mem = max((getattr(d, "memory_bytes", None) or 0 for d in eligible), default=0)

    scored = sorted(
        eligible,
        key=lambda d: _score_device(d, cfg, pool_max_ne, pool_max_mem),
        reverse=True,
    )

    target = cfg.target_devices
    if target is None:
        return scored
    # Clamp target to at least min_devices
    target = max(target, min_devices)
    return scored[:target]
