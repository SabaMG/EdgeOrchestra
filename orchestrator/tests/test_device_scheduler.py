"""Tests for device_scheduler: SchedulerConfig, eligibility, scoring, selection."""

import uuid
from types import SimpleNamespace

import pytest

from orchestrator.services.device_scheduler import (
    SchedulerConfig,
    _is_eligible,
    _score_device,
    select_devices,
)


def _make_device(**kwargs) -> SimpleNamespace:
    defaults = {
        "id": uuid.uuid4(),
        "battery_level": 0.8,
        "battery_state": "discharging",
        "metrics": {"cpu_usage": 0.3, "memory_usage": 0.4, "thermal_pressure": 0.2},
        "neural_engine_cores": 16,
        "memory_bytes": 8_000_000_000,
        "chip": "A17 Pro",
        "status": "online",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------
class TestSchedulerConfig:
    def test_from_none(self):
        cfg = SchedulerConfig.from_job_config(None)
        assert cfg.enabled is False

    def test_from_empty_dict(self):
        cfg = SchedulerConfig.from_job_config({})
        assert cfg.enabled is False
        assert cfg.min_battery == 0.20

    def test_partial_override(self):
        cfg = SchedulerConfig.from_job_config({
            "scheduler": {"enabled": True, "min_battery": 0.50}
        })
        assert cfg.enabled is True
        assert cfg.min_battery == 0.50
        assert cfg.max_thermal_pressure == 0.70  # default kept

    def test_weight_merge(self):
        cfg = SchedulerConfig.from_job_config({
            "scheduler": {"enabled": True, "weights": {"battery": 0.60}}
        })
        assert cfg.weights["battery"] == 0.60
        assert cfg.weights["thermal"] == 0.25  # default kept

    def test_full_config(self):
        cfg = SchedulerConfig.from_job_config({
            "scheduler": {
                "enabled": True,
                "target_devices": 5,
                "min_battery": 0.30,
                "allow_low_power_mode": True,
                "max_thermal_pressure": 0.50,
                "max_cpu_usage": 0.80,
                "weights": {
                    "battery": 0.10,
                    "thermal": 0.10,
                    "cpu_load": 0.30,
                    "memory_load": 0.30,
                    "hardware": 0.20,
                },
            }
        })
        assert cfg.target_devices == 5
        assert cfg.allow_low_power_mode is True
        assert cfg.weights["cpu_load"] == 0.30


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
class TestEligibility:
    def test_low_battery_excluded(self):
        cfg = SchedulerConfig(enabled=True, min_battery=0.20)
        d = _make_device(battery_level=0.10)
        assert _is_eligible(d, cfg) is False

    def test_battery_at_min_included(self):
        cfg = SchedulerConfig(enabled=True, min_battery=0.20)
        d = _make_device(battery_level=0.20)
        assert _is_eligible(d, cfg) is True

    def test_low_power_mode_excluded(self):
        cfg = SchedulerConfig(enabled=True, allow_low_power_mode=False)
        d = _make_device(metrics={"cpu_usage": 0.3, "memory_usage": 0.4, "thermal_pressure": 0.2, "is_low_power_mode": True})
        assert _is_eligible(d, cfg) is False

    def test_low_power_mode_allowed(self):
        cfg = SchedulerConfig(enabled=True, allow_low_power_mode=True)
        d = _make_device(metrics={"cpu_usage": 0.3, "memory_usage": 0.4, "thermal_pressure": 0.2, "is_low_power_mode": True})
        assert _is_eligible(d, cfg) is True

    def test_thermal_exceeded(self):
        cfg = SchedulerConfig(enabled=True, max_thermal_pressure=0.70)
        d = _make_device(metrics={"cpu_usage": 0.3, "memory_usage": 0.4, "thermal_pressure": 0.80})
        assert _is_eligible(d, cfg) is False

    def test_cpu_exceeded(self):
        cfg = SchedulerConfig(enabled=True, max_cpu_usage=0.90)
        d = _make_device(metrics={"cpu_usage": 0.95, "memory_usage": 0.4, "thermal_pressure": 0.2})
        assert _is_eligible(d, cfg) is False

    def test_none_metrics_eligible(self):
        cfg = SchedulerConfig(enabled=True)
        d = _make_device(battery_level=0.80, metrics=None)
        assert _is_eligible(d, cfg) is True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
class TestScoring:
    def test_charging_higher_than_discharging(self):
        cfg = SchedulerConfig(enabled=True)
        d_charge = _make_device(battery_level=0.5, battery_state="charging")
        d_discharge = _make_device(battery_level=0.5, battery_state="discharging")
        s1 = _score_device(d_charge, cfg, 16, 8_000_000_000)
        s2 = _score_device(d_discharge, cfg, 16, 8_000_000_000)
        assert s1 > s2

    def test_higher_battery_higher_score(self):
        cfg = SchedulerConfig(enabled=True)
        d_high = _make_device(battery_level=0.9)
        d_low = _make_device(battery_level=0.3)
        s1 = _score_device(d_high, cfg, 16, 8_000_000_000)
        s2 = _score_device(d_low, cfg, 16, 8_000_000_000)
        assert s1 > s2

    def test_lower_thermal_higher_score(self):
        cfg = SchedulerConfig(enabled=True)
        d_cool = _make_device(metrics={"cpu_usage": 0.3, "memory_usage": 0.4, "thermal_pressure": 0.1})
        d_hot = _make_device(metrics={"cpu_usage": 0.3, "memory_usage": 0.4, "thermal_pressure": 0.6})
        s1 = _score_device(d_cool, cfg, 16, 8_000_000_000)
        s2 = _score_device(d_hot, cfg, 16, 8_000_000_000)
        assert s1 > s2

    def test_hardware_normalization(self):
        cfg = SchedulerConfig(enabled=True, weights={"battery": 0, "thermal": 0, "cpu_load": 0, "memory_load": 0, "hardware": 1.0})
        d_big = _make_device(neural_engine_cores=16, memory_bytes=8_000_000_000)
        d_small = _make_device(neural_engine_cores=8, memory_bytes=4_000_000_000)
        s1 = _score_device(d_big, cfg, 16, 8_000_000_000)
        s2 = _score_device(d_small, cfg, 16, 8_000_000_000)
        assert s1 > s2

    def test_custom_weights(self):
        cfg = SchedulerConfig(enabled=True, weights={"battery": 1.0, "thermal": 0, "cpu_load": 0, "memory_load": 0, "hardware": 0})
        d = _make_device(battery_level=0.9, battery_state="discharging")
        score = _score_device(d, cfg, 16, 8_000_000_000)
        assert pytest.approx(score, abs=0.01) == 0.9


# ---------------------------------------------------------------------------
# select_devices
# ---------------------------------------------------------------------------
class TestSelectDevices:
    def test_disabled_returns_all(self):
        cfg = SchedulerConfig(enabled=False)
        devices = [_make_device() for _ in range(5)]
        result = select_devices(devices, cfg, min_devices=2)
        assert result is devices

    def test_not_enough_eligible_returns_none(self):
        cfg = SchedulerConfig(enabled=True, min_battery=0.50)
        devices = [_make_device(battery_level=0.10) for _ in range(3)]
        result = select_devices(devices, cfg, min_devices=2)
        assert result is None

    def test_target_limits_selection(self):
        cfg = SchedulerConfig(enabled=True, target_devices=2)
        devices = [_make_device() for _ in range(5)]
        result = select_devices(devices, cfg, min_devices=1)
        assert result is not None
        assert len(result) == 2

    def test_ordered_by_score(self):
        cfg = SchedulerConfig(enabled=True)
        d_best = _make_device(battery_level=0.95, metrics={"cpu_usage": 0.1, "memory_usage": 0.1, "thermal_pressure": 0.05})
        d_worst = _make_device(battery_level=0.25, metrics={"cpu_usage": 0.8, "memory_usage": 0.8, "thermal_pressure": 0.6})
        result = select_devices([d_worst, d_best], cfg, min_devices=1)
        assert result is not None
        assert result[0] is d_best

    def test_target_clamped_to_min(self):
        cfg = SchedulerConfig(enabled=True, target_devices=1)
        devices = [_make_device() for _ in range(5)]
        result = select_devices(devices, cfg, min_devices=3)
        assert result is not None
        assert len(result) == 3
