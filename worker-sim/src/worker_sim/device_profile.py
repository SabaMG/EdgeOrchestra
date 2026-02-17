from dataclasses import dataclass


@dataclass
class DeviceProfile:
    name: str
    device_model: str
    os_version: str
    chip: str
    memory_bytes: int
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int
    supported_frameworks: list[str]


PROFILES: dict[str, DeviceProfile] = {
    "iphone15pro": DeviceProfile(
        name="iPhone 15 Pro",
        device_model="iPhone16,1",
        os_version="17.4",
        chip="A17 Pro",
        memory_bytes=8 * 1024**3,
        cpu_cores=6,
        gpu_cores=6,
        neural_engine_cores=16,
        supported_frameworks=["coreml"],
    ),
    "iphone14": DeviceProfile(
        name="iPhone 14",
        device_model="iPhone15,2",
        os_version="17.3",
        chip="A15 Bionic",
        memory_bytes=6 * 1024**3,
        cpu_cores=6,
        gpu_cores=5,
        neural_engine_cores=16,
        supported_frameworks=["coreml"],
    ),
    "macbook-m2": DeviceProfile(
        name="MacBook Pro M2",
        device_model="Mac14,7",
        os_version="14.3",
        chip="Apple M2",
        memory_bytes=16 * 1024**3,
        cpu_cores=8,
        gpu_cores=10,
        neural_engine_cores=16,
        supported_frameworks=["coreml", "mlx"],
    ),
    "ipad-m4": DeviceProfile(
        name="iPad Pro M4",
        device_model="iPad16,3",
        os_version="17.4",
        chip="Apple M4",
        memory_bytes=16 * 1024**3,
        cpu_cores=10,
        gpu_cores=10,
        neural_engine_cores=16,
        supported_frameworks=["coreml", "mlx"],
    ),
}


def get_profile(name: str, index: int) -> DeviceProfile:
    base = PROFILES[name]
    return DeviceProfile(
        name=f"{base.name} #{index + 1}",
        device_model=base.device_model,
        os_version=base.os_version,
        chip=base.chip,
        memory_bytes=base.memory_bytes,
        cpu_cores=base.cpu_cores,
        gpu_cores=base.gpu_cores,
        neural_engine_cores=base.neural_engine_cores,
        supported_frameworks=list(base.supported_frameworks),
    )
