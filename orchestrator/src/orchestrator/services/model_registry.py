"""Registry of supported model architectures for federated learning."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelArchitecture:
    key: str
    name: str
    input_shape: tuple[int, ...]
    num_classes: int
    layer_names: list[str] = field(default_factory=list)
    layer_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)


ARCHITECTURES: dict[str, ModelArchitecture] = {
    "mnist": ModelArchitecture(
        key="mnist",
        name="MNIST Classifier (784\u2192128\u219210)",
        input_shape=(1, 28, 28),
        num_classes=10,
        layer_names=["hidden_weight", "hidden_bias", "output_weight", "output_bias"],
        layer_shapes={
            "hidden_weight": (128, 784),
            "hidden_bias": (128,),
            "output_weight": (10, 128),
            "output_bias": (10,),
        },
    ),
    "cifar10": ModelArchitecture(
        key="cifar10",
        name="CIFAR-10 Classifier (3072\u2192256\u2192128\u219210)",
        input_shape=(3, 32, 32),
        num_classes=10,
        layer_names=[
            "hidden1_weight", "hidden1_bias",
            "hidden2_weight", "hidden2_bias",
            "output_weight", "output_bias",
        ],
        layer_shapes={
            "hidden1_weight": (256, 3072),
            "hidden1_bias": (256,),
            "hidden2_weight": (128, 256),
            "hidden2_bias": (128,),
            "output_weight": (10, 128),
            "output_bias": (10,),
        },
    ),
}


def get_architecture(key: str) -> ModelArchitecture:
    if key not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {key!r}. Available: {list(ARCHITECTURES)}")
    return ARCHITECTURES[key]


def list_architectures() -> list[ModelArchitecture]:
    return list(ARCHITECTURES.values())
