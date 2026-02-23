import asyncio
import random
import struct

import numpy as np

# Layer shapes matching the CoreML MNIST model (784→128→10)
LAYER_SHAPES = {
    "hidden_weight": (128, 784),
    "hidden_bias": (128,),
    "output_weight": (10, 128),
    "output_bias": (10,),
}
LAYER_NAMES = ["hidden_weight", "hidden_bias", "output_weight", "output_bias"]


def serialize_weight_deltas(deltas: dict[str, np.ndarray]) -> bytes:
    """Serialize weight deltas to compact binary format (same as fed_avg.py and Swift)."""
    parts = []
    layers = [k for k in LAYER_NAMES if k in deltas]
    parts.append(struct.pack("<I", len(layers)))

    for name in layers:
        name_bytes = name.encode("utf-8")
        values = deltas[name].astype(np.float32).flatten()
        parts.append(struct.pack("<I", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<I", len(values)))
        parts.append(values.tobytes())

    return b"".join(parts)


async def simulate_local_training(
    model_weights: bytes,
    num_epochs: int = 1,
    num_samples: int = 100,
) -> tuple[bytes, int, dict]:
    """Simulate local training and return (gradients, num_samples, metrics).

    Now produces weight deltas in the new binary dict format compatible
    with the real CoreML trainer and the updated FedAvg aggregation.
    """
    # Simulate training time (1-3 seconds)
    await asyncio.sleep(random.uniform(1.0, 3.0))

    # Generate fake weight deltas (small random perturbations per layer)
    deltas = {}
    for name, shape in LAYER_SHAPES.items():
        deltas[name] = (np.random.randn(*shape) * 0.01).astype(np.float32)

    gradient_bytes = serialize_weight_deltas(deltas)

    # Compute a rough "norm" of model weights for simulated convergence metrics
    # The model is now a CoreML protobuf, so we just use round count proxy
    total_delta_norm = sum(np.linalg.norm(v) for v in deltas.values())
    base_loss = 2.0 / (1.0 + total_delta_norm * 0.5)
    loss = base_loss + random.uniform(-0.1, 0.1)
    accuracy = min(0.95, 0.3 + total_delta_norm * 0.02 + random.uniform(-0.05, 0.05))

    metrics = {
        "loss": round(max(0.01, loss), 4),
        "accuracy": round(max(0.0, min(1.0, accuracy)), 4),
        "num_epochs": float(num_epochs),
    }

    return gradient_bytes, num_samples, metrics
