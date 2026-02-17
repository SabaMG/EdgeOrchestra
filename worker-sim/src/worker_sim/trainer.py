import asyncio
import random

import numpy as np

MODEL_SIZE = 7850  # 784*10 weights + 10 biases (single-layer MNIST)


async def simulate_local_training(
    model_weights: bytes,
    num_epochs: int = 1,
    num_samples: int = 100,
) -> tuple[bytes, int, dict]:
    """Simulate local training and return (gradients, num_samples, metrics)."""
    weights = np.frombuffer(model_weights, dtype=np.float64)

    # Simulate training time (1-3 seconds)
    await asyncio.sleep(random.uniform(1.0, 3.0))

    # Generate fake gradients (small random perturbations)
    gradients = np.random.randn(MODEL_SIZE) * 0.01

    # Fake metrics - loss decreases with model norm (simulating convergence)
    model_norm = np.linalg.norm(weights)
    base_loss = 2.0 / (1.0 + model_norm * 0.1)
    loss = base_loss + random.uniform(-0.1, 0.1)
    accuracy = min(0.95, 0.3 + model_norm * 0.05 + random.uniform(-0.05, 0.05))

    metrics = {
        "loss": round(max(0.01, loss), 4),
        "accuracy": round(max(0.0, min(1.0, accuracy)), 4),
        "num_epochs": float(num_epochs),
    }

    return gradients.tobytes(), num_samples, metrics
