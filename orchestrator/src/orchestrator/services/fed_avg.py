import numpy as np

MODEL_SIZE = 7850  # 784*10 weights + 10 biases (single-layer MNIST)


def create_initial_model() -> bytes:
    return np.zeros(MODEL_SIZE, dtype=np.float64).tobytes()


def aggregate_gradients(gradients: list[tuple[bytes, int]]) -> bytes:
    total_samples = sum(n for _, n in gradients)
    if total_samples == 0:
        return np.zeros(MODEL_SIZE, dtype=np.float64).tobytes()

    weighted_sum = np.zeros(MODEL_SIZE, dtype=np.float64)
    for grad_bytes, num_samples in gradients:
        grad = np.frombuffer(grad_bytes, dtype=np.float64)
        weighted_sum += grad * num_samples

    averaged = weighted_sum / total_samples
    return averaged.tobytes()


def apply_gradients(
    model_weights: bytes,
    averaged_gradients: bytes,
    learning_rate: float = 0.01,
) -> bytes:
    weights = np.frombuffer(model_weights, dtype=np.float64).copy()
    grads = np.frombuffer(averaged_gradients, dtype=np.float64)
    weights -= learning_rate * grads
    return weights.tobytes()
