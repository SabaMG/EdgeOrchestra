"""Federated Averaging with binary weight-delta serialization.

Binary format (shared with Swift):
  [layer_count: uint32_le]
  For each layer:
    [name_length: uint32_le]
    [name: utf8_bytes]
    [element_count: uint32_le]
    [values: float32_le × element_count]

Layer names: hidden_weight, hidden_bias, output_weight, output_bias
"""

import struct

import numpy as np

LAYER_NAMES = ["hidden_weight", "hidden_bias", "output_weight", "output_bias"]


def serialize_weight_deltas(
    deltas: dict[str, np.ndarray],
    layer_names: list[str] | None = None,
) -> bytes:
    """Serialize weight deltas to compact binary format."""
    parts = []
    if layer_names is not None:
        layers = [k for k in layer_names if k in deltas]
    else:
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


def deserialize_weight_deltas(data: bytes) -> dict[str, np.ndarray]:
    """Deserialize weight deltas from compact binary format."""
    offset = 0
    (layer_count,) = struct.unpack_from("<I", data, offset)
    offset += 4

    result = {}
    for _ in range(layer_count):
        (name_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
        (elem_count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        values = np.frombuffer(data, dtype=np.float32, count=elem_count, offset=offset).copy()
        offset += elem_count * 4
        result[name] = values

    return result


def aggregate_gradients(gradients: list[tuple[bytes, int]]) -> dict[str, np.ndarray]:
    """FedAvg: weighted average of gradient deltas from multiple devices."""
    total_samples = sum(n for _, n in gradients)
    if total_samples == 0:
        return {}

    accumulated: dict[str, np.ndarray] = {}
    for grad_bytes, num_samples in gradients:
        deltas = deserialize_weight_deltas(grad_bytes)
        weight = num_samples / total_samples
        for name, values in deltas.items():
            if name in accumulated:
                accumulated[name] += values * weight
            else:
                accumulated[name] = values * weight

    return accumulated


def apply_gradients(
    weights: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    learning_rate: float = 0.01,
) -> dict[str, np.ndarray]:
    """Apply averaged weight deltas to global weights.

    Workers send weight deltas (not raw loss gradients), and aggregate_gradients
    computes their weighted average. We apply them directly:
        new_weight = old_weight + averaged_delta

    The learning_rate parameter is kept for API compatibility but the actual
    learning rate is baked into the on-device training step that produces the deltas.
    """
    result = {}
    for name, w in weights.items():
        if name in grads:
            result[name] = w + grads[name].reshape(w.shape)
        else:
            result[name] = w.copy()
    return result
