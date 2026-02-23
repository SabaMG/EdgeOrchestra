"""Tests for binary serialization and FedAvg aggregation."""

import struct

import numpy as np
import pytest

from orchestrator.services.fed_avg import (
    LAYER_NAMES,
    aggregate_gradients,
    apply_gradients,
    deserialize_weight_deltas,
    serialize_weight_deltas,
)


def _make_deltas() -> dict[str, np.ndarray]:
    """Helper: create small deterministic weight deltas."""
    rng = np.random.RandomState(42)
    return {
        "hidden_weight": rng.randn(128, 784).astype(np.float32),
        "hidden_bias": rng.randn(128).astype(np.float32),
        "output_weight": rng.randn(10, 128).astype(np.float32),
        "output_bias": rng.randn(10).astype(np.float32),
    }


class TestSerialization:
    def test_serialize_deserialize_roundtrip(self):
        deltas = _make_deltas()
        data = serialize_weight_deltas(deltas)
        result = deserialize_weight_deltas(data)

        assert set(result.keys()) == set(deltas.keys())
        for name in deltas:
            np.testing.assert_allclose(result[name], deltas[name].flatten(), rtol=0)

    def test_serialize_layer_order(self):
        deltas = _make_deltas()
        data = serialize_weight_deltas(deltas)

        # Parse layer names from binary to verify order
        offset = 0
        (layer_count,) = struct.unpack_from("<I", data, offset)
        offset += 4

        names_in_order = []
        for _ in range(layer_count):
            (name_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            name = data[offset : offset + name_len].decode("utf-8")
            names_in_order.append(name)
            offset += name_len
            (elem_count,) = struct.unpack_from("<I", data, offset)
            offset += 4 + elem_count * 4

        assert names_in_order == LAYER_NAMES

    def test_deserialize_empty_data_raises(self):
        with pytest.raises((struct.error, Exception)):
            deserialize_weight_deltas(b"")


class TestAggregation:
    def test_aggregate_single_device(self):
        deltas = _make_deltas()
        grad_bytes = serialize_weight_deltas(deltas)
        result = aggregate_gradients([(grad_bytes, 100)])

        for name in deltas:
            np.testing.assert_allclose(
                result[name], deltas[name].flatten(), rtol=1e-6
            )

    def test_aggregate_two_devices_equal_weight(self):
        d1 = {k: np.ones_like(v) for k, v in _make_deltas().items()}
        d2 = {k: np.ones_like(v) * 3.0 for k, v in _make_deltas().items()}
        b1 = serialize_weight_deltas(d1)
        b2 = serialize_weight_deltas(d2)

        result = aggregate_gradients([(b1, 50), (b2, 50)])
        for name in d1:
            expected = 2.0  # (1*0.5 + 3*0.5)
            np.testing.assert_allclose(result[name], expected, rtol=1e-6)

    def test_aggregate_weighted_by_samples(self):
        d1 = {k: np.ones_like(v) for k, v in _make_deltas().items()}
        d2 = {k: np.ones_like(v) * 4.0 for k, v in _make_deltas().items()}
        b1 = serialize_weight_deltas(d1)
        b2 = serialize_weight_deltas(d2)

        # 25% weight to d1, 75% weight to d2
        result = aggregate_gradients([(b1, 10), (b2, 30)])
        for name in d1:
            expected = 1.0 * 0.25 + 4.0 * 0.75  # 3.25
            np.testing.assert_allclose(result[name], expected, rtol=1e-6)

    def test_aggregate_empty_list(self):
        result = aggregate_gradients([])
        assert result == {}


    def test_aggregate_with_compressed_gradients(self):
        """Compress → decompress → aggregate produces correct result."""
        from orchestrator.services.gradient_codec import (
            compress_gradients,
            decompress_gradients,
        )

        d1 = {k: np.ones_like(v) for k, v in _make_deltas().items()}
        d2 = {k: np.ones_like(v) * 3.0 for k, v in _make_deltas().items()}

        b1_raw = serialize_weight_deltas(d1)
        b2_raw = serialize_weight_deltas(d2)

        # Compress then decompress (simulates wire transfer)
        b1 = decompress_gradients(compress_gradients(b1_raw))
        b2 = decompress_gradients(compress_gradients(b2_raw))

        result = aggregate_gradients([(b1, 50), (b2, 50)])
        for name in d1:
            expected = 2.0  # (1*0.5 + 3*0.5)
            np.testing.assert_allclose(result[name], expected, rtol=1e-3)


class TestApplyGradients:
    def test_apply_gradients(self):
        weights = {
            "hidden_weight": np.ones((128, 784), dtype=np.float32),
            "hidden_bias": np.zeros(128, dtype=np.float32),
        }
        grads = {
            "hidden_weight": np.full(128 * 784, 0.5, dtype=np.float32),
            "hidden_bias": np.full(128, 0.1, dtype=np.float32),
        }
        result = apply_gradients(weights, grads)

        np.testing.assert_allclose(result["hidden_weight"], 1.5)
        np.testing.assert_allclose(result["hidden_bias"], 0.1)

    def test_apply_gradients_missing_layer(self):
        weights = {
            "hidden_weight": np.ones((128, 784), dtype=np.float32),
            "output_weight": np.ones((10, 128), dtype=np.float32),
        }
        grads = {
            "hidden_weight": np.full(128 * 784, 0.5, dtype=np.float32),
        }
        result = apply_gradients(weights, grads)

        np.testing.assert_allclose(result["hidden_weight"], 1.5)
        np.testing.assert_allclose(result["output_weight"], 1.0)  # unchanged
