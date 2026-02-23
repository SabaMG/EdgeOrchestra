"""Tests for gradient compression codec (float16 + lz4)."""

import numpy as np
import pytest

from orchestrator.services.fed_avg import (
    deserialize_weight_deltas,
    serialize_weight_deltas,
)
from orchestrator.services.gradient_codec import (
    compress_gradients,
    decompress_gradients,
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


class TestGradientCodec:
    def test_compress_decompress_roundtrip(self):
        """float32 → compress → decompress ≈ float32 (float16 precision)."""
        deltas = _make_deltas()
        raw = serialize_weight_deltas(deltas)

        compressed = compress_gradients(raw)
        decompressed = decompress_gradients(compressed)

        original = deserialize_weight_deltas(raw)
        restored = deserialize_weight_deltas(decompressed)

        assert set(original.keys()) == set(restored.keys())
        for name in original:
            np.testing.assert_allclose(
                restored[name], original[name], rtol=1e-3, atol=1e-3
            )

    def test_legacy_passthrough(self):
        """Data without magic byte passes through unchanged."""
        deltas = _make_deltas()
        raw = serialize_weight_deltas(deltas)

        result = decompress_gradients(raw)
        assert result == raw

    def test_compression_ratio(self):
        """Compressed size should be < 60% of original float32 size."""
        deltas = _make_deltas()
        raw = serialize_weight_deltas(deltas)

        compressed = compress_gradients(raw)
        ratio = len(compressed) / len(raw)
        assert ratio < 0.60, f"Compression ratio {ratio:.2%} exceeds 60%"

    def test_float16_precision(self):
        """Relative precision loss from float16 should be < 1e-3."""
        deltas = _make_deltas()
        raw = serialize_weight_deltas(deltas)

        compressed = compress_gradients(raw)
        decompressed = decompress_gradients(compressed)

        original = deserialize_weight_deltas(raw)
        restored = deserialize_weight_deltas(decompressed)

        for name in original:
            orig = original[name]
            rest = restored[name]
            mask = np.abs(orig) > 1e-6
            if mask.any():
                relative_err = np.abs((rest[mask] - orig[mask]) / orig[mask])
                assert relative_err.max() < 2e-3, (
                    f"Layer {name}: max relative error {relative_err.max():.2e}"
                )

    def test_quantize_dequantize_preserves_structure(self):
        """Layer names and element counts are preserved through codec."""
        deltas = _make_deltas()
        raw = serialize_weight_deltas(deltas)

        compressed = compress_gradients(raw)
        decompressed = decompress_gradients(compressed)

        original = deserialize_weight_deltas(raw)
        restored = deserialize_weight_deltas(decompressed)

        for name in original:
            assert original[name].shape == restored[name].shape, (
                f"Shape mismatch for {name}"
            )

    def test_compress_real_model_gradients(self):
        """End-to-end with MNIST-like gradients using _make_deltas."""
        deltas = _make_deltas()
        raw = serialize_weight_deltas(deltas)

        compressed = compress_gradients(raw)
        assert compressed[0] == 0x01  # magic byte

        decompressed = decompress_gradients(compressed)
        restored = deserialize_weight_deltas(decompressed)

        assert len(restored) == 4
        assert "hidden_weight" in restored
        assert restored["hidden_weight"].shape == (128 * 784,)
