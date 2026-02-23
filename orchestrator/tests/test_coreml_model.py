"""Tests for CoreML model creation, weight extraction, and injection."""

import numpy as np
import pytest

from orchestrator.services.coreml_model import (
    LAYER_SHAPES,
    create_updatable_mlmodel,
    create_updatable_mlmodel_for_architecture,
    extract_weights,
    inject_weights,
)
from orchestrator.services.model_registry import get_architecture


class TestCoreMLModel:
    def test_create_model_returns_bytes(self):
        model_bytes = create_updatable_mlmodel()
        assert isinstance(model_bytes, bytes)
        assert len(model_bytes) > 0

    def test_extract_weights_shape(self):
        model_bytes = create_updatable_mlmodel()
        weights = extract_weights(model_bytes)

        assert set(weights.keys()) == set(LAYER_SHAPES.keys())
        for name, shape in LAYER_SHAPES.items():
            assert weights[name].shape == shape, f"{name}: expected {shape}, got {weights[name].shape}"

    def test_extract_inject_roundtrip(self):
        model_bytes = create_updatable_mlmodel()
        original = extract_weights(model_bytes)

        injected_bytes = inject_weights(model_bytes, original)
        recovered = extract_weights(injected_bytes)

        for name in original:
            np.testing.assert_allclose(recovered[name], original[name], rtol=0)

    def test_inject_custom_weights(self):
        model_bytes = create_updatable_mlmodel()

        custom = {
            "hidden_weight": np.full((128, 784), 0.01, dtype=np.float32),
            "hidden_bias": np.full(128, 0.02, dtype=np.float32),
            "output_weight": np.full((10, 128), 0.03, dtype=np.float32),
            "output_bias": np.full(10, 0.04, dtype=np.float32),
        }

        new_bytes = inject_weights(model_bytes, custom)
        recovered = extract_weights(new_bytes)

        for name in custom:
            np.testing.assert_allclose(recovered[name], custom[name], rtol=1e-6)

    def test_create_model_with_provided_weights(self):
        custom = {
            "hidden_weight": np.full((128, 784), 0.05, dtype=np.float32),
            "hidden_bias": np.full(128, 0.06, dtype=np.float32),
            "output_weight": np.full((10, 128), 0.07, dtype=np.float32),
            "output_bias": np.full(10, 0.08, dtype=np.float32),
        }

        model_bytes = create_updatable_mlmodel(weights=custom)
        recovered = extract_weights(model_bytes)

        for name in custom:
            np.testing.assert_allclose(recovered[name], custom[name], rtol=1e-6)

    def test_create_cifar10_model(self):
        arch = get_architecture("cifar10")
        model_bytes = create_updatable_mlmodel_for_architecture(arch)
        assert isinstance(model_bytes, bytes)
        assert len(model_bytes) > 0

        weights = extract_weights(model_bytes)
        for name, shape in arch.layer_shapes.items():
            assert name in weights, f"Missing layer {name}"
            assert weights[name].shape == shape, f"{name}: expected {shape}, got {weights[name].shape}"

    def test_cifar10_inject_roundtrip(self):
        arch = get_architecture("cifar10")
        model_bytes = create_updatable_mlmodel_for_architecture(arch)
        original = extract_weights(model_bytes)

        injected = inject_weights(model_bytes, original)
        recovered = extract_weights(injected)

        for name in original:
            np.testing.assert_allclose(recovered[name], original[name], rtol=0)
