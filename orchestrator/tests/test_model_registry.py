"""Tests for the model architecture registry."""

import pytest

from orchestrator.services.model_registry import get_architecture, list_architectures


class TestModelRegistry:
    def test_get_mnist_architecture(self):
        arch = get_architecture("mnist")
        assert arch.key == "mnist"
        assert arch.input_shape == (1, 28, 28)
        assert arch.num_classes == 10
        assert "hidden_weight" in arch.layer_names
        assert arch.layer_shapes["hidden_weight"] == (128, 784)

    def test_get_cifar10_architecture(self):
        arch = get_architecture("cifar10")
        assert arch.key == "cifar10"
        assert arch.input_shape == (3, 32, 32)
        assert arch.num_classes == 10
        assert "hidden1_weight" in arch.layer_names
        assert arch.layer_shapes["hidden1_weight"] == (256, 3072)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_architecture("resnet50")

    def test_list_architectures(self):
        archs = list_architectures()
        assert len(archs) >= 2
        keys = [a.key for a in archs]
        assert "mnist" in keys
        assert "cifar10" in keys
