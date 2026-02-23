"""Server-side model evaluation on held-out test data.

Performs a pure-numpy forward pass through the federated model
and computes accuracy + cross-entropy loss on a held-out test set.

Supports MNIST and CIFAR-10 architectures.
"""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()


class ServerEvaluator:
    """Evaluates model weights on a cached test set (MNIST or CIFAR-10)."""

    _instance: ServerEvaluator | None = None

    def __init__(self) -> None:
        self._datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def get_instance(cls) -> ServerEvaluator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_mnist(self) -> None:
        if "mnist" in self._datasets:
            return
        from sklearn.datasets import fetch_openml

        logger.info("server_evaluator_loading_mnist_test_data")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        self._datasets["mnist"] = (X[-2000:], y[-2000:])
        logger.info("server_evaluator_mnist_loaded", samples=2000)

    def _load_cifar10(self) -> None:
        if "cifar10" in self._datasets:
            return

        cache_dir = Path.home() / ".cache" / "edgeorchestra"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "cifar-10-python.tar.gz"

        if not cache_path.exists():
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            logger.info("server_evaluator_downloading_cifar10", url=url)
            urllib.request.urlretrieve(url, cache_path)

        logger.info("server_evaluator_loading_cifar10_test_data")
        # Load the test batch
        with tarfile.open(cache_path, "r:gz") as tar:
            for member in tar.getmembers():
                if "test_batch" in member.name:
                    f = tar.extractfile(member)
                    batch = pickle.load(f, encoding="bytes")
                    X = batch[b"data"].astype(np.float32) / 255.0
                    y = np.array(batch[b"labels"], dtype=np.int32)
                    # Use first 2000 as held-out test set
                    self._datasets["cifar10"] = (X[:2000], y[:2000])
                    logger.info("server_evaluator_cifar10_loaded", samples=2000)
                    return

        raise RuntimeError("Could not find test_batch in CIFAR-10 archive")

    def evaluate(
        self, weights: dict[str, np.ndarray], architecture: str = "mnist",
    ) -> tuple[float, float]:
        """Run forward pass and return (loss, accuracy)."""
        if architecture == "cifar10":
            return self._evaluate_cifar10(weights)
        return self._evaluate_mnist(weights)

    def _evaluate_mnist(self, weights: dict[str, np.ndarray]) -> tuple[float, float]:
        self._load_mnist()
        X_test, y_test = self._datasets["mnist"]

        H = np.maximum(0, X_test @ weights["hidden_weight"].T + weights["hidden_bias"])
        logits = H @ weights["output_weight"].T + weights["output_bias"]
        return self._compute_metrics(logits, y_test)

    def _evaluate_cifar10(self, weights: dict[str, np.ndarray]) -> tuple[float, float]:
        self._load_cifar10()
        X_test, y_test = self._datasets["cifar10"]

        H1 = np.maximum(0, X_test @ weights["hidden1_weight"].T + weights["hidden1_bias"])
        H2 = np.maximum(0, H1 @ weights["hidden2_weight"].T + weights["hidden2_bias"])
        logits = H2 @ weights["output_weight"].T + weights["output_bias"]
        return self._compute_metrics(logits, y_test)

    @staticmethod
    def _compute_metrics(logits: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        n = len(y_test)
        correct_probs = probs[np.arange(n), y_test]
        loss = float(-np.log(np.clip(correct_probs, 1e-12, 1.0)).mean())

        preds = np.argmax(logits, axis=1)
        accuracy = float((preds == y_test).mean())

        return loss, accuracy
