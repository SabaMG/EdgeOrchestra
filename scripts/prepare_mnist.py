#!/usr/bin/env python3
"""Download MNIST and write a compact binary file for iOS CoreML training.

Output format:
  [count: uint32_le]
  For each sample:
    [label: uint8]
    [pixels: float32_le × 784]

Usage:
    python scripts/prepare_mnist.py [--samples 5000] [--output ios-worker/Sources/EdgeOrchestraWorker/Resources/mnist_train.bin]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def fetch_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST data using sklearn."""
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    return X, y


def main() -> None:
    default_output = (
        Path(__file__).resolve().parent.parent
        / "ios-worker"
        / "Sources"
        / "EdgeOrchestraWorker"
        / "Resources"
        / "mnist_train.bin"
    )

    parser = argparse.ArgumentParser(description="Prepare MNIST binary for iOS training")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to include")
    parser.add_argument("--output", type=Path, default=default_output, help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    print(f"Fetching MNIST dataset...")
    X, y = fetch_mnist()

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(X), size=min(args.samples, len(X)), replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    print(f"Selected {len(indices)} samples, label distribution:")
    for digit in range(10):
        count = np.sum(y_subset == digit)
        print(f"  {digit}: {count}")

    # Write binary file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", len(indices)))
        for i in range(len(indices)):
            f.write(struct.pack("B", y_subset[i]))
            f.write(X_subset[i].astype(np.float32).tobytes())

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MB, {len(indices)} samples)")


if __name__ == "__main__":
    main()
