#!/usr/bin/env python3
"""Download CIFAR-10 and write a compact binary file for iOS CoreML training.

Output format:
  [count: uint32_le]
  For each sample:
    [label: uint8]
    [pixels: float32_le x 3072]

Usage:
    python scripts/prepare_cifar10.py [--samples 5000] [--output ios-worker/Sources/EdgeOrchestraWorker/Resources/cifar10_train.bin]
"""

import argparse
import io
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def fetch_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """Fetch CIFAR-10 from the official source (no sklearn/torch needed)."""
    cache_dir = Path.home() / ".cache" / "edgeorchestra"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "cifar-10-python.tar.gz"

    if not cache_path.exists():
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
        urllib.request.urlretrieve(CIFAR10_URL, cache_path)

    all_data = []
    all_labels = []

    with tarfile.open(cache_path, "r:gz") as tar:
        for member in tar.getmembers():
            if "data_batch" in member.name:
                f = tar.extractfile(member)
                batch = pickle.load(f, encoding="bytes")
                all_data.append(batch[b"data"])
                all_labels.extend(batch[b"labels"])

    X = np.concatenate(all_data).astype(np.float32) / 255.0
    y = np.array(all_labels, dtype=np.int32)
    return X, y


def main() -> None:
    default_output = (
        Path(__file__).resolve().parent.parent
        / "ios-worker"
        / "Sources"
        / "EdgeOrchestraWorker"
        / "Resources"
        / "cifar10_train.bin"
    )

    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 binary for iOS training")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to include")
    parser.add_argument("--output", type=Path, default=default_output, help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    print("Fetching CIFAR-10 dataset...")
    X, y = fetch_cifar10()

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(X), size=min(args.samples, len(X)), replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    print(f"Selected {len(indices)} samples, label distribution:")
    for cls in range(10):
        count = np.sum(y_subset == cls)
        print(f"  {cls}: {count}")

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
