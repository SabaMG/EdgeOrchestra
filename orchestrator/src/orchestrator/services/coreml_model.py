"""Create and manipulate updatable CoreML .mlmodel files for federated learning.

Supports multiple architectures via ModelArchitecture registry.

MNIST:   Flatten(1x28x28->784) -> Dense(784->128, ReLU) -> Dense(128->10, Softmax)
CIFAR10: Flatten(3x32x32->3072) -> Dense(3072->256, ReLU) -> Dense(256->128, ReLU) -> Dense(128->10, Softmax)

Updatable layers: all innerProduct layers
Loss: categorical cross-entropy, Optimizer: SGD
"""

from __future__ import annotations

import numpy as np

from orchestrator.services.model_registry import ARCHITECTURES, ModelArchitecture

LAYER_NAMES = ["hidden_weight", "hidden_bias", "output_weight", "output_bias"]
LAYER_SHAPES = {
    "hidden_weight": (128, 784),
    "hidden_bias": (128,),
    "output_weight": (10, 128),
    "output_bias": (10,),
}

# The softmax output layer name -- loss target will be "{SOFTMAX_OUTPUT}_true"
SOFTMAX_OUTPUT = "labelProbs"
LOSS_TARGET = f"{SOFTMAX_OUTPUT}_true"


def _init_weights(shape: tuple[int, ...], rng: np.random.RandomState) -> np.ndarray:
    """He initialization for weight matrices, zeros for biases."""
    if len(shape) == 1:
        return np.zeros(shape[0], dtype=np.float32)
    fan_in = shape[1]
    return (rng.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def create_updatable_mlmodel(weights: dict[str, np.ndarray] | None = None) -> bytes:
    """Create an updatable .mlmodel for MNIST classification (backward compat)."""
    return create_updatable_mlmodel_for_architecture(ARCHITECTURES["mnist"], weights=weights)


def create_updatable_mlmodel_for_architecture(
    arch: ModelArchitecture,
    weights: dict[str, np.ndarray] | None = None,
    learning_rate: float = 0.01,
) -> bytes:
    """Create an updatable .mlmodel for the given architecture."""
    if arch.key == "mnist":
        return _create_mnist_model(arch, weights, learning_rate=learning_rate)
    elif arch.key == "cifar10":
        return _create_cifar10_model(arch, weights, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported architecture: {arch.key}")


def _create_mnist_model(
    arch: ModelArchitecture, weights: dict[str, np.ndarray] | None = None,
    learning_rate: float = 0.01,
) -> bytes:
    """MNIST: Flatten(1x28x28->784) -> Dense(784->128, ReLU) -> Dense(128->10, Softmax)"""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

    input_features = [("image", ct.models.datatypes.Array(*arch.input_shape))]
    output_features = [
        (SOFTMAX_OUTPUT, ct.models.datatypes.Dictionary(ct.models.datatypes.Int64())),
    ]

    builder = NeuralNetworkBuilder(input_features, output_features, mode="classifier")

    if weights is None:
        rng = np.random.RandomState(0)
        w = {name: _init_weights(shape, rng) for name, shape in arch.layer_shapes.items()}
    else:
        w = {name: weights[name].astype(np.float32) for name in arch.layer_names}

    builder.add_flatten(name="flatten", mode=0, input_name="image", output_name="flatten_out")

    builder.add_inner_product(
        name="hidden", W=w["hidden_weight"], b=w["hidden_bias"],
        input_channels=784, output_channels=128, has_bias=True,
        input_name="flatten_out", output_name="hidden_out",
    )
    builder.add_activation(
        name="relu", non_linearity="RELU", input_name="hidden_out", output_name="relu_out",
    )

    builder.add_inner_product(
        name="output", W=w["output_weight"], b=w["output_bias"],
        input_channels=128, output_channels=10, has_bias=True,
        input_name="relu_out", output_name="output_presoftmax",
    )
    builder.add_softmax(name="softmax", input_name="output_presoftmax", output_name=SOFTMAX_OUTPUT)

    builder.make_updatable(["hidden", "output"])
    builder.set_categorical_cross_entropy_loss(name="loss", input=SOFTMAX_OUTPUT)
    builder.set_sgd_optimizer(SgdParams(lr=learning_rate, batch=32, momentum=0))
    builder.set_epochs(5)

    spec = builder.spec
    spec.description.metadata.author = "EdgeOrchestra"
    spec.description.metadata.shortDescription = "Updatable MNIST classifier for federated learning"

    nn = spec.neuralNetworkClassifier
    nn.int64ClassLabels.vector.extend(range(arch.num_classes))

    class_output = spec.description.output.add()
    class_output.name = "classLabel"
    class_output.type.int64Type.MergeFromString(b"")

    spec.description.predictedFeatureName = "classLabel"
    spec.description.predictedProbabilitiesName = SOFTMAX_OUTPUT

    return spec.SerializeToString()


def _create_cifar10_model(
    arch: ModelArchitecture, weights: dict[str, np.ndarray] | None = None,
    learning_rate: float = 0.01,
) -> bytes:
    """CIFAR-10: Flatten(3x32x32->3072) -> Dense(3072->256, ReLU) -> Dense(256->128, ReLU) -> Dense(128->10, Softmax)"""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

    input_features = [("image", ct.models.datatypes.Array(*arch.input_shape))]
    output_features = [
        (SOFTMAX_OUTPUT, ct.models.datatypes.Dictionary(ct.models.datatypes.Int64())),
    ]

    builder = NeuralNetworkBuilder(input_features, output_features, mode="classifier")

    if weights is None:
        rng = np.random.RandomState(0)
        w = {name: _init_weights(shape, rng) for name, shape in arch.layer_shapes.items()}
    else:
        w = {name: weights[name].astype(np.float32) for name in arch.layer_names}

    builder.add_flatten(name="flatten", mode=0, input_name="image", output_name="flatten_out")

    # Hidden1: 3072 -> 256
    builder.add_inner_product(
        name="hidden1", W=w["hidden1_weight"], b=w["hidden1_bias"],
        input_channels=3072, output_channels=256, has_bias=True,
        input_name="flatten_out", output_name="hidden1_out",
    )
    builder.add_activation(
        name="relu1", non_linearity="RELU", input_name="hidden1_out", output_name="relu1_out",
    )

    # Hidden2: 256 -> 128
    builder.add_inner_product(
        name="hidden2", W=w["hidden2_weight"], b=w["hidden2_bias"],
        input_channels=256, output_channels=128, has_bias=True,
        input_name="relu1_out", output_name="hidden2_out",
    )
    builder.add_activation(
        name="relu2", non_linearity="RELU", input_name="hidden2_out", output_name="relu2_out",
    )

    # Output: 128 -> 10
    builder.add_inner_product(
        name="output", W=w["output_weight"], b=w["output_bias"],
        input_channels=128, output_channels=10, has_bias=True,
        input_name="relu2_out", output_name="output_presoftmax",
    )
    builder.add_softmax(name="softmax", input_name="output_presoftmax", output_name=SOFTMAX_OUTPUT)

    builder.make_updatable(["hidden1", "hidden2", "output"])
    builder.set_categorical_cross_entropy_loss(name="loss", input=SOFTMAX_OUTPUT)
    builder.set_sgd_optimizer(SgdParams(lr=learning_rate, batch=32, momentum=0))
    builder.set_epochs(5)

    spec = builder.spec
    spec.description.metadata.author = "EdgeOrchestra"
    spec.description.metadata.shortDescription = "Updatable CIFAR-10 classifier for federated learning"

    nn = spec.neuralNetworkClassifier
    nn.int64ClassLabels.vector.extend(range(arch.num_classes))

    class_output = spec.description.output.add()
    class_output.name = "classLabel"
    class_output.type.int64Type.MergeFromString(b"")

    spec.description.predictedFeatureName = "classLabel"
    spec.description.predictedProbabilitiesName = SOFTMAX_OUTPUT

    return spec.SerializeToString()


def set_learning_rate(mlmodel_bytes: bytes, lr: float) -> bytes:
    """Modify the SGD learning rate in an existing .mlmodel protobuf."""
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString(mlmodel_bytes)

    nn = _get_nn(spec)
    if nn.updateParams.HasField("optimizer"):
        sgd = nn.updateParams.optimizer.sgdOptimizer
        sgd.learningRate.defaultValue = lr
        sgd.learningRate.range.minValue = lr
        sgd.learningRate.range.maxValue = lr

    return spec.SerializeToString()


def _get_nn(spec):
    """Get the neural network layers from the spec, regardless of type."""
    if spec.HasField("neuralNetworkClassifier"):
        return spec.neuralNetworkClassifier
    return spec.neuralNetwork


def extract_weights(mlmodel_bytes: bytes) -> dict[str, np.ndarray]:
    """Extract weight arrays from .mlmodel protobuf bytes.

    Dynamically reads all innerProduct layers instead of hardcoding names.
    """
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString(mlmodel_bytes)

    weights = {}
    nn = _get_nn(spec)

    for layer in nn.layers:
        if layer.HasField("innerProduct"):
            ip = layer.innerProduct
            out_ch = ip.outputChannels
            in_ch = ip.inputChannels
            weights[f"{layer.name}_weight"] = np.array(
                ip.weights.floatValue, dtype=np.float32
            ).reshape(out_ch, in_ch)
            weights[f"{layer.name}_bias"] = np.array(
                ip.bias.floatValue, dtype=np.float32
            )

    return weights


def inject_weights(mlmodel_bytes: bytes, weights: dict[str, np.ndarray]) -> bytes:
    """Replace weights in .mlmodel protobuf bytes and return new bytes."""
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString(mlmodel_bytes)

    nn = _get_nn(spec)
    for layer in nn.layers:
        if layer.HasField("innerProduct"):
            w_key = f"{layer.name}_weight"
            b_key = f"{layer.name}_bias"
            if w_key in weights:
                ip = layer.innerProduct
                del ip.weights.floatValue[:]
                ip.weights.floatValue.extend(
                    weights[w_key].astype(np.float32).flatten().tolist()
                )
            if b_key in weights:
                ip = layer.innerProduct
                del ip.bias.floatValue[:]
                ip.bias.floatValue.extend(
                    weights[b_key].astype(np.float32).flatten().tolist()
                )

    return spec.SerializeToString()
