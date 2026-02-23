import CoreML
import Foundation

/// Loads MNIST training data from the bundled `mnist_train.bin` file.
///
/// Binary format:
///   [count: uint32_le]
///   For each sample:
///     [label: uint8]
///     [pixels: float32_le × 784]
public final class MNISTDataProvider: @unchecked Sendable {

    public let batchProvider: MLArrayBatchProvider
    public let sampleCount: Int

    public let testBatchProvider: MLArrayBatchProvider
    public let testSampleCount: Int
    /// Labels for test samples, used for accuracy computation.
    public let testLabels: [Int]

    public convenience init() throws {
        try self.init(partitionIndex: 0, partitionTotal: 1)
    }

    public init(partitionIndex: Int, partitionTotal: Int, trainRatio: Double = 0.9) throws {
        guard let url = Bundle.module.url(forResource: "mnist_train", withExtension: "bin") else {
            throw MNISTError.resourceNotFound
        }

        let data = try Data(contentsOf: url)
        guard data.count >= 4 else {
            throw MNISTError.invalidFormat("File too small")
        }

        let count: UInt32 = data.withUnsafeBytes { $0.load(as: UInt32.self) }
        let totalSamples = Int(count)
        let expectedSize = 4 + totalSamples * (1 + 784 * 4)
        guard data.count >= expectedSize else {
            throw MNISTError.invalidFormat("Expected \(expectedSize) bytes, got \(data.count)")
        }

        // Parse all samples
        var allFeatures: [MNISTFeatureProvider] = []
        allFeatures.reserveCapacity(totalSamples)

        var offset = 4
        for _ in 0..<totalSamples {
            let label = Int(data[offset])
            offset += 1

            let imageArray = try MLMultiArray(shape: [1, 28, 28], dataType: .float32)
            data.withUnsafeBytes { rawBuffer in
                let floatPtr = rawBuffer.baseAddress!.advanced(by: offset)
                    .assumingMemoryBound(to: Float.self)
                let dst = imageArray.dataPointer.bindMemory(to: Float.self, capacity: 784)
                dst.update(from: floatPtr, count: 784)
            }
            offset += 784 * 4

            let provider = MNISTFeatureProvider(image: imageArray, label: label)
            allFeatures.append(provider)
        }

        // Deterministic shuffle (seed=42, matching prepare_mnist.py)
        var rng = SeededRNG(seed: 42)
        allFeatures.shuffle(using: &rng)

        // Partition slice
        let start = (totalSamples * partitionIndex) / partitionTotal
        let end = (totalSamples * (partitionIndex + 1)) / partitionTotal
        let partitionFeatures = Array(allFeatures[start..<end])

        // Train/test split
        let trainEnd = Int(Double(partitionFeatures.count) * trainRatio)
        let trainFeatures = Array(partitionFeatures[..<trainEnd])
        let testFeatures = Array(partitionFeatures[trainEnd...])

        self.batchProvider = MLArrayBatchProvider(array: trainFeatures)
        self.sampleCount = trainFeatures.count
        self.testBatchProvider = MLArrayBatchProvider(array: testFeatures)
        self.testSampleCount = testFeatures.count
        self.testLabels = testFeatures.map { $0.label }
    }
}

enum MNISTError: Error, CustomStringConvertible {
    case resourceNotFound
    case invalidFormat(String)

    var description: String {
        switch self {
        case .resourceNotFound:
            return "mnist_train.bin not found in bundle"
        case .invalidFormat(let detail):
            return "Invalid MNIST format: \(detail)"
        }
    }
}

/// A simple seeded random number generator for deterministic shuffling.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // SplitMix64
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}

/// A single MNIST sample as an `MLFeatureProvider`.
final class MNISTFeatureProvider: MLFeatureProvider, @unchecked Sendable {
    let image: MLMultiArray
    let label: Int

    var featureNames: Set<String> { ["image", "labelProbs_true"] }

    init(image: MLMultiArray, label: Int) {
        self.image = image
        self.label = label
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "image":
            return MLFeatureValue(multiArray: image)
        case "labelProbs_true":
            // CoreML categorical cross-entropy expects target as MLMultiArray [1] Int32
            let labelArray = try? MLMultiArray(shape: [1], dataType: .int32)
            labelArray?[0] = NSNumber(value: Int32(label))
            if let arr = labelArray {
                return MLFeatureValue(multiArray: arr)
            }
            return nil
        default:
            return nil
        }
    }
}
