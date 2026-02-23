import CoreML

/// Protocol for dataset providers used in federated training.
public protocol TrainingDataProvider: Sendable {
    var batchProvider: MLArrayBatchProvider { get }
    var sampleCount: Int { get }
    var testBatchProvider: MLArrayBatchProvider { get }
    var testSampleCount: Int { get }
    var testLabels: [Int] { get }
}

extension MNISTDataProvider: TrainingDataProvider {}
extension CIFAR10DataProvider: TrainingDataProvider {}
