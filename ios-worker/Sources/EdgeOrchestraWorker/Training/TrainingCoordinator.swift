import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2

public final class TrainingCoordinator: Sendable {

    private let modelClient: ModelClientWrapper
    private let deviceId: String

    public init(client: GRPCServiceClient, deviceId: String) throws {
        self.modelClient = ModelClientWrapper(client: client)
        self.deviceId = deviceId
    }

    public func runRound(
        jobId: String,
        modelId: String,
        round: String,
        partitionIndex: Int,
        partitionTotal: Int,
        architecture: String = "mnist"
    ) async throws -> TrainingRoundResult {
        // 1. Download CoreML .mlmodel
        let modelData = try await modelClient.downloadModel(modelId: modelId, deviceId: deviceId)

        // 2. Create data provider and trainer based on architecture
        let trainer: CoreMLTrainer
        switch architecture {
        case "cifar10":
            let dataProvider = try CIFAR10DataProvider(partitionIndex: partitionIndex, partitionTotal: partitionTotal)
            trainer = CoreMLTrainer(
                trainBatchProvider: dataProvider.batchProvider,
                trainSampleCount: dataProvider.sampleCount,
                testBatchProvider: dataProvider.testBatchProvider,
                testLabels: dataProvider.testLabels,
                layerMappings: [
                    ("hidden1_weight", "hidden1", .weights),
                    ("hidden1_bias", "hidden1", .biases),
                    ("hidden2_weight", "hidden2", .weights),
                    ("hidden2_bias", "hidden2", .biases),
                    ("output_weight", "output", .weights),
                    ("output_bias", "output", .biases),
                ],
                layerOrder: [
                    "hidden1_weight", "hidden1_bias",
                    "hidden2_weight", "hidden2_bias",
                    "output_weight", "output_bias",
                ]
            )
        default:
            let dataProvider = try MNISTDataProvider(partitionIndex: partitionIndex, partitionTotal: partitionTotal)
            trainer = CoreMLTrainer(dataProvider: dataProvider)
        }

        let (gradients, numSamples, metrics) = try await trainer.train(modelData: modelData)

        // 3. Submit weight deltas
        let accepted = try await modelClient.submitGradients(
            deviceId: deviceId,
            modelId: modelId,
            trainingRound: round,
            gradients: gradients,
            numSamples: UInt32(numSamples),
            metrics: metrics
        )

        return TrainingRoundResult(
            round: round,
            loss: metrics["loss"] ?? 0,
            accuracy: metrics["accuracy"] ?? 0,
            numSamples: numSamples
        )
    }
}
