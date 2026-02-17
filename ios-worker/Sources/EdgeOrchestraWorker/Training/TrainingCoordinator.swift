import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2

public final class TrainingCoordinator: Sendable {

    private let modelClient: ModelClientWrapper
    private let trainer: LocalTrainer
    private let deviceId: String

    public init(client: GRPCServiceClient, deviceId: String) {
        self.modelClient = ModelClientWrapper(client: client)
        self.trainer = LocalTrainer()
        self.deviceId = deviceId
    }

    public func runRound(
        jobId: String,
        modelId: String,
        round: String
    ) async throws -> TrainingRoundResult {
        // 1. Download model
        let modelData = try await modelClient.downloadModel(modelId: modelId, deviceId: deviceId)

        // 2. Train locally
        let (gradients, numSamples, metrics) = await trainer.train(modelWeights: modelData)

        // 3. Submit gradients
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
