import Foundation
import Accelerate

public struct LocalTrainer: Sendable {

    public static let modelSize = 7850 // 784*10 + 10

    public init() {}

    public func train(
        modelWeights: Data,
        numEpochs: Int = 1,
        numSamples: Int = 100
    ) async -> (gradients: Data, numSamples: Int, metrics: [String: Float]) {
        // Deserialize model weights
        let weights: [Double] = modelWeights.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Double.self))
        }

        // Simulate training delay
        let delay = Double.random(in: 1.0...3.0)
        try? await Task.sleep(for: .seconds(delay))

        // Generate gradients using vDSP
        var gradients = [Double](repeating: 0, count: Self.modelSize)
        for i in 0..<Self.modelSize {
            gradients[i] = Double.random(in: -0.01...0.01)
        }

        // Compute model norm for simulated metrics
        var sumOfSquares: Double = 0
        vDSP_svesqD(weights, 1, &sumOfSquares, vDSP_Length(weights.count))
        let modelNorm = sqrt(sumOfSquares)

        let baseLoss = 2.0 / (1.0 + modelNorm * 0.1)
        let loss = Float(max(0.01, baseLoss + Double.random(in: -0.1...0.1)))
        let accuracy = Float(min(0.95, max(0.0, 0.3 + modelNorm * 0.05 + Double.random(in: -0.05...0.05))))

        // Serialize gradients
        let gradientData = gradients.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }

        let metrics: [String: Float] = [
            "loss": loss,
            "accuracy": accuracy,
            "num_epochs": Float(numEpochs),
        ]

        return (gradientData, numSamples, metrics)
    }
}
