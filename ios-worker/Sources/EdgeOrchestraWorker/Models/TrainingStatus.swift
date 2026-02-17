public enum TrainingStatus: String, Sendable {
    case idle
    case downloading
    case training
    case submitting
}

public struct TrainingRoundResult: Sendable {
    public let round: String
    public let loss: Float
    public let accuracy: Float
    public let numSamples: Int

    public init(round: String, loss: Float, accuracy: Float, numSamples: Int) {
        self.round = round
        self.loss = loss
        self.accuracy = accuracy
        self.numSamples = numSamples
    }
}
