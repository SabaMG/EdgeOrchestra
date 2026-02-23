import Foundation

public enum TrainingStatus: String, Sendable {
    case idle
    case downloading
    case training
    case submitting
}

public struct TrainingRoundResult: Sendable, Identifiable {
    public var id: String { round }
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

public struct TrainingJob: Sendable, Identifiable {
    public let id: String
    public let jobId: String
    public let architecture: String
    public let startedAt: Date
    public var status: TrainingJobStatus
    public var rounds: [TrainingRoundResult]

    public var latestLoss: Float? { rounds.last?.loss }
    public var latestAccuracy: Float? { rounds.last?.accuracy }
    public var totalSamples: Int { rounds.reduce(0) { $0 + $1.numSamples } }

    public init(jobId: String, architecture: String) {
        self.id = jobId
        self.jobId = jobId
        self.architecture = architecture
        self.startedAt = Date()
        self.status = .running
        self.rounds = []
    }
}

public enum TrainingJobStatus: String, Sendable {
    case running
    case completed
    case failed
}
