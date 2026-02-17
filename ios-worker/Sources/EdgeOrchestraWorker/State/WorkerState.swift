import Foundation

public struct LogEntry: Identifiable, Sendable {
    public let id = UUID()
    public let timestamp: Date
    public let level: String
    public let message: String

    public init(level: String, message: String) {
        self.timestamp = Date()
        self.level = level
        self.message = message
    }
}

@Observable
@MainActor
public final class WorkerState {
    public var connectionStatus: ConnectionStatus = .disconnected
    public var discoveredHost: String?
    public var deviceId: String?
    public var deviceName: String = ""

    public var cpuUsage: Float = 0
    public var memoryUsage: Float = 0
    public var thermalPressure: Float = 0
    public var batteryLevel: Float = 1.0
    public var heartbeatSequence: UInt64 = 0
    public var heartbeatInterval: TimeInterval = 5.0

    public var trainingStatus: TrainingStatus = .idle
    public var currentRound: String?
    public var trainingHistory: [TrainingRoundResult] = []

    public var logEntries: [LogEntry] = []

    public init() {}

    public func log(_ level: String, _ message: String) {
        let entry = LogEntry(level: level, message: message)
        logEntries.append(entry)
        if logEntries.count > 200 {
            logEntries.removeFirst(logEntries.count - 200)
        }
        print("[\(level)] \(message)")
    }
}
