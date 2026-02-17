import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import EdgeOrchestraProtos

public struct HeartbeatCommand: Sendable {
    public let type: String
    public let parameters: [String: String]
}

public struct HeartbeatClientWrapper: Sendable {

    private let client: GRPCServiceClient
    private let deviceId: String
    private let metricsCollector: MetricsCollector

    public init(client: GRPCServiceClient, deviceId: String) {
        self.client = client
        self.deviceId = deviceId
        self.metricsCollector = MetricsCollector()
    }

    public func run(
        interval: TimeInterval = 5.0,
        onMetrics: @escaping @Sendable (Edgeorchestra_V1_DeviceMetrics) -> Void,
        onCommand: @escaping @Sendable (HeartbeatCommand) -> Void
    ) async throws {
        let stub = Edgeorchestra_V1_HeartbeatService.Client(wrapping: client)

        try await stub.heartbeat(
            requestProducer: { requestStream in
                // Send heartbeats at interval
                var sequence: UInt64 = 0
                while !Task.isCancelled {
                    sequence += 1
                    var request = Edgeorchestra_V1_HeartbeatRequest()
                    request.deviceID.value = deviceId
                    let metrics = metricsCollector.collect()
                    request.metrics = metrics
                    request.sequence = sequence
                    onMetrics(metrics)
                    try await requestStream.write(request)
                    try await Task.sleep(for: .seconds(interval))
                }
            },
            onResponse: { responseStream in
                // Read commands from server
                for try await response in responseStream.messages {
                    let cmd = mapCommand(response)
                    if let cmd { onCommand(cmd) }
                }
            }
        )
    }

    private func mapCommand(_ response: Edgeorchestra_V1_HeartbeatResponse) -> HeartbeatCommand? {
        switch response.command {
        case .unspecified, .ack, .UNRECOGNIZED:
            return nil
        case .updateInterval:
            return HeartbeatCommand(type: "update_interval", parameters: Dictionary(uniqueKeysWithValues: response.parameters.map { ($0.key, $0.value) }))
        case .startTraining:
            return HeartbeatCommand(type: "start_training", parameters: Dictionary(uniqueKeysWithValues: response.parameters.map { ($0.key, $0.value) }))
        case .stopTraining:
            return HeartbeatCommand(type: "stop_training", parameters: [:])
        case .shutdown:
            return HeartbeatCommand(type: "shutdown", parameters: [:])
        }
    }
}
