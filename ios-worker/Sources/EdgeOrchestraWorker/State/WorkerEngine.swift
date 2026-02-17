import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2

public final class WorkerEngine: Sendable {

    public let state: WorkerState

    private let deviceInfoCollector = DeviceInfoCollector()
    private let discovery = BonjourDiscovery()

    @MainActor
    public init(state: WorkerState) {
        self.state = state
    }

    public func run() async {
        while !Task.isCancelled {
            do {
                try await runLifecycle()
            } catch {
                await state.log("ERROR", "Lifecycle error: \(error)")
                await MainActor.run { state.connectionStatus = .disconnected }
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    private func runLifecycle() async throws {
        // 1. Discover
        await MainActor.run { state.connectionStatus = .discovering }
        await state.log("INFO", "Discovering orchestrator via Bonjour...")

        guard let orchestrator = await discovery.discover(timeout: 30) else {
            await state.log("WARN", "No orchestrator found, retrying...")
            return
        }

        await MainActor.run { state.discoveredHost = "\(orchestrator.host):\(orchestrator.grpcPort)" }
        await state.log("INFO", "Found orchestrator at \(orchestrator.host):\(orchestrator.grpcPort) (v\(orchestrator.version))")

        // 2. Connect
        await MainActor.run { state.connectionStatus = .connecting }

        let transport = try HTTP2ClientTransport.Posix(
            target: .dns(host: orchestrator.host, port: orchestrator.grpcPort),
            transportSecurity: .plaintext
        )
        let client = GRPCClient(transport: transport)

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Run the gRPC client transport
            group.addTask {
                try await client.runConnections()
            }

            // Run the worker logic
            group.addTask {
                try await self.runWorker(client: client)
            }

            // Wait for first completion (error or shutdown)
            try await group.next()
            group.cancelAll()
        }
    }

    private func runWorker(client: GRPCServiceClient) async throws {
        // 3. Register
        let registryClient = DeviceRegistryClient(client: client)
        let (name, model, osVersion, caps) = deviceInfoCollector.collectDeviceInfo()
        let metricsCollector = MetricsCollector()
        let initialMetrics = metricsCollector.collect()

        await MainActor.run { state.deviceName = name }

        let deviceId = try await registryClient.register(
            name: name,
            deviceModel: model,
            osVersion: osVersion,
            capabilities: caps,
            initialMetrics: initialMetrics
        )
        await MainActor.run {
            state.deviceId = deviceId
            state.connectionStatus = .connected
        }
        await state.log("INFO", "Registered as \(deviceId)")

        defer {
            Task {
                try? await registryClient.unregister(deviceId: deviceId)
                await state.log("INFO", "Unregistered")
            }
        }

        // 4. Heartbeat + command handling
        let heartbeat = HeartbeatClientWrapper(client: client, deviceId: deviceId)
        let trainingCoordinator = TrainingCoordinator(client: client, deviceId: deviceId)
        let heartbeatInterval = await state.heartbeatInterval

        try await heartbeat.run(
            interval: heartbeatInterval,
            onMetrics: { [state] metrics in
                Task { @MainActor in
                    state.cpuUsage = metrics.cpuUsage
                    state.memoryUsage = metrics.memoryUsage
                    state.thermalPressure = metrics.thermalPressure
                    state.batteryLevel = metrics.battery.level
                    state.heartbeatSequence += 1
                }
            }
        ) { [state] command in
            Task { @MainActor in
                state.log("INFO", "Command: \(command.type)")

                switch command.type {
                case "start_training":
                    let jobId = command.parameters["job_id"] ?? ""
                    let modelId = command.parameters["model_id"] ?? ""
                    let round = command.parameters["round"] ?? "0"
                    state.trainingStatus = .downloading
                    state.currentRound = round

                    Task {
                        do {
                            await MainActor.run { state.trainingStatus = .training }
                            let result = try await trainingCoordinator.runRound(
                                jobId: jobId, modelId: modelId, round: round
                            )
                            await MainActor.run {
                                state.trainingHistory.append(result)
                                state.trainingStatus = .idle
                            }
                            await state.log("INFO", "Round \(round) done - loss=\(result.loss) acc=\(result.accuracy)")
                        } catch {
                            await MainActor.run { state.trainingStatus = .idle }
                            await state.log("ERROR", "Training failed: \(error)")
                        }
                    }

                case "stop_training":
                    state.trainingStatus = .idle

                case "update_interval":
                    if let interval = command.parameters["interval_seconds"], let val = Double(interval) {
                        state.heartbeatInterval = val
                    }

                default:
                    break
                }
            }
        }
    }
}
