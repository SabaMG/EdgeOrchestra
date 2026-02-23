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
        // 1. Discover (or use manual address)
        let host: String
        let grpcPort: Int

        let manual = await MainActor.run { state.manualAddress }
        if let manual, !manual.isEmpty {
            // Manual address: "host:port"
            let parts = manual.split(separator: ":", maxSplits: 1)
            host = String(parts[0])
            grpcPort = parts.count > 1 ? Int(parts[1]) ?? 50051 : 50051
            await MainActor.run { state.discoveredHost = "\(host):\(grpcPort)" }
            await state.log("INFO", "Using manual address: \(host):\(grpcPort)")
        } else {
            await MainActor.run { state.connectionStatus = .discovering }
            await state.log("INFO", "Discovering orchestrator via Bonjour...")

            guard let orchestrator = await discovery.discover(timeout: 30) else {
                await state.log("WARN", "No orchestrator found, retrying...")
                return
            }

            host = orchestrator.host
            grpcPort = orchestrator.grpcPort
            await MainActor.run { state.discoveredHost = "\(host):\(grpcPort)" }
            await state.log("INFO", "Found orchestrator at \(host):\(grpcPort) (v\(orchestrator.version))")
        }

        // 2. Connect
        await MainActor.run { state.connectionStatus = .connecting }

        let tlsConfig: TLSClientConfig? = await MainActor.run { state.tlsEnabled ? state.certBundle : nil }
        let connection = try OrchestratorConnection(host: host, port: grpcPort, tlsConfig: tlsConfig)

        var interceptors: [any ClientInterceptor] = []
        let apiKey = await MainActor.run { state.apiKey }
        if let apiKey, !apiKey.isEmpty {
            interceptors.append(ApiKeyInterceptor(apiKey: apiKey))
        }
        let client = GRPCClient(transport: connection.transport, interceptors: interceptors)

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
        let trainingCoordinator = try TrainingCoordinator(client: client, deviceId: deviceId)
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
                // Update server metrics from heartbeat metadata
                if let accStr = command.metadata["server_accuracy"], let acc = Float(accStr) {
                    state.serverAccuracy = acc
                }
                if let lossStr = command.metadata["server_loss"], let loss = Float(lossStr) {
                    state.serverLoss = loss
                }

                guard command.type != "ack" else { return }
                state.log("INFO", "Command: \(command.type)")

                switch command.type {
                case "start_training":
                    let jobId = command.parameters["job_id"] ?? ""
                    let modelId = command.parameters["model_id"] ?? ""
                    let round = command.parameters["round"] ?? "0"
                    let partitionIndex = Int(command.parameters["partition_index"] ?? "0") ?? 0
                    let partitionTotal = Int(command.parameters["partition_total"] ?? "1") ?? 1
                    let architecture = command.parameters["architecture"] ?? "mnist"
                    state.trainingStatus = .downloading
                    state.currentRound = round

                    // Mark previous job as completed if a new one starts
                    if let prevId = state.activeJobId, prevId != jobId,
                       let idx = state.trainingJobs.firstIndex(where: { $0.jobId == prevId && $0.status == .running }) {
                        state.trainingJobs[idx].status = .completed
                    }
                    state.activeJobId = jobId

                    // Create job entry if it doesn't exist yet
                    if !state.trainingJobs.contains(where: { $0.jobId == jobId }) {
                        state.trainingJobs.insert(
                            TrainingJob(jobId: jobId, architecture: architecture),
                            at: 0
                        )
                    }

                    Task {
                        do {
                            await MainActor.run { state.trainingStatus = .training }
                            let result = try await trainingCoordinator.runRound(
                                jobId: jobId, modelId: modelId, round: round,
                                partitionIndex: partitionIndex, partitionTotal: partitionTotal,
                                architecture: architecture
                            )
                            await MainActor.run {
                                state.trainingHistory.append(result)
                                // Add round result to the job
                                if let idx = state.trainingJobs.firstIndex(where: { $0.jobId == jobId }) {
                                    state.trainingJobs[idx].rounds.append(result)
                                }
                                state.trainingStatus = .idle
                            }
                            await state.log("INFO", "Round \(round) done - loss=\(result.loss) acc=\(result.accuracy)")
                        } catch {
                            await MainActor.run {
                                state.trainingStatus = .idle
                                // Mark job as failed
                                if let idx = state.trainingJobs.firstIndex(where: { $0.jobId == jobId }) {
                                    state.trainingJobs[idx].status = .failed
                                }
                            }
                            await state.log("ERROR", "Training failed: \(error)")
                        }
                    }

                case "stop_training":
                    state.trainingStatus = .idle
                    // Mark the active job as completed
                    if let activeId = state.activeJobId,
                       let idx = state.trainingJobs.firstIndex(where: { $0.jobId == activeId }) {
                        state.trainingJobs[idx].status = .completed
                    }
                    state.activeJobId = nil

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
