import GRPCCore
import GRPCNIOTransportHTTP2
import EdgeOrchestraProtos

public struct DeviceRegistryClient: Sendable {

    private let client: GRPCServiceClient

    public init(client: GRPCServiceClient) {
        self.client = client
    }

    public func register(
        name: String,
        deviceModel: String,
        osVersion: String,
        capabilities: Edgeorchestra_V1_DeviceCapabilities,
        initialMetrics: Edgeorchestra_V1_DeviceMetrics
    ) async throws -> String {
        let stub = Edgeorchestra_V1_DeviceRegistry.Client(wrapping: client)

        var request = Edgeorchestra_V1_RegisterRequest()
        request.name = name
        request.deviceModel = deviceModel
        request.osVersion = osVersion
        request.capabilities = capabilities
        request.initialMetrics = initialMetrics

        let response = try await stub.register(request)
        return response.deviceID.value
    }

    public func unregister(deviceId: String) async throws {
        let stub = Edgeorchestra_V1_DeviceRegistry.Client(wrapping: client)

        var request = Edgeorchestra_V1_UnregisterRequest()
        request.deviceID.value = deviceId

        _ = try await stub.unregister(request)
    }
}
