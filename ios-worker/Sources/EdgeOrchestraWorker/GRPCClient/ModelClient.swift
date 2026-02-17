import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import EdgeOrchestraProtos

public struct ModelClientWrapper: Sendable {

    private let client: GRPCServiceClient

    public init(client: GRPCServiceClient) {
        self.client = client
    }

    public func downloadModel(modelId: String, deviceId: String) async throws -> Data {
        let stub = Edgeorchestra_V1_ModelService.Client(wrapping: client)

        var request = Edgeorchestra_V1_DownloadModelRequest()
        request.modelID = modelId
        request.deviceID.value = deviceId

        let modelData: Data = try await stub.downloadModel(request) { response in
            var data = Data()
            for try await chunk in response.messages {
                if case .chunk = chunk.data {
                    data.append(chunk.chunk)
                }
            }
            return data
        }
        return modelData
    }

    public func submitGradients(
        deviceId: String,
        modelId: String,
        trainingRound: String,
        gradients: Data,
        numSamples: UInt32,
        metrics: [String: Float]
    ) async throws -> Bool {
        let stub = Edgeorchestra_V1_ModelService.Client(wrapping: client)

        var request = Edgeorchestra_V1_SubmitGradientsRequest()
        request.deviceID.value = deviceId
        request.modelID = modelId
        request.trainingRound = trainingRound
        request.gradients = gradients
        request.numSamples = numSamples
        request.metrics = metrics

        let response = try await stub.submitGradients(request)
        return response.accepted
    }
}
