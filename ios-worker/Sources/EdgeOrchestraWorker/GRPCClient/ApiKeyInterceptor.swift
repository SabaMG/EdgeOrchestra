import GRPCCore

public struct ApiKeyInterceptor: ClientInterceptor {
    public let apiKey: String

    public init(apiKey: String) {
        self.apiKey = apiKey
    }

    public func intercept<Input: Sendable, Output: Sendable>(
        request: StreamingClientRequest<Input>,
        context: ClientContext,
        next: (StreamingClientRequest<Input>, ClientContext) async throws -> StreamingClientResponse<Output>
    ) async throws -> StreamingClientResponse<Output> {
        var request = request
        request.metadata.addString(apiKey, forKey: "x-api-key")
        return try await next(request, context)
    }
}
