import GRPCCore
import GRPCNIOTransportHTTP2

public typealias GRPCServiceClient = GRPCClient<HTTP2ClientTransport.Posix>

public final class OrchestratorConnection: Sendable {

    public let host: String
    public let port: Int

    private let transport: HTTP2ClientTransport.Posix

    public init(host: String, port: Int) throws {
        self.host = host
        self.port = port
        self.transport = try HTTP2ClientTransport.Posix(
            target: .dns(host: host, port: port),
            transportSecurity: .plaintext
        )
    }

    public var grpcClient: GRPCServiceClient {
        GRPCServiceClient(transport: transport)
    }
}
