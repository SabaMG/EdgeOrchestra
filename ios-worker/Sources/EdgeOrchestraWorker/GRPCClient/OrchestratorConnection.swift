import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2

public typealias GRPCServiceClient = GRPCClient<HTTP2ClientTransport.Posix>

public struct TLSClientConfig: Sendable {
    public let caCert: Data
    public let clientCert: Data
    public let clientKey: Data

    public init(caCert: Data, clientCert: Data, clientKey: Data) {
        self.caCert = caCert
        self.clientCert = clientCert
        self.clientKey = clientKey
    }
}

public final class OrchestratorConnection: Sendable {

    public let host: String
    public let port: Int
    public let transport: HTTP2ClientTransport.Posix

    public init(host: String, port: Int, tlsConfig: TLSClientConfig? = nil) throws {
        self.host = host
        self.port = port

        if let tls = tlsConfig {
            self.transport = try HTTP2ClientTransport.Posix(
                target: .dns(host: host, port: port),
                transportSecurity: .mTLS(
                    certificateChain: [.bytes([UInt8](tls.clientCert), format: .pem)],
                    privateKey: .bytes([UInt8](tls.clientKey), format: .pem)
                ) { config in
                    config.trustRoots = .certificates([.bytes([UInt8](tls.caCert), format: .pem)])
                    config.serverCertificateVerification = .noHostnameVerification
                }
            )
        } else {
            self.transport = try HTTP2ClientTransport.Posix(
                target: .dns(host: host, port: port),
                transportSecurity: .plaintext
            )
        }
    }

    public var grpcClient: GRPCServiceClient {
        GRPCServiceClient(transport: transport)
    }
}
