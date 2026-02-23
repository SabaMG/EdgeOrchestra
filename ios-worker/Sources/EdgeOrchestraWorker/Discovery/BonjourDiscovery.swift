import Foundation
import Network

public struct DiscoveredOrchestrator: Sendable {
    public let host: String
    public let grpcPort: Int
    public let apiPort: Int
    public let version: String
    public let tlsEnabled: Bool
}

public final class BonjourDiscovery: Sendable {

    private let serviceType = "_edgeorchestra._tcp"
    private let domain = "local."

    public init() {}

    public func discover(timeout: TimeInterval = 10) async -> DiscoveredOrchestrator? {
        await withCheckedContinuation { continuation in
            let serialQueue = DispatchQueue(label: "bonjour.discover")
            let browser = NWBrowser(for: .bonjour(type: serviceType, domain: domain), using: .tcp)
            nonisolated(unsafe) var resumed = false

            let resumeOnce: @Sendable (DiscoveredOrchestrator?) -> Void = { value in
                serialQueue.async {
                    guard !resumed else { return }
                    resumed = true
                    browser.cancel()
                    continuation.resume(returning: value)
                }
            }

            browser.stateUpdateHandler = { state in
                if case .failed = state {
                    resumeOnce(nil)
                }
            }

            browser.browseResultsChangedHandler = { results, _ in
                serialQueue.async {
                    guard !resumed else { return }
                    for result in results {
                        if case .service = result.endpoint {
                            let params = NWParameters.tcp
                            let connection = NWConnection(to: result.endpoint, using: params)
                            connection.stateUpdateHandler = { connState in
                                if case .ready = connState {
                                    if let endpoint = connection.currentPath?.remoteEndpoint,
                                       case let .hostPort(host, port) = endpoint {
                                        var hostStr: String
                                        switch host {
                                        case .ipv4(let addr):
                                            hostStr = "\(addr)"
                                        case .ipv6(let addr):
                                            hostStr = "\(addr)"
                                        case .name(let name, _):
                                            hostStr = name
                                        @unknown default:
                                            hostStr = "localhost"
                                        }
                                        if let pctIdx = hostStr.firstIndex(of: "%") {
                                            hostStr = String(hostStr[hostStr.startIndex..<pctIdx])
                                        }

                                        var grpcPort = Int(port.rawValue)
                                        var apiPort = 8000
                                        var version = "unknown"
                                        var tls = false

                                        if case let .bonjour(txtRecord) = result.metadata {
                                            let dict = Self.parseTXTRecord(txtRecord)
                                            if let gp = dict["grpc_port"] { grpcPort = Int(gp) ?? grpcPort }
                                            if let ap = dict["api_port"] { apiPort = Int(ap) ?? apiPort }
                                            if let v = dict["version"] { version = v }
                                            tls = dict["tls"] == "1"
                                        }

                                        connection.cancel()
                                        resumeOnce(DiscoveredOrchestrator(
                                            host: hostStr,
                                            grpcPort: grpcPort,
                                            apiPort: apiPort,
                                            version: version,
                                            tlsEnabled: tls
                                        ))
                                    }
                                }
                            }
                            connection.start(queue: serialQueue)
                            return
                        }
                    }
                }
            }

            browser.start(queue: serialQueue)

            // Timeout
            serialQueue.asyncAfter(deadline: .now() + timeout) {
                resumeOnce(nil)
            }
        }
    }

    private static func parseTXTRecord(_ txtRecord: NWTXTRecord) -> [String: String] {
        var result: [String: String] = [:]
        for (key, value) in txtRecord {
            switch value {
            case .string(let str):
                result[key] = str
            case .data(let data):
                if let str = String(data: data, encoding: .utf8) {
                    result[key] = str
                }
            case .none, .empty:
                result[key] = ""
            @unknown default:
                break
            }
        }
        return result
    }
}
