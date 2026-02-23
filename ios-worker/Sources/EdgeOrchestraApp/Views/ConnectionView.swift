import SwiftUI
import EdgeOrchestraWorker

struct ConnectionView: View {
    let state: WorkerState
    @State private var manualAddressInput: String = ""

    private var localIPAddress: String {
        var address = "unknown"
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let firstAddr = ifaddr else { return address }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: firstAddr, next: { $0.pointee.ifa_next }) {
            let sa = ptr.pointee.ifa_addr.pointee
            guard sa.sa_family == UInt8(AF_INET) else { continue }
            let name = String(cString: ptr.pointee.ifa_name)
            guard name == "en0" || name == "en1" else { continue }
            var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            getnameinfo(ptr.pointee.ifa_addr, socklen_t(sa.sa_len),
                        &hostname, socklen_t(hostname.count),
                        nil, 0, NI_NUMERICHOST)
            address = String(cString: hostname)
            break
        }
        return address
    }

    var body: some View {
        List {
            Section("Manual Connection") {
                HStack {
                    TextField("host:port (e.g. 172.20.10.2:50051)", text: $manualAddressInput)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        #endif
                    Button(manualAddressInput.isEmpty ? "Auto" : "Connect") {
                        state.manualAddress = manualAddressInput.isEmpty ? nil : manualAddressInput
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(manualAddressInput.isEmpty ? .gray : .blue)
                }
                Text("Leave empty for Bonjour auto-discovery")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Network") {
                LabeledContent("Local IP", value: localIPAddress)
                HStack {
                    Circle()
                        .fill(state.connectionStatus == .discovering ? Color.yellow : (state.discoveredHost != nil ? Color.green : Color.red))
                        .frame(width: 10, height: 10)
                    Text(state.connectionStatus == .discovering ? "Searching via Bonjour…" : (state.discoveredHost != nil ? "Bonjour: found" : "Bonjour: not found"))
                        .font(.subheadline)
                }
            }

            Section("Status") {
                HStack {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 12, height: 12)
                    Text(state.connectionStatus.rawValue.capitalized)
                        .font(.headline)
                }

                if let host = state.discoveredHost {
                    LabeledContent("Orchestrator", value: host)
                }
            }

            Section("Device") {
                if !state.deviceName.isEmpty {
                    LabeledContent("Name", value: state.deviceName)
                }
                if let id = state.deviceId {
                    LabeledContent("Device ID", value: String(id.prefix(8)) + "...")
                }
            }

            Section("Heartbeat") {
                LabeledContent("Sequence", value: "\(state.heartbeatSequence)")
                LabeledContent("Interval", value: "\(state.heartbeatInterval)s")
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        #endif
        .navigationTitle("Connection")
    }

    private var statusColor: Color {
        switch state.connectionStatus {
        case .connected: return .green
        case .connecting, .discovering: return .yellow
        case .disconnected: return .red
        }
    }
}
