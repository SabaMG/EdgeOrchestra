import SwiftUI
import EdgeOrchestraWorker

struct ConnectionView: View {
    let state: WorkerState

    var body: some View {
        List {
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
