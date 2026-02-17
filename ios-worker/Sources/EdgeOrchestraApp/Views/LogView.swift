import SwiftUI
import EdgeOrchestraWorker

struct LogView: View {
    let state: WorkerState

    var body: some View {
        List {
            ForEach(state.logEntries.reversed()) { entry in
                HStack(alignment: .top, spacing: 8) {
                    Text(entry.timestamp, style: .time)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text(entry.level)
                        .font(.caption.bold())
                        .foregroundStyle(levelColor(entry.level))
                        .frame(width: 40, alignment: .leading)
                    Text(entry.message)
                        .font(.caption)
                }
            }
        }
        .navigationTitle("Logs")
    }

    private func levelColor(_ level: String) -> Color {
        switch level {
        case "ERROR": return .red
        case "WARN": return .yellow
        case "INFO": return .blue
        default: return .secondary
        }
    }
}
