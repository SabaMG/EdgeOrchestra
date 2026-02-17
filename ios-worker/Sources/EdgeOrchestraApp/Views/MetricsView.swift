import SwiftUI
import EdgeOrchestraWorker

struct MetricsView: View {
    let state: WorkerState

    var body: some View {
        List {
            Section("System") {
                GaugeRow(label: "CPU", value: state.cpuUsage, format: .percent, color: .blue)
                GaugeRow(label: "Memory", value: state.memoryUsage, format: .percent, color: .orange)
                GaugeRow(label: "Thermal", value: state.thermalPressure, format: .percent, color: .red)
            }

            Section("Battery") {
                GaugeRow(label: "Level", value: state.batteryLevel, format: .exact, color: .green)
            }
        }
        .navigationTitle("Metrics")
    }
}

enum GaugeFormat {
    case percent   // shows "42%"
    case exact     // shows "87.3%"
}

struct GaugeRow: View {
    let label: String
    let value: Float
    var format: GaugeFormat = .percent
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(formattedValue)
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .animation(.easeInOut(duration: 0.3), value: value)
            }
            ProgressView(value: Double(min(max(value, 0), 1)))
                .tint(color)
                .animation(.easeInOut(duration: 0.3), value: value)
        }
        .padding(.vertical, 4)
    }

    private var formattedValue: String {
        switch format {
        case .percent:
            return "\(Int(value * 100))%"
        case .exact:
            return String(format: "%.1f%%", value * 100)
        }
    }
}
