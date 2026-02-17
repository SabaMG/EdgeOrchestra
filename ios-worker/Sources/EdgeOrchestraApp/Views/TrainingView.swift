import SwiftUI
import Charts
import EdgeOrchestraWorker

struct TrainingView: View {
    let state: WorkerState

    var body: some View {
        List {
            Section("Status") {
                HStack {
                    Text("Training Status")
                    Spacer()
                    Text(state.trainingStatus.rawValue.capitalized)
                        .foregroundStyle(trainingColor)
                        .bold()
                }
                if let round = state.currentRound {
                    LabeledContent("Current Round", value: round)
                }
            }

            if !state.trainingHistory.isEmpty {
                Section("Loss") {
                    Chart {
                        ForEach(Array(state.trainingHistory.enumerated()), id: \.offset) { _, result in
                            LineMark(
                                x: .value("Round", "R\(result.round)"),
                                y: .value("Loss", result.loss)
                            )
                            .foregroundStyle(.red)
                            .symbol(.circle)
                            .interpolationMethod(.catmullRom)

                            AreaMark(
                                x: .value("Round", "R\(result.round)"),
                                y: .value("Loss", result.loss)
                            )
                            .foregroundStyle(.red.opacity(0.1))
                            .interpolationMethod(.catmullRom)
                        }
                    }
                    .frame(height: 150)
                    .chartYAxisLabel("Loss")
                }

                Section("Accuracy") {
                    Chart {
                        ForEach(Array(state.trainingHistory.enumerated()), id: \.offset) { _, result in
                            LineMark(
                                x: .value("Round", "R\(result.round)"),
                                y: .value("Accuracy", result.accuracy * 100)
                            )
                            .foregroundStyle(.green)
                            .symbol(.square)
                            .interpolationMethod(.catmullRom)

                            AreaMark(
                                x: .value("Round", "R\(result.round)"),
                                y: .value("Accuracy", result.accuracy * 100)
                            )
                            .foregroundStyle(.green.opacity(0.1))
                            .interpolationMethod(.catmullRom)
                        }
                    }
                    .frame(height: 150)
                    .chartYScale(domain: 0...100)
                    .chartYAxisLabel("Accuracy %")
                }

                Section("Round History") {
                    ForEach(state.trainingHistory.reversed(), id: \.round) { result in
                        HStack {
                            Text("Round \(result.round)")
                                .font(.headline)
                            Spacer()
                            VStack(alignment: .trailing) {
                                Text("Loss: \(result.loss, specifier: "%.4f")")
                                    .foregroundStyle(.red)
                                Text("Acc: \(result.accuracy * 100, specifier: "%.1f")%")
                                    .foregroundStyle(.green)
                            }
                            .font(.caption.monospacedDigit())
                        }
                    }
                }
            }
        }
        .navigationTitle("Training")
    }

    private var trainingColor: Color {
        switch state.trainingStatus {
        case .idle: return .secondary
        case .downloading: return .blue
        case .training: return .orange
        case .submitting: return .green
        }
    }
}
