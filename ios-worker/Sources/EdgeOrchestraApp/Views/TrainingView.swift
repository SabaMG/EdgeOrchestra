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
                if let acc = state.serverAccuracy {
                    LabeledContent("Server Accuracy", value: "\(String(format: "%.1f", acc * 100))%")
                }
                if let loss = state.serverLoss {
                    LabeledContent("Server Loss", value: String(format: "%.4f", loss))
                }
            }

            Section("Training Jobs") {
                if state.trainingJobs.isEmpty {
                    Text("No training jobs yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(state.trainingJobs) { job in
                        NavigationLink(value: job.jobId) {
                            JobCardRow(job: job, isActive: job.jobId == state.activeJobId)
                        }
                    }
                }
            }
        }
        .navigationTitle("Training")
        .navigationDestination(for: String.self) { jobId in
            if let job = state.trainingJobs.first(where: { $0.jobId == jobId }) {
                JobDetailView(job: job, state: state)
            }
        }
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

// MARK: - Job Card Row

private struct JobCardRow: View {
    let job: TrainingJob
    let isActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(job.architecture.uppercased(), systemImage: "brain")
                    .font(.headline)
                Spacer()
                StatusBadge(status: job.status, isActive: isActive)
            }

            HStack(spacing: 16) {
                if let acc = job.latestAccuracy {
                    Label("\(String(format: "%.1f", acc * 100))%", systemImage: "checkmark.circle")
                        .foregroundStyle(.green)
                        .font(.caption.monospacedDigit())
                }
                if let loss = job.latestLoss {
                    Label(String(format: "%.4f", loss), systemImage: "xmark.circle")
                        .foregroundStyle(.red)
                        .font(.caption.monospacedDigit())
                }
                Label("\(job.rounds.count) rounds", systemImage: "arrow.trianglehead.2.counterclockwise")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text(job.startedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("ID: \(String(job.jobId.prefix(8)))")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Status Badge

private struct StatusBadge: View {
    let status: TrainingJobStatus
    let isActive: Bool

    var body: some View {
        HStack(spacing: 4) {
            if isActive && status == .running {
                ProgressView()
                    .controlSize(.mini)
            }
            Text(displayText)
                .font(.caption.bold())
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(backgroundColor.opacity(0.15))
        .foregroundStyle(backgroundColor)
        .clipShape(Capsule())
    }

    private var displayText: String {
        if isActive && status == .running { return "Active" }
        return status.rawValue.capitalized
    }

    private var backgroundColor: Color {
        switch status {
        case .running: return isActive ? .orange : .blue
        case .completed: return .green
        case .failed: return .red
        }
    }
}

// MARK: - Job Detail View

struct JobDetailView: View {
    let job: TrainingJob
    let state: WorkerState

    var body: some View {
        List {
            Section("Info") {
                LabeledContent("Architecture", value: job.architecture.uppercased())
                LabeledContent("Status", value: job.status.rawValue.capitalized)
                LabeledContent("Rounds", value: "\(job.rounds.count)")
                LabeledContent("Total Samples", value: "\(job.totalSamples)")
                LabeledContent("Started", value: job.startedAt, format: .dateTime)
                LabeledContent("Job ID", value: String(job.jobId.prefix(8)))
            }

            if !job.rounds.isEmpty {
                Section("Loss") {
                    Chart {
                        ForEach(job.rounds) { result in
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
                        ForEach(job.rounds) { result in
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

                Section("Rounds") {
                    ForEach(job.rounds.reversed()) { result in
                        HStack {
                            Text("Round \(result.round)")
                                .font(.headline)
                            Spacer()
                            VStack(alignment: .trailing) {
                                Text("Loss: \(result.loss, specifier: "%.4f")")
                                    .foregroundStyle(.red)
                                Text("Acc: \(result.accuracy * 100, specifier: "%.1f")%")
                                    .foregroundStyle(.green)
                                Text("\(result.numSamples) samples")
                                    .foregroundStyle(.secondary)
                            }
                            .font(.caption.monospacedDigit())
                        }
                    }
                }
            }
        }
        .navigationTitle(job.architecture.uppercased())
    }
}
