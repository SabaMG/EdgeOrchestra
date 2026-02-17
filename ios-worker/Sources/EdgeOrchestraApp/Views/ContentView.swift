import SwiftUI
import EdgeOrchestraWorker

struct ContentView: View {
    let state: WorkerState

    var body: some View {
        TabView {
            NavigationStack {
                ConnectionView(state: state)
            }
            .tabItem { Label("Connection", systemImage: "network") }

            NavigationStack {
                MetricsView(state: state)
            }
            .tabItem { Label("Metrics", systemImage: "gauge.medium") }

            NavigationStack {
                TrainingView(state: state)
            }
            .tabItem { Label("Training", systemImage: "brain") }

            NavigationStack {
                LogView(state: state)
            }
            .tabItem { Label("Logs", systemImage: "text.justify.left") }
        }
        #if os(macOS)
        .frame(minWidth: 600, minHeight: 400)
        #endif
    }
}
