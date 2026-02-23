import SwiftUI
import EdgeOrchestraWorker

@main
struct EdgeOrchestraApp: App {
    @State private var state = WorkerState()
    @State private var engine: WorkerEngine?
    @State private var engineTask: Task<Void, Never>?

    var body: some Scene {
        WindowGroup {
            ContentView(state: state)
                .onAppear {
                    #if os(iOS)
                    UIApplication.shared.isIdleTimerDisabled = true
                    #endif
                    let eng = WorkerEngine(state: state)
                    engine = eng
                    engineTask = Task {
                        await eng.run()
                    }
                }
                .onDisappear {
                    engineTask?.cancel()
                }
        }
    }
}
