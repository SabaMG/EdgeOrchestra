import Foundation
#if canImport(UIKit)
import UIKit
#endif
import EdgeOrchestraProtos

public struct DeviceInfoCollector: Sendable {

    public init() {}

    public func collectDeviceInfo() -> (name: String, model: String, osVersion: String, capabilities: Edgeorchestra_V1_DeviceCapabilities) {
        let name = hostName()
        let model = deviceModel()
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString

        var caps = Edgeorchestra_V1_DeviceCapabilities()
        caps.chip = chipName()
        caps.memoryBytes = UInt64(ProcessInfo.processInfo.physicalMemory)
        caps.cpuCores = UInt32(ProcessInfo.processInfo.activeProcessorCount)
        caps.gpuCores = estimatedGPUCores()
        caps.neuralEngineCores = 16 // All modern Apple chips have 16 NE cores
        caps.supportedFrameworks = ["coreml"]

        return (name, model, osVersion, caps)
    }

    private func hostName() -> String {
        #if os(iOS)
        return UIDevice.current.name
        #else
        return Host.current().localizedName ?? ProcessInfo.processInfo.hostName
        #endif
    }

    private func deviceModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        return String(cString: model)
    }

    private func chipName() -> String {
        #if arch(arm64)
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        if size > 0 {
            var brand = [CChar](repeating: 0, count: size)
            sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
            return String(cString: brand)
        }
        return "Apple Silicon"
        #else
        return "Intel"
        #endif
    }

    private func estimatedGPUCores() -> UInt32 {
        // Apple doesn't expose GPU core count via API, estimate from model
        let model = deviceModel().lowercased()
        if model.contains("mac14") || model.contains("mac15") { return 10 }
        if model.contains("mac16") { return 16 }
        return 8
    }
}
