@preconcurrency import Darwin
import Foundation
#if canImport(UIKit)
import UIKit
#endif
#if os(macOS)
import IOKit.ps
#endif
import EdgeOrchestraProtos
import SwiftProtobuf

public final class MetricsCollector: @unchecked Sendable {

    private var previousTicks: (user: Double, system: Double, idle: Double, nice: Double)?

    public init() {
        #if os(iOS)
        Task { @MainActor in
            UIDevice.current.isBatteryMonitoringEnabled = true
        }
        #endif
    }

    public func collect() -> Edgeorchestra_V1_DeviceMetrics {
        var metrics = Edgeorchestra_V1_DeviceMetrics()
        metrics.cpuUsage = cpuUsage()
        metrics.memoryUsage = memoryUsage()
        metrics.thermalPressure = thermalPressure()
        metrics.battery = batteryInfo()
        metrics.collectedAt = .init(date: Date())
        return metrics
    }

    private func cpuUsage() -> Float {
        var loadInfo = host_cpu_load_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &loadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        let user = Double(loadInfo.cpu_ticks.0)
        let system = Double(loadInfo.cpu_ticks.1)
        let idle = Double(loadInfo.cpu_ticks.2)
        let nice = Double(loadInfo.cpu_ticks.3)

        defer {
            previousTicks = (user, system, idle, nice)
        }

        guard let prev = previousTicks else {
            // First call — return cumulative ratio
            let total = user + system + idle + nice
            guard total > 0 else { return 0 }
            return Float((user + system) / total)
        }

        let dUser = user - prev.user
        let dSystem = system - prev.system
        let dIdle = idle - prev.idle
        let dNice = nice - prev.nice
        let dTotal = dUser + dSystem + dIdle + dNice
        guard dTotal > 0 else { return 0 }
        return Float((dUser + dSystem) / dTotal)
    }

    private func memoryUsage() -> Float {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let pageSize = UInt64(vm_kernel_page_size)
        let active = UInt64(stats.active_count) * pageSize
        let wired = UInt64(stats.wire_count) * pageSize
        let total = ProcessInfo.processInfo.physicalMemory
        return Float(Double(active + wired) / Double(total))
    }

    private func thermalPressure() -> Float {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return 0.1
        case .fair: return 0.4
        case .serious: return 0.7
        case .critical: return 1.0
        @unknown default: return 0.2
        }
    }

    private func batteryInfo() -> Edgeorchestra_V1_BatteryInfo {
        var info = Edgeorchestra_V1_BatteryInfo()
        #if os(iOS)
        let level = UIDevice.current.batteryLevel
        // batteryLevel returns -1.0 if monitoring not enabled yet
        info.level = level >= 0 ? level : 1.0
        switch UIDevice.current.batteryState {
        case .charging: info.state = .charging
        case .full: info.state = .full
        case .unplugged: info.state = .discharging
        default: info.state = .notCharging
        }
        info.isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        #elseif os(macOS)
        let (level, state) = Self.macOSBatteryInfo()
        info.level = level
        info.state = state
        info.isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        #endif
        return info
    }

    #if os(macOS)
    private static func macOSBatteryInfo() -> (Float, Edgeorchestra_V1_BatteryState) {
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as [CFTypeRef]

        guard let source = sources.first,
              let desc = IOPSGetPowerSourceDescription(snapshot, source).takeUnretainedValue() as? [String: Any]
        else {
            // No battery (Mac mini, Mac Pro, etc.) — report as plugged in, full
            return (1.0, .unspecified)
        }

        let capacity = desc[kIOPSCurrentCapacityKey as String] as? Int ?? 100
        let maxCapacity = desc[kIOPSMaxCapacityKey as String] as? Int ?? 100
        let level = Float(capacity) / Float(max(maxCapacity, 1))

        let isCharging = desc[kIOPSIsChargingKey as String] as? Bool ?? false
        let isPluggedIn = desc[kIOPSPowerSourceStateKey as String] as? String == kIOPSACPowerValue as String

        let state: Edgeorchestra_V1_BatteryState
        if level >= 0.99 && isPluggedIn {
            state = .full
        } else if isCharging {
            state = .charging
        } else if isPluggedIn {
            state = .notCharging
        } else {
            state = .discharging
        }

        return (level, state)
    }
    #endif
}
