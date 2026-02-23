import Compression
@preconcurrency import CoreML
import Foundation

/// Performs real on-device training using CoreML's `MLUpdateTask`.
///
/// Flow:
/// 1. Write .mlmodel bytes to temp file
/// 2. Compile with `MLModel.compileModel(at:)`
/// 3. Run `MLUpdateTask` with MNIST data
/// 4. Extract pre-training weights at `trainingBegin`
/// 5. Extract post-training weights at completion
/// 6. Compute deltas (post - pre), serialize
/// 7. Clean up temp files
public struct CoreMLTrainer: Sendable {

    private let trainBatchProvider: MLBatchProvider
    private let trainSampleCount: Int
    private let testBatchProvider: MLBatchProvider?
    private let testLabels: [Int]?
    private let layerMappings: [(String, String, MLParameterKey)]
    private let layerOrder: [String]

    public init(dataProvider: MNISTDataProvider) {
        self.trainBatchProvider = dataProvider.batchProvider
        self.trainSampleCount = dataProvider.sampleCount
        self.testBatchProvider = dataProvider.testBatchProvider
        self.testLabels = dataProvider.testLabels
        self.layerMappings = [
            ("hidden_weight", "hidden", .weights),
            ("hidden_bias", "hidden", .biases),
            ("output_weight", "output", .weights),
            ("output_bias", "output", .biases),
        ]
        self.layerOrder = ["hidden_weight", "hidden_bias", "output_weight", "output_bias"]
    }

    public init(
        trainBatchProvider: MLBatchProvider,
        trainSampleCount: Int,
        testBatchProvider: MLBatchProvider?,
        testLabels: [Int]?,
        layerMappings: [(String, String, MLParameterKey)],
        layerOrder: [String]
    ) {
        self.trainBatchProvider = trainBatchProvider
        self.trainSampleCount = trainSampleCount
        self.testBatchProvider = testBatchProvider
        self.testLabels = testLabels
        self.layerMappings = layerMappings
        self.layerOrder = layerOrder
    }

    public func train(
        modelData: Data
    ) async throws -> (gradients: Data, numSamples: Int, metrics: [String: Float]) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let mlmodelPath = tempDir.appendingPathComponent("model.mlmodel")
        try modelData.write(to: mlmodelPath)

        // Compile the model
        let compiledURL = try await MLModel.compileModel(at: mlmodelPath)
        defer {
            try? FileManager.default.removeItem(at: compiledURL)
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        // Run MLUpdateTask — extracts pre/post weights inside the task handlers
        let result = try await runUpdateTask(
            compiledModelURL: compiledURL,
            configuration: config,
            trainingData: trainBatchProvider
        )

        // Compute deltas and serialize
        let deltas = computeDeltas(pre: result.preWeights, post: result.postWeights)
        let serialized = serializeWeightDeltas(deltas)

        var metrics = computeMetrics(pre: result.preWeights, post: result.postWeights)
        metrics["loss"] = result.finalLoss

        // Compute real accuracy on local test set if available
        if let testBatch = testBatchProvider, let labels = testLabels, !labels.isEmpty {
            defer { try? FileManager.default.removeItem(at: result.updatedModelURL) }

            let predModel = try MLModel(contentsOf: result.updatedModelURL, configuration: config)
            var correct = 0
            for i in 0..<testBatch.count {
                let sample = testBatch.features(at: i)
                if let prediction = try? await predModel.prediction(from: sample),
                   let classLabel = prediction.featureValue(for: "classLabel")?.int64Value {
                    if Int(classLabel) == labels[i] {
                        correct += 1
                    }
                }
            }
            metrics["accuracy"] = Float(correct) / Float(labels.count)
            print("[CoreMLTrainer] Local validation: \(correct)/\(labels.count) = \(metrics["accuracy"]!)")
        } else {
            try? FileManager.default.removeItem(at: result.updatedModelURL)
            // Fallback: estimate accuracy from loss
            let randomLoss: Float = 2.302
            metrics["accuracy"] = max(0.0, min(1.0, 1.0 - result.finalLoss / randomLoss))
        }

        return (serialized, trainSampleCount, metrics)
    }

    // MARK: - Private

    private struct UpdateResult: Sendable {
        let preWeights: [String: [Float]]
        let postWeights: [String: [Float]]
        let finalLoss: Float
        /// URL to the compiled (.mlmodelc) updated model directory for local validation.
        let updatedModelURL: URL
    }

    private func runUpdateTask(
        compiledModelURL: URL,
        configuration: MLModelConfiguration,
        trainingData: MLBatchProvider
    ) async throws -> UpdateResult {
        let mappings = self.layerMappings
        return try await withCheckedThrowingContinuation { continuation in
            nonisolated(unsafe) var preWeights: [String: [Float]] = [:]
            nonisolated(unsafe) var lastLoss: Float = 0

            let progressHandlers = MLUpdateProgressHandlers(
                forEvents: [.trainingBegin, .epochEnd],
                progressHandler: { context in
                    if context.event == .trainingBegin {
                        // Extract pre-training weights (only accessible inside task context)
                        if let weights = try? Self.extractWeights(from: context.model, mappings: mappings) {
                            preWeights = weights
                            print("[CoreMLTrainer] Pre-training weights extracted (\(weights.count) layers)")
                        } else {
                            print("[CoreMLTrainer] WARNING: Failed to extract pre-training weights")
                        }
                    }
                    if context.event == .epochEnd {
                        let loss = context.metrics[.lossValue] as? Double ?? -1
                        lastLoss = Float(loss)
                        print("[CoreMLTrainer] Epoch end - loss: \(loss)")
                    }
                },
                completionHandler: { context in
                    if let error = context.task.error {
                        continuation.resume(throwing: error)
                    } else {
                        do {
                            let postWeights = try Self.extractWeights(from: context.model, mappings: mappings)
                            let finalLoss = context.metrics[.lossValue] as? Double
                            let loss = Float(finalLoss ?? Double(lastLoss))

                            // Write updated compiled model for local validation
                            let tmpURL = FileManager.default.temporaryDirectory
                                .appendingPathComponent(UUID().uuidString + ".mlmodelc")
                            try context.model.write(to: tmpURL)

                            continuation.resume(returning: UpdateResult(
                                preWeights: preWeights,
                                postWeights: postWeights,
                                finalLoss: loss,
                                updatedModelURL: tmpURL
                            ))
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    }
                }
            )

            do {
                let updateTask = try MLUpdateTask(
                    forModelAt: compiledModelURL,
                    trainingData: trainingData,
                    configuration: configuration,
                    progressHandlers: progressHandlers
                )
                updateTask.resume()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    private static func extractWeights(
        from model: MLModel,
        mappings: [(String, String, MLParameterKey)]
    ) throws -> [String: [Float]] {
        var weights: [String: [Float]] = [:]

        for (name, scope, key) in mappings {
            let scopedKey = key.scoped(to: scope)
            let value = try model.parameterValue(for: scopedKey)
            guard let param = value as? MLMultiArray else {
                throw CoreMLTrainerError.weightExtractionFailed(name)
            }
            let count = param.count
            let ptr = param.dataPointer.bindMemory(to: Float.self, capacity: count)
            weights[name] = Array(UnsafeBufferPointer(start: ptr, count: count))
        }

        return weights
    }

    private func computeDeltas(
        pre: [String: [Float]],
        post: [String: [Float]]
    ) -> [String: [Float]] {
        var deltas: [String: [Float]] = [:]
        for (name, preValues) in pre {
            guard let postValues = post[name] else { continue }
            var delta = [Float](repeating: 0, count: preValues.count)
            for i in 0..<preValues.count {
                delta[i] = postValues[i] - preValues[i]
            }
            deltas[name] = delta
        }
        return deltas
    }

    private func serializeWeightDeltas(_ deltas: [String: [Float]]) -> Data {
        let float16Payload = serializeFloat16(deltas)

        // Compress with raw LZ4 block format (compatible with Python lz4.block)
        guard let compressed = lz4RawCompress(float16Payload) else {
            // Fallback to uncompressed float32 if compression fails
            return serializeFloat32Fallback(deltas)
        }

        // Wire format: [0x01] [original_size: UInt32 LE] [lz4_compressed_data]
        var wire = Data()
        wire.append(0x01)
        var originalSize = UInt32(float16Payload.count)
        wire.append(Data(bytes: &originalSize, count: 4))
        wire.append(compressed)

        return wire
    }

    /// Raw LZ4 block compression compatible with Python's lz4.block.
    /// Apple's NSData.compressed(using: .lz4) uses a proprietary framing
    /// that is NOT compatible with lz4.block.decompress.
    private func lz4RawCompress(_ input: Data) -> Data? {
        let sourceSize = input.count
        // LZ4 worst case: input size + input/255 + 16
        let destinationCapacity = sourceSize + sourceSize / 255 + 16
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destinationCapacity)
        defer { destinationBuffer.deallocate() }

        let compressedSize = input.withUnsafeBytes { srcPtr -> Int in
            guard let srcBase = srcPtr.baseAddress else { return 0 }
            return compression_encode_buffer(
                destinationBuffer, destinationCapacity,
                srcBase.assumingMemoryBound(to: UInt8.self), sourceSize,
                nil,
                COMPRESSION_LZ4_RAW
            )
        }

        guard compressedSize > 0 else { return nil }
        return Data(bytes: destinationBuffer, count: compressedSize)
    }

    private func serializeFloat16(_ deltas: [String: [Float]]) -> Data {
        let layers = layerOrder.filter { deltas[$0] != nil }

        var data = Data()

        var layerCount = UInt32(layers.count)
        data.append(Data(bytes: &layerCount, count: 4))

        for name in layers {
            guard let values = deltas[name] else { continue }
            let nameData = Data(name.utf8)
            var nameLen = UInt32(nameData.count)
            data.append(Data(bytes: &nameLen, count: 4))
            data.append(nameData)

            var elemCount = UInt32(values.count)
            data.append(Data(bytes: &elemCount, count: 4))

            // Quantize Float → Float16 (2 bytes per element)
            let float16Values = values.map { Float16($0) }
            float16Values.withUnsafeBufferPointer { buffer in
                data.append(buffer)
            }
        }

        return data
    }

    private func serializeFloat32Fallback(_ deltas: [String: [Float]]) -> Data {
        let layers = layerOrder.filter { deltas[$0] != nil }

        var data = Data()

        var layerCount = UInt32(layers.count)
        data.append(Data(bytes: &layerCount, count: 4))

        for name in layers {
            guard let values = deltas[name] else { continue }
            let nameData = Data(name.utf8)
            var nameLen = UInt32(nameData.count)
            data.append(Data(bytes: &nameLen, count: 4))
            data.append(nameData)

            var elemCount = UInt32(values.count)
            data.append(Data(bytes: &elemCount, count: 4))

            values.withUnsafeBufferPointer { buffer in
                data.append(UnsafeBufferPointer(start: buffer.baseAddress, count: buffer.count))
            }
        }

        return data
    }

    private func computeMetrics(
        pre: [String: [Float]],
        post: [String: [Float]]
    ) -> [String: Float] {
        var totalDeltaNorm: Float = 0
        for (name, preValues) in pre {
            guard let postValues = post[name] else { continue }
            for i in 0..<preValues.count {
                let d = postValues[i] - preValues[i]
                totalDeltaNorm += d * d
            }
        }
        totalDeltaNorm = sqrtf(totalDeltaNorm)

        return [
            "loss": 0.0,
            "accuracy": 0.0,
            "delta_norm": totalDeltaNorm,
        ]
    }
}

enum CoreMLTrainerError: Error, CustomStringConvertible {
    case weightExtractionFailed(String)

    var description: String {
        switch self {
        case .weightExtractionFailed(let layer):
            return "Failed to extract weights for layer: \(layer)"
        }
    }
}
