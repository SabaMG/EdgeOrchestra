// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "EdgeOrchestraWorker",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .executable(name: "EdgeOrchestraApp", targets: ["EdgeOrchestraApp"]),
        .library(name: "EdgeOrchestraWorker", targets: ["EdgeOrchestraWorker"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.28.0"),
        .package(url: "https://github.com/grpc/grpc-swift-2.git", from: "2.0.0"),
        .package(url: "https://github.com/grpc/grpc-swift-nio-transport.git", from: "2.0.0"),
        .package(url: "https://github.com/grpc/grpc-swift-protobuf.git", from: "2.0.0"),
    ],
    targets: [
        .target(
            name: "EdgeOrchestraProtos",
            dependencies: [
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
                .product(name: "GRPCProtobuf", package: "grpc-swift-protobuf"),
            ]
        ),
        .target(
            name: "EdgeOrchestraWorker",
            dependencies: [
                "EdgeOrchestraProtos",
                .product(name: "GRPCCore", package: "grpc-swift-2"),
                .product(name: "GRPCNIOTransportHTTP2", package: "grpc-swift-nio-transport"),
            ],
            resources: [
                .copy("Resources/mnist_train.bin"),
                .copy("Resources/cifar10_train.bin"),
            ]
        ),
        .executableTarget(
            name: "EdgeOrchestraApp",
            dependencies: ["EdgeOrchestraWorker"]
        ),
        .testTarget(
            name: "EdgeOrchestraWorkerTests",
            dependencies: ["EdgeOrchestraWorker"]
        ),
    ]
)
