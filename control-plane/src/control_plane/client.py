import grpc


class GRPCClient:
    def __init__(self, target: str = "localhost:50051") -> None:
        self.channel = grpc.insecure_channel(target)

    def list_devices(self) -> list:
        from orchestrator.generated import device_pb2, device_pb2_grpc

        stub = device_pb2_grpc.DeviceRegistryStub(self.channel)
        response = stub.ListDevices(device_pb2.ListDevicesRequest())
        return list(response.devices)

    def get_device(self, device_id: str):
        from orchestrator.generated import common_pb2, device_pb2, device_pb2_grpc

        stub = device_pb2_grpc.DeviceRegistryStub(self.channel)
        response = stub.GetDevice(
            device_pb2.GetDeviceRequest(device_id=common_pb2.DeviceId(value=device_id))
        )
        return response.device

    def register_device(self, name: str, model: str, os_version: str):
        from orchestrator.generated import device_pb2, device_pb2_grpc

        stub = device_pb2_grpc.DeviceRegistryStub(self.channel)
        response = stub.Register(
            device_pb2.RegisterRequest(
                name=name,
                device_model=model,
                os_version=os_version,
            )
        )
        return response

    def close(self) -> None:
        self.channel.close()
