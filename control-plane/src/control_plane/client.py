import grpc


class GRPCClient:
    def __init__(
        self,
        target: str = "localhost:50051",
        tls: bool = False,
        ca_cert: str | None = None,
        client_cert: str | None = None,
        client_key: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if tls:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=open(ca_cert, "rb").read() if ca_cert else None,
                private_key=open(client_key, "rb").read() if client_key else None,
                certificate_chain=open(client_cert, "rb").read() if client_cert else None,
            )
            self.channel = grpc.secure_channel(target, credentials)
        else:
            self.channel = grpc.insecure_channel(target)

        self._metadata = [("x-api-key", api_key)] if api_key else []

    def list_devices(self) -> list:
        from orchestrator.generated import device_pb2, device_pb2_grpc

        stub = device_pb2_grpc.DeviceRegistryStub(self.channel)
        response = stub.ListDevices(device_pb2.ListDevicesRequest(), metadata=self._metadata)
        return list(response.devices)

    def get_device(self, device_id: str):
        from orchestrator.generated import common_pb2, device_pb2, device_pb2_grpc

        stub = device_pb2_grpc.DeviceRegistryStub(self.channel)
        response = stub.GetDevice(
            device_pb2.GetDeviceRequest(device_id=common_pb2.DeviceId(value=device_id)),
            metadata=self._metadata,
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
            ),
            metadata=self._metadata,
        )
        return response

    def close(self) -> None:
        self.channel.close()
