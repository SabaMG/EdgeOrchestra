import httpx


class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        self.base_url = base_url
        headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(base_url=base_url, timeout=10.0, headers=headers)

    def health(self) -> dict:
        resp = self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def list_devices(self, status: str | None = None) -> list[dict]:
        params = {}
        if status:
            params["status"] = status
        resp = self.client.get("/api/v1/devices", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_device(self, device_id: str) -> dict:
        resp = self.client.get(f"/api/v1/devices/{device_id}")
        resp.raise_for_status()
        return resp.json()

    def delete_device(self, device_id: str) -> None:
        resp = self.client.delete(f"/api/v1/devices/{device_id}")
        resp.raise_for_status()

    def get_device_metrics(self, device_id: str) -> dict:
        resp = self.client.get(f"/api/v1/devices/{device_id}/metrics")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.client.close()
