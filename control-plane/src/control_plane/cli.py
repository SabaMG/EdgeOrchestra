from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from control_plane.api_client import APIClient

app = typer.Typer(name="eo", help="EdgeOrchestra CLI")
console = Console()

# Global state set by the callback
_global_api_key: str | None = None
_global_tls: bool = False
_global_ca_cert: str | None = None
_global_client_cert: str | None = None
_global_client_key: str | None = None


@app.callback()
def main(
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="EO_API_KEY", help="API key"),
    tls: bool = typer.Option(False, "--tls", envvar="EO_TLS_ENABLED", help="Enable TLS"),
    ca_cert: Optional[str] = typer.Option(None, "--ca-cert", help="CA certificate path"),
    client_cert: Optional[str] = typer.Option(
        None, "--client-cert", help="Client certificate path"
    ),
    client_key: Optional[str] = typer.Option(None, "--client-key", help="Client key path"),
):
    """EdgeOrchestra CLI — manage your federated learning cluster."""
    global _global_api_key, _global_tls, _global_ca_cert, _global_client_cert, _global_client_key
    _global_api_key = api_key
    _global_tls = tls
    _global_ca_cert = ca_cert
    _global_client_cert = client_cert
    _global_client_key = client_key


def _get_api(host: str = "localhost", port: int = 8000) -> APIClient:
    return APIClient(base_url=f"http://{host}:{port}", api_key=_global_api_key)


def _get_grpc(host: str = "localhost", port: int = 50051):
    from control_plane.client import GRPCClient

    return GRPCClient(
        target=f"{host}:{port}",
        tls=_global_tls,
        ca_cert=_global_ca_cert,
        client_cert=_global_client_cert,
        client_key=_global_client_key,
        api_key=_global_api_key,
    )


@app.command()
def ping(
    host: str = typer.Option("localhost", help="Orchestrator host"),
    port: int = typer.Option(8000, help="API port"),
):
    """Check if the orchestrator is reachable."""
    try:
        client = _get_api(host, port)
        result = client.health()
        client.close()
        console.print(f"[green]Orchestrator is up:[/green] {result}")
    except Exception as e:
        console.print(f"[red]Orchestrator unreachable:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def status(
    host: str = typer.Option("localhost", help="Orchestrator host"),
    port: int = typer.Option(8000, help="API port"),
):
    """Show orchestrator status and connected devices."""
    try:
        client = _get_api(host, port)
        health = client.health()
        devices = client.list_devices()
        client.close()

        console.print(f"[green]Status:[/green] {health.get('status', 'unknown')}")
        online = sum(1 for d in devices if d.get("status") == "online")
        console.print(f"Devices: {len(devices)} total, {online} online")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def devices(
    status_filter: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    host: str = typer.Option("localhost", help="Orchestrator host"),
    port: int = typer.Option(8000, help="API port"),
):
    """List all registered devices."""
    try:
        client = _get_api(host, port)
        device_list = client.list_devices(status=status_filter)
        client.close()

        if not device_list:
            console.print("[dim]No devices registered.[/dim]")
            return

        table = Table(title="Devices")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Name", style="bold")
        table.add_column("Model")
        table.add_column("Chip")
        table.add_column("Status")
        table.add_column("Battery")
        table.add_column("Last Seen")

        for d in device_list:
            status_style = "green" if d["status"] == "online" else "red"
            battery = f"{d.get('battery_level', 0) * 100:.0f}%" if d.get("battery_level") else "-"
            table.add_row(
                str(d["id"])[:8],
                d["name"],
                d.get("device_model", "-"),
                d.get("chip", "-"),
                f"[{status_style}]{d['status']}[/{status_style}]",
                battery,
                d.get("last_seen_at", "-")[:19] if d.get("last_seen_at") else "-",
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def device(
    device_id: str = typer.Argument(help="Device UUID"),
    host: str = typer.Option("localhost", help="Orchestrator host"),
    port: int = typer.Option(8000, help="API port"),
):
    """Show details for a specific device."""
    try:
        client = _get_api(host, port)
        d = client.get_device(device_id)
        client.close()

        console.print(f"[bold]Device: {d['name']}[/bold]")
        console.print(f"  ID:     {d['id']}")
        console.print(f"  Model:  {d.get('device_model', '-')}")
        console.print(f"  OS:     {d.get('os_version', '-')}")
        console.print(f"  Chip:   {d.get('chip', '-')}")
        console.print(f"  Status: {d['status']}")
        if d.get("battery_level") is not None:
            console.print(f"  Battery: {d['battery_level'] * 100:.0f}% ({d.get('battery_state', '-')})")
        console.print(f"  Registered: {d.get('registered_at', '-')}")
        console.print(f"  Last seen:  {d.get('last_seen_at', '-')}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def discover(timeout: int = typer.Option(5, help="Discovery timeout in seconds")):
    """Discover EdgeOrchestra services via mDNS."""
    from zeroconf import ServiceBrowser, Zeroconf
    import time

    console.print(f"[dim]Scanning for EdgeOrchestra services ({timeout}s)...[/dim]")
    found = []

    class Listener:
        def __init__(self, zc: Zeroconf):
            self.zc = zc

        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info:
                addresses = [addr for addr in info.parsed_addresses()]
                found.append({
                    "name": name,
                    "addresses": addresses,
                    "port": info.port,
                    "properties": {
                        k.decode(): v.decode() if isinstance(v, bytes) else v
                        for k, v in info.properties.items()
                    },
                })

        def remove_service(self, zc, type_, name):
            pass

        def update_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    listener = Listener(zc)
    browser = ServiceBrowser(zc, "_edgeorchestra._tcp.local.", listener)

    time.sleep(timeout)
    zc.close()

    if not found:
        console.print("[yellow]No EdgeOrchestra services found.[/yellow]")
        return

    table = Table(title="Discovered Services")
    table.add_column("Name")
    table.add_column("Address")
    table.add_column("gRPC Port")
    table.add_column("API Port")
    table.add_column("Version")
    table.add_column("TLS")

    for svc in found:
        props = svc["properties"]
        tls_status = "yes" if props.get("tls") == "1" else "no"
        table.add_row(
            svc["name"],
            ", ".join(svc["addresses"]),
            str(svc["port"]),
            props.get("api_port", "-"),
            props.get("version", "-"),
            tls_status,
        )
    console.print(table)


if __name__ == "__main__":
    app()
