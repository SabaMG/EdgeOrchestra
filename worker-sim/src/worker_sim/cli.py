import logging

import typer
from rich.console import Console
from rich.table import Table

from worker_sim.device_profile import PROFILES
from worker_sim.manager import run_workers

app = typer.Typer(name="worker-sim", help="Simulated device workers for EdgeOrchestra")
console = Console()


@app.command()
def start(
    target: str = typer.Option("localhost:50051", "-t", "--target", help="gRPC server address"),
    profile: str = typer.Option("iphone15pro", "-p", "--profile", help="Device profile name"),
    count: int = typer.Option(1, "-n", "--count", help="Number of simulated devices"),
    interval: float = typer.Option(5.0, "-i", "--interval", help="Heartbeat interval (seconds)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging"),
):
    """Start simulated device workers."""
    if profile not in PROFILES:
        console.print(f"[red]Unknown profile:[/red] {profile}")
        console.print(f"Available: {', '.join(PROFILES.keys())}")
        raise typer.Exit(code=1)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = PROFILES[profile]
    console.print(f"[bold]EdgeOrchestra Worker Simulator[/bold]")
    console.print(f"  Target:    {target}")
    console.print(f"  Profile:   {profile} ({p.chip}, {p.memory_bytes // (1024**3)}GB)")
    console.print(f"  Workers:   {count}")
    console.print(f"  Interval:  {interval}s")
    console.print()

    run_workers(target, profile, count, interval)


@app.command("profiles")
def list_profiles():
    """List available device profiles."""
    table = Table(title="Device Profiles")
    table.add_column("Key", style="bold")
    table.add_column("Name")
    table.add_column("Chip")
    table.add_column("RAM")
    table.add_column("CPU")
    table.add_column("GPU")
    table.add_column("NE")
    table.add_column("Frameworks")

    for key, p in PROFILES.items():
        table.add_row(
            key,
            p.name,
            p.chip,
            f"{p.memory_bytes // (1024**3)}GB",
            str(p.cpu_cores),
            str(p.gpu_cores),
            str(p.neural_engine_cores),
            ", ".join(p.supported_frameworks),
        )
    console.print(table)


if __name__ == "__main__":
    app()
