 EdgeOrchestra - Plan de Setup du Projet               
                                                  
 Contexte

 Projet de federated learning / edge computing qui transforme des devices Apple inutilises en cluster ML
 distribue. On setup la base: orchestrateur Python (FastAPI + gRPC), protocole de communication Protobuf,
 decouverte mDNS, Docker Compose, et CLI de controle. Le Mac sert d'orchestrateur pour le dev (pas de
 Raspberry Pi pour l'instant).

 ---
 Structure Monorepo

 EdgeOrchestra/
 ├── .gitignore
 ├── .env.example
 ├── pyproject.toml                    # Root workspace (uv)
 ├── docker-compose.yml
 ├── Makefile
 ├── protos/                           # Schemas Protobuf
 │   └── edgeorchestra/v1/
 │       ├── common.proto              # Types partages (DeviceId, Battery, etc.)
 │       ├── device.proto              # Service DeviceRegistry
 │       ├── heartbeat.proto           # Service Heartbeat (bidirectional streaming)
 │       └── model.proto              # Service Model (upload/download/gradients)
 ├── orchestrator/                     # Serveur principal
 │   ├── pyproject.toml
 │   ├── Dockerfile
 │   ├── alembic.ini
 │   ├── alembic/
 │   └── src/orchestrator/
 │       ├── main.py                   # Entrypoint: FastAPI + gRPC + mDNS en async
 │       ├── config.py                 # Pydantic Settings (env vars EO_*)
 │       ├── generated/                # Code genere par protoc
 │       ├── db/
 │       │   ├── engine.py             # SQLAlchemy async engine
 │       │   ├── models.py             # ORM: Device table
 │       │   └── repositories.py       # CRUD async
 │       ├── api/
 │       │   ├── app.py                # FastAPI factory
 │       │   └── routes/
 │       │       ├── devices.py        # GET/DELETE /api/v1/devices
 │       │       └── health.py         # GET /health
 │       ├── grpc_server/
 │       │   ├── server.py             # gRPC async server setup
 │       │   ├── device_service.py     # Register/Unregister/ListDevices
 │       │   ├── heartbeat_service.py  # Bidirectional streaming heartbeat
 │       │   └── model_service.py      # Upload/Download model (stub)
 │       ├── discovery/
 │       │   └── mdns.py               # Bonjour: _edgeorchestra._tcp
 │       └── services/
 │           ├── device_manager.py     # Logique metier device lifecycle
 │           └── heartbeat_monitor.py  # Detecte devices offline
 ├── control-plane/                    # CLI
 │   ├── pyproject.toml
 │   └── src/control_plane/
 │       ├── cli.py                    # Typer: eo discover/devices/ping/status
 │       ├── client.py                 # gRPC client wrapper
 │       └── api_client.py             # HTTP client (httpx)
 ├── ios-worker/                       # Placeholder (Phase 2)
 ├── dashboard/                        # Placeholder (Phase 2)
 ├── experiments/
 ├── papers/
 └── docs/


 ---
 Etapes d'Implementation (dans l'ordre)

 Etape 1 - Structure monorepo et config racine

 - pyproject.toml racine avec [tool.uv.workspace] (members: orchestrator, control-plane)
 - .gitignore (Python, Docker, protobuf genere, .env, .DS_Store)
 - .env.example avec toutes les variables EO_*
 - Makefile avec targets: proto-gen, dev-up, dev-down, lint, test, migrate
 - Creer les dossiers placeholder avec .gitkeep
 - Verification: uv sync fonctionne

 Etape 2 - Schemas Protobuf

 - common.proto: DeviceId, DeviceCapabilities, BatteryInfo, BatteryState, DeviceStatus
 - device.proto: service DeviceRegistry (Register, Unregister, ListDevices)
 - heartbeat.proto: service HeartbeatService avec bidirectional streaming
 - model.proto: service ModelService (UploadModel streaming, DownloadModel, SubmitGradients)
 - Script de generation dans le Makefile
 - Verification: make proto-gen produit les fichiers _pb2.py et _pb2_grpc.py

 Etape 3 - Orchestrator: config + database

 - pyproject.toml orchestrator: FastAPI, grpcio, SQLAlchemy async, asyncpg, redis, zeroconf, structlog
 - config.py: Pydantic Settings avec prefix EO_
 - db/engine.py: create_async_engine + async_sessionmaker
 - db/models.py: table Device (id UUID, name, device_model, os_version, chip, battery, status, metrics,
 timestamps)
 - db/repositories.py: DeviceRepository (CRUD async)
 - Verification: imports fonctionnent, Settings() charge les env vars

 Etape 4 - Docker Compose

 - PostgreSQL 16 + Redis 7 (avec healthchecks)
 - Container orchestrator avec hot-reload (volume mount src/)
 - Volumes persistes: pgdata, redisdata, model-storage
 - Note: pour mDNS, network_mode: host necessaire OU run orchestrator nativement
 - Verification: docker compose up postgres redis - les deux healthy

 Etape 5 - Alembic + migration initiale

 - alembic.ini + alembic/env.py configure pour async
 - Migration initiale qui cree la table devices
 - Verification: alembic upgrade head cree la table

 Etape 6 - FastAPI REST API

 - api/app.py: factory avec lifespan (init engine + sessions)
 - api/routes/health.py: GET /health
 - api/routes/devices.py: GET /api/v1/devices, GET /{id}, DELETE /{id}, GET /{id}/metrics
 - schemas/device.py: Pydantic v2 response models
 - Verification: curl localhost:8000/health retourne OK

 Etape 7 - Serveur gRPC

 - grpc_server/server.py: setup avec grpc.aio + reflection
 - grpc_server/device_service.py: Register, Unregister, ListDevices
 - grpc_server/heartbeat_service.py: bidirectional streaming (process heartbeat -> update Redis/DB ->
 respond with command or ACK)
 - grpc_server/model_service.py: stubs pour Phase 3
 - Verification: grpcurl -plaintext localhost:50051 list montre les services

 Etape 8 - main.py (entrypoint unifie)

 - Lance FastAPI (uvicorn), gRPC server, mDNS, et heartbeat monitor en parallele avec asyncio
 - Graceful shutdown
 - Verification: les deux serveurs repondent simultanement

 Etape 9 - mDNS Discovery

 - discovery/mdns.py: enregistre _edgeorchestra._tcp.local. avec IP locale et ports
 - Properties: api_port, grpc_port, version
 - Verification: dns-sd -B _edgeorchestra._tcp trouve le service

 Etape 10 - Heartbeat Monitor

 - services/heartbeat_monitor.py: process_heartbeat (Redis + DB), get_pending_command (Redis queue),
 run_stale_device_checker (background loop)
 - Devices marques offline apres 3 heartbeats manques (90s par defaut)
 - Verification: device auto-marque offline apres timeout

 Etape 11 - Docker Compose complet

 - Ajouter le container orchestrator au docker-compose.yml
 - Tester le stack complet
 - Verification: docker compose up - tout fonctionne

 Etape 12 - CLI Control Plane

 - cli.py avec Typer: commandes discover, devices, device, status, ping
 - client.py: wrapper gRPC
 - api_client.py: wrapper httpx pour REST
 - Output avec Rich (tables formatees)
 - Verification: eo ping, eo discover, eo devices fonctionnent

 ---
 Tech Stack

 ┌───────────────────────┬────────────────────────────────────────────────┐
 │       Composant       │                  Technologie                   │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Package manager       │ uv (workspace monorepo)                        │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ API REST              │ FastAPI + uvicorn                              │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Communication devices │ gRPC async (grpcio) + Protobuf                 │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Base de donnees       │ PostgreSQL 16 + SQLAlchemy 2.0 async + asyncpg │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Cache / queue         │ Redis 7                                        │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Discovery             │ python-zeroconf (Bonjour/mDNS)                 │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Migrations            │ Alembic                                        │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ CLI                   │ Typer + Rich                                   │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Logging               │ structlog                                      │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Containerisation      │ Docker Compose                                 │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Linting               │ Ruff                                           │
 ├───────────────────────┼────────────────────────────────────────────────┤
 │ Types                 │ mypy + mypy-protobuf                           │
 └───────────────────────┴────────────────────────────────────────────────┘

 ---
 Verification Finale

 1. docker compose up lance Postgres + Redis + Orchestrator
 2. eo ping confirme que l'orchestrateur repond
 3. eo discover trouve le service via mDNS
 4. grpcurl -plaintext localhost:50051 list montre les 3 services gRPC
 5. curl localhost:8000/api/v1/devices retourne une liste vide []
 6. curl localhost:8000/health retourne {"status": "ok"}