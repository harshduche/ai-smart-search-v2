# Deployment Guide

This document covers all deployment options: local development, Docker containers, and distributed architectures.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | None (CPU fallback) | NVIDIA GPU, 8+ GB VRAM |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 10 GB | 50+ GB (for video data) |
| **CPU** | 4 cores | 8+ cores |

### Software Requirements

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- NVIDIA Container Toolkit (for GPU in Docker)
- FFmpeg (for video processing)

---

## Option 1: Local Development

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd video-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
# Option A: Docker (recommended)
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Option B: Check if running
curl http://localhost:6333/collections
```

### 3. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

Key settings to adjust:
- `DEVICE=cuda` (or `cpu` if no GPU)
- `BATCH_SIZE=4` (lower for less VRAM)
- `PRELOAD_MODELS=true` (set to `false` for faster dev restarts)

### 4. Start the Server

```bash
python run.py
```

Access:
- Dashboard: http://localhost:8000/static/index.html
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 5. Ingest Data

```bash
# Single video
python scripts/ingest_data.py /path/to/video.mp4 --zone main_gate

# Directory
python scripts/ingest_data.py /path/to/footage/ --stats-output stats.json

# From URL
python scripts/ingest_data.py "https://example.com/video.mp4" --from-url
```

---

## Option 2: Docker Deployment

### Production (Self-Hosted)

Uses `docker/docker-compose.self-hosted.yml` with full GPU support, health checks, and restart policies.

**Services:**
- RabbitMQ (message queue + management UI)
- Qdrant (vector database)
- API (FastAPI with GPU)
- Worker (embedding worker with GPU)

```bash
cd docker

# Start all services
docker compose -f docker-compose.self-hosted.yml up -d

# Check status
docker compose -f docker-compose.self-hosted.yml ps

# View logs
docker compose -f docker-compose.self-hosted.yml logs -f

# Scale workers
docker compose -f docker-compose.self-hosted.yml up -d --scale worker=3
```

**Service URLs:**
| Service | URL |
|---------|-----|
| API / Dashboard | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| RabbitMQ Management | http://localhost:15672 (guest/guest) |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### Development Mode

Uses `docker/docker-compose.dev.yml` with hot-reload, volume mounts for live code changes, and `PRELOAD_MODELS=false` for fast restarts.

```bash
cd docker

# Start development stack
docker compose -f docker-compose.dev.yml up -d

# Code changes are reflected immediately via volume mounts
```

### Standalone Worker

Uses `docker/docker-compose.worker.yml` for running just the worker container, connecting to external RabbitMQ and Qdrant services.

```bash
cd docker

# Configure .env with external service addresses
# RABBITMQ_HOST=rabbitmq.example.com
# QDRANT_HOST=qdrant.example.com

docker compose -f docker-compose.worker.yml up -d
```

### CPU-Only Deployment

For machines without NVIDIA GPUs, use the `Dockerfile.cpu`:

1. Edit the compose file to use `Dockerfile.cpu` instead of `Dockerfile`
2. Set `DEVICE=cpu` in environment
3. Remove GPU device reservations from compose file

---

## Option 3: Distributed Architecture

For high-throughput ingestion, separate the GPU-intensive model serving from CPU-based workers.

### Architecture

```
┌─────────────────────────────────────────────┐
│              Model Server (GPU)              │
│  Hosts: Qwen3-VL-Embedding-2B               │
│  Port: 8001                                  │
│  Handles: /embed/text, /embed/image, etc.    │
└──────────────────┬──────────────────────────┘
                   │ HTTP
        ┌──────────┴──────────┐
        │                     │
┌───────▼──────┐      ┌──────▼───────┐
│  Worker 1    │      │  Worker 2    │
│  (CPU)       │      │  (CPU)       │
│  RabbitMQ    │      │  RabbitMQ    │
│  Consumer    │      │  Consumer    │
└──────────────┘      └──────────────┘
```

### Step 1: Start the Model Server

```bash
# Standard model server
python model_server.py

# Or batched model server (better GPU utilization)
python model_server_batched.py

# Or use the helper script
bash scripts/start-batched-server.sh
```

The model server listens on port 8001 by default.

### Step 2: Start CPU Workers

```bash
# Set environment for remote embedding
export EMBEDDING_MODE=remote
export MODEL_SERVER_URL=http://gpu-machine:8001

# Start worker
python worker.py

# Or use the helper script
bash scripts/start-worker-cpu.sh
```

### Step 3: Publish Jobs

```bash
# From any machine with RabbitMQ access
python scripts/publish_embedding_job.py \
  --video-url "https://example.com/video.mp4" \
  --metadata '{"zone": "perimeter", "site_id": "site_a"}'

# Batch publish
python scripts/publish_embedding_job.py \
  --batch-file examples/sample_jobs.json
```

---

## Docker Management Script

The `docker/docker-manage.sh` script provides convenient commands for all Docker operations:

```bash
cd docker

# Production
./docker-manage.sh start              # Start production stack
./docker-manage.sh stop               # Stop all services
./docker-manage.sh restart             # Restart services
./docker-manage.sh logs                # View all logs
./docker-manage.sh logs-worker         # View worker logs only

# Development
./docker-manage.sh start-dev           # Start dev stack (hot reload)

# Scaling
./docker-manage.sh scale 3             # Scale to 3 workers
./docker-manage.sh scale-worker 5      # Scale standalone workers

# Monitoring
./docker-manage.sh status              # Service status
./docker-manage.sh health              # Health checks
./docker-manage.sh stats               # Resource usage (docker stats)
./docker-manage.sh gpu-check           # Check GPU availability

# Maintenance
./docker-manage.sh build               # Rebuild images
./docker-manage.sh clean               # Remove all containers/volumes
./docker-manage.sh publish-job         # Publish test job
./docker-manage.sh shell-api           # Shell into API container
./docker-manage.sh shell-worker        # Shell into worker container
```

---

## Remote Access

### SSH Tunnel

For accessing the dashboard from your local machine when the server runs on a remote GPU machine:

```bash
# Forward port 8000
ssh -L 8000:localhost:8000 user@gpu-machine

# Then open in browser
http://localhost:8000/static/index.html
```

### Multi-Port Forwarding

```bash
ssh -L 8000:localhost:8000 \  # API
    -L 6333:localhost:6333 \  # Qdrant
    -L 15672:localhost:15672 \ # RabbitMQ
    user@gpu-machine
```

---

## Dockerfiles

### `docker/Dockerfile` (GPU)

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Includes: Python 3.10, FFmpeg, Git, system libraries
- Installs all Python dependencies
- Creates data directories
- Health check against `/health` endpoint
- Default command: `python3 run.py`

### `docker/Dockerfile.cpu` (CPU)

- Base: `python:3.10-slim`
- Same structure but without CUDA
- Sets `DEVICE=cpu` by default
- Suitable for workers using remote model server

---

## Health Checks

### Manual Health Verification

```bash
# API health
curl http://localhost:8000/health

# Qdrant health
curl http://localhost:6333/collections

# RabbitMQ health (if using workers)
curl http://localhost:15672/api/health/checks/alarms \
  -u guest:guest

# Model server health (if using distributed mode)
curl http://localhost:8001/health
```

### Docker Health Checks

All Docker services include built-in health checks:
- **API**: HTTP GET to `/health` (120s start period for model loading)
- **Qdrant**: HTTP GET to `/` 
- **RabbitMQ**: `rabbitmq-diagnostics ping`
- **Worker**: `pgrep -f worker.py`

---

## Troubleshooting

### Common Issues

**Qdrant connection error:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker restart qdrant

# Check Qdrant logs
docker logs qdrant
```

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in `.env` (try 2 or 1)
- Use `--batch-size 2` flag in ingest scripts
- Enable semantic clips to reduce per-item memory
- Use FP16 (enabled by default on CUDA)

**Model not loading:**
```bash
# Check internet (first run downloads ~4GB model)
curl -I https://huggingface.co

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check disk space for model cache
df -h ~/.cache/huggingface/
```

**Worker not connecting to RabbitMQ:**
```bash
# Check RabbitMQ is accessible
curl http://localhost:15672

# Verify credentials in .env
# RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USER, RABBITMQ_PASSWORD
```

**Slow embedding generation:**
- Ensure GPU is being used: check `DEVICE=cuda` in `.env`
- Enable model preloading: `PRELOAD_MODELS=true`
- Use batched model server for distributed workloads
- Check GPU utilization: `nvidia-smi`
