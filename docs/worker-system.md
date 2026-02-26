# Worker System

This document covers the distributed worker architecture, RabbitMQ integration, job format, model servers, and scaling strategies.

---

## Overview

The worker system enables **distributed, asynchronous video ingestion** by decoupling job submission from processing. Jobs are published to a RabbitMQ queue and consumed by worker processes that can run on separate machines.

```
┌───────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Job Publisher  │     │  RabbitMQ    │     │  Worker Process  │
│ (API / Script) │────▶│  Queue:      │────▶│  (GPU or CPU)    │
│               │     │  embedding.  │     │                  │
│               │     │  jobs        │     │  Download video  │
└───────────────┘     └──────────────┘     │  Extract frames  │
                                           │  Generate embeds │
                                           │  Store in Qdrant │
                                           └──────────────────┘
```

---

## Worker Process (`worker.py`)

### `EmbeddingWorker` Class

The main worker class that:
1. Initializes ML models and services at startup
2. Connects to RabbitMQ with automatic reconnection
3. Consumes messages from the `embedding.jobs` queue
4. Processes each video ingestion job
5. Acknowledges or rejects messages based on success/failure
6. Tracks statistics (jobs processed, failed, success rate)
7. Integrates with Langfuse for observability tracing

### Lifecycle

```
1. Worker Start
   └─▶ Initialize Services
       ├── Load Embedding Model (or connect to model server)
       ├── Connect to Qdrant Vector Store
       └── Create Ingestion Pipeline
   └─▶ Connect to RabbitMQ
       ├── Declare queue (durable, 24h TTL, max 10k messages)
       └── Set prefetch count (QoS)
   └─▶ Start Consuming Messages
       ├── Parse JSON job data
       ├── Process job (download → extract → embed → store)
       ├── ACK on success
       └── NACK + requeue on failure (NACK + no requeue for invalid JSON)

2. Worker Stop (SIGINT / SIGTERM)
   └─▶ Stop consuming
   └─▶ Close RabbitMQ connection
   └─▶ Flush Langfuse traces
   └─▶ Print statistics
```

### Running the Worker

```bash
# Default configuration (localhost RabbitMQ)
python worker.py

# With custom configuration via environment
RABBITMQ_HOST=rabbitmq.example.com \
RABBITMQ_PORT=5672 \
RABBITMQ_USER=worker \
RABBITMQ_PASSWORD=secret \
DEVICE=cuda \
python worker.py
```

---

## Job Format

### Job Message Structure

Jobs are JSON messages published to the `embedding.jobs` RabbitMQ queue:

```json
{
  "video_url": "https://s3.amazonaws.com/bucket/video.mp4?signature=...",
  "metadata": {
    "zone": "perimeter_1",
    "organization_id": "org_123",
    "site_id": "site_456",
    "drone_id": "drone_789",
    "flight_id": "flight_001",
    "clip_duration": 4.0,
    "use_semantic_clips": true,
    "max_frames_per_clip": 32,
    "batch_size": 100,
    "save_full_frames": false,
    "cleanup_after": true
  }
}
```

### Job Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_url` | string | Yes | Video URL (HTTP/HTTPS/S3) or local filesystem path |
| `metadata` | object | No | Job configuration and metadata |

### Metadata Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `zone` | string | "unknown" | Zone/location identifier |
| `use_semantic_clips` | bool | true | Use semantic clip mode |
| `clip_duration` | float | 4.0 | Clip duration in seconds |
| `max_frames_per_clip` | int | 32 | Max frames per clip |
| `batch_size` | int | 100 | Qdrant batch insert size |
| `save_full_frames` | bool | false | Save full-resolution frames |
| `cleanup_after` | bool | true | Delete downloaded file after |
| `organization_id` | string | — | Organization identifier |
| `site_id` | string | — | Site identifier |
| `drone_id` | string | — | Drone identifier |
| `flight_id` | string | — | Flight identifier |

---

## Publishing Jobs

### Using the CLI Script

```bash
# Single job
python scripts/publish_embedding_job.py \
  --video-url "https://example.com/video.mp4" \
  --metadata '{"zone": "perimeter", "site_id": "site_a"}'

# Batch from JSON file
python scripts/publish_embedding_job.py \
  --batch-file examples/sample_jobs.json

# Custom queue and priority
python scripts/publish_embedding_job.py \
  --video-url "https://example.com/video.mp4" \
  --queue embedding.jobs.high_priority
```

### Batch File Format (`examples/sample_jobs.json`)

```json
[
  {
    "video_url": "https://s3.amazonaws.com/bucket/site_a_morning.mp4",
    "metadata": {
      "organization_id": "flytbase",
      "site_id": "construction_site_a",
      "drone_id": "dji_mavic_001",
      "flight_id": "flight_20240207_001",
      "zone": "north_perimeter",
      "use_semantic_clips": true,
      "clip_duration": 4.0
    }
  },
  {
    "video_url": "https://s3.amazonaws.com/bucket/site_b_evening.mp4",
    "metadata": {
      "organization_id": "flytbase",
      "site_id": "warehouse_b",
      "drone_id": "dji_mavic_002",
      "flight_id": "flight_20240207_002",
      "zone": "loading_dock"
    }
  }
]
```

### Publishing from Python

```python
import json
import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost')
)
channel = connection.channel()
channel.queue_declare(queue='embedding.jobs', durable=True)

job = {
    "video_url": "https://example.com/video.mp4",
    "metadata": {"zone": "perimeter", "site_id": "site_a"}
}

channel.basic_publish(
    exchange='',
    routing_key='embedding.jobs',
    body=json.dumps(job),
    properties=pika.BasicProperties(delivery_mode=2)  # Persistent
)
```

---

## Message Acknowledgment

The worker uses **manual acknowledgment** for reliability:

| Scenario | Action | Message Fate |
|----------|--------|-------------|
| Success | `basic_ack` | Removed from queue |
| Processing error | `basic_nack(requeue=True)` | Returned to queue for retry |
| Invalid JSON | `basic_nack(requeue=False)` | Discarded (goes to DLQ if configured) |

### Queue Configuration

```python
channel.queue_declare(
    queue='embedding.jobs',
    durable=True,            # Survives broker restart
    arguments={
        'x-message-ttl': 86400000,  # 24-hour message TTL
        'x-max-length': 10000,      # Max 10,000 messages
    }
)
```

---

## Model Server

For distributed deployments where GPU resources should be shared across multiple CPU workers.

### Standard Model Server (`model_server.py`)

Serves the Qwen3-VL model over HTTP with endpoints for each embedding type.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed/text` | POST | Text embedding |
| `/embed/image` | POST | Single image embedding (base64) |
| `/embed/video-clip` | POST | Video clip embedding (list of base64 frames) |
| `/embed/multimodal` | POST | Text + image embedding |
| `/embed/text/batch` | POST | Batch text embedding |
| `/embed/image/batch` | POST | Batch image embedding |
| `/health` | GET | Health check |

**Running:**

```bash
python model_server.py
# Listens on port 8001
```

### Batched Model Server (`model_server_batched.py`)

Optimized version that **batches incoming requests** for better GPU utilization.

**How Batching Works:**

1. Requests arrive individually from multiple workers
2. A `BatchProcessor` collects requests into a queue
3. After a time window (default: 50ms) or reaching max batch size, requests are processed together
4. Results are dispatched back to individual callers via futures

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Batch window | 50ms | Time to collect requests before processing |
| Max batch size | 8 | Maximum requests per batch |
| Separate queues | Yes | Image and video clip requests batched separately |

**Additional Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | Batch processing statistics |

**Running:**

```bash
python model_server_batched.py

# Or use the optimized script (12GB VRAM settings)
bash scripts/start-batched-server.sh
```

**Statistics Response:**

```json
{
  "total_requests": 1523,
  "total_batches": 245,
  "average_batch_size": 6.2,
  "queue_size": 0
}
```

---

## Scaling Strategies

### Vertical Scaling (Single Machine)

```bash
# Increase batch size for better GPU utilization
BATCH_SIZE=16 python worker.py

# Use batched model server
python model_server_batched.py
```

### Horizontal Scaling (Multiple Workers)

```bash
# Docker Compose scaling
docker compose -f docker-compose.self-hosted.yml up -d --scale worker=5

# Or standalone workers on separate machines
# Machine 1:
RABBITMQ_HOST=queue.example.com python worker.py

# Machine 2:
RABBITMQ_HOST=queue.example.com python worker.py
```

### GPU + CPU Separation

The most efficient architecture for expensive GPU resources:

```
GPU Machine:
  └─ model_server_batched.py (port 8001)

CPU Machine 1:
  └─ worker.py (EMBEDDING_MODE=remote, MODEL_SERVER_URL=http://gpu:8001)

CPU Machine 2:
  └─ worker.py (EMBEDDING_MODE=remote, MODEL_SERVER_URL=http://gpu:8001)
```

### Fair Dispatch

With `prefetch_count=1`, RabbitMQ ensures fair dispatch:
- Each worker processes one job at a time
- A busy worker won't receive new jobs until it ACKs the current one
- Jobs are distributed round-robin to available workers

---

## Observability

### Langfuse Tracing

Each worker job is traced with Langfuse (when enabled):

```python
with trace_operation(
    name="worker-process-job",
    operation_type="span",
    user_id=metadata.get("organization_id"),
    session_id=metadata.get("flight_id"),
    metadata={"zone": zone, "site_id": site_id},
    tags=["worker", "ingestion"],
) as trace:
    # ... processing ...
    trace.update(output=result, status_message="completed")
```

**Traced Data:**
- Job input (video URL, metadata)
- Processing time
- Frames/clips processed
- Errors (if any)
- Organization, site, and flight grouping

### Worker Statistics

On shutdown, the worker prints:
```
Worker Statistics:
  - Uptime: 3600.00s (60.00 minutes)
  - Jobs processed: 42
  - Jobs failed: 2
  - Success rate: 95.5%
```

### Logging

Worker logs are written to both stdout and `logs/worker.log`:

```
2024-02-07 10:30:15 - worker - INFO - Processing video: https://example.com/video.mp4
2024-02-07 10:30:15 - worker - INFO - Metadata: {"zone": "perimeter", ...}
2024-02-07 10:35:42 - worker - INFO - ✓ Successfully processed video
2024-02-07 10:35:42 - worker - INFO -   - Frames/Clips: 450
2024-02-07 10:35:42 - worker - INFO -   - Processing time: 327.15s
```

---

## RabbitMQ Management

### Reset Queue

```bash
# Using the helper script
bash scripts/rabbitmq_reset.sh

# Or manually
docker exec video-rag-rabbitmq rabbitmqctl delete_queue embedding.jobs
```

### Monitor Queue

Access the RabbitMQ Management UI at http://localhost:15672 (guest/guest):
- View queue depth and message rates
- Monitor consumer connections
- Inspect individual messages
- Purge queues if needed

### Queue Properties

| Property | Value | Description |
|----------|-------|-------------|
| `durable` | true | Queue survives broker restarts |
| `x-message-ttl` | 86400000 | 24-hour message expiry |
| `x-max-length` | 10000 | Max queue depth |
| `delivery_mode` | 2 | Persistent messages (survive restart) |
