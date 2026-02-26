# Configuration Reference

All configuration is managed through environment variables, loaded via `python-dotenv` from a `.env` file or system environment. The central configuration module is `config.py`.

---

## Quick Setup

```bash
# Copy the template
cp .env.example .env

# Edit with your settings
nano .env
```

---

## Environment Variables

### Qdrant (Vector Database)

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `security_footage` | Name of the vector collection |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-VL-Embedding-2B` | Hugging Face model ID for embeddings |
| `RERANKER_MODEL_NAME` | `Qwen/Qwen3-VL-Reranker-2B` | Hugging Face model ID for reranker |
| `DEVICE` | `cuda` | Compute device: `cuda`, `cpu`, or `mps` |
| `BATCH_SIZE` | `8` | Batch size for embedding generation |
| `USE_RERANKER` | `false` | Enable/disable the reranker model |
| `PRELOAD_MODELS` | `true` | Preload models at server startup |

**Notes:**
- `EMBEDDING_DIM` is hardcoded to `2048` (Qwen3-VL output dimension)
- FP16 is automatically enabled when `DEVICE=cuda`
- Setting `PRELOAD_MODELS=false` enables hot-reload for development

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |

### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Root data directory |
| `RAW_DATA_DIR` | `./data/raw` | Raw video/image storage |
| `FRAMES_DIR` | `./data/frames` | Full-resolution extracted frames |
| `THUMBNAILS_DIR` | `./data/thumbnails` | 224x224 thumbnail images |

All directories are created automatically at startup if they don't exist.

### Frame Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_RATE` | `1.0` | Frames per second to extract from videos |
| `THUMBNAIL_SIZE` | `224` | Thumbnail dimension (pixels, square) |
| `SAVE_FULL_FRAMES` | `true` | Save full-resolution frames for popup view |

### Long Video Optimization

These settings apply to videos longer than `LONG_VIDEO_THRESHOLD_SECONDS`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LONG_VIDEO_THRESHOLD_SECONDS` | `3600` | Threshold (seconds) for "long video" mode |
| `LONG_VIDEO_FRAME_RATE` | `0.5` | Reduced FPS for long videos |
| `SEMANTIC_CLIP_DURATION` | `4.0` | Duration of each semantic clip (seconds) |
| `SEMANTIC_CLIP_MAX_FRAMES` | `32` | Max frames per semantic clip (prevents OOM) |
| `RERANKER_BATCH_SIZE` | `8` | Batch size for reranker inference |

### Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TOP_K` | `20` | Default number of search results |
| `MAX_TOP_K` | `100` | Maximum allowed top_k value |

These are set in `config.py` and not configurable via environment variables.

### RabbitMQ (Message Queue)

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_HOST` | *(empty)* | RabbitMQ server hostname |
| `RABBITMQ_PORT` | *(empty)* | RabbitMQ server port (default: 5672) |
| `RABBITMQ_USER` | *(empty)* | RabbitMQ username |
| `RABBITMQ_PASSWORD` | *(empty)* | RabbitMQ password |
| `RABBITMQ_QUEUE` | *(empty)* | Queue name (default: `embedding.jobs`) |
| `RABBITMQ_PREFETCH_COUNT` | *(empty)* | Messages to prefetch per worker |

The worker defaults to `localhost:5672` / `guest:guest` / `embedding.jobs` / `1` if not set.

### Langfuse (Observability)

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_ENABLED` | *(empty)* | Enable Langfuse tracing (`true`/`false`) |
| `LANGFUSE_PUBLIC_KEY` | *(empty)* | Langfuse public API key |
| `LANGFUSE_SECRET_KEY` | *(empty)* | Langfuse secret API key |
| `LANGFUSE_BASE_URL` | *(empty)* | Langfuse server URL |

### Hugging Face Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `./models/hf_cache` | Hugging Face home directory |
| `HF_HUB_CACHE` | `./models/hf_cache` | Hugging Face hub cache directory |

Using local directories avoids permission issues and keeps model weights with the project.

### Embedding Mode (for distributed workers)

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODE` | *(not set)* | `local` for on-machine, `remote` for model server |
| `MODEL_SERVER_URL` | *(not set)* | URL of the remote model server (e.g., `http://gpu:8001`) |

---

## Configuration Module (`config.py`)

The `config.py` file is the central configuration module. It:

1. Loads `.env` file via `python-dotenv`
2. Reads environment variables with sensible defaults
3. Creates required directories at import time
4. Exports typed configuration constants

```python
# Example usage in other modules
import config

print(config.QDRANT_HOST)        # "localhost"
print(config.MODEL_NAME)         # "Qwen/Qwen3-VL-Embedding-2B"
print(config.EMBEDDING_DIM)      # 2048
print(config.FRAMES_DIR)         # Path("./data/frames")
```

---

## Configuration Profiles

### Development Profile

Optimized for fast iteration:

```env
DEVICE=cuda
BATCH_SIZE=4
PRELOAD_MODELS=false     # Enable hot-reload
USE_RERANKER=false       # Skip reranker for speed
FRAME_RATE=1.0
SAVE_FULL_FRAMES=false   # Save disk space
```

### Production Profile

Optimized for quality and performance:

```env
DEVICE=cuda
BATCH_SIZE=8
PRELOAD_MODELS=true      # Pre-warm models
USE_RERANKER=true        # Enable reranker for better results
FRAME_RATE=1.0
SAVE_FULL_FRAMES=true    # Full-quality popup view
```

### Low-Memory GPU Profile (< 8GB VRAM)

```env
DEVICE=cuda
BATCH_SIZE=2
PRELOAD_MODELS=true
USE_RERANKER=false       # Saves ~2GB VRAM
FRAME_RATE=0.5
SEMANTIC_CLIP_MAX_FRAMES=16
```

### CPU-Only Profile

```env
DEVICE=cpu
BATCH_SIZE=1
PRELOAD_MODELS=false     # Lazy load to save memory
USE_RERANKER=false
FRAME_RATE=0.5           # Reduce workload
```

### Distributed Worker Profile

```env
EMBEDDING_MODE=remote
MODEL_SERVER_URL=http://gpu-machine:8001
DEVICE=cpu               # Workers don't need GPU
RABBITMQ_HOST=rabbitmq.example.com
RABBITMQ_PORT=5672
RABBITMQ_USER=worker
RABBITMQ_PASSWORD=secret
RABBITMQ_QUEUE=embedding.jobs
RABBITMQ_PREFETCH_COUNT=1
```

---

## Tuning Guide

### Batch Size vs VRAM

| VRAM | Recommended `BATCH_SIZE` | Notes |
|------|-------------------------|-------|
| 4 GB | 1 | Very constrained |
| 8 GB | 2-4 | Typical consumer GPU |
| 12 GB | 4-8 | Good for most workloads |
| 16 GB | 8-16 | Comfortable headroom |
| 24 GB+ | 16-32 | High throughput |

### Frame Rate vs Quality

| `FRAME_RATE` | Frames/Hour | Use Case |
|-------------|-------------|----------|
| 2.0 | 7,200 | High-granularity analysis |
| 1.0 | 3,600 | Standard (recommended) |
| 0.5 | 1,800 | Long videos, storage-conscious |
| 0.25 | 900 | Very long videos, overview only |

### Semantic Clip Duration

| `SEMANTIC_CLIP_DURATION` | Use Case |
|-------------------------|----------|
| 2.0s | Fast-changing scenes |
| 4.0s | General surveillance (default) |
| 6.0s | Slow-moving drone footage |
| 10.0s | Static camera with rare events |
