# System Architecture

This document provides a deep dive into the architecture of the Visual Search Engine for Security Operations.

---

## High-Level Overview

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  Web Dashboard (HTML/CSS/JS)  │  REST API (FastAPI + OpenAPI)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      SERVICE LAYER                               │
│  SearchService  │  EmbeddingService  │  IngestPipeline          │
│  RerankerService│  EmbeddingFactory  │  VideoProcessor          │
└─────────┬───────────────┬────────────────────┬──────────────────┘
          │               │                    │
┌─────────▼───────┐ ┌────▼──────────┐ ┌───────▼──────────────────┐
│  STORAGE LAYER  │ │  ML LAYER     │ │  MESSAGE QUEUE LAYER     │
│  Qdrant Vector  │ │  Qwen3-VL-2B  │ │  RabbitMQ                │
│  Store          │ │  (+ Reranker) │ │  (embedding.jobs queue)  │
│  Static Files   │ │  Model Server │ │  Worker Process(es)      │
└─────────────────┘ └───────────────┘ └──────────────────────────┘
```

---

## Component Architecture

### 1. FastAPI Backend (`api/`)

The API layer is the central gateway for all client interactions.

```
api/
├── main.py              # Application factory, lifespan, middleware
├── models/
│   ├── requests.py      # Pydantic request validation schemas
│   └── responses.py     # Pydantic response serialization schemas
└── routes/
    ├── health.py        # Health + stats endpoints
    ├── search.py        # All 4 search modalities
    └── ingest.py        # 5 ingestion endpoints + collection delete
```

**Key Design Decisions:**

- **Lifespan Context Manager**: Models are preloaded at startup (configurable via `PRELOAD_MODELS`) to eliminate cold-start latency on first request.
- **CORS Middleware**: Allows all origins for development; should be restricted in production.
- **Static File Mounts**: Thumbnails, frames, and raw videos are served directly by FastAPI at `/thumbnails/`, `/frames/`, and `/raw/` paths.
- **Singleton Services**: `SearchService`, `EmbeddingService`, `VectorStore`, and `RerankerService` all use singleton patterns for efficient resource sharing.

### 2. Ingestion Pipeline (`ingestion/`)

The ingestion subsystem handles the full lifecycle from raw media to stored vector embeddings.

```
ingestion/
├── ingest_pipeline.py         # Orchestrator: coordinates all steps
├── video_processor.py         # OpenCV-based frame extraction
├── image_processor.py         # Image loading + thumbnail generation
├── embedding_service.py       # High-level embedding API
├── embedding_factory.py       # Decides: local model vs remote server
├── qwen3_vl_model.py         # Low-level Qwen3-VL model wrapper
├── qwen3_vl_reranker.py      # Qwen3-VL reranker model wrapper
└── remote_embedding_client.py # HTTP client for model server
```

**Ingestion Flow:**

```
Input (Video/Image/URL/RTSP)
        │
        ▼
┌──────────────────┐
│ Video Processor   │ ─── Extract frames at configurable FPS
│ (or Image Proc.)  │ ─── Generate thumbnails (224x224)
└────────┬─────────┘     ─── Detect night/day (brightness analysis)
         │                ─── Save full-resolution frames (optional)
         ▼
┌──────────────────┐
│ Embedding Service │ ─── Convert frames → 2048-dim vectors
│ (Qwen3-VL-2B)    │ ─── Text, image, multimodal, video clip modes
└────────┬─────────┘     ─── Batch processing with CUDA memory mgmt
         │
         ▼
┌──────────────────┐
│ Vector Store      │ ─── Upsert to Qdrant with payload indexes
│ (Qdrant)          │ ─── Cosine similarity metric
└──────────────────┘     ─── Batch inserts (configurable batch size)
```

**Two Ingestion Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Frame-by-Frame** | Each frame gets its own embedding | Short videos, high granularity |
| **Semantic Clips** | Groups of frames (e.g., 4-second clips) get a single video embedding | Long videos (3+ hours), temporal context |

### 3. Search Engine (`search/`)

The search subsystem converts queries into vector lookups with optional reranking.

```
search/
├── search_service.py      # Orchestrates embed → search → rerank
├── vector_store.py        # Qdrant client wrapper
└── reranker_service.py    # Optional second-stage reranking
```

**Search Flow:**

```
Query (text / image / text+image / OCR text)
        │
        ▼
┌──────────────────┐
│ Embedding Service │ ─── Convert query to 2048-dim vector
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Vector Store      │ ─── Cosine similarity search in Qdrant
│ (Qdrant)          │ ─── Apply filters (zone, time, site, etc.)
└────────┬─────────┘     ─── Return top_k * 3 candidates (if reranking)
         │
         ▼
┌──────────────────┐
│ Reranker Service  │ ─── (Optional) Rerank with Qwen3-VL-Reranker
│ (Qwen3-VL-2B)    │ ─── Score each (query, result) pair
└────────┬─────────┘     ─── Return final top_k results
         │
         ▼
    Formatted Results
    (with thumbnails, timestamps, metadata)
```

### 4. Worker System (`worker.py`)

The distributed processing layer uses RabbitMQ for async job processing.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Job          │     │  RabbitMQ    │     │  Worker      │
│  Publisher    │────▶│  Queue       │────▶│  Process     │
│  (API/Script) │     │  embedding.  │     │  (GPU)       │
└──────────────┘     │  jobs        │     └──────┬───────┘
                     └──────────────┘            │
                                                 ▼
                                          Download → Extract
                                          → Embed → Store
```

**Worker Features:**
- Manual message acknowledgment (reliability)
- Automatic reconnection on RabbitMQ failures
- Langfuse tracing per job
- Statistics tracking (jobs processed, failed, success rate)
- Configurable prefetch count for fair dispatch

### 5. Model Server (`model_server.py`, `model_server_batched.py`)

For distributed deployments, a standalone HTTP server hosts the embedding model.

```
┌────────────────────────────────────────┐
│          Model Server (GPU)            │
│  /embed/text, /embed/image             │
│  /embed/video-clip, /embed/multimodal  │
│  /embed/text/batch, /embed/image/batch │
└───────────────┬────────────────────────┘
                │ HTTP (base64-encoded images)
        ┌───────┴──────────┐
        │                  │
  ┌─────▼──────┐    ┌─────▼──────┐
  │ Worker 1   │    │ Worker 2   │
  │ (CPU)      │    │ (CPU)      │
  └────────────┘    └────────────┘
```

**Batched Model Server** (`model_server_batched.py`):
- Collects requests within a configurable time window (default 50ms)
- Processes them as a single batch for GPU efficiency
- Separate queues for image and video clip requests
- Tracks statistics (total requests, batches, avg batch size)

---

## Data Flow Diagrams

### Ingestion Data Flow (URL-based)

```
1. Client POST /ingest/url
        │
2. Download video from URL → save to data/raw/
        │
3. OpenCV extracts frames at configured FPS
        │
4. For each frame/clip:
   a. Generate 224x224 thumbnail → data/thumbnails/
   b. (Optional) Save full frame → data/frames/
   c. Compute brightness → determine night/day
        │
5. Group frames into semantic clips (if enabled)
        │
6. Generate 2048-dim embedding per frame/clip
        │
7. Upsert embedding + metadata to Qdrant
        │
8. Return statistics to client
```

### Search Data Flow

```
1. Client POST /search/text
   Body: {"query": "person near fence", "top_k": 10}
        │
2. EmbeddingService.embed_text("person near fence")
   → 2048-dim float32 vector
        │
3. VectorStore.search(embedding, top_k=10)
   → Qdrant cosine similarity search
   → Apply any filters (zone, time, site)
        │
4. (Optional) RerankerService.rerank(query, results)
   → Score each (query, thumbnail) pair
   → Re-sort by reranker score
        │
5. Convert file paths to static serving URLs
        │
6. Return SearchResponse with results, scores, metadata
```

---

## Qdrant Collection Schema

**Collection Name:** `security_footage` (configurable)

**Vector Configuration:**
- Dimension: 2048
- Distance metric: Cosine

**Payload Indexes:**

| Field | Type | Purpose |
|-------|------|---------|
| `source_file` | Keyword | Filter by source video/image |
| `zone` | Keyword | Filter by zone/location |
| `timestamp` | Datetime | Temporal range filtering |
| `frame_number` | Integer | Frame-level lookup |
| `is_night` | Bool | Day/night filtering |
| `organization_id` | Keyword | Multi-org filtering |
| `site_id` | Keyword | Multi-site filtering |
| `drone_id` | Keyword | Drone-specific filtering |
| `flight_id` | Keyword | Flight-specific filtering |

**Payload Fields (not indexed):**

- `thumbnail_path` — Path to 224px thumbnail
- `frame_path` — Path to full-resolution frame
- `video_path` — Path to source video
- `seconds_offset` — Time offset in video
- `brightness` — Average pixel brightness
- `clip_index`, `clip_start_seconds`, `clip_end_seconds` — Semantic clip bounds
- `organization_name`, `site_name`, `drone_model`, `flight_purpose`, etc. — Multi-site metadata

---

## Singleton Pattern

The system makes heavy use of singletons to share expensive resources:

| Singleton | Module | Purpose |
|-----------|--------|---------|
| `get_embedding_service()` | `ingestion/embedding_service.py` | Share loaded ML model |
| `get_vector_store()` | `search/vector_store.py` | Share Qdrant connection |
| `get_search_service()` | `search/search_service.py` | Share search orchestrator |
| `get_reranker_service()` | `search/reranker_service.py` | Share reranker model |

All singletons are lazily initialized on first access and can be reset for testing.

---

## Scalability Considerations

### Vertical Scaling
- GPU VRAM is the primary bottleneck for embedding generation
- Batch size tuning: lower for smaller GPUs (4 for <16GB), higher for larger GPUs
- FP16 inference on CUDA halves memory usage with minimal quality loss

### Horizontal Scaling
- Multiple workers can consume from the same RabbitMQ queue
- Model server separates GPU workload from CPU workers
- Qdrant supports distributed mode for very large collections
- Docker Compose `--scale worker=N` for easy scaling

### Long Video Optimization
- Videos longer than `LONG_VIDEO_THRESHOLD_SECONDS` (default: 1 hour) automatically use reduced frame rates
- Semantic clips group multiple frames into single embeddings, reducing storage ~4x
- Periodic CUDA cache clearing prevents OOM during long processing runs
