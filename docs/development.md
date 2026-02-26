# Development Guide

This document covers development setup, project conventions, testing, and contributing guidelines.

---

## Development Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended; CPU fallback available)
- Docker (for Qdrant and optional services)
- FFmpeg (for video processing)

### Quick Start

```bash
# 1. Clone and setup
cd video-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy environment config
cp .env.example .env
# Edit .env: set DEVICE=cuda (or cpu), PRELOAD_MODELS=false for dev

# 3. Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# 4. Start server with hot-reload
PRELOAD_MODELS=false python run.py
```

### Development Configuration

For fast iteration, use these `.env` settings:

```env
DEVICE=cuda           # or cpu
PRELOAD_MODELS=false  # Enable hot-reload, models load on first request
USE_RERANKER=false    # Saves VRAM during development
BATCH_SIZE=4
SAVE_FULL_FRAMES=false
```

With `PRELOAD_MODELS=false`, the server starts in seconds with hot-reload enabled. Models load lazily on the first search/ingest request.

### Hot Reload

When `PRELOAD_MODELS=false`, uvicorn runs with `--reload` and only watches code directories:
- `api/`
- `ingestion/`
- `search/`
- `scripts/`
- `config.py`

Model cache files and data directories are excluded to prevent reload loops.

---

## Project Structure Conventions

### Module Organization

```
video-rag/
├── api/              # FastAPI application layer
│   ├── models/       # Pydantic schemas (request/response validation)
│   └── routes/       # Endpoint handlers (thin controllers)
├── ingestion/        # Data processing layer
├── search/           # Search and retrieval layer
├── observability/    # Cross-cutting concerns (tracing, monitoring)
├── scripts/          # CLI tools (not imported by server)
├── frontend/         # Static web UI
├── docker/           # Container configuration
├── examples/         # Example data and scripts
├── tests/            # Test suite
└── docs/             # Documentation
```

### Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Singleton** | `get_*_service()` functions | Share expensive resources (models, connections) |
| **Factory** | `embedding_factory.py` | Choose local vs remote embedding based on config |
| **Pipeline** | `IngestPipeline` | Orchestrate multi-step ingestion workflow |
| **Strategy** | `EmbeddingService` / `RemoteEmbeddingClient` | Swap embedding backends |

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase` (e.g., `EmbeddingService`, `VectorStore`)
- **Functions**: `snake_case` (e.g., `embed_text`, `search_image`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `EMBEDDING_DIM`, `MAX_TOP_K`)
- **Routes**: Kebab-style paths (e.g., `/search/text`, `/ingest/drone-footage`)
- **Queue names**: Dot-separated (e.g., `embedding.jobs`)

### Configuration Pattern

All configuration lives in `config.py`:
1. Environment variables are read with `os.getenv()`
2. Sensible defaults are provided
3. Values are exported as module-level constants
4. Other modules import from `config` directly

```python
# Other modules:
import config
print(config.QDRANT_HOST)  # "localhost"
```

---

## Key Workflows

### Adding a New Search Type

1. **Add request model** in `api/models/requests.py`:
   ```python
   class MyNewSearchRequest(BaseModel):
       query: str
       top_k: int = 20
   ```

2. **Add response handling** (use existing `SearchResponse` if applicable)

3. **Add search method** in `search/search_service.py`:
   ```python
   def search_my_type(self, query, top_k, filters):
       embedding = self.embedding_service.embed_text(query)
       return self.vector_store.search(embedding, top_k, filters)
   ```

4. **Add route** in `api/routes/search.py`:
   ```python
   @router.post("/my-type", response_model=SearchResponse)
   async def search_my_type(request: MyNewSearchRequest):
       ...
   ```

5. **Add frontend tab** in `frontend/index.html` and handler in `frontend/app.js`

### Adding a New Ingestion Source

1. **Add processing method** in `ingestion/ingest_pipeline.py`
2. **Add route** in `api/routes/ingest.py`
3. **Add request model** in `api/models/requests.py`
4. **Update worker** in `worker.py` if async processing is needed

### Adding a New Filter

1. **Add to `SearchFilters`** in `api/models/requests.py`
2. **Add filter condition** in `search/vector_store.py` `search()` method
3. **Add payload index** in `vector_store.py` `_create_collection()` (for performance)
4. **Add to filter conversion** in `api/routes/search.py` `filters_to_dict()`
5. **Add to frontend** filter panel in `index.html`

---

## Testing

### Test Directory

```
tests/
└── __init__.py    # Currently a placeholder
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_search.py

# Run with coverage
pytest --cov=api --cov=search --cov=ingestion
```

### Writing Tests

Tests can use `httpx` for API testing:

```python
import pytest
from httpx import AsyncClient
from api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_text_search(client):
    response = await client.post("/search/text", json={
        "query": "test query",
        "top_k": 5
    })
    assert response.status_code == 200
    assert "results" in response.json()
```

### Mock Embeddings for Testing

When the Qwen3-VL model is unavailable, the `EmbeddingService` automatically falls back to mock embeddings (random normalized vectors). This allows testing the full pipeline without a GPU:

```python
from ingestion.embedding_service import get_embedding_service, reset_embedding_service

# Force mock mode for testing
service = get_embedding_service()
assert service.is_mock()  # True if model failed to load
```

---

## Debugging

### Server Logs

The FastAPI server prints detailed startup information:
```
==============================================================
Visual Search Engine for Security Operations
==============================================================
Starting server at http://0.0.0.0:8000
Dashboard: http://0.0.0.0:8000/static/index.html
API Docs: http://0.0.0.0:8000/docs
==============================================================
[1/3] Loading embedding model...
  Using float16 precision
  Model loaded successfully on cuda!
[2/3] Reranker disabled
[3/3] Connecting to Qdrant...
  Collection 'security_footage' already exists
CUDA memory - Allocated: 3.45 GB, Reserved: 3.72 GB
==============================================================
All models loaded and ready! Server is accepting requests.
==============================================================
```

### Worker Logs

Worker logs go to both stdout and `logs/worker.log`:
```bash
# Tail worker logs
tail -f logs/worker.log
```

### API Interactive Docs

FastAPI auto-generates interactive documentation:
- **Swagger UI**: http://localhost:8000/docs (try requests directly)
- **ReDoc**: http://localhost:8000/redoc (read-only reference)

### CUDA Debugging

```python
import torch
print(torch.cuda.is_available())           # True/False
print(torch.cuda.get_device_name(0))       # GPU name
print(torch.cuda.memory_allocated() / 1e9) # Allocated GB
print(torch.cuda.memory_reserved() / 1e9)  # Reserved GB
```

### Qdrant Debugging

```bash
# List collections
curl http://localhost:6333/collections

# Collection info
curl http://localhost:6333/collections/security_footage

# Count points
curl http://localhost:6333/collections/security_footage/points/count

# Scroll through points (see payloads)
curl -X POST http://localhost:6333/collections/security_footage/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 5, "with_payload": true}'
```

### Clear and Reset

```bash
# Clear Qdrant collection
python scripts/clear_qdrant.py

# Reset RabbitMQ queue
bash scripts/rabbitmq_reset.sh

# Reset embedding service singleton (in Python)
from ingestion.embedding_service import reset_embedding_service
reset_embedding_service()
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0.0 | PyTorch deep learning framework |
| `torchvision` | >= 0.15.0 | Computer vision utilities |
| `transformers` | >= 4.40.0 | Hugging Face model loading |
| `accelerate` | >= 0.25.0 | Training/inference acceleration |
| `qwen-vl-utils` | latest | Qwen VL model utilities |
| `qdrant-client` | >= 1.7.0 | Qdrant vector database client |
| `fastapi` | >= 0.109.0 | Web framework |
| `uvicorn[standard]` | >= 0.27.0 | ASGI server |
| `python-multipart` | >= 0.0.9 | File upload handling |
| `opencv-python` | >= 4.9.0 | Video/image processing |
| `Pillow` | >= 10.0.0 | Image manipulation |
| `pydantic` | >= 2.5.0 | Data validation |
| `python-dotenv` | >= 1.0.0 | Environment variable management |
| `pika` | >= 1.3.0 | RabbitMQ client |
| `langfuse` | >= 3.0.0 | Observability/tracing |
| `tqdm` | >= 4.66.0 | Progress bars |
| `numpy` | >= 1.24.0 | Numerical computing |
| `requests` | >= 2.31.0 | HTTP client |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 7.4.0 | Testing framework |
| `httpx` | >= 0.25.0 | Async HTTP client for API testing |

### Adding Dependencies

```bash
# Install and add to requirements.txt
pip install package-name
pip freeze | grep package-name >> requirements.txt
```

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/ingest_data.py` | CLI for video/image ingestion |
| `scripts/run_demo.py` | Run 15 demo search scenarios |
| `scripts/clear_qdrant.py` | Delete Qdrant collection |
| `scripts/publish_embedding_job.py` | Publish jobs to RabbitMQ |
| `scripts/remove_embedding_traces.py` | Remove Langfuse tracing from code |
| `scripts/docker-start.sh` | Docker deployment launcher |
| `scripts/start-batched-server.sh` | Start batched model server |
| `scripts/start-worker-cpu.sh` | Start CPU worker |
| `scripts/rabbitmq_reset.sh` | Reset RabbitMQ queue |

---

## Observability

### Langfuse Integration

The `observability/langfuse_integration.py` module provides:

| Decorator/Function | Purpose |
|-------------------|---------|
| `@observe(name, operation_type)` | General function tracing |
| `@trace_embedding_generation(model, type)` | Embedding operation tracing |
| `@trace_search(type, top_k)` | Search operation tracing |
| `@trace_ingestion(source_type)` | Ingestion operation tracing |
| `trace_operation()` | Context manager for custom traces |
| `flush_langfuse()` | Flush pending events |

### Enabling Langfuse

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

When disabled, all tracing decorators and context managers become no-ops with zero overhead.
