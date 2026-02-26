# Video-RAG Documentation

> Comprehensive documentation for the **Visual Search Engine for Security Operations** — an AI-powered visual search system built with Qwen3-VL-Embedding, Qdrant, FastAPI, and RabbitMQ.

Built for the **FlytBase AI Hackathon 2026**.

---

## Table of Contents

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System architecture, component interactions, data flow diagrams |
| [API Reference](./api-reference.md) | Complete REST API endpoint documentation with examples |
| [Ingestion Pipeline](./ingestion-pipeline.md) | Video/image processing, frame extraction, embedding generation |
| [Search Engine](./search-engine.md) | Vector search internals, reranking, filtering strategies |
| [Worker System](./worker-system.md) | RabbitMQ workers, distributed processing, job format |
| [Deployment](./deployment.md) | Docker, local, and distributed deployment guides |
| [Configuration](./configuration.md) | All environment variables and configuration options |
| [Frontend](./frontend.md) | Web dashboard UI, features, and customization |
| [Development](./development.md) | Development setup, testing, coding conventions |

---

## Project Overview

### What is Video-RAG?

Video-RAG is a **Retrieval-Augmented Generation** system designed for security operations. It ingests video and image data from multiple sources (local files, URLs, RTSP streams), processes them into vector embeddings using a state-of-the-art vision-language model, and enables sub-second semantic search across the entire footage library.

### Core Capabilities

- **Text Search** — Natural language queries like *"person near fence at night"*
- **Image Search** — Upload a reference image to find visually similar scenes
- **Multimodal Search** — Combine text and image for precise, hybrid queries
- **OCR Search** — Find text visible in footage (license plates, signs, labels)
- **Semantic Video Clips** — Temporal understanding via grouped frame embeddings
- **Multi-Site Support** — Manage footage across organizations, sites, drones, and flights
- **RTSP Stream Ingestion** — Ingest live camera feeds in real-time
- **Distributed Processing** — Scale ingestion with RabbitMQ workers and model servers

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Model** | Qwen3-VL-Embedding-2B | Multimodal embeddings (text, image, video) |
| **Reranker** | Qwen3-VL-Reranker-2B | Result quality improvement (optional) |
| **Vector DB** | Qdrant | Similarity search and filtered retrieval |
| **API** | FastAPI + Uvicorn | REST API with OpenAPI docs |
| **Message Queue** | RabbitMQ | Distributed job processing |
| **Video Processing** | OpenCV + FFmpeg | Frame extraction, thumbnail generation |
| **Frontend** | Vanilla HTML/CSS/JS | Web search dashboard |
| **Observability** | Langfuse | Tracing and monitoring |
| **Containerization** | Docker + NVIDIA Container Toolkit | GPU-accelerated deployment |

### Quick Architecture Diagram

```
                    ┌──────────────────────────────────────────┐
                    │           Web Dashboard (Frontend)        │
                    │  Text | Image | Multimodal | OCR Search   │
                    └──────────────────┬───────────────────────┘
                                       │ REST API
                    ┌──────────────────▼───────────────────────┐
                    │            FastAPI Backend                │
                    │  /search/* | /ingest/* | /health | /stats │
                    └───┬──────────────┬──────────────┬────────┘
                        │              │              │
         ┌──────────────▼──┐  ┌───────▼────────┐  ┌──▼──────────────┐
         │  Qwen3-VL-2B    │  │    Qdrant      │  │  Static Files   │
         │  Embedding Model │  │  Vector Store  │  │  (Thumbnails,   │
         │  (+ Reranker)    │  │  (Cosine Sim)  │  │   Frames, Raw)  │
         └─────────────────┘  └────────────────┘  └─────────────────┘

                        ┌───────────────────────────┐
                        │  RabbitMQ Message Queue    │
                        │  (embedding.jobs)          │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │        Worker Process(es)          │
                    │  Download → Extract → Embed → Store│
                    └───────────────────────────────────┘
```

### Repository Structure

```
video-rag/
├── api/                           # FastAPI application
│   ├── main.py                    # App entry point, lifespan, middleware
│   ├── models/                    # Pydantic request/response schemas
│   │   ├── requests.py            # TextSearchRequest, IngestRequest, etc.
│   │   └── responses.py           # SearchResponse, IngestResponse, etc.
│   └── routes/                    # API endpoint handlers
│       ├── health.py              # GET /health, GET /stats
│       ├── ingest.py              # POST /ingest/*, DELETE /ingest/collection
│       └── search.py              # POST /search/text|image|multimodal|ocr
├── ingestion/                     # Data processing pipeline
│   ├── embedding_service.py       # EmbeddingService (Qwen3-VL wrapper)
│   ├── embedding_factory.py       # Factory: local vs remote embedding
│   ├── qwen3_vl_model.py         # Qwen3VLEmbedder low-level model
│   ├── qwen3_vl_reranker.py      # Qwen3VLReranker model
│   ├── remote_embedding_client.py # Client for remote model server
│   ├── video_processor.py         # Frame extraction from video/RTSP
│   ├── image_processor.py         # Image loading, thumbnails, metadata
│   └── ingest_pipeline.py         # Orchestrates full ingestion workflow
├── search/                        # Search functionality
│   ├── vector_store.py            # Qdrant client wrapper
│   ├── search_service.py          # Search orchestration layer
│   └── reranker_service.py        # Reranking with Qwen3-VL-Reranker
├── frontend/                      # Web dashboard
│   ├── index.html                 # Main HTML page
│   ├── styles.css                 # Stylesheet
│   └── app.js                     # Frontend JavaScript
├── scripts/                       # CLI tools and shell scripts
│   ├── ingest_data.py             # CLI for ingesting video/image data
│   ├── run_demo.py                # Demo showcasing 15 search scenarios
│   ├── clear_qdrant.py            # Clear the vector collection
│   ├── publish_embedding_job.py   # Publish jobs to RabbitMQ
│   ├── docker-start.sh            # Docker deployment launcher
│   ├── start-batched-server.sh    # Launch batched model server
│   ├── start-worker-cpu.sh        # Launch CPU worker
│   └── rabbitmq_reset.sh          # Reset RabbitMQ queue
├── docker/                        # Container configuration
│   ├── Dockerfile                 # GPU image (NVIDIA CUDA)
│   ├── Dockerfile.cpu             # CPU-only image
│   ├── docker-compose.dev.yml     # Development stack
│   ├── docker-compose.self-hosted.yml  # Production stack
│   ├── docker-compose.worker.yml  # Standalone worker
│   └── docker-manage.sh           # Docker management CLI
├── examples/                      # Example data and scripts
│   ├── multi_site_ingestion_example.py
│   ├── multi_site_urls.json
│   ├── sample_jobs.json
│   ├── urls_example.json
│   └── urls_example.txt
├── observability/                 # Monitoring and tracing
│   └── langfuse_integration.py    # Langfuse decorators and context managers
├── tests/                         # Test suite (placeholder)
├── logs/                          # Log output directory
├── config.py                      # Central configuration (env vars)
├── run.py                         # Server entry point
├── model_server.py                # Standalone embedding HTTP server
├── model_server_batched.py        # Batched embedding server (GPU optimized)
├── worker.py                      # RabbitMQ worker process
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── docs/                          # This documentation
```

---

## Getting Started

For quick setup instructions, see the main [README.md](../README.md) in the project root.

For detailed deployment options, see [Deployment](./deployment.md).

For development workflow, see [Development](./development.md).
