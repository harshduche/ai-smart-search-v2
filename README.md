# Visual Search Engine for Security Operations

AI-powered visual search system using Qwen3-VL-Embedding that enables security teams to search across video/image data using natural language queries, reference images, or multi-modal searches.

## Features

- **Text Search**: Natural language queries like "person near fence at night"
- **Image Search**: Upload a reference image to find similar scenes
- **Multimodal Search**: Combine text + image for precise queries
- **OCR Search**: Find text visible in footage (license plates, signs)
- **URL Ingestion**: Download and ingest videos directly from S3/HTTP URLs
- **Temporal Filtering**: Search within time ranges
- **Spatial Filtering**: Filter by zones/locations
- **Sub-second Query Response**: Optimized for <2s search times

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Web Dashboard                              │
│  Text Search | Image Search | Multimodal | OCR               │
└─────────────────────────────┬────────────────────────────────┘
                              │ REST API
┌─────────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                            │
│  /search/text | /search/image | /search/multimodal | /search/ocr│
└─────────────────────────────┬────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Qwen3-VL-2B   │   │   Qdrant      │   │ Static Files  │
│ (Embeddings)  │   │ (Vector DB)   │   │ (Thumbnails)  │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Quick Start

### 1. Clone and Setup

```bash
# SSH into your machine
ssh deair@100.87.139.27

# Navigate to project
cd /path/to/video-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant (Vector Database)

```bash
# Using Docker
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/collections
```

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit as needed
nano .env
```

Key settings:
- `DEVICE`: Set to `cuda` for GPU, `cpu` otherwise
- `MODEL_NAME`: `Qwen/Qwen3-VL-Embedding-2B`
- `BATCH_SIZE`: Adjust based on GPU VRAM (4-8 recommended for <16GB)

### 4. Ingest Video/Image Data

```bash
# Ingest a single video
python scripts/ingest_data.py /path/to/video.mp4 --zone main_gate

# Ingest a directory
python scripts/ingest_data.py /path/to/footage/ --stats-output ingestion_stats.json

# Ingest only videos
python scripts/ingest_data.py /path/to/footage/ --videos-only

# Ingest from URL (S3, HTTP, HTTPS)
python scripts/ingest_data.py "https://s3.amazonaws.com/bucket/video.mp4?signature=..." --from-url --zone camera1

# Batch ingest from multiple URLs
python scripts/ingest_from_urls_batch.py examples/urls_example.json
```

### 5. Start the Server

```bash
python run.py
```

Server will be available at:
- Dashboard: http://localhost:8000/static/index.html
- API Docs: http://localhost:8000/docs

### 6. Access from Local Machine

```bash
# SSH tunnel for local access
ssh -L 8000:localhost:8000 deair@100.87.139.27

# Open in browser
http://localhost:8000/static/index.html
```

## API Usage

### Text Search

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "person near fence", "top_k": 10}'
```

### Image Search

```bash
curl -X POST http://localhost:8000/search/image \
  -F "image=@reference.jpg" \
  -F "top_k=10"
```

### Multimodal Search

```bash
curl -X POST http://localhost:8000/search/multimodal \
  -F "image=@reference.jpg" \
  -F "query=similar scene but at night" \
  -F "top_k=10"
```

### OCR Search

```bash
curl -X POST http://localhost:8000/search/ocr \
  -H "Content-Type: application/json" \
  -d '{"text": "7829", "top_k": 10}'
```

## Demo Scenarios

Run the demo to showcase all 15 required search scenarios:

```bash
python scripts/run_demo.py
```

This tests:
1. People near perimeter fence
2. Vehicles in unauthorized zones
3. Ladders/climbing equipment
4. Unattended bags
5. Safety vests
6. Nighttime gate footage
7. Empty guard posts
8. Open gates
9. Crowd gatherings
10. Adverse weather
11. Security breaches
12. Infrastructure damage
13. Specific vehicles
14. Emergency vehicles
15. Anomalous activity

## Project Structure

```
video-rag/
├── api/                    # FastAPI application
│   ├── main.py            # App entry point
│   ├── routes/            # API endpoints
│   └── models/            # Request/response schemas
├── ingestion/             # Data processing
│   ├── embedding_service.py   # Qwen3-VL wrapper
│   ├── video_processor.py     # Frame extraction
│   ├── image_processor.py     # Image loading
│   └── ingest_pipeline.py     # Main pipeline
├── search/                # Search functionality
│   ├── vector_store.py    # Qdrant client
│   └── search_service.py  # Search logic
├── frontend/              # Web dashboard
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── scripts/               # CLI tools
│   ├── ingest_data.py
│   └── run_demo.py
├── docker-compose.yml     # Qdrant setup
├── requirements.txt
├── config.py
└── run.py                 # Server entry point
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `MODEL_NAME` | Qwen/Qwen3-VL-Embedding-2B | Embedding model |
| `DEVICE` | cuda | Device (cuda/cpu/mps) |
| `BATCH_SIZE` | 4 | Batch size for embedding |
| `FRAME_RATE` | 1.0 | Frames per second to extract |
| `API_PORT` | 8000 | API server port |

## Performance Notes

For <16GB VRAM:
- Use `Qwen3-VL-Embedding-2B` model
- Set `BATCH_SIZE=4`
- FP16 is enabled automatically on CUDA

For 3+ hours of footage:
- ~10,800 frames at 1 fps
- ~10-20 GB VRAM during ingestion
- Ingestion time: varies with hardware

## Troubleshooting

**Qdrant connection error:**
```bash
docker-compose restart
```

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in .env
- Use `--batch-size 2` flag in ingest script

**Model not loading:**
- Check internet connection (first run downloads model)
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## License

Built for FlytBase AI Hackathon 2026
