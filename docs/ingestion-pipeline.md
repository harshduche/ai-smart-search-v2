# Ingestion Pipeline

This document covers the complete data ingestion system, from raw video/image input to stored vector embeddings in Qdrant.

---

## Overview

The ingestion pipeline handles:
1. **Frame extraction** from videos (or direct image loading)
2. **Thumbnail generation** for UI display
3. **Metadata extraction** (brightness, timestamps, night detection)
4. **Embedding generation** using Qwen3-VL-Embedding-2B
5. **Vector storage** in Qdrant with rich payload metadata

---

## Pipeline Architecture

```
ingestion/
├── ingest_pipeline.py         # Main orchestrator (IngestPipeline class)
├── video_processor.py         # Frame extraction (VideoProcessor class)
├── image_processor.py         # Image loading/processing (ImageProcessor class)
├── embedding_service.py       # High-level embedding API (EmbeddingService class)
├── embedding_factory.py       # Factory: local vs remote embedding
├── qwen3_vl_model.py         # Low-level Qwen3VL model wrapper
├── qwen3_vl_reranker.py      # Reranker model wrapper
└── remote_embedding_client.py # HTTP client for remote model server
```

---

## Ingestion Modes

### 1. Frame-by-Frame Ingestion

Each extracted frame gets its own individual embedding. Best for short videos where per-frame granularity matters.

```python
pipeline = create_ingest_pipeline()
count = pipeline.ingest_video("/path/to/video.mp4", zone="main_gate")
```

**Process:**
1. Extract frames at configured FPS (default: 1.0 fps)
2. Generate thumbnail (224x224) for each frame
3. Optionally save full-resolution frame
4. Compute brightness, determine day/night
5. Generate one 2048-dim embedding per frame
6. Store embedding + metadata in Qdrant

### 2. Semantic Clip Ingestion

Groups consecutive frames into clips (default: 4 seconds each) and generates a single video embedding per clip using Qwen3-VL's native video understanding. Best for long videos (3+ hours).

```python
pipeline = create_ingest_pipeline()
count = pipeline.ingest_video_semantic_clips(
    "/path/to/video.mp4",
    zone="perimeter",
    clip_duration=4.0,          # 4-second clips
    max_frames_per_clip=32,     # Memory limit per clip
)
```

**Process:**
1. Extract frames at configured FPS
2. Group frames into clips of `clip_duration` seconds
3. Cap each clip at `max_frames_per_clip` frames
4. Generate one 2048-dim **video** embedding per clip
5. Store embedding with clip metadata (start/end times, frame count)

**Advantages of Semantic Clips:**
- ~4x fewer embeddings to store
- Temporal context (model sees motion/changes across frames)
- Better for long surveillance footage
- Lower storage and query costs

### 3. Image Ingestion

Process standalone images (not extracted from video).

```python
pipeline = create_ingest_pipeline()
count = pipeline.ingest_images([Path("img1.jpg"), Path("img2.png")], zone="entrance")
```

### 4. URL-Based Ingestion

Download a video from a remote URL (S3, HTTP, HTTPS), ingest it, and optionally clean up.

```python
pipeline = create_ingest_pipeline()
result = pipeline.ingest_video_from_url(
    video_url="https://s3.amazonaws.com/bucket/video.mp4?signature=...",
    zone="camera_1",
    cleanup_after=True,      # Delete downloaded file after processing
    metadata={"site_id": "construction_site_a"},
)
```

### 5. RTSP Stream Ingestion

Capture frames from a live RTSP camera feed for a specified duration or frame count.

```python
pipeline = create_ingest_pipeline()
result = pipeline.ingest_rtsp_stream(
    rtsp_url="rtsp://192.168.1.100:554/stream1",
    zone="main_entrance",
    duration_seconds=300,        # 5 minutes
    use_semantic_clips=True,
    reconnect_on_failure=True,
)
```

### 6. Pre-Extracted Frame Ingestion

Process a directory of already-extracted frame images (e.g., from external tools).

```python
pipeline = create_ingest_pipeline()
count = pipeline.ingest_pre_extracted_frames(
    frames_dir=Path("/path/to/frames/"),
    zone="warehouse",
    source_name="external_camera",
)
```

---

## Video Processor

**File:** `ingestion/video_processor.py` — `VideoProcessor` class

### Frame Extraction

Uses OpenCV (`cv2.VideoCapture`) to extract frames from video files or RTSP streams.

**Key Parameters:**
- `frame_rate`: Frames per second to extract (default: 1.0 fps)
- `thumbnail_size`: Thumbnail dimension (default: 224px)
- `save_full_frames`: Whether to save full-resolution frames

**Long Video Optimization:**
Videos longer than `LONG_VIDEO_THRESHOLD_SECONDS` (default: 3600s / 1 hour) automatically use `LONG_VIDEO_FRAME_RATE` (default: 0.5 fps) to reduce the number of frames.

**Frame Extraction Process:**
```
Video File (e.g., 30fps, 1 hour)
        │
        ▼
cv2.VideoCapture → Read frames
        │
        ▼  Sample at configured FPS (e.g., 1 frame/sec)
        │
        ▼
For each sampled frame:
  1. Save as 224x224 thumbnail → data/thumbnails/{video_name}/frame_{N}.jpg
  2. (Optional) Save full frame → data/frames/{video_name}/frame_{N}.jpg
  3. Compute brightness (mean pixel value)
  4. Determine is_night (brightness < threshold)
  5. Calculate timestamp and seconds_offset
```

### Night Detection

Brightness-based heuristic:
1. Convert frame to grayscale
2. Compute mean pixel value (0-255)
3. If mean < threshold → classify as nighttime

### RTSP Stream Handling

- Configurable connection timeout
- Automatic reconnection on stream drops
- Retry with configurable delay and max attempts
- Frame capture continues until duration or frame count limit

---

## Image Processor

**File:** `ingestion/image_processor.py` — `ImageProcessor` class

Handles:
- Loading images from paths (JPEG, PNG, BMP, WebP)
- Creating 224x224 thumbnails with aspect ratio preservation
- Extracting metadata (file size, dimensions, brightness)
- Batch processing of image directories

---

## Embedding Service

**File:** `ingestion/embedding_service.py` — `EmbeddingService` class

The central service for generating vector embeddings. Wraps the Qwen3-VL-Embedding model.

### Embedding Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| `embed_text(text)` | String | 2048-dim vector | Text queries |
| `embed_image(image)` | PIL Image or path | 2048-dim vector | Image search, frame indexing |
| `embed_multimodal(text, image)` | String + PIL Image | 2048-dim vector | Combined queries |
| `embed_video_clip(frames)` | List[PIL Image] | 2048-dim vector | Semantic clip indexing |
| `embed_images_batch(images)` | List of images | List of vectors | Batch frame processing |
| `embed_video_clips_batch(clips)` | List of frame lists | List of vectors | Batch clip processing |

### Mock Embeddings

If the Qwen3-VL model fails to load (e.g., no GPU, missing dependencies), the service automatically falls back to **mock embeddings** — random normalized vectors. This allows development and testing without a GPU.

Check with: `embedding_service.is_mock()`.

### Memory Management

- CUDA cache is cleared every 50 clips (for `embed_video_clips_batch`)
- CUDA cache is cleared every 100 images (for `embed_images_batch`)
- FP16 inference is automatically enabled on CUDA devices

---

## Embedding Factory

**File:** `ingestion/embedding_factory.py`

Decides whether to use a **local** embedding service (loads model on same machine) or a **remote** embedding client (connects to a model server via HTTP).

```python
# Environment variable driven:
# EMBEDDING_MODE=local   → EmbeddingService (local GPU)
# EMBEDDING_MODE=remote  → RemoteEmbeddingClient (HTTP to model server)
# (default)              → EmbeddingService (local)

from ingestion.embedding_factory import get_embedding_service
service = get_embedding_service()
```

---

## Remote Embedding Client

**File:** `ingestion/remote_embedding_client.py`

HTTP client that connects to a model server (`model_server.py` or `model_server_batched.py`) for embedding generation. Used by CPU-only workers that don't have GPU access.

**Features:**
- Sends images as base64-encoded data
- Supports text, image, multimodal, and video clip embeddings
- Health check against model server
- Same interface as `EmbeddingService` (drop-in replacement)

---

## Qwen3-VL Model

**File:** `ingestion/qwen3_vl_model.py` — `Qwen3VLEmbedder` class

Low-level wrapper around the Hugging Face `Qwen3-VL-Embedding-2B` model.

**Key Details:**
- Model architecture: Vision-Language Transformer
- Output dimension: 2048
- Supports text-only, image-only, text+image, and video (list of frames) inputs
- Uses the official `qwen-vl-utils` for input processing
- FP16 inference on CUDA for memory efficiency

**Input Format (to `process()`):**

```python
# Text only
inputs = [{'text': 'person near fence'}]

# Image only
inputs = [{'image': pil_image}]

# Text + Image (multimodal)
inputs = [{'text': 'describe this', 'image': pil_image}]

# Video (list of frames)
inputs = [{'video': [frame1, frame2, frame3, ...]}]
```

---

## Metadata Stored Per Embedding

Each vector in Qdrant includes the following payload:

### Core Fields
| Field | Type | Description |
|-------|------|-------------|
| `source_file` | string | Source video/image filename |
| `frame_number` | int | Frame index within extraction |
| `timestamp` | string | ISO 8601 timestamp |
| `seconds_offset` | float | Seconds from video start |
| `original_frame_number` | int | Actual frame number in video |
| `sample_rate` | int | Video FPS |
| `source_type` | string | "video", "image", "video_clip", "rtsp_stream" |
| `thumbnail_path` | string | Path to 224px thumbnail |
| `frame_path` | string | Path to full-resolution frame (if saved) |
| `video_path` | string | Path to source video |
| `zone` | string | Zone/location identifier |
| `brightness` | float | Average pixel brightness (0-255) |
| `is_night` | bool | Whether frame is nighttime |

### Semantic Clip Fields
| Field | Type | Description |
|-------|------|-------------|
| `clip_index` | int | Clip index within video |
| `clip_start_timestamp` | string | Clip start time |
| `clip_end_timestamp` | string | Clip end time |
| `clip_start_seconds` | float | Clip start (seconds) |
| `clip_end_seconds` | float | Clip end (seconds) |
| `num_frames` | int | Number of frames in clip |

### Multi-Site Fields
| Field | Type | Description |
|-------|------|-------------|
| `organization_id` | string | Organization identifier |
| `organization_name` | string | Organization display name |
| `site_id` | string | Site identifier |
| `site_name` | string | Site display name |
| `site_location` | object | `{lat, lon, city, state, country}` |
| `drone_id` | string | Drone identifier |
| `drone_model` | string | Drone model name |
| `flight_id` | string | Flight identifier |
| `flight_purpose` | string | Purpose of the flight |
| `zone_type` | string | Zone type (fence, building, etc.) |
| `weather_condition` | string | Weather during capture |
| `operator` | string | Operator/pilot name |
| `tags` | list[string] | Additional categorization tags |

---

## CLI Ingestion Scripts

### `scripts/ingest_data.py`

Full-featured CLI for data ingestion:

```bash
# Single video
python scripts/ingest_data.py /path/to/video.mp4 --zone main_gate

# Directory of files
python scripts/ingest_data.py /path/to/footage/ --stats-output stats.json

# Videos only (skip images)
python scripts/ingest_data.py /path/to/footage/ --videos-only

# URL-based ingestion
python scripts/ingest_data.py "https://s3.amazonaws.com/video.mp4" --from-url --zone camera1

# Semantic clips
python scripts/ingest_data.py /path/to/video.mp4 --semantic-clips --clip-duration 4.0

# Pre-extracted frames
python scripts/ingest_data.py /path/to/frames/ --pre-extracted
```

### `scripts/publish_embedding_job.py`

Publish ingestion jobs to RabbitMQ for async worker processing:

```bash
# Single job
python scripts/publish_embedding_job.py \
  --video-url "https://example.com/video.mp4" \
  --metadata '{"zone": "perimeter", "site_id": "site_a"}'

# Batch from file
python scripts/publish_embedding_job.py --batch-file examples/sample_jobs.json
```

---

## Performance Considerations

### Frame Rate Tuning

| Video Length | Recommended FPS | Frames (1 hour) |
|-------------|----------------|------------------|
| < 1 hour | 1.0 fps | 3,600 |
| 1-3 hours | 0.5 fps | 1,800-5,400 |
| 3+ hours | 0.25 fps | 2,700+ |

### Memory Usage

- Each Qwen3-VL-2B inference uses ~2-4 GB VRAM
- Batch size 4 with FP16: ~6-8 GB peak VRAM
- Batch size 8 with FP16: ~10-14 GB peak VRAM
- Semantic clips with 32 frames: ~4-6 GB per clip

### Storage Requirements

| Component | Size Per Frame |
|-----------|---------------|
| Thumbnail (224x224 JPEG) | ~10-20 KB |
| Full frame (1080p JPEG) | ~100-500 KB |
| Embedding vector (2048 float32) | 8 KB |
| Qdrant metadata payload | ~1-2 KB |
