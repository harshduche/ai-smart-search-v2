# API Reference

Complete REST API documentation for the Visual Search Engine. The API is built with FastAPI and auto-generates interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc).

**Base URL:** `http://localhost:8000`

---

## Search Endpoints

All search endpoints return a `SearchResponse` object containing results with similarity scores, metadata, and static file URLs.

### POST `/search/text`

Search footage using a natural language text query.

**Request Body (JSON):**

```json
{
  "query": "person walking near the perimeter fence at night",
  "top_k": 10,
  "use_reranker": false,
  "filters": {
    "zone": "perimeter",
    "is_night": true,
    "start_time": "2024-01-01T00:00:00",
    "end_time": "2024-01-31T23:59:59",
    "source_file": "camera1.mp4",
    "organization_id": "flytbase",
    "site_id": "construction_site_a",
    "drone_id": "dji_mavic_001",
    "flight_id": "flight_001",
    "zone_type": "fence",
    "flight_purpose": "perimeter_inspection"
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | — | Natural language search query |
| `top_k` | int | No | 20 | Number of results (1-100) |
| `use_reranker` | bool | No | false | Enable Qwen3-VL reranking |
| `filters` | object | No | null | Filter criteria (see Filters section) |

**Response:**

```json
{
  "query_type": "text",
  "query": "person walking near the perimeter fence at night",
  "total_results": 10,
  "search_time_ms": 45.2,
  "results": [
    {
      "id": "video1.mp4_120",
      "score": 0.856,
      "source_file": "video1.mp4",
      "frame_number": 120,
      "timestamp": "2024-01-15T14:30:45",
      "seconds_offset": 120.0,
      "original_frame_number": 3600,
      "sample_rate": 30,
      "source_type": "video",
      "thumbnail_path": "/thumbnails/video1/frame_000120.jpg",
      "frame_path": "/frames/video1/frame_000120.jpg",
      "video_path": "/raw/video1.mp4",
      "zone": "perimeter",
      "is_night": true,
      "clip_index": 5,
      "clip_start_timestamp": "2024-01-15T14:30:40",
      "clip_end_timestamp": "2024-01-15T14:30:44",
      "clip_start_seconds": 116.0,
      "clip_end_seconds": 120.0,
      "num_frames": 4,
      "organization_id": "flytbase",
      "site_id": "construction_site_a",
      "drone_id": "dji_mavic_001",
      "flight_id": "flight_001"
    }
  ]
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "person near fence", "top_k": 10}'
```

---

### POST `/search/image`

Search footage using a reference image (finds visually similar scenes).

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | file | Yes | — | Reference image file |
| `top_k` | int | No | 20 | Number of results (1-100) |
| `zone` | string | No | null | Zone filter |
| `is_night` | bool | No | null | Night filter |
| `use_reranker` | bool | No | false | Enable reranking |

**cURL Example:**

```bash
curl -X POST http://localhost:8000/search/image \
  -F "image=@reference.jpg" \
  -F "top_k=10"
```

**Response:** Same `SearchResponse` format as text search.

---

### POST `/search/multimodal`

Search using combined text and image query for precise hybrid matching.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | file | Yes | — | Reference image file |
| `query` | string | Yes | — | Text query to combine with image |
| `top_k` | int | No | 20 | Number of results (1-100) |
| `zone` | string | No | null | Zone filter |
| `is_night` | bool | No | null | Night filter |
| `start_time` | string | No | null | Start time filter (ISO 8601) |
| `end_time` | string | No | null | End time filter (ISO 8601) |
| `use_reranker` | bool | No | false | Enable reranking |

**cURL Example:**

```bash
curl -X POST http://localhost:8000/search/multimodal \
  -F "image=@reference.jpg" \
  -F "query=similar scene but at night" \
  -F "top_k=10"
```

---

### POST `/search/ocr`

Search for text visible in footage (license plates, signs, labels).

**Request Body (JSON):**

```json
{
  "text": "7829",
  "top_k": 10,
  "use_reranker": true,
  "filters": {
    "zone": "parking"
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | — | Text to search for in footage |
| `top_k` | int | No | 20 | Number of results (1-100) |
| `use_reranker` | bool | No | false | Enable reranking |
| `filters` | object | No | null | Filter criteria |

**cURL Example:**

```bash
curl -X POST http://localhost:8000/search/ocr \
  -H "Content-Type: application/json" \
  -d '{"text": "7829", "top_k": 10}'
```

**Implementation Note:** OCR search constructs an enhanced query `"Text visible in image showing: {text}"` and delegates to the text search pipeline.

---

## Ingestion Endpoints

### POST `/ingest/`

Ingest video/image data from a filesystem path.

**Request Body (JSON):**

```json
{
  "path": "/path/to/video.mp4",
  "zone": "main_gate",
  "process_videos": true,
  "process_images": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | Yes | — | Filesystem path to file or directory |
| `zone` | string | No | null | Zone/location identifier |
| `process_videos` | bool | No | true | Process video files |
| `process_images` | bool | No | true | Process image files |

**Response:**

```json
{
  "status": "completed",
  "videos_processed": 1,
  "images_processed": 0,
  "frames_extracted": 1800,
  "embeddings_generated": 1800,
  "vectors_stored": 1800,
  "duration_seconds": 245.3,
  "errors": []
}
```

---

### POST `/ingest/upload`

Upload a video file via the web dashboard and ingest it.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | Yes | — | Video file to upload |
| `zone` | string | No | null | Zone identifier |
| `semantic_video` | bool | No | false | Use semantic clip mode |
| `clip_duration` | float | No | 4.0 | Clip duration (seconds) |
| `max_frames_per_clip` | int | No | 32 | Max frames per clip |
| `save_full_frames` | bool | No | null | Save full-res frames |

**cURL Example:**

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@security_footage.mp4" \
  -F "zone=main_gate" \
  -F "semantic_video=true"
```

---

### POST `/ingest/url`

Download and ingest a video from a remote URL (S3, HTTP, HTTPS).

**Request Body (JSON):**

```json
{
  "video_url": "https://s3.amazonaws.com/bucket/video.mp4?signature=...",
  "zone": "camera_1",
  "clip_duration": 4.0,
  "max_frames_per_clip": 32,
  "batch_size": 50,
  "save_full_frames": false,
  "cleanup_after": true,
  "organization_id": "flytbase",
  "site_id": "construction_site_a",
  "drone_id": "dji_mavic_001",
  "flight_id": "flight_001"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_url` | string | Yes | — | URL to download video from |
| `zone` | string | No | null | Zone identifier |
| `clip_duration` | float | No | 4.0 | Semantic clip duration |
| `max_frames_per_clip` | int | No | 32 | Max frames per clip |
| `batch_size` | int | No | 50 | Batch size for storage |
| `save_full_frames` | bool | No | null | Save full-res frames |
| `cleanup_after` | bool | No | true | Delete downloaded file after processing |
| `organization_id` | string | No | null | Organization identifier |
| `site_id` | string | No | null | Site identifier |
| `drone_id` | string | No | null | Drone identifier |
| `flight_id` | string | No | null | Flight identifier |

---

### POST `/ingest/drone-footage`

Ingest drone footage with comprehensive multi-site metadata. Designed for organizations managing footage across multiple sites.

**Request Body (JSON):**

```json
{
  "video_url": "https://s3.amazonaws.com/bucket/site_a_morning.mp4",
  "organization_id": "flytbase_security",
  "organization_name": "FlytBase Security",
  "site_id": "construction_site_a",
  "site_name": "Construction Site A - Mumbai",
  "site_location": {
    "lat": 19.0760,
    "lon": 72.8777,
    "city": "Mumbai",
    "state": "Maharashtra",
    "country": "India"
  },
  "drone_id": "dji_mavic_001",
  "drone_model": "DJI Mavic 3 Enterprise",
  "drone_serial": "ABC123XYZ001",
  "flight_id": "flight_20240207_001",
  "flight_date": "2024-02-07",
  "flight_time": "morning",
  "flight_purpose": "perimeter_inspection",
  "zone": "north_perimeter",
  "zone_type": "fence",
  "weather_condition": "clear",
  "operator": "john_doe",
  "tags": ["perimeter", "security", "routine"],
  "notes": "Morning patrol route",
  "clip_duration": 4.0,
  "max_frames_per_clip": 32,
  "cleanup_after": true
}
```

See `api/models/requests.py` for the full `DroneFootageIngestRequest` schema.

---

### POST `/ingest/rtsp`

Ingest live video from an RTSP stream (IP cameras, surveillance systems).

**Request Body (JSON):**

```json
{
  "rtsp_url": "rtsp://192.168.1.100:554/stream1",
  "zone": "main_entrance",
  "duration_seconds": 300,
  "max_frames": null,
  "use_semantic_clips": true,
  "clip_duration": 4.0,
  "max_frames_per_clip": 32,
  "reconnect_on_failure": true,
  "max_reconnect_attempts": 5,
  "reconnect_delay_seconds": 2.0,
  "connection_timeout_seconds": 10.0,
  "camera_id": "cam_001",
  "site_id": "office_building_a"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `rtsp_url` | string | Yes | — | RTSP stream URL |
| `zone` | string | No | null | Zone identifier |
| `duration_seconds` | float | No* | null | Capture duration (1-3600s) |
| `max_frames` | int | No* | null | Max frames to capture (1-10000) |
| `use_semantic_clips` | bool | No | true | Use semantic clip grouping |
| `clip_duration` | float | No | 4.0 | Clip duration (2-20s) |
| `reconnect_on_failure` | bool | No | true | Auto-reconnect on drops |
| `max_reconnect_attempts` | int | No | 5 | Max retry attempts |

*At least one of `duration_seconds` or `max_frames` is required.

---

### DELETE `/ingest/collection`

**WARNING: Destructive operation.** Deletes the entire Qdrant vector collection and all stored embeddings.

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/ingest/collection
```

**Response:**

```json
{
  "status": "deleted",
  "message": "Collection deleted successfully"
}
```

---

## Health & Stats Endpoints

### GET `/health`

Returns service health status, model info, and vector store status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
  "using_mock_embeddings": false,
  "vector_store": {
    "name": "security_footage",
    "indexed_vectors_count": 15420,
    "points_count": 15420,
    "status": "green"
  },
  "search_service": {
    "collection": { ... },
    "embedding_dim": 2048,
    "using_mock_embeddings": false,
    "using_mock_reranker": true
  }
}
```

### GET `/stats`

Returns collection statistics.

**Response:**

```json
{
  "name": "security_footage",
  "vectors_count": 15420,
  "points_count": 15420,
  "status": "green"
}
```

### GET `/`

Returns API info and documentation links.

**Response:**

```json
{
  "name": "Visual Search Engine API",
  "version": "1.0.0",
  "docs": "/docs",
  "dashboard": "/static/index.html"
}
```

---

## Search Filters

All search endpoints support a common set of filters:

### Basic Filters

| Filter | Type | Description |
|--------|------|-------------|
| `zone` | string | Filter by zone/location (e.g., "perimeter", "main_gate") |
| `is_night` | bool | Filter by day/night (based on brightness analysis) |
| `source_file` | string | Filter by source video/image filename |
| `start_time` | datetime | Filter results after this timestamp |
| `end_time` | datetime | Filter results before this timestamp |

### Multi-Site Filters

| Filter | Type | Description |
|--------|------|-------------|
| `organization_id` | string | Filter by organization |
| `site_id` | string | Filter by site |
| `drone_id` | string | Filter by drone |
| `flight_id` | string | Filter by flight |
| `zone_type` | string | Filter by zone type (fence, building, parking) |
| `flight_purpose` | string | Filter by flight purpose |

---

## Response Models

### SearchResult

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique result identifier |
| `score` | float | Similarity score (0-1, higher = better) |
| `source_file` | string | Source video/image filename |
| `frame_number` | int | Frame number within source |
| `timestamp` | string | Frame timestamp (ISO 8601) |
| `seconds_offset` | float | Seconds from video start |
| `source_type` | string | "video", "image", "video_clip", etc. |
| `thumbnail_path` | string | URL path to 224px thumbnail |
| `frame_path` | string | URL path to full-resolution frame |
| `video_path` | string | URL path to source video |
| `zone` | string | Zone/location identifier |
| `is_night` | bool | Whether frame is nighttime |
| `clip_index` | int | Semantic clip index |
| `clip_start_seconds` | float | Clip start time (seconds) |
| `clip_end_seconds` | float | Clip end time (seconds) |
| `num_frames` | int | Frames in semantic clip |
| `organization_id` | string | Organization identifier |
| `site_id` | string | Site identifier |
| `drone_id` | string | Drone identifier |
| `flight_id` | string | Flight identifier |

### IngestResponse

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "completed" or error status |
| `videos_processed` | int | Videos processed count |
| `images_processed` | int | Images processed count |
| `frames_extracted` | int | Frames extracted count |
| `embeddings_generated` | int | Embeddings generated count |
| `vectors_stored` | int | Vectors stored in Qdrant |
| `duration_seconds` | float | Total processing time |
| `errors` | list[string] | Any errors encountered |
| `video_url` | string | Source URL (if URL-based) |
| `site_id` | string | Site identifier (if multi-site) |
| `file_size_mb` | float | Video file size in MB |

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Meaning | Common Cause |
|------|---------|-------------|
| `400` | Bad Request | Invalid URL, missing required fields, connection failure |
| `404` | Not Found | Filesystem path does not exist |
| `500` | Internal Server Error | Model failure, Qdrant unavailable, processing error |

Error responses include a `detail` field with a human-readable message:

```json
{
  "detail": "Path not found: /nonexistent/path"
}
```

---

## Interactive Documentation

When the server is running, full interactive API docs are available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
