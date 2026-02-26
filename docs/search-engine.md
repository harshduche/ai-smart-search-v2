# Search Engine

This document covers the search subsystem internals, including vector similarity search, filtering, reranking, and the four search modalities.

---

## Overview

The search engine converts user queries (text, images, or both) into vector embeddings, performs cosine similarity search against Qdrant, and optionally reranks results using a second-stage model.

```
search/
├── search_service.py      # SearchService: orchestrates the full search flow
├── vector_store.py        # VectorStore: Qdrant client wrapper
└── reranker_service.py    # RerankerService: optional result reranking
```

---

## Search Service

**File:** `search/search_service.py` — `SearchService` class

The `SearchService` is the main orchestrator. It coordinates three singleton services:

1. **EmbeddingService** — converts queries to vectors
2. **VectorStore** — performs similarity search in Qdrant
3. **RerankerService** — optionally re-scores results

### Search Methods

#### `search_text(query, top_k, filters, use_reranker)`

1. Embed the text query → 2048-dim vector
2. Search Qdrant with cosine similarity
3. If `use_reranker=True`: fetch 3x candidates, rerank, return top_k
4. Return formatted results with scores and metadata

#### `search_image(image, top_k, filters, use_reranker)`

1. Embed the image → 2048-dim vector (same vector space as text)
2. Search Qdrant (cross-modal search: image query vs image/clip embeddings)
3. Optional reranking
4. Return results

#### `search_multimodal(text, image, top_k, filters, use_reranker)`

1. Generate a **combined** embedding from text + image simultaneously
2. The Qwen3-VL model processes both modalities together, producing a fused vector
3. Search Qdrant with this fused embedding
4. Optional reranking
5. Return results

#### `search_ocr(text_query, top_k, filters, use_reranker)`

1. Construct enhanced query: `"Text visible in image showing: {text_query}"`
2. Delegate to `search_text()` with the enhanced query
3. The VL model's training on OCR-capable data helps find frames with visible text

### Reranker Integration

When `use_reranker=True`:
- Initial retrieval fetches `min(top_k * 3, MAX_TOP_K)` candidates
- Reranker scores each (query, result) pair
- Results are re-sorted by reranker score
- Final top_k results are returned

This two-stage approach (retrieve then rerank) improves result quality at the cost of additional latency.

---

## Vector Store

**File:** `search/vector_store.py` — `VectorStore` class

Wrapper around `qdrant-client` that handles all interactions with the Qdrant vector database.

### Collection Setup

On first use, the VectorStore creates a collection with:

```python
# Vector configuration
size = 2048                    # Qwen3-VL embedding dimension
distance = Distance.COSINE     # Cosine similarity metric

# Payload indexes for efficient filtering
indexes = [
    ("source_file", KEYWORD),
    ("zone", KEYWORD),
    ("timestamp", DATETIME),
    ("frame_number", INTEGER),
    ("is_night", BOOL),
    ("organization_id", KEYWORD),
    ("site_id", KEYWORD),
    ("drone_id", KEYWORD),
    ("flight_id", KEYWORD),
]
```

### Insert Operations

#### `insert(embedding, metadata, point_id)`

Insert a single vector with metadata. Point ID is auto-generated from `{source_file}_{frame_number}` if not provided. Uses `hash(point_id) % 2^63` for Qdrant's integer ID requirement.

#### `insert_batch(embeddings, metadata_list, batch_size=100)`

Batch insert with configurable batch size. Points are accumulated and flushed to Qdrant in chunks of `batch_size`.

### Search Operations

#### `search(query_embedding, top_k, filters)`

Performs a filtered cosine similarity search.

**Filter Construction:**

Filters are translated into Qdrant's `Filter` DSL with `must` conditions:

```python
# Example: zone="perimeter" AND is_night=True AND timestamp >= "2024-01-01"
filter = Filter(must=[
    FieldCondition(key="zone", match=MatchValue(value="perimeter")),
    FieldCondition(key="is_night", match=MatchValue(value=True)),
    FieldCondition(key="timestamp", range=Range(gte="2024-01-01T00:00:00")),
])
```

**Supported Filter Types:**

| Filter Key | Qdrant Condition | Description |
|-----------|-----------------|-------------|
| `source_file` | MatchValue (keyword) | Exact match on filename |
| `zone` | MatchValue (keyword) | Exact match on zone |
| `is_night` | MatchValue (bool) | Boolean match |
| `organization_id` | MatchValue (keyword) | Exact match |
| `site_id` | MatchValue (keyword) | Exact match |
| `drone_id` | MatchValue (keyword) | Exact match |
| `flight_id` | MatchValue (keyword) | Exact match |
| `start_time` | Range (gte) | Timestamp >= value |
| `end_time` | Range (lte) | Timestamp <= value |

**Result Format:**

Each result includes:
- `score`: Cosine similarity (0 to 1)
- `id`: Original point ID from payload
- All payload fields (metadata)

### Administrative Operations

| Method | Description |
|--------|-------------|
| `get_collection_info()` | Returns collection name, vector count, point count, status |
| `delete_collection()` | Deletes the entire collection (destructive) |
| `count()` | Returns total point count |

---

## Reranker Service

**File:** `search/reranker_service.py` — `RerankerService` class

Optional second-stage reranking using the **Qwen3-VL-Reranker-2B** model.

### How It Works

1. Takes the initial search results (top_k * 3 candidates)
2. For each result, loads the thumbnail image
3. Evaluates (query, thumbnail) pairs through the reranker model
4. The reranker uses binary classification (yes/no relevance) to produce a relevance score
5. Re-sorts results by reranker score
6. Returns the top_k results

### Reranker Model

**File:** `ingestion/qwen3_vl_reranker.py` — `Qwen3VLReranker` class

- Model: `Qwen/Qwen3-VL-Reranker-2B`
- Input: (query text, document image) pair
- Output: Relevance score (higher = more relevant)
- Batch processing for efficiency (configurable `RERANKER_BATCH_SIZE`)

### Fallback Behavior

If the reranker model fails to load, the service falls back to **mock reranking** — results are returned with their original scores, unmodified. Check with `reranker_service.is_mock()`.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_RERANKER` | false | Enable/disable reranker globally |
| `RERANKER_MODEL_NAME` | Qwen/Qwen3-VL-Reranker-2B | Reranker model |
| `RERANKER_BATCH_SIZE` | 8 | Batch size for reranker inference |

---

## Cross-Modal Search

One of the most powerful features of this system is **cross-modal search**. Because text and images are embedded into the same 2048-dimensional vector space by Qwen3-VL, you can:

1. **Search with text** to find relevant frames (text → image matching)
2. **Search with an image** to find visually similar frames (image → image matching)
3. **Search with text + image** for hybrid queries (multimodal → image matching)
4. **Search for OCR text** leveraging the model's OCR-aware training

All four modalities produce vectors in the same space, enabling seamless cross-modal retrieval.

---

## Search Performance

### Latency Breakdown

| Stage | Typical Latency |
|-------|----------------|
| Text embedding | 20-50ms (GPU), 200-500ms (CPU) |
| Image embedding | 50-100ms (GPU), 500-2000ms (CPU) |
| Qdrant search (10k vectors) | 5-15ms |
| Qdrant search (100k vectors) | 15-50ms |
| Reranking (20 candidates) | 200-500ms (GPU) |
| Total (text, no rerank) | 25-65ms |
| Total (text, with rerank) | 250-600ms |

### Optimizing Search Speed

1. **Disable reranker** for faster queries (`use_reranker=false`)
2. **Use filters** to reduce candidate set before vector search
3. **Lower `top_k`** if you don't need many results
4. **Preload models** at startup (`PRELOAD_MODELS=true`) to avoid cold-start latency
5. **Use GPU** for embedding generation (10-20x faster than CPU)

### Scoring

- **Cosine Similarity**: Scores range from 0 (completely dissimilar) to 1 (identical)
- Scores > 0.7 typically indicate strong matches
- Scores 0.5-0.7 are moderate matches
- Scores < 0.5 are weak matches
- Reranker scores provide more calibrated relevance judgments

---

## Search Examples

### Security Use Cases (from Demo Scenarios)

| Query | Description |
|-------|-------------|
| "people near the perimeter fence" | Detect unauthorized persons near boundaries |
| "vehicles in parking zones or unauthorized areas" | Vehicle detection and zone violation |
| "ladders or climbing equipment" | Detect potential intrusion tools |
| "unattended bags or suspicious packages" | Unattended item detection |
| "person wearing safety vest" | PPE compliance monitoring |
| "nighttime footage from the main gate area" | Time-filtered search |
| "empty security post or guard station" | Unmanned post detection |
| "open gate or unlocked entrance" | Access control monitoring |
| "crowd or gathering of multiple people" | Crowd detection |
| "rain or adverse weather conditions" | Weather-based filtering |
| "security breach or intrusion at fence" | Intrusion detection |
| "damage to fence or broken infrastructure" | Infrastructure monitoring |
| "white sedan or light colored car" | Specific vehicle search |
| "emergency vehicle or ambulance" | Emergency vehicle detection |
| "unusual activity or anomalous behavior" | Anomaly detection |

### Filter Combinations

```python
# Site-specific night search
filters = {"site_id": "construction_site_a", "is_night": True}

# Cross-site drone search
filters = {"organization_id": "flytbase", "drone_id": "dji_mavic_001"}

# Time-range search with zone filter
filters = {
    "zone": "perimeter",
    "start_time": "2024-01-15T08:00:00",
    "end_time": "2024-01-15T18:00:00"
}

# Flight-specific search
filters = {"flight_id": "flight_20240207_001", "zone_type": "fence"}
```
