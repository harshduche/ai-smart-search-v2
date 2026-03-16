"""Configuration settings for the Video RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
FRAMES_DIR = Path(os.getenv("FRAMES_DIR", DATA_DIR / "frames"))
THUMBNAILS_DIR = Path(os.getenv("THUMBNAILS_DIR", DATA_DIR / "thumbnails"))

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, FRAMES_DIR, THUMBNAILS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = os.getenv("QDRANT_URL", "")        # Cloud URL (e.g. https://xxx.cloud.qdrant.io:6333)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") # Cloud API key
# Note: Collections are now per-organization (org_<id>), managed by VectorStore

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-Embedding-2B")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-VL-Reranker-2B")
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
EMBEDDING_DIM = 2048  # Qwen3-VL-Embedding-2B output dimension
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"  # Enable/disable reranker
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"  # Preload models at startup for low latency

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Frame Extraction
FRAME_RATE = float(os.getenv("FRAME_RATE", 1.0))  # Frames per second
THUMBNAIL_SIZE = int(os.getenv("THUMBNAIL_SIZE", 224))
SAVE_FULL_FRAMES = os.getenv("SAVE_FULL_FRAMES", "true").lower() == "true"  # Save full-resolution frames for popup view

# Long Video Optimization (for 3+ hour footage)
LONG_VIDEO_THRESHOLD_SECONDS = float(os.getenv("LONG_VIDEO_THRESHOLD_SECONDS", 3600))  # 1 hour
LONG_VIDEO_FRAME_RATE = float(os.getenv("LONG_VIDEO_FRAME_RATE", 0.5))  # Reduced frame rate for long videos
SEMANTIC_CLIP_DURATION = float(os.getenv("SEMANTIC_CLIP_DURATION", 4.0))  # Duration of each semantic clip in seconds
SEMANTIC_CLIP_MAX_FRAMES = int(os.getenv("SEMANTIC_CLIP_MAX_FRAMES", 32))  # Max frames per clip
CLIP_EMBED_BATCH_SIZE = int(os.getenv("CLIP_EMBED_BATCH_SIZE", 2))  # Clips to accumulate before embedding (keep low to bound memory)
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", 8))  # Batch size for reranker

# Search Configuration
DEFAULT_TOP_K = 20
MAX_TOP_K = 100

# AWS / S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")  # Optional: custom endpoint (e.g. MinIO)
S3_PRESIGNED_URL_EXPIRATION = int(os.getenv("S3_PRESIGNED_URL_EXPIRATION", 3600))  # seconds
# 's3' = SigV2 (AWSAccessKeyId/Signature/Expires params, path-style URL)
# 's3v4' = SigV4 (X-Amz-* params, virtual-hosted URL) — default for newer AWS SDK
S3_SIGNATURE_VERSION = os.getenv("S3_SIGNATURE_VERSION", "s3v4")
# 'path' = https://s3.region.amazonaws.com/bucket/key
# 'virtual' = https://bucket.s3.region.amazonaws.com/key
S3_ADDRESSING_STYLE = os.getenv("S3_ADDRESSING_STYLE", "path")
S3_USE_ACCELERATE = os.getenv("S3_USE_ACCELERATE", "true").lower() == "true"

# MongoDB (for geospatial queries)
MONGODB_URI      = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "video_rag")
