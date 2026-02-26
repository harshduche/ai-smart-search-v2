"""Ingestion module for video/image processing and embedding generation."""

from .video_processor import VideoProcessor
from .image_processor import ImageProcessor
from .embedding_service import EmbeddingService
from .ingest_pipeline import IngestPipeline

__all__ = ["VideoProcessor", "ImageProcessor", "EmbeddingService", "IngestPipeline"]
