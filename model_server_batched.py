"""
Model Server with Request Batching for Parallel Processing.

This version collects multiple requests within a time window and processes
them together in a batch, enabling parallel GPU utilization while sharing
a single model instance.
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import deque
from datetime import datetime
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import base64

sys.path.insert(0, str(Path(__file__).parent))

import config
from ingestion.embedding_service import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Request/Response Models (same as before)
class TextEmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed")


class ImageEmbedRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")


class VideoClipEmbedRequest(BaseModel):
    images_base64: List[str] = Field(..., description="List of base64 encoded images (frames)")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")


class MultimodalEmbedRequest(BaseModel):
    text: str = Field(..., description="Text component")
    image_base64: str = Field(..., description="Base64 encoded image")


class BatchTextEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")


class BatchImageEmbedRequest(BaseModel):
    images_base64: List[str] = Field(..., description="List of base64 encoded images")


class BatchVideoClipEmbedRequest(BaseModel):
    clips: List[List[str]] = Field(..., description="List of clips, each a list of base64 images")


class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")
    request_id: Optional[str] = Field(None, description="Request ID if provided")


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    batch_processing: bool
    queue_size: int


class BatchProcessor:
    """Handles request batching and parallel processing."""

    def __init__(self, embedding_service: EmbeddingService,
                 batch_window_ms: int = 50,
                 max_batch_size: int = 16):
        """
        Initialize batch processor.

        Args:
            embedding_service: The embedding service instance
            batch_window_ms: Time window to collect requests (milliseconds)
            max_batch_size: Maximum batch size for GPU processing
        """
        self.embedding_service = embedding_service
        self.batch_window_ms = batch_window_ms / 1000.0  # Convert to seconds
        self.max_batch_size = max_batch_size

        self.request_queue = deque()
        self.video_clip_queue = deque()
        self.processing = False
        self.processing_clips = False
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'total_video_clips': 0,
            'total_clip_batches': 0,
        }

    async def add_request(self, image: Image.Image, request_id: Optional[str] = None) -> np.ndarray:
        """
        Add a request to the batch queue and wait for result.

        Args:
            image: PIL Image to embed
            request_id: Optional request ID

        Returns:
            Embedding vector
        """
        # Create a future to wait for the result
        future = asyncio.get_event_loop().create_future()

        # Add to queue
        self.request_queue.append({
            'image': image,
            'request_id': request_id,
            'future': future,
            'timestamp': time.time()
        })

        self.stats['total_requests'] += 1

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def add_video_clip_request(self, frames: List[Image.Image], request_id: Optional[str] = None) -> np.ndarray:
        """
        Add a video clip request to the batch queue and wait for result.

        Args:
            frames: List of PIL Images (video frames)
            request_id: Optional request ID

        Returns:
            Embedding vector
        """
        # Create a future to wait for the result
        future = asyncio.get_event_loop().create_future()

        # Add to queue
        self.video_clip_queue.append({
            'frames': frames,
            'request_id': request_id,
            'future': future,
            'timestamp': time.time()
        })

        self.stats['total_video_clips'] += 1

        # Start processing if not already running
        if not self.processing_clips:
            asyncio.create_task(self._process_video_clip_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process accumulated requests as a batch."""
        if self.processing:
            return

        self.processing = True

        try:
            # Wait for batch window to collect more requests
            await asyncio.sleep(self.batch_window_ms)

            # Collect requests up to max_batch_size
            batch = []
            futures = []
            request_ids = []

            while self.request_queue and len(batch) < self.max_batch_size:
                req = self.request_queue.popleft()
                batch.append(req['image'])
                futures.append(req['future'])
                request_ids.append(req.get('request_id'))

            if not batch:
                return

            logger.info(f"Processing batch of {len(batch)} requests")

            # Process batch in separate thread to not block event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._embed_batch,
                batch
            )

            # Return results to waiting requests
            for future, embedding, req_id in zip(futures, embeddings, request_ids):
                if not future.done():
                    future.set_result(embedding)

            # Update stats
            self.stats['total_batches'] += 1
            self.stats['avg_batch_size'] = (
                self.stats['total_requests'] / self.stats['total_batches']
            )

            logger.info(f"Batch complete. Avg batch size: {self.stats['avg_batch_size']:.2f}")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all waiting futures
            while self.request_queue:
                req = self.request_queue.popleft()
                if not req['future'].done():
                    req['future'].set_exception(e)
        finally:
            self.processing = False

            # If more requests came in, process them
            if self.request_queue:
                asyncio.create_task(self._process_batch())

    def _embed_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Embed a batch of images in a single GPU forward pass.
        This runs in a thread pool to not block the event loop.
        """
        return self.embedding_service.embed_images_batch_gpu(images)

    async def _process_video_clip_batch(self):
        """Process accumulated video clip requests as a batch."""
        if self.processing_clips:
            return

        self.processing_clips = True

        try:
            # Wait for batch window to collect more requests
            await asyncio.sleep(self.batch_window_ms)

            # Collect requests up to max_batch_size
            batch = []
            futures = []
            request_ids = []

            while self.video_clip_queue and len(batch) < self.max_batch_size:
                req = self.video_clip_queue.popleft()
                batch.append(req['frames'])
                futures.append(req['future'])
                request_ids.append(req.get('request_id'))

            if not batch:
                return

            logger.info(f"Processing video clip batch of {len(batch)} requests")

            # Process batch in separate thread to not block event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._embed_video_clip_batch,
                batch
            )

            # Return results to waiting requests
            for future, embedding, req_id in zip(futures, embeddings, request_ids):
                if not future.done():
                    future.set_result(embedding)

            # Update stats
            self.stats['total_clip_batches'] += 1

            logger.info(f"Video clip batch complete. Total clips: {self.stats['total_video_clips']}")

        except Exception as e:
            logger.error(f"Error processing video clip batch: {e}")
            # Set exception for all waiting futures
            while self.video_clip_queue:
                req = self.video_clip_queue.popleft()
                if not req['future'].done():
                    req['future'].set_exception(e)
        finally:
            self.processing_clips = False

            # If more requests came in, process them
            if self.video_clip_queue:
                asyncio.create_task(self._process_video_clip_batch())

    def _embed_video_clip_batch(self, clips: List[List[Image.Image]]) -> List[np.ndarray]:
        """
        Embed a batch of video clips in a single GPU forward pass.
        This runs in a thread pool to not block the event loop.
        """
        return self.embedding_service.embed_video_clips_batch_gpu(clips)


class BatchedModelServer:
    """Model server with request batching for improved throughput."""

    def __init__(self, batch_window_ms: int = 50, max_batch_size: int = 16):
        """
        Initialize batched model server.

        Args:
            batch_window_ms: Time window to collect requests (default: 50ms)
            max_batch_size: Maximum batch size (default: 16)
        """
        self.embedding_service: Optional[EmbeddingService] = None
        self.batch_processor: Optional[BatchProcessor] = None
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        self.app = FastAPI(
            title="Video-RAG Batched Model Server",
            description="Embedding model server with request batching for parallel processing",
            version="2.0.0"
        )
        self._setup_routes()

    def load_model(self):
        """Load the embedding model and initialize batch processor."""
        logger.info("Loading embedding model...")
        try:
            self.embedding_service = EmbeddingService()
            # IMPORTANT: Actually initialize and load the model into GPU memory
            self.embedding_service.initialize()
            logger.info(f"Model loaded successfully on {self.embedding_service.device}")

            self.batch_processor = BatchProcessor(
                self.embedding_service,
                batch_window_ms=self.batch_window_ms,
                max_batch_size=self.max_batch_size
            )
            logger.info(f"Batch processing enabled: window={self.batch_window_ms}ms, max_size={self.max_batch_size}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _decode_image(self, image_base64: str) -> Image.Image:
        """Decode base64 image to PIL Image."""
        try:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.embedding_service else "not_ready",
                model_loaded=self.embedding_service is not None,
                device=self.embedding_service.device if self.embedding_service else "unknown",
                batch_processing=True,
                queue_size=len(self.batch_processor.request_queue) if self.batch_processor else 0
            )

        @self.app.post("/embed/text", response_model=EmbedResponse)
        async def embed_text(request: TextEmbedRequest):
            """Embed text."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")
            try:
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self.embedding_service.embed_text, request.text
                )
                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding),
                )
            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/image", response_model=EmbedResponse)
        async def embed_image(request: ImageEmbedRequest):
            """Embed single image with batching."""
            if not self.batch_processor:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                image = self._decode_image(request.image_base64)
                embedding = await self.batch_processor.add_request(image, request.request_id)

                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding),
                    request_id=request.request_id
                )
            except Exception as e:
                logger.error(f"Error embedding image: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/video-clip", response_model=EmbedResponse)
        async def embed_video_clip(request: VideoClipEmbedRequest):
            """Embed video clip (sequence of frames) with batching."""
            if not self.batch_processor:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                frames = [self._decode_image(img_b64) for img_b64 in request.images_base64]
                embedding = await self.batch_processor.add_video_clip_request(frames, request.request_id)

                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding),
                    request_id=request.request_id
                )
            except Exception as e:
                logger.error(f"Error embedding video clip: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/multimodal", response_model=EmbedResponse)
        async def embed_multimodal(request: MultimodalEmbedRequest):
            """Embed text + image."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")
            try:
                image = self._decode_image(request.image_base64)
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self.embedding_service.embed_multimodal, request.text, image
                )
                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding),
                )
            except Exception as e:
                logger.error(f"Error embedding multimodal: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/text/batch", response_model=BatchEmbedResponse)
        async def embed_text_batch(request: BatchTextEmbedRequest):
            """Embed multiple texts in one call."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")
            try:
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, self.embedding_service.embed_texts_batch, request.texts
                )
                return BatchEmbedResponse(
                    embeddings=[emb.tolist() for emb in embeddings],
                    dimension=len(embeddings[0]) if embeddings else 0,
                    count=len(embeddings),
                )
            except Exception as e:
                logger.error(f"Error embedding text batch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/image/batch", response_model=BatchEmbedResponse)
        async def embed_image_batch(request: BatchImageEmbedRequest):
            """Embed multiple images in a single GPU forward pass."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")
            try:
                images = [self._decode_image(img_b64) for img_b64 in request.images_base64]
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, self.embedding_service.embed_images_batch_gpu, images
                )
                return BatchEmbedResponse(
                    embeddings=[emb.tolist() for emb in embeddings],
                    dimension=len(embeddings[0]) if embeddings else 0,
                    count=len(embeddings),
                )
            except Exception as e:
                logger.error(f"Error embedding image batch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/video-clip/batch", response_model=BatchEmbedResponse)
        async def embed_video_clip_batch(request: BatchVideoClipEmbedRequest):
            """Embed multiple video clips in a single GPU forward pass."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")
            try:
                clips = [
                    [self._decode_image(img_b64) for img_b64 in clip]
                    for clip in request.clips
                ]
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, self.embedding_service.embed_video_clips_batch_gpu, clips
                )
                return BatchEmbedResponse(
                    embeddings=[emb.tolist() for emb in embeddings],
                    dimension=len(embeddings[0]) if embeddings else 0,
                    count=len(embeddings),
                )
            except Exception as e:
                logger.error(f"Error embedding video clip batch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats")
        async def get_stats():
            """Get batch processing statistics."""
            if not self.batch_processor:
                return {"error": "Batch processor not initialized"}

            return {
                "total_image_requests": self.batch_processor.stats['total_requests'],
                "total_image_batches": self.batch_processor.stats['total_batches'],
                "total_video_clips": self.batch_processor.stats['total_video_clips'],
                "total_clip_batches": self.batch_processor.stats['total_clip_batches'],
                "avg_batch_size": round(self.batch_processor.stats['avg_batch_size'], 2),
                "current_image_queue": len(self.batch_processor.request_queue),
                "current_clip_queue": len(self.batch_processor.video_clip_queue)
            }


def main():
    """Main entry point."""
    host = os.getenv("MODEL_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MODEL_SERVER_PORT", 8001))
    batch_window_ms = int(os.getenv("BATCH_WINDOW_MS", 50))
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 16))

    logger.info("Starting Batched Model Server")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Batch window: {batch_window_ms}ms")
    logger.info(f"Max batch size: {max_batch_size}")

    server = BatchedModelServer(
        batch_window_ms=batch_window_ms,
        max_batch_size=max_batch_size
    )
    server.load_model()

    uvicorn.run(
        server.app,
        host=host,
        port=port,
        timeout_keep_alive=300,
        limit_concurrency=64,
    )


if __name__ == "__main__":
    main()
