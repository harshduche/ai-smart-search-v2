"""
Model Server for serving embedding models via HTTP API.

This allows multiple workers to share a single model instance,
reducing memory usage and enabling workers without GPU.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from ingestion.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# Request/Response Models
class TextEmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed")


class ImageEmbedRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")


class VideoClipEmbedRequest(BaseModel):
    images_base64: List[str] = Field(..., description="List of base64 encoded images (frames)")


class MultimodalEmbedRequest(BaseModel):
    text: str = Field(..., description="Text component")
    image_base64: str = Field(..., description="Base64 encoded image")


class BatchTextEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")


class BatchImageEmbedRequest(BaseModel):
    images_base64: List[str] = Field(..., description="List of base64 encoded images")


class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


class ModelServer:
    """Model server that loads embedding model once and serves requests."""

    def __init__(self):
        """Initialize model server."""
        self.embedding_service: Optional[EmbeddingService] = None
        self.app = FastAPI(
            title="Video-RAG Model Server",
            description="Embedding model serving API for distributed workers",
            version="1.0.0"
        )
        self._setup_routes()

    def load_model(self):
        """Load the embedding model."""
        logger.info("Loading embedding model...")
        try:
            self.embedding_service = EmbeddingService()
            logger.info(f"Model loaded successfully on device: {self.embedding_service.device}")
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
                model_name=config.MODEL_NAME if self.embedding_service else "unknown"
            )

        @self.app.post("/embed/text", response_model=EmbedResponse)
        async def embed_text(request: TextEmbedRequest):
            """Embed text."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                embedding = self.embedding_service.embed_text(request.text)
                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding)
                )
            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/image", response_model=EmbedResponse)
        async def embed_image(request: ImageEmbedRequest):
            """Embed single image."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                image = self._decode_image(request.image_base64)
                embedding = self.embedding_service.embed_image(image)
                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding)
                )
            except Exception as e:
                logger.error(f"Error embedding image: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/video-clip", response_model=EmbedResponse)
        async def embed_video_clip(request: VideoClipEmbedRequest):
            """Embed video clip (sequence of frames)."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                images = [self._decode_image(img_b64) for img_b64 in request.images_base64]
                embedding = self.embedding_service.embed_video_clip(images)
                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding)
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
                embedding = self.embedding_service.embed_multimodal(request.text, image)
                return EmbedResponse(
                    embedding=embedding.tolist(),
                    dimension=len(embedding)
                )
            except Exception as e:
                logger.error(f"Error embedding multimodal: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/text/batch", response_model=BatchEmbedResponse)
        async def embed_text_batch(request: BatchTextEmbedRequest):
            """Embed multiple texts."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                embeddings = self.embedding_service.embed_texts_batch(request.texts)
                return BatchEmbedResponse(
                    embeddings=[emb.tolist() for emb in embeddings],
                    dimension=len(embeddings[0]) if embeddings else 0,
                    count=len(embeddings)
                )
            except Exception as e:
                logger.error(f"Error embedding text batch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/embed/image/batch", response_model=BatchEmbedResponse)
        async def embed_image_batch(request: BatchImageEmbedRequest):
            """Embed multiple images."""
            if not self.embedding_service:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                images = [self._decode_image(img_b64) for img_b64 in request.images_base64]
                embeddings = self.embedding_service.embed_images_batch(images)
                return BatchEmbedResponse(
                    embeddings=[emb.tolist() for emb in embeddings],
                    dimension=len(embeddings[0]) if embeddings else 0,
                    count=len(embeddings)
                )
            except Exception as e:
                logger.error(f"Error embedding image batch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Video-RAG Model Server",
                "status": "running",
                "model_loaded": self.embedding_service is not None,
                "endpoints": {
                    "health": "GET /health",
                    "embed_text": "POST /embed/text",
                    "embed_image": "POST /embed/image",
                    "embed_video_clip": "POST /embed/video-clip",
                    "embed_multimodal": "POST /embed/multimodal",
                    "embed_text_batch": "POST /embed/text/batch",
                    "embed_image_batch": "POST /embed/image/batch",
                }
            }


def main():
    """Run the model server."""
    # Configuration
    host = os.getenv("MODEL_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MODEL_SERVER_PORT", "8001"))

    logger.info(f"Starting Model Server on {host}:{port}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME}")

    # Initialize server
    server = ModelServer()

    # Load model
    server.load_model()

    # Start server
    logger.info("Model Server ready to accept requests")
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
