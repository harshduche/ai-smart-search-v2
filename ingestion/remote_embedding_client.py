"""
Remote Embedding Client for connecting to Model Server.

This client allows workers to use embeddings from a remote model server
instead of loading the model locally.
"""

import base64
import io
import logging
import time
from typing import List
import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def _build_session(max_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Build a requests Session with retry + exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class RemoteEmbeddingClient:
    """Client for remote embedding model server."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 300):
        """
        Initialize remote embedding client.

        Args:
            base_url: Base URL of model server
            timeout: Request timeout in seconds (default 5 minutes for large batches)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = _build_session()
        logger.info(f"Initialized remote embedding client: {self.base_url}")

        # Check server health
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            logger.info(f"Model server status: {health}")

            if not health.get('model_loaded'):
                logger.warning("Model server reports model not loaded!")
        except Exception as e:
            logger.error(f"Failed to connect to model server: {e}")
            raise RuntimeError(f"Model server not available at {self.base_url}")

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 JPEG (much smaller than PNG for photos)."""
        buffer = io.BytesIO()
        img = image.convert('RGB') if image.mode == 'RGBA' else image
        img.save(buffer, format='JPEG', quality=90)
        image_bytes = buffer.getvalue()
        buffer.close()
        return base64.b64encode(image_bytes).decode('utf-8')

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using remote model server.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.session.post(
                f"{self.base_url}/embed/text",
                json={"text": text},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error calling remote embed_text: {e}")
            raise

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed image using remote model server.

        Args:
            image: PIL Image

        Returns:
            Embedding vector as numpy array
        """
        try:
            image_b64 = self._encode_image(image)
            response = self.session.post(
                f"{self.base_url}/embed/image",
                json={"image_base64": image_b64},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error calling remote embed_image: {e}")
            raise

    def embed_video_clip(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed video clip (sequence of frames) using remote model server.

        Args:
            images: List of PIL Images (frames)

        Returns:
            Embedding vector as numpy array
        """
        try:
            images_b64 = [self._encode_image(img) for img in images]
            response = self.session.post(
                f"{self.base_url}/embed/video-clip",
                json={"images_base64": images_b64},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error calling remote embed_video_clip: {e}")
            raise

    def embed_multimodal(self, text: str, image: Image.Image) -> np.ndarray:
        """
        Embed text + image using remote model server.

        Args:
            text: Text component
            image: PIL Image

        Returns:
            Embedding vector as numpy array
        """
        try:
            image_b64 = self._encode_image(image)
            response = self.session.post(
                f"{self.base_url}/embed/multimodal",
                json={
                    "text": text,
                    "image_base64": image_b64
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error calling remote embed_multimodal: {e}")
            raise

    def embed_texts_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts using remote model server.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        try:
            response = self.session.post(
                f"{self.base_url}/embed/text/batch",
                json={"texts": texts},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return [np.array(emb, dtype=np.float32) for emb in result['embeddings']]
        except Exception as e:
            logger.error(f"Error calling remote embed_texts_batch: {e}")
            raise

    def embed_images_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Embed multiple images using remote model server.

        Args:
            images: List of PIL Images

        Returns:
            List of embedding vectors
        """
        try:
            images_b64 = [self._encode_image(img) for img in images]
            response = self.session.post(
                f"{self.base_url}/embed/image/batch",
                json={"images_base64": images_b64},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return [np.array(emb, dtype=np.float32) for emb in result['embeddings']]
        except Exception as e:
            logger.error(f"Error calling remote embed_images_batch: {e}")
            raise

    def embed_video_clips_batch(self, clips: List[List[Image.Image]]) -> List[np.ndarray]:
        """
        Embed multiple video clips in a single request to the batch endpoint.

        Args:
            clips: List of clips, where each clip is a list of PIL Images

        Returns:
            List of embedding vectors
        """
        try:
            clips_b64 = [
                [self._encode_image(img) for img in clip]
                for clip in clips
            ]
            response = self.session.post(
                f"{self.base_url}/embed/video-clip/batch",
                json={"clips": clips_b64},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return [np.array(emb, dtype=np.float32) for emb in result['embeddings']]
        except Exception as e:
            logger.warning(f"Batch video-clip endpoint failed ({e}), falling back to sequential")
            embeddings = []
            for clip in clips:
                embedding = self.embed_video_clip(clip)
                embeddings.append(embedding)
            return embeddings

    def embed_images_batch_gpu(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Alias for embed_images_batch for API compatibility with EmbeddingService."""
        return self.embed_images_batch(images)

    def embed_video_clips_batch_gpu(self, clips: List[List[Image.Image]]) -> List[np.ndarray]:
        """Alias for embed_video_clips_batch for API compatibility with EmbeddingService."""
        return self.embed_video_clips_batch(clips)
