"""Embedding service using Qwen3-VL-Embedding model.

Based on official implementation: https://github.com/QwenLM/Qwen3-VL-Embedding
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
from tqdm import tqdm

import config
from ingestion.qwen3_vl_model import Qwen3VLEmbedder
# Note: Individual embedding calls are not traced to reduce overhead
# Tracing is done at the pipeline level (ingest_pipeline.py)


class EmbeddingService:
    """
    Service for generating embeddings using Qwen3-VL-Embedding model.

    This model maps both text AND images to the same 2048-dimensional vector space,
    enabling cross-modal similarity search.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        use_float16: bool = True,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: HuggingFace model name (default: Qwen/Qwen3-VL-Embedding-2B)
            device: Device to run on ('cuda', 'cpu', 'mps')
            batch_size: Batch size for processing
            use_float16: Whether to use float16 (for CUDA only)
        """
        self.model_name = model_name or config.MODEL_NAME
        self.device = device or config.DEVICE
        self.batch_size = batch_size or config.BATCH_SIZE
        self.use_float16 = use_float16 and self.device == "cuda"
        self.embedder = None
        self._initialized = False
        self._use_mock = False

    def initialize(self):
        """Load the model and processor."""
        if self._initialized:
            return

        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")

        try:
            # Prepare kwargs for model loading
            model_kwargs = {}

            if self.use_float16:
                model_kwargs["dtype"] = torch.float16
                print("Using float16 precision")

            # Initialize the Qwen3VL embedder
            self.embedder = Qwen3VLEmbedder(
                model_name_or_path=self.model_name,
                **model_kwargs
            )

            self._initialized = True
            print(f"Model loaded successfully on {self.device}!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock embeddings for development...")
            self._initialized = True
            self._use_mock = True

    def _get_mock_embedding(self, seed: int = None) -> np.ndarray:
        """Generate a mock embedding for development/testing."""
        if seed is not None:
            np.random.seed(seed)
        embedding = np.random.randn(config.EMBEDDING_DIM).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text query.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector (2048-dim)
        """
        self.initialize()

        if self._use_mock:
            return self._get_mock_embedding(seed=hash(text) % 2**32)

        try:
            # Use the official Qwen3VL embedder
            inputs = [{'text': text}]
            embeddings = self.embedder.process(inputs, normalize=True)

            # Convert to numpy and return first embedding
            return embeddings[0].cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return self._get_mock_embedding(seed=hash(text) % 2**32)

    def embed_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Generate embedding for an image.

        Args:
            image: Image path or PIL Image object

        Returns:
            Normalized embedding vector (2048-dim)
        """
        self.initialize()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image).convert("RGB")
        else:
            image_path = "pil_image"
            pil_image = image.convert("RGB") if image.mode != "RGB" else image

        if self._use_mock:
            return self._get_mock_embedding(seed=hash(image_path) % 2**32)

        try:
            # Use the official Qwen3VL embedder
            inputs = [{'image': pil_image}]
            embeddings = self.embedder.process(inputs, normalize=True)

            # Convert to numpy and return first embedding
            return embeddings[0].cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return self._get_mock_embedding()

    def embed_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Generate embedding for combined text and image query.

        Args:
            text: Text query
            image: Image path or PIL Image object

        Returns:
            Normalized embedding vector (2048-dim)
        """
        self.initialize()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB") if image.mode != "RGB" else image

        if self._use_mock:
            combined = f"{text}_{hash(str(image))}"
            return self._get_mock_embedding(seed=hash(combined) % 2**32)

        try:
            # Use the official Qwen3VL embedder
            inputs = [{'text': text, 'image': pil_image}]
            embeddings = self.embedder.process(inputs, normalize=True)

            # Convert to numpy and return first embedding
            return embeddings[0].cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"Error generating multimodal embedding: {e}")
            return self._get_mock_embedding()

    def embed_video_clip(
        self,
        frames: List[Image.Image],
    ) -> np.ndarray:
        """
        Generate an embedding for a short video clip represented as a
        sequence of frames.

        This uses Qwen3-VL's native video embedding path by passing the
        frames as a ``video`` input, allowing the model to reason over
        temporal context instead of treating each frame independently.

        Args:
            frames: Ordered list of PIL Image frames from a clip.

        Returns:
            Normalized embedding vector (2048-dim)
        """
        self.initialize()

        if not frames:
            raise ValueError("embed_video_clip requires at least one frame")

        if self._use_mock:
            # Use a deterministic seed based on the first frame's id
            seed = id(frames[0]) % 2**32
            return self._get_mock_embedding(seed=seed)

        try:
            # Qwen3VLEmbedder will treat a list of PIL.Image frames as a video
            inputs = [{'video': frames}]
            embeddings = self.embedder.process(inputs, normalize=True)

            return embeddings[0].cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"Error generating video clip embedding: {e}")
            return self._get_mock_embedding()

    def embed_video_clips_batch(
        self,
        clips: List[List[Image.Image]],
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple video clips efficiently.

        Uses GPU batch processing when available, with periodic memory cleanup.

        Args:
            clips: List of video clips, each clip is a list of PIL Image frames
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        self.initialize()

        if not clips:
            return []

        batch_size = self.batch_size
        all_embeddings = []

        chunks = [clips[i:i + batch_size] for i in range(0, len(clips), batch_size)]
        iterator = tqdm(chunks, desc="Embedding video clip batches") if show_progress else chunks

        for chunk in iterator:
            batch_embs = self.embed_video_clips_batch_gpu(chunk)
            all_embeddings.extend(batch_embs)

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return all_embeddings

    def embed_texts_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts in a single forward pass.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        self.initialize()

        if not texts:
            return []

        if self._use_mock:
            return [self._get_mock_embedding(seed=hash(t) % 2**32) for t in texts]

        try:
            inputs = [{'text': t} for t in texts]
            embeddings_tensor = self.embedder.process(inputs, normalize=True)
            return [
                embeddings_tensor[i].cpu().numpy().astype(np.float32)
                for i in range(embeddings_tensor.shape[0])
            ]
        except Exception as e:
            print(f"Error in batch text embedding, falling back to sequential: {e}")
            return [self.embed_text(t) for t in texts]

    def embed_images_batch_gpu(
        self,
        images: List[Image.Image],
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of PIL Images using a single GPU
        forward pass via the Qwen3VL model's native batching support.

        This is significantly faster than calling embed_image() in a loop
        because it pads inputs into a single tensor and runs one forward pass.

        Args:
            images: List of PIL Image objects (already loaded into memory)

        Returns:
            List of embedding vectors
        """
        self.initialize()

        if not images:
            return []

        if self._use_mock:
            return [self._get_mock_embedding() for _ in images]

        try:
            pil_images = [
                img.convert("RGB") if img.mode != "RGB" else img
                for img in images
            ]
            inputs = [{'image': img} for img in pil_images]
            embeddings_tensor = self.embedder.process(inputs, normalize=True)
            return [
                embeddings_tensor[i].cpu().numpy().astype(np.float32)
                for i in range(embeddings_tensor.shape[0])
            ]
        except Exception as e:
            print(f"Error in batch GPU embedding, falling back to sequential: {e}")
            return [self.embed_image(img) for img in images]

    def embed_video_clips_batch_gpu(
        self,
        clips: List[List[Image.Image]],
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple video clips in a single GPU forward
        pass via the Qwen3VL model's native video batching support.

        Args:
            clips: List of video clips, each clip is a list of PIL Image frames

        Returns:
            List of embedding vectors
        """
        self.initialize()

        if not clips:
            return []

        if self._use_mock:
            return [self._get_mock_embedding() for _ in clips]

        try:
            inputs = [{'video': frames} for frames in clips]
            embeddings_tensor = self.embedder.process(inputs, normalize=True)
            return [
                embeddings_tensor[i].cpu().numpy().astype(np.float32)
                for i in range(embeddings_tensor.shape[0])
            ]
        except Exception as e:
            print(f"Error in batch GPU clip embedding, falling back to sequential: {e}")
            return [self.embed_video_clip(frames) for frames in clips]

    def embed_images_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of images.

        Args:
            images: List of image paths or PIL Image objects
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        self.initialize()

        embeddings = []
        iterator = tqdm(images, desc="Embedding images") if show_progress else images

        for img in iterator:
            embedding = self.embed_image(img)
            embeddings.append(embedding)

            # Clear CUDA cache periodically to avoid OOM
            if self.device == "cuda" and len(embeddings) % 100 == 0:
                torch.cuda.empty_cache()

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return config.EMBEDDING_DIM

    def is_mock(self) -> bool:
        """Check if using mock embeddings."""
        return self._use_mock


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def reset_embedding_service():
    """Reset the singleton (useful for testing)."""
    global _embedding_service
    _embedding_service = None
