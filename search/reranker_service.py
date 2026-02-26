"""Reranker service using Qwen3-VL-Reranker model."""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image

import config
from ingestion.qwen3_vl_reranker import Qwen3VLReranker

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Service for reranking search results using Qwen3-VL-Reranker model.

    This model evaluates query-document pairs to provide more accurate relevance scores.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        use_float16: bool = True,
    ):
        """
        Initialize the reranker service.

        Args:
            model_name: HuggingFace model name (default: uses RERANKER_MODEL_NAME from config)
            device: Device to run on ('cuda', 'cpu', 'mps')
            use_float16: Whether to use float16 (for CUDA only)
        """
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.device = device or config.DEVICE
        self.use_float16 = use_float16 and self.device == "cuda"
        self.reranker = None
        self._initialized = False
        self._use_mock = False

    def initialize(self):
        """Load the reranker model and processor."""
        if self._initialized:
            return

        print(f"Loading reranker model: {self.model_name}")
        print(f"Device: {self.device}")

        try:
            # Prepare kwargs for model loading
            model_kwargs = {}

            if self.use_float16:
                model_kwargs["dtype"] = torch.float16
                print("Using float16 precision")

            # Initialize the Qwen3VL reranker
            self.reranker = Qwen3VLReranker(
                model_name_or_path=self.model_name,
                **model_kwargs
            )

            self._initialized = True
            print(f"Reranker model loaded successfully on {self.device}!")

        except Exception as e:
            print(f"Error loading reranker model: {e}")
            print("Falling back to mock reranking (no reordering)...")
            self._initialized = True
            self._use_mock = True

    def rerank(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Union[str, Path, Image.Image]] = None,
        results: List[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results based on query.

        Args:
            query_text: Text query
            query_image: Image query (path or PIL Image)
            results: List of search results to rerank
            top_k: Optional limit on number of results to return after reranking

        Returns:
            Reranked list of search results with updated scores
        """
        self.initialize()

        if not results:
            return []

        if self._use_mock:
            # Mock reranking - just return results as-is
            logger.warning("Using mock reranker - results not reranked")
            return results[:top_k] if top_k else results

        try:
            # Prepare query
            query = {}
            if query_text:
                query['text'] = query_text
            if query_image:
                if isinstance(query_image, (str, Path)):
                    query['image'] = Image.open(query_image).convert("RGB")
                else:
                    query['image'] = query_image.convert("RGB") if query_image.mode != "RGB" else query_image

            # Prepare documents from results
            documents = []
            for result in results:
                doc = {}

                # Use thumbnail as document image if available
                thumbnail_path = result.get('thumbnail_path')
                if thumbnail_path:
                    try:
                        # Handle both absolute paths and relative paths
                        if not Path(thumbnail_path).is_absolute():
                            thumbnail_path = config.THUMBNAILS_DIR / thumbnail_path
                        doc['image'] = Image.open(thumbnail_path).convert("RGB")
                    except Exception as e:
                        logger.warning(f"Could not load thumbnail {thumbnail_path}: {e}")

                # Add metadata as text context
                metadata_parts = []
                if result.get('zone'):
                    metadata_parts.append(f"Zone: {result['zone']}")
                if result.get('is_night') is not None:
                    time_of_day = "night" if result['is_night'] else "day"
                    metadata_parts.append(f"Time: {time_of_day}")
                if result.get('timestamp'):
                    metadata_parts.append(f"Timestamp: {result['timestamp']}")

                if metadata_parts:
                    doc['text'] = ", ".join(metadata_parts)

                documents.append(doc)

            # Get reranking scores
            rerank_input = {
                "query": query,
                "documents": documents,
                "instruction": "Given a search query, retrieve relevant candidates that answer the query."
            }

            scores = self.reranker.process(rerank_input)

            # Update results with new scores
            reranked_results = []
            for i, result in enumerate(results):
                result_copy = result.copy()
                result_copy['original_score'] = result.get('score', 0.0)
                result_copy['rerank_score'] = scores[i] if i < len(scores) else 0.0
                result_copy['score'] = scores[i] if i < len(scores) else 0.0
                reranked_results.append(result_copy)

            # Sort by rerank score (descending)
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

            # Apply top_k if specified
            if top_k:
                reranked_results = reranked_results[:top_k]

            return reranked_results

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Returning original results without reranking")
            return results[:top_k] if top_k else results

    def is_mock(self) -> bool:
        """Check if using mock reranker."""
        return self._use_mock


# Singleton instance
_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create the singleton reranker service."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service


def reset_reranker_service():
    """Reset the singleton (useful for testing)."""
    global _reranker_service
    _reranker_service = None
