"""Search service for querying the vector store."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from .vector_store import VectorStore, get_vector_store
from .reranker_service import RerankerService, get_reranker_service
from ingestion.embedding_service import EmbeddingService, get_embedding_service
import config


class SearchService:
    """Service for searching the security footage database."""

    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_service: EmbeddingService = None,
        reranker_service: RerankerService = None,
    ):
        """
        Initialize the search service.

        Args:
            vector_store: VectorStore instance (uses singleton if not provided)
            embedding_service: EmbeddingService instance (uses singleton if not provided)
            reranker_service: RerankerService instance (uses singleton if not provided)
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.reranker_service = reranker_service or get_reranker_service()

    def search_text(
        self,
        query: str,
        organization_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = False,
        media_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search using a text query.

        Args:
            query: Text query (e.g., "person near fence at night")
            organization_id: Organization identifier (determines Qdrant collection)
            top_k: Number of results to return
            filters: Optional metadata filters
            use_reranker: Whether to rerank results using the reranker model
            media_ids: Optional list of media IDs to restrict search (geo pre-filter)

        Returns:
            List of search results with scores and metadata
        """
        search_limit = min(top_k * 3, config.MAX_TOP_K) if use_reranker else min(top_k, config.MAX_TOP_K)

        query_embedding = self.embedding_service.embed_text(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            organization_id=organization_id,
            top_k=search_limit,
            filters=filters,
            media_ids=media_ids,
        )

        if use_reranker and results:
            results = self.reranker_service.rerank(
                query_text=query,
                results=results,
                top_k=top_k,
            )

        return results

    def search_image(
        self,
        image: Union[str, Path, Image.Image],
        organization_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search using a reference image.

        Args:
            image: Image path or PIL Image object
            organization_id: Organization identifier (determines Qdrant collection)
            top_k: Number of results to return
            filters: Optional metadata filters
            use_reranker: Whether to rerank results using the reranker model

        Returns:
            List of search results with scores and metadata
        """
        search_limit = min(top_k * 3, config.MAX_TOP_K) if use_reranker else min(top_k, config.MAX_TOP_K)

        query_embedding = self.embedding_service.embed_image(image)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            organization_id=organization_id,
            top_k=search_limit,
            filters=filters,
        )

        if use_reranker and results:
            results = self.reranker_service.rerank(
                query_image=image,
                results=results,
                top_k=top_k,
            )

        return results

    def search_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
        organization_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search using combined text and image query.

        Args:
            text: Text query
            image: Image path or PIL Image object
            organization_id: Organization identifier (determines Qdrant collection)
            top_k: Number of results to return
            filters: Optional metadata filters
            use_reranker: Whether to rerank results using the reranker model

        Returns:
            List of search results with scores and metadata
        """
        search_limit = min(top_k * 3, config.MAX_TOP_K) if use_reranker else min(top_k, config.MAX_TOP_K)

        query_embedding = self.embedding_service.embed_multimodal(text, image)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            organization_id=organization_id,
            top_k=search_limit,
            filters=filters,
        )

        if use_reranker and results:
            results = self.reranker_service.rerank(
                query_text=text,
                query_image=image,
                results=results,
                top_k=top_k,
            )

        return results

    def search_ocr(
        self,
        text_query: str,
        organization_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for text visible in footage (OCR).

        Args:
            text_query: Text to search for (e.g., "license plate 7829")
            organization_id: Organization identifier (determines Qdrant collection)
            top_k: Number of results
            filters: Optional metadata filters
            use_reranker: Whether to rerank results using the reranker model

        Returns:
            List of search results
        """
        enhanced_query = f"Text visible in image showing: {text_query}"

        return self.search_text(
            query=enhanced_query,
            organization_id=organization_id,
            top_k=top_k,
            filters=filters,
            use_reranker=use_reranker,
        )

    def search_similar_to_id(
        self,
        point_id: str,
        organization_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find frames similar to a given frame by ID.

        Args:
            point_id: ID of the reference frame
            organization_id: Organization identifier (determines Qdrant collection)
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of similar frames
        """
        raise NotImplementedError("Search by ID requires embedding storage")

    def get_stats(self, organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Get search service statistics."""
        stats: Dict[str, Any] = {
            "embedding_dim": self.embedding_service.get_embedding_dim(),
            "using_mock_embeddings": self.embedding_service.is_mock(),
            "using_mock_reranker": self.reranker_service.is_mock(),
        }
        if organization_id:
            stats["collection"] = self.vector_store.get_collection_info(organization_id)
        return stats


# Singleton instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create the singleton search service."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
