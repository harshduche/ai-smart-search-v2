"""Health check routes."""

from fastapi import APIRouter

from api.models.responses import HealthResponse, CollectionStats
from search.vector_store import get_vector_store
from search.search_service import get_search_service
from ingestion.embedding_service import get_embedding_service
import config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and configuration info.
    """
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    search_service = get_search_service()

    # List all org collections instead of querying a single one
    try:
        vector_store._connect()
        org_collections = vector_store.list_org_collections()
        collection_info = {
            "status": "connected",
            "org_collections": len(org_collections),
        }
    except Exception as e:
        collection_info = {"error": str(e)}

    try:
        search_stats = search_service.get_stats()
    except Exception as e:
        search_stats = {"error": str(e)}

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        embedding_model=config.MODEL_NAME,
        using_mock_embeddings=embedding_service.is_mock() if embedding_service._initialized else False,
        vector_store=collection_info,
        search_service=search_stats,
    )


@router.get("/stats/{organization_id}", response_model=CollectionStats)
async def get_stats(organization_id: str):
    """
    Get collection statistics for a specific organization.
    """
    vector_store = get_vector_store()
    info = vector_store.get_collection_info(organization_id)

    return CollectionStats(
        name=info["name"],
        vectors_count=info.get("indexed_vectors_count", 0),
        points_count=info.get("points_count", 0),
        status=str(info.get("status", "unknown")),
    )
