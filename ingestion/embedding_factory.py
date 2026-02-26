"""
Embedding Factory - Creates local or remote embedding service based on configuration.
"""

import os
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def get_embedding_service(force_local: bool = False, force_remote: bool = False):
    """
    Get embedding service (local or remote based on configuration).

    Args:
        force_local: Force local embedding service
        force_remote: Force remote embedding service

    Returns:
        Embedding service instance (local or remote)
    """
    # Check environment variable for mode
    use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "false").lower() == "true"

    # Override with force flags
    if force_local:
        use_remote = False
    elif force_remote:
        use_remote = True

    if use_remote:
        logger.info("Using REMOTE embedding service (model server)")
        from ingestion.remote_embedding_client import RemoteEmbeddingClient

        model_server_url = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")
        timeout = int(os.getenv("MODEL_SERVER_TIMEOUT", "300"))

        return RemoteEmbeddingClient(base_url=model_server_url, timeout=timeout)
    else:
        logger.info("Using LOCAL embedding service")
        from ingestion.embedding_service import EmbeddingService

        service = EmbeddingService()
        service.initialize()
        return service
