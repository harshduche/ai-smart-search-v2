"""Qdrant vector store for storing and searching embeddings.

Uses per-organization collections (collection per org) so each
organization's data is fully isolated at the Qdrant level.
"""

import os
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Set
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    PayloadSchemaType,
)

import config

logger = logging.getLogger(__name__)

# Prefix used for all org-based collections
_COLLECTION_PREFIX = ""


def collection_name_for_org(organization_id: str) -> str:
    """Return the Qdrant collection name for a given organization."""
    safe_id = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in organization_id
    )
    return f"{_COLLECTION_PREFIX}{safe_id}"


class VectorStore:
    """Qdrant-based vector store with per-organization collections.

    Supports two connection modes:
      - **Local / self-hosted**: pass ``host`` + ``port``, or set
        ``QDRANT_HOST`` / ``QDRANT_PORT`` in the environment.
      - **Qdrant Cloud**: pass ``url`` + ``api_key``, or set
        ``QDRANT_URL`` / ``QDRANT_API_KEY`` in the environment.

    When both ``url`` and host/port are provided, the cloud URL takes
    priority.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        url: str = None,
        api_key: str = None,
        timeout: int = None,
    ):
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        self.url = url or config.QDRANT_URL or ""
        self.api_key = api_key or config.QDRANT_API_KEY or ""
        self.timeout = timeout or int(os.getenv("QDRANT_TIMEOUT", "120"))
        self.client: Optional[QdrantClient] = None
        self._connected = False
        self._ensured_collections: Set[str] = set()

    @property
    def _use_cloud(self) -> bool:
        return bool(self.url)

    def _connect(self):
        """Establish the Qdrant client connection (idempotent)."""
        if self._connected:
            return

        if self._use_cloud:
            print(f"Connecting to Qdrant Cloud at {self.url} (timeout={self.timeout}s)")
            kwargs: Dict[str, Any] = {"url": self.url, "timeout": self.timeout}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self.client = QdrantClient(**kwargs)
        else:
            print(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)

        self._connected = True

    def ensure_collection(self, organization_id: str) -> str:
        """Ensure the collection for the given organization exists."""
        self._connect()
        col_name = collection_name_for_org(organization_id)

        if col_name in self._ensured_collections:
            return col_name

        existing = {c.name for c in self.client.get_collections().collections}
        if col_name not in existing:
            self._create_collection(col_name)
        else:
            print(f"Collection '{col_name}' already exists")

        self._ensured_collections.add(col_name)
        return col_name

    def _create_collection(self, collection_name: str):
        """Create a vector collection with the full payload index schema."""
        print(f"Creating collection '{collection_name}'")

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

        # Keyword indexes
        keyword_fields = [
            "media_id", "source_file", "file_type", "source_type",
            "flight_id", "flight_type", "mission_id", "mission_type",
            "organization_id", "site_id", "zone",
        ]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        # Integer indexes
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="frame_number",
            field_schema=PayloadSchemaType.INTEGER,
        )

        # Float indexes (geo coordinates — job-level and drone telemetry)
        for field in ["latitude", "longitude", "drone_lat", "drone_lng",
                      "drone_alt_rel", "drone_alt_abs"]:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.FLOAT,
            )

        # Datetime indexes
        for field in ["timestamp", "capture_timestamp"]:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.DATETIME,
            )

        # Boolean indexes
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="is_night",
            field_schema=PayloadSchemaType.BOOL,
        )

        print(f"Collection '{collection_name}' created with indexes")

    def insert(
        self,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        organization_id: str,
        point_id: Optional[str] = None,
    ) -> str:
        """Insert a single embedding with metadata.

        Args:
            embedding: The embedding vector.
            metadata: Metadata dict.
            organization_id: Organization identifier (determines collection).
            point_id: Optional ID (auto-generated if not provided).

        Returns:
            The point ID.
        """
        col_name = self.ensure_collection(organization_id)

        if point_id is None:
            source = metadata.get("source_file", "unknown")
            frame = metadata.get("frame_number", 0)
            point_id = f"{source}_{frame}"

        vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        point = PointStruct(
            id=hash(point_id) % (2**63),
            vector=vector,
            payload={**metadata, "point_id": point_id},
        )

        self.client.upsert(collection_name=col_name, points=[point])
        return point_id

    def _upsert_with_retry(self, col_name: str, points: List[PointStruct],
                            max_retries: int = 3):
        """Upsert points with exponential backoff retry for transient errors."""
        for attempt in range(max_retries):
            try:
                self.client.upsert(collection_name=col_name, points=points)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"Qdrant upsert failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

    def insert_batch(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
        organization_id: str,
        batch_size: int = 100,
    ) -> int:
        """Insert a batch of embeddings with metadata.

        Args:
            embeddings: List of embedding vectors.
            metadata_list: List of metadata dicts.
            organization_id: Organization identifier (determines collection).
            batch_size: Number of points to insert at once.

        Returns:
            Number of points inserted.
        """
        col_name = self.ensure_collection(organization_id)

        points = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            source = metadata.get("source_file", "unknown")
            frame = metadata.get("frame_number", i)
            point_id = f"{source}_{frame}"

            vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            point = PointStruct(
                id=hash(point_id) % (2**63),
                vector=vector,
                payload={**metadata, "point_id": point_id},
            )
            points.append(point)

            if len(points) >= batch_size:
                self._upsert_with_retry(col_name, points)
                points = []

        if points:
            self._upsert_with_retry(col_name, points)

        return len(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        organization_id: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        media_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in an organization's collection.

        Args:
            query_embedding: The query embedding vector.
            organization_id: Organization identifier (determines collection).
            top_k: Number of results to return.
            filters: Optional filters dict.  Supported keys:
                Keyword: source_file, zone, site_id, flight_id,
                    flight_type, mission_id, mission_type, media_id,
                    file_type, source_type, organization_id.
                Boolean: is_night.
                Datetime range: start_time/end_time (on timestamp),
                    capture_start/capture_end (on capture_timestamp).
                Float range: latitude_min/latitude_max,
                    longitude_min/longitude_max.

        Returns:
            List of search results with score and metadata.
        """
        col_name = self.ensure_collection(organization_id)

        filter_conditions = []

        if filters:
            # Keyword match filters
            _keyword_keys = [
                "source_file", "zone", "site_id", "flight_id",
                "flight_type", "mission_id", "mission_type",
                "media_id", "file_type", "source_type", "organization_id",
            ]
            for key in _keyword_keys:
                if key in filters:
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=filters[key]))
                    )

            # Boolean
            if "is_night" in filters:
                filter_conditions.append(
                    FieldCondition(key="is_night", match=MatchValue(value=filters["is_night"]))
                )

            # Temporal range on timestamp
            if "start_time" in filters or "end_time" in filters:
                rng = {}
                if "start_time" in filters:
                    rng["gte"] = filters["start_time"]
                if "end_time" in filters:
                    rng["lte"] = filters["end_time"]
                filter_conditions.append(
                    FieldCondition(key="timestamp", range=Range(**rng))
                )

            # Temporal range on capture_timestamp
            if "capture_start" in filters or "capture_end" in filters:
                rng = {}
                if "capture_start" in filters:
                    rng["gte"] = filters["capture_start"]
                if "capture_end" in filters:
                    rng["lte"] = filters["capture_end"]
                filter_conditions.append(
                    FieldCondition(key="capture_timestamp", range=Range(**rng))
                )

            # Geo bounding box
            if "latitude_min" in filters or "latitude_max" in filters:
                rng = {}
                if "latitude_min" in filters:
                    rng["gte"] = filters["latitude_min"]
                if "latitude_max" in filters:
                    rng["lte"] = filters["latitude_max"]
                filter_conditions.append(
                    FieldCondition(key="latitude", range=Range(**rng))
                )
            if "longitude_min" in filters or "longitude_max" in filters:
                rng = {}
                if "longitude_min" in filters:
                    rng["gte"] = filters["longitude_min"]
                if "longitude_max" in filters:
                    rng["lte"] = filters["longitude_max"]
                filter_conditions.append(
                    FieldCondition(key="longitude", range=Range(**rng))
                )

        # Geo pre-filter: restrict to specific media IDs resolved from MongoDB
        if media_ids:
            filter_conditions.append(
                FieldCondition(key="media_id", match=MatchAny(any=media_ids))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        query_vector = (
            query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else query_embedding
        )

        response = self.client.query_points(
            collection_name=col_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        formatted_results = []
        for result in response.points:
            formatted_results.append({
                "score": result.score,
                "id": result.payload.get("point_id", str(result.id)),
                **result.payload,
            })

        return formatted_results

    def search_by_text(
        self,
        text: str,
        embedding_service,
        organization_id: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        media_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using a text query."""
        query_embedding = embedding_service.embed_text(text)
        return self.search(
            query_embedding,
            organization_id=organization_id,
            top_k=top_k,
            filters=filters,
            media_ids=media_ids,
        )

    # ------------------------------------------------------------------
    # Collection info / management
    # ------------------------------------------------------------------

    def get_collection_info(self, organization_id: str) -> Dict[str, Any]:
        """Get information about an organization's collection."""
        col_name = self.ensure_collection(organization_id)
        info = self.client.get_collection(col_name)
        return {
            "name": col_name,
            "organization_id": organization_id,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    def delete_collection(self, organization_id: str):
        """Delete an organization's collection."""
        self._connect()
        col_name = collection_name_for_org(organization_id)
        self.client.delete_collection(col_name)
        self._ensured_collections.discard(col_name)
        print(f"Collection '{col_name}' deleted")

    def count(self, organization_id: str) -> int:
        """Get the number of points in an organization's collection."""
        col_name = self.ensure_collection(organization_id)
        info = self.client.get_collection(col_name)
        return info.points_count or 0

    def list_org_collections(self) -> List[Dict[str, str]]:
        """List all organization collections in Qdrant."""
        self._connect()
        result = []
        for col in self.client.get_collections().collections:
            if col.name.startswith(_COLLECTION_PREFIX):
                org_id = col.name[len(_COLLECTION_PREFIX):]
                result.append({
                    "collection_name": col.name,
                    "organization_id": org_id,
                })
        return result


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
