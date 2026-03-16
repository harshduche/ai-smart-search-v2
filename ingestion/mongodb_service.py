"""MongoDB service for geospatial flight telemetry storage and queries,
and ingestion state tracking.

Follows the same singleton / graceful-degradation pattern as s3_service.py.
Returns None from get_mongodb_service() when MONGODB_URI is not configured,
so all callers can safely skip geo operations without additional checks.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import pymongo
    import pymongo.errors
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed – MongoDB geospatial layer unavailable")


class MongoDBService:
    """Thin wrapper around MongoDB for flight telemetry and ingestion state."""

    _TELEMETRY_COLLECTION = "flight_telemetry"
    _INGESTION_COLLECTION = "ingestion_state"
    _PROCESSING_REQUEST_COLLECTION = "processing_requests"

    def __init__(self, uri: str, database: str):
        if not _PYMONGO_AVAILABLE:
            raise RuntimeError("pymongo is not installed")

        self._client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[database]
        self._col = self._db[self._TELEMETRY_COLLECTION]
        self._ingestion_col = self._db[self._INGESTION_COLLECTION]
        self._processing_request_col = self._db[self._PROCESSING_REQUEST_COLLECTION]
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create 2dsphere and scalar indexes (idempotent)."""
        try:
            self._col.create_index([("flightPath",   pymongo.GEOSPHERE)])
            self._col.create_index([("coverageArea", pymongo.GEOSPHERE)])
            self._col.create_index("organizationId")
            self._col.create_index("flightId")
            self._col.create_index("siteId")
            logger.debug("MongoDB indexes ensured on %s", self._TELEMETRY_COLLECTION)

            self._ingestion_col.create_index("mediaId")
            self._ingestion_col.create_index("organizationId")
            self._ingestion_col.create_index("status")
            logger.debug("MongoDB indexes ensured on %s", self._INGESTION_COLLECTION)

            self._processing_request_col.create_index("mediaIds")
            self._processing_request_col.create_index("organizationId")
            self._processing_request_col.create_index("status")
            logger.debug("MongoDB indexes ensured on %s", self._PROCESSING_REQUEST_COLLECTION)
        except Exception as exc:
            logger.warning("MongoDB index creation failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_flight_telemetry(self, doc: Dict[str, Any]) -> bool:
        """Upsert a flight telemetry document by _id (= mediaId).

        Returns True on success, False on error.
        """
        try:
            media_id = doc.get("_id")
            if not media_id:
                logger.warning("upsert_flight_telemetry: doc has no _id, skipping")
                return False

            self._col.replace_one(
                {"_id": media_id},
                doc,
                upsert=True,
            )
            return True
        except Exception as exc:
            logger.warning("MongoDB upsert failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Reads / geo queries
    # ------------------------------------------------------------------

    @staticmethod
    def _project(exclude_frames: bool) -> Dict:
        return {"frames": 0} if exclude_frames else {}

    def find_coverage(
        self,
        org_id: str,
        point: Dict,
        exclude_frames: bool = True,
    ) -> List[Dict]:
        """Return flights whose *coverageArea* contains *point*.

        point – GeoJSON Point dict: ``{"type": "Point", "coordinates": [lng, lat]}``
        """
        try:
            cursor = self._col.find(
                {
                    "organizationId": org_id,
                    "coverageArea": {
                        "$geoIntersects": {"$geometry": point},
                    },
                },
                projection=self._project(exclude_frames),
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("find_coverage query failed: %s", exc)
            return []

    def find_flights_in_polygon(
        self,
        org_id: str,
        polygon: Dict,
        exclude_frames: bool = True,
    ) -> List[Dict]:
        """Return flights whose *flightPath* intersects *polygon*.

        polygon – GeoJSON Polygon dict.
        """
        try:
            cursor = self._col.find(
                {
                    "organizationId": org_id,
                    "flightPath": {
                        "$geoIntersects": {"$geometry": polygon},
                    },
                },
                projection=self._project(exclude_frames),
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("find_flights_in_polygon query failed: %s", exc)
            return []

    def find_nearby_flights(
        self,
        org_id: str,
        point: Dict,
        max_distance_m: float,
        exclude_frames: bool = True,
    ) -> List[Dict]:
        """Return flights whose *flightPath* is within *max_distance_m* metres.

        Results are sorted by distance (nearest first) via $nearSphere.
        """
        try:
            cursor = self._col.find(
                {
                    "organizationId": org_id,
                    "flightPath": {
                        "$nearSphere": {
                            "$geometry": point,
                            "$maxDistance": max_distance_m,
                        },
                    },
                },
                projection=self._project(exclude_frames),
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("find_nearby_flights query failed: %s", exc)
            return []

    def get_telemetry(self, media_id: str) -> Optional[Dict]:
        """Return the full flight document (including frames) for *media_id*."""
        try:
            return self._col.find_one({"_id": media_id})
        except Exception as exc:
            logger.warning("get_telemetry query failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ingestion state tracking
    # ------------------------------------------------------------------

    def _ingestion_doc_id(self, media_id: str, pipeline_version: str) -> str:
        return f"{media_id}::{pipeline_version}"

    def update_ingestion_status(
        self,
        media_id: str,
        status: str,
        pipeline_version: str = "v1",
        error: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Update the ``ingestion_state`` document for *media_id*.

        Sets ``status``, ``updatedAt``, and (when provided) ``requestId``.
        Also sets ``lastError`` when *status* is ``"failed"``.
        Returns True on success.
        """
        try:
            doc_id = self._ingestion_doc_id(media_id, pipeline_version)
            now = datetime.now(timezone.utc)

            update: Dict[str, Any] = {
                "$set": {
                    "status": status,
                    "updatedAt": now,
                },
            }
            if request_id is not None:
                update["$set"]["requestId"] = request_id
            if status == "failed" and error is not None:
                update["$set"]["lastError"] = error
            elif status == "completed":
                update["$set"]["lastError"] = None

            result = self._ingestion_col.update_one(
                {"_id": doc_id},
                update,
            )
            if result.matched_count == 0:
                logger.warning(
                    "ingestion_state document not found for _id=%s", doc_id,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("update_ingestion_status failed: %s", exc)
            return False

    def increment_processing_request(
        self,
        transition: str,
        processing_request_id: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> bool:
        """Atomically update counters on a ``processing_requests`` document.

        *transition* must be one of:

        * ``"queued_to_processing"``  – job picked up by a worker
        * ``"processing_to_done"``    – job completed successfully
        * ``"processing_to_failed"``  – job failed permanently
        * ``"queued_to_skipped"``     – job skipped (already processed, etc.)

        The document is located by *processing_request_id* when provided;
        otherwise by ``{ mediaIds: media_id }`` as a fallback.

        After each terminal transition (done / failed / skipped) the method
        checks whether ``done + failed + skipped >= totalMedia`` and, if so,
        sets ``status`` to ``"completed"``.

        Returns True on success, False on any error or if the document was
        not found.
        """
        _TRANSITIONS: Dict[str, Dict[str, int]] = {
            "queued_to_processing": {"queued": -1, "processing": 1},
            "processing_to_done":   {"processing": -1, "done": 1},
            "processing_to_failed": {"processing": -1, "failed": 1},
            "queued_to_skipped":    {"queued": -1, "skipped": 1},
        }

        if transition not in _TRANSITIONS:
            logger.warning("increment_processing_request: unknown transition %r", transition)
            return False

        if not processing_request_id and not media_id:
            logger.warning("increment_processing_request: need processing_request_id or media_id")
            return False

        try:
            increments = _TRANSITIONS[transition]
            is_terminal = transition in (
                "processing_to_done", "processing_to_failed", "queued_to_skipped"
            )

            query: Dict[str, Any]
            if processing_request_id:
                query = {"_id": processing_request_id}
            else:
                query = {"mediaIds": media_id}

            # Step 1: atomically apply the counter increment.
            result = self._processing_request_col.find_one_and_update(
                query,
                {
                    "$inc": increments,
                    "$set": {"updatedAt": datetime.now(timezone.utc)},
                },
                return_document=True,
            )

            if result is None:
                logger.warning(
                    "processing_requests document not found "
                    "(request_id=%s, media_id=%s)",
                    processing_request_id, media_id,
                )
                return False

            # Step 2 (terminal transitions only): atomically flip status to
            # "completed" when done + failed + skipped >= totalMedia.
            # Uses an aggregation-pipeline update so the condition and the
            # write are evaluated together by the server — no TOCTOU race
            # between concurrent workers.
            if is_terminal:
                self._processing_request_col.update_one(
                    {
                        "_id": result["_id"],
                        "status": {"$ne": "completed"},
                        "$expr": {
                            "$and": [
                                {"$gt": ["$totalMedia", 0]},
                                {
                                    "$gte": [
                                        {"$add": ["$done", "$failed", "$skipped"]},
                                        "$totalMedia",
                                    ]
                                },
                            ]
                        },
                    },
                    [{"$set": {"status": "completed", "updatedAt": "$$NOW"}}],
                )

            return True

        except Exception as exc:
            logger.warning("increment_processing_request failed: %s", exc)
            return False

    def find_media_ids_in_area(
        self,
        org_id: str,
        geo_filter: Dict,
    ) -> List[str]:
        """Return mediaId strings matching a geographic filter.

        geo_filter shapes::

            {"type": "radius",  "center": [lng, lat], "radiusMeters": 500}
            {"type": "polygon", "polygon": {"type": "Polygon", "coordinates": [...]}}

        Returns an empty list on error or no match.
        """
        try:
            filter_type = geo_filter.get("type")

            if filter_type == "radius":
                center = geo_filter.get("center")          # [lng, lat]
                radius = geo_filter.get("radiusMeters", 200)
                if not center or len(center) < 2:
                    return []
                geo_query = {
                    "$nearSphere": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": center,
                        },
                        "$maxDistance": radius,
                    }
                }

            elif filter_type == "polygon":
                poly = geo_filter.get("polygon")
                if not poly:
                    return []
                geo_query = {
                    "$geoIntersects": {"$geometry": poly}
                }

            else:
                logger.warning("Unknown geo_filter type: %s", filter_type)
                return []

            cursor = self._col.find(
                {
                    "organizationId": org_id,
                    "flightPath": geo_query,
                },
                projection={"_id": 1},
            )
            return [str(doc["_id"]) for doc in cursor]

        except Exception as exc:
            logger.warning("find_media_ids_in_area query failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_mongodb_service: Optional[MongoDBService] = None


def get_mongodb_service() -> Optional[MongoDBService]:
    """Return the singleton MongoDBService, or None if not configured.

    MongoDB is considered unconfigured when MONGODB_URI is empty.
    """
    global _mongodb_service
    if _mongodb_service is not None:
        return _mongodb_service

    import config as _config

    uri = getattr(_config, "MONGODB_URI", "") or ""
    if not uri:
        logger.debug("MONGODB_URI not configured; MongoDBService disabled")
        return None

    database = getattr(_config, "MONGODB_DATABASE", "video_rag")

    try:
        _mongodb_service = MongoDBService(uri=uri, database=database)
        # Ping to verify connectivity
        _mongodb_service._client.admin.command("ping")
        logger.info("MongoDBService initialised (uri=%s, db=%s)", uri, database)
        return _mongodb_service
    except Exception as exc:
        logger.warning(
            "MongoDB connection failed (geospatial layer disabled): %s", exc
        )
        return None
