"""Geospatial API routes.

All endpoints require MongoDB to be configured (MONGODB_URI env var).
Returns HTTP 503 if MongoDB is unavailable.
"""

from fastapi import APIRouter, HTTPException

from api.models.geo_models import (
    CoverageRequest,
    FlightIntersectionRequest,
    FlightRecord,
    GeoSearchResponse,
    NearbyRequest,
    TelemetryResponse,
    FrameTelemetry,
)
from ingestion.mongodb_service import get_mongodb_service

router = APIRouter(prefix="/geo", tags=["geospatial"])


def _require_mongodb():
    """Return MongoDBService or raise 503."""
    svc = get_mongodb_service()
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Geospatial layer unavailable: MONGODB_URI is not configured",
        )
    return svc


def _doc_to_flight_record(doc: dict) -> FlightRecord:
    return FlightRecord(
        mediaId=str(doc.get("_id", "")) or None,
        sourceFile=doc.get("sourceFile"),
        flightPath=doc.get("flightPath"),
        coverageArea=doc.get("coverageArea"),
        duration=doc.get("duration"),
        siteId=doc.get("siteId"),
        flightId=doc.get("flightId"),
        captureTimestamp=doc.get("captureTimestamp"),
        storagePath=doc.get("storagePath"),
    )


@router.post("/coverage", response_model=GeoSearchResponse)
async def coverage_lookup(request: CoverageRequest):
    """Return flights whose *coverageArea* polygon contains the given point."""
    svc = _require_mongodb()
    point = request.point.model_dump()
    docs = svc.find_coverage(org_id=request.orgId, point=point)
    results = [_doc_to_flight_record(d) for d in docs]
    return GeoSearchResponse(count=len(results), results=results)


@router.post("/flights", response_model=GeoSearchResponse)
async def flight_intersection(request: FlightIntersectionRequest):
    """Return flights whose *flightPath* intersects the given polygon."""
    svc = _require_mongodb()
    polygon = request.polygon.model_dump()
    docs = svc.find_flights_in_polygon(org_id=request.orgId, polygon=polygon)
    results = [_doc_to_flight_record(d) for d in docs]
    return GeoSearchResponse(count=len(results), results=results)


@router.post("/nearby", response_model=GeoSearchResponse)
async def nearby_search(request: NearbyRequest):
    """Return flights within *maxDistance* metres of the given point (nearest first)."""
    svc = _require_mongodb()
    point = request.point.model_dump()
    docs = svc.find_nearby_flights(
        org_id=request.orgId,
        point=point,
        max_distance_m=request.maxDistance,
    )
    results = [_doc_to_flight_record(d) for d in docs]
    return GeoSearchResponse(count=len(results), results=results)


@router.get("/telemetry/{media_id}", response_model=TelemetryResponse)
async def get_telemetry(media_id: str):
    """Return full telemetry (including per-second frames) for a media asset."""
    svc = _require_mongodb()
    doc = svc.get_telemetry(media_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No telemetry found for media_id={media_id!r}")

    raw_frames = doc.get("frames") or []
    frames = [
        FrameTelemetry(
            ts=f.get("ts", 0.0),
            lat=f["lat"],
            lng=f["lng"],
            altRel=f.get("altRel"),
            altAbs=f.get("altAbs"),
            yaw=f.get("yaw"),
            pitch=f.get("pitch"),
            roll=f.get("roll"),
            focalLen=f.get("focalLen"),
            dzoom=f.get("dzoom"),
        )
        for f in raw_frames
        if "lat" in f and "lng" in f
    ]

    return TelemetryResponse(
        mediaId=str(doc.get("_id", media_id)),
        duration=doc.get("duration"),
        flightPath=doc.get("flightPath"),
        coverageArea=doc.get("coverageArea"),
        frames=frames,
    )
