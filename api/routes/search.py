"""Search API routes."""

import time
import io
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image

from api.models.requests import (
    TextSearchRequest,
    ImageSearchRequest,
    MultimodalSearchRequest,
    OCRSearchRequest,
    SearchFilters,
)
from api.models.geo_models import GeoFilter
from api.models.responses import (
    SearchResponse,
    SearchResult,
    GroupedSearchResponse,
    VideoClipGroup,
    ClipTimestamp,
)
from search.search_service import get_search_service
from ingestion.s3_service import get_s3_service
from ingestion.mongodb_service import get_mongodb_service
import config

router = APIRouter(prefix="/search", tags=["search"])


def path_to_url(file_path: Optional[str]) -> Optional[str]:
    """Convert absolute or relative file path to URL for static file serving."""
    if not file_path:
        return None

    file_path_obj = Path(file_path)
    path_str = str(file_path_obj)

    # Handle absolute paths
    thumbnails_dir = str(config.THUMBNAILS_DIR)
    frames_dir = str(config.FRAMES_DIR)
    raw_dir = str(config.RAW_DATA_DIR)

    if path_str.startswith(thumbnails_dir):
        relative_path = path_str[len(thumbnails_dir):].lstrip('/')
        return f"/thumbnails/{relative_path}"

    if path_str.startswith(frames_dir):
        relative_path = path_str[len(frames_dir):].lstrip('/')
        return f"/frames/{relative_path}"

    if path_str.startswith(raw_dir):
        relative_path = path_str[len(raw_dir):].lstrip('/')
        return f"/raw/{relative_path}"

    # Handle relative paths (e.g., "data/raw/video.mp4")
    # Remove leading slash if present
    path_str_clean = path_str.lstrip('/')

    if path_str_clean.startswith("data/thumbnails/"):
        relative_path = path_str_clean.replace("data/thumbnails/", "")
        return f"/thumbnails/{relative_path}"

    if path_str_clean.startswith("data/frames/"):
        relative_path = path_str_clean.replace("data/frames/", "")
        return f"/frames/{relative_path}"

    if path_str_clean.startswith("data/raw/"):
        relative_path = path_str_clean.replace("data/raw/", "")
        return f"/raw/{relative_path}"

    # If path doesn't match any known pattern, return as-is
    return file_path


def filters_to_dict(filters: Optional[SearchFilters]) -> Optional[dict]:
    """Convert SearchFilters to dict for the search service."""
    if filters is None:
        return None

    filter_dict = {}

    # Keyword filters
    _keyword_fields = [
        "zone", "source_file", "site_id", "media_id",
        "flight_id", "flight_type", "mission_id", "mission_type", "file_type",
    ]
    for field in _keyword_fields:
        val = getattr(filters, field, None)
        if val is not None:
            filter_dict[field] = val

    # Boolean
    if filters.is_night is not None:
        filter_dict["is_night"] = filters.is_night

    # Datetime range on timestamp
    if filters.start_time:
        filter_dict["start_time"] = filters.start_time.isoformat()
    if filters.end_time:
        filter_dict["end_time"] = filters.end_time.isoformat()

    # Datetime range on capture_timestamp
    if filters.capture_start:
        filter_dict["capture_start"] = filters.capture_start.isoformat()
    if filters.capture_end:
        filter_dict["capture_end"] = filters.capture_end.isoformat()

    # Geo bounding box
    if filters.latitude_min is not None:
        filter_dict["latitude_min"] = filters.latitude_min
    if filters.latitude_max is not None:
        filter_dict["latitude_max"] = filters.latitude_max
    if filters.longitude_min is not None:
        filter_dict["longitude_min"] = filters.longitude_min
    if filters.longitude_max is not None:
        filter_dict["longitude_max"] = filters.longitude_max

    return filter_dict if filter_dict else None


def format_results(raw_results: list) -> list[SearchResult]:
    """Format raw search results into SearchResult objects."""
    s3_service = get_s3_service()
    results = []
    for r in raw_results:
        # Convert file paths to URLs for static file serving
        thumbnail_url = path_to_url(r.get("thumbnail_path"))
        frame_url = path_to_url(r.get("original_image_path") or r.get("frame_path"))
        video_url = path_to_url(r.get("video_path"))

        # Generate a presigned URL from the stored S3 storage path (if available)
        storage_path = r.get("storage_path")
        presigned_url: Optional[str] = None
        if storage_path and s3_service is not None:
            presigned_url = s3_service.try_generate_presigned_download_url(storage_path)

        results.append(SearchResult(
            id=r.get("id", r.get("point_id", "unknown")),
            score=r.get("score", 0.0),
            source_file=r.get("source_file", "unknown"),
            frame_number=r.get("frame_number", 0),
            timestamp=r.get("timestamp"),
            seconds_offset=r.get("seconds_offset"),
            original_frame_number=r.get("original_frame_number"),
            sample_rate=r.get("sample_rate"),
            source_type=r.get("source_type"),
            thumbnail_path=thumbnail_url,
            frame_path=frame_url,
            video_path=video_url,
            zone=r.get("zone"),
            is_night=r.get("is_night"),
            # Semantic clip fields
            clip_index=r.get("clip_index"),
            clip_start_timestamp=r.get("clip_start_timestamp"),
            clip_end_timestamp=r.get("clip_end_timestamp"),
            clip_start_seconds=r.get("clip_start_seconds"),
            clip_end_seconds=r.get("clip_end_seconds"),
            num_frames=r.get("num_frames"),
            # Organisation / media fields
            organization_id=r.get("organization_id"),
            site_id=r.get("site_id"),
            media_id=r.get("media_id"),
            flight_id=r.get("flight_id"),
            flight_type=r.get("flight_type"),
            mission_id=r.get("mission_id"),
            mission_type=r.get("mission_type"),
            file_type=str(r["file_type"]) if r.get("file_type") is not None else None,
            capture_timestamp=r.get("capture_timestamp"),
            latitude=r.get("latitude"),
            longitude=r.get("longitude"),
            pipeline_version=r.get("pipeline_version"),
            # S3 fields
            storage_path=storage_path,
            presigned_url=presigned_url,
            # Drone telemetry fields (from SRT)
            drone_lat=r.get("drone_lat"),
            drone_lng=r.get("drone_lng"),
            drone_alt_rel=r.get("drone_alt_rel"),
            drone_alt_abs=r.get("drone_alt_abs"),
            gimbal_yaw=r.get("gimbal_yaw"),
            gimbal_pitch=r.get("gimbal_pitch"),
            gimbal_roll=r.get("gimbal_roll"),
            focal_len=r.get("focal_len"),
            dzoom=r.get("dzoom"),
            path_lats=r.get("path_lats"),
            path_lngs=r.get("path_lngs"),
            path_yaws=r.get("path_yaws"),
            # Extra DJI flight fields (EXIF, images only)
            flight_yaw=r.get("flight_yaw"),
            flight_pitch=r.get("flight_pitch"),
            flight_roll=r.get("flight_roll"),
            flight_x_speed=r.get("flight_x_speed"),
            flight_y_speed=r.get("flight_y_speed"),
            flight_z_speed=r.get("flight_z_speed"),
            rtk_flag=r.get("rtk_flag"),
            gps_status=r.get("gps_status"),
            telemetry_path=r.get("telemetry_path"),
        ))
    return results


def resolve_geo_media_ids(
    organization_id: str,
    geo_filter: Optional[GeoFilter],
) -> Optional[List[str]]:
    """Return a list of matching mediaIds from MongoDB, or None if no geo filter.

    Returns an empty list when the filter is valid but matches nothing (callers
    should short-circuit and return empty results in that case).
    """
    if geo_filter is None:
        return None

    mongodb_service = get_mongodb_service()
    if mongodb_service is None:
        return None  # MongoDB disabled; silently ignore geo_filter

    geo_dict = geo_filter.model_dump()
    return mongodb_service.find_media_ids_in_area(organization_id, geo_dict)


@router.post("/text", response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search using a text query.

    Requires ``organization_id`` to target the correct Qdrant collection.
    Optionally accepts a ``geo_filter`` to restrict results to flights that
    flew within a given radius or polygon.
    """
    start_time = time.time()

    try:
        search_service = get_search_service()
        filters = filters_to_dict(request.filters)

        # Geo pre-filter: resolve matching mediaIds from MongoDB
        media_ids = resolve_geo_media_ids(request.organization_id, request.geo_filter)
        if media_ids is not None and len(media_ids) == 0:
            # Geo filter matched nothing → return empty results immediately
            return SearchResponse(
                query_type="text",
                query=request.query,
                total_results=0,
                results=[],
                search_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        raw_results = search_service.search_text(
            query=request.query,
            organization_id=request.organization_id,
            top_k=request.top_k,
            filters=filters,
            use_reranker=request.use_reranker,
            media_ids=media_ids,
        )

        results = format_results(raw_results)
        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query_type="text",
            query=request.query,
            total_results=len(results),
            results=results,
            search_time_ms=round(search_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def group_results_by_video(results: list[SearchResult]) -> list[VideoClipGroup]:
    """Group flat search results into per-video clip groups."""
    from collections import OrderedDict

    groups: OrderedDict[str, dict] = OrderedDict()
    for r in results:
        key = r.source_file
        if key not in groups:
            groups[key] = {
                "source_file": r.source_file,
                "video_path": r.video_path,
                "presigned_url": r.presigned_url,
                "storage_path": r.storage_path,
                "source_type": r.source_type,
                "organization_id": r.organization_id,
                "site_id": r.site_id,
                "media_id": r.media_id,
                "flight_id": r.flight_id,
                "capture_timestamp": r.capture_timestamp,
                "best_score": r.score,
                "clips": [],
            }
        group = groups[key]
        if r.score > group["best_score"]:
            group["best_score"] = r.score

        group["clips"].append(ClipTimestamp(
            id=r.id,
            score=r.score,
            frame_number=r.frame_number,
            timestamp=r.timestamp,
            seconds_offset=r.seconds_offset,
            clip_index=r.clip_index,
            clip_start_timestamp=r.clip_start_timestamp,
            clip_end_timestamp=r.clip_end_timestamp,
            clip_start_seconds=r.clip_start_seconds,
            clip_end_seconds=r.clip_end_seconds,
            thumbnail_path=r.thumbnail_path,
            frame_path=r.frame_path,
        ))

    video_groups = []
    for g in groups.values():
        clips = sorted(
            g.pop("clips"),
            key=lambda c: (c.clip_start_seconds or c.seconds_offset or 0),
        )
        video_groups.append(VideoClipGroup(
            **g,
            total_clips=len(clips),
            clips=clips,
        ))

    video_groups.sort(key=lambda v: v.best_score, reverse=True)
    return video_groups


@router.post("/text/grouped", response_model=GroupedSearchResponse)
async def search_text_grouped(request: TextSearchRequest):
    """
    Search using a text query and return results grouped by video.

    Each video entry contains an array of matching clip timestamps, sorted
    chronologically. Videos are sorted by their best matching score.
    """
    start_time = time.time()

    try:
        search_service = get_search_service()
        filters = filters_to_dict(request.filters)

        media_ids = resolve_geo_media_ids(request.organization_id, request.geo_filter)
        if media_ids is not None and len(media_ids) == 0:
            return GroupedSearchResponse(
                query_type="text",
                query=request.query,
                total_videos=0,
                total_clips=0,
                videos=[],
                search_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        raw_results = search_service.search_text(
            query=request.query,
            organization_id=request.organization_id,
            top_k=request.top_k,
            filters=filters,
            use_reranker=request.use_reranker,
            media_ids=media_ids,
        )

        results = format_results(raw_results)
        video_groups = group_results_by_video(results)
        total_clips = sum(v.total_clips for v in video_groups)
        search_time_ms = (time.time() - start_time) * 1000

        return GroupedSearchResponse(
            query_type="text",
            query=request.query,
            total_videos=len(video_groups),
            total_clips=total_clips,
            videos=video_groups,
            search_time_ms=round(search_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image", response_model=SearchResponse)
async def search_image(
    image: UploadFile = File(..., description="Reference image to search for"),
    organization_id: str = Form(..., description="Organization identifier"),
    top_k: int = Form(20, ge=1, le=100),
    zone: Optional[str] = Form(None),
    site_id: Optional[str] = Form(None),
    flight_id: Optional[str] = Form(None),
    mission_id: Optional[str] = Form(None),
    mission_type: Optional[str] = Form(None),
    is_night: Optional[bool] = Form(None),
    use_reranker: bool = Form(False),
):
    """
    Search using a reference image.

    Requires ``organization_id`` to target the correct Qdrant collection.
    """
    start_time = time.time()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        filters = {}
        if zone:
            filters["zone"] = zone
        if site_id:
            filters["site_id"] = site_id
        if flight_id:
            filters["flight_id"] = flight_id
        if mission_id:
            filters["mission_id"] = mission_id
        if mission_type:
            filters["mission_type"] = mission_type
        if is_night is not None:
            filters["is_night"] = is_night

        search_service = get_search_service()

        raw_results = search_service.search_image(
            image=pil_image,
            organization_id=organization_id,
            top_k=top_k,
            filters=filters if filters else None,
            use_reranker=use_reranker,
        )

        results = format_results(raw_results)
        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query_type="image",
            query=f"Image: {image.filename}",
            total_results=len(results),
            results=results,
            search_time_ms=round(search_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal", response_model=SearchResponse)
async def search_multimodal(
    image: UploadFile = File(..., description="Reference image"),
    organization_id: str = Form(..., description="Organization identifier"),
    query: str = Form(..., description="Text query to combine with image"),
    top_k: int = Form(20, ge=1, le=100),
    zone: Optional[str] = Form(None),
    site_id: Optional[str] = Form(None),
    flight_id: Optional[str] = Form(None),
    mission_id: Optional[str] = Form(None),
    is_night: Optional[bool] = Form(None),
    start_time_filter: Optional[str] = Form(None, alias="start_time"),
    end_time_filter: Optional[str] = Form(None, alias="end_time"),
    use_reranker: bool = Form(False),
):
    """
    Search using combined text and image query.

    Requires ``organization_id`` to target the correct Qdrant collection.
    """
    search_start = time.time()

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        filters = {}
        if zone:
            filters["zone"] = zone
        if site_id:
            filters["site_id"] = site_id
        if flight_id:
            filters["flight_id"] = flight_id
        if mission_id:
            filters["mission_id"] = mission_id
        if is_night is not None:
            filters["is_night"] = is_night
        if start_time_filter:
            filters["start_time"] = start_time_filter
        if end_time_filter:
            filters["end_time"] = end_time_filter

        search_service = get_search_service()

        raw_results = search_service.search_multimodal(
            text=query,
            image=pil_image,
            organization_id=organization_id,
            top_k=top_k,
            filters=filters if filters else None,
            use_reranker=use_reranker,
        )

        results = format_results(raw_results)
        search_time_ms = (time.time() - search_start) * 1000

        return SearchResponse(
            query_type="multimodal",
            query=f"{query} + Image: {image.filename}",
            total_results=len(results),
            results=results,
            search_time_ms=round(search_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=SearchResponse)
async def search_ocr(request: OCRSearchRequest):
    """
    Search for text visible in footage (OCR).

    Requires ``organization_id`` to target the correct Qdrant collection.
    """
    start_time = time.time()

    try:
        search_service = get_search_service()
        filters = filters_to_dict(request.filters)

        raw_results = search_service.search_ocr(
            text_query=request.text,
            organization_id=request.organization_id,
            top_k=request.top_k,
            filters=filters,
            use_reranker=request.use_reranker,
        )

        results = format_results(raw_results)
        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query_type="ocr",
            query=f"OCR: {request.text}",
            total_results=len(results),
            results=results,
            search_time_ms=round(search_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
