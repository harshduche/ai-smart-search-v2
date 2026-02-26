"""Ingestion API routes."""

from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel, HttpUrl

from api.models.requests import IngestRequest, DroneFootageIngestRequest, RTSPIngestRequest
from api.models.responses import IngestResponse
from ingestion.ingest_pipeline import create_ingest_pipeline
import config

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Track background ingestion tasks (reserved for future use)
_active_ingestions = {}


class IngestURLRequest(BaseModel):
    """Request model for URL-based video ingestion."""
    organization_id: str
    video_url: str
    zone: Optional[str] = None
    clip_duration: float = 4.0
    max_frames_per_clip: int = 32
    batch_size: int = 50
    save_full_frames: Optional[bool] = None
    cleanup_after: bool = True

    # Optional metadata
    site_id: Optional[str] = None
    site_name: Optional[str] = None
    flight_id: Optional[str] = None


@router.post("/", response_model=IngestResponse)
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest video/image data into the vector store.

    This endpoint starts ingestion and returns immediately.
    For large datasets, the ingestion continues in the background.
    """
    input_path = Path(request.path)

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Path not found: {request.path}"
        )

    pipeline = create_ingest_pipeline()

    try:
        if input_path.is_file():
            # Single file
            ext = input_path.suffix.lower()
            video_exts = {".mp4", ".avi", ".mov", ".mkv"}
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

            if ext in video_exts:
                count = pipeline.ingest_video(input_path, organization_id=request.organization_id, zone=request.zone)
                stats = pipeline.stats
            elif ext in image_exts:
                count = pipeline.ingest_images([input_path], organization_id=request.organization_id, zone=request.zone)
                stats = pipeline.stats
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {ext}"
                )

        elif input_path.is_dir():
            stats = pipeline.ingest_directory(
                input_path,
                organization_id=request.organization_id,
                process_videos=request.process_videos,
                process_images=request.process_images,
            )

        return IngestResponse(
            status="completed",
            videos_processed=stats.get("videos_processed", 0),
            images_processed=stats.get("images_processed", 0),
            frames_extracted=stats.get("frames_extracted", 0),
            embeddings_generated=stats.get("embeddings_generated", 0),
            vectors_stored=stats.get("vectors_stored", 0),
            duration_seconds=stats.get("duration_seconds"),
            errors=stats.get("errors", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest_video(
    file: UploadFile = File(..., description="Video file to ingest"),
    organization_id: str = Form(..., description="Organization identifier"),
    zone: str | None = Form(None, description="Zone/location identifier"),
    semantic_video: bool = Form(
        False,
        description="If true, ingest as semantic clips using Qwen3-VL video embeddings",
    ),
    clip_duration: float = Form(
        4.0,
        description="Target clip duration in seconds for semantic ingestion",
    ),
    max_frames_per_clip: int = Form(
        32,
        description="Maximum frames per semantic clip for memory control",
    ),
    save_full_frames: bool | None = Form(
        None,
        description="Save full-resolution frames for high-quality popup view (uses config default if not specified)",
    ),
):
    """
    Upload a video file and ingest it into the vector store.

    This endpoint is primarily used by the web dashboard to allow
    browser-based uploads. For filesystem paths, prefer the standard
    `/ingest` JSON endpoint.
    """
    # Basic content-type guard
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type for upload: {file.content_type!r}",
        )

    # Persist uploaded file to RAW_DATA_DIR with a timestamped name
    raw_dir = config.RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    target_path = raw_dir / f"{timestamp}_{safe_name}"

    try:
        with target_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {e}",
        )

    pipeline = create_ingest_pipeline()

    try:
        if semantic_video:
            clips_count = pipeline.ingest_video_semantic_clips(
                target_path,
                organization_id=organization_id,
                zone=zone,
                clip_duration=clip_duration,
                max_frames_per_clip=max_frames_per_clip,
                save_full_frames=save_full_frames,
            )
            stats = pipeline.stats
        else:
            _ = pipeline.ingest_video(target_path, organization_id=organization_id, zone=zone, save_full_frames=save_full_frames)
            stats = pipeline.stats

        return IngestResponse(
            status="completed",
            videos_processed=stats.get("videos_processed", 0),
            images_processed=stats.get("images_processed", 0),
            frames_extracted=stats.get("frames_extracted", 0),
            embeddings_generated=stats.get("embeddings_generated", 0),
            vectors_stored=stats.get("vectors_stored", 0),
            duration_seconds=stats.get("duration_seconds"),
            errors=stats.get("errors", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/url", response_model=IngestResponse)
async def ingest_from_url(request: IngestURLRequest):
    """
    Download and ingest a video from a URL using semantic clips.

    This endpoint is designed for ingesting videos from remote sources like:
    - S3 pre-signed URLs
    - Public HTTP/HTTPS video URLs
    - Any downloadable video link

    The video is downloaded to a temporary location, ingested using semantic
    clips for optimal temporal context, and optionally cleaned up after processing.

    **Use Cases:**
    - Ingesting videos stored in cloud storage (S3, GCS, Azure Blob)
    - Processing videos from CDNs or public URLs
    - Batch ingestion from a list of remote video URLs

    **Performance Notes:**
    - Download time depends on video size and network speed
    - Semantic clips ingestion is optimized for long videos
    - Set `cleanup_after=false` to retain downloaded file for debugging

    **Example:**
    ```json
    {
        "video_url": "https://s3.amazonaws.com/bucket/video.mp4?signature=...",
        "zone": "camera_1",
        "clip_duration": 4.0,
        "max_frames_per_clip": 32,
        "cleanup_after": true,
        "site_id": "construction_site_a",
        "drone_id": "dji_mavic_001"
    }
    ```
    """
    pipeline = create_ingest_pipeline()

    try:
        result = pipeline.ingest_video_from_url(
            video_url=request.video_url,
            organization_id=request.organization_id,
            zone=request.zone,
            clip_duration=request.clip_duration,
            max_frames_per_clip=request.max_frames_per_clip,
            batch_size=request.batch_size,
            save_full_frames=request.save_full_frames,
            cleanup_after=request.cleanup_after,
        )

        stats = result["stats"]

        return IngestResponse(
            status="completed",
            videos_processed=stats.get("videos_processed", 0),
            images_processed=stats.get("images_processed", 0),
            frames_extracted=stats.get("frames_extracted", 0),
            embeddings_generated=stats.get("embeddings_generated", 0),
            vectors_stored=stats.get("vectors_stored", 0),
            duration_seconds=stats.get("duration_seconds"),
            errors=stats.get("errors", []),
            video_url=request.video_url,
            site_id=request.site_id,
            flight_id=request.flight_id,
            file_size_mb=result.get("file_size_mb"),
        )

    except ValueError as e:
        # Handle validation errors (invalid URL, download failures)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other processing errors
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drone-footage", response_model=IngestResponse)
async def ingest_drone_footage(request: DroneFootageIngestRequest):
    """
    Ingest drone footage with comprehensive multi-site metadata.

    This endpoint is specifically designed for organizations managing drone
    footage across multiple sites. It accepts rich metadata including:
    - Organization and site information
    - Drone details (ID, model, serial)
    - Flight information (ID, date, purpose)
    - Zone and location data
    - Weather conditions and operator details

    All metadata is indexed for fast filtering and cross-site analysis.

    **Use Cases:**
    - Multi-site security surveillance
    - Construction progress monitoring
    - Infrastructure inspection
    - Site compliance verification

    **Architecture:**
    - Single Qdrant collection with site filtering
    - Fast site-specific or cross-site searches
    - Semantic clips for temporal understanding

    **Example:**
    ```json
    {
        "video_url": "https://s3.amazonaws.com/bucket/site_a_morning.mp4?signature=...",
        "organization_id": "flytbase_security",
        "site_id": "construction_site_a",
        "site_name": "Construction Site A - Mumbai",
        "site_location": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai"},
        "drone_id": "dji_mavic_001",
        "drone_model": "DJI Mavic 3 Enterprise",
        "flight_id": "flight_20240207_001",
        "flight_purpose": "perimeter_inspection",
        "zone": "north_perimeter",
        "zone_type": "fence",
        "weather_condition": "clear",
        "operator": "john_doe",
        "tags": ["perimeter", "security", "routine"]
    }
    ```

    **Search Examples:**
    ```python
    # Site-specific search
    filters = {"site_id": "construction_site_a", "zone_type": "fence"}

    # Cross-site search
    filters = {"organization_id": "flytbase_security", "flight_purpose": "perimeter_inspection"}

    # Drone-specific search
    filters = {"drone_id": "dji_mavic_001", "is_night": True}
    ```
    """
    pipeline = create_ingest_pipeline()

    try:
        # Build zone parameter (combine site_id and zone for uniqueness)
        zone_param = f"{request.site_id}_{request.zone}" if request.zone else request.site_id

        # Build site_location dict
        site_location_dict = None
        if request.site_location:
            site_location_dict = {
                "lat": request.site_location.lat,
                "lon": request.site_location.lon,
                "city": request.site_location.city,
                "state": request.site_location.state,
                "country": request.site_location.country,
            }

        # Build comprehensive metadata dict to add to each clip
        metadata = {
            # Organization & Site
            "organization_id": request.organization_id,
            "site_id": request.site_id,
            "site_name": request.site_name,

            # Drone
            "drone_id": request.drone_id,
            "drone_model": request.drone_model,

            # Flight
            "flight_id": request.flight_id,
            "flight_purpose": request.flight_purpose,

            # Context
            "weather_condition": request.weather_condition,
            "video_url": request.video_url,
        }

        # Add optional fields if provided
        if request.organization_name:
            metadata["organization_name"] = request.organization_name
        if site_location_dict:
            metadata["site_location"] = site_location_dict
        if request.drone_serial:
            metadata["drone_serial"] = request.drone_serial
        if request.flight_date:
            metadata["flight_date"] = request.flight_date
        if request.flight_time:
            metadata["flight_time"] = request.flight_time
        if request.zone_type:
            metadata["zone_type"] = request.zone_type
        if request.operator:
            metadata["operator"] = request.operator
        if request.tags:
            metadata["tags"] = request.tags
        if request.notes:
            metadata["notes"] = request.notes

        # Ingest with URL method and metadata
        result = pipeline.ingest_video_from_url(
            video_url=request.video_url,
            organization_id=request.organization_id,
            zone=zone_param,
            clip_duration=request.clip_duration,
            max_frames_per_clip=request.max_frames_per_clip,
            batch_size=request.batch_size,
            save_full_frames=request.save_full_frames,
            cleanup_after=request.cleanup_after,
            metadata=metadata,
        )

        stats = result["stats"]

        return IngestResponse(
            status="completed",
            videos_processed=stats.get("videos_processed", 0),
            images_processed=stats.get("images_processed", 0),
            frames_extracted=stats.get("frames_extracted", 0),
            embeddings_generated=stats.get("embeddings_generated", 0),
            vectors_stored=stats.get("vectors_stored", 0),
            duration_seconds=stats.get("duration_seconds"),
            errors=stats.get("errors", []),
            video_url=request.video_url,
            site_id=request.site_id,
            site_name=request.site_name,
            flight_id=request.flight_id,
            file_size_mb=result.get("file_size_mb"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rtsp", response_model=IngestResponse)
async def ingest_rtsp_stream(
    request: RTSPIngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest live video from an RTSP stream.

    This endpoint connects to RTSP streams from IP cameras, surveillance systems,
    or other live video sources. It captures frames for a specified duration or
    frame count, processes them into semantic clips, and stores embeddings in
    the vector database.

    **Key Features:**
    - Supports standard RTSP URLs (rtsp://host:port/path)
    - Duration-based or frame-count-based capture
    - Automatic reconnection on stream failures
    - Real-time frame extraction and processing
    - Semantic clips for temporal context

    **Use Cases:**
    - IP camera surveillance ingestion
    - Live monitoring integration
    - Scheduled periodic captures from static cameras
    - Multi-camera site monitoring

    **Stream Termination:**
    - Specify `duration_seconds` to capture for a fixed time (e.g., 300 for 5 minutes)
    - Specify `max_frames` to capture a fixed number of frames (e.g., 1000 frames)
    - At least one termination condition must be provided

    **Processing Mode:**
    - `use_semantic_clips=true` (recommended): Groups frames into 4-second clips
      for better temporal understanding and reduced storage
    - `use_semantic_clips=false`: Processes individual frames (higher granularity,
      more storage)

    **Reconnection:**
    - Automatically reconnects if stream drops (configurable)
    - Configurable retry attempts and delays
    - Useful for unreliable network connections

    **Example - Fixed Duration:**
    ```json
    {
        "rtsp_url": "rtsp://192.168.1.100:554/stream1",
        "zone": "main_entrance",
        "duration_seconds": 300,
        "use_semantic_clips": true,
        "reconnect_on_failure": true,
        "site_id": "office_building_a",
        "camera_id": "cam_001"
    }
    ```

    **Example - Fixed Frame Count:**
    ```json
    {
        "rtsp_url": "rtsp://admin:password@camera.local/live",
        "zone": "parking_lot",
        "max_frames": 1000,
        "clip_duration": 4.0,
        "camera_id": "parking_cam_02",
        "camera_name": "Parking Lot Camera 2"
    }
    ```

    **RTSP URL Format:**
    - Basic: `rtsp://host:port/path`
    - With auth: `rtsp://username:password@host:port/path`
    - Default port: 554

    **Performance Notes:**
    - Capture happens synchronously (blocks until complete or error)
    - For long captures (>10 minutes), consider breaking into smaller sessions
    - Network bandwidth affects frame capture rate
    - Processing speed depends on GPU availability for embeddings

    **Error Handling:**
    - Connection failures after max retries return 400
    - Invalid RTSP URLs return 400
    - Processing errors return 500
    """
    pipeline = create_ingest_pipeline()

    try:
        # Build additional metadata
        metadata = {
            "organization_id": request.organization_id,
        }
        if request.site_id:
            metadata["site_id"] = request.site_id
        if request.site_name:
            metadata["site_name"] = request.site_name
        if request.camera_id:
            metadata["camera_id"] = request.camera_id
        if request.camera_name:
            metadata["camera_name"] = request.camera_name
        if request.camera_location:
            metadata["camera_location"] = request.camera_location

        # Ingest RTSP stream
        result = pipeline.ingest_rtsp_stream(
            rtsp_url=request.rtsp_url,
            organization_id=request.organization_id,
            zone=request.zone,
            duration_seconds=request.duration_seconds,
            max_frames=request.max_frames,
            use_semantic_clips=request.use_semantic_clips,
            clip_duration=request.clip_duration,
            max_frames_per_clip=request.max_frames_per_clip,
            batch_size=request.batch_size,
            save_full_frames=request.save_full_frames,
            reconnect_on_failure=request.reconnect_on_failure,
            max_reconnect_attempts=request.max_reconnect_attempts,
            reconnect_delay_seconds=request.reconnect_delay_seconds,
            connection_timeout_seconds=request.connection_timeout_seconds,
            additional_metadata=metadata if metadata else None,
        )

        stats = result["stats"]

        return IngestResponse(
            status="completed",
            videos_processed=stats.get("videos_processed", 0),
            images_processed=stats.get("images_processed", 0),
            frames_extracted=stats.get("frames_extracted", 0),
            embeddings_generated=stats.get("embeddings_generated", 0),
            vectors_stored=stats.get("vectors_stored", 0),
            duration_seconds=result.get("capture_duration_seconds"),
            errors=stats.get("errors", []),
            video_url=request.rtsp_url,  # Use rtsp_url as video_url
            site_id=request.site_id,
        )

    except ValueError as e:
        # Connection failures, invalid URLs
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Configuration errors (missing termination condition)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Other processing errors
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection/{organization_id}")
async def delete_collection(organization_id: str):
    """
    Delete the vector collection for a specific organization.

    WARNING: This is destructive and cannot be undone!
    """
    from search.vector_store import get_vector_store

    try:
        vector_store = get_vector_store()
        vector_store.delete_collection(organization_id)
        return {
            "status": "deleted",
            "organization_id": organization_id,
            "message": f"Collection for org '{organization_id}' deleted successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
