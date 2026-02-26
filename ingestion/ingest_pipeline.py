"""Main ingestion pipeline for processing videos/images and storing embeddings."""

import gc
import json
import logging
import re
import shutil
import tempfile
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm

from .video_processor import VideoProcessor, get_video_files
from .image_processor import ImageProcessor, get_image_files
from .embedding_service import EmbeddingService, get_embedding_service
from .srt_parser import extract_srt_from_video, parse_srt_to_telemetry, get_clip_telemetry
from search.vector_store import VectorStore, get_vector_store
from observability.langfuse_integration import trace_operation
import config

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Pipeline for ingesting videos and images into the vector store."""

    def __init__(
        self,
        video_processor: VideoProcessor = None,
        image_processor: ImageProcessor = None,
        embedding_service: EmbeddingService = None,
        vector_store: VectorStore = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            video_processor: VideoProcessor instance
            image_processor: ImageProcessor instance
            embedding_service: EmbeddingService instance
            vector_store: VectorStore instance
        """
        self.video_processor = video_processor or VideoProcessor()
        self.image_processor = image_processor or ImageProcessor()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

        # Track ingestion progress
        self.stats = {
            "videos_processed": 0,
            "images_processed": 0,
            "frames_extracted": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "errors": [],
        }

    def ingest_video(
        self,
        video_path: Path,
        organization_id: str,
        zone: Optional[str] = None,
        batch_size: int = 50,
        save_full_frames: Optional[bool] = None,
    ) -> int:
        """
        Ingest a single video file.

        Args:
            video_path: Path to the video file
            organization_id: Organization identifier (determines Qdrant collection)
            zone: Zone/location identifier
            batch_size: Number of frames to process at once
            save_full_frames: Whether to save full-resolution frames (overrides config if provided)

        Returns:
            Number of frames ingested
        """
        video_path = Path(video_path)
        print(f"\n{'='*60}")
        print(f"Ingesting video: {video_path.name}")
        print(f"{'='*60}")

        _save_full_frames = save_full_frames if save_full_frames is not None else config.SAVE_FULL_FRAMES
        frames_metadata = self.video_processor.extract_and_save_frames(
            video_path,
            zone=zone,
            save_full_frames=_save_full_frames,
        )

        if not frames_metadata:
            print(f"No frames extracted from {video_path.name}")
            return 0

        self.stats["frames_extracted"] += len(frames_metadata)

        embeddings = []
        metadata_batch = []
        embed_batch_size = getattr(config, 'BATCH_SIZE', 8)

        # Collect images + metadata for batch embedding
        pending_images: List[Image.Image] = []
        pending_metas: List[Dict[str, Any]] = []

        print(f"\nGenerating embeddings for {len(frames_metadata)} frames (batch_size={embed_batch_size})...")

        for i, metadata in enumerate(tqdm(frames_metadata, desc="Embedding frames")):
            pil_image = metadata.pop("_pil_image", None)

            if pil_image is None:
                thumb_path = metadata.get("thumbnail_path")
                if thumb_path:
                    pil_image = Image.open(thumb_path).convert("RGB")
                else:
                    continue

            pending_images.append(pil_image)
            pending_metas.append(metadata)

            if len(pending_images) >= embed_batch_size:
                try:
                    batch_embs = self.embedding_service.embed_images_batch_gpu(pending_images)
                    embeddings.extend(batch_embs)
                    metadata_batch.extend(pending_metas)
                    self.stats["embeddings_generated"] += len(batch_embs)
                except Exception as e:
                    self.stats["errors"].append(f"Batch embedding error: {str(e)}")
                pending_images = []
                pending_metas = []

            if len(embeddings) >= batch_size:
                self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                self.stats["vectors_stored"] += len(embeddings)
                embeddings = []
                metadata_batch = []

        # Flush remaining pending images
        if pending_images:
            try:
                batch_embs = self.embedding_service.embed_images_batch_gpu(pending_images)
                embeddings.extend(batch_embs)
                metadata_batch.extend(pending_metas)
                self.stats["embeddings_generated"] += len(batch_embs)
            except Exception as e:
                self.stats["errors"].append(f"Batch embedding error: {str(e)}")

        if embeddings:
            self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
            self.stats["vectors_stored"] += len(embeddings)

        self.stats["videos_processed"] += 1
        print(f"Completed: {len(frames_metadata)} frames from {video_path.name}")

        return len(frames_metadata)

    def ingest_video_auto(
        self,
        video_path: Path,
        organization_id: str,
        zone: Optional[str] = None,
        batch_size: int = 50,
        save_full_frames: Optional[bool] = None,
    ) -> int:
        """
        Automatically ingest a video with optimized settings based on duration.

        Args:
            video_path: Path to the video file
            organization_id: Organization identifier (determines Qdrant collection)
            zone: Zone/location identifier
            batch_size: Number of clips/frames to process at once
            save_full_frames: Whether to save full-resolution frames (overrides config if provided)

        Returns:
            Number of items ingested (clips or frames)
        """
        video_path = Path(video_path)
        video_info = self.video_processor.get_video_info(video_path)
        duration = video_info['duration_seconds']

        print(f"\nVideo duration: {video_info['duration_formatted']} ({duration:.1f}s)")

        if duration > config.LONG_VIDEO_THRESHOLD_SECONDS:
            print(f"Long video detected (>{config.LONG_VIDEO_THRESHOLD_SECONDS}s)")
            print(f"Using optimized settings:")
            print(f"  - Frame rate: {config.LONG_VIDEO_FRAME_RATE} fps")
            print(f"  - Semantic clips: {config.SEMANTIC_CLIP_DURATION}s each")
            print(f"  - Max frames per clip: {config.SEMANTIC_CLIP_MAX_FRAMES}")

            original_frame_rate = self.video_processor.frame_rate
            self.video_processor.frame_rate = config.LONG_VIDEO_FRAME_RATE

            try:
                result = self.ingest_video_semantic_clips(
                    video_path,
                    organization_id=organization_id,
                    zone=zone,
                    clip_duration=config.SEMANTIC_CLIP_DURATION,
                    max_frames_per_clip=config.SEMANTIC_CLIP_MAX_FRAMES,
                    batch_size=batch_size,
                    save_full_frames=save_full_frames,
                )
            finally:
                self.video_processor.frame_rate = original_frame_rate

            return result
        else:
            print("Using standard video ingestion")
            return self.ingest_video(
                video_path, organization_id=organization_id,
                zone=zone, batch_size=batch_size, save_full_frames=save_full_frames,
            )

    def ingest_video_semantic_clips(
        self,
        video_path: Path,
        organization_id: str,
        zone: Optional[str] = None,
        clip_duration: float = 4.0,
        max_frames_per_clip: int = 32,
        batch_size: int = 50,
        save_full_frames: Optional[bool] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Ingest a single video using semantic clips instead of individual frames.

        Args:
            video_path: Path to the video file.
            organization_id: Organization identifier (determines Qdrant collection).
            zone: Zone/location identifier.
            clip_duration: Target duration (in seconds) for each clip.
            max_frames_per_clip: Upper bound on frames per clip (for memory).
            batch_size: Number of clips to write to the vector store at once.
            save_full_frames: Whether to save full-resolution frames (overrides config if provided).
            additional_metadata: Optional dict of metadata to add to each clip.

        Returns:
            Number of clips ingested.
        """
        video_path = Path(video_path)
        print(f"\n{'='*60}")
        print(f"Ingesting video (semantic clips): {video_path.name}")
        print(f"{'='*60}")

        # Directory for clip thumbnails
        video_name = video_path.stem
        clips_thumb_dir = config.THUMBNAILS_DIR / f"{video_name}_clips"
        clips_thumb_dir.mkdir(parents=True, exist_ok=True)

        # Use the low-level frame generator (no saving to disk yet)
        frame_generator = self.video_processor.extract_frames(
            video_path,
            zone=zone,
        )

        embeddings: List[np.ndarray] = []
        metadata_batch: List[Dict[str, Any]] = []

        # Pending clips waiting to be batch-embedded
        pending_clip_frames: List[List[Image.Image]] = []
        pending_clip_metas: List[Dict[str, Any]] = []

        clip_frames: List[Image.Image] = []
        clip_metas: List[Dict[str, Any]] = []
        clip_start_sec: Optional[float] = None
        clip_index: int = 0
        total_frames: int = 0
        total_clips: int = 0

        embed_batch_size = getattr(config, 'CLIP_EMBED_BATCH_SIZE', 2)

        def finalize_clip():
            """Build metadata for a completed clip and add to pending batch."""
            nonlocal clip_frames, clip_metas, clip_start_sec, clip_index
            nonlocal pending_clip_frames, pending_clip_metas

            if not clip_frames:
                return

            start_meta = clip_metas[0]
            end_meta = clip_metas[-1]

            avg_brightness = float(
                np.mean([m.get("avg_brightness", 0.0) for m in clip_metas])
            )
            night_votes = sum(1 for m in clip_metas if m.get("is_night"))
            is_night = night_votes >= max(1, len(clip_metas) // 2)

            rep_idx = len(clip_frames) // 2
            rep_image = clip_frames[rep_idx]

            thumb = rep_image.copy()
            thumb.thumbnail((config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE))
            thumb_path = clips_thumb_dir / f"clip_{clip_index:06d}.jpg"
            thumb.save(thumb_path, "JPEG", quality=85)

            clip_meta: Dict[str, Any] = {
                "source_file": video_path.name,
                "video_path": str(video_path),
                "frame_number": clip_index,
                "clip_index": clip_index,
                "clip_start_timestamp": start_meta.get("timestamp"),
                "clip_end_timestamp": end_meta.get("timestamp"),
                "clip_start_seconds": start_meta.get("seconds_offset"),
                "clip_end_seconds": end_meta.get("seconds_offset"),
                "zone": start_meta.get("zone", zone or "unknown"),
                "is_night": bool(is_night),
                "avg_brightness": avg_brightness,
                "num_frames": len(clip_frames),
                "thumbnail_path": str(thumb_path),
                "source_type": "video_clip",
            }

            if telemetry:
                t_start = float(start_meta.get("seconds_offset", 0.0))
                t_end = float(end_meta.get("seconds_offset", t_start))
                drone_fields = get_clip_telemetry(telemetry, t_start, t_end)
                clip_meta.update(drone_fields)

            if additional_metadata:
                clip_meta.update(additional_metadata)

            _save_full_frames = save_full_frames if save_full_frames is not None else config.SAVE_FULL_FRAMES
            if _save_full_frames:
                video_name = video_path.stem
                frames_dir = config.FRAMES_DIR / f"{video_name}_clips"
                frames_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frames_dir / f"clip_{clip_index:06d}.jpg"
                rep_image.save(frame_path, "JPEG", quality=95)
                clip_meta["frame_path"] = str(frame_path)

            pending_clip_frames.append(clip_frames)
            pending_clip_metas.append(clip_meta)

            clip_index += 1
            clip_frames = []
            clip_metas = []
            clip_start_sec = None

        def flush_pending_embeddings():
            """Batch-embed all pending clips and release frame memory."""
            nonlocal pending_clip_frames, pending_clip_metas
            nonlocal embeddings, metadata_batch, total_clips

            if not pending_clip_frames:
                return

            try:
                batch_embeddings = self.embedding_service.embed_video_clips_batch_gpu(
                    pending_clip_frames
                )
                embeddings.extend(batch_embeddings)
                metadata_batch.extend(pending_clip_metas)
                self.stats["embeddings_generated"] += len(batch_embeddings)
                total_clips += len(batch_embeddings)
            except Exception as e:
                self.stats["errors"].append(f"Batch clip embedding error: {str(e)}")

            for clip in pending_clip_frames:
                for img in clip:
                    img.close()
                clip.clear()
            pending_clip_frames.clear()
            pending_clip_metas.clear()
            pending_clip_frames = []
            pending_clip_metas = []
            gc.collect()

        with trace_operation(
            name="process-video-clips",
            operation_type="span",
            metadata={
                "video_name": video_path.name,
                "clip_duration": clip_duration,
                "max_frames_per_clip": max_frames_per_clip,
                "zone": zone
            },
            tags=["processing", "ingestion"]
        ) as process_trace:
            print(f"Building semantic clips and batch-embedding (batch_size={embed_batch_size})...")
            for pil_image, meta in tqdm(frame_generator, desc="Processing frames for clips"):
                total_frames += 1

                sec_offset = float(meta.get("seconds_offset", 0.0))
                if clip_start_sec is None:
                    clip_start_sec = sec_offset

                clip_frames.append(pil_image)
                clip_metas.append(meta)

                duration = sec_offset - clip_start_sec
                if duration >= clip_duration or len(clip_frames) >= max_frames_per_clip:
                    finalize_clip()

                    # When enough clips are pending, batch-embed them all at once
                    if len(pending_clip_frames) >= embed_batch_size:
                        flush_pending_embeddings()

                    # Store vectors when full
                    if len(embeddings) >= batch_size:
                        with trace_operation(
                            name="store-vectors-batch",
                            operation_type="span",
                            metadata={"batch_size": len(embeddings)},
                            tags=["storage", "qdrant"]
                        ) as storage_trace:
                            batch_size_to_store = len(embeddings)
                            self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                            self.stats["vectors_stored"] += batch_size_to_store

                            if storage_trace:
                                storage_trace.update(
                                    output={
                                        "vectors_stored": batch_size_to_store,
                                        "status": "completed"
                                    }
                                )

                            embeddings = []
                            metadata_batch = []

            # Flush last partial clip and any remaining pending embeddings
            finalize_clip()
            flush_pending_embeddings()

            if embeddings:
                with trace_operation(
                    name="store-vectors-final",
                    operation_type="span",
                    metadata={"batch_size": len(embeddings)},
                    tags=["storage", "qdrant"]
                ) as storage_trace:
                    final_batch_size = len(embeddings)
                    self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                    self.stats["vectors_stored"] += final_batch_size

                    if storage_trace:
                        storage_trace.update(
                            output={
                                "vectors_stored": final_batch_size,
                                "status": "completed"
                            }
                        )

            self.stats["frames_extracted"] += total_frames
            self.stats["videos_processed"] += 1

            if process_trace:
                process_trace.update(
                    output={
                        "total_clips": total_clips,
                        "total_frames": total_frames,
                        "embeddings_generated": total_clips,
                        "status": "completed"
                    }
                )

            print(f"Completed: {total_clips} semantic clips from {video_path.name}")
            return total_clips

    def ingest_images(
        self,
        image_paths: List[Path],
        organization_id: str,
        zone: Optional[str] = None,
        batch_size: int = 50,
    ) -> int:
        """
        Ingest multiple images.

        Args:
            image_paths: List of image file paths
            organization_id: Organization identifier (determines Qdrant collection)
            zone: Zone/location identifier
            batch_size: Number of images to process at once

        Returns:
            Number of images ingested
        """
        print(f"\n{'='*60}")
        print(f"Ingesting {len(image_paths)} images")
        print(f"{'='*60}")

        embeddings = []
        metadata_batch = []

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                _, metadata = self.image_processor.process_image(
                    image_path,
                    zone=zone,
                )

                # Get the PIL image
                pil_image = metadata.pop("_pil_image", None)

                if pil_image is None:
                    continue

                # Generate embedding
                embedding = self.embedding_service.embed_image(pil_image)
                embeddings.append(embedding)

                # Update frame number for sequential ordering
                metadata["frame_number"] = i
                metadata_batch.append(metadata)

                self.stats["embeddings_generated"] += 1
                self.stats["images_processed"] += 1

            except Exception as e:
                self.stats["errors"].append(f"Image error {image_path}: {str(e)}")
                continue

            # Store batch when full
            if len(embeddings) >= batch_size:
                self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                self.stats["vectors_stored"] += len(embeddings)
                embeddings = []
                metadata_batch = []

        # Store remaining embeddings
        if embeddings:
            self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
            self.stats["vectors_stored"] += len(embeddings)

        print(f"Completed: {self.stats['images_processed']} images ingested")
        return self.stats["images_processed"]

    def ingest_video_frames(
        self,
        frames_dir: Path,
        organization_id: str,
        sample_rate: int = 1,
        source_video_name: Optional[str] = None,
        original_fps: Optional[float] = None,
        zone: Optional[str] = None,
        base_timestamp: Optional[datetime] = None,
        batch_size: int = 50,
    ) -> int:
        """
        Ingest pre-extracted video frames (images) with frame sampling.

        This method is useful when you have a folder of images that were
        extracted from a video and want to ingest them with frame sampling,
        treating them as video frames with proper metadata.

        Args:
            frames_dir: Directory containing the frame images
            sample_rate: Sample every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)
            source_video_name: Optional name to identify the source video
            original_fps: FPS of the original video (for timestamp calculation)
            zone: Zone/location identifier
            base_timestamp: Base timestamp for the first frame
            batch_size: Number of frames to process at once

        Returns:
            Number of frames ingested
        """
        frames_dir = Path(frames_dir)

        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Get all image files from the directory
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        frame_files = []
        for ext in image_extensions:
            frame_files.extend(frames_dir.glob(f"*{ext}"))
            frame_files.extend(frames_dir.glob(f"*{ext.upper()}"))

        if not frame_files:
            print(f"No frame images found in {frames_dir}")
            return 0

        # Sort frames naturally (handle frame_001, frame_002, ..., frame_010, etc.)
        def natural_sort_key(path: Path) -> tuple:
            """Extract numbers from filename for natural sorting."""
            numbers = re.findall(r'\d+', path.stem)
            if numbers:
                # Return the last number found (usually the frame number)
                return (int(numbers[-1]),)
            return (0,)

        frame_files = sorted(frame_files, key=natural_sort_key)
        total_frames = len(frame_files)

        # Apply frame sampling
        sampled_frames = frame_files[::sample_rate]
        num_sampled = len(sampled_frames)

        # Determine source name
        source_name = source_video_name or frames_dir.name

        print(f"\n{'='*60}")
        print(f"Ingesting video frames from: {frames_dir.name}")
        print(f"{'='*60}")
        print(f"Total frames found: {total_frames}")
        print(f"Sample rate: every {sample_rate} frame(s)")
        print(f"Frames to ingest: {num_sampled}")
        print(f"Source video name: {source_name}")

        # Base timestamp (use current time if not provided)
        if base_timestamp is None:
            # Try to get from first file's modification time
            try:
                file_mtime = frame_files[0].stat().st_mtime
                base_timestamp = datetime.fromtimestamp(file_mtime)
            except:
                base_timestamp = datetime.now()

        # Default FPS for timestamp calculation
        fps = original_fps or 30.0

        # Create thumbnail directory for this video
        thumb_dir = config.THUMBNAILS_DIR / source_name
        thumb_dir.mkdir(parents=True, exist_ok=True)

        embeddings = []
        metadata_batch = []
        frames_processed = 0

        print(f"\nProcessing {num_sampled} sampled frames...")

        for idx, frame_path in enumerate(tqdm(sampled_frames, desc="Processing frames")):
            try:
                # Load the image
                pil_image = Image.open(frame_path)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                # Calculate the original frame number (before sampling)
                original_frame_number = idx * sample_rate

                # Calculate timestamp for this frame
                seconds_offset = original_frame_number / fps
                frame_timestamp = base_timestamp + timedelta(seconds=seconds_offset)

                # Calculate average brightness for night detection
                img_array = np.array(pil_image.convert("L"))
                avg_brightness = float(img_array.mean())
                is_night = avg_brightness < 50

                # Build metadata (similar to video frame metadata)
                metadata = {
                    "source_file": source_name,
                    "video_path": str(frames_dir),
                    "original_image_path": str(frame_path),
                    "frame_number": idx,  # Sequential number of sampled frames
                    "original_frame_number": original_frame_number,
                    "timestamp": frame_timestamp.isoformat(),
                    "seconds_offset": seconds_offset,
                    "zone": zone or "unknown",
                    "is_night": bool(is_night),
                    "avg_brightness": avg_brightness,
                    "sample_rate": sample_rate,
                    "source_type": "pre_extracted_frames",
                }

                # Create and save thumbnail
                thumbnail = pil_image.copy()
                thumbnail.thumbnail((config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE))
                thumb_path = thumb_dir / f"frame_{idx:06d}.jpg"
                thumbnail.save(thumb_path, "JPEG", quality=85)
                metadata["thumbnail_path"] = str(thumb_path)

                # Generate embedding
                embedding = self.embedding_service.embed_image(pil_image)
                embeddings.append(embedding)
                metadata_batch.append(metadata)
                self.stats["embeddings_generated"] += 1
                frames_processed += 1

            except Exception as e:
                self.stats["errors"].append(f"Frame error {frame_path}: {str(e)}")
                continue

            # Store batch when full
            if len(embeddings) >= batch_size:
                self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                self.stats["vectors_stored"] += len(embeddings)
                embeddings = []
                metadata_batch = []

        # Store remaining embeddings
        if embeddings:
            self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
            self.stats["vectors_stored"] += len(embeddings)

        self.stats["frames_extracted"] += frames_processed
        self.stats["videos_processed"] += 1

        print(f"Completed: {frames_processed} frames ingested from {source_name}")
        return frames_processed

    def ingest_directory(
        self,
        directory: Path,
        organization_id: str,
        zone_mapping: Optional[Dict[str, str]] = None,
        process_videos: bool = True,
        process_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest all videos and images from a directory.

        Args:
            directory: Directory containing media files
            organization_id: Organization identifier (determines Qdrant collection)
            zone_mapping: Optional mapping of filename to zone
            process_videos: Whether to process video files
            process_images: Whether to process image files

        Returns:
            Ingestion statistics
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        print(f"\n{'='*60}")
        print(f"Ingesting directory: {directory}")
        print(f"{'='*60}")

        start_time = datetime.now()

        # Process videos
        if process_videos:
            video_files = get_video_files(directory)
            print(f"Found {len(video_files)} video files")

            for video_path in video_files:
                zone = None
                if zone_mapping:
                    zone = zone_mapping.get(video_path.name)

                self.ingest_video(video_path, organization_id=organization_id, zone=zone)

        # Process images
        if process_images:
            image_files = get_image_files(directory)
            print(f"Found {len(image_files)} image files")

            if image_files:
                # Build zone list if mapping provided
                zones = []
                for img_path in image_files:
                    if zone_mapping:
                        zones.append(zone_mapping.get(img_path.name))
                    else:
                        zones.append(None)

                self.ingest_images(image_files, organization_id=organization_id)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.stats["duration_seconds"] = duration
        self.stats["start_time"] = start_time.isoformat()
        self.stats["end_time"] = end_time.isoformat()

        print(f"\n{'='*60}")
        print("Ingestion Complete!")
        print(f"{'='*60}")
        print(f"Videos processed: {self.stats['videos_processed']}")
        print(f"Images processed: {self.stats['images_processed']}")
        print(f"Frames extracted: {self.stats['frames_extracted']}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']}")
        print(f"Vectors stored: {self.stats['vectors_stored']}")
        print(f"Duration: {duration:.1f} seconds")

        if self.stats["errors"]:
            print(f"Errors: {len(self.stats['errors'])}")

        return self.stats

    def ingest_video_from_url(
        self,
        video_url: str,
        organization_id: str,
        zone: Optional[str] = None,
        clip_duration: float = 4.0,
        max_frames_per_clip: int = 32,
        batch_size: int = 50,
        save_full_frames: Optional[bool] = None,
        cleanup_after: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        s3_storage_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a video from a URL and ingest it using semantic clips.

        Args:
            video_url: URL of the video to download (S3, HTTP, HTTPS)
            organization_id: Organization identifier (determines Qdrant collection)
            zone: Zone/location identifier
            clip_duration: Target duration in seconds for each clip (default: 4.0)
            max_frames_per_clip: Maximum frames per semantic clip (default: 32)
            batch_size: Number of clips to process at once (default: 50)
            save_full_frames: Whether to save full-resolution frames (overrides config if provided)
            cleanup_after: Whether to delete the downloaded video after ingestion (default: True)
            metadata: Optional dict of additional metadata to add to each clip
            s3_storage_path: Original ``s3://bucket/key`` path. When provided
                (and boto3 is available), the video is downloaded directly via
                boto3's multipart transfer instead of streaming through the
                presigned HTTP URL, which is significantly faster for large files.

        Returns:
            Dictionary containing ingestion stats and metadata
        """
        print(f"\n{'='*60}")
        print(f"Ingesting video from URL: {video_url}")
        print(f"{'='*60}")

        # Validate URL
        if not video_url.startswith(('http://', 'https://', 's3://')):
            raise ValueError(f"Invalid URL scheme. Expected http://, https://, or s3://. Got: {video_url}")

        # Download to the mounted data volume (not /tmp) so large files
        # don't exhaust the container overlay filesystem.
        download_root = config.DATA_DIR / "tmp"
        download_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="video_rag_download_", dir=str(download_root)))

        try:
            # Extract filename from URL or generate one
            url_path = video_url.split('?')[0]  # Remove query parameters
            filename = Path(url_path).name
            if not filename or '.' not in filename:
                # Generate filename with timestamp if URL doesn't have a clear filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"video_{timestamp}.mp4"

            video_path = temp_dir / filename

            # Download the video with tracing
            with trace_operation(
                name="download-video",
                operation_type="span",
                metadata={"url": video_url[:100], "filename": filename},
                tags=["download", "ingestion"]
            ) as download_trace:
                print(f"Downloading video from: {video_url}")
                print(f"Saving to: {video_path}")

                try:
                    downloaded_via_s3 = False

                    # Fast path: direct S3 multipart download (parallel threads)
                    if s3_storage_path:
                        from ingestion.s3_service import get_s3_service
                        s3_svc = get_s3_service()
                        if s3_svc is not None:
                            total_bytes = s3_svc.get_object_size(s3_storage_path)
                            total_mb = (total_bytes / (1024 * 1024)) if total_bytes else None
                            logger.info(
                                "Starting S3 multipart download (%s)",
                                f"{total_mb:.1f} MB" if total_mb else "unknown size",
                            )

                            downloaded_bytes = [0]
                            last_log_bytes = [0]
                            log_interval = 100 * 1024 * 1024  # log every 100 MB
                            dl_start = time.time()

                            def _s3_progress(chunk_bytes: int):
                                downloaded_bytes[0] += chunk_bytes
                                if downloaded_bytes[0] - last_log_bytes[0] >= log_interval:
                                    last_log_bytes[0] = downloaded_bytes[0]
                                    cur_mb = downloaded_bytes[0] / (1024 * 1024)
                                    elapsed = time.time() - dl_start
                                    speed = cur_mb / elapsed if elapsed > 0 else 0
                                    if total_mb:
                                        pct = downloaded_bytes[0] / total_bytes * 100
                                        logger.info(
                                            "S3 download: %.0f / %.0f MB (%.0f%%) — %.1f MB/s",
                                            cur_mb, total_mb, pct, speed,
                                        )
                                    else:
                                        logger.info(
                                            "S3 download: %.0f MB — %.1f MB/s",
                                            cur_mb, speed,
                                        )

                            try:
                                s3_svc.download_file(
                                    storage_path=s3_storage_path,
                                    local_path=str(video_path),
                                    callback=_s3_progress,
                                )
                                dl_elapsed = time.time() - dl_start
                                downloaded_via_s3 = True
                            except RuntimeError as s3_err:
                                logger.warning(
                                    "Direct S3 download failed (%s) — falling back to HTTP presigned URL download.",
                                    s3_err,
                                )

                    # Fallback: stream download via HTTP (presigned URL or public URL)
                    if not downloaded_via_s3:
                        response = requests.get(video_url, stream=True, timeout=30)
                        response.raise_for_status()

                        total_size = int(response.headers.get('content-length', 0))
                        total_size_mb = total_size / (1024 * 1024) if total_size else 0
                        chunk_size = 1024 * 1024  # 1 MB
                        downloaded_bytes = 0
                        last_log_bytes = 0
                        log_interval = 100 * 1024 * 1024  # log every 100 MB
                        dl_start = time.time()

                        logger.info(
                            "Starting HTTP download (%s)",
                            f"{total_size_mb:.1f} MB" if total_size else "unknown size",
                        )

                        with open(video_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_bytes += len(chunk)
                                    if downloaded_bytes - last_log_bytes >= log_interval:
                                        last_log_bytes = downloaded_bytes
                                        cur_mb = downloaded_bytes / (1024 * 1024)
                                        elapsed = time.time() - dl_start
                                        speed = cur_mb / elapsed if elapsed > 0 else 0
                                        if total_size:
                                            pct = downloaded_bytes / total_size * 100
                                            logger.info(
                                                "HTTP download: %.0f / %.0f MB (%.0f%%) — %.1f MB/s",
                                                cur_mb, total_size_mb, pct, speed,
                                            )
                                        else:
                                            logger.info(
                                                "HTTP download: %.0f MB — %.1f MB/s",
                                                cur_mb, speed,
                                            )

                        dl_elapsed = time.time() - dl_start

                    file_size_mb = video_path.stat().st_size / (1024 * 1024)
                    logger.info(
                        "Download completed: %.2f MB in %.1fs (%.1f MB/s)",
                        file_size_mb, dl_elapsed, file_size_mb / dl_elapsed if dl_elapsed > 0 else 0,
                    )

                    video_info = self.video_processor.get_video_info(video_path)
                    video_duration_sec = video_info["duration_seconds"]
                    logger.info(
                        "Video duration: %s (%.1fs)",
                        video_info["duration_formatted"], video_duration_sec,
                    )

                    if download_trace:
                        download_trace.update(
                            output={
                                "file_size_mb": round(file_size_mb, 2),
                                "video_duration_seconds": round(video_duration_sec, 2),
                                "video_duration_formatted": video_info["duration_formatted"],
                                "video_resolution": f"{video_info['width']}x{video_info['height']}",
                                "video_fps": video_info["fps"],
                                "download_seconds": round(dl_elapsed, 2),
                                "download_speed_mbps": round(file_size_mb / dl_elapsed, 2) if dl_elapsed > 0 else 0,
                                "status": "completed",
                            }
                        )

                except requests.exceptions.RequestException as e:
                    if download_trace:
                        download_trace.update(level="ERROR", status_message=str(e))
                    raise ValueError(f"Failed to download video from URL: {e}")
                except RuntimeError as e:
                    if download_trace:
                        download_trace.update(level="ERROR", status_message=str(e))
                    raise ValueError(f"Failed to download video from S3: {e}")

            # Verify the downloaded file exists and has content
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise ValueError(f"Downloaded video file is empty or doesn't exist: {video_path}")

            # Extract embedded SRT telemetry (DJI drone footage)
            telemetry_array: Optional[List[Dict[str, Any]]] = None
            logger.info("Extracting embedded SRT telemetry...")
            with trace_operation(
                name="srt-telemetry-extraction",
                operation_type="span",
                metadata={"video_file": video_path.name},
                tags=["telemetry", "srt"],
            ) as srt_trace:
                t0 = time.time()
                srt_text = extract_srt_from_video(video_path)
                t_extract = time.time() - t0

                if srt_text:
                    t1 = time.time()
                    telemetry_array = parse_srt_to_telemetry(srt_text)
                    t_parse = time.time() - t1
                    total_srt = time.time() - t0
                    logger.info(
                        "SRT extracted and parsed in %.2fs "
                        "(extract=%.3fs, parse=%.3fs) — %d telemetry entries",
                        total_srt, t_extract, t_parse, len(telemetry_array),
                    )
                    if srt_trace:
                        srt_trace.update(output={
                            "total_seconds": round(total_srt, 3),
                            "extract_seconds": round(t_extract, 3),
                            "parse_seconds": round(t_parse, 3),
                            "telemetry_entries": len(telemetry_array),
                            "srt_found": True,
                        })
                else:
                    logger.info(
                        "No SRT subtitle stream found in %.3fs — telemetry fields will be null",
                        t_extract,
                    )
                    if srt_trace:
                        srt_trace.update(output={
                            "total_seconds": round(t_extract, 3),
                            "extract_seconds": round(t_extract, 3),
                            "srt_found": False,
                            "telemetry_entries": 0,
                        })

            # Ingest the video using semantic clips
            print(f"\nProcessing downloaded video...")
            clips_count = self.ingest_video_semantic_clips(
                video_path=video_path,
                organization_id=organization_id,
                zone=zone,
                clip_duration=clip_duration,
                max_frames_per_clip=max_frames_per_clip,
                batch_size=batch_size,
                save_full_frames=save_full_frames,
                additional_metadata=metadata,
                telemetry=telemetry_array,
            )

            # Prepare result
            result = {
                "status": "completed",
                "video_url": video_url,
                "downloaded_path": str(video_path),
                "clips_ingested": clips_count,
                "file_size_mb": video_path.stat().st_size / (1024 * 1024),
                "zone": zone,
                "telemetry": telemetry_array,
                "stats": self.stats,
            }

            print(f"\n{'='*60}")
            print(f"URL Ingestion Complete!")
            print(f"{'='*60}")
            print(f"Video URL: {video_url}")
            print(f"Clips ingested: {clips_count}")
            print(f"File size: {result['file_size_mb']:.2f} MB")

            return result

        finally:
            # Cleanup downloaded file if requested
            if cleanup_after:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"\nCleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Warning: Failed to cleanup temporary files: {e}")
            else:
                print(f"\nDownloaded video retained at: {video_path}")

    def ingest_rtsp_stream(
        self,
        rtsp_url: str,
        organization_id: str,
        zone: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        max_frames: Optional[int] = None,
        use_semantic_clips: bool = True,
        clip_duration: float = 4.0,
        max_frames_per_clip: int = 32,
        batch_size: int = 50,
        save_full_frames: Optional[bool] = None,
        reconnect_on_failure: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 2.0,
        connection_timeout_seconds: float = 10.0,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest frames from an RTSP live stream.

        This method connects to an RTSP stream (IP camera, surveillance system, etc.)
        and continuously captures frames for a specified duration or frame count.

        Unlike file-based ingestion, this handles:
        - Infinite stream sources
        - Real-time frame capture
        - Connection failures and automatic reconnection
        - Duration-based or frame-count-based termination

        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://192.168.1.100:554/stream1)
            zone: Zone/location identifier (e.g., "north_entrance")
            duration_seconds: Duration to capture in seconds (None = until max_frames)
            max_frames: Maximum number of frames to capture (None = until duration)
            use_semantic_clips: Use semantic clips (recommended, default True)
            clip_duration: Target clip duration in seconds
            max_frames_per_clip: Maximum frames per semantic clip
            batch_size: Batch size for vector storage
            save_full_frames: Save full-resolution frames (overrides config)
            reconnect_on_failure: Attempt reconnection if stream drops
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay_seconds: Delay between reconnection attempts
            connection_timeout_seconds: Timeout for initial connection
            additional_metadata: Optional metadata (camera_id, site_id, etc.)

        Returns:
            Dictionary with ingestion stats and metadata

        Raises:
            ValueError: If connection fails or invalid parameters
            RuntimeError: If both duration_seconds and max_frames are None
        """
        print(f"\n{'='*60}")
        print(f"Ingesting RTSP stream: {rtsp_url}")
        print(f"{'='*60}")

        start_time = datetime.now()

        # Get frame generator from video processor
        frame_generator = self.video_processor.extract_frames_from_stream(
            rtsp_url=rtsp_url,
            duration_seconds=duration_seconds,
            max_frames=max_frames,
            zone=zone,
            reconnect_on_failure=reconnect_on_failure,
            max_reconnect_attempts=max_reconnect_attempts,
            reconnect_delay_seconds=reconnect_delay_seconds,
            connection_timeout_seconds=connection_timeout_seconds,
        )

        total_frames = 0
        total_items = 0  # clips or frames

        if use_semantic_clips:
            # Process as semantic clips (recommended for streams)
            print("Using semantic clips for stream ingestion...")

            # Create clip storage directory
            stream_name = f"rtsp_{zone or 'stream'}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            clips_thumb_dir = config.THUMBNAILS_DIR / f"{stream_name}_clips"
            clips_thumb_dir.mkdir(parents=True, exist_ok=True)

            embeddings: List[np.ndarray] = []
            metadata_batch: List[Dict[str, Any]] = []

            clip_frames: List[Image.Image] = []
            clip_metas: List[Dict[str, Any]] = []
            clip_start_sec: Optional[float] = None
            clip_index: int = 0

            def flush_clip():
                nonlocal clip_frames, clip_metas, clip_start_sec, clip_index
                nonlocal embeddings, metadata_batch, total_items

                if not clip_frames:
                    return

                # Aggregate metadata
                start_meta = clip_metas[0]
                end_meta = clip_metas[-1]

                avg_brightness = float(
                    np.mean([m.get("avg_brightness", 0.0) for m in clip_metas])
                )
                night_votes = sum(1 for m in clip_metas if m.get("is_night"))
                is_night = night_votes >= max(1, len(clip_metas) // 2)

                # Representative frame
                rep_idx = len(clip_frames) // 2
                rep_image = clip_frames[rep_idx]

                # Save thumbnail
                thumb = rep_image.copy()
                thumb.thumbnail((config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE))
                thumb_path = clips_thumb_dir / f"clip_{clip_index:06d}.jpg"
                thumb.save(thumb_path, "JPEG", quality=85)

                # Build clip metadata
                clip_meta: Dict[str, Any] = {
                    "source_file": stream_name,
                    "video_path": rtsp_url,
                    "frame_number": clip_index,
                    "clip_index": clip_index,
                    "clip_start_timestamp": start_meta.get("timestamp"),
                    "clip_end_timestamp": end_meta.get("timestamp"),
                    "clip_start_seconds": start_meta.get("seconds_offset"),
                    "clip_end_seconds": end_meta.get("seconds_offset"),
                    "zone": zone or "unknown",
                    "is_night": bool(is_night),
                    "avg_brightness": avg_brightness,
                    "num_frames": len(clip_frames),
                    "thumbnail_path": str(thumb_path),
                    "source_type": "rtsp_stream_clip",
                    "stream_url": rtsp_url,
                }

                # Add additional metadata (camera_id, site_id, etc.)
                if additional_metadata:
                    clip_meta.update(additional_metadata)

                # Optionally save full frame
                _save_full_frames = save_full_frames if save_full_frames is not None else config.SAVE_FULL_FRAMES
                if _save_full_frames:
                    frames_dir = config.FRAMES_DIR / f"{stream_name}_clips"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = frames_dir / f"clip_{clip_index:06d}.jpg"
                    rep_image.save(frame_path, "JPEG", quality=95)
                    clip_meta["frame_path"] = str(frame_path)

                # Generate embedding
                try:
                    embedding = self.embedding_service.embed_video_clip(clip_frames)
                    embeddings.append(embedding)
                    metadata_batch.append(clip_meta)
                    self.stats["embeddings_generated"] += 1
                    total_items += 1
                except Exception as e:
                    self.stats["errors"].append(f"RTSP clip embedding error: {str(e)}")

                clip_index += 1
                clip_frames = []
                clip_metas = []
                clip_start_sec = None

            # Process frames into clips
            print("Processing stream frames into semantic clips...")
            for pil_image, meta in tqdm(frame_generator, desc="Capturing stream"):
                total_frames += 1

                sec_offset = float(meta.get("seconds_offset", 0.0))
                if clip_start_sec is None:
                    clip_start_sec = sec_offset

                clip_frames.append(pil_image)
                clip_metas.append(meta)

                # Flush clip when ready
                duration = sec_offset - clip_start_sec
                if duration >= clip_duration or len(clip_frames) >= max_frames_per_clip:
                    flush_clip()

                    # Store batch when full
                    if len(embeddings) >= batch_size:
                        self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                        self.stats["vectors_stored"] += len(embeddings)
                        embeddings = []
                        metadata_batch = []

            # Flush last partial clip
            flush_clip()

            # Store remaining embeddings
            if embeddings:
                self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                self.stats["vectors_stored"] += len(embeddings)

            print(f"\nCompleted: {total_items} semantic clips from RTSP stream")

        else:
            # Process as individual frames (alternative approach)
            print("Using individual frame ingestion...")

            stream_name = f"rtsp_{zone or 'stream'}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            thumb_dir = config.THUMBNAILS_DIR / stream_name
            thumb_dir.mkdir(parents=True, exist_ok=True)

            embeddings = []
            metadata_batch = []

            for pil_image, meta in tqdm(frame_generator, desc="Capturing stream"):
                total_frames += 1

                # Save thumbnail
                thumb = pil_image.copy()
                thumb.thumbnail((config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE))
                thumb_path = thumb_dir / f"frame_{meta['frame_number']:06d}.jpg"
                thumb.save(thumb_path, "JPEG", quality=85)
                meta["thumbnail_path"] = str(thumb_path)

                # Optionally save full frame
                _save_full_frames = save_full_frames if save_full_frames is not None else config.SAVE_FULL_FRAMES
                if _save_full_frames:
                    frames_dir = config.FRAMES_DIR / stream_name
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = frames_dir / f"frame_{meta['frame_number']:06d}.jpg"
                    pil_image.save(frame_path, "JPEG", quality=95)
                    meta["frame_path"] = str(frame_path)

                # Add additional metadata
                if additional_metadata:
                    meta.update(additional_metadata)

                # Generate embedding
                try:
                    embedding = self.embedding_service.embed_image(pil_image)
                    embeddings.append(embedding)
                    metadata_batch.append(meta)
                    self.stats["embeddings_generated"] += 1
                    total_items += 1
                except Exception as e:
                    self.stats["errors"].append(f"RTSP frame embedding error: {str(e)}")
                    continue

                # Store batch when full
                if len(embeddings) >= batch_size:
                    self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                    self.stats["vectors_stored"] += len(embeddings)
                    embeddings = []
                    metadata_batch = []

            # Store remaining embeddings
            if embeddings:
                self.vector_store.insert_batch(embeddings, metadata_batch, organization_id=organization_id)
                self.stats["vectors_stored"] += len(embeddings)

            print(f"\nCompleted: {total_items} frames from RTSP stream")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.stats["frames_extracted"] += total_frames
        self.stats["videos_processed"] += 1  # Count stream as a "video"

        result = {
            "status": "completed",
            "rtsp_url": rtsp_url,
            "zone": zone,
            "capture_duration_seconds": duration,
            "frames_captured": total_frames,
            "items_ingested": total_items,
            "use_semantic_clips": use_semantic_clips,
            "stats": self.stats,
        }

        print(f"\n{'='*60}")
        print(f"RTSP Stream Ingestion Complete!")
        print(f"{'='*60}")
        print(f"Stream URL: {rtsp_url}")
        print(f"Capture duration: {duration:.1f}s")
        print(f"Frames captured: {total_frames}")
        print(f"Items ingested: {total_items} {'clips' if use_semantic_clips else 'frames'}")

        return result

    def get_stats(self, organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current ingestion statistics."""
        stats = {**self.stats}
        if organization_id:
            collection_info = self.vector_store.get_collection_info(organization_id)
            stats["collection"] = collection_info
        return stats

    def save_stats(self, output_path: Path):
        """Save statistics to a JSON file."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats saved to: {output_path}")


def create_ingest_pipeline() -> IngestPipeline:
    """Create a new ingestion pipeline with default settings."""
    return IngestPipeline()
