"""Image processing module for loading and preprocessing images."""

import logging
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

import config
from .exif_parser import extract_exif_telemetry
from observability.langfuse_integration import trace_operation

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images for embedding and storage."""

    def __init__(
        self,
        thumbnail_size: int = None,
        output_dir: Path = None,
    ):
        """
        Initialize the image processor.

        Args:
            thumbnail_size: Size of thumbnails
            output_dir: Directory to save thumbnails
        """
        self.thumbnail_size = thumbnail_size or config.THUMBNAIL_SIZE
        self.thumbnails_dir = output_dir or config.THUMBNAILS_DIR
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image

        Returns:
            PIL Image object
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def create_thumbnail(
        self,
        image: Image.Image,
        save_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Create a thumbnail from an image.

        Args:
            image: PIL Image object
            save_path: Optional path to save the thumbnail

        Returns:
            Thumbnail as PIL Image
        """
        thumbnail = image.copy()
        thumbnail.thumbnail((self.thumbnail_size, self.thumbnail_size))

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            thumbnail.save(save_path, "JPEG", quality=85)

        return thumbnail

    def get_image_metadata(
        self,
        image_path: Path,
        zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metadata for an image.

        Args:
            image_path: Path to the image
            zone: Zone/location identifier

        Returns:
            Metadata dict
        """
        image_path = Path(image_path)
        image = self.load_image(image_path)

        # Get file modification time
        try:
            file_mtime = os.path.getmtime(image_path)
            timestamp = datetime.fromtimestamp(file_mtime)
        except:
            timestamp = datetime.now()

        # Calculate average brightness
        img_array = np.array(image.convert("L"))
        avg_brightness = float(img_array.mean())
        is_night = avg_brightness < 50

        # Extract DJI EXIF/XMP drone telemetry (graceful no-op if exiftool absent)
        with trace_operation(
            name="exif-telemetry-extraction",
            operation_type="span",
            metadata={"image_file": image_path.name},
            tags=["telemetry", "exif"],
        ) as exif_trace:
            t0 = time.time()
            exif = extract_exif_telemetry(image_path)
            t_exif = time.time() - t0
            has_gps = exif.get("drone_lat") is not None
            logger.info(
                "EXIF extraction for %s in %.3fs (has_gps=%s)",
                image_path.name, t_exif, has_gps,
            )
            if exif_trace:
                exif_trace.update(output={
                    "extraction_seconds": round(t_exif, 3),
                    "has_gps": has_gps,
                    "has_gimbal": exif.get("gimbal_yaw") is not None,
                    "has_capture_timestamp": exif.get("capture_timestamp") is not None,
                })

        metadata: Dict[str, Any] = {
            "source_file": image_path.name,
            "image_path": str(image_path),
            "frame_number": 0,  # Single image
            "timestamp": timestamp.isoformat(),
            "zone": zone or "unknown",
            "is_night": bool(is_night),
            "avg_brightness": avg_brightness,
            "width": image.width,
            "height": image.height,
            "source_type": "image",
        }

        # Merge drone telemetry — keep only non-None values so Qdrant payload
        # stays clean for non-DJI images.
        drone_fields = {k: v for k, v in exif.items() if v is not None}

        # Prefer EXIF capture_timestamp over file-mtime if available.
        if drone_fields.get("capture_timestamp"):
            metadata["capture_timestamp"] = drone_fields.pop("capture_timestamp")
        else:
            drone_fields.pop("capture_timestamp", None)

        metadata.update(drone_fields)
        return metadata

    def process_image(
        self,
        image_path: Path,
        zone: Optional[str] = None,
        save_thumbnail: bool = True,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Process a single image.

        Args:
            image_path: Path to the image
            zone: Zone/location identifier
            save_thumbnail: Whether to save thumbnail

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        image_path = Path(image_path)
        image = self.load_image(image_path)
        metadata = self.get_image_metadata(image_path, zone=zone)

        # Create and save thumbnail
        if save_thumbnail:
            thumb_dir = self.thumbnails_dir / "images"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = thumb_dir / f"{image_path.stem}_thumb.jpg"
            self.create_thumbnail(image, save_path=thumb_path)
            metadata["thumbnail_path"] = str(thumb_path)

        metadata["_pil_image"] = image

        return image, metadata

    def process_images_batch(
        self,
        image_paths: List[Path],
        zone_mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images.

        Args:
            image_paths: List of image file paths
            zone_mapping: Optional mapping of filename to zone

        Returns:
            List of metadata dicts
        """
        all_metadata = []

        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            zone = None

            if zone_mapping:
                zone = zone_mapping.get(image_path.name)

            try:
                _, metadata = self.process_image(image_path, zone=zone)
                metadata["frame_number"] = i  # Sequential numbering
                all_metadata.append(metadata)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        return all_metadata


def get_image_files(
    directory: Path,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> List[Path]:
    """
    Find all image files in a directory.

    Args:
        directory: Directory to search
        extensions: Image file extensions to look for

    Returns:
        List of image file paths
    """
    directory = Path(directory)
    image_files = []

    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_files)
