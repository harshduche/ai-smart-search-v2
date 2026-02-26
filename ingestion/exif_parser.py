"""DJI EXIF/XMP telemetry extraction for still images.

Runs ``exiftool -j -n <image_path>`` to get all metadata as JSON (numeric GPS
values), then maps DJI-specific fields to the same drone telemetry schema used
by the SRT parser so search results are uniform across images and video clips.

Public API
----------
extract_exif_telemetry(image_path) -> Dict[str, Any]
    Returns a dict with drone telemetry fields (all may be None if exiftool is
    unavailable or the image has no DJI metadata).
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _run_exiftool(image_path: Path) -> Optional[Dict[str, Any]]:
    """Run ``exiftool -j -n`` on *image_path* and return the first tag dict.

    ``-j``  → JSON output
    ``-n``  → numeric values (GPS decimal degrees, float focal length, etc.)

    Returns ``None`` if exiftool is not installed or the command fails.
    """
    cmd = ["exiftool", "-j", "-n", str(image_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout or "[]")
        if isinstance(data, list) and data:
            return data[0]
        return None
    except FileNotFoundError:
        logger.warning("exiftool not found; EXIF drone telemetry extraction skipped")
        return None
    except Exception as exc:
        logger.warning("exiftool failed for %s: %s", image_path.name, exc)
        return None


def _to_float(value: Any) -> Optional[float]:
    """Coerce *value* to float, handling signed strings like '+578.845'."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _parse_exif_datetime(value: Any) -> Optional[str]:
    """Convert an exiftool datetime string to ISO-8601.

    Handles:
    - ``"2026:02:17 17:28:54"``          → ``"2026-02-17T17:28:54"``
    - ``"2026:02:17 11:59:13.710396"``   → ``"2026-02-17T11:59:13.710396"``
    """
    if not value:
        return None
    raw = str(value).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return None


def extract_exif_telemetry(image_path: Path) -> Dict[str, Any]:
    """Extract DJI drone/gimbal telemetry from an image's EXIF/XMP metadata.

    Uses ``exiftool`` (must be installed on the system). Gracefully returns a
    dict of ``None`` values when exiftool is absent or the image lacks DJI tags.

    Returned keys match the SRT-derived drone fields stored on video clips:
      ``drone_lat``, ``drone_lng``, ``drone_alt_rel``, ``drone_alt_abs``,
      ``gimbal_yaw``, ``gimbal_pitch``, ``gimbal_roll``,
      ``focal_len``, ``dzoom``, ``capture_timestamp``

    Extra DJI flight fields (XMP-only, not in video schema):
      ``flight_yaw``, ``flight_pitch``, ``flight_roll``,
      ``flight_x_speed``, ``flight_y_speed``, ``flight_z_speed``,
      ``rtk_flag``, ``gps_status``
    """
    empty: Dict[str, Any] = {
        "drone_lat": None,
        "drone_lng": None,
        "drone_alt_rel": None,
        "drone_alt_abs": None,
        "gimbal_yaw": None,
        "gimbal_pitch": None,
        "gimbal_roll": None,
        "focal_len": None,
        "dzoom": None,
        "capture_timestamp": None,
        "flight_yaw": None,
        "flight_pitch": None,
        "flight_roll": None,
        "flight_x_speed": None,
        "flight_y_speed": None,
        "flight_z_speed": None,
        "rtk_flag": None,
        "gps_status": None,
    }

    tags = _run_exiftool(Path(image_path))
    if not tags:
        return empty

    result: Dict[str, Any] = {}

    # GPS (numeric with -n → already decimal degrees)
    result["drone_lat"] = _to_float(tags.get("GPSLatitude"))
    result["drone_lng"] = _to_float(tags.get("GPSLongitude"))

    if result["drone_lat"] is not None:
        lat_ref = tags.get("GPSLatitudeRef", "")
        if isinstance(lat_ref, str) and lat_ref.upper().startswith("S"):
            result["drone_lat"] = -abs(result["drone_lat"])

    if result["drone_lng"] is not None:
        lng_ref = tags.get("GPSLongitudeRef", "")
        if isinstance(lng_ref, str) and lng_ref.upper().startswith("W"):
            result["drone_lng"] = -abs(result["drone_lng"])

    # DJI XMP altitude (stored as signed strings like "+79.995")
    result["drone_alt_rel"] = _to_float(tags.get("RelativeAltitude"))
    result["drone_alt_abs"] = _to_float(tags.get("AbsoluteAltitude"))

    # Gimbal angles
    result["gimbal_yaw"] = _to_float(tags.get("GimbalYawDegree"))
    result["gimbal_pitch"] = _to_float(tags.get("GimbalPitchDegree"))
    result["gimbal_roll"] = _to_float(tags.get("GimbalRollDegree"))

    # Optics
    result["focal_len"] = _to_float(tags.get("FocalLength"))
    result["dzoom"] = _to_float(tags.get("DigitalZoomRatio"))

    # Capture timestamp (prefer UTC, fall back to local DateTimeOriginal)
    result["capture_timestamp"] = (
        _parse_exif_datetime(tags.get("UTCAtExposure"))
        or _parse_exif_datetime(tags.get("DateTimeOriginal"))
        or _parse_exif_datetime(tags.get("CreateDate"))
    )

    # Extra DJI flight fields
    result["flight_yaw"] = _to_float(tags.get("FlightYawDegree"))
    result["flight_pitch"] = _to_float(tags.get("FlightPitchDegree"))
    result["flight_roll"] = _to_float(tags.get("FlightRollDegree"))
    result["flight_x_speed"] = _to_float(tags.get("FlightXSpeed"))
    result["flight_y_speed"] = _to_float(tags.get("FlightYSpeed"))
    result["flight_z_speed"] = _to_float(tags.get("FlightZSpeed"))
    result["rtk_flag"] = tags.get("RtkFlag")
    result["gps_status"] = tags.get("GpsStatus") or tags.get("GPSStatus")

    for key in empty:
        result.setdefault(key, None)

    return result
