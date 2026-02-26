"""DJI SRT telemetry extraction and parsing.

Extracts an embedded SRT subtitle track from an MP4 file using ffprobe/ffmpeg,
then parses the DJI-format telemetry entries into a list of dicts indexed by
seconds offset from the start of the video.

Public API
----------
extract_srt_from_video(video_path)          -> Optional[str]
parse_srt_to_telemetry(srt_text)            -> List[Dict]
get_clip_telemetry(telemetry, start, end)   -> Dict
"""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum number of path-array points returned per clip
_MAX_PATH_POINTS = 10


# ---------------------------------------------------------------------------
# SRT extraction
# ---------------------------------------------------------------------------

def extract_srt_from_video(video_path: Path) -> Optional[str]:
    """Extract an embedded SRT subtitle stream from *video_path* using ffmpeg.

    Returns the raw SRT text, or ``None`` if no subtitle stream is found or
    if ffprobe/ffmpeg is unavailable.
    """
    video_path = Path(video_path)

    # 1. Probe for subtitle streams
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        str(video_path),
    ]
    try:
        probe = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        probe_data = json.loads(probe.stdout or "{}")
        streams = probe_data.get("streams", [])
        if not streams:
            logger.debug("No subtitle streams in %s", video_path.name)
            return None
        stream_index = streams[0].get("index", "s:0")
    except FileNotFoundError:
        logger.warning("ffprobe not found; SRT extraction skipped")
        return None
    except Exception as exc:
        logger.warning("ffprobe error for %s: %s", video_path.name, exc)
        return None

    # 2. Extract to a temp SRT file
    tmp_path = Path(tempfile.mktemp(suffix=".srt"))
    extract_cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        str(tmp_path),
        "-y",
    ]
    try:
        subprocess.run(extract_cmd, capture_output=True, timeout=120, check=True)
        srt_text = tmp_path.read_text(encoding="utf-8", errors="replace")
        return srt_text.strip() or None
    except FileNotFoundError:
        logger.warning("ffmpeg not found; SRT extraction skipped")
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg SRT extraction failed for %s: %s", video_path.name, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected SRT extraction error for %s: %s", video_path.name, exc)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def _parse_srt_timestamp(ts: str) -> float:
    """Convert ``HH:MM:SS,mmm`` (or ``.mmm``) to seconds (float)."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def _extract_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _extract_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _parse_dji_block(content: str) -> Dict[str, Any]:
    """Parse the content lines of one DJI telemetry SRT block.

    Example content::

        FrameCnt: 0 2025-12-25 15:40:25.273
        [iso: 150] [shutter: 1/1250.0] [fnum: 2.8] [ev: 0] ...
        [latitude: 18.563137] [longitude: 73.701113]
        [rel_alt: 100.001 abs_alt: 607.081]
        [gb_yaw: 80.8 gb_pitch: -46.8 gb_roll: 0.0]
    """
    result: Dict[str, Any] = {}

    # Frame counter
    m = re.search(r"FrameCnt:\s*(\d+)", content)
    if m:
        result["frame_cnt"] = int(m.group(1))

    # Wall-clock datetime embedded in the block
    m = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", content)
    if m:
        result["datetime_str"] = m.group(1).strip()

    result["latitude"] = _extract_float(r"\[latitude:\s*([-\d.]+)\]", content)
    result["longitude"] = _extract_float(r"\[longitude:\s*([-\d.]+)\]", content)
    result["rel_alt"] = _extract_float(r"\[rel_alt:\s*([-\d.]+)", content)
    result["abs_alt"] = _extract_float(r"abs_alt:\s*([-\d.]+)\]", content)
    result["gb_yaw"] = _extract_float(r"\[gb_yaw:\s*([-\d.]+)", content)
    result["gb_pitch"] = _extract_float(r"gb_pitch:\s*([-\d.]+)", content)
    result["gb_roll"] = _extract_float(r"gb_roll:\s*([-\d.]+)\]", content)
    result["focal_len"] = _extract_float(r"\[focal_len:\s*([-\d.]+)\]", content)
    result["dzoom"] = _extract_float(r"\[dzoom_ratio:\s*([-\d.]+)\]", content)
    result["iso"] = _extract_int(r"\[iso:\s*(\d+)\]", content)
    result["fnum"] = _extract_float(r"\[fnum:\s*([-\d.]+)\]", content)

    # Drop None values to keep dicts lean
    return {k: v for k, v in result.items() if v is not None}


def parse_srt_to_telemetry(srt_text: str) -> List[Dict[str, Any]]:
    """Parse raw SRT text into a list of telemetry dicts.

    Each dict includes:
    - ``seconds_offset``  – midpoint of the subtitle block (float)
    - ``start_seconds``   – block start in seconds
    - ``end_seconds``     – block end in seconds
    - All DJI telemetry fields extracted from the block content

    The returned list is sorted by ``seconds_offset``.
    """
    telemetry: List[Dict[str, Any]] = []

    # Split into subtitle blocks on blank lines
    blocks = re.split(r"\n\s*\n", srt_text.strip())

    for block in blocks:
        lines = [l.rstrip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        # Find the timing line (contains "-->")
        timing_line: Optional[str] = None
        content_start = 2  # lines[0]=index, lines[1]=timing, lines[2+]=content
        for idx, line in enumerate(lines):
            if "-->" in line:
                timing_line = line
                content_start = idx + 1
                break

        if timing_line is None:
            continue

        try:
            start_str, end_str = timing_line.split("-->")
            start_sec = _parse_srt_timestamp(start_str)
            end_sec = _parse_srt_timestamp(end_str)
        except Exception:
            continue

        content = "\n".join(lines[content_start:])
        entry = _parse_dji_block(content)
        entry["seconds_offset"] = (start_sec + end_sec) / 2.0
        entry["start_seconds"] = start_sec
        entry["end_seconds"] = end_sec
        telemetry.append(entry)

    telemetry.sort(key=lambda x: x["seconds_offset"])
    return telemetry


# ---------------------------------------------------------------------------
# Clip-level telemetry sampling
# ---------------------------------------------------------------------------

def sample_telemetry_at(
    telemetry: List[Dict[str, Any]],
    seconds: float,
) -> Optional[Dict[str, Any]]:
    """Return the telemetry entry whose ``seconds_offset`` is closest to *seconds*.

    Returns ``None`` if *telemetry* is empty.
    """
    if not telemetry:
        return None
    return min(telemetry, key=lambda t: abs(t["seconds_offset"] - seconds))


def get_clip_telemetry(
    telemetry: List[Dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> Dict[str, Any]:
    """Build a drone-telemetry dict for the clip ``[start_seconds, end_seconds]``.

    Returns a dict with:

    Single-point values (sampled at clip midpoint):
      ``drone_lat``, ``drone_lng``, ``drone_alt_rel``, ``drone_alt_abs``,
      ``gimbal_yaw``, ``gimbal_pitch``, ``gimbal_roll``,
      ``focal_len``, ``dzoom``

    Path arrays (all entries within the clip, downsampled to ≤ 10 points):
      ``path_lats``, ``path_lngs``, ``path_yaws``

    All values may be ``None`` when telemetry is absent for a field.
    """
    if not telemetry:
        return {
            "drone_lat": None, "drone_lng": None,
            "drone_alt_rel": None, "drone_alt_abs": None,
            "gimbal_yaw": None, "gimbal_pitch": None, "gimbal_roll": None,
            "focal_len": None, "dzoom": None,
            "path_lats": None, "path_lngs": None, "path_yaws": None,
        }

    mid_seconds = (start_seconds + end_seconds) / 2.0
    mid = sample_telemetry_at(telemetry, mid_seconds)

    # Entries within the clip window
    clip_entries = [
        t for t in telemetry
        if start_seconds <= t["seconds_offset"] <= end_seconds
    ]

    # Downsample path arrays to at most _MAX_PATH_POINTS
    if len(clip_entries) > _MAX_PATH_POINTS:
        step = max(1, len(clip_entries) // _MAX_PATH_POINTS)
        clip_entries = clip_entries[::step]

    result: Dict[str, Any] = {}

    # Single-point telemetry
    if mid:
        result["drone_lat"] = mid.get("latitude")
        result["drone_lng"] = mid.get("longitude")
        result["drone_alt_rel"] = mid.get("rel_alt")
        result["drone_alt_abs"] = mid.get("abs_alt")
        result["gimbal_yaw"] = mid.get("gb_yaw")
        result["gimbal_pitch"] = mid.get("gb_pitch")
        result["gimbal_roll"] = mid.get("gb_roll")
        result["focal_len"] = mid.get("focal_len")
        result["dzoom"] = mid.get("dzoom")
    else:
        for key in ("drone_lat", "drone_lng", "drone_alt_rel", "drone_alt_abs",
                    "gimbal_yaw", "gimbal_pitch", "gimbal_roll", "focal_len", "dzoom"):
            result[key] = None

    # Path arrays
    lats = [e["latitude"] for e in clip_entries if e.get("latitude") is not None]
    lngs = [e["longitude"] for e in clip_entries if e.get("longitude") is not None]
    yaws = [e["gb_yaw"] for e in clip_entries if e.get("gb_yaw") is not None]
    result["path_lats"] = lats or None
    result["path_lngs"] = lngs or None
    result["path_yaws"] = yaws or None

    return result
