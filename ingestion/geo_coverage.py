"""Geospatial coverage computation for drone flight telemetry.

Builds GeoJSON primitives (LineString, Polygon) from per-frame SRT telemetry
that has been parsed by srt_parser.py.

Key functions
-------------
calculate_fov_footprint  – project camera FOV onto the ground plane
build_coverage_polygon   – union of FOV footprints sampled at regular intervals
build_flight_path        – GeoJSON LineString from telemetry lat/lng
build_bounds             – GeoJSON Polygon (bounding box)
build_frames             – downsample telemetry to ~1 fps for video-map sync
"""

import math
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies (shapely + pyproj)
# ---------------------------------------------------------------------------
try:
    from shapely.geometry import Polygon, MultiPolygon, mapping
    from shapely.ops import unary_union
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False
    logger.warning("shapely not installed – FOV coverage polygon unavailable")

try:
    from pyproj import Transformer
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False
    logger.warning("pyproj not installed – UTM projection unavailable")


# ---------------------------------------------------------------------------
# FOV footprint
# ---------------------------------------------------------------------------

# DJI SRT reports focal_len as 35mm-equivalent.
# Sensor size for a full-frame (35mm) camera: 36 mm × 24 mm.
_SENSOR_HALF_W_MM = 18.0   # half of 36 mm
_SENSOR_HALF_H_MM = 12.0   # half of 24 mm

# Skip footprint when pitch is too near horizontal (rays won't hit ground)
_MIN_PITCH_DEG = -10.0


def _enu_camera_axes(yaw_deg: float, pitch_deg: float):
    """Return (forward, right, up) unit vectors in ENU frame.

    ENU convention: X=East, Y=North, Z=Up.
    yaw_deg  – compass bearing, clockwise from North (0=N, 90=E, …)
    pitch_deg – elevation angle from horizontal, negative = nose down
                (-90 = straight down / nadir)
    """
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Forward vector (direction camera is pointing)
    fwd_e = math.sin(yaw) * math.cos(pitch)
    fwd_n = math.cos(yaw) * math.cos(pitch)
    fwd_u = math.sin(pitch)
    forward = (fwd_e, fwd_n, fwd_u)

    # Right vector (horizontal, perpendicular to forward in yaw direction)
    right_e =  math.cos(yaw)
    right_n = -math.sin(yaw)
    right_u = 0.0
    right = (right_e, right_n, right_u)

    # Up vector (image top direction) = right × forward (re-orthogonalise)
    up_e = right_n * fwd_u - right_u * fwd_n
    up_n = right_u * fwd_e - right_e * fwd_u
    up_u = right_e * fwd_n - right_n * fwd_e
    # Normalise
    mag = math.sqrt(up_e**2 + up_n**2 + up_u**2) or 1.0
    up = (up_e / mag, up_n / mag, up_u / mag)

    return forward, right, up


def _ground_intersection(ray_e, ray_n, ray_u, alt_m: float):
    """Return (east_m, north_m) where ray from (0,0,alt_m) hits z=0.

    Returns None if the ray points upward or nearly horizontal.
    """
    if ray_u >= 0:
        return None  # points upward, no ground intersection
    t = alt_m / (-ray_u)
    return (ray_e * t, ray_n * t)


def calculate_fov_footprint(
    lat: float,
    lng: float,
    alt_rel: float,
    gimbal_pitch_deg: float,
    gimbal_yaw_deg: float,
    focal_len_mm: float,
    dzoom: float = 1.0,
) -> "Optional[Polygon]":
    """Project the camera FOV onto the ground and return a shapely Polygon.

    Uses full 3-D projective geometry in a local UTM frame, then converts
    corners back to WGS84.

    Parameters
    ----------
    lat, lng        – drone WGS84 position
    alt_rel         – drone altitude above ground (metres)
    gimbal_pitch_deg – gimbal pitch (0 = horizontal, -90 = nadir)
    gimbal_yaw_deg  – gimbal yaw compass bearing (0=N, 90=E …)
    focal_len_mm    – 35mm-equivalent focal length reported in DJI SRT
    dzoom           – digital zoom factor

    Returns None if:
    - shapely / pyproj not available
    - pitch is too near horizontal
    - any corner ray doesn't intersect the ground
    - altitude is ≤ 0
    """
    if not _SHAPELY_AVAILABLE or not _PYPROJ_AVAILABLE:
        return None

    if alt_rel <= 0:
        return None

    if gimbal_pitch_deg > _MIN_PITCH_DEG:
        # Near-horizontal; footprint would be at (near) infinity
        return None

    f_eff = focal_len_mm * dzoom
    if f_eff <= 0:
        return None

    # Half-angles
    half_hfov = math.atan(_SENSOR_HALF_W_MM / f_eff)
    half_vfov = math.atan(_SENSOR_HALF_H_MM / f_eff)

    tan_h = math.tan(half_hfov)
    tan_v = math.tan(half_vfov)

    forward, right, up = _enu_camera_axes(gimbal_yaw_deg, gimbal_pitch_deg)

    # Four image corners (dx, dy) relative to image centre
    corners_offsets = [
        (-1, -1),  # bottom-left
        ( 1, -1),  # bottom-right
        ( 1,  1),  # top-right
        (-1,  1),  # top-left
    ]

    ground_pts_enu = []
    for dx, dy in corners_offsets:
        ray_e = forward[0] + dx * tan_h * right[0] + dy * tan_v * up[0]
        ray_n = forward[1] + dx * tan_h * right[1] + dy * tan_v * up[1]
        ray_u = forward[2] + dx * tan_h * right[2] + dy * tan_v * up[2]

        pt = _ground_intersection(ray_e, ray_n, ray_u, alt_rel)
        if pt is None:
            return None  # at least one corner doesn't reach the ground
        ground_pts_enu.append(pt)

    # Convert ENU offsets (metres) to WGS84 via local UTM
    try:
        # UTM zone string, e.g. "+proj=utm +zone=43 +datum=WGS84"
        zone = int((lng + 180) / 6) + 1
        hemisphere = "south" if lat < 0 else "north"
        utm_crs = f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"

        to_utm   = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        from_utm = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        drone_e, drone_n = to_utm.transform(lng, lat)

        wgs84_corners = []
        for de, dn in ground_pts_enu:
            corner_e = drone_e + de
            corner_n = drone_n + dn
            corner_lng, corner_lat = from_utm.transform(corner_e, corner_n)
            wgs84_corners.append((corner_lng, corner_lat))  # GeoJSON order: lng, lat

        return Polygon(wgs84_corners)

    except Exception as exc:
        logger.debug("FOV footprint projection failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Telemetry → GeoJSON helpers
# ---------------------------------------------------------------------------

def build_flight_path(telemetry: List[Dict]) -> Optional[Dict]:
    """Return a GeoJSON LineString from telemetry lat/lng fields.

    Returns None if fewer than 2 valid points are found.
    """
    coords = []
    for entry in telemetry:
        lat = entry.get("latitude")
        lng = entry.get("longitude")
        if lat is not None and lng is not None:
            coords.append([lng, lat])  # GeoJSON: [lng, lat]

    if len(coords) < 2:
        return None

    return {"type": "LineString", "coordinates": coords}


def build_bounds(telemetry: List[Dict]) -> Optional[Dict]:
    """Return a GeoJSON Polygon bounding box from telemetry lat/lng."""
    lats, lngs = [], []
    for entry in telemetry:
        lat = entry.get("latitude")
        lng = entry.get("longitude")
        if lat is not None and lng is not None:
            lats.append(lat)
            lngs.append(lng)

    if len(lats) < 2:
        return None

    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)

    # Closed ring: 5 points (first == last)
    ring = [
        [min_lng, min_lat],
        [max_lng, min_lat],
        [max_lng, max_lat],
        [min_lng, max_lat],
        [min_lng, min_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def build_coverage_polygon(
    telemetry: List[Dict],
    step_seconds: float = 5.0,
) -> Optional[Dict]:
    """Union of camera FOV footprints sampled every *step_seconds*.

    Returns a GeoJSON Polygon dict, or None if shapely/pyproj are not
    available or insufficient telemetry has geodetic + camera data.
    """
    if not _SHAPELY_AVAILABLE or not _PYPROJ_AVAILABLE:
        return None

    if not telemetry:
        return None

    footprints = []
    last_sampled = -step_seconds  # ensure first entry is included

    # Sort by time so sampling is chronological
    sorted_telem = sorted(telemetry, key=lambda e: e.get("seconds_offset", 0))

    for entry in sorted_telem:
        ts = entry.get("seconds_offset", 0)
        if ts - last_sampled < step_seconds:
            continue

        lat = entry.get("latitude")
        lng = entry.get("longitude")
        alt_rel = entry.get("rel_alt")
        pitch = entry.get("gb_pitch")
        yaw = entry.get("gb_yaw")
        focal_len = entry.get("focal_len")

        if any(v is None for v in (lat, lng, alt_rel, pitch, yaw, focal_len)):
            continue

        dzoom = entry.get("dzoom", 1.0) or 1.0

        fp = calculate_fov_footprint(
            lat=lat,
            lng=lng,
            alt_rel=alt_rel,
            gimbal_pitch_deg=pitch,
            gimbal_yaw_deg=yaw,
            focal_len_mm=focal_len,
            dzoom=dzoom,
        )
        if fp is not None and fp.is_valid and not fp.is_empty:
            footprints.append(fp)
            last_sampled = ts

    if not footprints:
        return None

    try:
        union = unary_union(footprints)
        # Simplify to ~1 m tolerance (0.00001 deg ≈ 1.1 m)
        simplified = union.simplify(0.00001, preserve_topology=True)

        # Ensure we have a single Polygon (take convex hull of MultiPolygon)
        if isinstance(simplified, MultiPolygon):
            simplified = simplified.convex_hull

        if simplified.is_empty:
            return None

        return dict(mapping(simplified))

    except Exception as exc:
        logger.warning("Coverage polygon union failed: %s", exc)
        return None


def build_frames(telemetry: List[Dict]) -> List[Dict]:
    """Downsample SRT telemetry (~30 fps) to ~1 fps for video-map sync.

    Picks the first entry encountered in each integer-second bucket.
    Returns a list of dicts with camelCase keys matching the MongoDB schema.
    """
    if not telemetry:
        return []

    seen_seconds: set = set()
    frames = []

    sorted_telem = sorted(telemetry, key=lambda e: e.get("seconds_offset", 0))

    for entry in sorted_telem:
        ts = entry.get("seconds_offset", 0)
        bucket = int(ts)

        if bucket in seen_seconds:
            continue
        seen_seconds.add(bucket)

        lat = entry.get("latitude")
        lng = entry.get("longitude")
        if lat is None or lng is None:
            continue

        frame = {
            "ts":       round(ts, 3),
            "lat":      lat,
            "lng":      lng,
        }

        if entry.get("rel_alt") is not None:
            frame["altRel"] = entry["rel_alt"]
        if entry.get("abs_alt") is not None:
            frame["altAbs"] = entry["abs_alt"]
        if entry.get("gb_yaw") is not None:
            frame["yaw"] = entry["gb_yaw"]
        if entry.get("gb_pitch") is not None:
            frame["pitch"] = entry["gb_pitch"]
        if entry.get("gb_roll") is not None:
            frame["roll"] = entry["gb_roll"]
        if entry.get("focal_len") is not None:
            frame["focalLen"] = entry["focal_len"]
        if entry.get("dzoom") is not None:
            frame["dzoom"] = entry["dzoom"]

        frames.append(frame)

    return frames
