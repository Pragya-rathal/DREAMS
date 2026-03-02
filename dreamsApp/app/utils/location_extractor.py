"""Location extraction and semantic enrichment for DREAMS photo memories.

Pipeline:
    1. extract_gps_from_image()  — pull raw GPS + timestamp from EXIF
    2. reverse_geocode()         — coords → place metadata via OSM Nominatim
    3. format_location_text()    — metadata → semantic text (no geography)
    4. get_location_embedding()  — text → 384-dim vector (all-MiniLM-L6-v2)
    5. enrich_location()         — orchestrates steps 2-4 in one call
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_USER_AGENT = (
    "DREAMS-Research/1.0 (https://github.com/KathiraveluLab/DREAMS)"
)
_MIN_REQUEST_INTERVAL = 1.1  # Nominatim policy: max 1 request / second
_CACHE_PRECISION = 5         # decimal places ≈ 1 m resolution
_MAX_CACHE_SIZE = 10_000     # bounded LRU-style eviction (oldest first)

# City-like keys Nominatim may use, in priority order
_CITY_KEYS = ("city", "town", "village", "hamlet")


# ── Module-level state ──────────────────────────────────────────────────

_geocode_cache: Dict[tuple, dict] = {}
_last_request_time: float = 0.0
_nominatim_lock = threading.Lock()  # protects _geocode_cache + _last_request_time
_embedding_model = None


# ── Internal helpers ─────────────────────────────────────────────────────

def _get_embedding_model():
    """Lazily load the SentenceTransformer singleton (avoids import-time cost)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-V2")
    return _embedding_model


def _dms_to_decimal(dms_value) -> float:
    """Convert an EXIF GPS coordinate from DMS (degrees/minutes/seconds) to
    decimal degrees.

    Each component may be a plain number or a (numerator, denominator) tuple.
    """
    if not isinstance(dms_value, (tuple, list)) or len(dms_value) != 3:
        raise ValueError(f"Expected 3-element DMS sequence, got: {dms_value}")

    decimal = 0.0
    for idx, component in enumerate(dms_value):
        if isinstance(component, tuple):
            numerator, denominator = component
            if denominator == 0:
                raise ValueError(f"Zero denominator in DMS component: {component}")
            value = numerator / denominator
        else:
            value = float(component)
        # idx 0 → degrees (÷1), idx 1 → minutes (÷60), idx 2 → seconds (÷3600)
        decimal += value / (60 ** idx)

    return decimal


def _parse_gps_timestamp(gps_info: dict) -> Optional[str]:
    """Try to build an ISO-8601 timestamp from GPSDateStamp + GPSTimeStamp."""
    if "GPSDateStamp" not in gps_info or "GPSTimeStamp" not in gps_info:
        return None
    try:
        year, month, day = map(int, gps_info["GPSDateStamp"].split(":"))
        hours, minutes, seconds_raw = (float(p) for p in gps_info["GPSTimeStamp"])

        whole_seconds = int(seconds_raw)
        microseconds = int((seconds_raw - whole_seconds) * 1_000_000)

        dt_utc = datetime(
            year, month, day,
            int(hours), int(minutes), whole_seconds, microseconds,
            tzinfo=timezone.utc,
        )
        return dt_utc.isoformat()
    except (ValueError, TypeError, IndexError):
        logger.warning("Could not parse GPSDateStamp / GPSTimeStamp")
        return None


def _parse_exif_datetime(raw: str) -> Optional[str]:
    """Parse EXIF DateTimeOriginal string into ISO-8601 (no timezone)."""
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").isoformat()
    except (ValueError, TypeError):
        logger.warning("Could not parse EXIF DateTimeOriginal: '%s'", raw)
        return None


def _rate_limit() -> None:
    """Sleep if necessary to respect Nominatim's 1-request-per-second policy."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)


def _resolve_city(address_raw: dict) -> Optional[str]:
    """Pick the best 'city' field from a Nominatim address dict."""
    for key in _CITY_KEYS:
        if address_raw.get(key):
            return address_raw[key]
    return None


def _resolve_place_name(geocode_result: dict) -> Optional[str]:
    """Find the specific place name from either the address dict or the
    display_name string.

    Nominatim stores the place name under a key matching its OSM class:
        address["amenity"] = "St. Mary's Church"
        address["leisure"] = "Westchester Lagoon"
    """
    osm_class = geocode_result.get("osm_class", "")
    address = geocode_result.get("address", {})

    # Best case: keyed by OSM class
    if osm_class and address.get(osm_class):
        return address[osm_class]

    # Fallback: first segment of display_name (usually the place itself)
    display_name = geocode_result.get("display_name", "")
    if display_name:
        return display_name.split(",")[0].strip()

    return None


# ── Public API ───────────────────────────────────────────────────────────

def extract_gps_from_image(image_path: str) -> Optional[Dict[str, Any]]:
    """Extract GPS coordinates and timestamp from an image's EXIF metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        ``{"lat": float, "lon": float}`` with an optional ``"timestamp"``
        key, or ``None`` if no GPS data is available.
    """
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None

            # Scan EXIF tags for GPS block and capture timestamp
            gps_info = None
            datetime_original = None

            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    gps_info = {GPSTAGS.get(t, t): value[t] for t in value}
                elif tag_name == "DateTimeOriginal":
                    datetime_original = value
                if gps_info is not None and datetime_original is not None:
                    break

            if not gps_info:
                return None
            if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
                return None

            # Convert DMS → decimal degrees
            lat = _dms_to_decimal(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat

            lon = _dms_to_decimal(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon

            result: Dict[str, Any] = {"lat": lat, "lon": lon}

            # Prefer GPS timestamp (has timezone); fall back to DateTimeOriginal
            timestamp = _parse_gps_timestamp(gps_info)
            if not timestamp and datetime_original:
                timestamp = _parse_exif_datetime(datetime_original)
            if timestamp:
                result["timestamp"] = timestamp

            return result

    except (AttributeError, KeyError, IndexError, TypeError, ValueError, IOError) as exc:
        logger.error("Failed to extract GPS from '%s': %s", image_path, exc)
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Reverse-geocode GPS coordinates via the OSM Nominatim API.

    Features:
        - In-memory cache keyed at ~1 m precision (same spot → one API call)
        - Rate limiting (≥1.1 s between requests, per Nominatim policy)
        - Coordinate validation (rejects out-of-range values)

    Args:
        lat: Latitude  (must be in [-90, 90]).
        lon: Longitude (must be in [-180, 180]).

    Returns:
        Dict with keys ``display_name``, ``place_category``, ``place_type``,
        ``address``, ``osm_class``  — or ``None`` on any failure.
    """
    global _last_request_time

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logger.warning("Invalid coordinates: lat=%s, lon=%s", lat, lon)
        return None

    with _nominatim_lock:
        cache_key = (round(lat, _CACHE_PRECISION), round(lon, _CACHE_PRECISION))
        if cache_key in _geocode_cache:
            return _geocode_cache[cache_key]

        _rate_limit()

        try:
            response = requests.get(
                _NOMINATIM_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "accept-language": "en",
                },
                headers={"User-Agent": _NOMINATIM_USER_AGENT},
                timeout=10,
            )
            _last_request_time = time.time()
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning("Nominatim error: %s", data["error"])
                return None

            address_raw = data.get("address", {})
            osm_class = data.get("category", "")

            result = {
                "display_name": data.get("display_name", ""),
                "place_category": data.get("type", ""),
                "place_type": osm_class,
                "osm_class": osm_class,
                "address": {
                    "road": address_raw.get("road"),
                    "city": _resolve_city(address_raw),
                    "state": address_raw.get("state"),
                    "country": address_raw.get("country"),
                    "country_code": address_raw.get("country_code"),
                },
            }

            _geocode_cache[cache_key] = result
            if len(_geocode_cache) > _MAX_CACHE_SIZE:
                _geocode_cache.pop(next(iter(_geocode_cache)))
            return result

        except (requests.RequestException, ValueError, KeyError) as exc:
            _last_request_time = time.time()
            logger.error("Reverse geocoding failed for (%s, %s): %s", lat, lon, exc)
            return None


def format_location_text(
    geocode_result: Optional[Dict[str, Any]],
    lat: float,
    lon: float,
) -> str:
    """Build a semantic location string suitable for embedding.

    Design decision: the text contains **place-type semantics only** — no
    geographic identifiers (city, state, country).  Two churches in
    different cities should produce similar embeddings; geography is
    captured separately by the GEO edge in the proximity graph.

    Examples:
        ``"place of worship amenity St. Mary's Church"``
        ``"park leisure Westchester Lagoon"``
        ``"unknown location at coordinates 61.2181 -149.9003"``

    Args:
        geocode_result: Output of :func:`reverse_geocode`, or ``None``.
        lat: Latitude  (used only in the fallback string).
        lon: Longitude (used only in the fallback string).
    """
    fallback = f"unknown location at coordinates {lat} {lon}"

    if geocode_result is None:
        return fallback

    category = geocode_result.get("place_category", "").replace("_", " ")
    place_type = geocode_result.get("place_type", "").replace("_", " ")

    parts: List[str] = []
    if category:
        parts.append(category)
    if place_type and place_type != category:
        parts.append(place_type)

    place_name = _resolve_place_name(geocode_result)
    if place_name:
        parts.append(place_name)

    return " ".join(parts) if parts else fallback


def get_location_embedding(
    location_text: str,
    model: Any = None,
) -> List[float]:
    """Encode location text into a 384-dimensional semantic vector.

    Args:
        location_text: Descriptive string from :func:`format_location_text`.
        model: Optional pre-loaded ``SentenceTransformer`` instance.
               If ``None``, a module-level singleton is loaded lazily.

    Returns:
        List of 384 floats.
    """
    if model is None:
        model = _get_embedding_model()
    return model.encode(location_text).tolist()


def enrich_location(
    lat: float,
    lon: float,
    model: Any = None,
) -> Dict[str, Any]:
    """Run the full enrichment pipeline: geocode → text → embedding.

    Always returns a dict (never ``None``).  If geocoding fails the
    embedding is still computed from a coordinate-based fallback string.

    Args:
        lat: Latitude.
        lon: Longitude.
        model: Optional pre-loaded ``SentenceTransformer``.

    Returns:
        Dict with at least ``location_text`` and ``location_embedding``.
        On successful geocoding also includes ``display_name``,
        ``place_category``, ``place_type``, and ``address``.
    """
    geocode = reverse_geocode(lat, lon)
    text = format_location_text(geocode, lat, lon)
    embedding = get_location_embedding(text, model=model)

    enrichment: Dict[str, Any] = {
        "location_text": text,
        "location_embedding": embedding,
    }

    if geocode:
        enrichment["display_name"] = geocode["display_name"]
        enrichment["place_category"] = geocode["place_category"]
        enrichment["place_type"] = geocode["place_type"]
        enrichment["address"] = geocode["address"]

    return enrichment
