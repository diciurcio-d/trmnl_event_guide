#!/usr/bin/env python3
"""Simple Flask server for Venue Scout with distance filtering."""

import json
import sys
import importlib.util
import re
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.distance import get_travel_time, get_travel_times_batch
from venue_scout.observability import increment, log_event, record_failure, snapshot
from venue_scout.paths import TRAVEL_CACHE_FILE, ensure_data_dir

app = Flask(__name__, static_folder='.')
CORS(app)

# Rate limiting — protects against abuse and runaway API costs.
# Storage is in-memory: limits reset on container restart and are not shared
# across instances, but good enough for a hobbyist app.
# Expensive routes (LLM, scraping, Maps API) get tighter per-route limits below.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per hour"],
    storage_uri="memory://",
)

# Persistent cache for travel times
CACHE_FILE = TRAVEL_CACHE_FILE


def _load_cache() -> dict:
    """Load travel time cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: dict):
    """Save travel time cache to disk."""
    ensure_data_dir()
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


_travel_cache = _load_cache()
_TZ = ZoneInfo("America/New_York")
_origin_coord_cache: dict[str, tuple[float, float] | None] = {}
_venue_coord_lookup_cache: dict[str, dict[str, tuple[float, float]]] | None = None
_DEFAULT_ORIGIN_ADDRESS = "167 W 74th St, New York, NY 10023"
_DEFAULT_ORIGIN_COORDS = (40.780602, -73.983528)


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


def _load_config() -> dict:
    """Load local runtime config."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _to_float_or_none(value) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_lookup_text(value) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _origin_geocode_cache_key(address: str) -> str:
    """Stable cache key for geocoded origin addresses."""
    return f"geocode|origin|{_normalize_lookup_text(address)}"


def _fallback_origin_coords_for_address(address: str) -> tuple[float, float] | None:
    """Return known fallback coordinates for default configured addresses."""
    if _normalize_lookup_text(address) == _normalize_lookup_text(_DEFAULT_ORIGIN_ADDRESS):
        return _DEFAULT_ORIGIN_COORDS
    return None


def _load_venue_coord_lookup(force_reload: bool = False) -> dict[str, dict[str, tuple[float, float]]]:
    """Load venue coordinate indexes keyed by normalized name and address."""
    global _venue_coord_lookup_cache
    if _venue_coord_lookup_cache is not None and not force_reload:
        return _venue_coord_lookup_cache

    by_name: dict[str, tuple[float, float]] = {}
    by_address: dict[str, tuple[float, float]] = {}

    try:
        from venue_scout.cache import read_cached_venues
        venues = read_cached_venues()
    except Exception:
        venues = []

    for venue in venues:
        lat = _to_float_or_none(venue.get("lat"))
        lng = _to_float_or_none(venue.get("lng"))
        if lat is None or lng is None:
            continue

        name_key = _normalize_lookup_text(venue.get("name", ""))
        address_key = _normalize_lookup_text(venue.get("address", ""))

        if name_key and name_key not in by_name:
            by_name[name_key] = (lat, lng)
        if address_key and address_key not in by_address:
            by_address[address_key] = (lat, lng)

    _venue_coord_lookup_cache = {
        "by_name": by_name,
        "by_address": by_address,
    }
    return _venue_coord_lookup_cache


def _event_destination_coords(event: dict) -> tuple[float, float] | None:
    """Resolve destination coordinates for an event from row or venue cache."""
    lat = _to_float_or_none(event.get("lat"))
    lng = _to_float_or_none(event.get("lng"))
    if lat is not None and lng is not None:
        return lat, lng

    lookup = _load_venue_coord_lookup()
    address_key = _normalize_lookup_text(event.get("address", ""))
    if address_key:
        coords = lookup["by_address"].get(address_key)
        if coords:
            return coords

    name_key = _normalize_lookup_text(event.get("venue_name", ""))
    if name_key:
        coords = lookup["by_name"].get(name_key)
        if coords:
            return coords

    return None


def _geocode_address(address: str) -> tuple[float, float] | None:
    """Geocode an address to lat/lng using Google Geocoding API."""
    normalized = address.strip()
    if not normalized:
        return None
    cache_key = _origin_geocode_cache_key(normalized)

    # Persistent cache on disk (shared with travel cache file).
    cached_row = _travel_cache.get(cache_key)
    if isinstance(cached_row, dict):
        cached_lat = _to_float_or_none(cached_row.get("lat"))
        cached_lng = _to_float_or_none(cached_row.get("lng"))
        if cached_lat is not None and cached_lng is not None:
            coords = (cached_lat, cached_lng)
            _origin_coord_cache[normalized] = coords
            return coords

    in_memory_cached = _origin_coord_cache.get(normalized)
    if in_memory_cached is not None:
        return in_memory_cached

    config = _load_config()
    api_key = str(config.get("google_cloud", {}).get("api_key", "")).strip()
    if not api_key:
        fallback = _fallback_origin_coords_for_address(normalized)
        if fallback is not None:
            _origin_coord_cache[normalized] = fallback
        return fallback

    settings = _load_settings()
    timeout = int(getattr(settings, "VENUE_ENRICH_PLACES_TIMEOUT_SEC", 10))

    coords = None

    # Primary geocoding path: Geocoding API.
    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": normalized, "key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "OK":
            first = (payload.get("results") or [{}])[0]
            location = first.get("geometry", {}).get("location", {})
            lat = _to_float_or_none(location.get("lat"))
            lng = _to_float_or_none(location.get("lng"))
            if lat is not None and lng is not None:
                coords = (lat, lng)
    except Exception:
        coords = None

    # Fallback path: Places API Text Search.
    if coords is None:
        try:
            response = requests.post(
                "https://places.googleapis.com/v1/places:searchText",
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": "places.location",
                },
                json={
                    "textQuery": normalized,
                    "maxResultCount": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            first_place = (payload.get("places") or [{}])[0]
            location = first_place.get("location", {})
            lat = _to_float_or_none(location.get("latitude"))
            lng = _to_float_or_none(location.get("longitude"))
            if lat is not None and lng is not None:
                coords = (lat, lng)
        except Exception:
            coords = None

    if coords is None:
        fallback = _fallback_origin_coords_for_address(normalized)
        if fallback is not None:
            _origin_coord_cache[normalized] = fallback
        return fallback

    _origin_coord_cache[normalized] = coords
    if coords is not None:
        _travel_cache[cache_key] = {"lat": coords[0], "lng": coords[1]}
        _save_cache(_travel_cache)
    return coords


def _resolve_origin_coords(filters: dict) -> tuple[float, float] | None:
    """Resolve origin coordinates from filters or config."""
    filter_lat = _to_float_or_none(filters.get("origin_lat"))
    filter_lng = _to_float_or_none(filters.get("origin_lng"))
    if filter_lat is not None and filter_lng is not None:
        return filter_lat, filter_lng

    filter_address = str(filters.get("origin_address", "")).strip()
    if filter_address:
        # Do not silently fall back to home coordinates when user provided
        # a specific origin and geocoding failed.
        return _geocode_address(filter_address)

    config = _load_config()
    user_cfg = config.get("user", {}) if isinstance(config, dict) else {}
    cfg_lat = _to_float_or_none(user_cfg.get("home_lat"))
    cfg_lng = _to_float_or_none(user_cfg.get("home_lng"))
    if cfg_lat is not None and cfg_lng is not None:
        return cfg_lat, cfg_lng

    home_address = str(user_cfg.get("home_address", "")).strip()
    return _geocode_address(home_address) if home_address else None


def _sanitize_targomo_error(exc: Exception) -> str:
    """Return a user-safe Targomo error message without sensitive request details."""
    if isinstance(exc, RuntimeError):
        return str(exc)
    return "Targomo request failed"


def _parse_targomo_times(payload: dict) -> dict[str, int]:
    """Extract target-id -> travel minutes from Targomo response payload."""
    times: dict[str, int] = {}
    data = payload.get("data")
    if isinstance(data, list):
        for source in data:
            targets = source.get("targets")
            if not isinstance(targets, list):
                continue
            for target in targets:
                target_id = str(target.get("id", "")).strip()
                seconds = _to_int_or_none(target.get("travelTime"))
                if target_id and seconds is not None:
                    times[target_id] = max(1, int(round(seconds / 60)))

    if times:
        return times

    targets = payload.get("targets")
    if isinstance(targets, list):
        for target in targets:
            target_id = str(target.get("id", "")).strip()
            seconds = _to_int_or_none(target.get("travelTime"))
            if target_id and seconds is not None:
                times[target_id] = max(1, int(round(seconds / 60)))
    return times


def _targomo_travel_times(
    origin: tuple[float, float],
    targets: list[dict],
    mode: str,
    max_minutes: int,
) -> dict[str, int]:
    """Fetch one-to-many travel times from Targomo."""
    config = _load_config()
    targomo_cfg = config.get("targomo", {}) if isinstance(config, dict) else {}
    api_key = str(targomo_cfg.get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError("Targomo API key is not configured")

    region = str(targomo_cfg.get("region", "northamerica")).strip() or "northamerica"
    base_url = str(targomo_cfg.get("base_url", "https://api.targomo.com")).rstrip("/")
    max_edge_weight = int(max(1800, (max_minutes + 30) * 60))
    endpoint = f"{base_url}/{region}/v1/time"
    travel_type = "walk" if mode == "walking" else "transit"

    payload = {
        "edgeWeight": "time",
        "maxEdgeWeight": max_edge_weight,
        "travelType": travel_type,
        "sources": [{"id": "origin", "lat": origin[0], "lng": origin[1]}],
        "targets": targets,
    }

    settings = _load_settings()
    timeout = int(getattr(settings, "GOOGLE_DISTANCE_MATRIX_BATCH_TIMEOUT_SEC", 20))

    try:
        response = requests.post(
            endpoint,
            params={"key": api_key},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Targomo request failed (network): {exc.__class__.__name__}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Targomo request failed ({response.status_code})")

    parsed = response.json()
    if str(parsed.get("code", "")).lower() not in ("", "ok"):
        message = str(parsed.get("message", "Unknown Targomo error")).strip() or "Unknown Targomo error"
        raise RuntimeError(f"Targomo error: {message}")

    return _parse_targomo_times(parsed)


def _travel_cache_key(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
) -> str:
    if mode == "transit":
        bucket = datetime.now(_TZ).strftime("%Y-%m-%dT%H")
    else:
        bucket = "static"
    return (
        f"targomo|{mode}|"
        f"{origin[0]:.5f},{origin[1]:.5f}|{destination[0]:.5f},{destination[1]:.5f}|{bucket}"
    )


def _filter_and_enrich_by_distance(
    matches: list[dict],
    filters: dict,
    mode: str,
    max_minutes: int,
) -> tuple[list[dict], dict, str]:
    """
    Estimate travel time for LLM matches, then apply distance filter.
    Returns (kept_events, metadata, warning).
    """
    origin = _resolve_origin_coords(filters)
    if origin is None:
        return matches, {"mode": mode, "applied": False}, "Distance filter skipped (origin location unavailable)."

    to_lookup: list[dict] = []
    per_index_minutes: dict[int, int | None] = {}
    cache_updates = 0

    for idx, event in enumerate(matches):
        coords = _event_destination_coords(event)
        if coords is None:
            per_index_minutes[idx] = None
            continue
        key = _travel_cache_key(origin, coords, mode)
        cached = _travel_cache.get(key)
        if isinstance(cached, dict) and _to_int_or_none(cached.get("minutes")) is not None:
            per_index_minutes[idx] = int(cached["minutes"])
            continue
        to_lookup.append(
            {
                "index": idx,
                "id": str(idx),
                "lat": coords[0],
                "lng": coords[1],
                "cache_key": key,
            }
        )

    warning = ""
    if to_lookup:
        batch_size = 80
        try:
            for start in range(0, len(to_lookup), batch_size):
                batch = to_lookup[start:start + batch_size]
                targets = [{"id": row["id"], "lat": row["lat"], "lng": row["lng"]} for row in batch]
                response_times = _targomo_travel_times(origin, targets, mode=mode, max_minutes=max_minutes)
                for row in batch:
                    minutes = response_times.get(row["id"])
                    per_index_minutes[row["index"]] = minutes
                    if minutes is not None:
                        _travel_cache[row["cache_key"]] = {"minutes": minutes, "text": f"{minutes} min"}
                        cache_updates += 1
        except Exception as exc:
            warning = (
                f"Distance filter unavailable ({_sanitize_targomo_error(exc)}). "
                "Showing unfiltered ranked results."
            )
            enriched = []
            for idx, event in enumerate(matches):
                row = dict(event)
                row["travel_minutes"] = per_index_minutes.get(idx)
                enriched.append(row)
            return enriched, {"mode": mode, "applied": False}, warning

    if cache_updates:
        _save_cache(_travel_cache)

    kept: list[dict] = []
    dropped_missing = 0
    dropped_over = 0
    resolved = 0

    for idx, event in enumerate(matches):
        minutes = per_index_minutes.get(idx)
        row = dict(event)
        row["travel_minutes"] = minutes
        if minutes is None:
            dropped_missing += 1
            continue
        resolved += 1
        if minutes <= max_minutes:
            kept.append(row)
        else:
            dropped_over += 1

    meta = {
        "mode": mode,
        "applied": True,
        "max_minutes": max_minutes,
        "input_matches": len(matches),
        "resolved_minutes": resolved,
        "kept": len(kept),
        "dropped_missing": dropped_missing,
        "dropped_over_limit": dropped_over,
    }
    return kept, meta, warning


def _serialize_events(events: list[dict]) -> list[dict]:
    """Convert datetime fields to JSON-safe strings."""
    events_data = []
    for event in events:
        event_dict = dict(event)
        dt_value = event_dict.get("datetime")
        if dt_value:
            if hasattr(dt_value, "isoformat"):
                event_dict["datetime"] = dt_value.isoformat()
            else:
                event_dict["datetime"] = str(dt_value)
        events_data.append(event_dict)
    return events_data


def _parse_event_datetime(event: dict) -> datetime | None:
    """Best-effort parser for event datetime/date fields."""
    dt_value = event.get("datetime")
    if isinstance(dt_value, datetime):
        return dt_value if dt_value.tzinfo else dt_value.replace(tzinfo=_TZ)

    if isinstance(dt_value, str) and dt_value and dt_value != "None":
        try:
            parsed = datetime.fromisoformat(dt_value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=_TZ)
        except ValueError:
            pass

    date_str = event.get("date_str", "")
    if isinstance(date_str, str) and date_str:
        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed_date.replace(tzinfo=_TZ)
        except ValueError:
            return None

    return None


def _normalize_category(value) -> str:
    return str(value or "").strip().lower()


def _to_str_list(value) -> list[str]:
    """Parse a JSON value into a list of non-empty strings."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        items = value
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _category_similarity(term: str, category: str) -> int:
    """Return fuzzy similarity score between two category-like strings."""
    term_n = _normalize_category(term)
    category_n = _normalize_category(category)
    if not term_n or not category_n:
        return 0
    if term_n in category_n or category_n in term_n:
        return 100
    return int(round(SequenceMatcher(None, term_n, category_n).ratio() * 100))


def _to_int_or_none(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _apply_event_filters(
    events: list[dict],
    filters: dict,
    *,
    apply_distance_filter: bool = True,
) -> tuple[list[dict], dict]:
    """Apply structured filters before natural-language ranking."""
    filtered = list(events)
    applied: dict = {}

    raw_categories = _to_str_list(filters.get("categories"))
    categories = {_normalize_category(c) for c in raw_categories if _normalize_category(c)}
    if categories:
        filtered = [
            event for event in filtered
            if _normalize_category(event.get("event_type", "")) in categories
        ]
        applied["categories"] = sorted(categories)

    enable_fuzzy_categories = bool(filters.get("enable_fuzzy_categories", False))
    fuzzy_categories = _to_str_list(filters.get("fuzzy_categories") or filters.get("fuzzy_category_terms"))
    fuzzy_threshold = _to_int_or_none(filters.get("fuzzy_category_threshold"))
    if fuzzy_threshold is None:
        fuzzy_threshold = 72
    fuzzy_threshold = max(0, min(100, fuzzy_threshold))

    if enable_fuzzy_categories and fuzzy_categories:
        fuzzy_filtered = []
        for event in filtered:
            event_category = event.get("event_type", "")
            if not str(event_category or "").strip():
                continue
            best_term = ""
            best_score = 0
            for term in fuzzy_categories:
                score = _category_similarity(term, event_category)
                if score > best_score:
                    best_score = score
                    best_term = term
            if best_score >= fuzzy_threshold:
                event_copy = dict(event)
                event_copy["_fuzzy_category_score"] = best_score
                event_copy["_fuzzy_category_term"] = best_term
                fuzzy_filtered.append(event_copy)
        filtered = fuzzy_filtered
        applied["fuzzy_categories"] = fuzzy_categories
        applied["fuzzy_category_threshold"] = fuzzy_threshold

    if apply_distance_filter:
        max_travel_raw = filters.get("max_travel_minutes")
        include_unknown_distance = bool(filters.get("include_unknown_distance", True))
        max_travel = _to_int_or_none(max_travel_raw)

        if max_travel is not None and max_travel >= 0:
            def within_distance(event: dict) -> bool:
                travel = event.get("travel_minutes")
                if travel in (None, "", "None"):
                    return include_unknown_distance
                try:
                    return int(travel) <= max_travel
                except (TypeError, ValueError):
                    return include_unknown_distance

            filtered = [event for event in filtered if within_distance(event)]
            applied["max_travel_minutes"] = max_travel
            applied["include_unknown_distance"] = include_unknown_distance

    days_ahead = _to_int_or_none(filters.get("days_ahead"))

    if days_ahead is not None and days_ahead > 0:
        now = datetime.now(_TZ)
        cutoff = now + timedelta(days=days_ahead)
        in_window = []
        for event in filtered:
            dt = _parse_event_datetime(event)
            if dt and now <= dt <= cutoff:
                in_window.append(event)
        filtered = in_window
        applied["days_ahead"] = days_ahead

    return filtered, applied


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/travel-times', methods=['POST'])
@limiter.limit("60 per hour")  # Google Maps API — cached but still costs on cache miss
def calculate_travel_times():
    """
    Calculate travel times from origin to multiple destinations.
    Uses batch API calls for efficiency (25 destinations per API call).

    Request body:
    {
        "origin": "123 Main St, New York, NY",
        "destinations": [
            {"name": "Venue 1", "address": "456 Broadway, New York, NY"},
            ...
        ],
        "mode": "transit"  // transit, driving, walking
    }

    Response:
    {
        "results": [
            {"name": "Venue 1", "minutes": 25, "text": "25 mins"},
            ...
        ]
    }
    """
    increment("server.api.travel_times.calls")
    data = request.json
    origin = data.get('origin', '')
    destinations = data.get('destinations', [])
    mode = data.get('mode', 'transit')

    if not origin:
        return jsonify({"error": "Origin address required"}), 400

    results = []
    uncached = []  # (index, name, address) for venues not in cache

    # Addresses that are too vague to be useful
    BAD_ADDRESSES = {'nyc', 'new york', 'brooklyn', 'manhattan', 'queens', 'bronx', 'staten island', ''}

    # First pass: check cache
    for i, dest in enumerate(destinations):
        name = dest.get('name', '')
        address = dest.get('address', '').strip()

        # Skip vague addresses
        if address.lower().replace(',', '').strip() in BAD_ADDRESSES:
            results.append({"name": name, "minutes": None, "text": "No address"})
            continue

        cache_key = f"{origin}|{address}|{mode}"
        if cache_key in _travel_cache:
            cached = _travel_cache[cache_key]
            results.append({"name": name, "minutes": cached["minutes"], "text": cached["text"]})
        else:
            results.append(None)  # Placeholder
            uncached.append((i, name, address))

    # Batch API call for uncached venues
    if uncached:
        addresses = [addr for _, _, addr in uncached]
        batch_results = get_travel_times_batch(origin, addresses, mode)

        for (i, name, address), travel in zip(uncached, batch_results):
            cache_key = f"{origin}|{address}|{mode}"

            if travel["status"] == "OK":
                minutes = travel["duration_minutes"]
                text = travel["duration_text"]
                _travel_cache[cache_key] = {"minutes": minutes, "text": text}
                results[i] = {"name": name, "minutes": minutes, "text": text}
            else:
                results[i] = {"name": name, "minutes": None, "text": "Error"}

        # Save cache after batch
        _save_cache(_travel_cache)

    return jsonify({"results": results})


@app.route('/api/travel-time', methods=['GET'])
@limiter.limit("60 per hour")  # Google Maps API
def calculate_single_travel_time():
    """Calculate travel time for a single destination."""
    increment("server.api.travel_time.calls")
    origin = request.args.get('origin', '')
    destination = request.args.get('destination', '')
    mode = request.args.get('mode', 'transit')

    if not origin or not destination:
        return jsonify({"error": "Origin and destination required"}), 400

    cache_key = f"{origin}|{destination}|{mode}"
    if cache_key in _travel_cache:
        return jsonify(_travel_cache[cache_key])

    travel = get_travel_time(origin, destination, mode)

    if travel["status"] == "OK":
        result = {
            "minutes": travel["duration_minutes"],
            "text": travel["duration_text"],
            "distance": travel["distance_text"],
        }
        _travel_cache[cache_key] = result
        _save_cache(_travel_cache)
        return jsonify(result)
    else:
        return jsonify({"error": travel.get("error", "Unknown error")}), 400


@app.route('/api/geocode-origin', methods=['POST'])
def geocode_origin():
    """Validate and geocode a user-provided origin address."""
    increment("server.api.geocode_origin.calls")
    data = request.json or {}
    address = str(data.get("address", "") or "").strip()
    if not address:
        return jsonify({"valid": False, "error": "Origin address is required."}), 400

    try:
        coords = _geocode_address(address)
        if coords is None:
            return jsonify(
                {
                    "valid": False,
                    "error": "Address could not be geocoded. Please enter a valid address.",
                }
            ), 400
        return jsonify({"valid": True, "lat": coords[0], "lng": coords[1]})
    except Exception:
        return jsonify(
            {
                "valid": False,
                "error": "Address validation failed. Please try again.",
            }
        ), 500


@app.route('/api/preview-cost', methods=['POST'])
def preview_cost():
    """Preview how many API calls will be made (for cost estimation)."""
    increment("server.api.preview_cost.calls")
    data = request.json
    origin = data.get('origin', '')
    destinations = data.get('destinations', [])
    mode = data.get('mode', 'transit')

    BAD_ADDRESSES = {'nyc', 'new york', 'brooklyn', 'manhattan', 'queens', 'bronx', 'staten island', ''}

    cached = 0
    uncached = 0
    skipped = 0

    for dest in destinations:
        address = dest.get('address', '').strip()

        if address.lower().replace(',', '').strip() in BAD_ADDRESSES:
            skipped += 1
            continue

        cache_key = f"{origin}|{address}|{mode}"
        if cache_key in _travel_cache:
            cached += 1
        else:
            uncached += 1

    # Cost: $5 per 1000 elements
    cost = (uncached / 1000) * 5

    return jsonify({
        "total": len(destinations),
        "cached": cached,
        "uncached": uncached,
        "skipped": skipped,
        "estimated_cost": f"${cost:.3f}",
    })


@app.route('/api/cache-stats')
def cache_stats():
    """Get cache statistics."""
    increment("server.api.cache_stats.calls")
    # Group by origin
    origins = {}
    for key in _travel_cache:
        origin = key.split('|')[0]
        origins[origin] = origins.get(origin, 0) + 1

    return jsonify({
        "total_cached": len(_travel_cache),
        "by_origin": origins,
    })


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the travel time cache."""
    increment("server.api.clear_cache.calls")
    global _travel_cache
    _travel_cache = {}
    _save_cache(_travel_cache)
    return jsonify({"status": "ok", "message": "Cache cleared"})


# ============================================================================
# Event Fetching Endpoints
# ============================================================================

@app.route('/api/fetch-events', methods=['POST'])
@limiter.limit("15 per hour")  # Playwright + LLM — most expensive endpoint
def fetch_events():
    """
    Fetch events for selected venues (batch).
    For progress updates, use /api/fetch-venue instead.
    """
    increment("server.api.fetch_events.calls")
    from venue_scout.event_fetcher import fetch_events_for_venues

    data = request.json
    venues = data.get('venues', [])
    force_refresh = data.get('force_refresh', False)
    city = data.get('city', 'NYC')

    if not venues:
        record_failure("server.fetch_events", "no_venues_provided")
        return jsonify({"error": "No venues provided"}), 400

    try:
        log_event("api_fetch_events_start", venue_count=len(venues), city=city, force_refresh=force_refresh)
        results = fetch_events_for_venues(
            venues=venues,
            force_refresh=force_refresh,
            city=city,
            save_to_sheet=True,
        )

        # Convert FetchResult to JSON-serializable format
        response_results = {}
        total_events = 0
        for venue_name, result in results.items():
            events_data = _serialize_events(result.events)

            response_results[venue_name] = {
                "events": events_data,
                "source_used": result.source_used,
                "error": result.error,
                "skipped": result.skipped,
                "attempted_sources": result.attempted_sources,
                "source_errors": result.source_errors,
                "warnings": result.warnings,
            }
            total_events += len(events_data)

        log_event("api_fetch_events_done", venue_count=len(results), total_events=total_events, city=city)
        return jsonify({
            "results": response_results,
            "total_events": total_events,
        })

    except Exception as e:
        record_failure("server.fetch_events", str(e), city=city)
        return jsonify({"error": str(e)}), 500


@app.route('/api/fetch-venue', methods=['POST'])
@limiter.limit("15 per hour")  # Playwright + LLM
def fetch_single_venue():
    """
    Fetch events for a single venue. Use this for progress updates.

    Request body:
    {
        "venue": {"name": "Beacon Theatre", "category": "concert halls", ...},
        "force_refresh": false,
        "city": "NYC"
    }

    Response:
    {
        "venue_name": "Beacon Theatre",
        "events": [...],
        "source_used": "ticketmaster",
        "event_count": 15,
        "skipped": false,
        "error": null
    }
    """
    increment("server.api.fetch_venue.calls")
    from venue_scout.event_fetcher import fetch_venue_events
    from venue_scout.venue_events_sheet import append_venue_events

    data = request.json
    venue = data.get('venue', {})
    force_refresh = data.get('force_refresh', False)
    city = data.get('city', 'NYC')

    if not venue or not venue.get('name'):
        record_failure("server.fetch_venue", "no_venue_provided")
        return jsonify({"error": "No venue provided"}), 400

    try:
        log_event(
            "api_fetch_venue_start",
            venue_name=venue.get("name", ""),
            city=city,
            force_refresh=force_refresh,
        )
        result = fetch_venue_events(
            venue=venue,
            force_refresh=force_refresh,
            city=city,
        )

        # Save to sheet if we got events
        if result.events:
            append_venue_events(result.events, result.venue_name)

        # Convert events to JSON-serializable format
        events_data = _serialize_events(result.events)

        if result.warnings:
            log_event(
                "api_fetch_venue_warning",
                venue_name=result.venue_name,
                warnings=result.warnings,
                source_errors=result.source_errors,
            )
        return jsonify({
            "venue_name": result.venue_name,
            "events": events_data,
            "source_used": result.source_used,
            "event_count": len(events_data),
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "error": result.error,
            "attempted_sources": result.attempted_sources,
            "source_errors": result.source_errors,
            "warnings": result.warnings,
        })

    except Exception as e:
        record_failure("server.fetch_venue", str(e), venue_name=venue.get("name", ""), city=city)
        return jsonify({
            "venue_name": venue.get('name', ''),
            "events": [],
            "error": str(e),
        }), 500


@app.route('/api/events/<venue_name>')
def get_venue_events(venue_name):
    """
    Get cached events for a specific venue.

    Response:
    {
        "venue_name": "Beacon Theatre",
        "events": [...],
        "count": 15
    }
    """
    increment("server.api.events_by_venue.calls")
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet

    try:
        events = read_venue_events_from_sheet(venue_name)
        events_data = _serialize_events(events)

        return jsonify({
            "venue_name": venue_name,
            "events": events_data,
            "count": len(events_data),
        })

    except Exception as e:
        record_failure("server.events_by_venue", str(e), venue_name=venue_name)
        return jsonify({"error": str(e)}), 500


@app.route('/api/events')
def get_all_events():
    """
    Get all cached events.

    Query params:
        venue: Filter by venue name
        matched: If 'true', only return matched events

    Response:
    {
        "events": [...],
        "count": 100,
        "by_venue": {"Beacon Theatre": 15, ...}
    }
    """
    increment("server.api.events.calls")
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet

    venue_filter = request.args.get('venue')
    matched_only = request.args.get('matched', '').lower() == 'true'

    try:
        if venue_filter:
            events = read_venue_events_from_sheet(venue_filter)
        else:
            events = read_venue_events_from_sheet()

        # Filter to matched only if requested
        if matched_only:
            events = [e for e in events if e.get('matched_artist')]

        events_data = _serialize_events(events)

        # Group by venue
        by_venue = {}
        for event in events:
            venue = event.get('venue_name', 'Unknown')
            by_venue[venue] = by_venue.get(venue, 0) + 1

        return jsonify({
            "events": events_data,
            "count": len(events_data),
            "by_venue": by_venue,
        })

    except Exception as e:
        record_failure("server.events", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/matched-events')
def get_matched_events():
    """
    Get all events matching user's YouTube Music artists.

    Response:
    {
        "events": [...],
        "count": 10,
        "by_artist": {"Artist Name": [...], ...}
    }
    """
    increment("server.api.matched_events.calls")
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet
    from venue_scout.concert_matcher import match_events_to_artists, get_user_artists

    try:
        # Get events
        events = read_venue_events_from_sheet()

        # Get user's artists and match
        artists = get_user_artists()
        if artists:
            events = match_events_to_artists(events, artists)

        # Filter to matched only
        matched = [e for e in events if e.get('matched_artist')]

        # Convert to JSON-serializable format
        events_data = []
        by_artist = {}
        for event in matched:
            event_dict = dict(event)
            if event_dict.get("datetime"):
                event_dict["datetime"] = event_dict["datetime"].isoformat()
            events_data.append(event_dict)

            artist = event.get('matched_artist')
            if artist not in by_artist:
                by_artist[artist] = []
            by_artist[artist].append(event_dict)

        return jsonify({
            "events": events_data,
            "count": len(events_data),
            "by_artist": by_artist,
        })

    except Exception as e:
        record_failure("server.matched_events", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/event-cache-status')
def event_cache_status():
    """Get venue event cache status."""
    increment("server.api.event_cache_status.calls")
    from venue_scout.event_cache import get_cache_summary

    try:
        summary = get_cache_summary()
        return jsonify(summary)
    except Exception as e:
        record_failure("server.event_cache_status", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/event-filter-options')
def event_filter_options():
    """Return filter options derived from cached event rows."""
    increment("server.api.event_filter_options.calls")
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet

    try:
        events = read_venue_events_from_sheet()

        category_counts: dict[str, int] = {}
        with_distance = 0
        without_distance = 0

        for event in events:
            category = str(event.get("event_type", "") or "").strip()
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1

            travel = event.get("travel_minutes")
            if travel in (None, "", "None"):
                without_distance += 1
            else:
                with_distance += 1

        categories = [
            {"name": name, "count": count}
            for name, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0].lower()))
        ]

        return jsonify(
            {
                "categories": categories,
                "total_events": len(events),
                "with_distance": with_distance,
                "without_distance": without_distance,
            }
        )
    except Exception as e:
        record_failure("server.event_filter_options", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/query-events', methods=['POST'])
def query_events():
    """Filter already-fetched events using natural language."""
    increment("server.api.query_events.calls")
    from venue_scout.query_filter import query_events_with_llm

    data = request.json or {}
    query = (data.get("query") or "").strip()
    events = data.get("events")
    filters = data.get("filters") or {}
    max_results = _to_int_or_none(data.get("max_results"))
    if max_results is None or max_results <= 0:
        max_results = 10
    force_fallback = bool(data.get("force_fallback", False))

    if not query:
        record_failure("server.query_events", "missing_query")
        return jsonify({"error": "Query is required"}), 400

    if events is None:
        from venue_scout.venue_events_sheet import read_venue_events_from_sheet
        events = read_venue_events_from_sheet()

    if not isinstance(events, list):
        record_failure("server.query_events", "events_must_be_list")
        return jsonify({"error": "events must be a list"}), 400

    try:
        settings = _load_settings()
        normalized_filters = filters if isinstance(filters, dict) else {}
        distance_mode_raw = str(normalized_filters.get("distance_mode", "transit")).strip().lower()
        distance_mode = "none"
        if distance_mode_raw in ("walking", "transit"):
            distance_mode = distance_mode_raw

        max_travel = _to_int_or_none(normalized_filters.get("max_travel_minutes"))
        if max_travel is None or max_travel < 1:
            max_travel = int(getattr(settings, "DEFAULT_MAX_TRAVEL_MINUTES", 60))

        filtered_events, applied_filters = _apply_event_filters(
            events,
            normalized_filters,
            apply_distance_filter=False,
        )
        applied_filters["distance_mode"] = distance_mode
        if distance_mode in ("walking", "transit"):
            applied_filters["max_travel_minutes"] = max_travel
        if not filtered_events:
            return jsonify(
                {
                    "interpretation": "No events match your selected filters.",
                    "filters": {},
                    "applied_filters": applied_filters,
                    "matches": [],
                    "count": 0,
                    "input_count": len(events),
                    "filtered_count": 0,
                    "warning": "",
                    "mode": "filtered_empty",
                }
            )

        result = query_events_with_llm(
            query=query,
            events=filtered_events,
            max_results=max_results,
            force_fallback=force_fallback,
        )
        pre_distance_matches = list(result.get("matches", []))
        distance_meta: dict = {"mode": distance_mode, "applied": False}
        distance_warning = ""
        if distance_mode in ("walking", "transit"):
            pre_distance_count = len(pre_distance_matches)
            distance_matches, distance_meta, distance_warning = _filter_and_enrich_by_distance(
                pre_distance_matches,
                filters=normalized_filters,
                mode=distance_mode,
                max_minutes=max_travel,
            )
            if distance_meta.get("applied"):
                distance_meta["pre_distance_count"] = pre_distance_count
            pre_distance_matches = distance_matches

        warning_parts = [str(result.get("warning", "") or "").strip()]
        if distance_warning:
            warning_parts.append(distance_warning)
        warning = " ".join([part for part in warning_parts if part]).strip()

        matches = _serialize_events(pre_distance_matches)
        payload = {
            "interpretation": result.get("interpretation", ""),
            "filters": result.get("filters", {}),
            "applied_filters": applied_filters,
            "distance_filter": distance_meta,
            "matches": matches,
            "count": len(matches),
            "input_count": len(events),
            "filtered_count": len(filtered_events),
            "warning": warning,
            "mode": result.get("mode", "unknown"),
        }
        if payload["warning"]:
            log_event("api_query_events_warning", warning=payload["warning"], mode=payload["mode"])
        return jsonify(payload)
    except Exception as e:
        record_failure("server.query_events", str(e))
        return jsonify(
            {
                "interpretation": "Filtering failed.",
                "filters": {},
                "matches": [],
                "count": 0,
                "warning": str(e),
                "mode": "error",
            }
        ), 500


@app.route('/api/debug/health')
def debug_health():
    """Expose lightweight process health and recent failures."""
    settings = _load_settings()
    payload = {
        "status": "ok",
        "settings": {
            "venue_event_cache_days": getattr(settings, "VENUE_EVENT_CACHE_DAYS", None),
            "venue_cache_threshold_days": getattr(settings, "VENUE_CACHE_THRESHOLD_DAYS", None),
            "venue_fetch_delay": getattr(settings, "VENUE_FETCH_DELAY", None),
            "jina_request_delay": getattr(settings, "JINA_REQUEST_DELAY", None),
        },
        "travel_cache_entries": len(_travel_cache),
        "observability": snapshot(),
    }
    return jsonify(payload)


if __name__ == '__main__':
    print("Starting Venue Scout server...")
    print("Open http://localhost:8000 in your browser")
    app.run(port=8000, debug=True)
