#!/usr/bin/env python3
"""Simple Flask server for Venue Scout with distance filtering."""

import json
import sys
import importlib.util
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.distance import get_travel_time, get_travel_times_batch
from venue_scout.observability import increment, log_event, record_failure, snapshot
from venue_scout.paths import TRAVEL_CACHE_FILE, ensure_data_dir

app = Flask(__name__, static_folder='.')
CORS(app)

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


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


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


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/travel-times', methods=['POST'])
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
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet, get_events_by_venue

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


@app.route('/api/query-events', methods=['POST'])
def query_events():
    """Filter already-fetched events using natural language."""
    increment("server.api.query_events.calls")
    from venue_scout.query_filter import query_events_with_llm

    data = request.json or {}
    query = (data.get("query") or "").strip()
    events = data.get("events")
    max_results = int(data.get("max_results", 10))
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
        result = query_events_with_llm(
            query=query,
            events=events,
            max_results=max_results,
            force_fallback=force_fallback,
        )
        matches = _serialize_events(result.get("matches", []))
        payload = {
            "interpretation": result.get("interpretation", ""),
            "filters": result.get("filters", {}),
            "matches": matches,
            "count": len(matches),
            "warning": result.get("warning", ""),
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
