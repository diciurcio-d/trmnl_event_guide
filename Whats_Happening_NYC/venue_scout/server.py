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
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.firestore_client import get_db
from venue_scout.observability import increment, log_event, record_failure, snapshot

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

_TZ = ZoneInfo("America/New_York")

# Subway station cache: populated once per process from Firestore.
# Shape: {"1": [{"id": "101", "name": "Van Cortlandt Park-242 St"}, ...], ...}
_subway_stations_cache: dict[str, list[dict]] | None = None

# Travel times cache: {station_id: {route_planner_id: {w: int, s: int}}}
# Loaded from GCS at startup; falls back to per-request Firestore reads if missing.
_travel_times_cache: dict[str, dict] = {}


def _preload() -> None:
    """Eagerly load events and travel times into memory at container startup.

    Called once at module load time (before gunicorn forks workers) so every
    worker starts with warm caches and the first user request is fast.
    """
    import threading

    def _load_events():
        try:
            from venue_scout.venue_events_sheet import read_venue_events_from_sheet
            events = read_venue_events_from_sheet()
            print(f"[preload] Events ready: {len(events):,}")
        except Exception as exc:
            print(f"[preload] Events load failed: {exc}")

    def _load_travel_times():
        try:
            from venue_scout.gcs_sync import pull_travel_times_cache
            data = pull_travel_times_cache(verbose=False)
            if data:
                _travel_times_cache.update(data)
                print(f"[preload] Travel times ready: {len(data):,} stations")
                return
        except Exception as exc:
            print(f"[preload] GCS travel times unavailable ({exc.__class__.__name__}), falling back to Firestore.")
        try:
            db = get_db()
            data = {}
            for doc in db.collection("subway_travel_times").stream():
                data[doc.id] = (doc.to_dict() or {}).get("times", {})
            _travel_times_cache.update(data)
            print(f"[preload] Travel times ready (Firestore): {len(data):,} stations")
        except Exception as exc:
            print(f"[preload] Travel times load failed: {exc}")

    def _load_venue_index():
        try:
            _load_venue_route_planner_index()
            idx = _venue_route_planner_index or {}
            n = len(idx.get("_by_name", {}))
            print(f"[preload] Venue route-planner index ready: {n:,} venues")
        except Exception as exc:
            print(f"[preload] Venue index load failed: {exc}")

    t1 = threading.Thread(target=_load_events, daemon=True)
    t2 = threading.Thread(target=_load_travel_times, daemon=True)
    t3 = threading.Thread(target=_load_venue_index, daemon=True)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
_SUBWAY_LINE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "E", "F", "G", "J", "L", "M", "N", "Q", "R", "S"]

# Venue route_planner_id lookup: {route_planner_id: True} populated lazily.
# We keep the per-event lookup as a flat map: {route_planner_id -> True} built
# from the venues collection when first needed.
_venue_route_planner_index: dict[str, str] | None = None  # {route_planner_id: venue_name}


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


def _load_subway_stations() -> dict[str, list[dict]]:
    """Load and cache subway station list from Firestore (station_name + line only)."""
    global _subway_stations_cache
    if _subway_stations_cache is not None:
        return _subway_stations_cache

    db = get_db()
    by_line: dict[str, list[dict]] = {}

    try:
        docs = db.collection("subway_travel_times").select(["station_name", "lines"]).stream()
        for doc in docs:
            data = doc.to_dict() or {}
            name = str(data.get("station_name", "") or "").strip()
            if not name:
                continue
            # `lines` is a list of line labels; fall back to legacy `line` string
            raw_lines = data.get("lines")
            if isinstance(raw_lines, list):
                line_labels = [str(l).strip() for l in raw_lines if str(l).strip()]
            else:
                legacy = str(data.get("line", "") or "").strip()
                line_labels = [legacy] if legacy else []
            for line in line_labels:
                by_line.setdefault(line, []).append({"id": doc.id, "name": name})
    except Exception:
        pass

    # Sort stations alphabetically within each line; preserve defined line order
    result: dict[str, list[dict]] = {}
    for line in _SUBWAY_LINE_ORDER:
        if line in by_line:
            result[line] = sorted(by_line[line], key=lambda s: s["name"])
    # Include any lines not in the predefined order (shouldn't happen, but be safe)
    for line, stations in by_line.items():
        if line not in result:
            result[line] = sorted(stations, key=lambda s: s["name"])

    _subway_stations_cache = result
    return _subway_stations_cache


def _load_venue_route_planner_index() -> dict[str, str]:
    """
    Build {route_planner_id: normalized_venue_name} from cached venues.
    Used to look up a venue's route_planner_id from an event's venue_name or ticketmaster_id.
    """
    global _venue_route_planner_index
    if _venue_route_planner_index is not None:
        return _venue_route_planner_index

    index: dict[str, str] = {}  # route_planner_id -> normalized_name (not critical, just for reference)
    # Also build reverse: normalized_name -> route_planner_id and tm_id -> route_planner_id
    try:
        from venue_scout.cache import read_cached_venues
        venues = read_cached_venues()
    except Exception:
        venues = []

    name_to_rpid: dict[str, str] = {}
    tm_to_rpid: dict[str, str] = {}

    for venue in venues:
        rpid = str(venue.get("route_planner_id", "") or "").strip()
        if not rpid:
            continue
        name_key = _normalize_lookup_text(venue.get("name", ""))
        if name_key:
            name_to_rpid[name_key] = rpid
        tm_id = str(venue.get("ticketmaster_venue_id", "") or "").strip()
        if tm_id:
            tm_to_rpid[tm_id] = rpid

    _venue_route_planner_index = {"_by_name": name_to_rpid, "_by_tm": tm_to_rpid}  # type: ignore[assignment]
    return _venue_route_planner_index


def _event_route_planner_id(event: dict) -> str | None:
    """Resolve a route_planner_id for an event via venue name or ticketmaster ID."""
    index = _load_venue_route_planner_index()
    by_name = index.get("_by_name", {})
    by_tm = index.get("_by_tm", {})

    tm_id = str(event.get("ticketmaster_venue_id", "") or "").strip()
    if tm_id:
        rpid = by_tm.get(tm_id)
        if rpid:
            return rpid

    name_key = _normalize_lookup_text(event.get("venue_name", ""))
    if name_key:
        return by_name.get(name_key)

    return None


def _filter_and_enrich_by_distance(
    matches: list[dict],
    filters: dict,
    mode: str,
    max_minutes: int,
) -> tuple[list[dict], dict, str]:
    """
    Estimate travel time for LLM matches, then apply distance filter.
    Returns (kept_events, metadata, warning).

    If filters contains origin_station_id, uses pre-computed subway travel times
    from Firestore. Otherwise returns matches unfiltered.
    """
    # ── Subway pre-computed path ──────────────────────────────────────────────
    station_id = str(filters.get("origin_station_id", "") or "").strip()
    if station_id:
        try:
            if station_id in _travel_times_cache:
                times_map: dict = _travel_times_cache[station_id]
            else:
                db = get_db()
                station_doc = db.collection("subway_travel_times").document(station_id).get()
                if not station_doc.exists:
                    return (
                        matches,
                        {"mode": mode, "applied": False},
                        f"Station '{station_id}' not found in travel time database. Showing unfiltered results.",
                    )
                times_map = (station_doc.to_dict() or {}).get("times", {})
                _travel_times_cache[station_id] = times_map
        except Exception as exc:
            return (
                matches,
                {"mode": mode, "applied": False},
                f"Subway travel time lookup failed ({exc.__class__.__name__}). Showing unfiltered results.",
            )

        # mode key: "s" for transit/subway, "w" for walking
        time_key = "w" if mode == "walking" else "s"

        kept: list[dict] = []
        dropped_over = 0
        no_rpid = 0
        no_entry = 0

        for event in matches:
            rpid = _event_route_planner_id(event)
            row = dict(event)
            if rpid is None:
                # No route_planner_id — include unfiltered per spec
                row["travel_minutes"] = None
                kept.append(row)
                no_rpid += 1
                continue

            entry = times_map.get(rpid)
            if entry is None:
                # No pre-computed path — include unfiltered
                row["travel_minutes"] = None
                kept.append(row)
                no_entry += 1
                continue

            minutes = entry.get(time_key)
            if minutes is None:
                # Mode not available (e.g. walk_status not ok) — include unfiltered
                row["travel_minutes"] = None
                kept.append(row)
                no_entry += 1
                continue

            row["travel_minutes"] = minutes
            if minutes <= max_minutes:
                kept.append(row)
            else:
                dropped_over += 1

        meta = {
            "mode": mode,
            "applied": True,
            "source": "subway_precomputed",
            "station_id": station_id,
            "max_minutes": max_minutes,
            "input_matches": len(matches),
            "kept": len(kept),
            "no_route_planner_id": no_rpid,
            "no_precomputed_entry": no_entry,
            "dropped_over_limit": dropped_over,
        }
        return kept, meta, ""

    # No station selected — distance filter not applicable.
    return matches, {"mode": mode, "applied": False}, ""


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

    if bool(filters.get("free_only", False)):
        filtered = [event for event in filtered if event.get("is_free") is True]
        applied["free_only"] = True

    if bool(filters.get("exclude_recurring", False)):
        filtered = [event for event in filtered if not event.get("is_recurring")]
        applied["exclude_recurring"] = True

    _time_of_day = str(filters.get("time_of_day") or "").strip().lower()
    if _time_of_day and _time_of_day != "none":
        _tod_hours: dict[str, tuple[int, int]] = {
            "morning":    (6, 11),
            "afternoon":  (12, 16),
            "evening":    (17, 21),
            "late_night": (22, 3),
        }
        _tod_range = _tod_hours.get(_time_of_day)
        if _tod_range:
            _start_h, _end_h = _tod_range
            _wraps = _end_h < _start_h
            _tod_filtered = []
            for _ev in filtered:
                if not _ev.get("datetime"):
                    _tod_filtered.append(_ev)
                    continue
                _dt = _parse_event_datetime(_ev)
                if _dt is None:
                    _tod_filtered.append(_ev)
                    continue
                _h = _dt.hour
                if (_wraps and (_h >= _start_h or _h <= _end_h)) or (not _wraps and _start_h <= _h <= _end_h):
                    _tod_filtered.append(_ev)
            filtered = _tod_filtered
            applied["time_of_day"] = _time_of_day

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

    if bool(filters.get("exclude_recurring", False)):
        filtered = [event for event in filtered if not event.get("is_recurring")]
        applied["exclude_recurring"] = True

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


@app.route('/api/subway-stations')
def subway_stations():
    """
    Return subway stations grouped by line.

    Response:
    {
        "lines": {
            "1": [{"id": "101", "name": "Van Cortlandt Park-242 St"}, ...],
            "A": [...],
            ...
        }
    }
    """
    increment("server.api.subway_stations.calls")
    try:
        lines = _load_subway_stations()
        return jsonify({"lines": lines})
    except Exception as exc:
        record_failure("server.subway_stations", str(exc))
        return jsonify({"error": str(exc)}), 500



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
    context = (data.get("context") or "").strip()
    history = (data.get("history") or "").strip()
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
        from venue_scout.query_filter import (
            extract_query_intent,
            filter_events_by_date_window,
            query_events_with_llm,
        )

        settings = _load_settings()
        normalized_filters = dict(filters if isinstance(filters, dict) else {})
        distance_mode_raw = str(normalized_filters.get("distance_mode", "transit")).strip().lower()
        distance_mode = "none"
        if distance_mode_raw in ("walking", "transit"):
            distance_mode = distance_mode_raw

        max_travel = _to_int_or_none(normalized_filters.get("max_travel_minutes"))
        if max_travel is None or max_travel < 1:
            max_travel = int(getattr(settings, "DEFAULT_MAX_TRAVEL_MINUTES", 60))

        input_event_count = len(events)

        # ── Step 1: Extract structured intent from query (single LLM tool call) ──
        intent = extract_query_intent(query)
        intent_warning = str(intent.pop("_warning", "") or "")

        # ── Step 2: Apply date filter from intent ─────────────────────────────────
        if intent.get("date_window_applied") and intent.get("date_start_dt") and intent.get("date_end_dt"):
            events = filter_events_by_date_window(
                events,
                intent["date_start_dt"],
                intent["date_end_dt"],
                intent.get("include_undated", False),
            )
            if not events:
                return jsonify({
                    "interpretation": "No events found in the requested date range.",
                    "filters": {"date": {k: v for k, v in intent.items() if not k.endswith("_dt")}},
                    "applied_filters": {},
                    "matches": [],
                    "count": 0,
                    "input_count": input_event_count,
                    "filtered_count": 0,
                    "warning": intent_warning,
                    "mode": "date_tool_empty",
                })

        # ── Step 3: Merge intent constraints into filters ─────────────────────────
        if intent.get("free_only"):
            normalized_filters["free_only"] = True
        if intent.get("exclude_recurring"):
            normalized_filters["exclude_recurring"] = True
        if intent.get("time_of_day"):
            normalized_filters["time_of_day"] = intent["time_of_day"]
        if intent.get("max_travel_minutes"):
            max_travel = intent["max_travel_minutes"]

        # ── Step 4: Apply structural filters (free, recurring, time-of-day) ───────
        filtered_events, applied_filters = _apply_event_filters(
            events,
            normalized_filters,
            apply_distance_filter=False,
        )
        applied_filters["distance_mode"] = distance_mode
        if distance_mode in ("walking", "transit"):
            applied_filters["max_travel_minutes"] = max_travel
        if not filtered_events:
            return jsonify({
                "interpretation": "No events match your selected filters.",
                "filters": {},
                "applied_filters": applied_filters,
                "matches": [],
                "count": 0,
                "input_count": input_event_count,
                "filtered_count": 0,
                "warning": intent_warning,
                "mode": "filtered_empty",
            })

        # ── Step 5: Apply distance filter PRE-LLM ────────────────────────────────
        distance_meta: dict = {"mode": distance_mode, "applied": False}
        distance_warning = ""
        if distance_mode in ("walking", "transit"):
            filtered_events, distance_meta, distance_warning = _filter_and_enrich_by_distance(
                filtered_events,
                filters=normalized_filters,
                mode=distance_mode,
                max_minutes=max_travel,
            )
            if distance_meta.get("applied") and not filtered_events:
                return jsonify({
                    "interpretation": "No events found within your travel time limit.",
                    "filters": {},
                    "applied_filters": applied_filters,
                    "distance_filter": distance_meta,
                    "matches": [],
                    "count": 0,
                    "input_count": input_event_count,
                    "filtered_count": 0,
                    "warning": distance_warning,
                    "mode": "distance_empty",
                })

        # ── Step 6: Semantic search + LLM re-ranking ─────────────────────────────
        result = query_events_with_llm(
            query=query,
            events=filtered_events,
            max_results=max_results,
            force_fallback=force_fallback,
            context=context,
            history=history,
            pre_applied_filters=intent,
        )

        # Enrich matches with cloudflare_protected so the frontend can pass it to /api/event-info
        try:
            from venue_scout.cache import read_cached_venues
            cf_lookup = {
                v.get("name", "").lower(): (v.get("cloudflare_protected", "") == "yes")
                for v in read_cached_venues()
            }
            for m in result.get("matches", []):
                m["cloudflare_protected"] = cf_lookup.get(
                    str(m.get("venue_name", "")).lower(), False
                )
        except Exception:
            pass  # best-effort; don't break the query on cache read failure

        warning_parts = [intent_warning, str(result.get("warning", "") or "").strip()]
        if distance_warning:
            warning_parts.append(distance_warning)
        warning = " ".join([part for part in warning_parts if part]).strip()

        matches = _serialize_events(result.get("matches", []))
        payload = {
            "interpretation": result.get("interpretation", ""),
            "filters": result.get("filters", {}),
            "applied_filters": applied_filters,
            "distance_filter": distance_meta,
            "matches": matches,
            "count": len(matches),
            "input_count": input_event_count,
            "filtered_count": len(filtered_events),
            "warning": warning,
            "mode": result.get("mode", "unknown"),
            "follow_up_question": result.get("follow_up_question", ""),
        }
        log_event(
            "api_query_events",
            query=query,
            result_count=len(matches),
            mode=payload["mode"],
            input_count=input_event_count,
            filtered_count=len(filtered_events),
        )
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


@app.route('/api/event-info', methods=['POST'])
@limiter.limit("30 per hour")  # LLM + page fetch — moderately expensive
def event_info():
    """Answer a question about a specific event by fetching its page or using web grounding."""
    increment("server.api.event_info.calls")
    payload = request.get_json(force=True) or {}
    event_url            = str(payload.get("event_url",            "")).strip()
    event_name           = str(payload.get("event_name",           "")).strip()
    question             = str(payload.get("question",             "")).strip()
    cloudflare_protected = bool(payload.get("cloudflare_protected", False))

    if not question:
        return jsonify({"error": "question is required"}), 400

    from venue_scout.event_info import answer_event_question
    result = answer_event_question(event_url, event_name, question, cloudflare_protected)
    return jsonify(result)


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
        },
        "observability": snapshot(),
    }
    return jsonify(payload)


_preload()

if __name__ == '__main__':
    print("Starting Venue Scout server...")
    print("Open http://localhost:8000 in your browser")
    app.run(port=8000, debug=True)
