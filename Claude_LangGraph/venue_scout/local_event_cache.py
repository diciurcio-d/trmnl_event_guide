"""Thread-safe local event cache for parallel fetching.

Stores events locally during parallel fetching to avoid race conditions
with Google Sheets writes. Merges to sheets at the end of a run.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .paths import DATA_DIR, LOCAL_EVENTS_CACHE_FILE, ensure_data_dir

_CACHE_DIR = DATA_DIR
_EVENTS_CACHE_FILE = LOCAL_EVENTS_CACHE_FILE
_LOCK = threading.Lock()


def _load_cache() -> dict:
    """Load the local events cache."""
    if not _EVENTS_CACHE_FILE.exists():
        return {
            "events": [],
            "venues_fetched": [],
            "started_at": None,
            "last_saved": None,
        }

    try:
        with open(_EVENTS_CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "events": [],
            "venues_fetched": [],
            "started_at": None,
            "last_saved": None,
        }


def _save_cache(cache: dict):
    """Save the local events cache."""
    cache["last_saved"] = datetime.now(ZoneInfo("America/New_York")).isoformat()

    ensure_data_dir()
    with open(_EVENTS_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def start_fetch_session(city: str):
    """Start a new fetch session, clearing old cache."""
    with _LOCK:
        cache = {
            "events": [],
            "venues_fetched": [],
            "city": city,
            "started_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "last_saved": None,
        }
        _save_cache(cache)
        print(f"Started new fetch session for {city}")


def add_events(events: list[dict], venue_name: str):
    """Add events for a venue to the local cache (thread-safe)."""
    if not events:
        # Still mark venue as fetched even if no events
        with _LOCK:
            cache = _load_cache()
            if venue_name not in cache["venues_fetched"]:
                cache["venues_fetched"].append(venue_name)
                _save_cache(cache)
        return

    # Serialize events for JSON storage
    serialized = []
    for event in events:
        e = event.copy()
        # Convert datetime to string
        if e.get("datetime"):
            dt = e["datetime"]
            if hasattr(dt, "isoformat"):
                e["datetime"] = dt.isoformat()
            else:
                e["datetime"] = str(dt) if dt else ""
        serialized.append(e)

    with _LOCK:
        cache = _load_cache()
        cache["events"].extend(serialized)
        if venue_name not in cache["venues_fetched"]:
            cache["venues_fetched"].append(venue_name)
        _save_cache(cache)


def get_fetched_venues() -> list[str]:
    """Get list of venues already fetched in current session."""
    with _LOCK:
        cache = _load_cache()
        return cache.get("venues_fetched", [])


def get_cached_events() -> list[dict]:
    """Get all events from local cache."""
    with _LOCK:
        cache = _load_cache()
        return cache.get("events", [])


def get_cache_stats() -> dict:
    """Get statistics about the local cache."""
    with _LOCK:
        cache = _load_cache()
        return {
            "event_count": len(cache.get("events", [])),
            "venues_fetched": len(cache.get("venues_fetched", [])),
            "city": cache.get("city", ""),
            "started_at": cache.get("started_at"),
            "last_saved": cache.get("last_saved"),
        }


def has_pending_events() -> bool:
    """Check if there are events in local cache not yet synced to sheets."""
    with _LOCK:
        cache = _load_cache()
        return len(cache.get("events", [])) > 0


def clear_cache():
    """Clear the local events cache."""
    with _LOCK:
        if _EVENTS_CACHE_FILE.exists():
            _EVENTS_CACHE_FILE.unlink()
        print("Cleared local events cache")


def merge_to_sheets() -> int:
    """
    Merge local cache events to Google Sheets.

    Returns:
        Number of events written to sheets
    """
    from .venue_events_sheet import read_venue_events_from_sheet, write_venue_events_to_sheet

    with _LOCK:
        cache = _load_cache()
        local_events = cache.get("events", [])

    if not local_events:
        print("No events in local cache to merge")
        return 0

    print(f"Merging {len(local_events)} events from local cache to Google Sheets...")

    # Convert datetime strings back to datetime objects
    for event in local_events:
        dt_str = event.get("datetime")
        if dt_str and isinstance(dt_str, str) and dt_str not in ("", "None"):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(dt_str)
                event["datetime"] = dt
            except ValueError:
                event["datetime"] = None
        elif not dt_str or dt_str == "None":
            event["datetime"] = None

    # Read existing events from sheets
    existing = read_venue_events_from_sheet()

    # Build deduplication key set
    existing_keys = set()
    for e in existing:
        key = (
            e.get("venue_name", "").lower(),
            e.get("name", "").lower(),
            e.get("date_str", ""),
        )
        existing_keys.add(key)

    # Filter to only new events
    new_events = []
    for event in local_events:
        key = (
            event.get("venue_name", "").lower(),
            event.get("name", "").lower(),
            event.get("date_str", ""),
        )
        if key not in existing_keys:
            new_events.append(event)
            existing_keys.add(key)

    if not new_events:
        print("All events already in sheets (no new events to add)")
        # Clear local cache since everything is synced
        clear_cache()
        return 0

    # Combine and write
    all_events = existing + new_events
    write_venue_events_to_sheet(all_events)

    print(f"Added {len(new_events)} new events to Google Sheets")

    # Clear local cache after successful merge
    clear_cache()

    return len(new_events)


def resume_info() -> dict | None:
    """
    Get info about a resumable fetch session.

    Returns:
        Dict with session info, or None if no resumable session
    """
    with _LOCK:
        cache = _load_cache()

        if not cache.get("started_at"):
            return None

        return {
            "city": cache.get("city", ""),
            "venues_fetched": len(cache.get("venues_fetched", [])),
            "events_collected": len(cache.get("events", [])),
            "started_at": cache.get("started_at"),
            "last_saved": cache.get("last_saved"),
        }
