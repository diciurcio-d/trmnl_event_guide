"""Event caching system with Google Sheets storage and freshness tracking."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sheets import read_events_from_sheet, write_events_to_sheet

# Metadata file for freshness tracking
CACHE_DIR = Path(__file__).parent
METADATA_FILE = CACHE_DIR / "cache_metadata.json"


def _ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_metadata() -> dict:
    """Load cache metadata (last update times per source)."""
    if not METADATA_FILE.exists():
        return {"sources": {}}

    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"sources": {}}


def _save_metadata(metadata: dict):
    """Save cache metadata."""
    _ensure_cache_dir()
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def get_source_last_updated(source_name: str) -> datetime | None:
    """Get the last update time for a source."""
    metadata = _load_metadata()
    timestamp = metadata.get("sources", {}).get(source_name)

    if timestamp:
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    return None


def is_source_fresh(source_name: str, threshold_days: int = 7) -> bool:
    """Check if a source's cached data is still fresh."""
    last_updated = get_source_last_updated(source_name)

    if last_updated is None:
        return False

    now = datetime.now(ZoneInfo("America/New_York"))
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=ZoneInfo("America/New_York"))

    age = now - last_updated
    return age < timedelta(days=threshold_days)


def mark_source_updated(source_name: str):
    """Mark a source as updated now."""
    metadata = _load_metadata()
    if "sources" not in metadata:
        metadata["sources"] = {}

    metadata["sources"][source_name] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    _save_metadata(metadata)


def read_cached_events(source_name: str | None = None) -> list[dict]:
    """
    Read events from Google Sheets cache.

    Args:
        source_name: If provided, only return events from this source.
                    If None, return all cached events.

    Returns:
        List of event dicts
    """
    return read_events_from_sheet(source_name)


def write_events_to_cache(events: list[dict], source_name: str | None = None):
    """
    Write events to Google Sheets cache.

    If source_name is provided, only updates events from that source
    (preserving events from other sources).

    Args:
        events: List of event dicts to cache
        source_name: If provided, only replace events from this source
    """
    write_events_to_sheet(events, source_name)


def clear_cache():
    """Clear metadata (sheets data must be cleared manually)."""
    if METADATA_FILE.exists():
        METADATA_FILE.unlink()


def get_cache_summary() -> dict:
    """Get a summary of the cache status."""
    metadata = _load_metadata()
    events = read_cached_events()

    # Count events by source
    by_source = {}
    for event in events:
        source = event.get("source", "Unknown")
        by_source[source] = by_source.get(source, 0) + 1

    return {
        "total_events": len(events),
        "events_by_source": by_source,
        "source_timestamps": metadata.get("sources", {}),
    }
