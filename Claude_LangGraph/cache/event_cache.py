"""Event caching system with CSV storage and freshness tracking."""

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Default cache directory (relative to this file)
CACHE_DIR = Path(__file__).parent
EVENTS_CSV = CACHE_DIR / "events.csv"
METADATA_FILE = CACHE_DIR / "cache_metadata.json"

# CSV columns
CSV_COLUMNS = [
    "name",
    "datetime",
    "date_str",
    "type",
    "sold_out",
    "source",
    "location",
    "description",
    "has_specific_time",
    "url",
    "travel_minutes",
]


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
    # Ensure last_updated has timezone info
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
    Read events from the CSV cache.

    Args:
        source_name: If provided, only return events from this source.
                    If None, return all cached events.

    Returns:
        List of event dicts
    """
    if not EVENTS_CSV.exists():
        return []

    events = []
    try:
        with open(EVENTS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by source if specified
                if source_name and row.get("source") != source_name:
                    continue

                # Convert types
                travel_mins_str = row.get("travel_minutes", "")
                travel_minutes = None
                if travel_mins_str and travel_mins_str not in ("", "None"):
                    try:
                        travel_minutes = int(travel_mins_str)
                    except ValueError:
                        pass

                event = {
                    "name": row.get("name", ""),
                    "datetime": None,
                    "date_str": row.get("date_str", ""),
                    "type": row.get("type", ""),
                    "sold_out": row.get("sold_out", "").lower() == "true",
                    "source": row.get("source", ""),
                    "location": row.get("location", ""),
                    "description": row.get("description", ""),
                    "has_specific_time": row.get("has_specific_time", "").lower() == "true",
                    "url": row.get("url", ""),
                    "travel_minutes": travel_minutes,
                }

                # Parse datetime
                dt_str = row.get("datetime", "")
                if dt_str and dt_str != "None" and dt_str != "":
                    try:
                        dt = datetime.fromisoformat(dt_str)
                        # Ensure timezone info is present
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                        event["datetime"] = dt
                    except ValueError:
                        pass

                events.append(event)
    except (IOError, csv.Error) as e:
        print(f"    Warning: Error reading cache: {e}")
        return []

    return events


def write_events_to_cache(events: list[dict], source_name: str | None = None):
    """
    Write events to the CSV cache.

    If source_name is provided, only updates events from that source
    (preserving events from other sources).

    Args:
        events: List of event dicts to cache
        source_name: If provided, only replace events from this source
    """
    _ensure_cache_dir()

    # If updating a specific source, merge with existing cache
    if source_name:
        existing_events = read_cached_events()
        # Remove old events from this source
        existing_events = [e for e in existing_events if e.get("source") != source_name]
        # Add new events from this source
        new_source_events = [e for e in events if e.get("source") == source_name]
        all_events = existing_events + new_source_events
    else:
        all_events = events

    # Sort by source (ascending) then date (descending)
    far_future = datetime(2099, 12, 31, tzinfo=ZoneInfo("America/New_York"))
    all_events.sort(
        key=lambda x: (
            x.get("source", ""),
            -(x["datetime"].timestamp() if x.get("datetime") else far_future.timestamp())
        )
    )

    # Write to CSV
    with open(EVENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for event in all_events:
            travel_mins = event.get("travel_minutes")
            row = {
                "name": event.get("name", ""),
                "datetime": event.get("datetime").isoformat() if event.get("datetime") else "",
                "date_str": event.get("date_str", ""),
                "type": event.get("type", ""),
                "sold_out": str(event.get("sold_out", False)),
                "source": event.get("source", ""),
                "location": event.get("location", ""),
                "description": event.get("description", ""),
                "has_specific_time": str(event.get("has_specific_time", False)),
                "url": event.get("url", ""),
                "travel_minutes": str(travel_mins) if travel_mins is not None else "",
            }
            writer.writerow(row)


def clear_cache():
    """Clear all cached data."""
    if EVENTS_CSV.exists():
        EVENTS_CSV.unlink()
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
