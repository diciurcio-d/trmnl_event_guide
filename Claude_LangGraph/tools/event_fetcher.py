"""Tool to fetch events from all configured sources with caching."""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.generic_scraper import fetch_events_from_source
from state import DEFAULT_SOURCES
from cache import (
    is_source_fresh,
    mark_source_updated,
    read_cached_events,
    write_events_to_cache,
    get_cache_summary,
)


@tool
def fetch_all_events(
    sources: list[dict] | None = None,
    force_update: bool = False,
    force_update_source: str | None = None,
    cache_threshold_days: int = 7,
) -> dict:
    """
    Fetch events from all configured sources, using cache when available.

    Args:
        sources: Optional list of source configs. If None, uses DEFAULT_SOURCES.
        force_update: If True, bypass cache and fetch fresh data for all sources.
        force_update_source: If provided, only force update this specific source
                            (use cache for all others). Overrides force_update.
        cache_threshold_days: Number of days before cached data is considered stale.
                             Default is 7 days.

    Returns:
        dict with:
        - events: List of all fetched events
        - summary: Dict mapping source name to event count
        - errors: List of any errors encountered
        - cache_status: Dict showing which sources were cached vs fetched
    """
    all_events = []
    summary = {}
    errors = []
    cache_status = {}

    # Use default sources if none provided
    if sources is None:
        sources = DEFAULT_SOURCES

    # Filter to enabled sources
    sources_to_fetch = [s for s in sources if s.get("enabled", True)]

    for source in sources_to_fetch:
        source_name = source["name"]

        # Determine if we should force update this source
        should_force = force_update or (force_update_source and source_name == force_update_source)

        # Check if we can use cached data
        if not should_force and is_source_fresh(source_name, cache_threshold_days):
            print(f"  Using cached data for {source_name}...", flush=True)
            cached_events = read_cached_events(source_name)
            all_events.extend(cached_events)
            summary[source_name] = len(cached_events)
            cache_status[source_name] = "cached"
            print(f"    ✓ Loaded {len(cached_events)} events from cache", flush=True)
        else:
            # Fetch fresh data
            try:
                reason = "force update" if should_force else "cache expired/missing"
                print(f"  Fetching from {source_name} ({reason})...", flush=True)
                events = fetch_events_from_source(source)

                # Filter out past events before caching
                now = datetime.now(ZoneInfo("America/New_York"))
                original_count = len(events)
                events = [e for e in events if e.get("datetime") and e["datetime"] >= now]
                if original_count != len(events):
                    print(f"    Filtered out {original_count - len(events)} past events", flush=True)

                all_events.extend(events)
                summary[source_name] = len(events)
                cache_status[source_name] = "fetched"
                print(f"    ✓ Found {len(events)} events", flush=True)

                # Update cache for this source
                write_events_to_cache(events, source_name)
                mark_source_updated(source_name)

            except Exception as e:
                errors.append(f"Error fetching from {source_name}: {str(e)}")
                summary[source_name] = 0
                cache_status[source_name] = "error"
                print(f"    ✗ Error: {str(e)}", flush=True)

                # Try to use stale cache as fallback
                cached_events = read_cached_events(source_name)
                if cached_events:
                    print(f"    ↳ Using stale cache ({len(cached_events)} events)", flush=True)
                    all_events.extend(cached_events)
                    summary[source_name] = len(cached_events)
                    cache_status[source_name] = "stale_cache"

    # Sort events by datetime (use a far-future date for None values)
    far_future = datetime(2099, 12, 31, tzinfo=ZoneInfo("America/New_York"))
    all_events.sort(key=lambda x: x["datetime"] if x["datetime"] else far_future)

    return {
        "events": all_events,
        "summary": summary,
        "total": len(all_events),
        "errors": errors,
        "cache_status": cache_status,
    }


def fetch_single_source(source: dict, force_update: bool = False) -> list[dict]:
    """
    Fetch events from a single source.

    Args:
        source: Source configuration dict
        force_update: If True, bypass cache

    Returns:
        List of events from that source
    """
    source_name = source["name"]

    if not force_update and is_source_fresh(source_name):
        return read_cached_events(source_name)

    events = fetch_events_from_source(source)

    # Filter out past events
    now = datetime.now(ZoneInfo("America/New_York"))
    events = [e for e in events if e.get("datetime") and e["datetime"] >= now]

    write_events_to_cache(events, source_name)
    mark_source_updated(source_name)
    return events


def fetch_source_by_name(source_name: str, force_update: bool = True) -> dict:
    """
    Fetch events from a single source by name.

    Args:
        source_name: Name of the source (e.g., "92NY", "Caveat NYC")
        force_update: If True, bypass cache (default True for single source fetches)

    Returns:
        dict with events, count, and any error
    """
    # Find the source config
    source = next((s for s in DEFAULT_SOURCES if s["name"] == source_name), None)

    if not source:
        available = [s["name"] for s in DEFAULT_SOURCES]
        return {
            "events": [],
            "count": 0,
            "error": f"Source '{source_name}' not found. Available: {available}",
        }

    if not source.get("enabled", True):
        return {
            "events": [],
            "count": 0,
            "error": f"Source '{source_name}' is disabled",
        }

    try:
        events = fetch_single_source(source, force_update=force_update)
        return {
            "events": events,
            "count": len(events),
            "error": None,
        }
    except Exception as e:
        return {
            "events": [],
            "count": 0,
            "error": str(e),
        }


def get_available_sources() -> list[str]:
    """Return list of available source names."""
    return [s["name"] for s in DEFAULT_SOURCES]


def get_cache_info() -> dict:
    """Get information about the current cache state."""
    return get_cache_summary()
