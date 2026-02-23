"""Per-venue event cache freshness tracking.

This module now delegates to cache.py which stores everything in Google Sheets.
The JSON-based venue_events_metadata.json is no longer used.
"""

from .cache import (
    is_venue_events_fresh,
    get_stale_venues,
    update_venue_event_tracking,
    read_cached_venues,
)


def mark_venue_fetched(
    venue_name: str,
    city: str,
    event_count: int,
    source_used: str,
):
    """
    Mark a venue as having been fetched.

    Delegates to cache.update_venue_event_tracking().

    Args:
        venue_name: Name of the venue
        city: City the venue is in
        event_count: Number of events found
        source_used: Source that was used ("ticketmaster", "scrape", "both")
    """
    update_venue_event_tracking(venue_name, city, event_count, source_used)


def get_venue_metadata(venue_name: str, city: str) -> dict | None:
    """
    Get metadata for a venue's last fetch.

    Returns:
        Dict with last_event_fetch, event_count, event_source, or None if not found
    """
    venues = read_cached_venues(city)
    venue_lower = venue_name.lower().strip()

    for v in venues:
        if v.get("name", "").lower().strip() == venue_lower:
            last_fetch = v.get("last_event_fetch", "")
            if last_fetch:
                return {
                    "last_fetched": last_fetch,
                    "event_count": v.get("event_count", 0),
                    "source_used": v.get("event_source", ""),
                    "venue_name": venue_name,
                    "city": city,
                }
    return None


def get_fresh_venues(venues: list[dict], city: str) -> list[dict]:
    """
    Filter to only venues with fresh event data.

    Args:
        venues: List of Venue dicts
        city: City to check

    Returns:
        List of venues that have fresh event data
    """
    fresh = []
    for venue in venues:
        venue_name = venue.get("name", "")
        venue_city = venue.get("city", city)
        if is_venue_events_fresh(venue_name, venue_city):
            fresh.append(venue)
    return fresh


def clear_venue_cache(venue_name: str | None = None, city: str | None = None):
    """
    Clear venue event tracking data.

    This clears the last_event_fetch, event_count, and event_source fields
    in the Venues sheet.

    Args:
        venue_name: If provided, only clear this venue
        city: If provided with venue_name, only clear that venue in that city
              If provided alone, clear all venues in that city
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from .cache import _get_sheets_service, get_or_create_venues_sheet, VENUE_COLUMNS, _venue_to_row

    venues = read_cached_venues()
    city_lower = city.lower().strip() if city else None
    venue_lower = venue_name.lower().strip() if venue_name else None

    updated = False
    for v in venues:
        should_clear = False

        if venue_lower and city_lower:
            # Clear specific venue in city
            if (v.get("name", "").lower().strip() == venue_lower and
                v.get("city", "").lower().strip() == city_lower):
                should_clear = True
        elif city_lower:
            # Clear all venues in city
            if v.get("city", "").lower().strip() == city_lower:
                should_clear = True
        else:
            # Clear all
            should_clear = True

        if should_clear and v.get("last_event_fetch"):
            v["last_event_fetch"] = ""
            v["event_count"] = 0
            v["event_source"] = ""
            updated = True

    if not updated:
        return

    # Write back
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    rows = [VENUE_COLUMNS]
    for venue in venues:
        rows.append(_venue_to_row(venue))

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:Q"
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

    except Exception as e:
        print(f"Error clearing venue cache: {e}")


def get_cache_summary() -> dict:
    """Get a summary of the venue event cache status."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    import importlib.util
    from pathlib import Path

    # Load settings for threshold
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    threshold_days = getattr(settings, "VENUE_EVENT_CACHE_DAYS", 7)

    venues = read_cached_venues()

    # Count by city and source
    by_city = {}
    by_source = {}
    total_events = 0
    fresh_count = 0
    stale_count = 0
    never_fetched = 0

    now = datetime.now(ZoneInfo("America/New_York"))

    for v in venues:
        city = v.get("city", "Unknown")
        source = v.get("event_source", "")
        count = v.get("event_count", 0)
        last_fetch = v.get("last_event_fetch", "")

        by_city[city] = by_city.get(city, 0) + 1
        if source:
            by_source[source] = by_source.get(source, 0) + 1
        total_events += count

        if not last_fetch:
            never_fetched += 1
        else:
            try:
                last_fetched = datetime.fromisoformat(last_fetch)
                if last_fetched.tzinfo is None:
                    last_fetched = last_fetched.replace(tzinfo=ZoneInfo("America/New_York"))
                age = now - last_fetched
                if age < timedelta(days=threshold_days):
                    fresh_count += 1
                else:
                    stale_count += 1
            except ValueError:
                stale_count += 1

    return {
        "total_venues": len(venues),
        "fresh_venues": fresh_count,
        "stale_venues": stale_count,
        "never_fetched": never_fetched,
        "total_events": total_events,
        "venues_by_city": by_city,
        "venues_by_source": by_source,
    }


if __name__ == "__main__":
    print("Venue Event Cache Summary")
    print("-" * 40)
    summary = get_cache_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
