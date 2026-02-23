"""Select random events for TRMNL display."""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

CACHE_DIR = Path(__file__).parent.parent / "cache"
EVENTS_CSV = CACHE_DIR / "events.csv"


def get_events_next_n_days(days: int = 6) -> list[dict]:
    """
    Read events from CSV and filter to next N days.

    Args:
        days: Number of days to look ahead (default 6)

    Returns:
        List of event dicts within the date range
    """
    if not EVENTS_CSV.exists():
        return []

    now = datetime.now(ZoneInfo("America/New_York"))
    cutoff = now + timedelta(days=days)

    events = []
    with open(EVENTS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_str = row.get("datetime", "")
            if not dt_str or dt_str == "None":
                continue

            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
            except ValueError:
                continue

            # Filter: must be in the future and within N days
            if now <= dt <= cutoff:
                events.append({
                    "name": row.get("name", ""),
                    "datetime": dt,
                    "type": row.get("type", ""),
                    "source": row.get("source", ""),
                    "location": row.get("location", ""),
                    "description": row.get("description", ""),
                    "sold_out": row.get("sold_out", "").lower() == "true",
                    "has_specific_time": row.get("has_specific_time", "").lower() == "true",
                })

    return events


def select_random_events(count: int = 6, days: int = 6) -> list[dict]:
    """
    Select random events from the next N days.

    Args:
        count: Number of events to select (default 6)
        days: Number of days to look ahead (default 6)

    Returns:
        List of randomly selected events, sorted by datetime
    """
    events = get_events_next_n_days(days)

    if len(events) <= count:
        selected = events
    else:
        selected = random.sample(events, count)

    # Sort by datetime
    selected.sort(key=lambda x: x["datetime"])

    return selected


def format_event_for_display(event: dict) -> dict:
    """
    Format an event for TRMNL display.

    Args:
        event: Event dict with datetime, name, etc.

    Returns:
        Dict with formatted display strings
    """
    dt = event["datetime"]
    has_time = event.get("has_specific_time", False)

    return {
        "day": dt.strftime("%a"),  # Mon, Tue, etc.
        "date": dt.strftime("%b %d"),  # Feb 03
        "time": dt.strftime("%I:%M%p").lstrip("0").lower() if has_time else None,
        "name": event["name"][:100],  # Allow longer names, CSS handles 2-line truncation
        "description": event.get("description", "")[:200],  # CSS handles 3-line truncation
        "source": event["source"],
        "type": event["type"],
        "sold_out": event["sold_out"],
        "has_specific_time": has_time,
    }


def get_trmnl_events(count: int = 6, days: int = 6) -> list[dict]:
    """
    Get formatted events ready for TRMNL display.

    Args:
        count: Number of events to select (default 6)
        days: Number of days to look ahead (default 6)

    Returns:
        List of formatted event dicts for display
    """
    events = select_random_events(count, days)
    return [format_event_for_display(e) for e in events]


if __name__ == "__main__":
    # Test the selector
    events = get_trmnl_events()
    print(f"Selected {len(events)} events:\n")
    for e in events:
        print(f"  {e['day']} {e['date']} @ {e['time']}")
        print(f"    {e['name']}")
        print(f"    ({e['source']})\n")
