"""Tool to search and filter events based on criteria."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from langchain_core.tools import tool


@tool
def search_events(
    events: list[dict],
    query: str | None = None,
    source: str | None = None,
    days_ahead: int | None = None,
    event_type: str | None = None,
    exclude_sold_out: bool = False,
    weekend_only: bool = False,
) -> list[dict]:
    """
    Search and filter events based on various criteria.

    Args:
        events: List of events to search through
        query: Text to search for in event name, description, or location
        source: Filter by source name (e.g., "Caveat NYC", "The Met")
        days_ahead: Only return events within this many days from now
        event_type: Filter by event type (e.g., "comedy", "book", "museum")
        exclude_sold_out: If True, exclude sold out events
        weekend_only: If True, only return events on Saturday or Sunday

    Returns:
        List of matching events, sorted by datetime
    """
    filtered = events.copy()
    now = datetime.now(ZoneInfo("America/New_York"))

    # Filter by date range
    if days_ahead is not None:
        cutoff = now + timedelta(days=days_ahead)
        filtered = [
            e for e in filtered
            if e.get("datetime") and now <= e["datetime"] <= cutoff
        ]

    # Filter to weekend only (Saturday=5, Sunday=6)
    if weekend_only:
        filtered = [
            e for e in filtered
            if e.get("datetime") and e["datetime"].weekday() in [5, 6]
        ]

    # Filter by source
    if source:
        source_lower = source.lower()
        filtered = [
            e for e in filtered
            if source_lower in e.get("source", "").lower()
        ]

    # Filter by event type
    if event_type:
        type_lower = event_type.lower()
        filtered = [
            e for e in filtered
            if type_lower in e.get("type", "").lower()
            or type_lower in e.get("name", "").lower()
            or type_lower in e.get("description", "").lower()
        ]

    # Filter by text query
    if query:
        query_lower = query.lower()
        filtered = [
            e for e in filtered
            if query_lower in e.get("name", "").lower()
            or query_lower in e.get("description", "").lower()
            or query_lower in e.get("location", "").lower()
            or query_lower in e.get("source", "").lower()
            or query_lower in e.get("type", "").lower()
        ]

    # Exclude sold out
    if exclude_sold_out:
        filtered = [e for e in filtered if not e.get("sold_out", False)]

    # Sort by datetime (use a far-future date for None values)
    far_future = datetime(2099, 12, 31, tzinfo=ZoneInfo("America/New_York"))
    filtered.sort(key=lambda x: x["datetime"] if x["datetime"] else far_future)

    return filtered


def format_events_for_display(events: list[dict], max_events: int = 20) -> str:
    """
    Format a list of events for display in chat.

    Args:
        events: List of events to format
        max_events: Maximum number of events to include

    Returns:
        Formatted string representation of events
    """
    if not events:
        return "No events found matching your criteria."

    events_to_show = events[:max_events]
    lines = []

    current_day = None
    for event in events_to_show:
        if not event.get("datetime"):
            continue

        day = event["datetime"].strftime("%A, %B %d")

        if day != current_day:
            current_day = day
            lines.append(f"\n**{day}**")

        time_str = (
            event["datetime"].strftime("%I:%M %p").lstrip("0")
            if event.get("has_specific_time", True)
            else "TBD"
        )

        name = event.get("name", "Unknown Event")
        if event.get("sold_out"):
            name += " [SOLD OUT]"

        location = event.get("location", "")
        source = event.get("source", "")

        lines.append(f"- {time_str}: **{name}**")
        lines.append(f"  📍 {location} ({source})")

        if event.get("description"):
            desc = event["description"][:100]
            if len(event["description"]) > 100:
                desc += "..."
            lines.append(f"  _{desc}_")

    result = "\n".join(lines)

    if len(events) > max_events:
        result += f"\n\n_...and {len(events) - max_events} more events_"

    return result


def get_events_summary(events: list[dict]) -> dict:
    """
    Get a summary of events by source and type.

    Args:
        events: List of events to summarize

    Returns:
        Dict with counts by source and type
    """
    by_source = {}
    by_type = {}

    for event in events:
        source = event.get("source", "Unknown")
        event_type = event.get("type", "Unknown")

        by_source[source] = by_source.get(source, 0) + 1
        by_type[event_type] = by_type.get(event_type, 0) + 1

    return {
        "total": len(events),
        "by_source": by_source,
        "by_type": by_type,
    }
