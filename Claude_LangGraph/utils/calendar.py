"""Google Calendar API for checking event conflicts."""

from datetime import datetime, timedelta

from googleapiclient.discovery import build

from .google_auth import get_credentials, is_authenticated, setup_auth


def get_calendar_service():
    """Get authenticated Google Calendar service."""
    creds = get_credentials()
    if not creds:
        return None
    return build("calendar", "v3", credentials=creds)


def get_events_in_range(
    start_time: datetime,
    end_time: datetime,
    calendar_id: str = "primary",
) -> list[dict]:
    """
    Get calendar events in a time range.

    Args:
        start_time: Start of range (timezone-aware datetime)
        end_time: End of range (timezone-aware datetime)
        calendar_id: Calendar ID (default "primary")

    Returns:
        List of event dicts with start, end, summary
    """
    service = get_calendar_service()
    if not service:
        return []

    try:
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = events_result.get("items", [])
        return [
            {
                "id": e.get("id"),
                "summary": e.get("summary", "Busy"),
                "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
                "end": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date"),
                "all_day": "date" in e.get("start", {}),
            }
            for e in events
        ]
    except Exception as e:
        print(f"Error fetching calendar events: {e}")
        return []


def has_conflict(
    event_start: datetime,
    event_end: datetime | None = None,
    buffer_minutes: int = 30,
) -> tuple[bool, list[dict]]:
    """
    Check if an event time conflicts with existing calendar events.

    Args:
        event_start: Event start time (timezone-aware)
        event_end: Event end time (if None, assumes 2 hours)
        buffer_minutes: Buffer time before/after event to check

    Returns:
        Tuple of (has_conflict, list of conflicting events)
    """
    if event_end is None:
        event_end = event_start + timedelta(hours=2)

    # Add buffer
    check_start = event_start - timedelta(minutes=buffer_minutes)
    check_end = event_end + timedelta(minutes=buffer_minutes)

    calendar_events = get_events_in_range(check_start, check_end)

    conflicts = []
    for cal_event in calendar_events:
        cal_start_str = cal_event["start"]
        cal_end_str = cal_event["end"]

        try:
            if cal_event["all_day"]:
                cal_date = datetime.fromisoformat(cal_start_str).date()
                if event_start.date() == cal_date:
                    conflicts.append(cal_event)
            else:
                cal_start = datetime.fromisoformat(cal_start_str)
                cal_end = datetime.fromisoformat(cal_end_str)

                if cal_start < event_end and cal_end > event_start:
                    conflicts.append(cal_event)
        except Exception:
            continue

    return len(conflicts) > 0, conflicts


def check_event_conflicts(events: list[dict]) -> list[dict]:
    """
    Check a list of events for calendar conflicts.

    Adds '_has_conflict' and '_conflicts' fields to each event.
    """
    service = get_calendar_service()
    if not service:
        for event in events:
            event["_has_conflict"] = None
            event["_conflicts"] = []
        return events

    for event in events:
        event_dt = event.get("datetime")
        if not event_dt:
            event["_has_conflict"] = None
            event["_conflicts"] = []
            continue

        has_conf, conflicts = has_conflict(event_dt)
        event["_has_conflict"] = has_conf
        event["_conflicts"] = conflicts

    return events


def is_calendar_authenticated() -> bool:
    """Check if we have valid Google authentication."""
    return is_authenticated()


def test_calendar():
    """Test the calendar integration."""
    print("Testing Google Calendar integration...")
    print()

    if not is_authenticated():
        print("Not authenticated. Running setup...")
        if not setup_auth():
            return

    print("Fetching next 7 days of events...")
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    week_later = now + timedelta(days=7)

    events = get_events_in_range(now, week_later)
    print(f"\nFound {len(events)} events:")

    for event in events[:10]:
        print(f"  - {event['summary']}")
        print(f"    {event['start']} to {event['end']}")


if __name__ == "__main__":
    test_calendar()
