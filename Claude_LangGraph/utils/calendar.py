"""Google Calendar API for checking event conflicts."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Scopes for read-only calendar access
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Paths for credentials
_CONFIG_DIR = Path(__file__).parent.parent / "concert_finder"
_CREDENTIALS_FILE = _CONFIG_DIR / "calendar_credentials.json"
_TOKEN_FILE = _CONFIG_DIR / "calendar_token.json"


def _get_credentials() -> Credentials | None:
    """Get valid Google Calendar credentials, refreshing or prompting auth as needed."""
    creds = None

    # Load existing token
    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            if not _CREDENTIALS_FILE.exists():
                print(f"Calendar credentials file not found: {_CREDENTIALS_FILE}")
                print("Please download OAuth credentials from Google Cloud Console")
                print("and save as 'calendar_credentials.json' in the concert_finder folder.")
                return None

            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for future use
        with open(_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return creds


def get_calendar_service():
    """Get authenticated Google Calendar service."""
    creds = _get_credentials()
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
        # Parse calendar event times
        cal_start_str = cal_event["start"]
        cal_end_str = cal_event["end"]

        try:
            if cal_event["all_day"]:
                # All-day events - check date overlap
                cal_date = datetime.fromisoformat(cal_start_str).date()
                if event_start.date() == cal_date:
                    conflicts.append(cal_event)
            else:
                cal_start = datetime.fromisoformat(cal_start_str)
                cal_end = datetime.fromisoformat(cal_end_str)

                # Check for time overlap
                if cal_start < event_end and cal_end > event_start:
                    conflicts.append(cal_event)
        except Exception:
            continue

    return len(conflicts) > 0, conflicts


def check_event_conflicts(events: list[dict]) -> list[dict]:
    """
    Check a list of events for calendar conflicts.

    Adds '_has_conflict' and '_conflicts' fields to each event.

    Args:
        events: List of event dicts with 'datetime' field

    Returns:
        Same list with conflict info added
    """
    service = get_calendar_service()
    if not service:
        # Can't check conflicts, return events unchanged
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


def is_calendar_configured() -> bool:
    """Check if calendar credentials are set up."""
    return _CREDENTIALS_FILE.exists()


def is_calendar_authenticated() -> bool:
    """Check if we have valid calendar authentication."""
    if not _TOKEN_FILE.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)
        return creds and creds.valid
    except Exception:
        return False


def setup_calendar_auth():
    """Interactive setup for Google Calendar authentication."""
    print("Setting up Google Calendar integration...")
    print()

    if not _CREDENTIALS_FILE.exists():
        print("Step 1: Download OAuth credentials")
        print("  1. Go to https://console.cloud.google.com/apis/credentials")
        print("  2. Create an OAuth 2.0 Client ID (Desktop app)")
        print("  3. Download the JSON file")
        print(f"  4. Save it as: {_CREDENTIALS_FILE}")
        print()
        input("Press Enter when ready...")

    if not _CREDENTIALS_FILE.exists():
        print("Credentials file still not found. Please try again.")
        return False

    print("\nStep 2: Authorize calendar access")
    print("A browser window will open for you to authorize access...")
    print()

    creds = _get_credentials()
    if creds:
        print("\nCalendar authentication successful!")
        return True
    else:
        print("\nCalendar authentication failed.")
        return False


def test_calendar():
    """Test the calendar integration."""
    print("Testing Google Calendar integration...")
    print()

    if not is_calendar_configured():
        print("Calendar not configured. Run setup_calendar_auth() first.")
        return

    if not is_calendar_authenticated():
        print("Calendar not authenticated. Attempting auth...")
        if not setup_calendar_auth():
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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_calendar_auth()
    else:
        test_calendar()
