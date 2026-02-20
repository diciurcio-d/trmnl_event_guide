"""Google Sheets storage for venue events."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.google_auth import get_credentials, is_authenticated


_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"


def _safe_lower(val) -> str:
    """Safely convert value to lowercase string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.lower()
    return str(val).lower()


VENUE_EVENTS_COLUMNS = [
    "name",
    "datetime",
    "date_str",
    "venue_name",
    "event_type",
    "url",
    "source",
    "matched_artist",
    "travel_minutes",
    "description",
    "event_source_url",
    "extraction_method",
    "relevance_score",
    "validation_confidence",
    "date_added",
]


def normalize_event(event: dict) -> dict:
    """Normalize a venue event into the canonical schema."""
    normalized = dict(event)
    normalized["name"] = normalized.get("name", "")
    normalized["datetime"] = normalized.get("datetime")
    normalized["date_str"] = normalized.get("date_str", "")
    normalized["venue_name"] = normalized.get("venue_name", "")
    normalized["event_type"] = normalized.get("event_type", "")
    normalized["url"] = normalized.get("url", "")
    normalized["source"] = normalized.get("source", "")
    normalized["matched_artist"] = normalized.get("matched_artist", "")
    normalized["travel_minutes"] = normalized.get("travel_minutes")
    normalized["description"] = normalized.get("description", "")
    normalized["event_source_url"] = normalized.get("event_source_url", "")
    normalized["extraction_method"] = normalized.get("extraction_method", "")
    normalized["relevance_score"] = normalized.get("relevance_score")
    normalized["validation_confidence"] = normalized.get("validation_confidence")
    normalized["date_added"] = normalized.get("date_added", "")
    return normalized


def _get_sheets_service():
    """Get authenticated Google Sheets service."""
    creds = get_credentials()
    if not creds:
        return None
    return build("sheets", "v4", credentials=creds)


def _load_sheets_config() -> dict:
    """Load sheet IDs from config."""
    if _SHEETS_CONFIG.exists():
        with open(_SHEETS_CONFIG) as f:
            return json.load(f)
    return {}


def _save_sheets_config(config: dict):
    """Save sheet IDs to config."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SHEETS_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


def _create_spreadsheet(title: str) -> str | None:
    """Create a new Google Spreadsheet and return its ID."""
    service = _get_sheets_service()
    if not service:
        return None

    spreadsheet = {"properties": {"title": title}}
    result = service.spreadsheets().create(body=spreadsheet).execute()
    return result.get("spreadsheetId")


def _write_header(sheet_id: str, columns: list[str]):
    """Write header row to a sheet."""
    service = _get_sheets_service()
    if not service:
        return

    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range="A1",
        valueInputOption="RAW",
        body={"values": [columns]},
    ).execute()


def get_or_create_venue_events_sheet() -> str | None:
    """Get or create the venue events spreadsheet."""
    config = _load_sheets_config()

    if "venue_events_sheet_id" in config:
        return config["venue_events_sheet_id"]

    creds = get_credentials()
    if not creds:
        print("Not authenticated with Google. Run: python -m utils.google_auth")
        return None

    print("Creating new Venue Events spreadsheet...")
    sheet_id = _create_spreadsheet("Venue Events Cache")
    if sheet_id:
        config["venue_events_sheet_id"] = sheet_id
        _save_sheets_config(config)
        _write_header(sheet_id, VENUE_EVENTS_COLUMNS)
        print(f"Created Venue Events sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def read_venue_events_from_sheet(venue_name: str | None = None) -> list[dict]:
    """Read venue events from Google Sheet."""
    sheet_id = get_or_create_venue_events_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:O",
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:
            return []

        header = rows[0]
        events = []

        for row in rows[1:]:
            while len(row) < len(header):
                row.append("")

            event = normalize_event(dict(zip(header, row)))

            if venue_name and _safe_lower(event.get("venue_name", "")) != venue_name.lower():
                continue

            dt_str = event.get("datetime", "")
            if dt_str and dt_str not in ("None", ""):
                try:
                    dt = datetime.fromisoformat(dt_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                    event["datetime"] = dt
                except ValueError:
                    event["datetime"] = None
            else:
                event["datetime"] = None

            travel_str = event.get("travel_minutes", "")
            if travel_str and travel_str not in ("", "None"):
                try:
                    event["travel_minutes"] = int(travel_str)
                except ValueError:
                    event["travel_minutes"] = None
            else:
                event["travel_minutes"] = None

            score_str = event.get("relevance_score", "")
            if score_str and score_str not in ("", "None"):
                try:
                    event["relevance_score"] = int(float(score_str))
                except ValueError:
                    event["relevance_score"] = None
            else:
                event["relevance_score"] = None

            conf_str = event.get("validation_confidence", "")
            if conf_str and conf_str not in ("", "None"):
                try:
                    event["validation_confidence"] = float(conf_str)
                except ValueError:
                    event["validation_confidence"] = None
            else:
                event["validation_confidence"] = None

            events.append(event)

        return events

    except Exception as e:
        print(f"Error reading venue events from sheet: {e}")
        return []


def write_venue_events_to_sheet(events: list[dict], venue_name: str | None = None):
    """Write venue events to Google Sheet."""
    sheet_id = get_or_create_venue_events_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    if venue_name:
        existing = read_venue_events_from_sheet()
        existing = [e for e in existing if _safe_lower(e.get("venue_name", "")) != venue_name.lower()]
        venue_events = [e for e in events if _safe_lower(e.get("venue_name", "")) == venue_name.lower()]
        all_events = existing + venue_events
    else:
        all_events = events

    all_events = [normalize_event(event) for event in all_events]

    today = datetime.now(ZoneInfo("America/New_York")).date()
    future_events = []
    for event in all_events:
        dt = event.get("datetime")
        if dt:
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt)
                except ValueError:
                    future_events.append(event)
                    continue
            if dt.date() >= today:
                future_events.append(event)
        else:
            date_str = event.get("date_str", "")
            if date_str:
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if event_date >= today:
                        future_events.append(event)
                except ValueError:
                    future_events.append(event)
            else:
                future_events.append(event)

    future_events.sort(
        key=lambda x: (
            _safe_lower(x.get("venue_name", "")),
            x.get("datetime").isoformat() if x.get("datetime") else "9999",
        )
    )

    rows = [VENUE_EVENTS_COLUMNS]
    for event in future_events:
        dt = event.get("datetime")
        if isinstance(dt, datetime):
            dt_str = dt.isoformat()
        elif dt:
            dt_str = str(dt)
        else:
            dt_str = ""

        row = [
            event.get("name", ""),
            dt_str,
            event.get("date_str", ""),
            event.get("venue_name", ""),
            event.get("event_type", ""),
            event.get("url", ""),
            event.get("source", ""),
            event.get("matched_artist", ""),
            str(event.get("travel_minutes")) if event.get("travel_minutes") is not None else "",
            event.get("description", ""),
            event.get("event_source_url", ""),
            event.get("extraction_method", ""),
            str(event.get("relevance_score")) if event.get("relevance_score") is not None else "",
            str(event.get("validation_confidence")) if event.get("validation_confidence") is not None else "",
            event.get("date_added", ""),
        ]
        rows.append(row)

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:O",
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()

        print(f"Wrote {len(future_events)} venue events to sheet")

    except Exception as e:
        print(f"Error writing venue events to sheet: {e}")


def append_venue_events(events: list[dict], venue_name: str):
    """Append events for a venue, deduplicating against existing."""
    if not events:
        return

    existing = read_venue_events_from_sheet()
    existing_keys = set()
    for event in existing:
        key = (
            _safe_lower(event.get("venue_name", "")),
            _safe_lower(event.get("name", "")),
            event.get("date_str", ""),
        )
        existing_keys.add(key)

    new_events = []
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    for event in events:
        event = normalize_event(event)
        key = (
            _safe_lower(event.get("venue_name", "")),
            _safe_lower(event.get("name", "")),
            event.get("date_str", ""),
        )
        if key not in existing_keys:
            # Set date_added for new events
            if not event.get("date_added"):
                event["date_added"] = now
            new_events.append(event)
            existing_keys.add(key)

    if not new_events:
        print(f"  No new events for {venue_name} (all duplicates)")
        return

    all_events = existing + new_events
    write_venue_events_to_sheet(all_events)
    print(f"  Added {len(new_events)} new events for {venue_name}")


def get_events_by_venue() -> dict[str, list[dict]]:
    """Get all events grouped by venue."""
    events = read_venue_events_from_sheet()
    by_venue = {}
    for event in events:
        venue = event.get("venue_name", "Unknown")
        by_venue.setdefault(venue, []).append(event)
    return by_venue


def get_matched_events() -> list[dict]:
    """Get all events that have a matched artist."""
    events = read_venue_events_from_sheet()
    return [event for event in events if event.get("matched_artist")]


def test_venue_events_sheet():
    """Test the venue events sheet integration."""
    print("Testing Venue Events Sheet integration...")

    if not is_authenticated():
        print("Not authenticated. Run: python -m utils.google_auth")
        return

    sheet_id = get_or_create_venue_events_sheet()
    print(f"Venue Events sheet: {sheet_id}")

    events = read_venue_events_from_sheet()
    print(f"\nCurrent data: {len(events)} events")

    by_venue = get_events_by_venue()
    print(f"\nBy venue: {len(by_venue)} venues")
    for venue, venue_events in sorted(by_venue.items()):
        print(f"  {venue}: {len(venue_events)} events")


if __name__ == "__main__":
    test_venue_events_sheet()
