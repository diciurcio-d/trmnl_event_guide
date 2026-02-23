"""Google Sheets API for storing events and concerts."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .google_auth import is_authenticated
from .sheets_core import (
    create_spreadsheet as _core_create_spreadsheet,
    get_sheets_service as _core_get_sheets_service,
    load_sheets_config as _core_load_sheets_config,
    save_sheets_config as _core_save_sheets_config,
    write_sheet_header as _core_write_sheet_header,
)

# Config file for sheet IDs
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"

# Column definitions
EVENTS_COLUMNS = [
    "name", "datetime", "date_str", "type", "sold_out", "source",
    "location", "description", "has_specific_time", "url", "travel_minutes"
]

CONCERTS_COLUMNS = [
    "artist", "artist_source", "artist_ytmusic_id", "artist_tm_id",
    "liked_songs", "event_id", "event_name", "date", "time", "venue",
    "city", "state", "url", "price_min", "price_max", "status", "travel_minutes"
]


def _get_sheets_service():
    """Get authenticated Google Sheets service."""
    return _core_get_sheets_service()


def _load_sheets_config() -> dict:
    """Load sheet IDs from config."""
    return _core_load_sheets_config(_SHEETS_CONFIG)


def _save_sheets_config(config: dict):
    """Save sheet IDs to config."""
    _core_save_sheets_config(config, _SHEETS_CONFIG)


def create_spreadsheet(title: str) -> str | None:
    """Create a new Google Spreadsheet and return its ID."""
    return _core_create_spreadsheet(title)


def get_or_create_events_sheet() -> str | None:
    """Get or create the events spreadsheet."""
    config = _load_sheets_config()

    if "events_sheet_id" in config:
        return config["events_sheet_id"]

    print("Creating new Events spreadsheet...")
    sheet_id = create_spreadsheet("NYC Events Cache")
    if sheet_id:
        config["events_sheet_id"] = sheet_id
        _save_sheets_config(config)

        # Add header row
        _write_header(sheet_id, EVENTS_COLUMNS)
        print(f"Created Events sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def get_or_create_concerts_sheet() -> str | None:
    """Get or create the concerts spreadsheet."""
    config = _load_sheets_config()

    if "concerts_sheet_id" in config:
        return config["concerts_sheet_id"]

    print("Creating new Concerts spreadsheet...")
    sheet_id = create_spreadsheet("NYC Concerts Cache")
    if sheet_id:
        config["concerts_sheet_id"] = sheet_id
        _save_sheets_config(config)

        # Add header row
        _write_header(sheet_id, CONCERTS_COLUMNS)
        print(f"Created Concerts sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def _write_header(sheet_id: str, columns: list[str]):
    """Write header row to a sheet."""
    _core_write_sheet_header(sheet_id, columns)


def read_events_from_sheet(source_name: str | None = None) -> list[dict]:
    """Read events from Google Sheet."""
    sheet_id = get_or_create_events_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:K"  # All columns
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:  # Only header or empty
            return []

        header = rows[0]
        events = []

        for row in rows[1:]:
            # Pad row to match header length
            while len(row) < len(header):
                row.append("")

            event = dict(zip(header, row))

            # Filter by source if specified
            if source_name and event.get("source") != source_name:
                continue

            # Convert types
            event["sold_out"] = event.get("sold_out", "").lower() == "true"
            event["has_specific_time"] = event.get("has_specific_time", "").lower() == "true"

            # Parse datetime
            dt_str = event.get("datetime", "")
            if dt_str and dt_str != "None" and dt_str != "":
                try:
                    dt = datetime.fromisoformat(dt_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                    event["datetime"] = dt
                except ValueError:
                    event["datetime"] = None
            else:
                event["datetime"] = None

            # Parse travel_minutes
            travel_str = event.get("travel_minutes", "")
            if travel_str and travel_str not in ("", "None"):
                try:
                    event["travel_minutes"] = int(travel_str)
                except ValueError:
                    event["travel_minutes"] = None
            else:
                event["travel_minutes"] = None

            events.append(event)

        return events

    except Exception as e:
        print(f"Error reading events from sheet: {e}")
        return []


def write_events_to_sheet(events: list[dict], source_name: str | None = None):
    """Write events to Google Sheet."""
    sheet_id = get_or_create_events_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    # If updating a specific source, merge with existing data
    if source_name:
        existing = read_events_from_sheet()
        # Remove old events from this source
        existing = [e for e in existing if e.get("source") != source_name]
        # Add new events
        new_source_events = [e for e in events if e.get("source") == source_name]
        all_events = existing + new_source_events
    else:
        all_events = events

    # Sort by source then date
    far_future = datetime(2099, 12, 31, tzinfo=ZoneInfo("America/New_York"))
    all_events.sort(
        key=lambda x: (
            x.get("source", ""),
            -(x["datetime"].timestamp() if x.get("datetime") else far_future.timestamp())
        )
    )

    # Convert to rows
    rows = [EVENTS_COLUMNS]  # Header
    for event in all_events:
        row = [
            event.get("name", ""),
            event.get("datetime").isoformat() if event.get("datetime") else "",
            event.get("date_str", ""),
            event.get("type", ""),
            str(event.get("sold_out", False)),
            event.get("source", ""),
            event.get("location", ""),
            event.get("description", ""),
            str(event.get("has_specific_time", False)),
            event.get("url", ""),
            str(event.get("travel_minutes")) if event.get("travel_minutes") is not None else "",
        ]
        rows.append(row)

    # Clear and write
    try:
        # Clear existing data
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:K"
        ).execute()

        # Write new data
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

    except Exception as e:
        print(f"Error writing events to sheet: {e}")


def read_concerts_from_sheet() -> list[dict]:
    """Read concerts from Google Sheet."""
    sheet_id = get_or_create_concerts_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:Q"  # All columns
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:
            return []

        header = rows[0]
        concerts = []

        for row in rows[1:]:
            while len(row) < len(header):
                row.append("")

            concert = dict(zip(header, row))

            # Convert types
            concert["liked_songs"] = int(concert.get("liked_songs", 0) or 0)

            price_min = concert.get("price_min", "")
            concert["price_min"] = float(price_min) if price_min else None

            price_max = concert.get("price_max", "")
            concert["price_max"] = float(price_max) if price_max else None

            travel_str = concert.get("travel_minutes", "")
            if travel_str and travel_str not in ("", "None"):
                try:
                    concert["travel_minutes"] = int(travel_str)
                except ValueError:
                    concert["travel_minutes"] = None
            else:
                concert["travel_minutes"] = None

            concerts.append(concert)

        return concerts

    except Exception as e:
        print(f"Error reading concerts from sheet: {e}")
        return []


def write_concerts_to_sheet(concerts: list[dict]):
    """Write concerts to Google Sheet."""
    sheet_id = get_or_create_concerts_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    # Filter out past concerts
    today = datetime.now(ZoneInfo("America/New_York")).date()
    future_concerts = []
    for concert in concerts:
        if concert.get("date"):
            try:
                concert_date = datetime.strptime(concert["date"], "%Y-%m-%d").date()
                if concert_date >= today:
                    future_concerts.append(concert)
            except ValueError:
                future_concerts.append(concert)
        else:
            future_concerts.append(concert)

    # Sort by date
    future_concerts.sort(key=lambda x: x.get("date", "9999-99-99"))

    # Convert to rows
    rows = [CONCERTS_COLUMNS]  # Header
    for concert in future_concerts:
        row = [
            concert.get("artist", ""),
            concert.get("artist_source", ""),
            concert.get("artist_ytmusic_id", ""),
            concert.get("artist_tm_id", ""),
            str(concert.get("liked_songs", 0)),
            concert.get("event_id", ""),
            concert.get("event_name", ""),
            concert.get("date", ""),
            concert.get("time", ""),
            concert.get("venue", ""),
            concert.get("city", ""),
            concert.get("state", ""),
            concert.get("url", ""),
            str(concert.get("price_min")) if concert.get("price_min") else "",
            str(concert.get("price_max")) if concert.get("price_max") else "",
            concert.get("status", ""),
            str(concert.get("travel_minutes")) if concert.get("travel_minutes") is not None else "",
        ]
        rows.append(row)

    # Clear and write
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
        print(f"Error writing concerts to sheet: {e}")


def test_sheets():
    """Test the Sheets integration."""
    print("Testing Google Sheets integration...")

    if not is_authenticated():
        print("Not authenticated. Run: python -m utils.google_auth")
        return

    # Test events sheet
    events_id = get_or_create_events_sheet()
    print(f"Events sheet: {events_id}")

    # Test concerts sheet
    concerts_id = get_or_create_concerts_sheet()
    print(f"Concerts sheet: {concerts_id}")

    # Read current data
    events = read_events_from_sheet()
    concerts = read_concerts_from_sheet()
    print(f"\nCurrent data: {len(events)} events, {len(concerts)} concerts")


if __name__ == "__main__":
    test_sheets()
