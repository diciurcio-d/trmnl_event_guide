"""Google Sheets storage for concert search results.

Replaces the JSON-based .concert_results.json with Google Sheets storage.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from googleapiclient.discovery import build
from utils.google_auth import get_credentials


# Config paths
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"

# Column definitions for concert results sheet
CONCERT_COLUMNS = [
    "artist_name",
    "event_name",
    "venue_name",
    "venue_city",
    "venue_state",
    "date",
    "time",
    "url",
    "price_range",
    "tm_event_id",
    "tm_attraction_id",
    "matched_via",       # "ticketmaster", "venue_event"
    "search_date",
]


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


def _get_sheets_service():
    """Get authenticated Google Sheets service."""
    creds = get_credentials()
    if not creds:
        return None
    return build("sheets", "v4", credentials=creds)


def _create_spreadsheet(title: str) -> str | None:
    """Create a new Google Spreadsheet and return its ID."""
    service = _get_sheets_service()
    if not service:
        return None

    spreadsheet = {"properties": {"title": title}}
    result = service.spreadsheets().create(body=spreadsheet).execute()
    return result.get("spreadsheetId")


def get_or_create_concerts_sheet() -> str | None:
    """Get or create the concert results spreadsheet."""
    config = _load_sheets_config()

    # Check for existing concerts_sheet_id
    if "concerts_sheet_id" in config:
        return config["concerts_sheet_id"]

    creds = get_credentials()
    if not creds:
        print("Not authenticated with Google. Run: python -m utils.google_auth")
        return None

    print("Creating new Concert Results spreadsheet...")
    sheet_id = _create_spreadsheet("Concert Search Results")
    if sheet_id:
        config["concerts_sheet_id"] = sheet_id
        _save_sheets_config(config)

        # Add header row
        service = _get_sheets_service()
        if service:
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range="A1",
                valueInputOption="RAW",
                body={"values": [CONCERT_COLUMNS]}
            ).execute()

        print(f"Created Concerts sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def read_concert_results() -> list[dict]:
    """Read all concert results from the Google Sheet."""
    sheet_id = get_or_create_concerts_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:M"
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:  # Only header or empty
            return []

        header = rows[0]
        concerts = []

        for row in rows[1:]:
            # Pad row to match header length
            while len(row) < len(header):
                row.append("")

            concert = dict(zip(header, row))
            concerts.append(concert)

        return concerts

    except Exception as e:
        print(f"Error reading concert results from sheet: {e}")
        return []


def write_concert_results(concerts: list[dict], append: bool = False):
    """
    Write concert results to the Google Sheet.

    Args:
        concerts: List of concert dicts with keys matching CONCERT_COLUMNS
        append: If True, append to existing data. If False, replace all.
    """
    sheet_id = get_or_create_concerts_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    # Get search date
    search_date = datetime.now(ZoneInfo("America/New_York")).isoformat()

    # If appending, read existing data first
    existing = []
    if append:
        existing = read_concert_results()

    # Build deduplication key set
    existing_keys = set()
    for c in existing:
        key = (
            c.get("artist_name", "").lower(),
            c.get("event_name", "").lower(),
            c.get("date", ""),
            c.get("venue_name", "").lower(),
        )
        existing_keys.add(key)

    # Filter to new concerts and add search date
    new_concerts = []
    for concert in concerts:
        key = (
            concert.get("artist_name", "").lower(),
            concert.get("event_name", "").lower(),
            concert.get("date", ""),
            concert.get("venue_name", "").lower(),
        )
        if key not in existing_keys:
            concert["search_date"] = search_date
            new_concerts.append(concert)
            existing_keys.add(key)

    # Combine
    all_concerts = existing + new_concerts

    # Convert to rows
    rows = [CONCERT_COLUMNS]
    for concert in all_concerts:
        row = [
            concert.get("artist_name", ""),
            concert.get("event_name", ""),
            concert.get("venue_name", ""),
            concert.get("venue_city", ""),
            concert.get("venue_state", ""),
            concert.get("date", ""),
            concert.get("time", ""),
            concert.get("url", ""),
            concert.get("price_range", ""),
            concert.get("tm_event_id", ""),
            concert.get("tm_attraction_id", ""),
            concert.get("matched_via", ""),
            concert.get("search_date", ""),
        ]
        rows.append(row)

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:M"
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        if append:
            print(f"Added {len(new_concerts)} new concerts (total: {len(all_concerts)})")
        else:
            print(f"Wrote {len(all_concerts)} concerts to sheet")

    except Exception as e:
        print(f"Error writing concert results to sheet: {e}")


def clear_concert_results():
    """Clear all concert results from the sheet."""
    sheet_id = get_or_create_concerts_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:M"
        ).execute()

        # Re-add header
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": [CONCERT_COLUMNS]}
        ).execute()

        print("Cleared concert results")

    except Exception as e:
        print(f"Error clearing concert results: {e}")


def migrate_from_json():
    """
    Migrate concert results from .concert_results.json to Google Sheets.

    This is a one-time migration function.
    """
    json_file = Path(__file__).parent / ".concert_results.json"
    if not json_file.exists():
        print("No .concert_results.json found to migrate")
        return

    with open(json_file) as f:
        data = json.load(f)

    search_date = data.get("search_date", "")
    artists_checked = data.get("artists_checked", [])

    # Build concert list from artists data
    concerts = []
    for artist in artists_checked:
        artist_name = artist.get("name", "")
        tm_attraction_id = artist.get("tm_attraction_id", "")

        for event in artist.get("events", []):
            concert = {
                "artist_name": artist_name,
                "event_name": event.get("name", ""),
                "venue_name": event.get("venue", {}).get("name", ""),
                "venue_city": event.get("venue", {}).get("city", ""),
                "venue_state": event.get("venue", {}).get("state", ""),
                "date": event.get("date", ""),
                "time": event.get("time", ""),
                "url": event.get("url", ""),
                "price_range": event.get("price_range", ""),
                "tm_event_id": event.get("id", ""),
                "tm_attraction_id": tm_attraction_id,
                "matched_via": "ticketmaster",
                "search_date": search_date,
            }
            concerts.append(concert)

    if concerts:
        write_concert_results(concerts, append=False)
        print(f"Migrated {len(concerts)} concerts from JSON to Google Sheets")

        # Rename the old file
        backup_file = json_file.with_suffix(".json.bak")
        json_file.rename(backup_file)
        print(f"Renamed {json_file} to {backup_file}")
    else:
        print("No concerts found in JSON file")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_from_json()
    else:
        concerts = read_concert_results()
        print(f"Found {len(concerts)} concert results in sheet")
        for concert in concerts[:5]:
            print(f"  - {concert.get('artist_name')}: {concert.get('event_name')} at {concert.get('venue_name')}")
