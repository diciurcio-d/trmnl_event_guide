"""Google Sheets storage for YouTube Music artists.

Replaces the JSON-based .artists_cache.json with Google Sheets storage.
"""

import json
import sys
from pathlib import Path

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from googleapiclient.discovery import build
from utils.google_auth import get_credentials


# Config paths
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"

# Column definitions for artists sheet
ARTIST_COLUMNS = [
    "name",
    "ytmusic_id",
    "subscribers",
    "source",           # "subscription", "liked_songs", "both"
    "liked_song_count",
    "tm_attraction_id", # Ticketmaster attraction ID if found
    "last_updated",
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


def get_or_create_artists_sheet() -> str | None:
    """Get or create the artists spreadsheet."""
    config = _load_sheets_config()

    if "artists_sheet_id" in config:
        return config["artists_sheet_id"]

    creds = get_credentials()
    if not creds:
        print("Not authenticated with Google. Run: python -m utils.google_auth")
        return None

    print("Creating new Artists spreadsheet...")
    sheet_id = _create_spreadsheet("YouTube Music Artists")
    if sheet_id:
        config["artists_sheet_id"] = sheet_id
        _save_sheets_config(config)

        # Add header row
        service = _get_sheets_service()
        if service:
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range="A1",
                valueInputOption="RAW",
                body={"values": [ARTIST_COLUMNS]}
            ).execute()

        print(f"Created Artists sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def read_artists_from_sheet() -> list[dict]:
    """Read all artists from the Google Sheet."""
    sheet_id = get_or_create_artists_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:G"
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:  # Only header or empty
            return []

        header = rows[0]
        artists = []

        for row in rows[1:]:
            # Pad row to match header length
            while len(row) < len(header):
                row.append("")

            artist = dict(zip(header, row))

            # Parse liked_song_count as int
            try:
                artist["liked_song_count"] = int(artist.get("liked_song_count", 0) or 0)
            except ValueError:
                artist["liked_song_count"] = 0

            artists.append(artist)

        return artists

    except Exception as e:
        print(f"Error reading artists from sheet: {e}")
        return []


def write_artists_to_sheet(artists: list[dict]):
    """
    Write artists to the Google Sheet (replaces all).

    Args:
        artists: List of artist dicts with keys matching ARTIST_COLUMNS
    """
    sheet_id = get_or_create_artists_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    # Convert to rows
    rows = [ARTIST_COLUMNS]
    for artist in artists:
        row = [
            artist.get("name", ""),
            artist.get("ytmusic_id", artist.get("id", "")),  # Support both id and ytmusic_id
            artist.get("subscribers", ""),
            artist.get("source", ""),
            str(artist.get("liked_song_count", 0)),
            artist.get("tm_attraction_id", ""),
            artist.get("last_updated", ""),
        ]
        rows.append(row)

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:G"
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        print(f"Wrote {len(artists)} artists to sheet")

    except Exception as e:
        print(f"Error writing artists to sheet: {e}")


def migrate_from_json():
    """
    Migrate artists from .artists_cache.json to Google Sheets.

    This is a one-time migration function.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    json_file = Path(__file__).parent / ".artists_cache.json"
    if not json_file.exists():
        print("No .artists_cache.json found to migrate")
        return

    with open(json_file) as f:
        data = json.load(f)

    artists = data.get("artists", [])
    if not artists:
        print("No artists in JSON file")
        return

    # Add last_updated timestamp
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    for artist in artists:
        artist["last_updated"] = now
        # Normalize id field
        if "id" in artist and "ytmusic_id" not in artist:
            artist["ytmusic_id"] = artist["id"]

    write_artists_to_sheet(artists)
    print(f"Migrated {len(artists)} artists from JSON to Google Sheets")

    # Optionally rename the old file
    backup_file = json_file.with_suffix(".json.bak")
    json_file.rename(backup_file)
    print(f"Renamed {json_file} to {backup_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_from_json()
    else:
        artists = read_artists_from_sheet()
        print(f"Found {len(artists)} artists in sheet")
        for artist in artists[:5]:
            print(f"  - {artist.get('name')}: {artist.get('liked_song_count')} liked songs")
