"""Concert caching system with Google Sheets storage and freshness tracking."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sheets import read_concerts_from_sheet, write_concerts_to_sheet

# Metadata file for freshness tracking
CACHE_DIR = Path(__file__).parent
METADATA_FILE = CACHE_DIR / "cache_metadata.json"


def _load_metadata() -> dict:
    """Load cache metadata."""
    if not METADATA_FILE.exists():
        return {"sources": {}, "concerts": {}}

    try:
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
            if "concerts" not in data:
                data["concerts"] = {}
            return data
    except (json.JSONDecodeError, IOError):
        return {"sources": {}, "concerts": {}}


def _save_metadata(metadata: dict):
    """Save cache metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def get_concerts_last_updated() -> datetime | None:
    """Get the last update time for concerts."""
    metadata = _load_metadata()
    timestamp = metadata.get("concerts", {}).get("last_updated")

    if timestamp:
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    return None


def is_concerts_cache_fresh(threshold_days: int = 7) -> bool:
    """Check if concert cache is still fresh."""
    last_updated = get_concerts_last_updated()

    if last_updated is None:
        return False

    now = datetime.now(ZoneInfo("America/New_York"))
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=ZoneInfo("America/New_York"))

    age = now - last_updated
    return age < timedelta(days=threshold_days)


def mark_concerts_updated():
    """Mark concerts cache as updated now."""
    metadata = _load_metadata()
    if "concerts" not in metadata:
        metadata["concerts"] = {}

    metadata["concerts"]["last_updated"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    _save_metadata(metadata)


def read_cached_concerts() -> list[dict]:
    """Read concerts from Google Sheets cache."""
    return read_concerts_from_sheet()


def write_concerts_to_cache(concerts: list[dict]):
    """Write concerts to Google Sheets cache."""
    write_concerts_to_sheet(concerts)
    mark_concerts_updated()


def clear_concerts_cache():
    """Clear concerts metadata (sheets data must be cleared manually)."""
    metadata = _load_metadata()
    if "concerts" in metadata:
        metadata["concerts"] = {}
        _save_metadata(metadata)


def get_concerts_cache_summary() -> dict:
    """Get a summary of the concert cache."""
    metadata = _load_metadata()
    concerts = read_cached_concerts()

    # Count by artist
    by_artist = {}
    for concert in concerts:
        artist = concert.get("artist", "Unknown")
        by_artist[artist] = by_artist.get(artist, 0) + 1

    return {
        "total_concerts": len(concerts),
        "concerts_by_artist": by_artist,
        "last_updated": metadata.get("concerts", {}).get("last_updated"),
    }
