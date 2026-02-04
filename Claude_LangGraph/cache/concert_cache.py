"""Concert caching system with CSV storage and freshness tracking."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Cache directory (same as events)
CACHE_DIR = Path(__file__).parent
CONCERTS_CSV = CACHE_DIR / "concerts.csv"
METADATA_FILE = CACHE_DIR / "cache_metadata.json"

# CSV columns for concerts
CSV_COLUMNS = [
    "artist",
    "artist_source",
    "artist_ytmusic_id",
    "artist_tm_id",
    "liked_songs",
    "event_id",
    "event_name",
    "date",
    "time",
    "venue",
    "city",
    "state",
    "url",
    "price_min",
    "price_max",
    "status",
    "travel_minutes",
]


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
    """Read concerts from the CSV cache."""
    if not CONCERTS_CSV.exists():
        return []

    concerts = []
    try:
        with open(CONCERTS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                travel_mins_str = row.get("travel_minutes", "")
                travel_minutes = None
                if travel_mins_str and travel_mins_str not in ("", "None"):
                    try:
                        travel_minutes = int(travel_mins_str)
                    except ValueError:
                        pass

                concert = {
                    "artist": row.get("artist", ""),
                    "artist_source": row.get("artist_source", ""),
                    "artist_ytmusic_id": row.get("artist_ytmusic_id", ""),
                    "artist_tm_id": row.get("artist_tm_id", ""),
                    "liked_songs": int(row.get("liked_songs", 0) or 0),
                    "event_id": row.get("event_id", ""),
                    "event_name": row.get("event_name", ""),
                    "date": row.get("date", ""),
                    "time": row.get("time", ""),
                    "venue": row.get("venue", ""),
                    "city": row.get("city", ""),
                    "state": row.get("state", ""),
                    "url": row.get("url", ""),
                    "price_min": float(row.get("price_min")) if row.get("price_min") else None,
                    "price_max": float(row.get("price_max")) if row.get("price_max") else None,
                    "status": row.get("status", ""),
                    "travel_minutes": travel_minutes,
                }
                concerts.append(concert)
    except (IOError, csv.Error) as e:
        print(f"Warning: Error reading concert cache: {e}")
        return []

    return concerts


def write_concerts_to_cache(concerts: list[dict]):
    """Write concerts to the CSV cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
                future_concerts.append(concert)  # Keep if date parsing fails
        else:
            future_concerts.append(concert)

    # Sort by date
    future_concerts.sort(key=lambda x: x.get("date", "9999-99-99"))

    with open(CONCERTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for concert in future_concerts:
            travel_mins = concert.get("travel_minutes")
            row = {
                "artist": concert.get("artist", ""),
                "artist_source": concert.get("artist_source", ""),
                "artist_ytmusic_id": concert.get("artist_ytmusic_id", ""),
                "artist_tm_id": concert.get("artist_tm_id", ""),
                "liked_songs": concert.get("liked_songs", 0),
                "event_id": concert.get("event_id", ""),
                "event_name": concert.get("event_name", ""),
                "date": concert.get("date", ""),
                "time": concert.get("time", ""),
                "venue": concert.get("venue", ""),
                "city": concert.get("city", ""),
                "state": concert.get("state", ""),
                "url": concert.get("url", ""),
                "price_min": concert.get("price_min") if concert.get("price_min") else "",
                "price_max": concert.get("price_max") if concert.get("price_max") else "",
                "status": concert.get("status", ""),
                "travel_minutes": str(travel_mins) if travel_mins is not None else "",
            }
            writer.writerow(row)

    mark_concerts_updated()


def clear_concerts_cache():
    """Clear concert cache."""
    if CONCERTS_CSV.exists():
        CONCERTS_CSV.unlink()


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
