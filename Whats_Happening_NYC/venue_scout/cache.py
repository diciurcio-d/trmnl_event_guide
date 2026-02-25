"""Venue caching system with Google Sheets storage and freshness tracking."""

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.google_auth import get_credentials, is_authenticated
from utils.sheets_core import (
    create_spreadsheet as _core_create_spreadsheet,
    get_sheets_service as _core_get_sheets_service,
    load_sheets_config as _core_load_sheets_config,
    save_sheets_config as _core_save_sheets_config,
    write_sheet_header as _core_write_sheet_header,
)

from .state import Venue
from .paths import DATA_DIR, SEED_VENUES_FILE, VENUE_CACHE_METADATA_FILE


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()

# Config and metadata paths
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"
_CACHE_DIR = DATA_DIR
_METADATA_FILE = VENUE_CACHE_METADATA_FILE

# Column definitions for venues sheet
VENUE_COLUMNS = [
    "name", "address", "lat", "lng", "city", "neighborhood",
    "website", "events_url", "category", "description", "source", "address_verified",
    "website_status", "website_attempts",
    "preferred_event_source", "api_endpoint", "ticketmaster_venue_id",
    "cloudflare_protected",
    "feed_url", "feed_type",
    "last_event_fetch", "event_count", "event_source",
]


def _sheet_col_label(col_index: int) -> str:
    """Convert a 1-based column index to A1 column label."""
    if col_index < 1:
        return "A"

    out = ""
    value = col_index
    while value:
        value, rem = divmod(value - 1, 26)
        out = chr(65 + rem) + out
    return out


def _sheet_full_range() -> str:
    """Return the full A1 range for all venue columns."""
    return f"A:{_sheet_col_label(len(VENUE_COLUMNS))}"


def _parse_coordinate(value) -> float | None:
    """Parse a sheet coordinate cell into float (or None)."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _format_coordinate(value) -> str:
    """Format a coordinate for sheet storage."""
    parsed = _parse_coordinate(value)
    if parsed is None:
        return ""
    return f"{parsed:.7f}".rstrip("0").rstrip(".")


def _venue_to_row(venue: dict) -> list:
    """Convert a venue dict to a row for the sheet."""
    return [
        venue.get("name", ""),
        venue.get("address", ""),
        _format_coordinate(venue.get("lat")),
        _format_coordinate(venue.get("lng")),
        venue.get("city", ""),
        venue.get("neighborhood", ""),
        venue.get("website", ""),
        venue.get("events_url", ""),
        venue.get("category", ""),
        venue.get("description", ""),
        venue.get("source", ""),
        venue.get("address_verified", ""),
        venue.get("website_status", ""),
        str(venue.get("website_attempts", 0) or ""),
        venue.get("preferred_event_source", ""),
        venue.get("api_endpoint", ""),
        venue.get("ticketmaster_venue_id", ""),
        venue.get("cloudflare_protected", ""),
        venue.get("feed_url", ""),
        venue.get("feed_type", ""),
        venue.get("last_event_fetch", ""),
        str(venue.get("event_count", 0) or ""),
        venue.get("event_source", ""),
    ]


def _get_sheets_service():
    """Get authenticated Google Sheets service."""
    return _core_get_sheets_service()


def _load_sheets_config() -> dict:
    """Load sheet IDs from config."""
    return _core_load_sheets_config(_SHEETS_CONFIG)


def _save_sheets_config(config: dict):
    """Save sheet IDs to config."""
    _core_save_sheets_config(config, _SHEETS_CONFIG, ensure_dir=True)


def _load_metadata() -> dict:
    """Load cache metadata (last update times per city and category)."""
    if not _METADATA_FILE.exists():
        return {"cities": {}, "categories": {}}

    try:
        with open(_METADATA_FILE, "r") as f:
            data = json.load(f)
            # Ensure categories key exists
            if "categories" not in data:
                data["categories"] = {}
            return data
    except (json.JSONDecodeError, IOError):
        return {"cities": {}, "categories": {}}


def _save_metadata(metadata: dict):
    """Save cache metadata."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _create_spreadsheet(title: str) -> str | None:
    """Create a new Google Spreadsheet and return its ID."""
    return _core_create_spreadsheet(title)


def _write_header(sheet_id: str, columns: list[str]):
    """Write header row to a sheet."""
    _core_write_sheet_header(sheet_id, columns)


def get_or_create_venues_sheet() -> str | None:
    """Get or create the venues spreadsheet."""
    config = _load_sheets_config()

    if "venues_sheet_id" in config:
        return config["venues_sheet_id"]

    # Try to get credentials (will refresh if needed)
    creds = get_credentials()
    if not creds:
        print("Not authenticated with Google. Run: python -m utils.google_auth")
        return None

    print("Creating new Venues spreadsheet...")
    sheet_id = _create_spreadsheet("Venue Scout Cache")
    if sheet_id:
        config["venues_sheet_id"] = sheet_id
        _save_sheets_config(config)

        # Add header row
        _write_header(sheet_id, VENUE_COLUMNS)
        print(f"Created Venues sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def get_city_last_updated(city: str) -> datetime | None:
    """Get the last update time for a city's venues."""
    metadata = _load_metadata()
    # Normalize city name
    city_key = city.lower().strip()
    timestamp = metadata.get("cities", {}).get(city_key)

    if timestamp:
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    return None


def is_city_fresh(city: str, threshold_days: int | None = None) -> bool:
    """Check if a city's cached venue data is still fresh."""
    if threshold_days is None:
        threshold_days = _settings.VENUE_CACHE_THRESHOLD_DAYS

    last_updated = get_city_last_updated(city)

    if last_updated is None:
        return False

    now = datetime.now(ZoneInfo("America/New_York"))
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=ZoneInfo("America/New_York"))

    age = now - last_updated
    return age < timedelta(days=threshold_days)


def mark_city_updated(city: str):
    """Mark a city's venues as updated now."""
    metadata = _load_metadata()
    if "cities" not in metadata:
        metadata["cities"] = {}

    city_key = city.lower().strip()
    metadata["cities"][city_key] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    _save_metadata(metadata)


def _category_key(city: str, category: str) -> str:
    """Create a unique key for city+category combination."""
    return f"{city.lower().strip()}:{category.lower().strip()}"


def is_category_searched(city: str, category: str, threshold_days: int | None = None) -> bool:
    """Check if a category has already been searched for a city."""
    if threshold_days is None:
        threshold_days = _settings.VENUE_CACHE_THRESHOLD_DAYS

    metadata = _load_metadata()
    key = _category_key(city, category)
    timestamp = metadata.get("categories", {}).get(key)

    if not timestamp:
        return False

    try:
        last_searched = datetime.fromisoformat(timestamp)
        now = datetime.now(ZoneInfo("America/New_York"))
        if last_searched.tzinfo is None:
            last_searched = last_searched.replace(tzinfo=ZoneInfo("America/New_York"))
        age = now - last_searched
        return age < timedelta(days=threshold_days)
    except ValueError:
        return False


def mark_category_searched(city: str, category: str):
    """Mark a category as searched for a city."""
    metadata = _load_metadata()
    if "categories" not in metadata:
        metadata["categories"] = {}

    key = _category_key(city, category)
    metadata["categories"][key] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    _save_metadata(metadata)


def get_searched_categories(city: str) -> list[str]:
    """Get list of categories already searched for a city."""
    metadata = _load_metadata()
    city_prefix = f"{city.lower().strip()}:"
    searched = []

    for key in metadata.get("categories", {}).keys():
        if key.startswith(city_prefix):
            category = key[len(city_prefix):]
            if is_category_searched(city, category):
                searched.append(category)

    return searched


def read_cached_venues(city: str | None = None) -> list[Venue]:
    """
    Read venues from Google Sheets cache.

    Args:
        city: If provided, only return venues from this city.
              If None, return all cached venues.

    Returns:
        List of Venue dicts
    """
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=_sheet_full_range(),  # All venue columns including event tracking fields
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:  # Only header or empty
            return []

        header = rows[0]
        venues = []

        for row in rows[1:]:
            # Pad row to match header length
            while len(row) < len(header):
                row.append("")

            venue = dict(zip(header, row))

            # Filter by city if specified
            if city:
                city_lower = city.lower().strip()
                venue_city = venue.get("city", "").lower().strip()
                if venue_city != city_lower:
                    continue

            # Parse website_attempts as int
            attempts_str = venue.get("website_attempts", "")
            try:
                website_attempts = int(attempts_str) if attempts_str else 0
            except ValueError:
                website_attempts = 0

            # Parse event_count as int
            event_count_str = venue.get("event_count", "")
            try:
                event_count = int(event_count_str) if event_count_str else 0
            except ValueError:
                event_count = 0

            venues.append(Venue(
                name=venue.get("name", ""),
                address=venue.get("address", ""),
                lat=_parse_coordinate(venue.get("lat", "")),
                lng=_parse_coordinate(venue.get("lng", "")),
                city=venue.get("city", ""),
                neighborhood=venue.get("neighborhood", ""),
                website=venue.get("website", ""),
                events_url=venue.get("events_url", ""),
                category=venue.get("category", ""),
                description=venue.get("description", ""),
                source=venue.get("source", ""),
                address_verified=venue.get("address_verified", ""),
                website_status=venue.get("website_status", ""),
                website_attempts=website_attempts,
                preferred_event_source=venue.get("preferred_event_source", ""),
                api_endpoint=venue.get("api_endpoint", ""),
                ticketmaster_venue_id=venue.get("ticketmaster_venue_id", ""),
                cloudflare_protected=venue.get("cloudflare_protected", ""),
                feed_url=venue.get("feed_url", ""),
                feed_type=venue.get("feed_type", ""),
                last_event_fetch=venue.get("last_event_fetch", ""),
                event_count=event_count,
                event_source=venue.get("event_source", ""),
            ))

        return venues

    except Exception as e:
        print(f"Error reading venues from sheet: {e}")
        return []


def append_venues_to_cache(venues: list[Venue], city: str, category: str):
    """
    Append venues to Google Sheets cache incrementally.

    Adds new venues for a category, deduplicating against existing venues.

    Args:
        venues: List of Venue dicts to add
        city: City these venues belong to
        category: Category that was searched
    """
    if not venues:
        mark_category_searched(city, category)
        return

    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    # Read existing venues
    existing = read_cached_venues()

    # Build set of existing venue names (normalized) for deduplication
    import re
    def normalize_name(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', name)
        name = re.sub(r'^the\s+', '', name)
        return name

    existing_names = {normalize_name(v.get("name", "")) for v in existing}

    # Filter to only new venues
    new_venues = []
    for venue in venues:
        norm_name = normalize_name(venue.get("name", ""))
        if norm_name and norm_name not in existing_names:
            existing_names.add(norm_name)  # Prevent duplicates within batch
            new_venues.append(venue)

    if not new_venues:
        print(f"    No new venues to add (all duplicates)")
        mark_category_searched(city, category)
        return

    # Combine and sort
    all_venues = existing + new_venues
    all_venues.sort(key=lambda x: (
        x.get("city", "").lower(),
        x.get("category", "").lower(),
        x.get("name", "").lower(),
    ))

    # Convert to rows
    rows = [VENUE_COLUMNS]  # Header
    for venue in all_venues:
        row = _venue_to_row(venue)
        rows.append(row)

    # Clear and write
    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=_sheet_full_range()
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        # Mark category as searched
        mark_category_searched(city, category)

        print(f"    Added {len(new_venues)} new venues (total: {len(all_venues)})")

    except Exception as e:
        print(f"Error appending venues to sheet: {e}")


def write_venues_to_cache(venues: list[Venue], city: str):
    """
    Write venues to Google Sheets cache (replaces all for city).

    Replaces all venues for the specified city while preserving
    venues from other cities.

    Args:
        venues: List of Venue dicts to cache
        city: City these venues belong to
    """
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    # Read existing venues from other cities
    all_venues = read_cached_venues()
    city_lower = city.lower().strip()

    # Remove old venues from this city
    other_city_venues = [
        v for v in all_venues
        if v.get("city", "").lower().strip() != city_lower
    ]

    # Add new venues for this city
    new_city_venues = [v for v in venues if v.get("city", "").lower().strip() == city_lower]
    all_venues = other_city_venues + new_city_venues

    # Sort by city then category then name
    all_venues.sort(key=lambda x: (
        x.get("city", "").lower(),
        x.get("category", "").lower(),
        x.get("name", "").lower(),
    ))

    # Convert to rows
    rows = [VENUE_COLUMNS]  # Header
    for venue in all_venues:
        rows.append(_venue_to_row(venue))

    # Clear and write
    try:
        # Clear existing data
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=_sheet_full_range()
        ).execute()

        # Write new data
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        # Mark city as updated
        mark_city_updated(city)

        print(f"Cached {len(new_city_venues)} venues for {city}")

    except Exception as e:
        print(f"Error writing venues to sheet: {e}")


def get_cache_summary() -> dict:
    """Get a summary of the venue cache status."""
    metadata = _load_metadata()
    venues = read_cached_venues()

    # Count venues by city
    by_city = {}
    for venue in venues:
        city = venue.get("city", "Unknown")
        by_city[city] = by_city.get(city, 0) + 1

    # Count venues by category
    by_category = {}
    for venue in venues:
        category = venue.get("category", "Unknown")
        by_category[category] = by_category.get(category, 0) + 1

    return {
        "total_venues": len(venues),
        "venues_by_city": by_city,
        "venues_by_category": by_category,
        "city_timestamps": metadata.get("cities", {}),
    }


def load_seed_venues(city: str) -> int:
    """
    Load seed venues from seed_venues.json for a city.

    Seed venues are known important venues that searches might miss.
    They are only added if not already in the cache.

    Returns count of venues added.
    """
    import re

    if not SEED_VENUES_FILE.exists():
        return 0

    with open(SEED_VENUES_FILE) as f:
        all_seeds = json.load(f)

    city_seeds = all_seeds.get(city, [])
    if not city_seeds:
        return 0

    # Get existing venues
    existing = read_cached_venues(city)

    def normalize_name(n: str) -> str:
        n = n.lower().strip()
        n = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', n)
        n = re.sub(r'^the\s+', '', n)
        return n

    existing_names = {normalize_name(v.get("name", "")) for v in existing}

    # Filter to only new seeds
    new_seeds = []
    for seed in city_seeds:
        norm_name = normalize_name(seed.get("name", ""))
        if norm_name and norm_name not in existing_names:
            new_seeds.append(Venue(
                name=seed["name"],
                address=seed.get("address", city),
                lat=None,
                lng=None,
                city=city,
                neighborhood=seed.get("neighborhood", ""),
                website=seed.get("website", ""),
                category=seed.get("category", "other"),
                description=seed.get("description", ""),
                source="seed",
                address_verified="yes",  # Seed venues have verified addresses
                website_status="",
                website_attempts=0,
                preferred_event_source="",
                api_endpoint="",
                ticketmaster_venue_id="",
                cloudflare_protected="",
                feed_url="",
                feed_type="",
                last_event_fetch="",
                event_count=0,
                event_source="",
            ))
            existing_names.add(norm_name)

    if not new_seeds:
        return 0

    # Append to cache
    append_venues_to_cache(new_seeds, city, "seed_venues")
    print(f"Added {len(new_seeds)} seed venues for {city}")
    return len(new_seeds)


def deduplicate_venues(city: str) -> int:
    """
    Remove duplicate venues from the cache.

    Keeps the first occurrence of each venue (by normalized name).
    Returns count of duplicates removed.
    """
    import re

    def normalize(name):
        name = name.lower().strip()
        name = re.sub(r'^the\s+', '', name)
        name = re.sub(r'^\(', '', name)  # handle (Le) Poisson Rouge
        name = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', name)
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name

    venues = read_cached_venues()
    city_lower = city.lower().strip()

    # Separate city venues from others
    city_venues = [v for v in venues if v.get("city", "").lower().strip() == city_lower]
    other_venues = [v for v in venues if v.get("city", "").lower().strip() != city_lower]

    # Deduplicate city venues
    seen = set()
    unique = []
    for v in city_venues:
        norm = normalize(v.get("name", ""))
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(v)

    removed = len(city_venues) - len(unique)

    if removed == 0:
        print(f"No duplicates found for {city}")
        return 0

    # Write back
    all_venues = other_venues + unique
    all_venues.sort(key=lambda x: (
        x.get("city", "").lower(),
        x.get("category", "").lower(),
        x.get("name", "").lower(),
    ))

    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return 0

    service = _get_sheets_service()
    if not service:
        return 0

    rows = [VENUE_COLUMNS]
    for venue in all_venues:
        rows.append(_venue_to_row(venue))

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=_sheet_full_range()
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        print(f"Removed {removed} duplicates for {city}. Total now: {len(unique)}")
        return removed

    except Exception as e:
        print(f"Error deduplicating: {e}")
        return 0


def update_venue_category(name: str, city: str, new_category: str) -> bool:
    """
    Update a venue's category.

    Returns True if updated, False if venue not found.
    """
    import re

    def normalize_name(n: str) -> str:
        n = n.lower().strip()
        n = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', n)
        n = re.sub(r'^the\s+', '', n)
        return n

    venues = read_cached_venues()
    target_norm = normalize_name(name)
    city_lower = city.lower().strip()

    updated = False
    for v in venues:
        if (normalize_name(v.get("name", "")) == target_norm and
            v.get("city", "").lower().strip() == city_lower):
            v["category"] = new_category
            updated = True
            break

    if not updated:
        print(f"Venue '{name}' not found in {city}")
        return False

    # Write back all venues
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return False

    service = _get_sheets_service()
    if not service:
        return False

    # Sort and write
    venues.sort(key=lambda x: (
        x.get("city", "").lower(),
        x.get("category", "").lower(),
        x.get("name", "").lower(),
    ))

    rows = [VENUE_COLUMNS]
    for venue in venues:
        rows.append(_venue_to_row(venue))

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=_sheet_full_range()
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        print(f"Updated '{name}' category to '{new_category}'")
        return True

    except Exception as e:
        print(f"Error updating venue: {e}")
        return False


def add_manual_venue(
    name: str,
    city: str,
    category: str,
    address: str = "",
    neighborhood: str = "",
    website: str = "",
    description: str = "",
) -> bool:
    """
    Manually add a venue that searches missed.

    Returns True if added, False if already exists.
    """
    import re

    # Check if venue already exists
    existing = read_cached_venues(city)

    def normalize_name(n: str) -> str:
        n = n.lower().strip()
        n = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', n)
        n = re.sub(r'^the\s+', '', n)
        return n

    norm_name = normalize_name(name)
    for v in existing:
        if normalize_name(v.get("name", "")) == norm_name:
            print(f"Venue '{name}' already exists")
            return False

    # Create venue
    venue = Venue(
        name=name,
        address=address or city,
        lat=None,
        lng=None,
        city=city,
        neighborhood=neighborhood,
        website=website,
        category=category,
        description=description,
        source="manual",
        address_verified="",
        website_status="",
        website_attempts=0,
        preferred_event_source="",
        api_endpoint="",
        ticketmaster_venue_id="",
        cloudflare_protected="",
        feed_url="",
        feed_type="",
        last_event_fetch="",
        event_count=0,
        event_source="",
    )

    # Append to cache
    append_venues_to_cache([venue], city, f"manual:{category}")
    print(f"Added venue: {name}")
    return True


def add_manual_venues_batch(venues_data: list[dict], city: str) -> int:
    """
    Add multiple manual venues at once.

    Each dict should have: name, category, and optionally:
    address, neighborhood, website, description

    Returns count of venues added.
    """
    added = 0
    for v in venues_data:
        if add_manual_venue(
            name=v["name"],
            city=city,
            category=v["category"],
            address=v.get("address", ""),
            neighborhood=v.get("neighborhood", ""),
            website=v.get("website", ""),
            description=v.get("description", ""),
        ):
            added += 1
    return added


def update_venues_batch(updated_venues: list[dict], city: str) -> int:
    """
    Update multiple venues in the cache with new data.

    Matches venues by name and city, updates all fields.

    Args:
        updated_venues: List of venue dicts with updated fields
        city: City these venues belong to

    Returns:
        Number of venues updated
    """
    import re

    def normalize_name(n: str) -> str:
        n = n.lower().strip()
        n = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', n)
        n = re.sub(r'^the\s+', '', n)
        return n

    # Read all venues
    all_venues = read_cached_venues()

    # Build lookup for updates
    updates = {}
    for v in updated_venues:
        key = (normalize_name(v.get("name", "")), v.get("city", city).lower())
        updates[key] = v

    # Apply updates
    updated_count = 0
    for i, venue in enumerate(all_venues):
        key = (normalize_name(venue.get("name", "")), venue.get("city", "").lower())
        if key in updates:
            # Merge updates into venue
            update = updates[key]
            for field in ["address", "lat", "lng", "neighborhood", "address_verified",
                          "website", "events_url", "website_status", "website_attempts",
                          "preferred_event_source", "api_endpoint", "ticketmaster_venue_id",
                          "cloudflare_protected",
                          "feed_url", "feed_type",
                          "last_event_fetch", "event_count", "event_source"]:
                if field in update:
                    all_venues[i][field] = update[field]
            updated_count += 1

    if updated_count == 0:
        return 0

    # Write back
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return 0

    service = _get_sheets_service()
    if not service:
        return 0

    # Sort and write
    all_venues.sort(key=lambda x: (
        x.get("city", "").lower(),
        x.get("category", "").lower(),
        x.get("name", "").lower(),
    ))

    rows = [VENUE_COLUMNS]
    for venue in all_venues:
        rows.append(_venue_to_row(venue))

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=_sheet_full_range()
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        return updated_count

    except Exception as e:
        print(f"Error updating venues: {e}")
        return 0


def update_venue_event_tracking(
    venue_name: str,
    city: str,
    event_count: int,
    event_source: str,
) -> bool:
    """
    Update a venue's event tracking fields after a fetch.

    This replaces the JSON-based venue_events_metadata.json.

    Args:
        venue_name: Name of the venue
        city: City the venue is in
        event_count: Number of events found
        event_source: Source used ("ticketmaster", "api", "scrape", etc.)

    Returns:
        True if updated, False if venue not found
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    venues = read_cached_venues()
    city_lower = city.lower().strip()
    venue_lower = venue_name.lower().strip()

    updated = False
    for v in venues:
        if (v.get("name", "").lower().strip() == venue_lower and
            v.get("city", "").lower().strip() == city_lower):
            v["last_event_fetch"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
            v["event_count"] = event_count
            v["event_source"] = event_source
            updated = True
            break

    if not updated:
        return False

    # Write back all venues
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return False

    service = _get_sheets_service()
    if not service:
        return False

    rows = [VENUE_COLUMNS]
    for venue in venues:
        rows.append(_venue_to_row(venue))

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=_sheet_full_range()
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        return True

    except Exception as e:
        print(f"Error updating venue event tracking: {e}")
        return False


def is_venue_events_fresh(
    venue_name: str,
    city: str,
    threshold_days: int | None = None,
) -> bool:
    """
    Check if a venue's event data is still fresh.

    This replaces the JSON-based freshness check in event_cache.py.

    Args:
        venue_name: Name of the venue
        city: City the venue is in
        threshold_days: Days before data is stale (default from settings)

    Returns:
        True if events are fresh, False if stale or not fetched
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    if threshold_days is None:
        threshold_days = getattr(_settings, "VENUE_EVENT_CACHE_DAYS", 7)

    venues = read_cached_venues(city)
    venue_lower = venue_name.lower().strip()

    for v in venues:
        if v.get("name", "").lower().strip() == venue_lower:
            last_fetch = v.get("last_event_fetch", "")
            if not last_fetch:
                return False

            try:
                last_fetched = datetime.fromisoformat(last_fetch)
                now = datetime.now(ZoneInfo("America/New_York"))
                if last_fetched.tzinfo is None:
                    last_fetched = last_fetched.replace(tzinfo=ZoneInfo("America/New_York"))

                age = now - last_fetched
                return age < timedelta(days=threshold_days)
            except ValueError:
                return False

    return False


def get_stale_venues(venues: list[dict], city: str) -> list[dict]:
    """
    Filter to only venues that need event fetching.

    Uses the last_event_fetch field directly from the venue dict,
    avoiding extra sheet reads. This is critical for batch operations.

    Args:
        venues: List of Venue dicts (must include last_event_fetch field)
        city: City to check

    Returns:
        List of venues that are stale or never fetched
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    threshold_days = getattr(_settings, "VENUE_EVENT_CACHE_DAYS", 7)
    now = datetime.now(ZoneInfo("America/New_York"))
    stale = []

    for venue in venues:
        last_fetch = venue.get("last_event_fetch", "")

        # No fetch recorded = stale
        if not last_fetch:
            stale.append(venue)
            continue

        # Check age of last fetch
        try:
            last_fetched = datetime.fromisoformat(last_fetch)
            if last_fetched.tzinfo is None:
                last_fetched = last_fetched.replace(tzinfo=ZoneInfo("America/New_York"))
            age = now - last_fetched
            if age >= timedelta(days=threshold_days):
                stale.append(venue)
        except ValueError:
            # Invalid date = stale
            stale.append(venue)

    return stale


def test_cache():
    """Test the venue cache integration."""
    print("Testing Venue Cache integration...")

    if not is_authenticated():
        print("Not authenticated. Run: python -m utils.google_auth")
        return

    # Get/create venues sheet
    sheet_id = get_or_create_venues_sheet()
    print(f"Venues sheet: {sheet_id}")

    # Read current data
    venues = read_cached_venues()
    print(f"\nCurrent data: {len(venues)} venues")

    # Print summary
    summary = get_cache_summary()
    print(f"\nBy city: {summary['venues_by_city']}")
    print(f"By category: {summary['venues_by_category']}")


if __name__ == "__main__":
    test_cache()
