"""Venue caching system with Firestore storage and freshness tracking.

Storage backend: Cloud Firestore, collection ``venues``.
Freshness metadata (city/category timestamps) continues to be stored in a
local JSON file so it survives container restarts without a Firestore round-trip.
"""

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Local metadata paths (freshness tracking stays on disk)
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_CACHE_DIR = DATA_DIR
_METADATA_FILE = VENUE_CACHE_METADATA_FILE

# Firestore collection
_VENUES_COLLECTION = "venues"
_FIRESTORE_BATCH_SIZE = 500

# Column list kept for schema documentation and backward compatibility
VENUE_COLUMNS = [
    "name", "address", "lat", "lng", "city", "neighborhood",
    "website", "events_url", "category", "description", "source", "address_verified",
    "website_status", "website_attempts",
    "preferred_event_source", "api_endpoint", "ticketmaster_venue_id",
    "cloudflare_protected",
    "feed_url", "feed_type",
    "last_event_fetch", "event_count", "event_source",
]


# ─────────────────────────── Firestore helpers ───────────────────────────────

def _get_db():
    """Return the Firestore client."""
    from venue_scout.firestore_client import get_db
    return get_db()


def _venue_doc_id(name: str) -> str:
    """Compute a stable Firestore document ID from a venue name.

    Uses the same normalisation as ``_normalize_venue_name`` so that
    lookups by name always resolve to the same doc.
    """
    text = str(name or "").lower().strip()
    text = re.sub(r'^the\s+', '', text)
    text = re.sub(r'^[(]', '', text)
    text = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', '_', text).strip('_')
    doc_id = text[:1490]
    return doc_id or hashlib.md5(str(name).encode()).hexdigest()


def _serialize_venue_for_db(venue: dict) -> dict:
    """Convert a venue dict to Firestore-compatible native types."""
    row = dict(venue)

    for field in ("lat", "lng"):
        val = row.get(field)
        if val is None or val == "":
            row[field] = None
        elif isinstance(val, str):
            try:
                row[field] = float(val.strip()) if val.strip() else None
            except ValueError:
                row[field] = None

    for field in ("website_attempts", "event_count"):
        val = row.get(field)
        if val is None or val == "":
            row[field] = 0
        elif isinstance(val, str):
            try:
                row[field] = int(val.strip()) if val.strip() else 0
            except ValueError:
                row[field] = 0

    return row


def _deserialize_venue_from_db(data: dict) -> Venue:
    """Convert a Firestore document dict to a Venue TypedDict."""
    lat_val = data.get("lat")
    lng_val = data.get("lng")

    attempts_raw = data.get("website_attempts", 0)
    try:
        website_attempts = int(attempts_raw) if attempts_raw is not None else 0
    except (ValueError, TypeError):
        website_attempts = 0

    count_raw = data.get("event_count", 0)
    try:
        event_count = int(count_raw) if count_raw is not None else 0
    except (ValueError, TypeError):
        event_count = 0

    return Venue(
        name=data.get("name", ""),
        address=data.get("address", ""),
        lat=float(lat_val) if lat_val is not None and lat_val != "" else None,
        lng=float(lng_val) if lng_val is not None and lng_val != "" else None,
        city=data.get("city", ""),
        neighborhood=data.get("neighborhood", ""),
        website=data.get("website", ""),
        events_url=data.get("events_url", ""),
        category=data.get("category", ""),
        description=data.get("description", ""),
        source=data.get("source", ""),
        address_verified=data.get("address_verified", ""),
        website_status=data.get("website_status", ""),
        website_attempts=website_attempts,
        preferred_event_source=data.get("preferred_event_source", ""),
        api_endpoint=data.get("api_endpoint", ""),
        ticketmaster_venue_id=data.get("ticketmaster_venue_id", ""),
        cloudflare_protected=data.get("cloudflare_protected", ""),
        feed_url=data.get("feed_url", ""),
        feed_type=data.get("feed_type", ""),
        last_event_fetch=data.get("last_event_fetch", ""),
        event_count=event_count,
        event_source=data.get("event_source", ""),
    )


def _batch_set_venues(db, col, venues: list[Venue]):
    """Batch-upsert venues into Firestore."""
    for start in range(0, len(venues), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        for venue in venues[start:start + _FIRESTORE_BATCH_SIZE]:
            doc_id = _venue_doc_id(venue.get("name", ""))
            batch.set(col.document(doc_id), _serialize_venue_for_db(venue))
        batch.commit()


def _batch_delete_refs(db, refs: list):
    """Batch-delete Firestore document references."""
    for start in range(0, len(refs), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        for ref in refs[start:start + _FIRESTORE_BATCH_SIZE]:
            batch.delete(ref)
        batch.commit()


def _normalize_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', n)
    n = re.sub(r'^the\s+', '', n)
    return n


# ─────────────────────────── Freshness metadata (local JSON) ─────────────────

def _load_metadata() -> dict:
    if not _METADATA_FILE.exists():
        return {"cities": {}, "categories": {}}
    try:
        with open(_METADATA_FILE, "r") as f:
            data = json.load(f)
            if "categories" not in data:
                data["categories"] = {}
            return data
    except (json.JSONDecodeError, IOError):
        return {"cities": {}, "categories": {}}


def _save_metadata(metadata: dict):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def get_city_last_updated(city: str) -> datetime | None:
    metadata = _load_metadata()
    timestamp = metadata.get("cities", {}).get(city.lower().strip())
    if timestamp:
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    return None


def is_city_fresh(city: str, threshold_days: int | None = None) -> bool:
    if threshold_days is None:
        threshold_days = _settings.VENUE_CACHE_THRESHOLD_DAYS

    last_updated = get_city_last_updated(city)
    if last_updated is None:
        return False

    now = datetime.now(ZoneInfo("America/New_York"))
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=ZoneInfo("America/New_York"))

    return (now - last_updated) < timedelta(days=threshold_days)


def mark_city_updated(city: str):
    metadata = _load_metadata()
    if "cities" not in metadata:
        metadata["cities"] = {}
    metadata["cities"][city.lower().strip()] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    _save_metadata(metadata)


def _category_key(city: str, category: str) -> str:
    return f"{city.lower().strip()}:{category.lower().strip()}"


def is_category_searched(city: str, category: str, threshold_days: int | None = None) -> bool:
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
        return (now - last_searched) < timedelta(days=threshold_days)
    except ValueError:
        return False


def mark_category_searched(city: str, category: str):
    metadata = _load_metadata()
    if "categories" not in metadata:
        metadata["categories"] = {}
    metadata["categories"][_category_key(city, category)] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    _save_metadata(metadata)


def get_searched_categories(city: str) -> list[str]:
    metadata = _load_metadata()
    city_prefix = f"{city.lower().strip()}:"
    return [
        key[len(city_prefix):]
        for key in metadata.get("categories", {}).keys()
        if key.startswith(city_prefix) and is_category_searched(city, key[len(city_prefix):])
    ]


# ─────────────────────────── Public venue storage API ────────────────────────

def read_cached_venues(city: str | None = None) -> list[Venue]:
    """Read venues from Firestore, optionally filtered by city."""
    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)

        # Use Firestore-level filter when city is provided; Python-side filter
        # as a safety net in case stored city casing differs.
        query = col.where("city", "==", city) if city else col
        city_lower = city.lower().strip() if city else None

        venues = []
        for doc in query.stream():
            data = doc.to_dict()
            if not data:
                continue
            if city_lower and data.get("city", "").lower().strip() != city_lower:
                continue
            venues.append(_deserialize_venue_from_db(data))

        return venues

    except Exception as e:
        print(f"Error reading venues from Firestore: {e}")
        return []


def append_venues_to_cache(venues: list[Venue], city: str, category: str):
    """Append new venues for a city/category, skipping duplicates."""
    if not venues:
        mark_category_searched(city, category)
        return

    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)

        # Fetch just the name field for existing city venues (fast)
        existing_names = set()
        for doc in col.where("city", "==", city).select(["name"]).stream():
            data = doc.to_dict() or {}
            norm = _normalize_name(data.get("name", ""))
            if norm:
                existing_names.add(norm)

        new_venues = []
        for venue in venues:
            norm = _normalize_name(venue.get("name", ""))
            if norm and norm not in existing_names:
                existing_names.add(norm)
                new_venues.append(venue)

        if not new_venues:
            print("    No new venues to add (all duplicates)")
            mark_category_searched(city, category)
            return

        _batch_set_venues(db, col, new_venues)
        mark_category_searched(city, category)
        print(f"    Added {len(new_venues)} new venues")

    except Exception as e:
        print(f"Error appending venues to Firestore: {e}")


def write_venues_to_cache(venues: list[Venue], city: str):
    """Replace all venues for a city (full city-scoped overwrite)."""
    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)
        city_lower = city.lower().strip()

        # Delete all existing docs for this city
        existing_refs = [
            doc.reference
            for doc in col.where("city", "==", city).select([]).stream()
        ]
        _batch_delete_refs(db, existing_refs)

        # Write only the city's venues from the provided list
        city_venues = [v for v in venues if v.get("city", "").lower().strip() == city_lower]
        _batch_set_venues(db, col, city_venues)

        mark_city_updated(city)
        print(f"Cached {len(city_venues)} venues for {city}")

    except Exception as e:
        print(f"Error writing venues to Firestore: {e}")


def get_cache_summary() -> dict:
    """Return a summary of the venue cache."""
    metadata = _load_metadata()
    venues = read_cached_venues()

    by_city: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for venue in venues:
        by_city[venue.get("city", "Unknown")] = by_city.get(venue.get("city", "Unknown"), 0) + 1
        by_category[venue.get("category", "Unknown")] = by_category.get(venue.get("category", "Unknown"), 0) + 1

    return {
        "total_venues": len(venues),
        "venues_by_city": by_city,
        "venues_by_category": by_category,
        "city_timestamps": metadata.get("cities", {}),
    }


def load_seed_venues(city: str) -> int:
    """Load seed venues for a city if not already in the cache."""
    if not SEED_VENUES_FILE.exists():
        return 0

    with open(SEED_VENUES_FILE) as f:
        all_seeds = json.load(f)

    city_seeds = all_seeds.get(city, [])
    if not city_seeds:
        return 0

    existing = read_cached_venues(city)
    existing_names = {_normalize_name(v.get("name", "")) for v in existing}

    new_seeds = []
    for seed in city_seeds:
        norm_name = _normalize_name(seed.get("name", ""))
        if norm_name and norm_name not in existing_names:
            new_seeds.append(Venue(
                name=seed["name"],
                address=seed.get("address", city),
                lat=None,
                lng=None,
                city=city,
                neighborhood=seed.get("neighborhood", ""),
                website=seed.get("website", ""),
                events_url=seed.get("events_url", ""),
                category=seed.get("category", "other"),
                description=seed.get("description", ""),
                source="seed",
                address_verified="yes",
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

    append_venues_to_cache(new_seeds, city, "seed_venues")
    print(f"Added {len(new_seeds)} seed venues for {city}")
    return len(new_seeds)


def deduplicate_venues(city: str) -> int:
    """Remove duplicate venues (by normalised name) from the cache for a city."""
    venues = read_cached_venues(city)

    seen: set[str] = set()
    unique: list[Venue] = []
    dupes: list[Venue] = []

    for v in venues:
        norm = _normalize_name(v.get("name", ""))
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(v)
        else:
            dupes.append(v)

    removed = len(dupes)
    if removed == 0:
        print(f"No duplicates found for {city}")
        return 0

    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)

        # Delete duplicate docs by their doc IDs
        dupe_ids = {_venue_doc_id(v.get("name", "")) for v in dupes}
        dupe_refs = [col.document(did) for did in dupe_ids]
        _batch_delete_refs(db, dupe_refs)

        print(f"Removed {removed} duplicates for {city}. Total now: {len(unique)}")
        return removed

    except Exception as e:
        print(f"Error deduplicating venues in Firestore: {e}")
        return 0


def update_venue_category(name: str, city: str, new_category: str) -> bool:
    """Update a venue's category."""
    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)

        doc_id = _venue_doc_id(name)
        doc_ref = col.document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            doc_ref.update({"category": new_category})
            print(f"Updated '{name}' category to '{new_category}'")
            return True

        # Fallback: city-scoped name-insensitive search
        city_lower = city.lower().strip()
        target_norm = _normalize_name(name)
        for doc in col.where("city", "==", city).stream():
            data = doc.to_dict() or {}
            if _normalize_name(data.get("name", "")) == target_norm:
                doc.reference.update({"category": new_category})
                print(f"Updated '{name}' category to '{new_category}'")
                return True

        print(f"Venue '{name}' not found in {city}")
        return False

    except Exception as e:
        print(f"Error updating venue category in Firestore: {e}")
        return False


def add_manual_venue(
    name: str,
    city: str,
    category: str,
    address: str = "",
    neighborhood: str = "",
    website: str = "",
    description: str = "",
    events_url: str = "",
) -> bool:
    """Add a venue manually. Returns True if added, False if already exists."""
    existing = read_cached_venues(city)
    norm_name = _normalize_name(name)
    for v in existing:
        if _normalize_name(v.get("name", "")) == norm_name:
            print(f"Venue '{name}' already exists")
            return False

    venue = Venue(
        name=name,
        address=address or city,
        lat=None,
        lng=None,
        city=city,
        neighborhood=neighborhood,
        website=website,
        events_url=events_url,
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

    append_venues_to_cache([venue], city, f"manual:{category}")
    print(f"Added venue: {name}")
    return True


def add_manual_venues_batch(venues_data: list[dict], city: str) -> int:
    """Add multiple manual venues. Returns count added."""
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
            events_url=v.get("events_url", ""),
        ):
            added += 1
    return added


def update_venues_batch(updated_venues: list[dict], city: str) -> int:
    """Update specific fields on multiple venues (matched by name)."""
    if not updated_venues:
        return 0

    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)

        # Build a normalised-name → update-payload map
        updates: dict[str, dict] = {}
        for v in updated_venues:
            norm = _normalize_name(v.get("name", ""))
            if norm:
                updates[norm] = v

        # Also build a doc-ID → update-payload map for O(1) direct lookups
        updates_by_doc_id: dict[str, dict] = {
            _venue_doc_id(v.get("name", "")): v
            for v in updated_venues
        }

        # Try direct doc-ID lookups first (fast path)
        updated_count = 0
        remaining_norms: set[str] = set(updates.keys())

        for doc_id, payload in updates_by_doc_id.items():
            doc_ref = col.document(doc_id)
            doc_snap = doc_ref.get()
            if doc_snap.exists:
                update_fields = {
                    k: payload[k] for k in [
                        "address", "lat", "lng", "neighborhood", "address_verified",
                        "website", "events_url", "website_status", "website_attempts",
                        "preferred_event_source", "api_endpoint", "ticketmaster_venue_id",
                        "cloudflare_protected", "feed_url", "feed_type",
                        "last_event_fetch", "event_count", "event_source",
                    ] if k in payload
                }
                if update_fields:
                    doc_ref.update(update_fields)
                    updated_count += 1
                    norm = _normalize_name(payload.get("name", ""))
                    remaining_norms.discard(norm)

        # Fallback: name-insensitive scan for anything not found by doc ID
        if remaining_norms:
            city_lower = city.lower().strip()
            for doc in col.where("city", "==", city).stream():
                data = doc.to_dict() or {}
                norm = _normalize_name(data.get("name", ""))
                if norm in remaining_norms:
                    payload = updates[norm]
                    update_fields = {
                        k: payload[k] for k in [
                            "address", "lat", "lng", "neighborhood", "address_verified",
                            "website", "events_url", "website_status", "website_attempts",
                            "preferred_event_source", "api_endpoint", "ticketmaster_venue_id",
                            "cloudflare_protected", "feed_url", "feed_type",
                            "last_event_fetch", "event_count", "event_source",
                        ] if k in payload
                    }
                    if update_fields:
                        doc.reference.update(update_fields)
                        updated_count += 1
                    remaining_norms.discard(norm)
                    if not remaining_norms:
                        break

        return updated_count

    except Exception as e:
        print(f"Error updating venues in Firestore: {e}")
        return 0


def update_venue_event_tracking(
    venue_name: str,
    city: str,
    event_count: int,
    event_source: str,
) -> bool:
    """Update a venue's event tracking fields (last_event_fetch, event_count, event_source)."""
    try:
        db = _get_db()
        col = db.collection(_VENUES_COLLECTION)

        now_iso = datetime.now(ZoneInfo("America/New_York")).isoformat()
        update_payload = {
            "last_event_fetch": now_iso,
            "event_count": event_count,
            "event_source": event_source,
        }

        # Fast path: direct doc-ID lookup
        doc_id = _venue_doc_id(venue_name)
        doc_ref = col.document(doc_id)
        doc_snap = doc_ref.get()
        if doc_snap.exists:
            doc_ref.update(update_payload)
            return True

        # Fallback: name-insensitive scan over city venues
        venue_lower = venue_name.lower().strip()
        for doc in col.where("city", "==", city).select(["name"]).stream():
            data = doc.to_dict() or {}
            if data.get("name", "").lower().strip() == venue_lower:
                doc.reference.update(update_payload)
                return True

        return False

    except Exception as e:
        print(f"Error updating venue event tracking in Firestore: {e}")
        return False


def is_venue_events_fresh(
    venue_name: str,
    city: str,
    threshold_days: int | None = None,
) -> bool:
    """Check if a venue's event data is still fresh."""
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
                return (now - last_fetched) < timedelta(days=threshold_days)
            except ValueError:
                return False

    return False


def get_stale_venues(venues: list[dict], city: str) -> list[dict]:
    """Filter to venues that need event fetching (stale or never fetched)."""
    threshold_days = getattr(_settings, "VENUE_EVENT_CACHE_DAYS", 7)
    now = datetime.now(ZoneInfo("America/New_York"))
    stale = []

    for venue in venues:
        last_fetch = venue.get("last_event_fetch", "")
        if not last_fetch:
            stale.append(venue)
            continue
        try:
            last_fetched = datetime.fromisoformat(last_fetch)
            if last_fetched.tzinfo is None:
                last_fetched = last_fetched.replace(tzinfo=ZoneInfo("America/New_York"))
            if (now - last_fetched) >= timedelta(days=threshold_days):
                stale.append(venue)
        except ValueError:
            stale.append(venue)

    return stale


# ─────────────────────────── Backward-compat stubs ───────────────────────────
# These were private Sheets helpers imported by enrich_addresses.py,
# venue_cleaner.py, and event_fetcher.py.  Those files have been updated to
# use the public API, but stubs are kept here to avoid import errors during
# any in-flight deployments.

def _venue_to_row(venue: dict) -> list:
    """DEPRECATED: Returns a row list from a venue dict (no-op stub)."""
    return [venue.get(col, "") for col in VENUE_COLUMNS]


def _get_sheets_service():
    """DEPRECATED: Sheets service stub — returns None (Firestore is used instead)."""
    return None


def get_or_create_venues_sheet() -> str | None:
    """DEPRECATED: Sheets stub — returns None (Firestore is used instead)."""
    return None


def _sheet_col_label(col_index: int) -> str:
    """DEPRECATED: A1 column label stub (kept for import compatibility)."""
    if col_index < 1:
        return "A"
    out = ""
    value = col_index
    while value:
        value, rem = divmod(value - 1, 26)
        out = chr(65 + rem) + out
    return out


def _sheet_full_range() -> str:
    """DEPRECATED: Sheets range stub."""
    return f"A:{_sheet_col_label(len(VENUE_COLUMNS))}"


# ─────────────────────────── Dev / test helpers ──────────────────────────────

def test_cache():
    """Smoke-test the Firestore venue cache."""
    print("Testing Venue Cache Firestore integration...")
    venues = read_cached_venues()
    print(f"\nCurrent data: {len(venues)} venues")
    summary = get_cache_summary()
    print(f"\nBy city: {summary['venues_by_city']}")
    print(f"By category: {summary['venues_by_category']}")


if __name__ == "__main__":
    test_cache()
