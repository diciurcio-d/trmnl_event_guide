#!/usr/bin/env python3
"""Enrich venue addresses using Google Places API."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.cache import (
    VENUE_COLUMNS,
    _get_sheets_service,
    _venue_to_row,
    get_or_create_venues_sheet,
    read_cached_venues,
)
from venue_scout.paths import PLACES_CACHE_FILE, ensure_data_dir


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


def _load_config() -> dict:
    """Load config from file."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _load_places_cache() -> dict:
    if PLACES_CACHE_FILE.exists():
        try:
            with open(PLACES_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_places_cache(cache: dict):
    ensure_data_dir()
    with open(PLACES_CACHE_FILE, "w") as f:
        json.dump(cache, f)


_places_cache = _load_places_cache()


BAD_ADDRESSES = {'nyc', 'new york', 'brooklyn', 'manhattan', 'queens', 'bronx', 'staten island', ''}


def is_bad_address(address: str) -> bool:
    """Check if an address is too vague to be useful."""
    return address.lower().replace(',', '').strip() in BAD_ADDRESSES


def _sheet_col_label(col_index: int) -> str:
    """Convert 1-based column index into A1 column label."""
    if col_index < 1:
        return "A"
    out = ""
    value = col_index
    while value:
        value, rem = divmod(value - 1, 26)
        out = chr(65 + rem) + out
    return out


def _sheet_full_range() -> str:
    """Return full A1 range covering all venue columns."""
    return f"A:{_sheet_col_label(len(VENUE_COLUMNS))}"


def _parse_coordinate(value) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _has_coordinates(venue: dict) -> bool:
    """Return True when venue has both latitude and longitude."""
    return _parse_coordinate(venue.get("lat")) is not None and _parse_coordinate(venue.get("lng")) is not None


def lookup_place(venue_name: str, city: str = "NYC", address_hint: str = "") -> dict | None:
    """
    Look up a venue using Google Places API (New).

    Returns dict with:
        - formatted_address: Full address
        - neighborhood: Extracted neighborhood/area
        - place_id: Google Place ID
        - lat: Latitude
        - lng: Longitude

    Or None if not found.
    """
    # Check cache first
    cache_key = f"{venue_name}|{city}|{address_hint.strip().lower()}"
    if cache_key in _places_cache:
        cached = _places_cache[cache_key]
        # Old cache entries may not include coordinates; refresh those.
        if cached is None:
            return None
        if cached.get("lat") is not None and cached.get("lng") is not None:
            return cached

    config = _load_config()
    api_key = config.get("google_cloud", {}).get("api_key")

    if not api_key:
        print("No Google Cloud API key configured")
        return None

    # Use new Places API (Text Search)
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.id,places.location",
    }
    query = f"{venue_name}, {city}"
    if address_hint.strip():
        query = f"{venue_name}, {address_hint}, {city}"
    body = {
        "textQuery": query,
        "maxResultCount": 1,
    }

    try:
        timeout = int(_settings.VENUE_ENRICH_PLACES_TIMEOUT_SEC)
        response = requests.post(url, headers=headers, json=body, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if data.get("places"):
            place = data["places"][0]
            formatted_address = place.get("formattedAddress", "")

            result = {
                "formatted_address": formatted_address,
                "name": place.get("displayName", {}).get("text", ""),
                "place_id": place.get("id", ""),
                "lat": place.get("location", {}).get("latitude"),
                "lng": place.get("location", {}).get("longitude"),
            }

            # Extract neighborhood from address (usually second component)
            addr_parts = formatted_address.split(",")
            if len(addr_parts) >= 2:
                result["neighborhood"] = addr_parts[1].strip()
            else:
                result["neighborhood"] = ""

            # Cache result
            _places_cache[cache_key] = result
            _save_places_cache(_places_cache)

            return result
        else:
            # Cache negative result to avoid repeated lookups
            _places_cache[cache_key] = None
            _save_places_cache(_places_cache)
            return None

    except requests.RequestException as e:
        print(f"Error looking up {venue_name}: {e}")
        return None


def enrich_venues_in_cache(city: str = "NYC") -> int:
    """
    Automatically enrich venues with bad addresses.
    Called after discovery to fill in missing addresses.
    Skips venues that have already been verified.

    Returns count of venues enriched.
    """
    venues = read_cached_venues(city)

    # Only enrich venues with bad addresses that haven't been verified yet
    to_enrich = [
        v for v in venues
        if is_bad_address(v.get("address", "")) and not v.get("address_verified")
    ]

    if not to_enrich:
        return 0

    print(f"  Enriching {len(to_enrich)} venues with missing addresses...")

    updated = 0
    for venue in to_enrich:
        name = venue["name"]

        # Rate limit
        if f"{name}|{city}" not in _places_cache:
            time.sleep(float(getattr(_settings, "VENUE_ENRICH_PLACES_DELAY_SEC", 0.1)))

        result = lookup_place(name, city, venue.get("address", ""))

        if result and result.get("formatted_address"):
            venue["address"] = result["formatted_address"]
            if result.get("neighborhood") and not venue.get("neighborhood"):
                venue["neighborhood"] = result["neighborhood"]
            venue["lat"] = result.get("lat")
            venue["lng"] = result.get("lng")
            venue["address_verified"] = "yes"
            updated += 1
        else:
            # Mark as verified even if not found, so we don't retry
            venue["address_verified"] = "not_found"

    if to_enrich:
        _save_venues_to_sheet(venues)
        print(f"  Enriched {updated} venue addresses ({len(to_enrich) - updated} not found)")

    return updated


def enrich_venues(city: str = "NYC", dry_run: bool = True, limit: int = None):
    """
    Enrich venues with bad addresses using Google Places API.

    Args:
        city: City to filter venues
        dry_run: If True, just print what would be updated
        limit: Max number of venues to process (for testing)
    """
    venues = read_cached_venues(city)

    # Find venues with bad addresses
    to_enrich = []
    for v in venues:
        if is_bad_address(v.get("address", "")):
            to_enrich.append(v)

    print(f"Found {len(to_enrich)} venues with bad addresses")

    if limit:
        to_enrich = to_enrich[:limit]
        print(f"Processing first {limit} venues")

    # Check how many are already cached
    cached = sum(1 for v in to_enrich if f"{v['name']}|{city}" in _places_cache)
    print(f"Already cached: {cached}")
    print(f"New API calls needed: {len(to_enrich) - cached}")
    print(f"Estimated cost: ${(len(to_enrich) - cached) * 0.017:.2f}")  # $17 per 1000 requests
    print()

    if dry_run:
        print("DRY RUN - not updating venues. Run with --apply to update.\n")

    updated = 0
    not_found = 0

    for i, venue in enumerate(to_enrich):
        name = venue["name"]

        # Rate limit (Places API allows 100 QPS but let's be gentle)
        if f"{name}|{city}" not in _places_cache:
            time.sleep(float(getattr(_settings, "VENUE_ENRICH_PLACES_DELAY_SEC", 0.1)))

        result = lookup_place(name, city, venue.get("address", ""))

        if result and result.get("formatted_address"):
            new_addr = result["formatted_address"]
            new_neighborhood = result.get("neighborhood", "")

            print(f"[{i+1}/{len(to_enrich)}] {name}")
            print(f"  Old: {venue['address']}")
            print(f"  New: {new_addr}")

            if not dry_run:
                venue["address"] = new_addr
                if new_neighborhood and not venue.get("neighborhood"):
                    venue["neighborhood"] = new_neighborhood
                venue["lat"] = result.get("lat")
                venue["lng"] = result.get("lng")

            updated += 1
        else:
            print(f"[{i+1}/{len(to_enrich)}] {name} - NOT FOUND")
            not_found += 1

    print(f"\nUpdated: {updated}, Not found: {not_found}")

    if not dry_run and updated > 0:
        # Write back to sheet
        print("\nSaving to Google Sheets...")
        _save_venues_to_sheet(venues)
        print("Done!")

    return updated, not_found


def _save_venues_to_sheet(venues: list):
    """Save all venues to Google Sheets."""
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    venues.sort(key=lambda x: (
        x.get("city", "").lower(),
        x.get("category", "").lower(),
        x.get("name", "").lower(),
    ))

    rows = [VENUE_COLUMNS]
    for venue in venues:
        rows.append(_venue_to_row(venue))

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


def backfill_venue_coordinates(
    city: str = "NYC",
    dry_run: bool = True,
    limit: int | None = None,
    include_bad_addresses: bool = False,
    require_events_url: bool = False,
) -> dict:
    """
    Backfill lat/lng for venues missing coordinates.

    Args:
        city: City to process
        dry_run: If True, do not write sheet updates
        limit: Max venues to process
        include_bad_addresses: If True, also attempt vague addresses
        require_events_url: If True, only process venues that have events_url
    """
    venues = read_cached_venues(city)
    to_backfill = []
    for venue in venues:
        if require_events_url and not str(venue.get("events_url", "") or "").strip():
            continue
        if _has_coordinates(venue):
            continue
        address = str(venue.get("address", "") or "").strip()
        if not address:
            continue
        if not include_bad_addresses and is_bad_address(address):
            continue
        to_backfill.append(venue)

    if limit:
        to_backfill = to_backfill[:limit]

    print(f"Venues missing coordinates: {len(to_backfill)}")

    updated = 0
    not_found = 0
    for i, venue in enumerate(to_backfill, 1):
        name = venue.get("name", "")
        address = venue.get("address", "")
        print(f"[{i}/{len(to_backfill)}] {name[:50]}...", end=" ", flush=True)

        if f"{name}|{city}|{str(address).strip().lower()}" not in _places_cache:
            time.sleep(float(getattr(_settings, "VENUE_ENRICH_PLACES_DELAY_SEC", 0.1)))

        result = lookup_place(name, city, str(address or ""))
        if result and result.get("lat") is not None and result.get("lng") is not None:
            print("ok")
            if not dry_run:
                venue["lat"] = result.get("lat")
                venue["lng"] = result.get("lng")
                if result.get("formatted_address") and is_bad_address(str(address)):
                    venue["address"] = result.get("formatted_address")
                    venue["address_verified"] = "yes"
                if result.get("neighborhood") and not venue.get("neighborhood"):
                    venue["neighborhood"] = result.get("neighborhood")
            updated += 1
        else:
            print("not found")
            not_found += 1

    print(f"Backfill results: updated={updated}, not_found={not_found}")

    if not dry_run and updated > 0:
        _save_venues_to_sheet(venues)
        print("Saved coordinate updates to sheet.")

    return {
        "targets": len(to_backfill),
        "updated": updated,
        "not_found": not_found,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich venue addresses using Google Places API")
    parser.add_argument("--apply", action="store_true", help="Actually update venues (default is dry run)")
    parser.add_argument("--limit", type=int, help="Limit number of venues to process")
    parser.add_argument("--city", default="NYC", help="City to process")
    parser.add_argument(
        "--backfill-coordinates",
        action="store_true",
        help="Backfill missing lat/lng for venues in cache",
    )
    parser.add_argument(
        "--include-bad-addresses",
        action="store_true",
        help="When backfilling coordinates, also attempt vague addresses",
    )
    parser.add_argument(
        "--events-url-only",
        action="store_true",
        help="When backfilling coordinates, only process venues with events_url",
    )

    args = parser.parse_args()

    if args.backfill_coordinates:
        backfill_venue_coordinates(
            city=args.city,
            dry_run=not args.apply,
            limit=args.limit,
            include_bad_addresses=args.include_bad_addresses,
            require_events_url=args.events_url_only,
        )
    else:
        enrich_venues(city=args.city, dry_run=not args.apply, limit=args.limit)
