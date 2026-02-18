#!/usr/bin/env python3
"""Venue cleaning module for Stage 3 of the venue pipeline.

This module handles:
1. Address verification using Google Places API
2. Duplicate removal based on normalized venue names
3. Venue description generation using Gemini

Usage:
    python -m venue_scout.cli clean-venues --limit 50
"""

import importlib.util
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.cache import read_cached_venues, VENUE_COLUMNS, _venue_to_row, _get_sheets_service
from venue_scout.enrich_addresses import lookup_place, is_bad_address


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()

# Verified venues sheet ID
VERIFIED_SHEET_ID = "1-Px113MllBDvZlkh8XUOh2tManLpoj9rZ_ZUSHUzjnc"


def normalize_name(name: str) -> str:
    """Normalize a venue name for deduplication."""
    name = name.lower().strip()
    name = re.sub(r'^the\s+', '', name)
    name = re.sub(r'^\(', '', name)  # handle (Le) Poisson Rouge
    name = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def verify_address(venue: dict, city: str = "NYC") -> dict:
    """
    Verify and update a venue's address using Google Places API.

    Args:
        venue: Venue dict to update
        city: City for lookup context

    Returns:
        Updated venue dict with address_verified field set
    """
    name = venue.get("name", "")
    current_address = venue.get("address", "")

    # Skip if already verified
    if venue.get("address_verified") == "yes":
        return venue

    # Look up in Google Places
    result = lookup_place(name, city)

    if result and result.get("formatted_address"):
        new_address = result["formatted_address"]

        # Check if address changed significantly
        if is_bad_address(current_address):
            venue["address"] = new_address
            venue["address_verified"] = "yes"
            if result.get("neighborhood") and not venue.get("neighborhood"):
                venue["neighborhood"] = result["neighborhood"]
        elif current_address.lower().strip() != new_address.lower().strip():
            # Use Places address as it's more authoritative
            venue["address"] = new_address
            venue["address_verified"] = "yes"
            if result.get("neighborhood") and not venue.get("neighborhood"):
                venue["neighborhood"] = result["neighborhood"]
        else:
            venue["address_verified"] = "yes"
    else:
        venue["address_verified"] = "not_found"

    return venue


def remove_duplicates(venues: list[dict]) -> tuple[list[dict], int]:
    """
    Remove duplicate venues based on normalized name.

    Keeps the venue with the most complete data (website, events_url, etc.).

    Args:
        venues: List of venue dicts

    Returns:
        Tuple of (deduplicated venues, count removed)
    """
    seen = {}

    def score_venue(v: dict) -> int:
        """Score a venue by data completeness."""
        score = 0
        if v.get("website"):
            score += 3
        if v.get("events_url"):
            score += 2
        if v.get("address") and not is_bad_address(v.get("address", "")):
            score += 2
        if v.get("description"):
            score += 1
        if v.get("website_status") == "verified":
            score += 2
        if v.get("ticketmaster_venue_id") and v.get("ticketmaster_venue_id") != "not_found":
            score += 2
        return score

    for venue in venues:
        norm = normalize_name(venue.get("name", ""))
        if not norm:
            continue

        if norm not in seen:
            seen[norm] = venue
        else:
            # Keep the one with better data
            if score_venue(venue) > score_venue(seen[norm]):
                seen[norm] = venue

    unique = list(seen.values())
    removed = len(venues) - len(unique)

    return unique, removed


def generate_description(venue: dict) -> str:
    """
    Generate a short description for a venue using Gemini.

    Args:
        venue: Venue dict with name, category, address, website

    Returns:
        Generated description string (1-2 sentences)
    """
    from utils.llm import generate_content

    name = venue.get("name", "")
    category = venue.get("category", "")
    address = venue.get("address", "")
    website = venue.get("website", "")

    prompt = f"""Write a brief 1-2 sentence description for this venue. Be factual and concise.

Venue: {name}
Category: {category}
Location: {address}
Website: {website}

Focus on what type of events or experiences the venue offers. Do not include phrases like "Check out" or promotional language. Just describe what the venue is."""

    try:
        response = generate_content(prompt)
        # Clean up the response
        description = response.strip()
        # Remove any quotation marks that might wrap the response
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]
        return description
    except Exception as e:
        print(f"    Error generating description: {e}")
        return ""


def clean_venues(
    city: str = "NYC",
    limit: int | None = None,
    dry_run: bool = False,
    skip_addresses: bool = False,
    skip_dedup: bool = False,
    skip_descriptions: bool = False,
) -> dict:
    """
    Clean and enrich verified venues.

    This is Stage 3 of the venue pipeline:
    1. Verify addresses using Google Places API
    2. Remove duplicates
    3. Add descriptions using Gemini

    Args:
        city: City to clean venues for
        limit: Max venues to process (for testing)
        dry_run: If True, don't write changes
        skip_addresses: Skip address verification step
        skip_dedup: Skip deduplication step
        skip_descriptions: Skip description generation step

    Returns:
        Dict with cleaning stats
    """
    print(f"Loading verified venues from sheet...")

    # Read from verified sheet
    service = _get_sheets_service()
    if not service:
        print("Error: Could not connect to Google Sheets")
        return {"error": "sheets_connection_failed"}

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=VERIFIED_SHEET_ID,
            range="A:R"
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:
            print("No venues found in verified sheet")
            return {"error": "no_venues"}

        header = rows[0]
        venues = []
        for row in rows[1:]:
            while len(row) < len(header):
                row.append("")
            venue = dict(zip(header, row))
            # Filter by city if specified
            if city and venue.get("city", "").lower() != city.lower():
                continue
            venues.append(venue)

        print(f"Found {len(venues)} venues for {city}")

    except Exception as e:
        print(f"Error reading from sheet: {e}")
        return {"error": str(e)}

    stats = {
        "total_venues": len(venues),
        "addresses_verified": 0,
        "addresses_not_found": 0,
        "addresses_skipped": 0,
        "duplicates_removed": 0,
        "descriptions_added": 0,
        "descriptions_skipped": 0,
    }

    # Keep all venues for saving, but only process a subset if limit specified
    all_venues = venues
    if limit:
        venues_to_process = venues[:limit]
        print(f"Limited to {limit} venues for processing (out of {len(all_venues)} total)")
    else:
        venues_to_process = venues

    # Step 1: Verify addresses
    if not skip_addresses:
        print(f"\n{'='*60}")
        print("Step 1: Verifying addresses with Google Places API...")
        print("="*60)

        for i, venue in enumerate(venues_to_process, 1):
            name = venue.get("name", "")[:40]
            current_status = venue.get("address_verified", "")

            if current_status == "yes":
                stats["addresses_skipped"] += 1
                continue

            print(f"[{i}/{len(venues_to_process)}] {name}...", end=" ", flush=True)

            verify_address(venue, city)

            if venue.get("address_verified") == "yes":
                print(f"verified")
                stats["addresses_verified"] += 1
            else:
                print(f"not found")
                stats["addresses_not_found"] += 1

            # Rate limit for Google Places API
            time.sleep(float(getattr(_settings, "VENUE_ENRICH_PLACES_DELAY_SEC", 0.1)))

        print(f"\nAddress verification complete:")
        print(f"  Verified: {stats['addresses_verified']}")
        print(f"  Not found: {stats['addresses_not_found']}")
        print(f"  Already verified: {stats['addresses_skipped']}")

    # Step 2: Remove duplicates (always runs on ALL venues, skipped when using --limit)
    if not skip_dedup:
        if limit:
            print(f"\n{'='*60}")
            print("Step 2: Skipping dedup (not meaningful with --limit)")
            print("="*60)
        else:
            print(f"\n{'='*60}")
            print("Step 2: Removing duplicates...")
            print("="*60)

            all_venues, removed = remove_duplicates(all_venues)
            # Update venues_to_process to point to the deduped list
            venues_to_process = all_venues
            stats["duplicates_removed"] = removed

            print(f"Removed {removed} duplicates. {len(all_venues)} unique venues remain.")

    # Step 3: Generate descriptions
    if not skip_descriptions:
        print(f"\n{'='*60}")
        print("Step 3: Generating descriptions with Gemini...")
        print("="*60)

        for i, venue in enumerate(venues_to_process, 1):
            name = venue.get("name", "")[:40]

            # Skip if already has description
            if venue.get("description"):
                stats["descriptions_skipped"] += 1
                continue

            print(f"[{i}/{len(venues_to_process)}] {name}...", end=" ", flush=True)

            description = generate_description(venue)

            if description:
                venue["description"] = description
                print(f"done ({len(description)} chars)")
                stats["descriptions_added"] += 1
            else:
                print(f"failed")

            # Rate limit for Gemini
            time.sleep(float(getattr(_settings, "VENUE_DESCRIPTION_DELAY_SEC", 0.5)))

        print(f"\nDescription generation complete:")
        print(f"  Added: {stats['descriptions_added']}")
        print(f"  Skipped (already had): {stats['descriptions_skipped']}")

    # Write back to sheet (always saves ALL venues, not just the processed subset)
    if not dry_run:
        print(f"\n{'='*60}")
        print("Saving cleaned venues to sheet...")
        print("="*60)

        # Sort all venues
        all_venues.sort(key=lambda x: (
            x.get("city", "").lower(),
            x.get("category", "").lower(),
            x.get("name", "").lower(),
        ))

        # Convert to rows
        rows = [VENUE_COLUMNS]
        for venue in all_venues:
            rows.append(_venue_to_row(venue))

        try:
            # Clear and write
            service.spreadsheets().values().clear(
                spreadsheetId=VERIFIED_SHEET_ID,
                range="A:R"
            ).execute()

            service.spreadsheets().values().update(
                spreadsheetId=VERIFIED_SHEET_ID,
                range="A1",
                valueInputOption="RAW",
                body={"values": rows}
            ).execute()

            print(f"Saved {len(all_venues)} venues to verified sheet")

        except Exception as e:
            print(f"Error saving to sheet: {e}")
            stats["error"] = str(e)
    else:
        print(f"\nDry run - not saving changes")

    # Summary
    print(f"\n{'='*60}")
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Venues processed: {len(venues_to_process)} (out of {len(all_venues)} total)")
    print(f"Addresses verified: {stats['addresses_verified']}")
    print(f"Addresses not found: {stats['addresses_not_found']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Descriptions added: {stats['descriptions_added']}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean and enrich verified venues")
    parser.add_argument("--city", default="NYC", help="City to process")
    parser.add_argument("--limit", type=int, help="Limit venues to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes")
    parser.add_argument("--skip-addresses", action="store_true", help="Skip address verification")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--skip-descriptions", action="store_true", help="Skip description generation")

    args = parser.parse_args()

    clean_venues(
        city=args.city,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_addresses=args.skip_addresses,
        skip_dedup=args.skip_dedup,
        skip_descriptions=args.skip_descriptions,
    )
