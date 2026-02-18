#!/usr/bin/env python3
"""
Run Gemini Grounding discovery for remaining categories.
Combines with MVP results and exports to a new Google Sheet.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from venue_scout.discovery_gemini import discover_venues
from venue_scout.cache import read_cached_venues, _get_sheets_service
import settings

# Categories already completed in MVP run
MVP_CATEGORIES = [
    "art galleries",
    "jazz clubs",
    "broadway theaters",
    "bookstores with author events",
    "comedy clubs standup",
    "outdoor concert venues amphitheaters",
    "museums",
]


def export_to_new_sheet(venues: list, sheet_name: str) -> str | None:
    """Export all venues to a new Google Sheet."""
    service = _get_sheets_service()
    if not service:
        print("ERROR: Could not get Sheets service")
        return None

    # Create new spreadsheet
    spreadsheet = service.spreadsheets().create(
        body={
            'properties': {'title': sheet_name},
            'sheets': [{'properties': {'title': 'Venues'}}]
        }
    ).execute()

    spreadsheet_id = spreadsheet['spreadsheetId']
    print(f"\nCreated spreadsheet: {sheet_name}")
    print(f"ID: {spreadsheet_id}")

    # Prepare data - all columns from Venue schema
    headers = [
        'name', 'address', 'city', 'neighborhood',
        'website', 'events_url', 'category', 'description', 'source',
        'address_verified', 'website_status', 'website_attempts',
        'preferred_event_source', 'api_endpoint', 'ticketmaster_venue_id',
        'last_event_fetch', 'event_count', 'event_source',
    ]

    rows = [headers]
    for v in venues:
        rows.append([
            v.get('name', ''),
            v.get('address', ''),
            v.get('city', ''),
            v.get('neighborhood', ''),
            v.get('website', ''),
            v.get('events_url', ''),
            v.get('category', ''),
            v.get('description', ''),
            v.get('source', ''),
            v.get('address_verified', ''),
            v.get('website_status', ''),
            str(v.get('website_attempts', 0) or ''),
            v.get('preferred_event_source', ''),
            v.get('api_endpoint', ''),
            v.get('ticketmaster_venue_id', ''),
            v.get('last_event_fetch', ''),
            str(v.get('event_count', 0) or ''),
            v.get('event_source', ''),
        ])

    # Write data
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range='Venues!A1',
        valueInputOption='RAW',
        body={'values': rows}
    ).execute()

    print(f"Wrote {len(venues)} venues to sheet")
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    print(f"URL: {url}")

    return spreadsheet_id


def main():
    print("=" * 60)
    print("RUNNING REMAINING CATEGORIES")
    print("=" * 60)

    # Get all categories from settings
    all_categories = settings.VENUE_CATEGORIES_ALL
    print(f"\nTotal categories defined: {len(all_categories)}")

    # Filter out MVP categories
    remaining = [cat for cat in all_categories if cat not in MVP_CATEGORIES]
    print(f"MVP categories already done: {len(MVP_CATEGORIES)}")
    print(f"Remaining categories to run: {len(remaining)}")

    print("\nCategories to discover:")
    for cat in remaining:
        print(f"  - {cat}")

    # Run discovery for remaining categories
    # Using force=False so it checks cache metadata
    print("\n" + "=" * 60)
    print("STARTING DISCOVERY")
    print("=" * 60)

    discover_venues(
        city="NYC",
        categories=remaining,
        force=True,  # Force re-run since we want fresh data
    )

    # Read all cached venues
    print("\n" + "=" * 60)
    print("READING ALL CACHED VENUES")
    print("=" * 60)

    all_venues = read_cached_venues("NYC")
    print(f"Total venues in cache: {len(all_venues)}")

    # Summary by category
    by_category = Counter(v.get('category', 'unknown') for v in all_venues)
    print("\nVenues by category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save JSON backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = Path(__file__).parent / f"full_results_{timestamp}.json"

    # Convert to regular dicts for JSON serialization
    venues_for_json = [dict(v) for v in all_venues]
    with open(json_path, 'w') as f:
        json.dump(venues_for_json, f, indent=2)
    print(f"\nSaved JSON backup to: {json_path}")

    # Export to new sheet
    sheet_name = f"Gemini Grounding Full Discovery {timestamp}"
    export_to_new_sheet(all_venues, sheet_name)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
