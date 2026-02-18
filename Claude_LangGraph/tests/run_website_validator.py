#!/usr/bin/env python3
"""
Run website validator on venues needing validation.
Updates the Venue Scout Cache sheet with results.
"""

import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.cache import read_cached_venues, update_venues_batch
from venue_scout.website_validator import validate_venue_website


def run_validation(batch_size: int = 50, max_venues: int = None):
    """Run validation on venues with empty website_status."""

    print("=" * 60)
    print("WEBSITE VALIDATOR")
    print("=" * 60)

    # Read all venues
    venues = read_cached_venues("NYC")
    print(f"Total venues in cache: {len(venues)}")

    # Filter to those needing validation
    needs_validation = [
        v for v in venues
        if not v.get("website_status")
    ]
    print(f"Venues needing validation: {len(needs_validation)}")

    if max_venues:
        needs_validation = needs_validation[:max_venues]
        print(f"Processing first {max_venues} venues")

    if not needs_validation:
        print("No venues need validation!")
        return

    # Process in batches
    total = len(needs_validation)
    validated = []
    errors = []

    for i in range(0, total, batch_size):
        batch = needs_validation[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch)} venues)")
        print(f"{'='*60}")

        batch_results = []
        for j, venue in enumerate(batch, 1):
            name = venue.get("name", "")[:40]
            print(f"\n[{i + j}/{total}] {name}")

            try:
                updated = validate_venue_website(
                    venue,
                    city="NYC",
                    max_attempts=3,
                    verify_with_llm=True,
                    find_events=True,
                )
                batch_results.append(updated)

                status = updated.get("website_status", "")
                website = updated.get("website", "")[:50] if updated.get("website") else ""
                events_url = updated.get("events_url", "")[:50] if updated.get("events_url") else ""

                print(f"    Status: {status}")
                if website:
                    print(f"    Website: {website}")
                if events_url:
                    print(f"    Events: {events_url}")

            except Exception as e:
                print(f"    ERROR: {e}")
                errors.append((name, str(e)))
                batch_results.append(venue)

            # Small delay between venues
            time.sleep(0.5)

        validated.extend(batch_results)

        # Write batch results to cache
        if batch_results:
            print(f"\nSaving batch {batch_num} to cache...")
            updated_count = update_venues_batch(batch_results, "NYC")
            print(f"Updated {updated_count} venues in cache")

        # Summary for this batch
        batch_statuses = Counter(v.get("website_status", "unknown") for v in batch_results)
        print(f"\nBatch summary: {dict(batch_statuses)}")

    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")

    final_statuses = Counter(v.get("website_status", "unknown") for v in validated)
    print("\nResults by status:")
    for status, count in final_statuses.most_common():
        print(f"  {status}: {count}")

    with_website = sum(1 for v in validated if v.get("website"))
    with_events = sum(1 for v in validated if v.get("events_url"))
    print(f"\nWith website: {with_website}")
    print(f"With events_url: {with_events}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, error in errors[:10]:
            print(f"  {name}: {error}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-venues", type=int, default=None)
    args = parser.parse_args()

    run_validation(batch_size=args.batch_size, max_venues=args.max_venues)
