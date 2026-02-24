"""
Daily event refresh job.

Runs as a Cloud Run Job on a daily schedule. Selects the 100 venues
with the oldest last_event_fetch (never-fetched first), fetches their
events, writes results to Google Sheets, and rebuilds the semantic index
incrementally (only new/changed events get re-embedded).
"""

import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

CITY = "NYC"
BATCH_SIZE = 100


def _sort_key(venue: dict):
    """Sort venues by last_event_fetch ascending — None/empty first (highest priority)."""
    raw = venue.get("last_event_fetch", "") or ""
    if not raw.strip():
        return ""  # sorts before any real date
    return raw.strip()


def main():
    start = time.time()
    now = datetime.now(ZoneInfo("America/New_York"))
    print(f"=== Whaddup NYC — Daily Event Refresh Job ===")
    print(f"Started: {now.isoformat()}")
    print(f"Batch size: {BATCH_SIZE} venues")
    print()

    # --- 1. Load all venues ---
    print("Loading venues from Google Sheets...")
    from venue_scout.cache import read_cached_venues
    all_venues = read_cached_venues(CITY)
    print(f"  Loaded {len(all_venues)} venues total")

    if not all_venues:
        print("ERROR: No venues found. Exiting.")
        sys.exit(1)

    # --- 1b. Filter to venues that have something to fetch ---
    # Only include venues with an events_url or a ticketmaster_venue_id.
    fetchable = [
        v for v in all_venues
        if v.get("events_url") or v.get("ticketmaster_venue_id")
    ]
    print(f"  {len(fetchable)} fetchable venues ({len(all_venues) - len(fetchable)} skipped — no events_url or TM ID)")

    # --- 2. Sort by last_event_fetch ASC (never-fetched first), take top BATCH_SIZE ---
    sorted_venues = sorted(fetchable, key=_sort_key)
    batch = sorted_venues[:BATCH_SIZE]

    never_fetched = sum(1 for v in batch if not (v.get("last_event_fetch") or "").strip())
    print(f"  Selected {len(batch)} venues for this run ({never_fetched} never fetched)")
    if batch:
        oldest = batch[-1].get("last_event_fetch", "never") or "never"
        print(f"  Oldest last fetch in batch: {oldest}")
    print()

    # --- 3. Fetch events ---
    print("Fetching events...")
    from venue_scout.event_fetcher import fetch_events_for_venues
    results = fetch_events_for_venues(
        venues=batch,
        force_refresh=True,   # we've already chosen which venues to refresh
        city=CITY,
        save_to_sheet=True,
        workers=1,            # sequential — avoids Sheets API rate limits
    )

    fetched = sum(1 for r in results.values() if r.events)
    errors = sum(1 for r in results.values() if r.error)
    total_events = sum(len(r.events) for r in results.values())
    print(f"\nFetch summary: {fetched} venues with events, {total_events} new events, {errors} errors")
    print()

    # --- 4. Rebuild semantic index (incremental — only new/changed events get re-embedded) ---
    print("Rebuilding semantic index...")
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet
    from venue_scout.semantic_search import build_semantic_index

    all_events = read_venue_events_from_sheet()
    print(f"  Loaded {len(all_events)} total events from sheet")

    index_result = build_semantic_index(all_events)
    reused = index_result.get("incremental_reused_count", 0)
    embedded = index_result.get("incremental_embedded_count", 0)
    print(f"  Index built: {embedded} events newly embedded, {reused} reused from cache")
    print()

    elapsed = time.time() - start
    print(f"=== Job complete in {elapsed / 60:.1f} minutes ===")


if __name__ == "__main__":
    main()
