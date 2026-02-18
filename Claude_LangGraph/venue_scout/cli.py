#!/usr/bin/env python3
"""CLI entry point for venue event fetching.

Usage:
    # Fetch events for specific venues
    python -m venue_scout.cli --venues "Beacon Theater" "Carnegie Hall"

    # Fetch events for venues in a category
    python -m venue_scout.cli --city NYC --categories "music venue" "comedy clubs"

    # Force refresh (ignore cache)
    python -m venue_scout.cli --venues "Beacon Theater" --force

    # Export to Google Sheets
    python -m venue_scout.cli --categories "music venue" --export

    # Show matched events only
    python -m venue_scout.cli --matched

    # Show cache status
    python -m venue_scout.cli --status
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.cache import read_cached_venues, update_venues_batch
from venue_scout.event_cache import get_cache_summary, clear_venue_cache
from venue_scout.event_fetcher import fetch_events_for_venues, fetch_venue_events
from venue_scout.concert_matcher import match_events_to_artists, get_user_artists, highlight_matched_events
from venue_scout.venue_events_sheet import (
    read_venue_events_from_sheet,
    write_venue_events_to_sheet,
    get_matched_events,
)
from venue_scout.website_validator import (
    validate_venue_website,
    is_aggregator_url,
    get_validation_summary,
)
from venue_scout.venue_cleaner import clean_venues


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


def _has_ticketmaster_id(venue: dict) -> bool:
    """Return True if a venue has a usable Ticketmaster venue ID."""
    tm_id = str(venue.get("ticketmaster_venue_id", "") or "").strip()
    return bool(tm_id and tm_id.lower() != "not_found")


def cmd_fetch_venues(args):
    """Fetch events for specific venues."""
    import os
    if getattr(args, 'skip_jina', False):
        os.environ["EVENT_FETCHER_SKIP_JINA"] = "1"
        print("Skip-Jina mode: using raw HTML instead of Jina Reader")

    venue_names = args.venues
    city = args.city
    force = args.force

    # Get venue details from cache
    all_venues = read_cached_venues(city)
    venues = []

    for name in venue_names:
        name_lower = name.lower()
        found = None
        for v in all_venues:
            if v.get("name", "").lower() == name_lower:
                found = v
                break
            # Partial match
            if name_lower in v.get("name", "").lower():
                found = v

        if found:
            venues.append(found)
        else:
            # Create minimal venue entry
            venues.append({
                "name": name,
                "category": "",
                "website": "",
                "city": city,
            })
            print(f"Note: Venue '{name}' not in cache, using minimal info")

    if not venues:
        print("No venues to fetch")
        return

    # Fetch events
    workers = getattr(args, 'workers', 1) or 1
    resume = getattr(args, 'resume', False)
    results = fetch_events_for_venues(
        venues=venues,
        force_refresh=force,
        city=city,
        save_to_sheet=args.export,
        workers=workers,
        resume=resume,
    )

    # Match to artists if requested
    if args.match:
        artists = get_user_artists()
        all_events = []
        for result in results.values():
            all_events.extend(result.events)

        matched_events = match_events_to_artists(all_events, artists)
        matched, other = highlight_matched_events(matched_events)

        if matched:
            print(f"\n{'='*50}")
            print(f"MATCHED EVENTS ({len(matched)}):")
            for event in matched:
                print(f"  [{event.get('matched_artist')}] {event.get('date_str')}: {event.get('name')}")


def cmd_fetch_categories(args):
    """Fetch events for venues in categories."""
    import os
    if getattr(args, 'skip_jina', False):
        os.environ["EVENT_FETCHER_SKIP_JINA"] = "1"
        print("Skip-Jina mode: using raw HTML instead of Jina Reader")

    categories = args.categories
    city = args.city
    force = args.force

    # Get all venues and filter by category
    all_venues = read_cached_venues(city)

    venues = []
    for v in all_venues:
        v_cat = v.get("category", "").lower()
        for cat in categories:
            if cat.lower() in v_cat or v_cat in cat.lower():
                venues.append(v)
                break

    if not venues:
        print(f"No venues found in categories: {categories}")
        print(f"Available categories: {set(v.get('category') for v in all_venues)}")
        return

    print(f"Found {len(venues)} venues in categories: {categories}")

    # Fetch events
    workers = getattr(args, 'workers', 1) or 1
    resume = getattr(args, 'resume', False)
    results = fetch_events_for_venues(
        venues=venues,
        force_refresh=force,
        city=city,
        save_to_sheet=args.export,
        workers=workers,
        resume=resume,
    )

    # Match to artists
    if args.match:
        artists = get_user_artists()
        all_events = []
        for result in results.values():
            all_events.extend(result.events)

        matched_events = match_events_to_artists(all_events, artists)
        matched, other = highlight_matched_events(matched_events)

        if matched:
            print(f"\n{'='*50}")
            print(f"MATCHED EVENTS ({len(matched)}):")
            for event in sorted(matched, key=lambda x: x.get("date_str", "")):
                print(f"  [{event.get('matched_artist')}] {event.get('date_str')}: {event.get('name')}")
                print(f"    @ {event.get('venue_name')}")


def cmd_fetch_all(args):
    """Fetch events for all verified venues."""
    import os
    if getattr(args, 'skip_jina', False):
        os.environ["EVENT_FETCHER_SKIP_JINA"] = "1"
        print("Skip-Jina mode: using raw HTML instead of Jina Reader")

    city = args.city
    force = args.force

    # Get all verified venues
    all_venues = read_cached_venues(city)
    venues = [v for v in all_venues if v.get("website_status") in ("verified", "reachable_no_events_page")]

    if not venues:
        print(f"No verified venues found for {city}")
        return

    print(f"Found {len(venues)} verified venues")

    # Show breakdown by source
    with_tm = len([v for v in venues if _has_ticketmaster_id(v)])
    with_api = len([v for v in venues if v.get("api_endpoint")])
    need_scrape = len(venues) - with_tm - with_api + len([v for v in venues if _has_ticketmaster_id(v) and v.get("api_endpoint")])

    print(f"  Ticketmaster: {with_tm}")
    print(f"  Direct API: {with_api}")
    print(f"  Need scraping: ~{len(venues) - with_tm - with_api}")

    # Fetch events
    workers = getattr(args, 'workers', 1) or 1
    resume = getattr(args, 'resume', False)
    results = fetch_events_for_venues(
        venues=venues,
        force_refresh=force,
        city=city,
        save_to_sheet=args.export,
        workers=workers,
        resume=resume,
    )

    # Match to artists if requested
    if args.match:
        artists = get_user_artists()
        all_events = []
        for result in results.values():
            all_events.extend(result.events)

        matched_events = match_events_to_artists(all_events, artists)
        matched, other = highlight_matched_events(matched_events)

        if matched:
            print(f"\n{'='*50}")
            print(f"MATCHED EVENTS ({len(matched)}):")
            for event in sorted(matched, key=lambda x: x.get("date_str", ""))[:20]:
                print(f"  [{event.get('matched_artist')}] {event.get('date_str')}: {event.get('name')}")
                print(f"    @ {event.get('venue_name')}")
            if len(matched) > 20:
                print(f"  ... and {len(matched) - 20} more")


def cmd_show_matched(args):
    """Show events matching user's artists."""
    # Get events from sheet
    events = read_venue_events_from_sheet()

    if not events:
        print("No events in cache. Run fetch first.")
        return

    # Get user's artists
    artists = get_user_artists()
    if not artists:
        print("Could not get user's artists from YouTube Music")
        return

    # Match events
    matched_events = match_events_to_artists(events, artists)
    matched, other = highlight_matched_events(matched_events)

    if not matched:
        print("No events match your artists")
        return

    print(f"Found {len(matched)} events matching your artists:\n")

    # Group by artist
    by_artist = {}
    for event in matched:
        artist = event.get("matched_artist")
        if artist not in by_artist:
            by_artist[artist] = []
        by_artist[artist].append(event)

    for artist, artist_events in sorted(by_artist.items()):
        print(f"\n{artist} ({len(artist_events)} events):")
        for event in sorted(artist_events, key=lambda x: x.get("date_str", "")):
            print(f"  {event.get('date_str')}: {event.get('name')}")
            print(f"    @ {event.get('venue_name')}")
            if event.get("url"):
                print(f"    {event.get('url')}")


def cmd_show_status(args):
    """Show cache status."""
    summary = get_cache_summary()

    print("Venue Event Cache Status")
    print("=" * 40)
    print(f"Total venues fetched: {summary['total_venues_fetched']}")
    print(f"Fresh venues: {summary['fresh_venues']}")
    print(f"Stale venues: {summary['stale_venues']}")
    print(f"Total events: {summary['total_events']}")

    if summary['venues_by_city']:
        print("\nBy city:")
        for city, count in summary['venues_by_city'].items():
            print(f"  {city}: {count} venues")

    if summary['venues_by_source']:
        print("\nBy source:")
        for source, count in summary['venues_by_source'].items():
            print(f"  {source}: {count} venues")


def cmd_clear_cache(args):
    """Clear venue event cache."""
    if args.venue:
        clear_venue_cache(venue_name=args.venue, city=args.city)
        print(f"Cleared cache for venue: {args.venue}")
    elif args.city:
        clear_venue_cache(city=args.city)
        print(f"Cleared cache for city: {args.city}")
    else:
        clear_venue_cache()
        print("Cleared all venue event cache")


def cmd_export(args):
    """Export events to Google Sheet."""
    events = read_venue_events_from_sheet()
    if not events:
        print("No events to export")
        return

    # Match to artists first
    artists = get_user_artists()
    if artists:
        events = match_events_to_artists(events, artists)

    write_venue_events_to_sheet(events)
    print(f"Exported {len(events)} events to Google Sheet")


def cmd_scan_apis(args):
    """Scan venues for API endpoints and save discovered endpoints."""
    import time
    city = args.city

    # Get all venues with websites but no known API endpoint
    all_venues = read_cached_venues(city)
    print(f"Total venues: {len(all_venues)}")

    # Filter to venues with websites but no API endpoint
    to_scan = []
    already_has_api = 0
    no_website = 0

    for v in all_venues:
        website = v.get("website", "")
        api_endpoint = v.get("api_endpoint", "")
        status = v.get("website_status", "")

        if api_endpoint:
            already_has_api += 1
        elif not website or status != "verified":
            no_website += 1
        else:
            # Filter by category if specified
            if args.category:
                v_cat = v.get("category", "").lower()
                if args.category.lower() not in v_cat:
                    continue
            to_scan.append(v)

    print(f"Already have API endpoint: {already_has_api}")
    print(f"No verified website: {no_website}")
    print(f"Candidates for API scan: {len(to_scan)}")

    if not to_scan:
        print("\nNo venues to scan!")
        return

    # Limit if specified
    if args.limit:
        to_scan = to_scan[:args.limit]
        print(f"\nLimited to {args.limit} venues")

    print(f"\nScanning {len(to_scan)} venues for API endpoints...")
    print("=" * 60)

    # Import API detector
    try:
        from .api_detector import detect_and_fetch
    except ImportError as e:
        print(f"Error: Could not import API detector: {e}")
        print("Make sure Playwright is installed: pip install playwright && playwright install")
        return

    found_apis = []
    total_found = 0
    no_api_found = 0
    errors = 0
    batch_size = 10  # Save every 10 venues

    for i, venue in enumerate(to_scan, 1):
        name = venue.get("name", "")[:40]
        website = venue.get("website", "")

        print(f"\n[{i}/{len(to_scan)}] {name}")
        print(f"    URL: {website[:60]}...")

        try:
            events, api_url = detect_and_fetch(website, venue.get("name", ""))

            if api_url and events:
                print(f"    ✓ Found API: {api_url[:60]}...")
                print(f"    ✓ Extracted {len(events)} events")

                # Update venue with API endpoint
                venue["api_endpoint"] = api_url
                venue["preferred_event_source"] = "api"
                found_apis.append(venue)
                total_found += 1
            else:
                print(f"    ✗ No API detected")
                no_api_found += 1

        except Exception as e:
            print(f"    ✗ Error: {e}")
            errors += 1

        # Save batch every N venues
        if len(found_apis) >= batch_size:
            print(f"\n  Saving batch of {len(found_apis)} venues with API endpoints...")
            update_venues_batch(found_apis, city)
            found_apis = []

        # Rate limit
        time.sleep(2)

    # Save remaining
    if found_apis:
        print(f"\n{'='*60}")
        print(f"Saving final batch of {len(found_apis)} venues with API endpoints...")
        update_venues_batch(found_apis, city)

    print(f"\n{'='*60}")
    print(f"Scan complete!")
    print(f"  Found API endpoints: {total_found}")
    print(f"  No API detected: {no_api_found}")
    print(f"  Errors: {errors}")


def cmd_scan_ticketmaster(args):
    """Scan venues to check if they're on Ticketmaster."""
    import time
    city = args.city

    # Categories likely to have ticketed events
    TICKETED_CATEGORIES = {
        'music venue', 'concert halls', 'rock music venues', 'jazz clubs',
        'broadway theaters', 'off broadway theaters', 'comedy clubs',
        'dance venues ballet contemporary', 'performing arts centers',
        'live music bars', 'amphitheaters', 'arenas stadiums',
    }

    # Get all venues
    all_venues = read_cached_venues(city)
    print(f"Total venues: {len(all_venues)}")

    # Filter to venues that might be on Ticketmaster
    to_scan = []
    already_checked = 0
    not_ticketed_category = 0

    for v in all_venues:
        tm_id = v.get("ticketmaster_venue_id", "")
        category = v.get("category", "").lower()

        # Skip if already checked
        if tm_id:  # Has ID or "not_found"
            already_checked += 1
            continue

        # Check if category is likely to have ticketed events
        is_ticketed = any(cat in category for cat in TICKETED_CATEGORIES)
        if not is_ticketed:
            not_ticketed_category += 1
            continue

        to_scan.append(v)

    print(f"Already checked: {already_checked}")
    print(f"Not ticketed category: {not_ticketed_category}")
    print(f"Need to scan: {len(to_scan)}")

    if not to_scan:
        print("\nNo venues to scan!")
        return

    # Limit if specified
    if args.limit:
        to_scan = to_scan[:args.limit]
        print(f"\nLimited to {args.limit} venues")

    print(f"\nScanning {len(to_scan)} venues on Ticketmaster...")
    print("=" * 60)

    # Initialize Ticketmaster client
    try:
        from concert_finder.ticketmaster_client import TicketmasterClient
        tm_client = TicketmasterClient()
    except Exception as e:
        print(f"Error: Could not initialize Ticketmaster client: {e}")
        return

    found = []
    not_found = []
    total_found = 0
    total_not_found = 0
    batch_size = 25

    for i, venue in enumerate(to_scan, 1):
        name = venue.get("name", "")
        print(f"[{i}/{len(to_scan)}] {name[:45]}", end="", flush=True)

        try:
            venue_id = tm_client.search_venue_id(name, city="New York", state_code="NY")

            if venue_id:
                print(f" ✓ Found: {venue_id}")
                venue["ticketmaster_venue_id"] = venue_id
                found.append(venue)
                total_found += 1
            else:
                print(" ✗ Not found")
                venue["ticketmaster_venue_id"] = "not_found"
                not_found.append(venue)
                total_not_found += 1

        except Exception as e:
            print(f" ✗ Error: {e}")
            total_not_found += 1

        # Save batch
        if len(found) + len(not_found) >= batch_size:
            print(f"  Saving batch of {len(found) + len(not_found)} venues...")
            update_venues_batch(found + not_found, city)
            found = []
            not_found = []

        # Rate limit (TM allows 5/sec, but be conservative)
        time.sleep(0.3)

    # Save remaining
    if found or not_found:
        print(f"\n{'='*60}")
        print(f"Saving final batch of {len(found) + len(not_found)} venues...")
        update_venues_batch(found + not_found, city)

    print(f"\n{'='*60}")
    print(f"Scan complete!")
    print(f"  Found on Ticketmaster: {total_found}")
    print(f"  Not on Ticketmaster: {total_not_found}")


def _find_events_page_worker(args: tuple) -> tuple[dict, bool]:
    """Worker function for parallel events page discovery."""
    venue, index, total = args
    name = venue.get("name", "")[:40]
    website = venue.get("website", "")

    print(f"[{index}/{total}] {name}", flush=True)

    from .website_validator import find_events_page
    events_url = find_events_page(website, venue.get("name", ""))

    if events_url:
        venue["events_url"] = events_url
        return venue, True
    else:
        print(f"    ✗ No events page found", flush=True)
        return venue, False


def cmd_find_events_pages(args):
    """Find events/calendar pages for verified venues."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    city = args.city
    workers = getattr(args, 'workers', 1) or 1

    # Get all verified venues without events_url
    all_venues = read_cached_venues(city)
    print(f"Total venues: {len(all_venues)}")

    # Limit to a single venue if specified
    if getattr(args, 'venue', None):
        venue_name = args.venue
        all_venues = [v for v in all_venues if v.get("name", "").lower() == venue_name.lower()]
        print(f"Found {len(all_venues)} venues matching '{venue_name}'")

    to_scan = []
    already_has_events_url = 0
    has_ticketmaster = 0
    not_verified = 0

    for v in all_venues:
        status = v.get("website_status", "")
        website = v.get("website", "")
        events_url = v.get("events_url", "")

        if events_url:
            already_has_events_url += 1
        elif _has_ticketmaster_id(v):
            # Has Ticketmaster - don't need to scrape website for events
            has_ticketmaster += 1
        elif status != "verified" or not website:
            not_verified += 1
        else:
            to_scan.append(v)

    print(f"Already have events_url: {already_has_events_url}")
    print(f"Have Ticketmaster (skip): {has_ticketmaster}")
    print(f"Not verified or no website: {not_verified}")
    print(f"Need events page discovery: {len(to_scan)}")

    if not to_scan:
        print("\nNo venues need events page discovery!")
        return

    # Limit if specified
    if args.limit:
        to_scan = to_scan[:args.limit]
        print(f"\nLimited to {args.limit} venues")

    print(f"\nFinding events pages for {len(to_scan)} venues...")
    if workers > 1:
        print(f"Using {workers} parallel workers")
    print("=" * 60)

    updated = []
    found = 0
    not_found = 0
    batch_size = 25
    total = len(to_scan)

    if workers > 1:
        # Parallel execution
        work_args = [(v, i, total) for i, v in enumerate(to_scan, 1)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_find_events_page_worker, args): args for args in work_args}

            for future in as_completed(futures):
                try:
                    venue, was_found = future.result()
                    updated.append(venue)
                    if was_found:
                        found += 1
                    else:
                        not_found += 1

                    # Save batch
                    if len(updated) >= batch_size:
                        print(f"  Saving batch of {len(updated)} venues...")
                        update_venues_batch(updated, city)
                        updated = []

                except Exception as e:
                    print(f"  Error: {e}", flush=True)
                    not_found += 1
    else:
        # Sequential execution
        import time
        sequential_delay = float(getattr(_settings, "WEBSITE_FIND_EVENTS_CLI_DELAY_SEC", 1.5))
        for i, venue in enumerate(to_scan, 1):
            venue, was_found = _find_events_page_worker((venue, i, total))
            updated.append(venue)
            if was_found:
                found += 1
            else:
                not_found += 1

            # Save batch
            if len(updated) >= batch_size:
                print(f"  Saving batch of {len(updated)} venues...")
                update_venues_batch(updated, city)
                updated = []

            # Rate limit for sequential
            time.sleep(sequential_delay)

    # Save remaining
    if updated:
        print(f"\n{'='*60}")
        print(f"Saving final batch of {len(updated)} venues...")
        update_venues_batch(updated, city)

    print(f"\nSummary:")
    print(f"  Events pages found: {found}")
    print(f"  Not found: {not_found}")


def _validate_website_worker(args: tuple) -> tuple[dict, str, str]:
    """Worker function for parallel website validation."""
    venue, city, max_attempts, find_events, index, total = args
    name = venue.get("name", "")[:40]
    old_url = venue.get("website", "")

    print(f"[{index}/{total}] {name}", flush=True)

    result = validate_venue_website(
        venue,
        city=city,
        max_attempts=max_attempts,
        find_events=find_events,
    )

    new_url = result.get("website", "")
    status = result.get("website_status", "")

    return result, old_url, status


def cmd_validate_websites(args):
    """Validate and discover venue websites."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Set skip-jina env var if flag is passed
    if getattr(args, "skip_jina", False):
        os.environ["WEBSITE_VALIDATOR_SKIP_JINA"] = "1"
        print("Skipping Jina Reader (using direct HTML fetch only)")

    city = args.city
    max_attempts = (
        getattr(args, "max_attempts", None)
        or int(getattr(_settings, "WEBSITE_VALIDATOR_MAX_ATTEMPTS", 3))
    )
    delay = (
        getattr(args, "delay", None)
        if getattr(args, "delay", None) is not None
        else float(getattr(_settings, "WEBSITE_VALIDATOR_CLI_DELAY_SEC", 0.4))
    )
    workers = getattr(args, 'workers', 1) or 1
    find_events = getattr(
        args,
        "find_events",
        bool(getattr(_settings, "WEBSITE_VALIDATOR_FIND_EVENTS_DURING_VALIDATION", True)),
    )

    # Get all venues
    all_venues = read_cached_venues(city)
    print(f"Total venues: {len(all_venues)}")

    # Filter to venues needing validation
    to_validate = []
    to_mark_verified = []  # Have website but no status
    already_verified = 0
    already_reachable_no_events = 0
    already_search_failed = 0
    already_ambiguous = 0
    already_closed = 0
    already_dead_site = 0
    already_unreachable = 0
    unknown_status = 0
    has_ticketmaster = 0

    for v in all_venues:
        status = v.get("website_status", "")
        website = v.get("website", "")
        events_url = str(v.get("events_url", "") or "").strip()
        attempts = v.get("website_attempts", 0)
        tm_id = str(v.get("ticketmaster_venue_id", "") or "").strip()

        # If Ticketmaster ID exists, skip website validation entirely.
        if _has_ticketmaster_id(v):
            has_ticketmaster += 1
            expected_reason = f"skipped website validation: has ticketmaster_venue_id={tm_id}"
            # Persist explicit skip status so future runs don't reconsider this venue.
            if status != "ticketmaster_skip" or v.get("validation_reason") != expected_reason:
                v["website_status"] = "ticketmaster_skip"
                v["validation_reason"] = expected_reason
                to_mark_verified.append(v)
            continue

        # Always re-validate if current URL is an aggregator (even if marked verified)
        if website and is_aggregator_url(website):
            v["website_status"] = ""  # Reset status to force re-validation
            to_validate.append(v)
        elif status == "verified":
            # Backfill missing events URLs during normal website validation pass.
            if find_events and website and not events_url:
                to_validate.append(v)
            else:
                already_verified += 1
        elif status == "reachable_no_events_page":
            # Re-probe these when events discovery is enabled.
            if find_events and website and not events_url:
                to_validate.append(v)
            else:
                already_reachable_no_events += 1
        elif status == "search_failed":
            # Retry search_failed while attempts remain under the active max-attempts.
            if attempts < max_attempts:
                to_validate.append(v)
            else:
                already_search_failed += 1
        elif status == "ambiguous":
            already_ambiguous += 1
        elif status == "closed":
            already_closed += 1
        elif status == "dead_site":
            already_dead_site += 1
        elif status == "unreachable":
            already_unreachable += 1
        elif not website:
            # Include search_failed if under max_attempts
            if status in ("search_failed", "unreachable") and attempts < max_attempts:
                to_validate.append(v)
            elif not status:
                to_validate.append(v)
            else:
                # Unknown non-empty status on a venue without website should be retried.
                unknown_status += 1
                to_validate.append(v)
        elif not status:
            # Has website but not verified - needs LLM verification
            to_validate.append(v)
        else:
            # Unknown status values should be revalidated, not silently skipped.
            unknown_status += 1
            to_validate.append(v)

    print(f"Already verified: {already_verified}")
    print(f"Already reachable_no_events_page: {already_reachable_no_events}")
    print(f"Already marked search_failed: {already_search_failed}")
    print(f"Already marked ambiguous: {already_ambiguous}")
    print(f"Already marked closed: {already_closed}")
    print(f"Already marked dead_site: {already_dead_site}")
    print(f"Already marked unreachable: {already_unreachable}")
    print(f"Unknown status (revalidate): {unknown_status}")
    print(f"Have Ticketmaster ID (skip): {has_ticketmaster}")
    print(f"Need validation: {len(to_validate)}")
    print(f"Inline events-page discovery: {'enabled' if find_events else 'disabled'}")

    # Batch update venues with existing valid websites
    if to_mark_verified:
        print(f"Updating {len(to_mark_verified)} venues with validation metadata...")
        update_venues_batch(to_mark_verified, city)

    if not to_validate:
        print("\nNo venues need validation!")
        return

    # Limit if specified
    if args.limit:
        to_validate = to_validate[:args.limit]
        print(f"\nLimited to {args.limit} venues")

    print(f"\nValidating {len(to_validate)} venues...")
    if workers > 1:
        print(f"Using {workers} parallel workers")
    print("=" * 60)

    validated = []
    found = 0
    reachable_no_events = 0
    search_failed = 0
    ambiguous = 0
    rejected = 0
    closed = 0
    dead_site = 0
    unreachable = 0
    aggregator_cleared = 0
    batch_size = 25  # Save every 25 venues
    total = len(to_validate)

    if workers > 1:
        # Parallel execution
        work_args = [(v, city, max_attempts, find_events, i, total) for i, v in enumerate(to_validate, 1)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_validate_website_worker, args): args for args in work_args}

            for future in as_completed(futures):
                try:
                    result, old_url, status = future.result()
                    validated.append(result)

                    new_url = result.get("website", "")

                    if old_url and is_aggregator_url(old_url) and not new_url:
                        aggregator_cleared += 1

                    if status == "verified" and new_url:
                        found += 1
                    elif status == "reachable_no_events_page" and new_url:
                        reachable_no_events += 1
                    elif status == "search_failed":
                        search_failed += 1
                    elif status == "ambiguous":
                        ambiguous += 1
                    elif status == "aggregator_rejected":
                        rejected += 1
                    elif status == "closed":
                        closed += 1
                    elif status == "dead_site":
                        dead_site += 1
                    elif status == "unreachable":
                        unreachable += 1

                    # Save batch
                    if len(validated) >= batch_size:
                        print(f"  Saving batch of {len(validated)} venues...")
                        update_venues_batch(validated, city)
                        validated = []

                except Exception as e:
                    print(f"  Error: {e}", flush=True)
    else:
        # Sequential execution
        import time
        for i, venue in enumerate(to_validate, 1):
            result, old_url, status = _validate_website_worker(
                (venue, city, max_attempts, find_events, i, total)
            )
            validated.append(result)

            new_url = result.get("website", "")

            if old_url and is_aggregator_url(old_url) and not new_url:
                aggregator_cleared += 1

            if status == "verified" and new_url:
                found += 1
            elif status == "reachable_no_events_page" and new_url:
                reachable_no_events += 1
            elif status == "search_failed":
                search_failed += 1
            elif status == "ambiguous":
                ambiguous += 1
            elif status == "aggregator_rejected":
                rejected += 1
            elif status == "closed":
                closed += 1
            elif status == "dead_site":
                dead_site += 1
            elif status == "unreachable":
                unreachable += 1

            # Save batch every 25 venues
            if len(validated) >= batch_size:
                print(f"  Saving batch of {len(validated)} venues...")
                update_venues_batch(validated, city)
                validated = []

            # Rate limit
            if delay and delay > 0:
                time.sleep(delay)

    # Save remaining
    if validated:
        print(f"\n{'='*60}")
        print(f"Saving final batch of {len(validated)} venues...")
        update_venues_batch(validated, city)

    print(f"\nSummary:")
    print(f"  Found websites: {found}")
    print(f"  Reachable but no events page: {reachable_no_events}")
    print(f"  Search failed: {search_failed}")
    print(f"  Ambiguous: {ambiguous}")
    print(f"  Aggregator rejected: {rejected}")
    print(f"  Closed venues: {closed}")
    print(f"  Dead sites: {dead_site}")
    print(f"  Unreachable: {unreachable}")
    print(f"  Aggregators cleared: {aggregator_cleared}")


def cmd_clean_venues(args):
    """Clean and enrich verified venues (Stage 3 of pipeline)."""
    city = args.city
    limit = getattr(args, "limit", None)
    dry_run = getattr(args, "dry_run", False)
    skip_addresses = getattr(args, "skip_addresses", False)
    skip_dedup = getattr(args, "skip_dedup", False)
    skip_descriptions = getattr(args, "skip_descriptions", False)

    stats = clean_venues(
        city=city,
        limit=limit,
        dry_run=dry_run,
        skip_addresses=skip_addresses,
        skip_dedup=skip_dedup,
        skip_descriptions=skip_descriptions,
    )

    if "error" in stats:
        print(f"\nError: {stats['error']}")
        return 1

    return 0


def cmd_validate_websites_sample(args):
    """Run validator on a small fixed sample for reliability checks."""
    city = args.city
    limit = args.limit or 15
    max_attempts = (
        getattr(args, "max_attempts", None)
        or int(getattr(_settings, "WEBSITE_VALIDATOR_MAX_ATTEMPTS", 3))
    )
    find_events = getattr(
        args,
        "find_events",
        bool(getattr(_settings, "WEBSITE_VALIDATOR_FIND_EVENTS_DURING_VALIDATION", True)),
    )

    all_venues = read_cached_venues(city)
    candidates = [v for v in all_venues if v.get("name")]
    sample = candidates[:limit]

    if not sample:
        print(f"No venues found for {city}")
        return

    print(f"Validating sample of {len(sample)} venues for {city}")
    print("=" * 60)

    results = []
    for i, venue in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] {venue.get('name', '')}")
        updated = validate_venue_website(
            venue,
            city=city,
            max_attempts=max_attempts,
            find_events=find_events,
        )
        results.append(updated)

    by_status = {}
    for venue in results:
        status = venue.get("website_status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    print("\nSample summary:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

    print("\nDetailed outcomes:")
    for venue in results:
        print(
            f"- {venue.get('name', '')}: {venue.get('website_status', '')} "
            f"| website={venue.get('website', '')} "
            f"| events_url={venue.get('events_url', '')} "
            f"| reason={venue.get('validation_reason', '')}"
        )


def main():
    default_validator_max_attempts = int(getattr(_settings, "WEBSITE_VALIDATOR_MAX_ATTEMPTS", 3))
    default_validator_delay = float(getattr(_settings, "WEBSITE_VALIDATOR_CLI_DELAY_SEC", 0.4))
    default_validator_find_events = bool(
        getattr(_settings, "WEBSITE_VALIDATOR_FIND_EVENTS_DURING_VALIDATION", True)
    )
    parser = argparse.ArgumentParser(
        description="Fetch and manage venue events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--city", default="NYC", help="City to search (default: NYC)")
    parser.add_argument("--force", "-f", action="store_true", help="Force refresh (ignore cache)")
    parser.add_argument("--export", "-e", action="store_true", help="Export to Google Sheet")
    parser.add_argument("--match", "-m", action="store_true", help="Match events to YouTube Music artists")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume interrupted fetch from local cache")
    parser.add_argument("--skip-jina", action="store_true", help="Skip Jina Reader, use raw HTML for LLM parsing (faster when Jina rate-limited)")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    # All subcommand
    subparsers.add_parser("all", help="Fetch events for ALL verified venues")

    # Venues subcommand
    venues_parser = subparsers.add_parser("venues", help="Fetch events for specific venues")
    venues_parser.add_argument("names", nargs="+", help="Venue names to fetch")

    # Categories subcommand
    cats_parser = subparsers.add_parser("categories", help="Fetch events for venue categories")
    cats_parser.add_argument("names", nargs="+", help="Category names to fetch")

    # Status subcommand
    subparsers.add_parser("status", help="Show cache status")

    # Matched subcommand
    subparsers.add_parser("matched", help="Show events matching your artists")

    # Clear subcommand
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--venue", help="Clear specific venue")

    # Export subcommand
    subparsers.add_parser("export", help="Export events to Google Sheet")

    # Validate websites subcommand
    validate_parser = subparsers.add_parser("validate-websites", help="Validate and discover venue websites")
    validate_parser.add_argument("--limit", type=int, help="Limit number of venues to validate")
    validate_parser.add_argument(
        "--max-attempts",
        type=int,
        default=default_validator_max_attempts,
        help=f"Max search attempts per venue (default: {default_validator_max_attempts})",
    )
    validate_parser.add_argument(
        "--delay",
        type=float,
        default=default_validator_delay,
        help=f"Delay between venues in seconds for sequential mode (default: {default_validator_delay})",
    )
    validate_events_group = validate_parser.add_mutually_exclusive_group()
    validate_events_group.add_argument(
        "--find-events",
        dest="find_events",
        action="store_true",
        default=default_validator_find_events,
        help=f"Discover events/calendar URLs during validation (default: {default_validator_find_events})",
    )
    validate_events_group.add_argument(
        "--no-find-events",
        dest="find_events",
        action="store_false",
        help="Skip events/calendar discovery during validation",
    )
    validate_parser.add_argument(
        "--skip-jina",
        action="store_true",
        default=False,
        help="Skip Jina Reader API, use only direct HTML fetch (faster when Jina is rate-limited)",
    )

    # Validate websites sample harness
    validate_sample_parser = subparsers.add_parser(
        "validate-websites-sample",
        help="Run validator on a small venue sample for reliability checks",
    )
    validate_sample_parser.add_argument("--limit", type=int, help="Number of sample venues to validate")
    validate_sample_parser.add_argument(
        "--max-attempts",
        type=int,
        default=default_validator_max_attempts,
        help=f"Max search attempts per venue (default: {default_validator_max_attempts})",
    )
    validate_sample_events_group = validate_sample_parser.add_mutually_exclusive_group()
    validate_sample_events_group.add_argument(
        "--find-events",
        dest="find_events",
        action="store_true",
        default=default_validator_find_events,
        help=f"Discover events/calendar URLs during validation (default: {default_validator_find_events})",
    )
    validate_sample_events_group.add_argument(
        "--no-find-events",
        dest="find_events",
        action="store_false",
        help="Skip events/calendar discovery during validation",
    )

    # Find events pages subcommand
    events_parser = subparsers.add_parser("find-events-pages", help="Find events/calendar pages for verified venues")
    events_parser.add_argument("--limit", type=int, help="Limit number of venues to scan")
    events_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")
    events_parser.add_argument("--venue", help="Scan a single venue by name")

    # Scan APIs subcommand
    scan_api_parser = subparsers.add_parser("scan-apis", help="Scan venues for API endpoints")
    scan_api_parser.add_argument("--limit", type=int, help="Limit number of venues to scan")
    scan_api_parser.add_argument("--category", help="Only scan venues in this category")

    # Scan Ticketmaster subcommand
    scan_tm_parser = subparsers.add_parser("scan-ticketmaster", help="Check which venues are on Ticketmaster")
    scan_tm_parser.add_argument("--limit", type=int, help="Limit number of venues to scan")

    # Clean venues subcommand (Stage 3 of pipeline)
    clean_parser = subparsers.add_parser(
        "clean-venues",
        help="Clean and enrich verified venues (verify addresses, remove duplicates, add descriptions)"
    )
    clean_parser.add_argument("--limit", type=int, help="Limit number of venues to process")
    clean_parser.add_argument("--dry-run", action="store_true", help="Don't save changes, just preview")
    clean_parser.add_argument("--skip-addresses", action="store_true", help="Skip address verification step")
    clean_parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication step")
    clean_parser.add_argument("--skip-descriptions", action="store_true", help="Skip description generation step")

    # Legacy args for backward compatibility
    parser.add_argument("--venues", nargs="+", help="Venue names (legacy)")
    parser.add_argument("--categories", nargs="+", help="Category names (legacy)")
    parser.add_argument("--status", action="store_true", help="Show status (legacy)")
    parser.add_argument("--matched", action="store_true", help="Show matched (legacy)")

    args = parser.parse_args()

    # Handle subcommands
    if args.command == "all":
        cmd_fetch_all(args)
    elif args.command == "venues":
        args.venues = args.names
        cmd_fetch_venues(args)
    elif args.command == "categories":
        args.categories = args.names
        cmd_fetch_categories(args)
    elif args.command == "status":
        cmd_show_status(args)
    elif args.command == "matched":
        cmd_show_matched(args)
    elif args.command == "clear":
        cmd_clear_cache(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "validate-websites":
        cmd_validate_websites(args)
    elif args.command == "validate-websites-sample":
        cmd_validate_websites_sample(args)
    elif args.command == "find-events-pages":
        cmd_find_events_pages(args)
    elif args.command == "scan-apis":
        cmd_scan_apis(args)
    elif args.command == "scan-ticketmaster":
        cmd_scan_ticketmaster(args)
    elif args.command == "clean-venues":
        cmd_clean_venues(args)
    # Legacy argument handling
    elif args.status:
        cmd_show_status(args)
    elif args.matched:
        cmd_show_matched(args)
    elif args.venues:
        cmd_fetch_venues(args)
    elif args.categories:
        cmd_fetch_categories(args)
    elif args.export:
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
