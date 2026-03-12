#!/usr/bin/env python3
"""One-time migration: copy existing Google Sheets data to Firestore.

Run this ONCE after deploying the Firestore migration to copy all existing
event and venue data from Sheets into Firestore collections.

Usage:
    python -m venue_scout.migrate_to_firestore
    python -m venue_scout.migrate_to_firestore --dry-run
    python -m venue_scout.migrate_to_firestore --skip-events
    python -m venue_scout.migrate_to_firestore --skip-venues
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))


def _migrate_events(dry_run: bool = False) -> int:
    """Read all events from Google Sheets and write them to Firestore.

    Returns the number of events migrated.
    """
    print("\n=== Migrating Events (Sheets → Firestore) ===")

    # --- Read from Sheets using the legacy API directly ---
    try:
        from utils.google_auth import get_credentials
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: google-api-python-client is required. Install with: pip install google-api-python-client")
        return 0

    config_path = Path(__file__).parent.parent / "config" / "sheets_config.json"
    if not config_path.exists():
        print("ERROR: config/sheets_config.json not found — cannot locate the Sheets ID.")
        return 0

    with open(config_path) as f:
        sheets_config = json.load(f)

    sheet_id = sheets_config.get("venue_events_sheet_id")
    if not sheet_id:
        print("ERROR: venue_events_sheet_id not in sheets_config.json")
        return 0

    creds = get_credentials()
    if not creds:
        print("ERROR: Not authenticated with Google. Run: python -m utils.google_auth")
        return 0

    service = build("sheets", "v4", credentials=creds)

    print(f"  Reading from sheet {sheet_id} ...")
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:U",
        ).execute()
    except Exception as exc:
        print(f"ERROR reading sheet: {exc}")
        return 0

    rows = result.get("values", [])
    if len(rows) <= 1:
        print("  No events found in sheet.")
        return 0

    header = rows[0]
    raw_events = []
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        raw_events.append(dict(zip(header, row)))

    print(f"  Found {len(raw_events)} events in Sheets.")

    if dry_run:
        print(f"  DRY RUN — would write {len(raw_events)} events to Firestore.")
        return len(raw_events)

    # --- Write to Firestore ---
    from venue_scout.venue_events_sheet import (
        normalize_event,
        _event_doc_id,
        _serialize_event_for_db,
        _EVENTS_COLLECTION,
        _FIRESTORE_BATCH_SIZE,
    )
    from venue_scout.firestore_client import get_db
    from zoneinfo import ZoneInfo

    db = get_db()
    col = db.collection(_EVENTS_COLLECTION)

    # Normalise and parse events
    parsed = []
    for raw in raw_events:
        event = normalize_event(raw)
        # Parse datetime field
        dt_str = raw.get("datetime", "")
        if dt_str and dt_str not in ("None", ""):
            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                event["datetime"] = dt
            except ValueError:
                event["datetime"] = None
        else:
            event["datetime"] = None
        parsed.append(event)

    # Batch write in chunks
    total = 0
    for start in range(0, len(parsed), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        chunk = parsed[start:start + _FIRESTORE_BATCH_SIZE]
        for event in chunk:
            doc_id = _event_doc_id(event)
            batch.set(col.document(doc_id), _serialize_event_for_db(event))
        batch.commit()
        total += len(chunk)
        print(f"  Wrote {total}/{len(parsed)} events...", end="\r")
        time.sleep(0.1)  # light throttle

    print(f"\n  Done. Migrated {total} events to Firestore.")
    return total


def _migrate_venues(dry_run: bool = False) -> int:
    """Read all venues from Google Sheets and write them to Firestore.

    Returns the number of venues migrated.
    """
    print("\n=== Migrating Venues (Sheets → Firestore) ===")

    try:
        from utils.google_auth import get_credentials
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: google-api-python-client is required.")
        return 0

    config_path = Path(__file__).parent.parent / "config" / "sheets_config.json"
    if not config_path.exists():
        print("ERROR: config/sheets_config.json not found.")
        return 0

    with open(config_path) as f:
        sheets_config = json.load(f)

    sheet_id = sheets_config.get("venues_sheet_id")
    if not sheet_id:
        print("ERROR: venues_sheet_id not in sheets_config.json")
        return 0

    creds = get_credentials()
    if not creds:
        print("ERROR: Not authenticated with Google. Run: python -m utils.google_auth")
        return 0

    service = build("sheets", "v4", credentials=creds)

    print(f"  Reading from sheet {sheet_id} ...")
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A:W",
        ).execute()
    except Exception as exc:
        print(f"ERROR reading sheet: {exc}")
        return 0

    rows = result.get("values", [])
    if len(rows) <= 1:
        print("  No venues found in sheet.")
        return 0

    header = rows[0]
    raw_venues = []
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        raw_venues.append(dict(zip(header, row)))

    print(f"  Found {len(raw_venues)} venues in Sheets.")

    if dry_run:
        print(f"  DRY RUN — would write {len(raw_venues)} venues to Firestore.")
        return len(raw_venues)

    # --- Write to Firestore ---
    from venue_scout.cache import (
        _venue_doc_id,
        _serialize_venue_for_db,
        _deserialize_venue_from_db,
        _VENUES_COLLECTION,
        _FIRESTORE_BATCH_SIZE,
    )
    from venue_scout.firestore_client import get_db

    db = get_db()
    col = db.collection(_VENUES_COLLECTION)

    total = 0
    for start in range(0, len(raw_venues), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        chunk = raw_venues[start:start + _FIRESTORE_BATCH_SIZE]
        for raw in chunk:
            venue = _deserialize_venue_from_db(raw)
            doc_id = _venue_doc_id(venue.get("name", ""))
            batch.set(col.document(doc_id), _serialize_venue_for_db(venue))
        batch.commit()
        total += len(chunk)
        print(f"  Wrote {total}/{len(raw_venues)} venues...", end="\r")
        time.sleep(0.1)

    print(f"\n  Done. Migrated {total} venues to Firestore.")
    return total


def _migrate_archive(dry_run: bool = False) -> int:
    """Read all archived events from the Archive tab and write them to Firestore.

    Returns the number of archived events migrated.
    """
    print("\n=== Migrating Archive (Sheets → Firestore) ===")

    try:
        from utils.google_auth import get_credentials
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: google-api-python-client is required.")
        return 0

    config_path = Path(__file__).parent.parent / "config" / "sheets_config.json"
    if not config_path.exists():
        print("ERROR: config/sheets_config.json not found.")
        return 0

    with open(config_path) as f:
        sheets_config = json.load(f)

    sheet_id = sheets_config.get("venue_events_sheet_id")
    if not sheet_id:
        print("ERROR: venue_events_sheet_id not in sheets_config.json")
        return 0

    creds = get_credentials()
    if not creds:
        print("ERROR: Not authenticated with Google.")
        return 0

    service = build("sheets", "v4", credentials=creds)

    print(f"  Reading Archive tab from sheet {sheet_id} ...")
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="'Archive'!A:U",
        ).execute()
    except Exception as exc:
        print(f"ERROR reading Archive tab: {exc}")
        return 0

    rows = result.get("values", [])
    if len(rows) <= 1:
        print("  No archived events found.")
        return 0

    header = rows[0]
    raw_events = []
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        raw_events.append(dict(zip(header, row)))

    print(f"  Found {len(raw_events)} archived events in Sheets.")

    if dry_run:
        print(f"  DRY RUN — would write {len(raw_events)} archived events to Firestore.")
        return len(raw_events)

    # --- Write to Firestore ---
    from venue_scout.venue_events_sheet import (
        normalize_event,
        _event_doc_id,
        _serialize_event_for_db,
        _ARCHIVE_COLLECTION,
        _FIRESTORE_BATCH_SIZE,
    )
    from venue_scout.firestore_client import get_db

    db = get_db()
    col = db.collection(_ARCHIVE_COLLECTION)

    parsed = []
    for raw in raw_events:
        event = normalize_event(raw)
        dt_str = raw.get("datetime", "")
        if dt_str and dt_str not in ("None", ""):
            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                event["datetime"] = dt
            except ValueError:
                event["datetime"] = None
        else:
            event["datetime"] = None
        parsed.append(event)

    total = 0
    for start in range(0, len(parsed), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        chunk = parsed[start:start + _FIRESTORE_BATCH_SIZE]
        for event in chunk:
            doc_id = _event_doc_id(event)
            batch.set(col.document(doc_id), _serialize_event_for_db(event))
        batch.commit()
        total += len(chunk)
        print(f"  Wrote {total}/{len(parsed)} archived events...", end="\r")
        time.sleep(0.1)

    print(f"\n  Done. Migrated {total} archived events to Firestore.")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Sheets data to Firestore (one-time migration)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing.")
    parser.add_argument("--skip-events", action="store_true", help="Skip event migration.")
    parser.add_argument("--skip-venues", action="store_true", help="Skip venue migration.")
    parser.add_argument("--skip-archive", action="store_true", help="Skip archive migration.")
    args = parser.parse_args()

    start = time.time()
    print("=== Firestore Migration ===")
    print(f"Started: {datetime.now(ZoneInfo('America/New_York')).isoformat()}")
    if args.dry_run:
        print("MODE: DRY RUN (no data will be written)")
    print()

    event_count = 0
    venue_count = 0
    archive_count = 0

    if not args.skip_events:
        event_count = _migrate_events(dry_run=args.dry_run)

    if not args.skip_venues:
        venue_count = _migrate_venues(dry_run=args.dry_run)

    if not args.skip_archive:
        archive_count = _migrate_archive(dry_run=args.dry_run)

    elapsed = time.time() - start
    print(f"\n=== Migration complete in {elapsed:.1f}s ===")
    print(f"  Events migrated: {event_count}")
    print(f"  Venues migrated: {venue_count}")
    print(f"  Archived events migrated: {archive_count}")


if __name__ == "__main__":
    main()
