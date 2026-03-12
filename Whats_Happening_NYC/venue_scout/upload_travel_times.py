#!/usr/bin/env python3
"""One-time script to upload pre-computed subway travel times to Firestore.

Usage:
    python -m venue_scout.upload_travel_times [--dry-run] [--skip-travel-times] [--skip-venue-mapping]
    python -m venue_scout.upload_travel_times --fix-lines-only [--dry-run]

Source data:
    Parquet: /Users/ddiciurcio/TRMNL/route_planner/outputs/matrix_20260307_151829.parquet
    Origins: /Users/ddiciurcio/TRMNL/route_planner/data/processed/origins.csv
    Destinations: /Users/ddiciurcio/TRMNL/route_planner/data/processed/destinations.csv
    GTFS: /Users/ddiciurcio/TRMNL/route_planner/data/otp/gtfs_subway/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROUTE_PLANNER_DIR = Path("/Users/ddiciurcio/TRMNL/route_planner")
PARQUET_FILE = ROUTE_PLANNER_DIR / "outputs" / "matrix_20260307_151829.parquet"
ORIGINS_CSV = ROUTE_PLANNER_DIR / "data" / "processed" / "origins.csv"
DESTINATIONS_CSV = ROUTE_PLANNER_DIR / "data" / "processed" / "destinations.csv"
GTFS_DIR = ROUTE_PLANNER_DIR / "data" / "otp" / "gtfs_subway"

FIRESTORE_COLLECTION = "subway_travel_times"

# Subway lines in display order (shown in UI)
LINE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "E", "F", "G", "J", "L", "M", "N", "Q", "R", "S", "H"]
_LINE_ORDER_SET = set(LINE_ORDER)

# Normalize GTFS route_ids that are variants of a base line (e.g. "6X" -> "6")
_ROUTE_NORMALIZE: dict[str, str] = {
    "6X": "6",
    "FX": "F",
    "GS": "S",   # Grand Central Shuttle
    "FS": "S",   # Franklin Ave Shuttle
    "H": "H",    # Rockaway Park Shuttle — keep as-is
}


def _normalize_route(route_id: str) -> str | None:
    """Return canonical line label for a GTFS route_id, or None to discard."""
    if route_id in _ROUTE_NORMALIZE:
        return _ROUTE_NORMALIZE[route_id]
    if route_id in _LINE_ORDER_SET:
        return route_id
    # Strip trailing X (express variant) and retry
    if route_id.endswith("X") and route_id[:-1] in _LINE_ORDER_SET:
        return route_id[:-1]
    return None  # unknown/discard


def _load_gtfs_station_lines(gtfs_dir: Path) -> dict[str, list[str]]:
    """
    Build {origin_id: sorted_list_of_lines} using MTA GTFS data.

    origin_ids are GTFS parent_station IDs (location_type=1).
    """
    # child stop_id -> parent_station_id
    parent: dict[str, str] = {}
    with open(gtfs_dir / "stops.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ps = row.get("parent_station", "").strip()
            if ps:
                parent[row["stop_id"].strip()] = ps

    # trip_id -> normalized route label
    trip_route: dict[str, str] = {}
    with open(gtfs_dir / "trips.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            line = _normalize_route(row["route_id"].strip())
            if line:
                trip_route[row["trip_id"].strip()] = line

    # Build parent_station_id -> set of lines
    station_lines: dict[str, set[str]] = {}
    with open(gtfs_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stop_id = row["stop_id"].strip()
            trip_id = row["trip_id"].strip()
            ps = parent.get(stop_id)
            line = trip_route.get(trip_id)
            if ps and line:
                station_lines.setdefault(ps, set()).add(line)

    # Sort each station's lines in display order
    return {
        sid: sorted(lines, key=lambda l: LINE_ORDER.index(l) if l in LINE_ORDER else 99)
        for sid, lines in station_lines.items()
    }


def _load_origins(path: Path) -> dict[str, dict]:
    """Load origins.csv → {origin_id: {station_name, lat, lng}}."""
    origins: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # column may be "origin_id" or "id"
            origin_id = str(row.get("origin_id", row.get("id", ""))).strip()
            if not origin_id:
                continue
            # name column may be "origin_name", "name", or "station_name"
            name = str(
                row.get("origin_name", row.get("name", row.get("station_name", "")))
            ).strip()
            origins[origin_id] = {
                "station_name": name,
                "lat": float(row.get("lat", 0) or 0),
                "lng": float(row.get("lng", row.get("lon", 0)) or 0),
            }
    return origins


def _load_destinations(path: Path) -> dict[str, str]:
    """Load destinations.csv → {dest_id: venue_name}."""
    dests: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # column may be "venue_id", "id", or "destination_id"
            dest_id = str(
                row.get("venue_id", row.get("id", row.get("destination_id", "")))
            ).strip()
            # name column may be "venue_name" or "name"
            name = str(row.get("venue_name", row.get("name", ""))).strip()
            if dest_id:
                dests[dest_id] = name
    return dests


def _build_station_docs(
    parquet_path: Path,
    origins: dict[str, dict],
    station_lines_map: dict[str, list[str]],
) -> dict[str, dict]:
    """
    Read parquet and build per-station document dicts.

    Returns {origin_id: {station_name, lat, lng, lines: [...], times: {dest_id: {w?, s?}}}}
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pydict()

    n_rows = len(df["origin_id"])
    print(f"  Parquet loaded: {n_rows:,} rows")

    station_docs: dict[str, dict] = {}

    # Pre-initialise docs for all known origins
    for origin_id, meta in origins.items():
        lines = station_lines_map.get(origin_id, [])
        station_docs[origin_id] = {
            "station_name": meta["station_name"],
            "lat": meta["lat"],
            "lng": meta["lng"],
            "lines": lines,
            "times": {},
        }

    # Pre-extract columns once to avoid repeated dict.get() with expensive defaults
    _none_list: list = [None] * n_rows
    origin_ids_col = df["origin_id"]
    dest_ids_col = df["destination_id"]
    walk_status_col = df.get("walk_status", _none_list)
    subway_status_col = df.get("subway_status", _none_list)
    walk_seconds_col = df.get("walk_seconds", _none_list)
    subway_seconds_col = df.get("subway_seconds", _none_list)

    skipped_no_origin = 0

    for i in range(n_rows):
        walk_status = str(walk_status_col[i] or "").lower()
        subway_status = str(subway_status_col[i] or "").lower()

        walk_ok = walk_status == "ok"
        subway_ok = subway_status == "ok"

        if not walk_ok and not subway_ok:
            continue  # nothing useful to store

        origin_id = str(origin_ids_col[i]).strip()
        dest_id = str(dest_ids_col[i]).strip()

        if origin_id not in station_docs:
            skipped_no_origin += 1
            continue

        entry: dict[str, int] = {}
        if walk_ok:
            walk_secs = walk_seconds_col[i]
            if walk_secs is not None:
                entry["w"] = max(1, round(float(walk_secs) / 60))
        if subway_ok:
            subway_secs = subway_seconds_col[i]
            if subway_secs is not None:
                entry["s"] = max(1, round(float(subway_secs) / 60))

        if entry:
            station_docs[origin_id]["times"][dest_id] = entry

    if skipped_no_origin:
        print(f"  Warning: {skipped_no_origin} rows had origin_id not in origins.csv")

    return station_docs


def _upload_station_docs(
    station_docs: dict[str, dict],
    dry_run: bool,
) -> None:
    """Batch-upload 496 station docs to subway_travel_times collection."""
    if dry_run:
        # Summarise without writing
        total_times = sum(len(doc["times"]) for doc in station_docs.values())
        print(f"  [DRY RUN] Would upload {len(station_docs)} station docs "
              f"with {total_times:,} total time entries")
        # Show a sample
        sample_id = next(iter(station_docs))
        sample = station_docs[sample_id]
        print(f"  Sample doc ({sample_id}): station_name={sample['station_name']!r}, "
              f"line={sample['line']!r}, times_count={len(sample['times'])}")
        return

    from venue_scout.firestore_client import get_db
    from google.cloud.firestore_v1 import WriteBatch

    db = get_db()
    # Each station doc is ~60KB (1,400 entries × ~45 bytes), so keep batches
    # small enough to stay under Firestore's 10MB batch request limit.
    BATCH_SIZE = 10
    batch: WriteBatch = db.batch()
    batch_count = 0
    total_written = 0

    for origin_id, doc in station_docs.items():
        ref = db.collection(FIRESTORE_COLLECTION).document(origin_id)
        batch.set(ref, doc)
        batch_count += 1
        total_written += 1

        if batch_count >= BATCH_SIZE:
            batch.commit()
            print(f"  Committed batch ({total_written} docs so far)…")
            batch = db.batch()
            batch_count = 0

    if batch_count:
        batch.commit()

    print(f"  Uploaded {total_written} station docs to '{FIRESTORE_COLLECTION}'")


def _map_venues(
    destinations: dict[str, str],
    dry_run: bool,
) -> None:
    """
    Match destination IDs to Firestore venue docs and write route_planner_id.

    - tm_* → match by ticketmaster_venue_id
    - v_*  → match by venue name (case-insensitive)
    """
    from venue_scout.firestore_client import get_db
    db = get_db()

    # Stream all venues
    venues_ref = db.collection("venues")
    venue_docs = list(venues_ref.stream())
    print(f"  Loaded {len(venue_docs)} venue docs from Firestore")

    # Build lookup indexes
    tm_index: dict[str, Any] = {}   # ticketmaster_id -> doc_ref
    name_index: dict[str, Any] = {}  # normalized_name -> doc_ref

    for vdoc in venue_docs:
        data = vdoc.to_dict() or {}
        tm_id = str(data.get("ticketmaster_venue_id", "") or "").strip()
        if tm_id:
            tm_index[tm_id] = vdoc.reference

        vname = str(data.get("name", "") or "").strip().lower()
        if vname and vname not in name_index:
            name_index[vname] = vdoc.reference

    matched = 0
    unmatched = 0

    for dest_id, venue_name in destinations.items():
        ref = None
        if dest_id.startswith("tm_"):
            tm_id = dest_id[3:]
            ref = tm_index.get(tm_id)
        elif dest_id.startswith("v_"):
            vname_lower = venue_name.lower()
            ref = name_index.get(vname_lower)

        if ref is None:
            unmatched += 1
            continue

        matched += 1
        if not dry_run:
            ref.update({"route_planner_id": dest_id})

    action = "[DRY RUN] Would write" if dry_run else "Wrote"
    print(f"  {action} route_planner_id to {matched} venue docs "
          f"({unmatched} unmatched out of {len(destinations)} destinations)")


def _fix_lines_only(dry_run: bool) -> None:
    """
    Fast targeted update: re-read GTFS and patch just the `lines` field on
    existing subway_travel_times docs without re-uploading the full times maps.
    """
    from venue_scout.firestore_client import get_db

    print("Loading GTFS station→lines mapping…")
    station_lines_map = _load_gtfs_station_lines(GTFS_DIR)
    print(f"  {len(station_lines_map)} stations mapped")

    db = get_db()
    BATCH_SIZE = 100  # only updating one small field — larger batches are fine
    batch = db.batch()
    batch_count = 0
    total = 0
    no_gtfs = 0

    for origin_id, lines in station_lines_map.items():
        ref = db.collection(FIRESTORE_COLLECTION).document(origin_id)
        if not dry_run:
            batch.update(ref, {"lines": lines})
        batch_count += 1
        total += 1

        if batch_count >= BATCH_SIZE:
            if not dry_run:
                batch.commit()
            print(f"  Patched {total} docs so far…")
            batch = db.batch()
            batch_count = 0

    if batch_count and not dry_run:
        batch.commit()

    action = "[DRY RUN] Would patch" if dry_run else "Patched"
    print(f"  {action} {total} station docs with corrected lines ({no_gtfs} had no GTFS match)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload subway travel times to Firestore")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--skip-travel-times", action="store_true", help="Skip uploading station travel time docs")
    parser.add_argument("--skip-venue-mapping", action="store_true", help="Skip writing route_planner_id to venue docs")
    parser.add_argument("--fix-lines-only", action="store_true",
                        help="Fast path: only update the 'lines' field on existing docs using GTFS data")
    args = parser.parse_args()

    # Fast path: just fix the lines field on existing docs
    if args.fix_lines_only:
        if not GTFS_DIR.exists():
            print(f"ERROR: GTFS directory not found: {GTFS_DIR}", file=sys.stderr)
            sys.exit(1)
        print("=== Fix Lines Only ===")
        if args.dry_run:
            print("Mode: DRY RUN (no writes)\n")
        _fix_lines_only(dry_run=args.dry_run)
        print("\nDone.")
        return

    # Validate source files exist
    for path in (PARQUET_FILE, ORIGINS_CSV, DESTINATIONS_CSV):
        if not path.exists():
            print(f"ERROR: Missing source file: {path}", file=sys.stderr)
            sys.exit(1)

    print("=== Subway Travel Times Upload ===")
    if args.dry_run:
        print("Mode: DRY RUN (no writes)\n")

    # Step 1: Load metadata
    print("Loading origins.csv…")
    origins = _load_origins(ORIGINS_CSV)
    print(f"  {len(origins)} stations loaded")

    print("Loading GTFS station→lines mapping…")
    station_lines_map = _load_gtfs_station_lines(GTFS_DIR) if GTFS_DIR.exists() else {}
    print(f"  {len(station_lines_map)} stations mapped from GTFS")

    print("Loading destinations.csv…")
    destinations = _load_destinations(DESTINATIONS_CSV)
    print(f"  {len(destinations)} destinations loaded")

    # Step 2: Upload station travel time docs
    if not args.skip_travel_times:
        print("\nBuilding station docs from parquet…")
        station_docs = _build_station_docs(PARQUET_FILE, origins, station_lines_map)
        print(f"  {len(station_docs)} station docs built")

        print("\nUploading to Firestore…")
        _upload_station_docs(station_docs, dry_run=args.dry_run)

    # Step 3: Map destinations to venue docs
    if not args.skip_venue_mapping:
        print("\nMapping destinations to venue docs…")
        _map_venues(destinations, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
