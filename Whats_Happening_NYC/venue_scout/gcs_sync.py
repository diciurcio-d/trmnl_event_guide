"""GCS sync for FAISS semantic index files and the events cache snapshot.

The nightly Cloud Run job rebuilds the semantic index and pushes it to GCS.
The web server Cloud Run service pulls it down on startup so it always has
the latest index without baking it into the Docker image.

Both containers also pull before any index operation so incremental rebuilds
can diff against the most recent index.

GCS paths:
    gs://whats-happening-nyc-data/semantic/semantic_events.faiss
    gs://whats-happening-nyc-data/semantic/semantic_events_metadata.json
    gs://whats-happening-nyc-data/events/events_cache.json.gz
    gs://whats-happening-nyc-data/events/travel_times_cache.json.gz
"""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path

from venue_scout.paths import (
    SEMANTIC_EVENTS_INDEX_FILE,
    SEMANTIC_EVENTS_METADATA_FILE,
    ensure_data_dir,
)

_GCS_BUCKET = os.environ.get("GCS_BUCKET", "whats-happening-nyc-data")
_INDEX_BLOB = "semantic/semantic_events.faiss"
_META_BLOB = "semantic/semantic_events_metadata.json"
_EVENTS_CACHE_BLOB = "events/events_cache.json.gz"
_TRAVEL_TIMES_CACHE_BLOB = "events/travel_times_cache.json.gz"

_FILES = [
    (SEMANTIC_EVENTS_INDEX_FILE, _INDEX_BLOB),
    (SEMANTIC_EVENTS_METADATA_FILE, _META_BLOB),
]


def _get_client():
    from google.cloud import storage
    return storage.Client()


def push_index(verbose: bool = True) -> None:
    """Upload local FAISS index + metadata to GCS.

    Silently skips any file that doesn't exist locally (e.g. first run).
    Raises on upload failure so the caller can decide whether to abort.
    """
    client = _get_client()
    bucket = client.bucket(_GCS_BUCKET)
    for local, blob_name in _FILES:
        if not local.exists():
            if verbose:
                print(f"  GCS push: {local.name} not found locally, skipping.")
            continue
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local))
        if verbose:
            size_kb = local.stat().st_size // 1024
            print(f"  Pushed {local.name} ({size_kb} KB) → gs://{_GCS_BUCKET}/{blob_name}")


def pull_index(verbose: bool = True) -> bool:
    """Download FAISS index + metadata from GCS when GCS version is newer.

    Returns True if any file was downloaded, False if already up to date.
    Returns False (without raising) if the blobs don't exist yet (first run).
    """
    ensure_data_dir()
    client = _get_client()
    bucket = client.bucket(_GCS_BUCKET)
    downloaded = False

    for local, blob_name in _FILES:
        blob = bucket.blob(blob_name)
        try:
            blob.reload()
        except Exception:
            # Blob doesn't exist in GCS yet — skip silently (first run).
            continue

        gcs_mtime = blob.updated.timestamp() if blob.updated else 0
        local_mtime = local.stat().st_mtime if local.exists() else 0

        if gcs_mtime <= local_mtime:
            if verbose:
                print(f"  GCS pull: {local.name} already up to date.")
            continue

        blob.download_to_filename(str(local))
        downloaded = True
        if verbose:
            size_kb = local.stat().st_size // 1024
            print(f"  Pulled gs://{_GCS_BUCKET}/{blob_name} → {local.name} ({size_kb} KB)")

    return downloaded


def push_travel_times_cache(data: dict[str, dict], verbose: bool = True) -> None:
    """Serialize all station travel times and upload to GCS.

    data is shaped {station_id: {route_planner_id: {w: int, s: int}}}.
    Called after uploading travel times to Firestore so the web server can
    load them from GCS at startup instead of streaming 496 Firestore docs.
    """
    payload = json.dumps(data).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=6)
    client = _get_client()
    bucket = client.bucket(_GCS_BUCKET)
    blob = bucket.blob(_TRAVEL_TIMES_CACHE_BLOB)
    blob.upload_from_file(
        io.BytesIO(compressed),
        content_type="application/json",
        size=len(compressed),
    )
    if verbose:
        size_kb = len(compressed) // 1024
        print(f"  Pushed {len(data):,} stations ({size_kb} KB compressed) → gs://{_GCS_BUCKET}/{_TRAVEL_TIMES_CACHE_BLOB}")


def pull_travel_times_cache(verbose: bool = True) -> dict[str, dict] | None:
    """Download and decompress the travel times cache from GCS.

    Returns {station_id: times_map} or None if the blob doesn't exist yet.
    """
    client = _get_client()
    bucket = client.bucket(_GCS_BUCKET)
    blob = bucket.blob(_TRAVEL_TIMES_CACHE_BLOB)
    try:
        blob.reload()
    except Exception:
        if verbose:
            print("  GCS travel times cache not found — will fall back to Firestore.")
        return None

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    data = json.loads(gzip.decompress(buf.read()).decode("utf-8"))
    if verbose:
        size_kb = buf.tell() // 1024
        print(f"  Pulled travel times for {len(data):,} stations from gs://{_GCS_BUCKET}/{_TRAVEL_TIMES_CACHE_BLOB}")
    return data


def push_events_cache(events: list[dict], verbose: bool = True) -> None:
    """Serialize events to gzipped JSON and upload to GCS.

    Called by the nightly job after writing events to Firestore so the web
    server can load them from GCS on startup instead of streaming Firestore.
    """
    payload = json.dumps(events, default=str).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=6)
    client = _get_client()
    bucket = client.bucket(_GCS_BUCKET)
    blob = bucket.blob(_EVENTS_CACHE_BLOB)
    blob.upload_from_file(
        io.BytesIO(compressed),
        content_type="application/json",
        size=len(compressed),
    )
    if verbose:
        size_kb = len(compressed) // 1024
        print(f"  Pushed {len(events):,} events ({size_kb} KB compressed) → gs://{_GCS_BUCKET}/{_EVENTS_CACHE_BLOB}")


def pull_events_cache(verbose: bool = True) -> list[dict] | None:
    """Download and decompress the events cache from GCS.

    Returns the list of event dicts, or None if the blob doesn't exist yet
    (first run before the nightly job has pushed anything).
    """
    client = _get_client()
    bucket = client.bucket(_GCS_BUCKET)
    blob = bucket.blob(_EVENTS_CACHE_BLOB)
    try:
        blob.reload()
    except Exception:
        if verbose:
            print("  GCS events cache not found — will fall back to Firestore.")
        return None

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    payload = gzip.decompress(buf.read()).decode("utf-8")
    events = json.loads(payload)
    if verbose:
        size_kb = buf.tell() // 1024
        print(f"  Pulled {len(events):,} events from gs://{_GCS_BUCKET}/{_EVENTS_CACHE_BLOB}")
    return events
