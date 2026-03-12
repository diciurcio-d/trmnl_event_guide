"""GCS sync for FAISS semantic index files.

The nightly Cloud Run job rebuilds the semantic index and pushes it to GCS.
The web server Cloud Run service pulls it down on startup so it always has
the latest index without baking it into the Docker image.

Both containers also pull before any index operation so incremental rebuilds
can diff against the most recent index.

GCS paths:
    gs://whats-happening-nyc-data/semantic/semantic_events.faiss
    gs://whats-happening-nyc-data/semantic/semantic_events_metadata.json
"""

from __future__ import annotations

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
