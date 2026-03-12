"""Shared Firestore client for Venue Scout.

Provides a singleton Firestore client used by venue_events_sheet.py and cache.py.

Credential resolution order:
  1. config/service_account.json (local dev) — if file exists and
     GOOGLE_APPLICATION_CREDENTIALS is not already set
  2. GOOGLE_APPLICATION_CREDENTIALS env var (explicit override)
  3. Application Default Credentials (Cloud Run, gcloud auth)
"""

from __future__ import annotations

import os
from pathlib import Path

_GCP_PROJECT = "gen-lang-client-0046008897"
_FIRESTORE_DATABASE = "whaddupnyc"
_SERVICE_ACCOUNT_PATH = Path(__file__).parent.parent / "config" / "service_account.json"

_client = None


def get_db():
    """Return the singleton Firestore client, initialising on first call."""
    global _client
    if _client is not None:
        return _client

    from google.cloud import firestore

    # Use the local service-account file when it exists and no explicit
    # GOOGLE_APPLICATION_CREDENTIALS is set — avoids needing `gcloud auth`
    # for local development.
    if _SERVICE_ACCOUNT_PATH.exists() and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            str(_SERVICE_ACCOUNT_PATH),
            scopes=["https://www.googleapis.com/auth/datastore"],
        )
        _client = firestore.Client(project=_GCP_PROJECT, database=_FIRESTORE_DATABASE, credentials=creds)
    else:
        # Cloud Run (attached service account) or explicit ADC
        _client = firestore.Client(project=_GCP_PROJECT, database=_FIRESTORE_DATABASE)

    return _client


def reset_client():
    """Reset the singleton — useful in tests or after credential rotation."""
    global _client
    _client = None
