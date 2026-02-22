"""Shared Google Sheets helpers used across modules."""

from __future__ import annotations

import json
from pathlib import Path

from googleapiclient.discovery import build

from .google_auth import get_credentials

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"
_DEFAULT_SHEETS_CONFIG = _DEFAULT_CONFIG_DIR / "sheets_config.json"


def get_sheets_service():
    """Return authenticated Google Sheets service or None."""
    creds = get_credentials()
    if not creds:
        return None
    return build("sheets", "v4", credentials=creds)


def load_sheets_config(config_path: Path | None = None) -> dict:
    """Load sheet IDs from config JSON."""
    path = config_path or _DEFAULT_SHEETS_CONFIG
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_sheets_config(
    config: dict,
    config_path: Path | None = None,
    *,
    ensure_dir: bool = False,
):
    """Persist sheet IDs to config JSON."""
    path = config_path or _DEFAULT_SHEETS_CONFIG
    if ensure_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def create_spreadsheet(title: str, service=None) -> str | None:
    """Create a new Google Spreadsheet and return its ID."""
    sheets_service = service or get_sheets_service()
    if not sheets_service:
        return None

    result = sheets_service.spreadsheets().create(
        body={"properties": {"title": title}}
    ).execute()
    return result.get("spreadsheetId")


def write_sheet_header(sheet_id: str, columns: list[str], service=None):
    """Write header row to a sheet."""
    sheets_service = service or get_sheets_service()
    if not sheets_service:
        return

    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range="A1",
        valueInputOption="RAW",
        body={"values": [columns]},
    ).execute()
