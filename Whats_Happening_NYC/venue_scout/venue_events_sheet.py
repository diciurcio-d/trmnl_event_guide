"""Google Sheets storage for venue events."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.google_auth import get_credentials, is_authenticated


_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"
_MAX_EVENT_DAYS_AHEAD = 365
_ARCHIVE_TAB = "Archive"


def _safe_lower(val) -> str:
    """Safely convert value to lowercase string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.lower()
    return str(val).lower()


VENUE_EVENTS_COLUMNS = [
    "name",
    "datetime",
    "date_str",
    "end_date",
    "venue_name",
    "address",
    "event_type",
    "url",
    "source",
    "matched_artist",
    "travel_minutes",
    "description",
    "event_source_url",
    "extraction_method",
    "relevance_score",
    "validation_confidence",
    "date_added",
    "in_semantic_index",
    "semantic_indexed_at",
]


def normalize_event(event: dict) -> dict:
    """Normalize a venue event into the canonical schema."""
    normalized = dict(event)
    normalized["name"] = normalized.get("name", "")
    normalized["datetime"] = normalized.get("datetime")
    normalized["date_str"] = normalized.get("date_str", "")
    normalized["end_date"] = normalized.get("end_date", "")
    normalized["venue_name"] = normalized.get("venue_name", "")
    normalized["address"] = normalized.get("address", "")
    normalized["event_type"] = _normalize_event_category(normalized.get("event_type", ""))
    normalized["url"] = normalized.get("url", "")
    normalized["source"] = normalized.get("source", "")
    normalized["matched_artist"] = normalized.get("matched_artist", "")
    normalized["travel_minutes"] = normalized.get("travel_minutes")
    normalized["description"] = normalized.get("description", "")
    normalized["event_source_url"] = normalized.get("event_source_url", "")
    normalized["extraction_method"] = normalized.get("extraction_method", "")
    normalized["relevance_score"] = normalized.get("relevance_score")
    normalized["validation_confidence"] = normalized.get("validation_confidence")
    normalized["date_added"] = normalized.get("date_added", "")
    normalized["in_semantic_index"] = normalized.get("in_semantic_index", False)
    normalized["semantic_indexed_at"] = normalized.get("semantic_indexed_at", "")
    return normalized


def _sheet_col_label(col_index: int) -> str:
    """1-based column index to A1 label."""
    if col_index < 1:
        return "A"

    out = ""
    value = col_index
    while value:
        value, rem = divmod(value - 1, 26)
        out = chr(65 + rem) + out
    return out


def _sheet_full_range() -> str:
    return f"A:{_sheet_col_label(len(VENUE_EVENTS_COLUMNS))}"


def _normalize_text(value) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_event_category(value) -> str:
    """Canonicalize event category labels to avoid case/syntax duplicates."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("&amp;", "&").replace("&#038;", "&")
    text = re.sub(r"\s*([/|,])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text.lower()


def _safe_date_token(event: dict) -> str:
    dt = event.get("datetime")
    if isinstance(dt, datetime):
        return dt.date().isoformat()

    raw = str(event.get("date_str", "") or "").strip()
    if not raw:
        return ""

    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


def _parse_date_str(raw: str):
    """Parse a YYYY-MM-DD or MM/DD/YYYY string to a date, or None."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _event_date_for_window(event: dict):
    """Return the date to use for the keep/drop window check.

    For ongoing events (exhibitions, festivals) that have an end_date,
    we keep the event as long as its end_date hasn't passed — even if
    the start date is already in the past.  For single-day events we
    use the start date as before.
    """
    # Try end_date first for ongoing-event window logic
    end_raw = str(event.get("end_date", "") or "").strip()
    if end_raw:
        end_parsed = _parse_date_str(end_raw)
        if end_parsed:
            return end_parsed

    # Fall back to start datetime / date_str
    dt = event.get("datetime")
    if isinstance(dt, datetime):
        return dt.date()

    raw = str(event.get("date_str", "") or "").strip()
    if not raw:
        return None
    return _parse_date_str(raw)


def _dedupe_canonical_link(event: dict) -> str:
    url = _normalize_text(event.get("url", ""))
    if url:
        return url
    return _normalize_text(event.get("event_source_url", ""))


def _event_dedupe_key(event: dict) -> tuple[str, str, str, str]:
    """
    Cross-venue dedupe key.

    If URL/source URL exists, dedupe across venues by (name, date, source, link).
    If no link exists, keep venue in key to avoid over-collapsing distinct no-link events.
    """
    name = _normalize_text(event.get("name", ""))
    date_token = _safe_date_token(event)
    source = _normalize_text(event.get("source", ""))
    link = _dedupe_canonical_link(event)
    if link:
        return name, date_token, source, f"link:{link}"
    venue = _normalize_text(event.get("venue_name", ""))
    return name, date_token, source, f"venue:{venue}"


def _event_quality_score(event: dict) -> float:
    score = 0.0
    if str(event.get("url", "") or "").strip():
        score += 3.0
    if str(event.get("event_source_url", "") or "").strip():
        score += 1.0
    if str(event.get("description", "") or "").strip():
        score += min(2.0, len(str(event.get("description", ""))) / 120.0)
    if str(event.get("address", "") or "").strip():
        score += 1.0
    if str(event.get("venue_name", "") or "").strip():
        score += 0.5
    return score


def _dedupe_events(events: list[dict]) -> tuple[list[dict], int]:
    """Collapse duplicate events across venues with deterministic quality tie-break."""
    best_by_key: dict[tuple[str, str, str, str], dict] = {}
    removed = 0

    for event in events:
        key = _event_dedupe_key(event)
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = event
            continue
        removed += 1
        if _event_quality_score(event) > _event_quality_score(existing):
            best_by_key[key] = event

    return list(best_by_key.values()), removed


def _normalize_venue_name(name: str) -> str:
    """Normalize venue name for cache lookups."""
    text = str(name or "").lower().strip()
    text = re.sub(r'^the\s+', '', text)
    text = re.sub(r'^[(]', '', text)
    text = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _load_venue_address_lookup() -> dict[str, str]:
    """
    Build a normalized venue-name -> address lookup from Venue Scout Cache.

    Returns empty mapping if cache read fails so event writes still proceed.
    """
    try:
        from venue_scout.cache import read_cached_venues
    except Exception:
        return {}

    try:
        venues = read_cached_venues()
    except Exception:
        return {}

    lookup: dict[str, str] = {}
    for venue in venues:
        norm_name = _normalize_venue_name(venue.get("name", ""))
        address = str(venue.get("address", "") or "").strip()
        if norm_name and address and norm_name not in lookup:
            lookup[norm_name] = address
    return lookup


def _normalize_url_host_path(raw_url: str) -> tuple[str, str]:
    """Normalize URL into (host, path) for matching."""
    raw = str(raw_url or "").strip()
    if not raw:
        return "", ""
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return "", ""
    host = (parsed.netloc or "").lower().replace("www.", "")
    path = (parsed.path or "/").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    path = re.sub(r"/+", "/", path).rstrip("/") or "/"
    return host, path


def _is_shared_feed_url(raw_url: str) -> bool:
    """Detect known shared feeds that should not map to a single venue address."""
    host, path = _normalize_url_host_path(raw_url)
    if host.endswith("nycgovparks.org") and path in ("/events", "/events/volunteer"):
        return True
    return False


def _load_event_source_address_lookup() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]], dict[str, str]]:
    """
    Build source-url lookup tables from venue cache.

    Returns:
        - exact_map: "host/path" -> single venue address when unambiguous
        - host_prefix_map: host -> list of (path_prefix, address), longest path first
        - host_unique_map: host -> address when host maps to exactly one venue address
    """
    try:
        from venue_scout.cache import read_cached_venues
    except Exception:
        return {}, {}, {}

    try:
        venues = read_cached_venues()
    except Exception:
        return {}, {}, {}

    exact_to_addresses: dict[str, set[str]] = {}
    host_path_to_addresses: dict[str, dict[str, set[str]]] = {}
    host_to_addresses: dict[str, set[str]] = {}

    for venue in venues:
        address = str(venue.get("address", "") or "").strip()
        if not address:
            continue

        for candidate_url in (
            str(venue.get("events_url", "") or "").strip(),
            str(venue.get("website", "") or "").strip(),
        ):
            if not candidate_url or _is_shared_feed_url(candidate_url):
                continue
            host, path = _normalize_url_host_path(candidate_url)
            if not host:
                continue

            exact_key = f"{host}{path}"
            exact_to_addresses.setdefault(exact_key, set()).add(address)
            host_path_to_addresses.setdefault(host, {}).setdefault(path, set()).add(address)
            host_to_addresses.setdefault(host, set()).add(address)

    exact_map: dict[str, str] = {
        key: next(iter(addresses))
        for key, addresses in exact_to_addresses.items()
        if len(addresses) == 1
    }
    host_prefix_map: dict[str, list[tuple[str, str]]] = {}
    for host, paths in host_path_to_addresses.items():
        entries = []
        for path, addresses in paths.items():
            if len(addresses) == 1:
                entries.append((path, next(iter(addresses))))
        entries.sort(key=lambda item: len(item[0]), reverse=True)
        if entries:
            host_prefix_map[host] = entries

    host_unique_map: dict[str, str] = {
        host: next(iter(addresses))
        for host, addresses in host_to_addresses.items()
        if len(addresses) == 1
    }
    return exact_map, host_prefix_map, host_unique_map


def _address_from_event_source_url(
    source_url: str,
    exact_map: dict[str, str],
    host_prefix_map: dict[str, list[tuple[str, str]]],
    host_unique_map: dict[str, str],
) -> str:
    """Resolve address via event_source_url using exact, prefix, then host-unique matching."""
    if not source_url or _is_shared_feed_url(source_url):
        return ""
    host, path = _normalize_url_host_path(source_url)
    if not host:
        return ""

    exact_key = f"{host}{path}"
    exact_match = exact_map.get(exact_key, "")
    if exact_match:
        return exact_match

    prefix_entries = host_prefix_map.get(host, [])
    for prefix_path, address in prefix_entries:
        if path == prefix_path or path.startswith(prefix_path + "/"):
            return address

    return host_unique_map.get(host, "")


def _populate_event_addresses(events: list[dict]) -> list[dict]:
    """Ensure each event has address from venue cache when available."""
    name_lookup = _load_venue_address_lookup()
    source_exact, source_prefix, source_host_unique = _load_event_source_address_lookup()

    if not name_lookup and not source_exact and not source_prefix and not source_host_unique:
        return events

    out = []
    for event in events:
        row = dict(event)
        if not str(row.get("address", "") or "").strip():
            norm_name = _normalize_venue_name(row.get("venue_name", ""))
            if norm_name and norm_name in name_lookup:
                row["address"] = name_lookup[norm_name]
            else:
                source_url = str(row.get("event_source_url", "") or row.get("url", "")).strip()
                source_address = _address_from_event_source_url(
                    source_url=source_url,
                    exact_map=source_exact,
                    host_prefix_map=source_prefix,
                    host_unique_map=source_host_unique,
                )
                if source_address:
                    row["address"] = source_address
        out.append(row)
    return out


def _get_sheets_service():
    """Get authenticated Google Sheets service."""
    creds = get_credentials()
    if not creds:
        return None
    return build("sheets", "v4", credentials=creds)


def _load_sheets_config() -> dict:
    """Load sheet IDs from config."""
    if _SHEETS_CONFIG.exists():
        with open(_SHEETS_CONFIG) as f:
            return json.load(f)
    return {}


def _save_sheets_config(config: dict):
    """Save sheet IDs to config."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SHEETS_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


def _create_spreadsheet(title: str) -> str | None:
    """Create a new Google Spreadsheet and return its ID."""
    service = _get_sheets_service()
    if not service:
        return None

    spreadsheet = {"properties": {"title": title}}
    result = service.spreadsheets().create(body=spreadsheet).execute()
    return result.get("spreadsheetId")


def _write_header(sheet_id: str, columns: list[str]):
    """Write header row to a sheet."""
    service = _get_sheets_service()
    if not service:
        return

    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range="A1",
        valueInputOption="RAW",
        body={"values": [columns]},
    ).execute()


def get_or_create_venue_events_sheet() -> str | None:
    """Get or create the venue events spreadsheet."""
    config = _load_sheets_config()

    if "venue_events_sheet_id" in config:
        return config["venue_events_sheet_id"]

    creds = get_credentials()
    if not creds:
        print("Not authenticated with Google. Run: python -m utils.google_auth")
        return None

    print("Creating new Venue Events spreadsheet...")
    sheet_id = _create_spreadsheet("Venue Events Cache")
    if sheet_id:
        config["venue_events_sheet_id"] = sheet_id
        _save_sheets_config(config)
        _write_header(sheet_id, VENUE_EVENTS_COLUMNS)
        print(f"Created Venue Events sheet: https://docs.google.com/spreadsheets/d/{sheet_id}")

    return sheet_id


def read_venue_events_from_sheet(venue_name: str | None = None) -> list[dict]:
    """Read venue events from Google Sheet."""
    sheet_id = get_or_create_venue_events_sheet()
    if not sheet_id:
        return []

    service = _get_sheets_service()
    if not service:
        return []

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=_sheet_full_range(),
        ).execute()

        rows = result.get("values", [])
        if len(rows) <= 1:
            return []

        header = rows[0]
        events = []

        for row in rows[1:]:
            while len(row) < len(header):
                row.append("")

            event = normalize_event(dict(zip(header, row)))

            if venue_name and _safe_lower(event.get("venue_name", "")) != venue_name.lower():
                continue

            dt_str = event.get("datetime", "")
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

            travel_str = event.get("travel_minutes", "")
            if travel_str and travel_str not in ("", "None"):
                try:
                    event["travel_minutes"] = int(travel_str)
                except ValueError:
                    event["travel_minutes"] = None
            else:
                event["travel_minutes"] = None

            score_str = event.get("relevance_score", "")
            if score_str and score_str not in ("", "None"):
                try:
                    event["relevance_score"] = int(float(score_str))
                except ValueError:
                    event["relevance_score"] = None
            else:
                event["relevance_score"] = None

            conf_str = event.get("validation_confidence", "")
            if conf_str and conf_str not in ("", "None"):
                try:
                    event["validation_confidence"] = float(conf_str)
                except ValueError:
                    event["validation_confidence"] = None
            else:
                event["validation_confidence"] = None

            semantic_flag = str(event.get("in_semantic_index", "") or "").strip().lower()
            event["in_semantic_index"] = semantic_flag in ("true", "1", "yes")
            event["semantic_indexed_at"] = str(event.get("semantic_indexed_at", "") or "").strip()

            events.append(event)

        return events

    except Exception as e:
        print(f"Error reading venue events from sheet: {e}")
        return []


def _ensure_archive_tab(sheet_id: str, service) -> bool:
    """Ensure the Archive tab exists in the spreadsheet. Returns True if ready."""
    try:
        spreadsheet = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        existing_titles = {s["properties"]["title"] for s in spreadsheet.get("sheets", [])}
        if _ARCHIVE_TAB in existing_titles:
            return True
        service.spreadsheets().batchUpdate(
            spreadsheetId=sheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": _ARCHIVE_TAB}}}]},
        ).execute()
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"{_ARCHIVE_TAB}!A1",
            valueInputOption="RAW",
            body={"values": [VENUE_EVENTS_COLUMNS]},
        ).execute()
        print(f"Created '{_ARCHIVE_TAB}' tab in venue events spreadsheet")
        return True
    except Exception as e:
        print(f"Warning: could not ensure archive tab: {e}")
        return False


def _read_archive_keys(sheet_id: str, service) -> set[tuple]:
    """Read existing archive event dedup keys to avoid re-archiving."""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{_ARCHIVE_TAB}!A:{_sheet_col_label(len(VENUE_EVENTS_COLUMNS))}",
        ).execute()
        rows = result.get("values", [])
        if not rows or len(rows) < 2:
            return set()
        headers = rows[0]
        events = []
        for row in rows[1:]:
            padded = row + [""] * max(0, len(headers) - len(row))
            events.append(dict(zip(headers, padded)))
        return {_event_dedupe_key(e) for e in events}
    except Exception:
        return set()


def _append_to_archive(sheet_id: str, service, past_events: list[dict]) -> int:
    """Append past events to the Archive tab, deduplicating against existing entries."""
    if not past_events:
        return 0
    if not _ensure_archive_tab(sheet_id, service):
        return 0

    existing_keys = _read_archive_keys(sheet_id, service)
    rows_to_append = []
    for event in past_events:
        if _event_dedupe_key(event) in existing_keys:
            continue
        dt = event.get("datetime")
        if isinstance(dt, datetime):
            dt_str = dt.isoformat()
        elif dt:
            dt_str = str(dt)
        else:
            dt_str = ""
        rows_to_append.append([
            event.get("name", ""),
            dt_str,
            event.get("date_str", ""),
            event.get("end_date", ""),
            event.get("venue_name", ""),
            event.get("address", ""),
            event.get("event_type", ""),
            event.get("url", ""),
            event.get("source", ""),
            event.get("matched_artist", ""),
            str(event.get("travel_minutes")) if event.get("travel_minutes") is not None else "",
            event.get("description", ""),
            event.get("event_source_url", ""),
            event.get("extraction_method", ""),
            str(event.get("relevance_score")) if event.get("relevance_score") is not None else "",
            str(event.get("validation_confidence")) if event.get("validation_confidence") is not None else "",
            event.get("date_added", ""),
            "TRUE" if bool(event.get("in_semantic_index", False)) else "FALSE",
            event.get("semantic_indexed_at", ""),
        ])

    if not rows_to_append:
        return 0

    try:
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range=f"{_ARCHIVE_TAB}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": rows_to_append},
        ).execute()
        return len(rows_to_append)
    except Exception as e:
        print(f"Warning: could not append to archive: {e}")
        return 0


def write_venue_events_to_sheet(events: list[dict], venue_name: str | None = None):
    """Write venue events to Google Sheet."""
    sheet_id = get_or_create_venue_events_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    if venue_name:
        existing = read_venue_events_from_sheet()
        existing = [e for e in existing if _safe_lower(e.get("venue_name", "")) != venue_name.lower()]
        venue_events = [e for e in events if _safe_lower(e.get("venue_name", "")) == venue_name.lower()]
        all_events = existing + venue_events
    else:
        all_events = events

    all_events = [normalize_event(event) for event in all_events]
    all_events = _populate_event_addresses(all_events)
    all_events, dedup_removed = _dedupe_events(all_events)

    today = datetime.now(ZoneInfo("America/New_York")).date()
    max_allowed_date = today + timedelta(days=_MAX_EVENT_DAYS_AHEAD)
    future_events = []
    past_events = []
    dropped_past = 0
    dropped_too_far = 0
    undated_kept = 0
    for event in all_events:
        event_date = _event_date_for_window(event)
        if event_date is None:
            undated_kept += 1
            future_events.append(event)
            continue

        if event_date < today:
            dropped_past += 1
            past_events.append(event)
            continue
        if event_date > max_allowed_date:
            dropped_too_far += 1
            continue
        future_events.append(event)

    future_events.sort(
        key=lambda x: (
            _safe_lower(x.get("venue_name", "")),
            x.get("datetime").isoformat() if isinstance(x.get("datetime"), datetime)
            else (x.get("datetime") or "9999"),
        )
    )

    rows = [VENUE_EVENTS_COLUMNS]
    for event in future_events:
        dt = event.get("datetime")
        if isinstance(dt, datetime):
            dt_str = dt.isoformat()
        elif dt:
            dt_str = str(dt)
        else:
            dt_str = ""

        row = [
            event.get("name", ""),
            dt_str,
            event.get("date_str", ""),
            event.get("end_date", ""),
            event.get("venue_name", ""),
            event.get("address", ""),
            event.get("event_type", ""),
            event.get("url", ""),
            event.get("source", ""),
            event.get("matched_artist", ""),
            str(event.get("travel_minutes")) if event.get("travel_minutes") is not None else "",
            event.get("description", ""),
            event.get("event_source_url", ""),
            event.get("extraction_method", ""),
            str(event.get("relevance_score")) if event.get("relevance_score") is not None else "",
            str(event.get("validation_confidence")) if event.get("validation_confidence") is not None else "",
            event.get("date_added", ""),
            "TRUE" if bool(event.get("in_semantic_index", False)) else "FALSE",
            event.get("semantic_indexed_at", ""),
        ]
        rows.append(row)

    try:
        # Write new data first (overwrites existing rows)
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()

        # Only clear extra rows AFTER successful write
        # This prevents data loss if write fails
        clear_start_row = len(rows) + 1
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=f"A{clear_start_row}:{_sheet_col_label(len(VENUE_EVENTS_COLUMNS))}",
        ).execute()

        archived_count = _append_to_archive(sheet_id, service, past_events)
        print(
            f"Wrote {len(future_events)} venue events to sheet "
            f"(dedup_removed={dedup_removed}, dropped_past={dropped_past}, "
            f"archived={archived_count}, dropped_too_far={dropped_too_far}, undated_kept={undated_kept})"
        )

    except Exception as e:
        print(f"Error writing venue events to sheet: {e}")
        raise  # Re-raise so callers know the write failed


def append_venue_events(events: list[dict], venue_name: str):
    """Append events for a venue, deduplicating against existing."""
    if not events:
        return

    existing = read_venue_events_from_sheet()
    existing_keys = set()
    for event in existing:
        key = (
            _safe_lower(event.get("venue_name", "")),
            _safe_lower(event.get("name", "")),
            event.get("date_str", ""),
        )
        existing_keys.add(key)

    new_events = []
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    for event in events:
        event = normalize_event(event)
        key = (
            _safe_lower(event.get("venue_name", "")),
            _safe_lower(event.get("name", "")),
            event.get("date_str", ""),
        )
        if key not in existing_keys:
            # Set date_added for new events
            if not event.get("date_added"):
                event["date_added"] = now
            new_events.append(event)
            existing_keys.add(key)

    if not new_events:
        print(f"  No new events for {venue_name} (all duplicates)")
        return

    all_events = existing + new_events
    write_venue_events_to_sheet(all_events)
    print(f"  Added {len(new_events)} new events for {venue_name}")


def sync_semantic_index_membership(included_event_keys: set[str], indexed_at: str | None = None) -> dict:
    """Persist semantic-index membership flags back to Venue Events sheet.

    This updates only `in_semantic_index` and `semantic_indexed_at` columns to avoid
    mutating core event fields (name/date/type/url) during membership sync.
    """
    from venue_scout.semantic_search import event_key  # Local import avoids circular import at module load.

    sheet_id = get_or_create_venue_events_sheet()
    if not sheet_id:
        return {"sheet_event_count": 0, "included_count": 0, "excluded_count": 0}

    service = _get_sheets_service()
    if not service:
        return {"sheet_event_count": 0, "included_count": 0, "excluded_count": 0}

    events = read_venue_events_from_sheet()
    if not events:
        return {"sheet_event_count": 0, "included_count": 0, "excluded_count": 0}

    included_count = 0
    indexed_at_value = str(indexed_at or datetime.now(ZoneInfo("America/New_York")).isoformat())
    membership_values: list[list[str]] = []
    for event in events:
        key = event_key(event)
        is_included = key in included_event_keys
        membership_values.append([
            "TRUE" if is_included else "FALSE",
            indexed_at_value if is_included else "",
        ])
        if is_included:
            included_count += 1

    start_col_idx = VENUE_EVENTS_COLUMNS.index("in_semantic_index") + 1
    end_col_idx = VENUE_EVENTS_COLUMNS.index("semantic_indexed_at") + 1
    start_col = _sheet_col_label(start_col_idx)
    end_col = _sheet_col_label(end_col_idx)
    update_range = f"{start_col}2:{end_col}{len(membership_values) + 1}"

    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=update_range,
        valueInputOption="RAW",
        body={"values": membership_values},
    ).execute()

    # Defensive cleanup if stale rows exist below current event count.
    clear_start_row = len(membership_values) + 2
    service.spreadsheets().values().clear(
        spreadsheetId=sheet_id,
        range=f"{start_col}{clear_start_row}:{end_col}",
    ).execute()

    persisted_events = read_venue_events_from_sheet()
    persisted_included = sum(1 for event in persisted_events if bool(event.get("in_semantic_index", False)))
    return {
        "sheet_event_count": len(persisted_events),
        "included_count": persisted_included,
        "excluded_count": len(persisted_events) - persisted_included,
        "semantic_indexed_at": indexed_at_value,
    }


def get_events_by_venue() -> dict[str, list[dict]]:
    """Get all events grouped by venue."""
    events = read_venue_events_from_sheet()
    by_venue = {}
    for event in events:
        venue = event.get("venue_name", "Unknown")
        by_venue.setdefault(venue, []).append(event)
    return by_venue


def get_matched_events() -> list[dict]:
    """Get all events that have a matched artist."""
    events = read_venue_events_from_sheet()
    return [event for event in events if event.get("matched_artist")]


def test_venue_events_sheet():
    """Test the venue events sheet integration."""
    print("Testing Venue Events Sheet integration...")

    if not is_authenticated():
        print("Not authenticated. Run: python -m utils.google_auth")
        return

    sheet_id = get_or_create_venue_events_sheet()
    print(f"Venue Events sheet: {sheet_id}")

    events = read_venue_events_from_sheet()
    print(f"\nCurrent data: {len(events)} events")

    by_venue = get_events_by_venue()
    print(f"\nBy venue: {len(by_venue)} venues")
    for venue, venue_events in sorted(by_venue.items()):
        print(f"  {venue}: {len(venue_events)} events")


if __name__ == "__main__":
    test_venue_events_sheet()
