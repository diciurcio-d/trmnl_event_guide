"""Firestore storage for venue events.

Storage backend: Cloud Firestore
  - Collection ``venue_events``  — active/upcoming events
  - Collection ``archived_events`` — past events (append-only archive)

The Run Log is still written to Google Sheets so it remains human-readable
and easy to inspect in a spreadsheet.
"""

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.google_auth import get_credentials, is_authenticated


# ─────────────────────────── Config / constants ───────────────────────────────

_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SHEETS_CONFIG = _CONFIG_DIR / "sheets_config.json"
_MAX_EVENT_DAYS_AHEAD = 365

# Firestore collections
_EVENTS_COLLECTION = "venue_events"
_ARCHIVE_COLLECTION = "archived_events"
_FIRESTORE_BATCH_SIZE = 500

# Transient HTTP status codes worth retrying (used only for Run Log writes)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRY_DELAYS = [10, 20, 40]

# ── In-process events cache ───────────────────────────────────────────────────
# Loaded once per container lifetime from GCS (fast) with Firestore as fallback.
# Invalidated after any write so the next read picks up fresh data.
_events_cache: list[dict] | None = None
_events_cache_lock = threading.Lock()


# ─────────────────────────── Sheets retry (Run Log only) ─────────────────────

def _execute_with_retry(request, label: str = ""):
    """Execute a Google Sheets API request with up to 3 retries on transient errors."""
    last_exc = None
    delays = [0] + _RETRY_DELAYS
    for attempt, delay in enumerate(delays):
        if delay:
            print(f"  [retry {attempt}/{len(_RETRY_DELAYS)}] {label or 'API call'} failed — waiting {delay}s...")
            time.sleep(delay)
        try:
            return request.execute()
        except HttpError as e:
            if e.resp.status in _RETRYABLE_STATUS_CODES:
                last_exc = e
                print(f"  Transient error (HTTP {e.resp.status}) on {label or 'API call'}: {e}")
                continue
            raise
    raise last_exc


# ─────────────────────────── Geo filter helpers ───────────────────────────────

_NYC_AREA_MARKERS = (
    "new york", ", ny", "nyc", "brooklyn", "manhattan", "queens",
    "bronx", "staten island", "hoboken", "jersey city", "newark", ", nj", ", ct",
)

_NON_NYC_MARKERS = (
    "london", "paris", "berlin", "amsterdam", "toronto", "sydney",
    "los angeles", "san francisco", "chicago", "boston", "miami",
    "seattle", "atlanta", "denver", "austin", "nashville",
    "philadelphia", "las vegas", "portland", "new orleans",
    "washington, dc", "washington dc", "baltimore", "cleveland",
    "detroit", "minneapolis", "dallas", "houston", "phoenix",
    ", ca", "california", ", tx", "texas", ", il", "illinois",
    ", fl", "florida", ", ma", "massachusetts",
    ", ga", "georgia", ", co", "colorado", ", pa", "pennsylvania",
    ", oh", "ohio", ", mi", "michigan", ", tn", "tennessee",
    ", nc", "north carolina", ", or", "oregon", ", va", "virginia",
    ", md", "maryland", ", mn", "minnesota", ", wa", "washington state",
    "united kingdom", "england", "france", "germany", "canada",
    "australia", "netherlands",
)


def _is_non_nyc_event(event: dict) -> bool:
    """Return True if the event's location is clearly outside the NYC metro area."""
    address = str(event.get("address", "") or "").strip().lower()
    if address:
        if any(m in address for m in _NYC_AREA_MARKERS):
            return False
        if any(m in address for m in _NON_NYC_MARKERS):
            return True

    venue = str(event.get("venue_name", "") or "").strip().lower()
    if venue:
        if any(m in venue for m in _NON_NYC_MARKERS):
            if not any(m in venue for m in _NYC_AREA_MARKERS):
                return True

    return False


def _safe_lower(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.lower()
    return str(val).lower()


# ─────────────────────────── Schema / normalisation ──────────────────────────

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
    "is_free",
    "price",
    "event_source_url",
    "extraction_method",
    "relevance_score",
    "validation_confidence",
    "date_added",
    "in_semantic_index",
    "semantic_indexed_at",
    "is_recurring",
]


def normalize_event(event: dict) -> dict:
    """Normalise a venue event into the canonical schema."""
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
    is_free = normalized.get("is_free")
    if isinstance(is_free, bool):
        normalized["is_free"] = is_free
    elif isinstance(is_free, str):
        s = is_free.strip().lower()
        if s in ("true", "1", "yes"):
            normalized["is_free"] = True
        elif s in ("false", "0", "no"):
            normalized["is_free"] = False
        else:
            normalized["is_free"] = None
    elif is_free is not None:
        normalized["is_free"] = bool(is_free)
    else:
        normalized["is_free"] = None
    normalized["price"] = normalized.get("price", "")
    normalized["event_source_url"] = normalized.get("event_source_url", "")
    normalized["extraction_method"] = normalized.get("extraction_method", "")
    normalized["relevance_score"] = normalized.get("relevance_score")
    normalized["validation_confidence"] = normalized.get("validation_confidence")
    normalized["date_added"] = normalized.get("date_added", "")
    normalized["in_semantic_index"] = normalized.get("in_semantic_index", False)
    normalized["semantic_indexed_at"] = normalized.get("semantic_indexed_at", "")
    is_recurring = normalized.get("is_recurring")
    if isinstance(is_recurring, bool):
        normalized["is_recurring"] = is_recurring
    elif isinstance(is_recurring, str):
        normalized["is_recurring"] = is_recurring.lower() in ("true", "1", "yes")
    else:
        normalized["is_recurring"] = False
    return normalized


def _normalize_text(value) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_event_category(value) -> str:
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
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _event_date_for_window(event: dict):
    """Return the date to use for keep/drop window checks."""
    end_raw = str(event.get("end_date", "") or "").strip()
    if end_raw:
        end_parsed = _parse_date_str(end_raw)
        if end_parsed:
            return end_parsed

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
    """Cross-venue dedup key — (name, date, source, link-or-venue)."""
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
    """Collapse duplicate events with deterministic quality tie-break."""
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
    text = str(name or "").lower().strip()
    text = re.sub(r'^the\s+', '', text)
    text = re.sub(r'^[(]', '', text)
    text = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _load_venue_address_lookup() -> dict[str, str]:
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
    host, path = _normalize_url_host_path(raw_url)
    if host.endswith("nycgovparks.org") and path in ("/events", "/events/volunteer"):
        return True
    return False


def _load_event_source_address_lookup() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]], dict[str, str]]:
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


# ─────────────────────────── Recurring event detection ──────────────────────

def _normalize_name_for_recurrence(name: str) -> str:
    """Reduce an event name to a stable key for grouping repeated instances.

    Strips trailing date/episode tokens so "Jazz Night – Apr 3" and
    "Jazz Night – Apr 10" collapse to the same key.
    """
    s = name.lower().strip()
    s = re.sub(r"[\-–—#|]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\d,]*$", "", s)
    s = re.sub(r"[\-–—#|]\s*\d[\d/.\-]*\s*$", "", s)
    s = re.sub(r"\s*(vol\.?|volume|ep\.?|episode|#)\s*\d+\s*$", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _compute_recurring_doc_ids(events: list[dict]) -> set[str]:
    """Return the set of doc IDs for events that recur more than once per week.

    An event name+venue group is considered recurring if any two consecutive
    instances (sorted by date) are ≤ 7 days apart.
    """
    from collections import defaultdict

    def _event_date(event: dict):
        dt = event.get("datetime")
        if isinstance(dt, datetime):
            return dt.date()
        raw = str(event.get("date_str") or "").strip()
        return _parse_date_str(raw) if raw else None

    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for event in events:
        venue = str(event.get("venue_name") or "").strip().lower()
        name = _normalize_name_for_recurrence(str(event.get("name") or ""))
        if not venue or not name:
            continue
        d = _event_date(event)
        groups[(venue, name)].append((d, _event_doc_id(event)))

    recurring: set[str] = set()
    for entries in groups.values():
        if len(entries) < 2:
            continue
        dated = sorted((d, doc_id) for d, doc_id in entries if d is not None)
        if len(dated) < 2:
            continue
        for (d1, _), (d2, _) in zip(dated, dated[1:]):
            if (d2 - d1).days <= 7:
                for _, doc_id in entries:
                    recurring.add(doc_id)
                break
    return recurring


def recompute_recurring_for_venue(venue_name: str) -> dict:
    """Recompute is_recurring for all events at a venue and update changed docs.

    Called after any write to a venue's events so the flag stays current.
    Returns a summary dict with counts of events set True/False.
    """
    db = _get_db()
    col = db.collection(_EVENTS_COLLECTION)

    events = read_venue_events_from_sheet(venue_name)
    if not events:
        return {"set_true": 0, "set_false": 0}

    recurring_ids = _compute_recurring_doc_ids(events)

    set_true = set_false = 0
    for start in range(0, len(events), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        for event in events[start:start + _FIRESTORE_BATCH_SIZE]:
            doc_id = _event_doc_id(event)
            new_val = doc_id in recurring_ids
            if event.get("is_recurring") != new_val:
                batch.set(col.document(doc_id), {"is_recurring": new_val}, merge=True)
                if new_val:
                    set_true += 1
                else:
                    set_false += 1
        batch.commit()

    return {"set_true": set_true, "set_false": set_false}


# ─────────────────────────── Firestore helpers ───────────────────────────────

def _get_db():
    """Return the Firestore client."""
    from venue_scout.firestore_client import get_db
    return get_db()


def _event_doc_id(event: dict) -> str:
    """Compute a stable Firestore document ID from the event dedup key."""
    key = _event_dedupe_key(event)
    key_str = "|".join(str(k) for k in key)
    return hashlib.md5(key_str.encode()).hexdigest()


def _serialize_event_for_db(event: dict) -> dict:
    """Convert an event dict to Firestore-native format."""
    row = dict(event)

    dt = row.get("datetime")
    if isinstance(dt, datetime):
        row["datetime"] = dt.isoformat()
    elif dt is None or dt == "" or dt == "None":
        row["datetime"] = None
    else:
        row["datetime"] = str(dt)

    # Normalise numeric/boolean fields so Firestore stores native types
    for field in ("travel_minutes", "relevance_score"):
        val = row.get(field)
        if val == "" or val == "None" or val is None:
            row[field] = None
        elif isinstance(val, str):
            try:
                row[field] = int(float(val))
            except (ValueError, TypeError):
                row[field] = None

    val = row.get("validation_confidence")
    if val == "" or val == "None" or val is None:
        row["validation_confidence"] = None
    elif isinstance(val, str):
        try:
            row["validation_confidence"] = float(val)
        except (ValueError, TypeError):
            row["validation_confidence"] = None

    return row


def _deserialize_event_from_db(data: dict) -> dict:
    """Convert Firestore document data to a properly-typed event dict."""
    event = normalize_event(data)

    # Parse datetime
    dt_val = event.get("datetime")
    if isinstance(dt_val, str) and dt_val and dt_val not in ("None", ""):
        try:
            dt = datetime.fromisoformat(dt_val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
            event["datetime"] = dt
        except ValueError:
            event["datetime"] = None
    elif hasattr(dt_val, "timestamp"):
        # Firestore Timestamp object
        event["datetime"] = dt_val.replace(tzinfo=ZoneInfo("America/New_York"))
    else:
        event["datetime"] = None

    # Parse numeric fields — Firestore may return them as native ints/floats
    for field, converter in [
        ("travel_minutes", lambda v: int(v)),
        ("relevance_score", lambda v: int(float(v))),
        ("validation_confidence", float),
    ]:
        val = event.get(field)
        if val is None or val == "" or val == "None":
            event[field] = None
        elif isinstance(val, (int, float)):
            pass  # already correct type
        else:
            try:
                event[field] = converter(str(val))
            except (ValueError, TypeError):
                event[field] = None

    # Ensure boolean
    si = event.get("in_semantic_index")
    if isinstance(si, bool):
        pass
    elif isinstance(si, str):
        event["in_semantic_index"] = si.lower() in ("true", "1", "yes")
    else:
        event["in_semantic_index"] = bool(si)

    return event


def _batch_set_events(db, col, events: list[dict]):
    """Batch-upsert events into a Firestore collection."""
    for start in range(0, len(events), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        for event in events[start:start + _FIRESTORE_BATCH_SIZE]:
            doc_id = _event_doc_id(event)
            batch.set(col.document(doc_id), _serialize_event_for_db(event))
        batch.commit()


def _batch_delete_refs(db, refs: list):
    """Batch-delete Firestore document references."""
    for start in range(0, len(refs), _FIRESTORE_BATCH_SIZE):
        batch = db.batch()
        for ref in refs[start:start + _FIRESTORE_BATCH_SIZE]:
            batch.delete(ref)
        batch.commit()


def _append_to_archive_db(db, past_events: list[dict]) -> int:
    """Append past events to the Firestore archive collection (dedup-safe)."""
    if not past_events:
        return 0

    archive_col = db.collection(_ARCHIVE_COLLECTION)

    # Only archive events not already present
    existing_ids = {doc.id for doc in archive_col.select([]).stream()}
    to_archive = [e for e in past_events if _event_doc_id(e) not in existing_ids]
    if not to_archive:
        return 0

    _batch_set_events(db, archive_col, to_archive)
    return len(to_archive)


def _process_events_for_write(
    all_events: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """
    Apply the full normalise → address-enrich → dedup → geo-filter → date-window
    pipeline.  Returns (future_events, past_events, stats_dict).
    """
    all_events = [normalize_event(e) for e in all_events]
    all_events = _populate_event_addresses(all_events)
    all_events, dedup_removed = _dedupe_events(all_events)

    geo_kept = [e for e in all_events if not _is_non_nyc_event(e)]
    geo_dropped = len(all_events) - len(geo_kept)
    all_events = geo_kept

    today = datetime.now(ZoneInfo("America/New_York")).date()
    max_allowed_date = today + timedelta(days=_MAX_EVENT_DAYS_AHEAD)
    future_events: list[dict] = []
    past_events: list[dict] = []
    dropped_past = dropped_too_far = undated_kept = 0

    for event in all_events:
        event_date = _event_date_for_window(event)
        if event_date is None:
            undated_kept += 1
            future_events.append(event)
        elif event_date < today:
            dropped_past += 1
            past_events.append(event)
        elif event_date > max_allowed_date:
            dropped_too_far += 1
        else:
            future_events.append(event)

    stats = {
        "dedup_removed": dedup_removed,
        "geo_dropped": geo_dropped,
        "dropped_past": dropped_past,
        "dropped_too_far": dropped_too_far,
        "undated_kept": undated_kept,
    }
    return future_events, past_events, stats


# ─────────────────────────── Public storage API ──────────────────────────────

def _serialize_events_for_gcs(events: list[dict]) -> list[dict]:
    """Convert event dicts to JSON-safe form for GCS storage."""
    out = []
    for event in events:
        e = dict(event)
        dt = e.get("datetime")
        if dt is not None and hasattr(dt, "isoformat"):
            e["datetime"] = dt.isoformat()
        out.append(e)
    return out


def _load_all_events_uncached() -> list[dict]:
    """Stream all events directly from Firestore (no cache)."""
    db = _get_db()
    events = []
    for doc in db.collection(_EVENTS_COLLECTION).stream():
        data = doc.to_dict()
        if data:
            events.append(_deserialize_event_from_db(data))
    return events


def _load_all_events() -> list[dict]:
    """Return all events, loading from GCS or Firestore on first call and
    caching in memory for the lifetime of the container process."""
    global _events_cache
    with _events_cache_lock:
        if _events_cache is not None:
            return _events_cache

        # Try GCS snapshot first (fast — ~1s vs ~25s for Firestore stream)
        try:
            from venue_scout.gcs_sync import pull_events_cache
            gcs_events = pull_events_cache(verbose=True)
            if gcs_events is not None:
                _events_cache = [_deserialize_event_from_db(e) for e in gcs_events]
                print(f"  Events cache loaded from GCS: {len(_events_cache):,} events")
                return _events_cache
        except Exception as exc:
            print(f"  GCS events cache unavailable ({exc.__class__.__name__}), falling back to Firestore.")

        # Fallback: stream from Firestore
        print("  Loading events from Firestore (cold start)...")
        _events_cache = _load_all_events_uncached()
        print(f"  Events loaded from Firestore: {len(_events_cache):,} events")
        return _events_cache


def invalidate_events_cache() -> None:
    """Clear the in-process events cache so the next read fetches fresh data."""
    global _events_cache
    with _events_cache_lock:
        _events_cache = None


def read_venue_events_from_sheet(venue_name: str | None = None) -> list[dict]:
    """Read venue events from Firestore.

    Full load (venue_name=None): served from the in-process cache which is
    populated from GCS on first call and held for the container lifetime.
    Per-venue reads always go directly to Firestore (used during nightly writes).
    """
    try:
        if venue_name is None:
            return _load_all_events()

        # Per-venue: always fresh from Firestore
        db = _get_db()
        col = db.collection(_EVENTS_COLLECTION)
        events = []
        for doc in col.where("venue_name", "==", venue_name).stream():
            data = doc.to_dict()
            if data:
                events.append(_deserialize_event_from_db(data))
        return events

    except Exception as e:
        print(f"Error reading venue events from Firestore: {e}")
        return []


def write_venue_events_to_sheet(events: list[dict], venue_name: str | None = None):
    """Write venue events to Firestore.

    Invalidates the in-process events cache so subsequent reads reflect the write.

    Venue-scoped write (venue_name provided):
        Processes only that venue's events.  Deletes the venue's stale docs
        and upserts the new set.  All other venues are untouched.

    Full overwrite (venue_name=None):
        Processes the entire provided event list.  Deletes docs not present
        in the new set and upserts the new set.
    """
    try:
        db = _get_db()
        col = db.collection(_EVENTS_COLLECTION)

        if venue_name:
            venue_lower = venue_name.lower()
            input_events = [e for e in events if _safe_lower(e.get("venue_name", "")) == venue_lower]
        else:
            input_events = list(events)

        future_events, past_events, stats = _process_events_for_write(input_events)

        # Archive past events
        archived_count = _append_to_archive_db(db, past_events)

        # Determine which existing docs to delete
        new_doc_ids = {_event_doc_id(e) for e in future_events}
        if venue_name:
            existing_docs = [
                (doc.id, doc.reference)
                for doc in col.where("venue_name", "==", venue_name).select([]).stream()
            ]
        else:
            existing_docs = [
                (doc.id, doc.reference)
                for doc in col.select([]).stream()
            ]
        to_delete = [ref for doc_id, ref in existing_docs if doc_id not in new_doc_ids]

        _batch_delete_refs(db, to_delete)
        _batch_set_events(db, col, future_events)
        invalidate_events_cache()

        if venue_name:
            recurring_result = recompute_recurring_for_venue(venue_name)
            if recurring_result["set_true"] or recurring_result["set_false"]:
                print(f"  Recurring recompute: {recurring_result['set_true']} marked recurring, {recurring_result['set_false']} unmarked")

        scope = venue_name or "all venues"
        print(
            f"Wrote {len(future_events)} venue events to Firestore [{scope}] "
            f"(dedup_removed={stats['dedup_removed']}, geo_dropped={stats['geo_dropped']}, "
            f"dropped_past={stats['dropped_past']}, archived={archived_count}, "
            f"dropped_too_far={stats['dropped_too_far']}, undated_kept={stats['undated_kept']})"
        )

    except Exception as e:
        print(f"Error writing venue events to Firestore: {e}")
        raise


def append_venue_events(events: list[dict], venue_name: str):
    """Append new events for a venue, deduplicating against what is already stored.

    For events that already exist (same dedupe key), updates is_free and price
    if the new fetch has non-null values for them — this keeps those fields
    fresh across repeated fetches without re-embedding or re-deduplicating.
    """
    if not events:
        return 0

    try:
        db = _get_db()
        col = db.collection(_EVENTS_COLLECTION)

        # Read existing events for this venue only
        existing = read_venue_events_from_sheet(venue_name)
        existing_by_key: dict[tuple, str] = {_event_dedupe_key(e): _event_doc_id(e) for e in existing}

        new_events: list[dict] = []
        price_updates: list[tuple[str, dict]] = []  # (doc_id, merge_fields)
        now = datetime.now(ZoneInfo("America/New_York")).isoformat()

        for event in events:
            event = normalize_event(event)
            key = _event_dedupe_key(event)
            if key in existing_by_key:
                # Existing event — refresh is_free/price if we have new data
                is_free_new = event.get("is_free")
                price_new = str(event.get("price") or "").strip()
                if is_free_new is not None or price_new:
                    price_updates.append((existing_by_key[key], {
                        "is_free": is_free_new,
                        "price": price_new,
                    }))
            else:
                if not event.get("date_added"):
                    event["date_added"] = now
                new_events.append(event)
                existing_by_key[key] = _event_doc_id(event)  # prevent within-batch dupes

        # Merge-update is_free/price on already-stored events
        if price_updates:
            for start in range(0, len(price_updates), _FIRESTORE_BATCH_SIZE):
                batch = db.batch()
                for doc_id, fields in price_updates[start:start + _FIRESTORE_BATCH_SIZE]:
                    batch.set(col.document(doc_id), fields, merge=True)
                batch.commit()

        if not new_events:
            print(f"  No new events for {venue_name} (all duplicates)")
            return 0

        _batch_set_events(db, col, new_events)
        invalidate_events_cache()
        print(f"  Added {len(new_events)} new events for {venue_name}")

        recurring_result = recompute_recurring_for_venue(venue_name)
        if recurring_result["set_true"] or recurring_result["set_false"]:
            print(f"  Recurring recompute: {recurring_result['set_true']} marked recurring, {recurring_result['set_false']} unmarked")

        return len(new_events)

    except Exception as e:
        print(f"Error appending venue events to Firestore: {e}")
        return 0


def sync_semantic_index_membership(included_event_keys: set[str], indexed_at: str | None = None) -> dict:
    """Persist semantic-index membership flags to Firestore.

    Only updates ``in_semantic_index`` and ``semantic_indexed_at``; all other
    event fields are left unchanged.
    """
    from venue_scout.semantic_search import event_key  # local import avoids circular

    try:
        db = _get_db()
        col = db.collection(_EVENTS_COLLECTION)

        events = read_venue_events_from_sheet()
        if not events:
            return {"sheet_event_count": 0, "included_count": 0, "excluded_count": 0}

        indexed_at_value = str(indexed_at or datetime.now(ZoneInfo("America/New_York")).isoformat())
        included_count = 0

        for start in range(0, len(events), _FIRESTORE_BATCH_SIZE):
            batch = db.batch()
            for event in events[start:start + _FIRESTORE_BATCH_SIZE]:
                key = event_key(event)
                is_included = key in included_event_keys
                doc_id = _event_doc_id(event)
                batch.set(
                    col.document(doc_id),
                    {
                        "in_semantic_index": is_included,
                        "semantic_indexed_at": indexed_at_value if is_included else "",
                    },
                    merge=True,
                )
                if is_included:
                    included_count += 1
            batch.commit()

        return {
            "sheet_event_count": len(events),
            "included_count": included_count,
            "excluded_count": len(events) - included_count,
            "semantic_indexed_at": indexed_at_value,
        }

    except Exception as e:
        print(f"Error syncing semantic index flags in Firestore: {e}")
        return {"sheet_event_count": 0, "included_count": 0, "excluded_count": 0}


def get_events_by_venue() -> dict[str, list[dict]]:
    """Return all events grouped by venue name."""
    events = read_venue_events_from_sheet()
    by_venue: dict[str, list[dict]] = {}
    for event in events:
        venue = event.get("venue_name", "Unknown")
        by_venue.setdefault(venue, []).append(event)
    return by_venue


def get_matched_events() -> list[dict]:
    """Return all events that have a matched artist."""
    return [e for e in read_venue_events_from_sheet() if e.get("matched_artist")]


# ─────────────────────────── Run Log (still in Sheets) ───────────────────────

_RUN_LOG_TAB = "Run Log"
_RUN_LOG_COLUMNS = [
    "date",
    "started_at",
    "duration_min",
    "venues_processed",
    "venues_with_events",
    "errors",
    "events_before",
    "events_added",
    "events_removed",
    "events_after",
    "status",
]


def _get_sheets_service():
    """Get authenticated Google Sheets service (for Run Log only)."""
    creds = get_credentials()
    if not creds:
        return None
    return build("sheets", "v4", credentials=creds)


def _load_sheets_config() -> dict:
    if _SHEETS_CONFIG.exists():
        with open(_SHEETS_CONFIG) as f:
            return json.load(f)
    return {}


def get_or_create_venue_events_sheet() -> str | None:
    """Return the venue-events spreadsheet ID (used only for Run Log tab)."""
    config = _load_sheets_config()
    return config.get("venue_events_sheet_id")


def _sheet_col_label(col_index: int) -> str:
    """1-based column index → A1 label (used only for Run Log)."""
    if col_index < 1:
        return "A"
    out = ""
    value = col_index
    while value:
        value, rem = divmod(value - 1, 26)
        out = chr(65 + rem) + out
    return out


def write_run_log(stats: dict) -> None:
    """Append a run-summary row to the 'Run Log' tab in the Venue Events sheet.

    Kept in Sheets so the log remains human-readable in a spreadsheet.

    stats keys (all optional):
        started_at, duration_min, venues_processed, venues_with_events,
        errors, events_before, events_added, events_removed, events_after, status
    """
    sheet_id = get_or_create_venue_events_sheet()
    if not sheet_id:
        print("  write_run_log: no sheet ID configured, skipping.")
        return

    service = _get_sheets_service()
    if not service:
        print("  write_run_log: no Sheets service, skipping.")
        return

    # Ensure the Run Log tab exists; create it if not.
    try:
        meta = _execute_with_retry(
            service.spreadsheets().get(spreadsheetId=sheet_id),
            "get spreadsheet metadata",
        )
        existing_tabs = {s["properties"]["title"] for s in meta.get("sheets", [])}
        if _RUN_LOG_TAB not in existing_tabs:
            _execute_with_retry(
                service.spreadsheets().batchUpdate(
                    spreadsheetId=sheet_id,
                    body={"requests": [{"addSheet": {"properties": {"title": _RUN_LOG_TAB}}}]},
                ),
                "create Run Log tab",
            )
            _execute_with_retry(
                service.spreadsheets().values().update(
                    spreadsheetId=sheet_id,
                    range=f"'{_RUN_LOG_TAB}'!A1",
                    valueInputOption="RAW",
                    body={"values": [_RUN_LOG_COLUMNS]},
                ),
                "write Run Log header",
            )
    except Exception as exc:
        print(f"  write_run_log: failed to ensure tab exists: {exc}")
        return

    now_et = datetime.now(ZoneInfo("America/New_York"))
    row = [
        now_et.strftime("%Y-%m-%d"),
        str(stats.get("started_at", "")),
        str(round(float(stats.get("duration_min", 0)), 1)),
        str(stats.get("venues_processed", "")),
        str(stats.get("venues_with_events", "")),
        str(stats.get("errors", "")),
        str(stats.get("events_before", "")),
        str(stats.get("events_added", "")),
        str(stats.get("events_removed", "")),
        str(stats.get("events_after", "")),
        str(stats.get("status", "success")),
    ]

    try:
        _execute_with_retry(
            service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range=f"'{_RUN_LOG_TAB}'!A1",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": [row]},
            ),
            "append Run Log row",
        )
        print(
            f"  Run log written: {row[0]} — "
            f"+{stats.get('events_added', '?')} added, "
            f"-{stats.get('events_removed', '?')} removed"
        )
    except Exception as exc:
        print(f"  write_run_log: failed to append row: {exc}")


# ─────────────────────────── Dev / test helpers ──────────────────────────────

def test_venue_events_sheet():
    """Smoke-test the Firestore integration."""
    print("Testing Venue Events Firestore integration...")

    events = read_venue_events_from_sheet()
    print(f"\nCurrent data: {len(events)} events")

    by_venue = get_events_by_venue()
    print(f"\nBy venue: {len(by_venue)} venues")
    for venue, venue_events in sorted(by_venue.items()):
        print(f"  {venue}: {len(venue_events)} events")


if __name__ == "__main__":
    test_venue_events_sheet()
