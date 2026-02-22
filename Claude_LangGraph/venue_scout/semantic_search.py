"""Semantic retrieval for Venue Scout events using Gemini embeddings + FAISS."""

from __future__ import annotations

import math
import hashlib
import importlib.util
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

from utils.llm import get_gemini_model
from venue_scout.paths import (
    SEMANTIC_EVENTS_INDEX_FILE,
    SEMANTIC_EVENTS_METADATA_FILE,
    ensure_data_dir,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()
_LOCK = RLock()
_TZ = ZoneInfo("America/New_York")

# Embedding cache for process lifetime to avoid repeated API calls.
_EMBED_CACHE: dict[str, "np.ndarray"] = {}


@dataclass
class _PersistedState:
    """Loaded semantic index state."""

    fingerprint: str
    model_name: str
    event_keys: list[str]
    event_records: list[dict]
    key_to_idx: dict[str, int]
    index: "faiss.Index"
    index_mtime_ns: int
    meta_mtime_ns: int


_STATE: _PersistedState | None = None


def _normalize_name(name: str) -> str:
    text = str(name or "").lower().strip()
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _datetime_token(value) -> str:
    """Serialize event datetime-like values into a stable token."""
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def event_key(event: dict) -> str:
    """Stable key for matching runtime event rows to indexed rows."""
    parts = [
        str(event.get("name", "") or "").strip(),
        str(event.get("venue_name", "") or "").strip(),
        str(event.get("date_str", "") or "").strip(),
        _datetime_token(event.get("datetime")),
        str(event.get("event_type", "") or "").strip(),
        str(event.get("url", "") or "").strip(),
    ]
    return "||".join(parts)


def _parse_event_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value and value != "None":
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None

    return dt if dt.tzinfo else dt.replace(tzinfo=_TZ)


def _event_datetime(event: dict) -> datetime | None:
    dt = _parse_event_datetime(event.get("datetime"))
    if dt:
        return dt

    date_str = str(event.get("date_str", "") or "").strip()
    if not date_str:
        return None

    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=_TZ)


def _event_date_iso(event: dict) -> str:
    dt = _event_datetime(event)
    return dt.date().isoformat() if dt else ""


def _event_datetime_iso(event: dict) -> str:
    dt = _event_datetime(event)
    return dt.isoformat() if dt else ""


def _is_past_event(event: dict, now: datetime) -> bool:
    dt = _event_datetime(event)
    if dt is None:
        return False
    # Date-only events are represented at midnight; keep same-day events.
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.date() < now.date()
    return dt < now


def _is_record_past(record: dict, now: datetime) -> bool:
    dt = _parse_event_datetime(record.get("event_datetime"))
    if dt is not None:
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return dt.date() < now.date()
        return dt < now

    date_str = str(record.get("event_date", "") or "").strip()
    if not date_str:
        return False
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date() < now.date()
    except ValueError:
        return False


def _load_neighborhood_lookup() -> dict[str, str]:
    """Best-effort venue->neighborhood mapping from Venue Scout cache."""
    try:
        from venue_scout.cache import read_cached_venues

        venues = read_cached_venues()
    except Exception:
        return {}

    lookup: dict[str, str] = {}
    for venue in venues:
        key = _normalize_name(venue.get("name", ""))
        neighborhood = str(venue.get("neighborhood", "") or "").strip()
        if key and neighborhood and key not in lookup:
            lookup[key] = neighborhood
    return lookup


def build_event_embedding_text(event: dict, neighborhood_lookup: dict[str, str] | None = None) -> str:
    """Build balanced semantic text for an event (without street address)."""
    neighborhood = str(event.get("neighborhood", "") or "").strip()
    if not neighborhood and neighborhood_lookup is not None:
        neighborhood = neighborhood_lookup.get(_normalize_name(event.get("venue_name", "")), "")

    parts = [
        str(event.get("name", "") or "").strip(),
        f"Type: {str(event.get('event_type', '') or '').strip()}",
        f"Description: {str(event.get('description', '') or '').strip()}",
        f"Venue: {str(event.get('venue_name', '') or '').strip()}",
        f"Neighborhood: {neighborhood}",
        f"Date: {str(event.get('date_str', '') or '').strip()}",
    ]
    text = ". ".join(part for part in parts if part and part != "Description: " and part != "Type: ")
    return text.strip()


def _event_fingerprint(keys: list[str], texts: list[str]) -> str:
    """Stable fingerprint for event set + semantic texts."""
    hasher = hashlib.sha256()
    for key, text in zip(keys, texts):
        hasher.update(key.encode("utf-8", errors="ignore"))
        hasher.update(b"\x1f")
        hasher.update(text.encode("utf-8", errors="ignore"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _text_hash(text: str) -> str:
    """Stable hash for semantic text payloads."""
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def _embedding_model_candidates(preferred: str) -> list[str]:
    """Return ordered embedding model candidates (first is preferred)."""
    candidates = [preferred, "gemini-embedding-001", "text-embedding-004"]
    out: list[str] = []
    seen: set[str] = set()
    for model in candidates:
        name = str(model or "").strip()
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("not_found" in text or "not found" in text) and "embed" in text and "model" in text


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota exceeded" in text


def _retry_delay_seconds(exc: Exception, default_seconds: float = 30.0) -> float:
    """Best-effort retry-delay extraction from provider error text."""
    text = str(exc)
    lower = text.lower()

    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", lower)
    if m:
        return max(1.0, float(m.group(1)))

    m = re.search(r"'retryDelay': '([0-9]+)s'", text)
    if m:
        return max(1.0, float(m.group(1)))

    return max(1.0, float(default_seconds))


def _coerce_positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _estimate_token_count(text: str) -> int:
    """Fast heuristic: ~4 chars per token for English-like text."""
    return max(1, int(math.ceil(len(str(text or "")) / 4.0)))


class _RollingMinuteLimiter:
    """Simple rolling 60s limiter across tokens/contents/requests."""

    def __init__(
        self,
        tokens_per_minute: int | None,
        contents_per_minute: int | None,
        requests_per_minute: int | None,
        safety_margin: float = 0.9,
    ) -> None:
        margin = min(1.0, max(0.1, float(safety_margin)))

        def _apply_margin(limit: int | None) -> int | None:
            if limit is None:
                return None
            return max(1, int(math.floor(limit * margin)))

        self.tokens_per_minute = _apply_margin(_coerce_positive_int(tokens_per_minute))
        self.contents_per_minute = _apply_margin(_coerce_positive_int(contents_per_minute))
        self.requests_per_minute = _apply_margin(_coerce_positive_int(requests_per_minute))

        self._events: deque[tuple[float, int, int, int]] = deque()
        self._tokens = 0
        self._contents = 0
        self._requests = 0

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0][0] <= cutoff:
            _, tokens, contents, requests = self._events.popleft()
            self._tokens -= tokens
            self._contents -= contents
            self._requests -= requests

    @staticmethod
    def _wait_seconds_for_limit(
        now: float,
        events: deque[tuple[float, int, int, int]],
        current: int,
        incoming: int,
        limit: int | None,
        value_idx: int,
    ) -> float:
        if limit is None:
            return 0.0
        if current == 0 and incoming > limit:
            # A single request can exceed configured budget; allow it rather than deadlock.
            return 0.0

        overflow = current + incoming - limit
        if overflow <= 0:
            return 0.0

        released = 0
        for ts, tokens, contents, requests in events:
            values = (tokens, contents, requests)
            released += values[value_idx]
            if released >= overflow:
                return max(0.01, (ts + 60.0) - now)
        return 60.0

    def acquire(self, token_cost: int, content_cost: int, request_cost: int = 1) -> None:
        if (
            self.tokens_per_minute is None
            and self.contents_per_minute is None
            and self.requests_per_minute is None
        ):
            return

        token_cost = max(0, int(token_cost))
        content_cost = max(0, int(content_cost))
        request_cost = max(0, int(request_cost))

        while True:
            now = time.monotonic()
            self._prune(now)

            waits = [
                self._wait_seconds_for_limit(
                    now=now,
                    events=self._events,
                    current=self._tokens,
                    incoming=token_cost,
                    limit=self.tokens_per_minute,
                    value_idx=0,
                ),
                self._wait_seconds_for_limit(
                    now=now,
                    events=self._events,
                    current=self._contents,
                    incoming=content_cost,
                    limit=self.contents_per_minute,
                    value_idx=1,
                ),
                self._wait_seconds_for_limit(
                    now=now,
                    events=self._events,
                    current=self._requests,
                    incoming=request_cost,
                    limit=self.requests_per_minute,
                    value_idx=2,
                ),
            ]
            wait_for = max(waits)
            if wait_for <= 0:
                self._events.append((now, token_cost, content_cost, request_cost))
                self._tokens += token_cost
                self._contents += content_cost
                self._requests += request_cost
                return

            time.sleep(wait_for)


def _embed_texts(
    texts: list[str],
    model_name: str,
    task_type: str,
    batch_size: int,
    allow_model_fallback: bool = True,
    retry_on_rate_limit: bool = False,
    max_rate_limit_retries: int = 20,
) -> tuple["np.ndarray", str]:
    """Embed a list of texts using Gemini embeddings with local memoization."""
    if np is None:
        raise RuntimeError("numpy is not installed")
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), model_name

    client = get_gemini_model()
    limiter = _RollingMinuteLimiter(
        tokens_per_minute=getattr(_settings, "SEMANTIC_EMBED_TOKENS_PER_MINUTE", None),
        contents_per_minute=getattr(_settings, "SEMANTIC_EMBED_CONTENTS_PER_MINUTE", None),
        requests_per_minute=getattr(_settings, "SEMANTIC_EMBED_REQUESTS_PER_MINUTE", None),
        safety_margin=float(getattr(_settings, "SEMANTIC_EMBED_RATE_LIMIT_HEADROOM", 0.9)),
    )
    max_tokens_per_call = _coerce_positive_int(
        getattr(_settings, "SEMANTIC_EMBED_MAX_TOKENS_PER_CALL", None)
    )
    candidates = [model_name]
    if allow_model_fallback:
        candidates = _embedding_model_candidates(model_name)

    last_error: Exception | None = None
    for candidate_model in candidates:
        vectors: list["np.ndarray"] = [None] * len(texts)  # type: ignore[assignment]
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        try:
            with _LOCK:
                for i, text in enumerate(texts):
                    cache_key = f"{candidate_model}|{task_type}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
                    cached = _EMBED_CACHE.get(cache_key)
                    if cached is not None:
                        vectors[i] = cached
                    else:
                        missing_indices.append(i)
                        missing_texts.append(text)
                        missing_keys.append(cache_key)

            for start in range(0, len(missing_texts), batch_size):
                batch_texts = missing_texts[start:start + batch_size]
                batch_indices = missing_indices[start:start + batch_size]
                batch_keys = missing_keys[start:start + batch_size]
                sub_start = 0
                while sub_start < len(batch_texts):
                    if max_tokens_per_call is None:
                        sub_end = len(batch_texts)
                    else:
                        sub_end = sub_start
                        running_tokens = 0
                        while sub_end < len(batch_texts):
                            est = _estimate_token_count(batch_texts[sub_end])
                            if sub_end > sub_start and (running_tokens + est) > max_tokens_per_call:
                                break
                            running_tokens += est
                            sub_end += 1
                            if running_tokens >= max_tokens_per_call:
                                break
                    sub_batch_texts = batch_texts[sub_start:sub_end]
                    sub_batch_indices = batch_indices[sub_start:sub_end]
                    sub_batch_keys = batch_keys[sub_start:sub_end]
                    est_tokens = sum(_estimate_token_count(text) for text in sub_batch_texts)

                    limiter.acquire(
                        token_cost=est_tokens,
                        content_cost=len(sub_batch_texts),
                        request_cost=1,
                    )

                    attempt = 0
                    while True:
                        try:
                            response = client.models.embed_content(
                                model=candidate_model,
                                contents=sub_batch_texts,
                                config={"task_type": task_type},
                            )
                            break
                        except Exception as exc:
                            # Some API paths/SDK versions reject config kwargs; retry bare request.
                            if "parameter is not supported" in str(exc).lower():
                                response = client.models.embed_content(
                                    model=candidate_model,
                                    contents=sub_batch_texts,
                                )
                                break
                            if _is_rate_limit_error(exc) and retry_on_rate_limit and attempt < max_rate_limit_retries:
                                wait_s = _retry_delay_seconds(exc)
                                left = start + sub_start + 1
                                right = start + sub_end
                                print(
                                    f"Rate-limited on embedding batch ({left}-{right}), "
                                    f"retrying in {wait_s:.1f}s...",
                                    flush=True,
                                )
                                time.sleep(wait_s)
                                attempt += 1
                                continue
                            raise

                    embeddings = response.embeddings or []
                    if len(embeddings) != len(sub_batch_texts):
                        raise RuntimeError(
                            f"Embedding count mismatch: expected {len(sub_batch_texts)}, got {len(embeddings)}"
                        )

                    for idx, key, emb in zip(sub_batch_indices, sub_batch_keys, embeddings):
                        values = emb.values or []
                        arr = np.asarray(values, dtype=np.float32)
                        vectors[idx] = arr
                        with _LOCK:
                            _EMBED_CACHE[key] = arr
                    sub_start = sub_end

            if any(vec is None for vec in vectors):
                raise RuntimeError("Missing embeddings after embedding pass")

            dim = len(vectors[0]) if vectors else 0
            if dim == 0:
                raise RuntimeError("Embeddings returned zero dimensions")
            for vec in vectors:
                if len(vec) != dim:
                    raise RuntimeError("Embedding dimensionality mismatch")

            return np.vstack(vectors).astype(np.float32), candidate_model
        except Exception as exc:
            last_error = exc
            if allow_model_fallback and _is_model_not_found_error(exc):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("No embedding model candidates available")


def _read_metadata() -> dict | None:
    if not SEMANTIC_EVENTS_METADATA_FILE.exists():
        return None
    try:
        with open(SEMANTIC_EVENTS_METADATA_FILE) as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def get_semantic_index_membership() -> tuple[set[str], str]:
    """Return indexed event keys and index build timestamp from metadata."""
    metadata = _read_metadata() or {}
    raw_keys = metadata.get("event_keys") or []
    keys = {
        str(key or "").strip()
        for key in raw_keys
        if str(key or "").strip()
    }
    indexed_at = str(metadata.get("built_at", "") or "")
    return keys, indexed_at


def build_semantic_index(
    events: list[dict],
    force: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Build and persist semantic index artifacts for fast query-time retrieval."""
    global _STATE

    if np is None or faiss is None:
        return {"error": "missing_dependencies", "message": "Install numpy and faiss-cpu"}
    if not events:
        return {"error": "no_events"}

    ensure_data_dir()
    embedding_model = str(getattr(_settings, "SEMANTIC_EMBEDDING_MODEL", "gemini-embedding-001"))
    batch = int(batch_size if batch_size is not None else getattr(_settings, "SEMANTIC_EMBED_BATCH_SIZE", 64))
    batch = max(1, batch)

    now = datetime.now(_TZ)
    input_event_count = len(events)
    indexable_events = [event for event in events if not _is_past_event(event, now)]
    skipped_past_count = input_event_count - len(indexable_events)

    if not indexable_events:
        # Remove stale index so query-time code falls back gracefully.
        if SEMANTIC_EVENTS_INDEX_FILE.exists():
            SEMANTIC_EVENTS_INDEX_FILE.unlink()
        metadata = {
            "version": 2,
            "built_at": now.isoformat(),
            "embedding_model": embedding_model,
            "event_count": 0,
            "indexed_event_count": 0,
            "input_event_count": input_event_count,
            "skipped_past_count": skipped_past_count,
            "embedding_dim": 0,
            "fingerprint": "",
            "event_keys": [],
            "event_records": [],
        }
        with open(SEMANTIC_EVENTS_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        with _LOCK:
            _STATE = None

        return {
            "status": "rebuilt_empty",
            "event_count": 0,
            "indexed_event_count": 0,
            "input_event_count": input_event_count,
            "skipped_past_count": skipped_past_count,
            "embedding_model": embedding_model,
            "index_file": str(SEMANTIC_EVENTS_INDEX_FILE),
            "metadata_file": str(SEMANTIC_EVENTS_METADATA_FILE),
            "fingerprint": "",
        }

    neighborhood_lookup = _load_neighborhood_lookup()
    keys = [event_key(event) for event in indexable_events]
    texts = [build_event_embedding_text(event, neighborhood_lookup) for event in indexable_events]
    text_hashes = [_text_hash(text) for text in texts]
    event_records = [
        {
            "key": key,
            "event_date": _event_date_iso(event),
            "event_datetime": _event_datetime_iso(event),
        }
        for key, event in zip(keys, indexable_events)
    ]
    fingerprint = _event_fingerprint(keys, texts)

    previous = _read_metadata() or {}
    if (
        not force
        and SEMANTIC_EVENTS_INDEX_FILE.exists()
        and previous.get("fingerprint") == fingerprint
        and previous.get("event_count") == len(indexable_events)
    ):
        return {
            "status": "unchanged",
            "event_count": len(indexable_events),
            "indexed_event_count": len(indexable_events),
            "input_event_count": input_event_count,
            "skipped_past_count": skipped_past_count,
            "fingerprint": fingerprint,
            "embedding_model": previous.get("embedding_model", embedding_model),
            "index_file": str(SEMANTIC_EVENTS_INDEX_FILE),
            "metadata_file": str(SEMANTIC_EVENTS_METADATA_FILE),
        }

    vectors = None
    resolved_model = embedding_model
    incremental_reused_count = 0
    incremental_embedded_count = 0

    can_incremental_reuse = (
        not force
        and SEMANTIC_EVENTS_INDEX_FILE.exists()
        and str(previous.get("embedding_model", "") or "").strip() == embedding_model
    )

    if can_incremental_reuse:
        previous_event_keys = previous.get("event_keys")
        previous_text_hashes = previous.get("event_text_hashes")
        if (
            isinstance(previous_event_keys, list)
            and isinstance(previous_text_hashes, list)
            and len(previous_event_keys) == len(previous_text_hashes)
            and len(previous_event_keys) > 0
        ):
            prev_state, prev_state_error = _load_persisted_state(force_reload=True)
            if prev_state is not None:
                prev_map: dict[str, tuple[str, int]] = {}
                for idx, (prev_key_raw, prev_hash_raw) in enumerate(zip(previous_event_keys, previous_text_hashes)):
                    prev_key = str(prev_key_raw or "")
                    prev_hash = str(prev_hash_raw or "")
                    if prev_key and prev_hash and prev_key not in prev_map:
                        prev_map[prev_key] = (prev_hash, idx)

                reused_vectors: dict[int, "np.ndarray"] = {}
                missing_indices: list[int] = []
                missing_texts: list[str] = []

                for idx, (key, text_hash, text) in enumerate(zip(keys, text_hashes, texts)):
                    prev_entry = prev_map.get(key)
                    if prev_entry and prev_entry[0] == text_hash:
                        try:
                            reconstructed = prev_state.index.reconstruct(prev_entry[1])
                            reused_vectors[idx] = np.asarray(reconstructed, dtype=np.float32)
                            continue
                        except Exception:
                            pass
                    missing_indices.append(idx)
                    missing_texts.append(text)

                if reused_vectors:
                    dim = len(next(iter(reused_vectors.values())))
                    if missing_texts:
                        missing_vectors, resolved_model = _embed_texts(
                            texts=missing_texts,
                            model_name=embedding_model,
                            task_type="RETRIEVAL_DOCUMENT",
                            batch_size=batch,
                            allow_model_fallback=False,
                            retry_on_rate_limit=True,
                        )
                        if missing_vectors.shape[1] != dim:
                            raise RuntimeError(
                                "Incremental semantic rebuild dimension mismatch; "
                                f"reused={dim}, new={missing_vectors.shape[1]}"
                            )
                    else:
                        missing_vectors = np.zeros((0, dim), dtype=np.float32)
                        resolved_model = embedding_model

                    vectors = np.zeros((len(indexable_events), dim), dtype=np.float32)
                    for idx, vec in reused_vectors.items():
                        vectors[idx] = vec
                    for local_idx, global_idx in enumerate(missing_indices):
                        vectors[global_idx] = missing_vectors[local_idx]

                    incremental_reused_count = len(reused_vectors)
                    incremental_embedded_count = len(missing_indices)
            elif prev_state_error:
                # Fall through to full rebuild when persisted state can't be loaded.
                pass

    if vectors is None:
        vectors, resolved_model = _embed_texts(
            texts=texts,
            model_name=embedding_model,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=batch,
            allow_model_fallback=True,
            retry_on_rate_limit=True,
        )
        incremental_reused_count = 0
        incremental_embedded_count = len(texts)

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(SEMANTIC_EVENTS_INDEX_FILE))

    metadata = {
        "version": 2,
        "built_at": datetime.now(_TZ).isoformat(),
        "embedding_model": resolved_model,
        "event_count": len(indexable_events),
        "indexed_event_count": len(indexable_events),
        "input_event_count": input_event_count,
        "skipped_past_count": skipped_past_count,
        "embedding_dim": int(vectors.shape[1]),
        "fingerprint": fingerprint,
        "event_keys": keys,
        "event_text_hashes": text_hashes,
        "event_records": event_records,
    }
    with open(SEMANTIC_EVENTS_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    with _LOCK:
        _STATE = None

    return {
        "status": "rebuilt",
        "event_count": len(indexable_events),
        "indexed_event_count": len(indexable_events),
        "input_event_count": input_event_count,
        "skipped_past_count": skipped_past_count,
        "embedding_model": resolved_model,
        "embedding_dim": int(vectors.shape[1]),
        "index_file": str(SEMANTIC_EVENTS_INDEX_FILE),
        "metadata_file": str(SEMANTIC_EVENTS_METADATA_FILE),
        "fingerprint": fingerprint,
        "incremental_reused_count": incremental_reused_count,
        "incremental_embedded_count": incremental_embedded_count,
    }


def _load_persisted_state(force_reload: bool = False) -> tuple[_PersistedState | None, str]:
    """Load persisted FAISS index + metadata into memory."""
    if np is None or faiss is None:
        return None, "missing_dependencies"

    if not SEMANTIC_EVENTS_INDEX_FILE.exists() or not SEMANTIC_EVENTS_METADATA_FILE.exists():
        return None, "missing_index"

    index_mtime_ns = SEMANTIC_EVENTS_INDEX_FILE.stat().st_mtime_ns
    meta_mtime_ns = SEMANTIC_EVENTS_METADATA_FILE.stat().st_mtime_ns

    global _STATE
    with _LOCK:
        if (
            not force_reload
            and _STATE is not None
            and _STATE.index_mtime_ns == index_mtime_ns
            and _STATE.meta_mtime_ns == meta_mtime_ns
        ):
            return _STATE, ""

    metadata = _read_metadata()
    if not metadata:
        return None, "bad_metadata"

    event_keys = metadata.get("event_keys")
    if not isinstance(event_keys, list) or not event_keys:
        return None, "missing_event_keys"

    event_records_raw = metadata.get("event_records")
    if (
        isinstance(event_records_raw, list)
        and len(event_records_raw) == len(event_keys)
        and all(isinstance(record, dict) for record in event_records_raw)
    ):
        event_records = list(event_records_raw)
    else:
        # Backward compatibility for older metadata without per-event records.
        event_records = [{"key": str(key or ""), "event_date": "", "event_datetime": ""} for key in event_keys]

    try:
        index = faiss.read_index(str(SEMANTIC_EVENTS_INDEX_FILE))
    except Exception as exc:
        return None, f"read_index_failed:{exc}"

    if int(index.ntotal) != len(event_keys):
        return None, "index_key_count_mismatch"

    key_to_idx: dict[str, int] = {}
    for idx, key in enumerate(event_keys):
        key_text = str(key or "")
        if key_text and key_text not in key_to_idx:
            key_to_idx[key_text] = idx

    state = _PersistedState(
        fingerprint=str(metadata.get("fingerprint", "")),
        model_name=str(metadata.get("embedding_model", "gemini-embedding-001")),
        event_keys=[str(key or "") for key in event_keys],
        event_records=event_records,
        key_to_idx=key_to_idx,
        index=index,
        index_mtime_ns=index_mtime_ns,
        meta_mtime_ns=meta_mtime_ns,
    )
    with _LOCK:
        _STATE = state
    return state, ""


def _lexical_score(query: str, event: dict) -> float:
    raw_tokens = [tok for tok in re.findall(r"[a-z0-9]+", query.lower()) if len(tok) > 2]
    if not raw_tokens:
        return 0.0

    expanded: set[str] = set()
    for token in raw_tokens:
        expanded.add(token)
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        if token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])

    if any(token in expanded for token in ("talk", "lecture", "panel", "seminar", "discussion", "workshop")):
        expanded.update({"talk", "lecture", "panel", "seminar", "discussion", "workshop", "presentation"})

    name = str(event.get("name", "") or "").lower()
    event_type = str(event.get("event_type", "") or "").lower()
    description = str(event.get("description", "") or "").lower()
    venue = str(event.get("venue_name", "") or "").lower()
    date_str = str(event.get("date_str", "") or "").lower()

    score = 0.0
    for token in expanded:
        if token in name:
            score += 3.0
        if token in event_type:
            score += 2.5
        if token in description:
            score += 1.5
        if token in venue:
            score += 0.5
        if token in date_str:
            score += 0.5

    return score


def lexical_rank_events(query: str, events: list[dict], top_k: int) -> list[dict]:
    """Deterministic lexical ranking fallback when semantic retrieval is unavailable."""
    ranked = sorted(events, key=lambda event: _lexical_score(query, event), reverse=True)
    return ranked[:max(1, min(top_k, len(ranked)))] if ranked else []


def retrieve_semantic_candidates(
    query: str,
    events: list[dict],
    top_k: int = 250,
) -> tuple[list[dict], dict, str]:
    """
    Retrieve top-K semantic candidates for query.

    Uses persisted index built by `build-semantic-index`.
    Returns:
        candidates, metadata, warning
    """
    if not events:
        return [], {"semantic_applied": False, "semantic_pool_size": 0}, ""

    top_k = max(1, min(int(top_k), len(events)))
    if not bool(getattr(_settings, "SEMANTIC_RETRIEVAL_ENABLED", True)):
        lexical = lexical_rank_events(query, events, top_k)
        return lexical, {
            "semantic_applied": False,
            "semantic_pool_size": len(events),
            "semantic_candidate_count": len(lexical),
            "semantic_mode": "lexical_disabled",
        }, ""

    if np is None or faiss is None:
        lexical = lexical_rank_events(query, events, top_k)
        return lexical, {
            "semantic_applied": False,
            "semantic_pool_size": len(events),
            "semantic_candidate_count": len(lexical),
            "semantic_mode": "lexical_missing_dep",
        }, "FAISS/numpy not available; used lexical retrieval."

    state, state_error = _load_persisted_state()
    if state is None:
        lexical = lexical_rank_events(query, events, top_k)
        warning = (
            "Semantic index unavailable "
            f"({state_error}); used lexical retrieval. Run `python -m venue_scout.cli build-semantic-index`."
        )
        return lexical, {
            "semantic_applied": False,
            "semantic_pool_size": len(events),
            "semantic_candidate_count": len(lexical),
            "semantic_mode": "lexical_missing_index",
        }, warning

    now = datetime.now(_TZ)
    allowed_by_idx: dict[int, dict] = {}
    skipped_past_index_rows = 0
    for event in events:
        idx = state.key_to_idx.get(event_key(event))
        if idx is None or idx in allowed_by_idx:
            continue

        if 0 <= idx < len(state.event_records) and _is_record_past(state.event_records[idx], now):
            skipped_past_index_rows += 1
            continue

        allowed_by_idx[idx] = event

    coverage = float(len(allowed_by_idx)) / float(max(1, len(events)))
    if not allowed_by_idx:
        lexical = lexical_rank_events(query, events, top_k)
        warning = (
            "Semantic index has no active overlap with current event set; used lexical retrieval."
        )
        return lexical, {
            "semantic_applied": False,
            "semantic_pool_size": len(events),
            "semantic_candidate_count": len(lexical),
            "semantic_mode": "lexical_zero_coverage",
            "semantic_index_coverage": coverage,
            "semantic_skipped_past_index_rows": skipped_past_index_rows,
        }, warning

    try:
        query_vec, resolved_model = _embed_texts(
            texts=[query],
            model_name=state.model_name,
            task_type="RETRIEVAL_QUERY",
            batch_size=1,
            allow_model_fallback=False,
        )
        if resolved_model != state.model_name:
            raise RuntimeError(
                f"query embedding model mismatch ({resolved_model} != {state.model_name})"
            )
        faiss.normalize_L2(query_vec)

        search_k = min(int(state.index.ntotal), max(top_k * 8, 400))
        scores, indices = state.index.search(query_vec, search_k)

        ranked_candidates: list[dict] = []
        seen: set[int] = set()
        for score, idx in zip(scores[0], indices[0]):
            idx_int = int(idx)
            if idx_int < 0 or idx_int in seen:
                continue
            if idx_int not in allowed_by_idx:
                continue
            seen.add(idx_int)
            event = dict(allowed_by_idx[idx_int])
            event["_semantic_score"] = float(score)
            ranked_candidates.append(event)
            if len(ranked_candidates) >= top_k:
                break

        if not ranked_candidates:
            lexical = lexical_rank_events(query, events, top_k)
            warning = "Semantic retrieval returned no candidates; used lexical retrieval."
            return lexical, {
                "semantic_applied": False,
                "semantic_pool_size": len(events),
                "semantic_candidate_count": len(lexical),
                "semantic_mode": "lexical_empty_semantic",
                "semantic_index_coverage": coverage,
            }, warning

        warning = ""
        if coverage < 0.85:
            warning = (
                f"Semantic index coverage is low ({coverage:.1%}); "
                "results may be incomplete until index rebuild."
            )

        metadata = {
            "semantic_applied": True,
            "semantic_pool_size": len(events),
            "semantic_candidate_count": len(ranked_candidates),
            "semantic_top_k": top_k,
            "semantic_embedding_model": state.model_name,
            "semantic_mode": "faiss_persisted",
            "semantic_index_coverage": coverage,
            "semantic_skipped_past_index_rows": skipped_past_index_rows,
            "semantic_score_min": float(min(event["_semantic_score"] for event in ranked_candidates)),
            "semantic_score_max": float(max(event["_semantic_score"] for event in ranked_candidates)),
        }
        return ranked_candidates, metadata, warning
    except Exception as exc:
        lexical = lexical_rank_events(query, events, top_k)
        if _is_rate_limit_error(exc):
            warning = "Semantic retrieval rate-limited; used lexical retrieval."
            mode = "lexical_rate_limited"
        else:
            warning = f"Semantic retrieval failed ({exc}); used lexical retrieval."
            mode = "lexical_error"
        return lexical, {
            "semantic_applied": False,
            "semantic_pool_size": len(events),
            "semantic_candidate_count": len(lexical),
            "semantic_mode": mode,
            "semantic_index_coverage": coverage,
            "semantic_skipped_past_index_rows": skipped_past_index_rows,
        }, warning
