"""Natural-language event filtering for Venue Scout web API."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from utils.llm import generate_content

_TZ = ZoneInfo("America/New_York")


def _parse_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=_TZ)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=_TZ)
        except ValueError:
            return None
    return None


def _event_view(events: list[dict], max_events: int = 150) -> list[dict]:
    rows = []
    for idx, event in enumerate(events[:max_events]):
        dt = _parse_datetime(event.get("datetime"))
        rows.append(
            {
                "index": idx,
                "name": event.get("name", ""),
                "venue_name": event.get("venue_name", ""),
                "event_type": event.get("event_type", ""),
                "date_str": event.get("date_str", ""),
                "datetime": dt.isoformat() if dt else "",
                "description": event.get("description", ""),
            }
        )
    return rows


def _safe_json_obj(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _apply_index_matches(events: list[dict], matched: list[dict], max_results: int) -> list[dict]:
    ranked = sorted(matched, key=lambda item: item.get("score", 0), reverse=True)
    out = []
    for row in ranked:
        idx = row.get("index")
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(events):
            continue
        event = dict(events[idx])
        event["_score"] = row.get("score", 0)
        event["_reason"] = row.get("reason", "")
        out.append(event)
        if len(out) >= max_results:
            break
    return out


def _fallback_filter(query: str, events: list[dict], max_results: int) -> tuple[list[dict], dict]:
    query_l = query.lower()
    now = datetime.now(_TZ)
    saturday = now + timedelta((5 - now.weekday()) % 7)

    filtered = []
    for event in events:
        dt = _parse_datetime(event.get("datetime"))
        venue = (event.get("venue_name", "") + " " + event.get("description", "")).lower()
        score = 50

        if "greenpoint" in query_l and "greenpoint" in venue:
            score += 30

        if "saturday" in query_l and dt and dt.date() == saturday.date():
            score += 20

        if dt is None and "saturday" in query_l:
            score -= 20

        if score >= 50:
            row = dict(event)
            row["_score"] = score
            row["_reason"] = "fallback_rule_match"
            filtered.append(row)

    filtered.sort(key=lambda item: item.get("_score", 0), reverse=True)
    return filtered[:max_results], {
        "mode": "fallback_rules",
        "weekend_target": saturday.date().isoformat(),
    }


def query_events_with_llm(
    query: str,
    events: list[dict],
    max_results: int = 10,
    force_fallback: bool = False,
) -> dict:
    """Rank events for a natural-language query with deterministic fallback."""
    if not events:
        return {
            "interpretation": "No events available to filter.",
            "filters": {},
            "matches": [],
            "warning": "",
            "mode": "empty",
        }

    prompt = f"""You are filtering NYC events for a user query.
Return strict JSON with keys:
- interpretation: string
- filters: object
- matched_events: list of objects with keys index (int), score (0-100), reason (string)

Only include events that truly match the query. Sort by score desc.
Limit to top {max_results}.

QUERY: {query}
NOW: {datetime.now(_TZ).isoformat()}
EVENTS:
{json.dumps(_event_view(events), ensure_ascii=True)}
"""

    warning = ""
    if not force_fallback:
        try:
            response = generate_content(prompt)
            payload = _safe_json_obj(response or "")
            if payload:
                matches = _apply_index_matches(events, payload.get("matched_events", []), max_results)
                return {
                    "interpretation": payload.get("interpretation", "Filtered events for your request."),
                    "filters": payload.get("filters", {}),
                    "matches": matches,
                    "warning": "",
                    "mode": "llm",
                }
            warning = "LLM response could not be parsed. Used fallback rules."
        except Exception as exc:
            warning = f"LLM filter unavailable ({exc}). Used fallback rules."
    else:
        warning = "Forced fallback mode enabled."

    fallback_matches, filters = _fallback_filter(query, events, max_results)
    return {
        "interpretation": "Applied fallback keyword/date filtering.",
        "filters": filters,
        "matches": fallback_matches,
        "warning": warning,
        "mode": "fallback",
    }
