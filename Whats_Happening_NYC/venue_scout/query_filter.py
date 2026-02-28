"""Natural-language event filtering for Venue Scout web API."""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from google.genai import types

from utils.llm import generate_content, get_gemini_model
from venue_scout.semantic_search import lexical_rank_events, retrieve_semantic_candidates

_TZ = ZoneInfo("America/New_York")
_DATE_TOOL_NAME = "filter_events_by_date"


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


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


def _event_datetime(event: dict) -> datetime | None:
    """Parse event datetime, falling back to date_str when needed."""
    dt = _parse_datetime(event.get("datetime"))
    if dt:
        return dt
    date_str = str(event.get("date_str", "") or "").strip()
    if not date_str:
        return None
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed_date.replace(tzinfo=_TZ)


def _parse_iso_date(value: str | None) -> datetime | None:
    """Parse YYYY-MM-DD into a timezone-aware datetime at midnight."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None
    return dt.replace(tzinfo=_TZ)


def _day_end(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def _window_from_relative(relative_window: str, now: datetime) -> tuple[datetime | None, datetime | None]:
    """Convert a relative date window string into absolute start/end datetimes."""
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = _day_end(today_start)
    key = str(relative_window or "").strip().lower()

    if key == "today":
        return today_start, today_end
    if key == "tomorrow":
        start = today_start + timedelta(days=1)
        return start, _day_end(start)
    if key == "this_week":
        end_of_week = today_start + timedelta(days=(6 - today_start.weekday()))
        return today_start, _day_end(end_of_week)
    if key == "next_week":
        days_to_next_monday = (7 - today_start.weekday()) % 7
        if days_to_next_monday == 0:
            days_to_next_monday = 7
        start = today_start + timedelta(days=days_to_next_monday)
        end = start + timedelta(days=6)
        return start, _day_end(end)
    if key == "this_weekend":
        days_to_saturday = (5 - today_start.weekday()) % 7
        saturday = today_start + timedelta(days=days_to_saturday)
        sunday = saturday + timedelta(days=1)
        return saturday, _day_end(sunday)
    if key == "next_30_days":
        return now, now + timedelta(days=30)

    return None, None


def _build_date_window_from_args(args: dict, now: datetime) -> tuple[datetime | None, datetime | None]:
    """Build absolute datetime window from tool arguments."""
    if not isinstance(args, dict):
        return None, None

    start = _parse_iso_date(args.get("start_date"))
    end = _parse_iso_date(args.get("end_date"))

    if start and end:
        return start, _day_end(end)

    relative_start, relative_end = _window_from_relative(str(args.get("relative_window", "") or ""), now)
    if relative_start and relative_end:
        return relative_start, relative_end

    days_ahead = args.get("days_ahead")
    if days_ahead not in (None, ""):
        try:
            days_ahead_int = int(days_ahead)
            if days_ahead_int > 0:
                return now, now + timedelta(days=days_ahead_int)
        except (TypeError, ValueError):
            pass

    if start and not end:
        return start, _day_end(start)
    if end and not start:
        day_start = end.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start, _day_end(end)

    return None, None


def _filter_events_by_date_window(
    events: list[dict],
    start: datetime | None,
    end: datetime | None,
    include_undated: bool,
) -> list[dict]:
    """Apply inclusive date filtering window."""
    if not start or not end:
        return events

    if end < start:
        start, end = end, start

    out = []
    for event in events:
        dt = _event_datetime(event)
        if dt is None:
            if include_undated:
                out.append(event)
            continue
        if start <= dt <= end:
            out.append(event)
    return out


def _date_tool_declaration() -> types.Tool:
    """Gemini function declaration for deterministic date filtering."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=_DATE_TOOL_NAME,
                description=(
                    "Extract concrete date constraints from a user query for event filtering. "
                    "Use explicit dates if present. Use relative_window for phrases like next week."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "relative_window": {
                            "type": "string",
                            "enum": [
                                "none",
                                "today",
                                "tomorrow",
                                "this_week",
                                "this_weekend",
                                "next_week",
                                "next_30_days",
                            ],
                        },
                        "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "days_ahead": {"type": "integer"},
                        "include_undated": {"type": "boolean"},
                    },
                },
            )
        ]
    )


def _apply_date_tool(query: str, events: list[dict]) -> tuple[list[dict], dict, str]:
    """
    Ask Gemini to call date-filter tool, execute locally, and return filtered events.

    Returns:
        (filtered_events, metadata, warning_message)
    """
    if not events:
        return events, {}, ""

    date_tool_timeout_sec = int(getattr(_settings, "QUERY_DATE_TOOL_TIMEOUT_SEC", 10))
    client = get_gemini_model(timeout_sec=date_tool_timeout_sec)
    model_name = str(
        getattr(
            _settings,
            "QUERY_DATE_TOOL_MODEL",
            getattr(_settings, "GEMINI_MODEL", "gemini-3-flash-preview"),
        )
    )
    seed = int(getattr(_settings, "GEMINI_SEED", 20260213))
    now = datetime.now(_TZ)

    tool_prompt = (
        "Determine whether the user query has date constraints.\n"
        f"Current datetime: {now.isoformat()}\n"
        "If no date constraint exists, do not call any function.\n"
        f"User query: {query}"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=tool_prompt,
            config={
                "temperature": 0,
                "seed": seed,
                "tools": [_date_tool_declaration()],
                "automatic_function_calling": {"disable": True},
            },
        )
    except Exception as exc:
        return events, {}, f"Date tool unavailable ({exc})."

    function_calls = response.function_calls or []
    if not function_calls:
        return events, {"date_tool_called": False}, ""

    call = function_calls[0]
    if call.name != _DATE_TOOL_NAME:
        return events, {"date_tool_called": False}, ""

    args = call.args or {}
    start, end = _build_date_window_from_args(args, now)
    include_undated = bool(args.get("include_undated", False))
    if not start or not end:
        return events, {"date_tool_called": True, "date_window_applied": False}, ""

    filtered = _filter_events_by_date_window(events, start, end, include_undated)
    metadata = {
        "date_tool_called": True,
        "date_window_applied": True,
        "date_window_start": start.isoformat(),
        "date_window_end": end.isoformat(),
        # Backward-compatible aliases for older clients/scripts.
        "date_start": start.date().isoformat(),
        "date_end": end.date().isoformat(),
        "include_undated": include_undated,
        "pre_date_count": len(events),
        "post_date_count": len(filtered),
        "pre_filter_count": len(events),
        "post_filter_count": len(filtered),
    }
    return filtered, metadata, ""


def _event_view(events: list[dict], max_events: int = 150) -> list[dict]:
    rows = []
    for idx, event in enumerate(events[:max_events]):
        dt = _event_datetime(event)
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


def _is_timeout_error(exc: Exception) -> bool:
    """Return True when exception text indicates provider timeout/deadline."""
    lowered = str(exc).lower()
    timeout_tokens = (
        "504",
        "deadline_exceeded",
        "deadline exceeded",
        "deadline expired",
        "timeout",
        "timed out",
    )
    return any(token in lowered for token in timeout_tokens)


def _fallback_model_list(value, default: tuple[str, ...]) -> list[str]:
    """Normalize fallback model setting into an ordered list."""
    raw = value if value is not None else default
    if isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw]
    else:
        text = str(raw or "").strip()
        items = [part.strip() for part in text.split(",")] if text else []
    return [item for item in items if item]


def _query_specificity_score(query: str) -> tuple[int, dict]:
    """
    Deterministic query specificity score to decide semantic retrieval usage.

    Higher score means more constrained/specific query.
    """
    text = str(query or "").strip().lower()
    tokens = re.findall(r"[a-z0-9$]+", text)
    score = 0
    signals: dict[str, list[str] | int] = {
        "category_terms": [],
        "location_terms": [],
        "price_terms": [],
        "time_terms": [],
        "audience_terms": [],
        "vague_terms": [],
        "token_count": len(tokens),
    }

    def _collect(haystack: str, phrases: tuple[str, ...]) -> list[str]:
        return [phrase for phrase in phrases if phrase in haystack]

    category_terms = _collect(
        text,
        (
            "music", "concert", "live music", "jazz", "classical", "opera",
            "comedy", "standup", "improv", "talk", "lecture", "panel",
            "workshop", "film", "movie", "art", "gallery", "museum",
            "dance", "theater", "theatre", "book", "poetry",
        ),
    )
    if category_terms:
        score += 2
        signals["category_terms"] = category_terms[:8]

    location_terms = _collect(
        text,
        (
            "brooklyn", "manhattan", "queens", "bronx", "staten island",
            "harlem", "greenpoint", "williamsburg", "soho", "chelsea",
            "lower east side", "upper west side", "upper east side",
            "midtown", "downtown",
        ),
    )
    if location_terms or re.search(r"\bin\s+[a-z]{3,}", text):
        score += 2
        signals["location_terms"] = location_terms[:8]

    price_terms = _collect(
        text,
        ("free", "cheap", "under $", "under ", "budget", "no cover"),
    )
    if price_terms or "$" in text:
        score += 1
        signals["price_terms"] = price_terms[:8]

    time_terms = _collect(
        text,
        (
            "today", "tomorrow", "tonight", "this weekend", "next weekend",
            "next week", "this week", "before the end of the month",
            "after", "before", "am", "pm",
        ),
    )
    if time_terms:
        score += 1
        signals["time_terms"] = time_terms[:8]

    audience_terms = _collect(
        text,
        (
            "kid-friendly", "family", "outdoor", "indoor", "date night",
            "beginner", "advanced", "networking", "meetup",
        ),
    )
    if audience_terms:
        score += 1
        signals["audience_terms"] = audience_terms[:8]

    vague_terms = _collect(
        text,
        (
            "what should i attend", "what should i do", "anything", "something",
            "interesting", "free time", "looking for ideas", "surprise me",
        ),
    )
    if vague_terms:
        score -= 2
        signals["vague_terms"] = vague_terms[:8]

    if len(tokens) <= 5:
        score -= 1

    return score, signals


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
        dt = _event_datetime(event)
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


def _lexical_fallback_response(
    query: str,
    ranked_events: list[dict],
    max_results: int,
    warning: str,
    date_filters: dict,
    semantic_filters: dict,
) -> dict:
    """Build response payload when LLM ranking fails and lexical fallback is used."""
    lexical_matches = lexical_rank_events(query, ranked_events, max_results)
    for event in lexical_matches:
        event["_reason"] = event.get("_reason") or "lexical_fallback_llm_unavailable"

    filters = {"mode": "lexical_fallback"}
    if date_filters:
        filters["date"] = date_filters
    if semantic_filters:
        filters["semantic"] = semantic_filters

    return {
        "interpretation": "Applied lexical fallback filtering.",
        "filters": filters,
        "matches": lexical_matches,
        "warning": warning,
        "mode": "lexical_fallback",
    }


def query_events_with_llm(
    query: str,
    events: list[dict],
    max_results: int = 10,
    force_fallback: bool = False,
    context: str = "",
    history: str = "",
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

    date_filters: dict = {}
    semantic_filters: dict = {}
    warning = ""
    ranked_events = events

    if not force_fallback:
        ranked_events, date_filters, date_warning = _apply_date_tool(query, events)
        if date_warning:
            warning = date_warning
        if date_filters.get("date_window_applied") and not ranked_events:
            return {
                "interpretation": "No events found in the requested date range.",
                "filters": {"date": date_filters},
                "matches": [],
                "warning": warning,
                "mode": "date_tool_empty",
            }
        semantic_top_k = int(getattr(_settings, "SEMANTIC_TOP_K", 250))
        semantic_top_k = max(10, semantic_top_k)
        specificity_score, specificity_signals = _query_specificity_score(query)
        specificity_threshold = int(getattr(_settings, "QUERY_SEMANTIC_SPECIFICITY_THRESHOLD", 2))
        skip_broad_semantic = bool(getattr(_settings, "QUERY_SEMANTIC_SKIP_BROAD", True))
        use_semantic = (not skip_broad_semantic) or (specificity_score >= specificity_threshold)

        if use_semantic:
            semantic_candidates, semantic_meta, semantic_warning = retrieve_semantic_candidates(
                query=query,
                events=ranked_events,
                top_k=min(semantic_top_k, len(ranked_events)),
            )
            if semantic_candidates:
                ranked_events = semantic_candidates
            semantic_filters = semantic_meta or {}
            semantic_filters["semantic_specificity_score"] = specificity_score
            semantic_filters["semantic_specificity_threshold"] = specificity_threshold
            if semantic_warning:
                warning = f"{warning} {semantic_warning}".strip()
        else:
            broad_top_k = int(getattr(_settings, "QUERY_BROAD_LEXICAL_TOP_K", 120))
            broad_top_k = max(25, broad_top_k)
            lexical_candidates = lexical_rank_events(
                query,
                ranked_events,
                min(broad_top_k, len(ranked_events)),
            )
            if lexical_candidates:
                ranked_events = lexical_candidates
            semantic_filters = {
                "semantic_applied": False,
                "semantic_pool_size": len(events),
                "semantic_candidate_count": len(ranked_events),
                "semantic_mode": "skipped_broad_query",
                "semantic_specificity_score": specificity_score,
                "semantic_specificity_threshold": specificity_threshold,
                "semantic_specificity_signals": specificity_signals,
            }

    llm_event_context_limit = int(getattr(_settings, "LLM_EVENT_CONTEXT_LIMIT", 250))
    llm_event_context_limit = max(25, llm_event_context_limit)

    context_section = f"\nFOLLOW-UP ANSWER: {context}" if context else ""
    history_section = f"\nPREVIOUS SEARCHES:\n{history}" if history else ""
    no_date_window = not date_filters.get("date_window_applied")
    near_term_hint = (
        "\nSCORING: No time window was specified. When two events are otherwise equally relevant,"
        " prefer the one happening sooner — events within the next 30 days should rank slightly"
        " above events many months away."
    ) if no_date_window else ""
    prompt = f"""You are filtering NYC events for a user query.
Return strict JSON with keys:
- interpretation: string (brief description of what you found, or what you are asking)
- filters: object
- matched_events: list of objects with keys index (int), score (0-100), reason (string)
- follow_up_question: string

FOLLOW-UP QUESTION RULES:
- Set follow_up_question to a short clarifying question ONLY when ALL of these are true:
  1. The query is intentionally vague or open-ended (e.g. "date spot", "fun night out", "something to do", "good show")
  2. A single question would meaningfully narrow the results (e.g. asking about vibe, activity type, or budget)
  3. No context has already been provided
- When follow_up_question is non-empty, set matched_events to [] — the user will answer before seeing results
- If the user said "I don't know", "anything", "surprise me", or provided any preference context, set follow_up_question to ""
- For specific queries (specific genre, artist, venue, or date), set follow_up_question to ""
- Default: follow_up_question is ""
{near_term_hint}
Only include events that truly match the query. Sort by score desc.
Limit to top {max_results}.

QUERY: {query}{context_section}{history_section}
NOW: {datetime.now(_TZ).isoformat()}
EVENTS:
{json.dumps(_event_view(ranked_events, max_events=min(llm_event_context_limit, len(ranked_events))), ensure_ascii=True)}
"""

    if not force_fallback:
        try:
            query_max_retries = int(getattr(_settings, "QUERY_LLM_MAX_RETRIES", 1))
            query_timeout_sec = int(getattr(_settings, "QUERY_LLM_TIMEOUT_SEC", 12))
            primary_model = str(
                getattr(
                    _settings,
                    "QUERY_RANKING_MODEL",
                    getattr(_settings, "GEMINI_MODEL", "gemini-3-flash-preview"),
                )
            )
            timeout_fallback_models = _fallback_model_list(
                getattr(
                    _settings,
                    "QUERY_LLM_TIMEOUT_FALLBACK_MODELS",
                    ("gemini-2.5-flash", "gemini-2.0-flash"),
                ),
                default=("gemini-2.5-flash", "gemini-2.0-flash"),
            )
            timeout_fallback_models = [m for m in timeout_fallback_models if m != primary_model]
            response = ""
            try:
                response = generate_content(
                    prompt,
                    max_retries=max(1, query_max_retries),
                    timeout_sec=max(5, query_timeout_sec),
                    model_name=primary_model,
                )
            except Exception as primary_exc:
                if not _is_timeout_error(primary_exc):
                    raise
                fallback_error: Exception = primary_exc
                for fallback_model in timeout_fallback_models:
                    try:
                        response = generate_content(
                            prompt,
                            max_retries=1,
                            timeout_sec=max(5, query_timeout_sec),
                            model_name=fallback_model,
                        )
                        warning = (
                            f"{warning} Primary ranking model timed out; retried with {fallback_model}."
                        ).strip()
                        break
                    except Exception as exc:
                        fallback_error = exc
                if not response:
                    raise fallback_error
            payload = _safe_json_obj(response or "")
            if payload:
                follow_up_question = str(payload.get("follow_up_question", "") or "").strip()
                # If the user already answered a clarifying question, never ask again
                if context:
                    follow_up_question = ""
                # If LLM wants to ask a follow-up, return immediately with no matches
                if follow_up_question:
                    merged_filters = payload.get("filters", {})
                    if date_filters:
                        merged_filters["date"] = date_filters
                    if semantic_filters:
                        merged_filters["semantic"] = semantic_filters
                    return {
                        "interpretation": payload.get("interpretation", ""),
                        "filters": merged_filters,
                        "matches": [],
                        "warning": warning,
                        "mode": "follow_up",
                        "follow_up_question": follow_up_question,
                    }
                matches = _apply_index_matches(ranked_events, payload.get("matched_events", []), max_results)
                merged_filters = payload.get("filters", {})
                if date_filters:
                    merged_filters["date"] = date_filters
                if semantic_filters:
                    merged_filters["semantic"] = semantic_filters
                if not matches and ranked_events:
                    matches = lexical_rank_events(query, ranked_events, max_results)
                    for event in matches:
                        event["_reason"] = event.get("_reason") or "lexical_fallback_after_llm_empty"
                    warning = f"{warning} LLM returned no matches; used lexical fallback.".strip()
                    return {
                        "interpretation": payload.get("interpretation", "Filtered events for your request."),
                        "filters": merged_filters,
                        "matches": matches,
                        "warning": warning,
                        "mode": "llm_empty_lexical_fallback",
                    }
                return {
                    "interpretation": payload.get("interpretation", "Filtered events for your request."),
                    "filters": merged_filters,
                    "matches": matches,
                    "warning": warning,
                    "mode": "llm",
                }
            extra = "LLM response could not be parsed. Used fallback rules."
            warning = f"{warning} {extra}".strip()
            return _lexical_fallback_response(
                query=query,
                ranked_events=ranked_events,
                max_results=max_results,
                warning=warning,
                date_filters=date_filters,
                semantic_filters=semantic_filters,
            )
        except Exception as exc:
            extra = f"LLM filter unavailable ({exc}). Used lexical fallback."
            warning = f"{warning} {extra}".strip()
            return _lexical_fallback_response(
                query=query,
                ranked_events=ranked_events,
                max_results=max_results,
                warning=warning,
                date_filters=date_filters,
                semantic_filters=semantic_filters,
            )
    else:
        warning = "Forced fallback mode enabled."

    fallback_matches, filters = _fallback_filter(query, ranked_events, max_results)
    if date_filters:
        filters["date"] = date_filters
    if semantic_filters:
        filters["semantic"] = semantic_filters
    return {
        "interpretation": "Applied fallback keyword/date filtering.",
        "filters": filters,
        "matches": fallback_matches,
        "warning": warning,
        "mode": "fallback",
    }
