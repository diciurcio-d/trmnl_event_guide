"""Use Gemini to semantically match user queries to events."""

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import generate_content
from utils.distance import get_max_travel_minutes
from utils.calendar import has_conflict, is_calendar_authenticated


QUERY_MATCHING_PROMPT = """You are helping a user find events in NYC. Analyze their query and the available events to find the best matches.

USER QUERY: "{query}"

TODAY'S DATE: {today}

AVAILABLE EVENTS (showing first {num_events}):
{events_json}

Your task:
1. Understand what the user is looking for (topic, mood, type of experience, time constraints)
2. Score each event from 0-100 based on how well it matches the query
3. Consider:
   - Semantic relevance (a "fun night out" could match comedy, music, social events)
   - Date/time preferences mentioned in query
   - Location preferences
   - Event type and description

Return ONLY a valid JSON object with this structure:
{{
  "interpretation": "Brief explanation of what you think the user wants",
  "date_filter": {{
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "weekend_only": true/false
  }},
  "matched_events": [
    {{"index": 0, "score": 85, "reason": "Why this matches"}},
    {{"index": 2, "score": 72, "reason": "Why this matches"}}
  ]
}}

Only include events with score >= 50. Sort by score descending. Limit to top 5 matches.
Return matched_events as empty array [] if nothing matches well.
"""


def match_events_to_query(query: str, events: list[dict], max_events_to_analyze: int = 100) -> dict:
    """
    Use Gemini to semantically match events to a user query.

    Args:
        query: The user's natural language query
        events: List of event dicts
        max_events_to_analyze: Max events to send to LLM (for token limits)

    Returns:
        Dict with interpretation, date_filter, and matched event indices with scores
    """
    now = datetime.now(ZoneInfo("America/New_York"))

    # Prepare events for LLM (simplified format, limited count)
    events_for_llm = []
    for i, event in enumerate(events[:max_events_to_analyze]):
        # Skip events without datetime or in the past
        if not event.get("datetime"):
            continue
        if event["datetime"] < now:
            continue

        events_for_llm.append({
            "index": i,
            "name": event.get("name", ""),
            "date": event["datetime"].strftime("%A, %B %d, %Y"),
            "time": event["datetime"].strftime("%I:%M %p") if event.get("has_specific_time") else "TBD",
            "type": event.get("type", ""),
            "source": event.get("source", ""),
            "location": event.get("location", ""),
            "description": event.get("description", ""),
            "sold_out": event.get("sold_out", False),
        })

    if not events_for_llm:
        return {
            "interpretation": "No upcoming events available",
            "date_filter": {"start_date": None, "end_date": None, "weekend_only": False},
            "matched_events": [],
        }

    prompt = QUERY_MATCHING_PROMPT.format(
        query=query,
        today=now.strftime("%A, %B %d, %Y"),
        num_events=len(events_for_llm),
        events_json=json.dumps(events_for_llm, indent=2),
    )

    try:
        response_text = generate_content(prompt).strip()

        # Parse JSON response
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Try to find JSON in response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # Fallback if parsing fails
        return {
            "interpretation": "Could not parse query",
            "date_filter": {"start_date": None, "end_date": None, "weekend_only": False},
            "matched_events": [],
        }

    except Exception as e:
        print(f"    LLM query matching error: {e}")
        return {
            "interpretation": f"Error: {str(e)}",
            "date_filter": {"start_date": None, "end_date": None, "weekend_only": False},
            "matched_events": [],
        }


def filter_events_by_llm_result(events: list[dict], llm_result: dict) -> list[dict]:
    """
    Filter and sort events based on LLM matching result.

    Args:
        events: Original list of events
        llm_result: Result from match_events_to_query

    Returns:
        Filtered and sorted list of events
    """
    matched = llm_result.get("matched_events", [])

    if not matched:
        return []

    # Sort by score descending
    matched.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Get events by index
    result = []
    for match in matched:
        idx = match.get("index", -1)
        if 0 <= idx < len(events):
            event = events[idx].copy()
            event["_match_score"] = match.get("score", 0)
            event["_match_reason"] = match.get("reason", "")
            result.append(event)

    return result


def apply_user_filters(
    events: list[dict],
    max_travel_minutes: int | None = None,
    exclude_conflicts: bool = True,
) -> list[dict]:
    """
    Apply user preference filters to events.

    Args:
        events: List of events (already matched/scored)
        max_travel_minutes: Max travel time filter (None = use config default)
        exclude_conflicts: Whether to exclude events with calendar conflicts

    Returns:
        Filtered list of events with conflict info added
    """
    if max_travel_minutes is None:
        max_travel_minutes = get_max_travel_minutes()

    # Filter by travel time
    filtered = []
    for event in events:
        travel = event.get("travel_minutes")
        # Include if no travel info or within limit
        if travel is None or travel <= max_travel_minutes:
            filtered.append(event)

    # Check calendar conflicts if authenticated
    if is_calendar_authenticated():
        for event in filtered:
            event_dt = event.get("datetime")
            if event_dt:
                has_conf, conflicts = has_conflict(event_dt)
                event["_has_conflict"] = has_conf
                event["_conflicts"] = conflicts
            else:
                event["_has_conflict"] = None
                event["_conflicts"] = []

        # Optionally exclude conflicting events
        if exclude_conflicts:
            filtered = [e for e in filtered if not e.get("_has_conflict")]
    else:
        for event in filtered:
            event["_has_conflict"] = None
            event["_conflicts"] = []

    return filtered


def format_matched_events(events: list[dict], interpretation: str, max_events: int = 5) -> str:
    """
    Format matched events for display, including match reasons.

    Args:
        events: List of matched events (with _match_score and _match_reason)
        interpretation: LLM's interpretation of the query
        max_events: Maximum events to show

    Returns:
        Formatted string for display
    """
    if not events:
        return f"_{interpretation}_\n\nNo events found matching your criteria."

    lines = [f"_{interpretation}_\n"]

    events_to_show = events[:max_events]
    current_day = None

    for event in events_to_show:
        if not event.get("datetime"):
            continue

        day = event["datetime"].strftime("%A, %B %d")

        if day != current_day:
            current_day = day
            lines.append(f"\n**{day}**")

        time_str = (
            event["datetime"].strftime("%I:%M %p").lstrip("0")
            if event.get("has_specific_time", True)
            else "TBD"
        )

        name = event.get("name", "Unknown Event")
        if event.get("sold_out"):
            name += " [SOLD OUT]"

        score = event.get("_match_score", 0)
        location = event.get("location", "")
        source = event.get("source", "")

        # Build info line with travel time
        travel_mins = event.get("travel_minutes")
        travel_str = f" | 🚇 {travel_mins} min" if travel_mins else ""

        lines.append(f"- {time_str}: **{name}** ({score}% match)")
        lines.append(f"  📍 {location} ({source}){travel_str}")

        # Show conflict warning if calendar is configured
        if event.get("_has_conflict"):
            conflicts = event.get("_conflicts", [])
            conflict_names = [c.get("summary", "Busy") for c in conflicts[:2]]
            lines.append(f"  ⚠️ _Conflicts with: {', '.join(conflict_names)}_")

        # Show match reason if available
        reason = event.get("_match_reason", "")
        if reason:
            lines.append(f"  _→ {reason}_")

    result = "\n".join(lines)

    if len(events) > max_events:
        result += f"\n\n_...and {len(events) - max_events} more matches_"

    return result


CONCERT_MATCHING_PROMPT = """You are helping a user find concerts in NYC based on their YouTube Music taste. Analyze their query and the available concerts to find the best matches.

USER QUERY: "{query}"

TODAY'S DATE: {today}

AVAILABLE CONCERTS (from user's liked artists):
{concerts_json}

Your task:
1. Understand what the user is looking for (artist, genre, mood, date preferences)
2. Score each concert from 0-100 based on how well it matches the query
3. Consider:
   - Artist name relevance to the query
   - Date/time preferences mentioned in query
   - Venue preferences
   - How much the user likes the artist (indicated by liked_songs count)
   - Whether the user is subscribed to the artist (★ indicator)

Return ONLY a valid JSON object with this structure:
{{
  "interpretation": "Brief explanation of what you think the user wants",
  "matched_concerts": [
    {{"index": 0, "score": 85, "reason": "Why this matches"}},
    {{"index": 2, "score": 72, "reason": "Why this matches"}}
  ]
}}

Only include concerts with score >= 50. Sort by score descending. Limit to top 5 matches.
Return matched_concerts as empty array [] if nothing matches well.
"""


def match_concerts_to_query(query: str, concerts: list[dict], max_concerts_to_analyze: int = 100) -> list[dict]:
    """
    Use Gemini to semantically match concerts to a user query.

    Args:
        query: The user's natural language query
        concerts: List of concert dicts
        max_concerts_to_analyze: Max concerts to send to LLM (for token limits)

    Returns:
        List of matched concerts with scores and reasons
    """
    now = datetime.now(ZoneInfo("America/New_York"))

    # Prepare concerts for LLM
    concerts_for_llm = []
    for i, concert in enumerate(concerts[:max_concerts_to_analyze]):
        # Parse date to check if in the past
        try:
            concert_date = datetime.strptime(concert.get("date", ""), "%Y-%m-%d")
            if concert_date.date() < now.date():
                continue
        except ValueError:
            pass  # Keep if date can't be parsed

        source_indicator = "★" if "subscription" in concert.get("artist_source", "") else ""

        concerts_for_llm.append({
            "index": i,
            "artist": f"{concert.get('artist', '')} {source_indicator}".strip(),
            "event_name": concert.get("event_name", ""),
            "date": concert.get("date", ""),
            "time": concert.get("time", ""),
            "venue": concert.get("venue", ""),
            "city": concert.get("city", ""),
            "liked_songs": concert.get("liked_songs", 0),
            "price_range": f"${concert.get('price_min', 'N/A')} - ${concert.get('price_max', 'N/A')}" if concert.get("price_min") else "N/A",
            "status": concert.get("status", ""),
        })

    if not concerts_for_llm:
        return []

    prompt = CONCERT_MATCHING_PROMPT.format(
        query=query,
        today=now.strftime("%A, %B %d, %Y"),
        concerts_json=json.dumps(concerts_for_llm, indent=2),
    )

    try:
        response_text = generate_content(prompt).strip()

        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        # Filter and sort concerts by score
        matched = result.get("matched_concerts", [])
        if not matched:
            return []

        matched.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Build result list with concert data
        matched_concerts = []
        for m in matched:
            idx = m.get("index", -1)
            if 0 <= idx < len(concerts):
                concert = concerts[idx].copy()
                concert["_match_score"] = m.get("score", 0)
                concert["_match_reason"] = m.get("reason", "")
                matched_concerts.append(concert)

        return matched_concerts

    except Exception as e:
        print(f"    LLM concert matching error: {e}")
        return []


def apply_user_filters_concerts(
    concerts: list[dict],
    max_travel_minutes: int | None = None,
    exclude_conflicts: bool = True,
) -> list[dict]:
    """
    Apply user preference filters to concerts.

    Args:
        concerts: List of concerts (already matched/scored)
        max_travel_minutes: Max travel time filter (None = use config default)
        exclude_conflicts: Whether to exclude concerts with calendar conflicts

    Returns:
        Filtered list of concerts with conflict info added
    """
    if max_travel_minutes is None:
        max_travel_minutes = get_max_travel_minutes()

    # Filter by travel time
    filtered = []
    for concert in concerts:
        travel = concert.get("travel_minutes")
        # Include if no travel info or within limit
        if travel is None or travel <= max_travel_minutes:
            filtered.append(concert)

    # Check calendar conflicts if authenticated
    if is_calendar_authenticated():
        for concert in filtered:
            date_str = concert.get("date", "")
            time_str = concert.get("time", "")
            if date_str:
                try:
                    # Parse concert datetime
                    if time_str:
                        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                    else:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                        dt = dt.replace(hour=20)  # Assume 8 PM if no time
                    dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

                    has_conf, conflicts = has_conflict(dt)
                    concert["_has_conflict"] = has_conf
                    concert["_conflicts"] = conflicts
                except ValueError:
                    concert["_has_conflict"] = None
                    concert["_conflicts"] = []
            else:
                concert["_has_conflict"] = None
                concert["_conflicts"] = []

        # Optionally exclude conflicting concerts
        if exclude_conflicts:
            filtered = [c for c in filtered if not c.get("_has_conflict")]
    else:
        for concert in filtered:
            concert["_has_conflict"] = None
            concert["_conflicts"] = []

    return filtered


def format_matched_concerts(concerts: list[dict], max_concerts: int = 5) -> str:
    """
    Format matched concerts for display.

    Args:
        concerts: List of matched concerts (with _match_score and _match_reason)
        max_concerts: Maximum concerts to show

    Returns:
        Formatted string for display
    """
    if not concerts:
        return ""

    lines = ["**🎵 Concerts from Your Artists**\n"]

    concerts_to_show = concerts[:max_concerts]

    for concert in concerts_to_show:
        artist = concert.get("artist", "Unknown Artist")
        source = concert.get("artist_source", "")
        source_indicator = " ★" if "subscription" in source else ""

        date_str = concert.get("date", "TBD")
        time_str = concert.get("time", "")
        venue = concert.get("venue", "")
        city = concert.get("city", "")

        score = concert.get("_match_score", 0)
        reason = concert.get("_match_reason", "")

        # Price info
        price_min = concert.get("price_min")
        price_max = concert.get("price_max")
        if price_min and price_max:
            price_str = f"${price_min:.0f}-${price_max:.0f}"
        elif price_min:
            price_str = f"from ${price_min:.0f}"
        else:
            price_str = ""

        status = concert.get("status", "")
        status_str = " [SOLD OUT]" if status.lower() == "offsale" else ""

        # Travel time
        travel_mins = concert.get("travel_minutes")
        travel_str = f" | 🚇 {travel_mins} min" if travel_mins else ""

        lines.append(f"- **{artist}{source_indicator}**{status_str} ({score}% match)")
        lines.append(f"  📅 {date_str} {time_str}")
        lines.append(f"  📍 {venue}, {city}{travel_str}")
        if price_str:
            lines.append(f"  💰 {price_str}")

        # Show conflict warning if present
        if concert.get("_has_conflict"):
            conflicts = concert.get("_conflicts", [])
            conflict_names = [c.get("summary", "Busy") for c in conflicts[:2]]
            lines.append(f"  ⚠️ _Conflicts with: {', '.join(conflict_names)}_")

        if reason:
            lines.append(f"  _→ {reason}_")
        lines.append("")

    result = "\n".join(lines)

    if len(concerts) > max_concerts:
        result += f"\n_...and {len(concerts) - max_concerts} more concerts_"

    return result
