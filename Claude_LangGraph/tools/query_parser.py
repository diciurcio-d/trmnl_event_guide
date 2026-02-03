"""Use Gemini to semantically match user queries to events."""

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import generate_content


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

        lines.append(f"- {time_str}: **{name}** ({score}% match)")
        lines.append(f"  📍 {location} ({source})")

        # Show match reason if available
        reason = event.get("_match_reason", "")
        if reason:
            lines.append(f"  _→ {reason}_")

    result = "\n".join(lines)

    if len(events) > max_events:
        result += f"\n\n_...and {len(events) - max_events} more matches_"

    return result
