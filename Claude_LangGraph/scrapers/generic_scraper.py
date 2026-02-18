"""Generic scraper that uses LLM to parse events from any source."""

import importlib.util
import json
import re
import time
import requests
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import generate_content
from utils.jina_reader import fetch_page_text_jina
from utils.distance import enrich_events_with_travel_time


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


EVENT_PARSING_PROMPT = """You are extracting event information from a webpage. Parse the text and return a JSON array of events.

Source: {source_name}
Default Location: {default_location}
Default Event Type: {default_event_type}
Parsing Hints: {parsing_hints}

Current year context: Events are in 2026. Use 2026 for any dates that don't specify a year.

For each event, extract:
- name: Event title (max 70 chars)
- date_str: The date as shown on the page
- time_str: The time if available (e.g., "7:00 PM"), or null
- location: Venue/location (use default if not specified)
- type: Event type/category (use default if not specified)
- description: Brief description (max 100 chars), or empty string
- sold_out: true if sold out, false otherwise
- url: Event URL if available, otherwise null

Return ONLY a valid JSON array. No markdown, no explanation. Example:
[
  {{"name": "Event Name", "date_str": "February 1, 2026", "time_str": "7:00 PM", "location": "Venue", "type": "Concert", "description": "A great show", "sold_out": false, "url": null}}
]

If no events found, return: []

PAGE TEXT:
{page_text}
"""


def _get_browser_context(playwright):
    """Create a browser context with anti-detection settings."""
    browser = playwright.chromium.launch(
        headless=True, args=["--disable-blink-features=AutomationControlled"]
    )
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    return browser, context


def fetch_page_text_playwright(
    url: str,
    wait_seconds: int | None = None,
    cloudflare_protected: bool = False
) -> str:
    """Fetch page text using Playwright."""
    if wait_seconds is None:
        wait_seconds = _settings.SCRAPER_PLAYWRIGHT_WAIT

    with sync_playwright() as p:
        browser, context = _get_browser_context(p)
        page = context.new_page()

        goto_timeout = int(_settings.GENERIC_SCRAPER_PLAYWRIGHT_GOTO_TIMEOUT_MS)
        page.goto(url, timeout=goto_timeout)

        # Handle Cloudflare challenge if needed
        if cloudflare_protected:
            for _ in range(_settings.SCRAPER_CLOUDFLARE_MAX_CHECKS):
                time.sleep(_settings.SCRAPER_CLOUDFLARE_WAIT)
                title = page.title().lower()
                if "moment" not in title and "checking" not in title:
                    break

        # Additional wait for dynamic content
        time.sleep(wait_seconds)

        # Get page text
        text = page.evaluate("() => document.body.innerText")

        browser.close()

    return text


def fetch_api_json(url: str) -> dict | list:
    """Fetch JSON from an API endpoint."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    timeout = int(_settings.GENERIC_SCRAPER_API_TIMEOUT_SEC)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_event_content(page_text: str, max_chars: int | None = None) -> str:
    """
    Extract the most relevant event content from a page.

    Tries to find where event listings start and extracts from there.
    """
    if max_chars is None:
        max_chars = _settings.SCRAPER_MAX_CONTENT_CHARS

    # Patterns that often appear before event listings
    start_patterns = [
        r'(?i)(upcoming events|event calendar|events calendar|what\'s on)',
        r'(?i)(january|february|march|april|may|june|july|august|september|october|november|december)\s+202[4-6]',
        r'(?i)sort by',
        r'(?i)items \d+-\d+ of',
    ]

    best_start = 0
    for pattern in start_patterns:
        match = re.search(pattern, page_text)
        if match and match.start() > best_start:
            # Start a bit before the match for context
            best_start = max(0, match.start() - 200)

    # If we found a good starting point, use it
    if best_start > 1000:  # Only if it's meaningfully into the document
        page_text = page_text[best_start:]

    # Truncate to max length
    if len(page_text) > max_chars:
        page_text = page_text[:max_chars] + "\n... [truncated]"

    return page_text


def parse_events_with_llm(
    page_text: str,
    source_name: str,
    default_location: str,
    default_event_type: str,
    parsing_hints: str,
) -> list[dict]:
    """Use Gemini to parse events from page text."""
    import re

    # Extract the most relevant content section
    page_text = _extract_event_content(page_text)

    prompt = EVENT_PARSING_PROMPT.format(
        source_name=source_name,
        default_location=default_location,
        default_event_type=default_event_type,
        parsing_hints=parsing_hints,
        page_text=page_text,
    )

    response_text = generate_content(prompt).strip()

    # Try to parse as JSON
    try:
        events = json.loads(response_text)
        if isinstance(events, list):
            return events
    except json.JSONDecodeError:
        # Try to find JSON array in response (might be wrapped in markdown)
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            try:
                events = json.loads(match.group())
                if isinstance(events, list):
                    return events
            except json.JSONDecodeError:
                pass

    return []


def parse_datetime(date_str: str, time_str: str | None) -> tuple[datetime | None, bool]:
    """Parse date and time strings into a datetime object."""
    from dateutil import parser as date_parser

    try:
        # Combine date and time if both available
        if time_str:
            dt = date_parser.parse(f"{date_str} {time_str}")
            has_specific_time = True
        else:
            dt = date_parser.parse(date_str)
            # Check if time was actually in the date_str (not midnight)
            has_specific_time = dt.hour != 0 or dt.minute != 0

        # Handle UTC timestamps (ending with Z or having tzinfo)
        if dt.tzinfo is not None:
            # Convert to Eastern time
            dt = dt.astimezone(ZoneInfo("America/New_York"))
            has_specific_time = True  # UTC timestamps always have specific time
        else:
            # Ensure year is 2026 if not specified or wrong
            if dt.year < 2026:
                dt = dt.replace(year=2026)
            # Add timezone for naive datetimes
            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        return dt, has_specific_time
    except Exception:
        return None, False


def parse_api_events_directly(data: list, source: dict) -> list[dict]:
    """
    Parse API JSON data directly without LLM for known formats.

    Returns list of events or empty list if parsing fails.
    """
    source_name = source["name"]
    default_location = source.get("default_location", source_name)
    default_event_type = source.get("default_event_type", "Event")
    api_format = source.get("api_format", "standard")
    events = []

    for item in data:
        if not isinstance(item, dict):
            continue

        # Handle Airtable-style APIs where data is nested in 'fields'
        if api_format == "airtable" and "fields" in item:
            item = item["fields"]

        # Try common field names for event data
        name = (
            item.get("Event") or
            item.get("title") or
            item.get("name") or
            item.get("event_name") or
            item.get("summary") or
            ""
        )

        if not name:
            continue

        # Try common date field names
        date_str = (
            item.get("Event start date and time") or
            item.get("dateTime") or
            item.get("start_date") or
            item.get("date") or
            item.get("start") or
            item.get("event_date") or
            ""
        )

        # Try to get location
        location = default_location
        if item.get("venue"):
            if isinstance(item["venue"], dict):
                location = item["venue"].get("name") or item["venue"].get("address") or default_location
            else:
                location = str(item["venue"])
        elif item.get("location"):
            if isinstance(item["location"], dict):
                location = item["location"].get("name") or item["location"].get("address") or default_location
            else:
                location = str(item["location"])

        # Try to get description
        description = (
            item.get("Short description") or
            item.get("description") or
            item.get("summary") or
            item.get("excerpt") or
            ""
        )
        if isinstance(description, str):
            # Strip HTML tags if present
            description = re.sub(r'<[^>]+>', '', description)[:100]

        # Check sold out status
        sold_out = (
            item.get("soldOut") or
            item.get("sold_out") or
            item.get("isSoldOut") or
            False
        )

        # Get event URL
        event_url = (
            item.get("url") or
            item.get("link") or
            item.get("event_url") or
            source["url"]
        )

        # Parse datetime
        dt, has_specific_time = parse_datetime(str(date_str), None)

        events.append({
            "name": str(name)[:70],
            "datetime": dt,
            "date_str": str(date_str),
            "type": item.get("type") or item.get("category") or default_event_type,
            "sold_out": bool(sold_out),
            "source": source_name,
            "location": location,
            "description": description,
            "has_specific_time": has_specific_time,
            "url": event_url,
            "travel_minutes": None,  # Will be populated by caller
        })

    # Enrich events with travel time from user's home
    events = enrich_events_with_travel_time(events)

    return events


def fetch_events_from_source(source: dict) -> list[dict]:
    """
    Fetch events from a source using its configuration.

    This is the main generic scraper function that works with any source.
    """
    source_name = source["name"]
    url = source["url"]
    method = source.get("method", "playwright")
    default_location = source.get("default_location", source_name)
    default_event_type = source.get("default_event_type", "Event")
    parsing_hints = source.get("parsing_hints", "Extract event name, date, time, and location.")

    if method == "api":
        # Fetch JSON from API and parse directly (no LLM needed)
        try:
            data = fetch_api_json(url)

            # Navigate to events array if path specified
            events_path = source.get("api_events_path", "")
            if events_path:
                for key in events_path.split("."):
                    if isinstance(data, dict) and key in data:
                        data = data[key]
                    else:
                        data = []
                        break

            if isinstance(data, list) and data:
                print(f"    Parsing {len(data)} items from API...", flush=True)
                events = parse_api_events_directly(data, source)
                if events:
                    return events
                # Fall through to LLM parsing if direct parsing returned nothing
                print(f"    Direct parsing failed, trying LLM...", flush=True)
                page_text = json.dumps(data, indent=2, default=str)
            else:
                return []

        except Exception as e:
            print(f"    API error: {e}")
            return []
    else:
        # Fetch page - use Jina for non-Cloudflare pages, Playwright otherwise
        cloudflare_protected = source.get("cloudflare_protected", False)

        if cloudflare_protected:
            # Cloudflare-protected sites need Playwright to handle the challenge
            wait_seconds = source.get("wait_seconds", 3)
            try:
                print(f"    Using Playwright (Cloudflare protected)...", flush=True)
                page_text = fetch_page_text_playwright(url, wait_seconds, cloudflare_protected)
            except Exception as e:
                print(f"    Playwright error: {e}")
                return []
        else:
            # Use Jina Reader for cleaner, LLM-friendly text
            try:
                print(f"    Using Jina Reader...", flush=True)
                page_text = fetch_page_text_jina(url)
            except Exception as e:
                print(f"    Jina error: {e}, falling back to Playwright...")
                wait_seconds = source.get("wait_seconds", 3)
                try:
                    page_text = fetch_page_text_playwright(url, wait_seconds, False)
                except Exception as e2:
                    print(f"    Playwright fallback error: {e2}")
                    return []

        # Parse with LLM
        print(f"    Parsing with LLM...", flush=True)

    # LLM parsing (for Playwright sources or API fallback)
    raw_events = parse_events_with_llm(
        page_text=page_text,
        source_name=source_name,
        default_location=default_location,
        default_event_type=default_event_type,
        parsing_hints=parsing_hints,
    )

    # Convert to standard Event format
    events = []
    for raw in raw_events:
        dt, has_specific_time = parse_datetime(
            raw.get("date_str", ""),
            raw.get("time_str"),
        )

        events.append({
            "name": raw.get("name", "Unknown Event")[:70],
            "datetime": dt,
            "date_str": raw.get("date_str", ""),
            "type": raw.get("type", default_event_type),
            "sold_out": raw.get("sold_out", False),
            "source": source_name,
            "location": raw.get("location", default_location),
            "description": raw.get("description", "")[:100],
            "has_specific_time": has_specific_time,
            "url": raw.get("url") or url,
            "travel_minutes": None,  # Will be populated below
        })

    # Enrich events with travel time from user's home
    events = enrich_events_with_travel_time(events)

    return events
