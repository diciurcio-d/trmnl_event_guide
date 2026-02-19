"""Main orchestrator for fetching events from venues.

Coordinates between Ticketmaster and website scraping based on
venue category and characteristics.
"""

import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from .source_strategy import determine_fetch_strategy, select_best_source, should_skip_ticketmaster
from .event_cache import is_venue_events_fresh, mark_venue_fetched, get_stale_venues
from .venue_events_sheet import append_venue_events
from .observability import increment, log_event, record_failure


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


def _save_discovered_endpoint(venue_name: str, api_endpoint: str, city: str):
    """Save a discovered API endpoint to the venue cache."""
    # Skip during batch fetching to avoid rate limits on Sheets API
    if os.environ.get("EVENT_FETCHER_BATCH_MODE"):
        return

    try:
        from .cache import read_cached_venues, update_venues_batch

        venues = read_cached_venues(city)
        for v in venues:
            if v.get("name") == venue_name:
                v["api_endpoint"] = api_endpoint
                v["preferred_event_source"] = "api"
                update_venues_batch([v], city)
                print(f"    Saved API endpoint for future use")
                break
    except Exception as e:
        print(f"    Could not save API endpoint: {e}")


@dataclass
class FetchResult:
    """Result of fetching events for a venue."""
    venue_name: str
    events: list[dict] = field(default_factory=list)
    source_used: str = ""
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""
    attempted_sources: list[str] = field(default_factory=list)
    source_errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize_event_schema(event: dict, venue_name: str, source: str) -> dict:
    """Normalize event shape so all sources write a consistent schema."""
    normalized = dict(event)
    normalized["name"] = normalized.get("name", "")
    normalized["datetime"] = normalized.get("datetime")
    normalized["date_str"] = normalized.get("date_str", "")
    normalized["venue_name"] = normalized.get("venue_name") or venue_name
    normalized["event_type"] = normalized.get("event_type", "")
    normalized["url"] = normalized.get("url", "")
    normalized["source"] = normalized.get("source") or source
    normalized["matched_artist"] = normalized.get("matched_artist", "")
    normalized["travel_minutes"] = normalized.get("travel_minutes")
    normalized["description"] = normalized.get("description", "")
    normalized["event_source_url"] = normalized.get("event_source_url") or normalized.get("url", "")
    normalized["extraction_method"] = normalized.get("extraction_method", source)
    normalized["relevance_score"] = normalized.get("relevance_score")
    normalized["validation_confidence"] = normalized.get("validation_confidence")
    return normalized


def _normalize_event_batch(events: list[dict], venue_name: str, source: str) -> list[dict]:
    """Normalize all events to the canonical venue event schema."""
    return [_normalize_event_schema(event, venue_name, source) for event in events]


def _get_ticketmaster_client():
    """Get Ticketmaster client instance."""
    try:
        from concert_finder.ticketmaster_client import TicketmasterClient
        return TicketmasterClient()
    except Exception as e:
        print(f"  Warning: Could not initialize Ticketmaster client: {e}")
        return None


def _fetch_from_ticketmaster(
    venue_name: str,
    city: str,
    state_code: str = "NY",
    months_ahead: int = 3,
    venue_id: str | None = None,
) -> list[dict]:
    """
    Fetch events from Ticketmaster for a venue.

    Args:
        venue_name: Name of the venue
        city: City name
        state_code: State code (default NY)
        months_ahead: How many months ahead to search
        venue_id: If provided, use direct lookup (fast). If None, skip TM.

    Returns:
        List of event dicts
    """
    # Only use Ticketmaster if we have a venue ID (from scan-ticketmaster)
    # This avoids slow venue name searches during event fetching
    if not venue_id or venue_id == "not_found":
        return []

    client = _get_ticketmaster_client()
    if not client:
        return []

    try:
        events = client.search_events_by_venue(
            venue_name=venue_name,
            city=city,
            state_code=state_code,
            months_ahead=months_ahead,
            venue_id=venue_id,
        )
        return events
    except Exception as e:
        print(f"  Ticketmaster error for {venue_name}: {e}")
        record_failure(
            "event_fetcher.ticketmaster",
            str(e),
            venue_name=venue_name,
            city=city,
        )
        return []


def _fetch_from_api(api_url: str, venue_name: str) -> list[dict]:
    """
    Fetch events from a known API endpoint.

    Returns:
        List of event dicts
    """
    if not api_url:
        return []

    try:
        from .api_detector import fetch_from_api
        return fetch_from_api(api_url, venue_name)
    except ImportError:
        record_failure(
            "event_fetcher.api_import",
            "api_detector import failed",
            venue_name=venue_name,
            api_url=api_url,
        )
        return []
    except Exception as e:
        print(f"  API fetch error for {venue_name}: {e}")
        record_failure(
            "event_fetcher.api",
            str(e),
            venue_name=venue_name,
            api_url=api_url,
        )
        return []


def _detect_and_fetch_api(url: str, venue_name: str) -> tuple[list[dict], str | None]:
    """
    Try to detect API endpoints and fetch events.

    Returns:
        Tuple of (events list, discovered api_endpoint or None)
    """
    if not url:
        return [], None

    try:
        from .api_detector import detect_and_fetch
        return detect_and_fetch(url, venue_name)
    except ImportError as e:
        print(f"  Could not import API detector: {e}")
        return [], None
    except Exception as e:
        print(f"  API detection error for {venue_name}: {e}")
        return [], None


def _should_skip_jina() -> bool:
    """Check if Jina should be skipped for event fetching."""
    import os
    # Check environment variable first (allows runtime override)
    env_val = os.environ.get("EVENT_FETCHER_SKIP_JINA", "").lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False
    # Fall back to settings
    return getattr(_settings, "EVENT_FETCHER_SKIP_JINA", False)


def _fetch_raw_html(url: str) -> str:
    """Fetch raw HTML for structured extraction and iframe discovery."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    timeout = int(_settings.EVENT_FETCHER_HTML_TIMEOUT_SEC)
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.text


def _fetch_with_playwright(url: str) -> tuple[str, str]:
    """
    Fetch page content using Playwright (browser automation).

    Used as fallback when raw HTML fetch fails (403) or returns empty content
    (JavaScript-rendered pages).

    Strategy: Try fast mode first (domcontentloaded), fall back to slow mode
    (networkidle) if no content is returned.

    Returns:
        Tuple of (raw_html, page_text) - HTML source and visible text content
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"    Playwright not installed, cannot use browser fallback")
        return "", ""

    # Fast mode settings
    timeout_ms = int(getattr(_settings, "EVENT_FETCHER_PLAYWRIGHT_TIMEOUT_MS", 12000))
    wait_ms = int(getattr(_settings, "EVENT_FETCHER_PLAYWRIGHT_WAIT_MS", 1500))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.new_page()

            # Try fast mode first: domcontentloaded + short JS wait
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            page.wait_for_timeout(wait_ms)

            # Get content
            raw_html = page.content()
            page_text = page.evaluate("() => document.body.innerText")

            # If fast mode returned empty, try slow mode (networkidle)
            # This handles sites like Ovationtix that need full network settling
            if not page_text or len(page_text.strip()) < 100:
                page.goto(url, timeout=timeout_ms + 8000, wait_until="networkidle")
                page.wait_for_timeout(1000)
                raw_html = page.content()
                page_text = page.evaluate("() => document.body.innerText")

            browser.close()

            return raw_html, page_text

    except Exception as e:
        print(f"    Playwright fetch failed: {e}")
        return "", ""


def _looks_like_js_calendar(html: str, stripped_text: str) -> bool:
    """
    Detect if HTML looks like a JavaScript-rendered calendar that needs Playwright.

    Returns True if content has signs of JS-rendered calendar:
    - Contains "loading" placeholder text
    - Has calendar-related JavaScript files
    - Very little actual text content despite HTML
    """
    if not html:
        return False

    html_lower = html.lower()
    text_lower = stripped_text.lower() if stripped_text else ""

    # Check for loading placeholders
    loading_patterns = [
        "loading events",
        "loading...",
        "loading calendar",
        "please wait",
        "fetching events",
    ]
    for pattern in loading_patterns:
        if pattern in text_lower:
            return True

    # Check for calendar-related JavaScript
    calendar_js_patterns = [
        "calendar.js",
        "events.js",
        "fullcalendar",
        "eventbrite",
        "ovationtix",
        "ticketleap",
        "dice.fm",
        "eventcalendar",
        "tribe/events",  # WordPress events plugin
    ]
    for pattern in calendar_js_patterns:
        if pattern in html_lower:
            return True

    # Check for very low text-to-HTML ratio (sign of JS-heavy page)
    if html and stripped_text:
        ratio = len(stripped_text) / len(html)
        # If we have lots of HTML but very little text, likely JS-rendered
        if ratio < 0.05 and len(html) > 5000:
            return True

    return False


def _extract_google_calendar_events(html: str, venue_name: str) -> list[dict]:
    """
    Extract events from embedded Google Calendar iframes.

    Looks for Google Calendar embed iframes, extracts the calendar ID,
    and fetches events from the public iCal feed.

    Args:
        html: Raw HTML of the page
        venue_name: Name of the venue for event records

    Returns:
        List of event dicts, or empty list if no Google Calendar found
    """
    import base64
    import urllib.request
    from urllib.parse import parse_qs, urlparse

    if not html:
        return []

    # Find Google Calendar embed iframe
    # Pattern: calendar.google.com/calendar/embed?...&src=BASE64_CALENDAR_ID
    iframe_pattern = r'calendar\.google\.com/calendar/embed\?[^"\'>\s]+'
    matches = re.findall(iframe_pattern, html, re.IGNORECASE)

    if not matches:
        return []

    print(f"    Found Google Calendar iframe, extracting events...")

    for match in matches[:3]:  # Try up to 3 calendar embeds
        try:
            # Parse the embed URL to get calendar ID
            # The src parameter contains base64-encoded calendar ID
            parsed = urlparse("https://" + match)
            params = parse_qs(parsed.query)

            src_values = params.get("src", [])
            if not src_values:
                continue

            # Decode the calendar ID (base64 encoded email)
            calendar_id_b64 = src_values[0]
            try:
                # Add padding if needed
                padding = 4 - len(calendar_id_b64) % 4
                if padding != 4:
                    calendar_id_b64 += "=" * padding
                calendar_id = base64.b64decode(calendar_id_b64).decode("utf-8")
            except Exception:
                # Maybe it's not base64 encoded, use as-is
                calendar_id = src_values[0]

            # Fetch the public iCal feed
            ical_url = f"https://calendar.google.com/calendar/ical/{urllib.parse.quote(calendar_id)}/public/basic.ics"

            try:
                req = urllib.request.Request(
                    ical_url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; EventBot/1.0)"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    ical_data = response.read().decode("utf-8")
            except Exception as e:
                print(f"    Could not fetch iCal feed: {e}")
                continue

            # Parse iCal data
            events = []
            current_event = {}
            now = datetime.now(ZoneInfo("America/New_York"))

            for line in ical_data.split("\n"):
                line = line.strip()
                # Handle line continuations (lines starting with space)
                if line.startswith(" ") and current_event:
                    continue

                if line == "BEGIN:VEVENT":
                    current_event = {}
                elif line == "END:VEVENT":
                    if current_event.get("summary") and current_event.get("dtstart"):
                        # Only include future events
                        event_dt = current_event["dtstart"]
                        if event_dt.tzinfo is None:
                            event_dt = event_dt.replace(tzinfo=ZoneInfo("America/New_York"))
                        if event_dt > now:
                            events.append({
                                "event_name": current_event["summary"],
                                "datetime": event_dt.isoformat(),
                                "date_str": event_dt.strftime("%Y-%m-%d"),
                                "venue_name": venue_name,
                                "event_source_url": ical_url,
                                "extraction_method": "google_calendar_ical",
                            })
                    current_event = {}
                elif line.startswith("SUMMARY:"):
                    # Handle escaped characters in summary
                    summary = line[8:].replace("\\,", ",").replace("\\;", ";").replace("\\n", " ")
                    current_event["summary"] = summary
                elif line.startswith("DTSTART"):
                    # Handle both DATE and DATETIME formats
                    try:
                        if "VALUE=DATE:" in line:
                            date_str = line.split(":")[-1]
                            current_event["dtstart"] = datetime.strptime(date_str, "%Y%m%d")
                        elif ":" in line:
                            date_str = line.split(":")[-1].replace("Z", "")
                            if "T" in date_str:
                                current_event["dtstart"] = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
                            else:
                                current_event["dtstart"] = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        pass

            if events:
                # Sort by date
                events.sort(key=lambda x: x["datetime"])
                print(f"    Google Calendar: {len(events)} upcoming events")
                return events

        except Exception as e:
            print(f"    Error parsing Google Calendar: {e}")
            continue

    return []


def _strip_html_for_llm(html: str) -> str:
    """Strip HTML tags to get plain text for LLM parsing.

    Used when skipping Jina Reader to still allow LLM event extraction.
    """
    if not html:
        return ""

    # Remove script and style elements entirely
    html = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html, flags=re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--[\s\S]*?-->', '', html)

    # Replace common block elements with newlines for readability
    html = re.sub(r'<(br|hr|p|div|li|tr|h[1-6])[^>]*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(p|div|li|tr|h[1-6])>', '\n', html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    html = re.sub(r'<[^>]+>', ' ', html)

    # Decode common HTML entities
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&quot;', '"')
    html = html.replace('&#39;', "'")

    # Collapse multiple whitespace and newlines
    html = re.sub(r'[ \t]+', ' ', html)
    html = re.sub(r'\n\s*\n+', '\n\n', html)

    return html.strip()


def _extract_jsonld_blocks(html: str) -> list[dict]:
    """Extract JSON-LD blocks from HTML."""
    blocks = []
    if not html:
        return blocks

    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>',
        html,
        re.IGNORECASE,
    ):
        payload = match.group(1).strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                blocks.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                blocks.append(data)
        except json.JSONDecodeError:
            continue
    return blocks


def _parse_date_flexible(value: str) -> datetime | None:
    """Parse a date/time string into timezone-aware datetime."""
    if not value:
        return None
    try:
        from dateutil import parser as date_parser

        parsed = date_parser.parse(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            parsed = parsed.astimezone(ZoneInfo("America/New_York"))
        return parsed
    except Exception:
        return None


def _extract_events_from_jsonld(html: str, venue_name: str, source_url: str) -> list[dict]:
    """Extract events from schema.org JSON-LD before LLM fallback."""
    blocks = _extract_jsonld_blocks(html)
    events = []

    for block in blocks:
        candidates = []
        if block.get("@type") == "Event":
            candidates.append(block)
        if isinstance(block.get("@graph"), list):
            candidates.extend(
                item for item in block["@graph"]
                if isinstance(item, dict) and item.get("@type") == "Event"
            )

        for item in candidates:
            name = str(item.get("name", "")).strip()
            start = str(item.get("startDate", "")).strip()
            if not name or not start:
                continue

            dt = _parse_date_flexible(start)
            date_str = dt.strftime("%Y-%m-%d") if dt else start[:10]
            location = ""
            loc = item.get("location")
            if isinstance(loc, dict):
                location = loc.get("name", "") or loc.get("address", "")
            elif isinstance(loc, str):
                location = loc

            events.append(
                {
                    "name": name,
                    "datetime": dt,
                    "date_str": date_str,
                    "venue_name": location or venue_name,
                    "event_type": "event",
                    "url": item.get("url", source_url),
                    "source": "scrape",
                    "matched_artist": "",
                    "travel_minutes": None,
                    "description": str(item.get("description", ""))[:200],
                    "event_source_url": source_url,
                    "extraction_method": "jsonld",
                    "validation_confidence": 0.9,
                }
            )

    return events


def _extract_iframe_srcs(html: str, base_url: str) -> list[str]:
    """Extract iframe source URLs from HTML."""
    links = []
    if not html:
        return links
    for match in re.finditer(r'<iframe[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE):
        src = match.group(1).strip()
        if not src or src.startswith(("javascript:", "data:", "#")):
            continue
        links.append(urljoin(base_url, src))
    return links


def _venue_tokens(venue: dict) -> set[str]:
    """Build normalized tokens that represent the venue's location identity."""
    text = " ".join(
        [
            venue.get("name", ""),
            venue.get("address", ""),
            venue.get("neighborhood", ""),
            venue.get("city", ""),
        ]
    ).lower()
    tokens = {t for t in re.findall(r"[a-z0-9]{3,}", text) if t not in {"venue", "street"}}
    return tokens


def _score_event_relevance(event: dict, venue: dict) -> int:
    """Score event relevance to a venue for shared multi-location event feeds."""
    venue_name = venue.get("name", "").lower()
    tokens = _venue_tokens(venue)
    haystack = " ".join(
        [
            str(event.get("name", "")),
            str(event.get("description", "")),
            str(event.get("venue_name", "")),
            str(event.get("url", "")),
        ]
    ).lower()

    score = 0
    if venue_name and venue_name in haystack:
        score += 60

    matched_tokens = sum(1 for token in tokens if token in haystack)
    score += min(matched_tokens * 8, 40)

    website = venue.get("website", "")
    event_url = event.get("url", "")
    if website and event_url:
        venue_domain = urlparse(website).netloc.lower().replace("www.", "")
        event_domain = urlparse(event_url).netloc.lower().replace("www.", "")
        if venue_domain and event_domain and (venue_domain == event_domain or event_domain.endswith("." + venue_domain)):
            score += 15

    return score


def _filter_events_for_venue(events: list[dict], venue: dict) -> list[dict]:
    """Filter out low-relevance events from shared feeds."""
    if not events:
        return []
    filtered = []
    for event in events:
        score = _score_event_relevance(event, venue)
        event["relevance_score"] = score
        # Keep high-confidence matches; allow unknown location pages to pass at lower score if tiny set.
        if score >= 20:
            filtered.append(event)

    if not filtered:
        # Avoid dropping all events for tiny pages with sparse metadata.
        if len(events) <= 2:
            return events
        # For larger mixed feeds, keep only stronger candidates.
        ranked = sorted(events, key=lambda e: e.get("relevance_score", 0), reverse=True)
        if ranked and ranked[0].get("relevance_score", 0) >= 12:
            return [event for event in ranked[: min(5, len(ranked))]]
        return []
    return filtered


def _fetch_from_website(
    url: str,
    venue_name: str,
    events_url: str = "",
    venue_context: dict | None = None,
) -> list[dict]:
    """
    Fetch events by scraping venue website.

    Uses structured extraction first (JSON-LD), then LLM extraction.
    Falls back to iframe pages when embedded event widgets are used.
    If events_url is provided, fetches from that instead of the homepage.

    When EVENT_FETCHER_SKIP_JINA is True, uses raw HTML with tag stripping
    instead of Jina Reader for LLM parsing (faster when Jina is rate-limited).

    Args:
        url: The venue's homepage URL
        venue_name: Name of the venue
        events_url: Direct URL to events/calendar page (preferred if available)
        venue_context: Optional venue metadata for downstream filtering

    Returns:
        List of event dicts
    """
    # Use events_url if available, otherwise fall back to homepage
    fetch_url = events_url if events_url else url

    if not fetch_url:
        return []

    skip_jina = _should_skip_jina()

    # Only import Jina if we're going to use it
    fetch_page_text_jina = None
    if not skip_jina:
        try:
            from utils.jina_reader import fetch_page_text_jina
        except ImportError as e:
            print(f"  Could not import Jina reader, falling back to raw HTML: {e}")
            skip_jina = True

    try:
        # Fetch page content
        if events_url:
            print(f"    Scraping events page: {fetch_url}...")
        else:
            print(f"    Scraping homepage: {fetch_url}...")

        raw_html = ""
        content = ""
        used_playwright = False
        fetch_failed_403 = False

        # Check if Playwright fallback is enabled
        use_playwright_fallback = getattr(_settings, "EVENT_FETCHER_PLAYWRIGHT_FALLBACK", True)

        # Always try raw HTML first (needed for JSON-LD and as Jina fallback)
        try:
            raw_html = _fetch_raw_html(fetch_url)
        except Exception as e:
            error_str = str(e)
            print(f"    Failed to fetch raw HTML: {e}")
            if "403" in error_str:
                fetch_failed_403 = True
            raw_html = ""

        # Playwright fallback for 403 errors
        if fetch_failed_403 and use_playwright_fallback:
            print(f"    Trying Playwright fallback for 403...")
            raw_html, playwright_text = _fetch_with_playwright(fetch_url)
            if raw_html or playwright_text:
                used_playwright = True
                if playwright_text and len(playwright_text) > 100:
                    content = playwright_text
                    print(f"    Playwright succeeded ({len(content)} chars)")

        # Get content for LLM parsing
        if not content:  # Only if we don't already have content from Playwright
            if skip_jina:
                # Use stripped HTML for LLM parsing
                if raw_html:
                    content = _strip_html_for_llm(raw_html)
                    if used_playwright:
                        print(f"    Using Playwright HTML (skip-jina mode)")
                    else:
                        print(f"    Using raw HTML (skip-jina mode)")
            else:
                # Use Jina Reader
                content = fetch_page_text_jina(fetch_url)
                if not content or len(content) < 100:
                    # Fallback to raw HTML if Jina fails
                    if raw_html:
                        print(f"    Jina returned no content, falling back to raw HTML")
                        content = _strip_html_for_llm(raw_html)

        # Playwright fallback for empty content (JavaScript-rendered pages)
        if (not content or len(content) < 100) and use_playwright_fallback and not used_playwright:
            print(f"    Content empty, trying Playwright fallback...")
            raw_html, playwright_text = _fetch_with_playwright(fetch_url)
            if playwright_text and len(playwright_text) > 100:
                content = playwright_text
                used_playwright = True
                print(f"    Playwright succeeded ({len(content)} chars)")
            elif raw_html:
                content = _strip_html_for_llm(raw_html)
                used_playwright = True

        if not content or len(content) < 100:
            print(f"    No content from {fetch_url}")
            record_failure(
                "event_fetcher.scrape_content",
                "empty_or_short_content",
                venue_name=venue_name,
                fetch_url=fetch_url,
            )
            return []

        # Try JSON-LD extraction first (doesn't need Jina)
        if raw_html:
            structured_events = _extract_events_from_jsonld(raw_html, venue_name, fetch_url)
            if structured_events:
                # Mark if Playwright was used
                if used_playwright:
                    for event in structured_events:
                        event["extraction_method"] = "jsonld_playwright"
                return structured_events

            # Try iframes for JSON-LD
            iframe_urls = _extract_iframe_srcs(raw_html, fetch_url)
            for iframe_url in iframe_urls[:5]:
                try:
                    iframe_html = _fetch_raw_html(iframe_url)
                    iframe_events = _extract_events_from_jsonld(iframe_html, venue_name, iframe_url)
                    if iframe_events:
                        for event in iframe_events:
                            event["event_source_url"] = iframe_url
                            event["extraction_method"] = "jsonld_iframe"
                        return iframe_events
                except Exception:
                    continue

        # Truncate content if too long
        max_chars = getattr(_settings, "SCRAPER_MAX_CONTENT_CHARS", 25000)
        if len(content) > max_chars:
            content = content[:max_chars]

        # Use LLM to extract events
        events = _parse_events_with_llm(content, venue_name, fetch_url)
        if events:
            # Mark extraction method based on source used
            if used_playwright:
                extraction_method = "llm_parse_playwright"
            elif skip_jina:
                extraction_method = "llm_parse_raw_html"
            else:
                extraction_method = "llm_parse"
            for event in events:
                event["extraction_method"] = extraction_method
            return events

        # LLM returned 0 events - check if this looks like a JS-rendered calendar
        # If so, try Playwright to get the actual content
        if not used_playwright and use_playwright_fallback and raw_html:
            if _looks_like_js_calendar(raw_html, content):
                print(f"    No events parsed, but page looks like JS calendar - trying Playwright...")
                pw_html, pw_text = _fetch_with_playwright(fetch_url)
                if pw_text and len(pw_text) > 100:
                    print(f"    Playwright got {len(pw_text)} chars, re-parsing...")
                    # Truncate if needed
                    if len(pw_text) > max_chars:
                        pw_text = pw_text[:max_chars]
                    events = _parse_events_with_llm(pw_text, venue_name, fetch_url)
                    if events:
                        for event in events:
                            event["extraction_method"] = "llm_parse_playwright_retry"
                        return events
                    print(f"    Still no events after Playwright retry")

        # Try iframes for LLM parsing
        if raw_html:
            iframe_urls = _extract_iframe_srcs(raw_html, fetch_url)
            for iframe_url in iframe_urls[:5]:
                try:
                    if skip_jina:
                        iframe_html = _fetch_raw_html(iframe_url)
                        iframe_content = _strip_html_for_llm(iframe_html)
                    else:
                        iframe_content = fetch_page_text_jina(iframe_url)

                    iframe_events = _parse_events_with_llm(iframe_content, venue_name, iframe_url)
                    if iframe_events:
                        extraction_method = "llm_iframe_raw_html" if skip_jina else "llm_iframe"
                        for event in iframe_events:
                            event["event_source_url"] = iframe_url
                            event["extraction_method"] = extraction_method
                        return iframe_events
                except Exception:
                    continue

        # Try Google Calendar iframe extraction as last resort
        if raw_html:
            gcal_events = _extract_google_calendar_events(raw_html, venue_name)
            if gcal_events:
                return gcal_events

        return []

    except Exception as e:
        print(f"  Website scraping error for {venue_name}: {e}")
        record_failure(
            "event_fetcher.scrape",
            str(e),
            venue_name=venue_name,
            fetch_url=fetch_url,
        )
        return []


def _parse_events_with_llm(content: str, venue_name: str, source_url: str) -> list[dict]:
    """
    Use LLM to extract events from webpage content.

    Returns:
        List of event dicts
    """
    from utils.llm import generate_content

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    prompt = f"""Extract upcoming events from this webpage content. Today's date is {today}.

WEBPAGE CONTENT:
{content}

Extract each event and return as a JSON array. For each event include:
- name: Event/show name
- date_str: Date in YYYY-MM-DD format (infer year if not shown, assume current/next year)
- time: Time if available (e.g. "7:00 PM")
- event_type: Type of event (concert, comedy, theater, reading, etc.)
- url: Event URL if available, otherwise empty string
- description: Brief description if available

Only include events from today ({today}) onwards.
If no events are found, return an empty array: []

Return ONLY the JSON array, no other text."""

    try:
        response = generate_content(prompt)

        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return []

        import json
        events_data = json.loads(json_match.group())

        # Convert to our event format
        events = []
        for e in events_data:
            date_str = e.get("date_str", "")
            time_str = e.get("time", "")

            # Try to parse datetime
            dt = None
            if date_str:
                try:
                    if time_str:
                        dt_str = f"{date_str} {time_str}"
                        for fmt in ["%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M", "%Y-%m-%d %I%p", "%Y-%m-%d"]:
                            try:
                                dt = datetime.strptime(dt_str, fmt)
                                dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                                break
                            except ValueError:
                                continue
                    if not dt:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                except ValueError:
                    pass

            events.append({
                "name": e.get("name", ""),
                "datetime": dt,
                "date_str": date_str,
                "venue_name": venue_name,
                "event_type": e.get("event_type", ""),
                "url": e.get("url", source_url),
                "source": "scrape",
                "matched_artist": "",
                "travel_minutes": None,
                "description": e.get("description", ""),
                "event_source_url": source_url,
                "extraction_method": "llm_parse",
                "validation_confidence": 0.6,
            })

        return events

    except Exception as e:
        print(f"    LLM parsing error: {e}")
        return []


def _normalize_tm_events(events: list[dict], venue_name: str) -> list[dict]:
    """Convert Ticketmaster events to our standard format."""
    normalized = []
    for e in events:
        date_str = e.get("date", "")
        time_str = e.get("time", "")

        # Parse datetime
        dt = None
        if date_str:
            try:
                if time_str:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                else:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
            except ValueError:
                pass

        normalized.append({
            "name": e.get("name", ""),
            "datetime": dt,
            "date_str": date_str,
            "venue_name": e.get("venue_name", venue_name),
            "event_type": "concert",  # TM is mostly music
            "url": e.get("url", ""),
            "source": "ticketmaster",
            "matched_artist": "",
            "travel_minutes": None,
            "description": e.get("info", ""),  # TM uses 'info' field
            "artists": e.get("artists", []),
            "price_min": e.get("price_min"),
            "price_max": e.get("price_max"),
            "event_source_url": e.get("url", ""),
            "extraction_method": "ticketmaster_api",
            "validation_confidence": 0.95,
        })

    return normalized


def _deduplicate_events(events: list[dict]) -> list[dict]:
    """Remove duplicate events by (name, date)."""
    seen = set()
    unique = []

    for event in events:
        # Create key from name and date
        name = event.get("name", "").lower().strip()
        date = event.get("date_str", "")
        key = (name, date)

        if key not in seen:
            seen.add(key)
            unique.append(event)

    return unique


def _batch_update_venue_metadata(results: dict, city: str):
    """
    Batch update venue metadata after parallel fetching.

    Updates last_event_fetch, event_count, and event_source for all
    venues in a single sheet write operation.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from .cache import (
        read_cached_venues, get_or_create_venues_sheet,
        _get_sheets_service, VENUE_COLUMNS, _venue_to_row
    )

    # Build updates dict
    updates = {}
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()

    for venue_name, result in results.items():
        if not result.skipped and not result.error:
            updates[venue_name.lower().strip()] = {
                "last_event_fetch": now,
                "event_count": len(result.events),
                "event_source": result.source_used,
            }

    if not updates:
        return

    # Read all venues
    venues = read_cached_venues()

    # Apply updates
    updated_count = 0
    for v in venues:
        key = v.get("name", "").lower().strip()
        if key in updates:
            v["last_event_fetch"] = updates[key]["last_event_fetch"]
            v["event_count"] = updates[key]["event_count"]
            v["event_source"] = updates[key]["event_source"]
            updated_count += 1

    if updated_count == 0:
        return

    # Write back
    sheet_id = get_or_create_venues_sheet()
    if not sheet_id:
        return

    service = _get_sheets_service()
    if not service:
        return

    rows = [VENUE_COLUMNS]
    for venue in venues:
        rows.append(_venue_to_row(venue))

    try:
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range="A:Q"
        ).execute()

        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": rows}
        ).execute()

        print(f"Updated metadata for {updated_count} venues")

    except Exception as e:
        print(f"Error batch updating venue metadata: {e}")


def fetch_venue_events(
    venue: dict,
    force_refresh: bool = False,
    city: str = "NYC",
    state_code: str = "NY",
    skip_metadata_update: bool = False,
) -> FetchResult:
    """
    Fetch events for a single venue using optimal strategy.

    Args:
        venue: Venue dict with name, category, website, etc.
        force_refresh: If True, fetch even if cache is fresh
        city: City name
        state_code: State code
        skip_metadata_update: If True, don't update venue metadata (for batch updates)

    Returns:
        FetchResult with events or error
    """
    venue_name = venue.get("name", "")
    category = venue.get("category", "")
    website = venue.get("website", "")
    events_url = venue.get("events_url", "")
    preferred_source = venue.get("preferred_event_source", "")
    ticketmaster_venue_id = venue.get("ticketmaster_venue_id", "")

    result = FetchResult(venue_name=venue_name)
    increment("event_fetcher.fetch_venue.calls")
    log_event(
        "event_fetch_start",
        venue_name=venue_name,
        city=city,
        force_refresh=force_refresh,
    )

    # Check cache freshness
    if not force_refresh and is_venue_events_fresh(venue_name, city):
        result.skipped = True
        result.skip_reason = "cached"
        increment("event_fetcher.fetch_venue.cached")
        log_event("event_fetch_cached", venue_name=venue_name, city=city)
        return result

    print(f"  Fetching events for: {venue_name}")

    # Determine strategy
    strategy = determine_fetch_strategy(venue_name, category, website, preferred_source)
    print(f"    Strategy: {strategy}")

    tm_events = None
    scrape_events = None

    # Execute strategy
    if strategy in ("ticketmaster_only", "both"):
        result.attempted_sources.append("ticketmaster")
        delay = getattr(_settings, "VENUE_FETCH_DELAY", 0.5)
        time.sleep(delay)
        tm_events = _fetch_from_ticketmaster(venue_name, city, state_code, venue_id=ticketmaster_venue_id)
        if tm_events:
            tm_events = _normalize_tm_events(tm_events, venue_name)
            tm_events = _normalize_event_batch(tm_events, venue_name, "ticketmaster")
            print(f"    Ticketmaster: {len(tm_events)} events")
            increment("event_fetcher.source.ticketmaster.success")
        else:
            result.source_errors["ticketmaster"] = "no_events_or_fetch_error"
            increment("event_fetcher.source.ticketmaster.empty")

    website_source = None  # Track whether we used 'api' or 'scrape'

    if strategy in ("scrape_only", "both", "scrape_first"):
        if website:
            result.attempted_sources.append("website")
            delay = getattr(_settings, "VENUE_FETCH_DELAY", 0.5)
            time.sleep(delay)

            # Check for known API endpoint first
            api_endpoint = venue.get("api_endpoint", "")

            if api_endpoint:
                print(f"    Using known API endpoint...")
                scrape_events = _fetch_from_api(api_endpoint, venue_name)
                if scrape_events:
                    scrape_events = _normalize_event_batch(scrape_events, venue_name, "api")
                    print(f"    API: {len(scrape_events)} events")
                    website_source = "api"
                    increment("event_fetcher.source.api.success")
                else:
                    result.source_errors["api"] = "known_endpoint_returned_no_events_or_failed"
                    increment("event_fetcher.source.api.empty")

            # Use Jina scraping if no known API or API failed
            if not scrape_events:
                scrape_events = _fetch_from_website(
                    website,
                    venue_name,
                    events_url,
                    venue_context=venue,
                )
                if scrape_events:
                    scrape_events = _normalize_event_batch(scrape_events, venue_name, "scrape")
                    print(f"    Scrape: {len(scrape_events)} events")
                    website_source = "scrape"
                    increment("event_fetcher.source.scrape.success")
                else:
                    result.source_errors["scrape"] = "website_scrape_returned_no_events_or_failed"
                    increment("event_fetcher.source.scrape.empty")
        else:
            result.source_errors["website"] = "missing_website_url"

    # Select best source
    source_used, events = select_best_source(tm_events, scrape_events)

    # Override source_used with actual source if we got website events
    if source_used == "scrape" and website_source:
        source_used = website_source

    # If both had results, merge and deduplicate
    if strategy == "both" and tm_events and scrape_events:
        events = _deduplicate_events(tm_events + scrape_events)
        source_used = "both" if website_source == "scrape" else f"ticketmaster+{website_source}"

    # Relevance filter for multi-location feeds.
    events = _filter_events_for_venue(events, venue)

    result.events = events
    result.source_used = source_used

    if not events:
        warning = (
            f"No events found via strategy '{strategy}'. "
            f"Source errors: {result.source_errors or {'all': 'no_events'}}"
        )
        result.warnings.append(warning)
        log_event(
            "event_fetch_empty",
            venue_name=venue_name,
            city=city,
            strategy=strategy,
            source_errors=result.source_errors,
        )
    else:
        increment("event_fetcher.fetch_venue.success")
        log_event(
            "event_fetch_success",
            venue_name=venue_name,
            city=city,
            strategy=strategy,
            source_used=source_used,
            event_count=len(events),
        )

    # Mark as fetched (skip in parallel mode - will batch update later)
    if not skip_metadata_update:
        mark_venue_fetched(venue_name, city, len(events), source_used)

    print(f"    Result: {len(events)} events (source: {source_used})")

    return result


def _fetch_single_venue(args: tuple) -> tuple[str, FetchResult]:
    """Helper function for parallel venue fetching."""
    venue, force_refresh, city, state_code, index, total, use_local_cache = args
    venue_name = venue.get("name", "")

    print(f"[{index}/{total}] {venue_name}", flush=True)

    result = fetch_venue_events(
        venue=venue,
        force_refresh=force_refresh,
        city=city,
        state_code=state_code,
        skip_metadata_update=use_local_cache,  # Skip per-venue updates in parallel mode
    )

    # Save to local cache if using parallel mode
    if use_local_cache and result.events:
        from .local_event_cache import add_events
        add_events(result.events, venue_name)

    return venue_name, result


def fetch_events_for_venues(
    venues: list[dict],
    force_refresh: bool = False,
    city: str = "NYC",
    state_code: str = "NY",
    save_to_sheet: bool = True,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, FetchResult]:
    """
    Fetch events for multiple venues.

    Args:
        venues: List of Venue dicts
        force_refresh: If True, fetch all venues regardless of cache
        city: City name
        state_code: State code
        save_to_sheet: If True, save results to Google Sheet
        workers: Number of parallel workers (1 = sequential)
        resume: If True, resume from previous interrupted run

    Returns:
        Dict mapping venue_name -> FetchResult
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    use_local_cache = workers > 1  # Use local cache for parallel fetching

    # Handle resume
    if resume and use_local_cache:
        from .local_event_cache import resume_info, get_fetched_venues
        info = resume_info()
        if info:
            print(f"Resuming previous session:")
            print(f"  Started: {info['started_at']}")
            print(f"  Venues fetched: {info['venues_fetched']}")
            print(f"  Events collected: {info['events_collected']}")

            # Filter out already-fetched venues
            fetched_names = set(v.lower() for v in get_fetched_venues())
            venues = [v for v in venues if v.get("name", "").lower() not in fetched_names]
            print(f"  Remaining venues: {len(venues)}")
        else:
            print("No previous session to resume, starting fresh")
            resume = False

    # Start new session if not resuming
    if use_local_cache and not resume:
        from .local_event_cache import start_fetch_session
        start_fetch_session(city)

    # Filter to stale venues unless force refresh
    if not force_refresh:
        stale_venues = get_stale_venues(venues, city)
        print(f"Found {len(stale_venues)} venues needing refresh (out of {len(venues)})")
        venues_to_fetch = stale_venues
    else:
        venues_to_fetch = venues
        print(f"Force refresh: fetching all {len(venues)} venues")

    if not venues_to_fetch:
        print("No venues to fetch")
        # Still merge if resuming with no new venues
        if save_to_sheet and use_local_cache:
            from .local_event_cache import merge_to_sheets, has_pending_events
            if has_pending_events():
                merge_to_sheets()
        return results

    total = len(venues_to_fetch)

    if workers > 1:
        print(f"Using {workers} parallel workers")
        print(f"Events saved locally, merged to sheets every 60s")

        # Enable batch mode to skip per-venue sheet writes (avoids rate limits)
        os.environ["EVENT_FETCHER_BATCH_MODE"] = "1"

        # Pre-load venues into memory cache (one-time sheet read)
        from .local_event_cache import load_venues_once
        load_venues_once(city)

        # Prepare arguments for parallel execution
        fetch_args = [
            (venue, force_refresh, city, state_code, i, total, use_local_cache)
            for i, venue in enumerate(venues_to_fetch, 1)
        ]

        # Execute in parallel - events saved to local cache, merged to sheets periodically
        all_events = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_single_venue, args): args for args in fetch_args}

            for future in as_completed(futures):
                try:
                    venue_name, result = future.result()
                    results[venue_name] = result
                    if result.events:
                        all_events.extend(result.events)
                    completed_count += 1

                    # Progress update and rate-limited sheet write every 100 venues
                    if completed_count % 100 == 0:
                        from .local_event_cache import get_cache_stats, rate_limited_merge_to_sheets
                        stats = get_cache_stats()
                        print(f"\n--- Progress: {completed_count}/{total} venues, {stats['event_count']} events cached ---")

                        # Try to merge to sheets (will skip if <60s since last write)
                        if save_to_sheet:
                            written = rate_limited_merge_to_sheets()
                            if written > 0:
                                print(f"--- Saved {written} events to Google Sheets ---\n")
                            else:
                                print(f"--- Sheet write skipped (rate limited) ---\n")

                except Exception as e:
                    args = futures[future]
                    venue_name = args[0].get("name", "unknown")
                    print(f"  Error fetching {venue_name}: {e}", flush=True)
                    results[venue_name] = FetchResult(venue_name=venue_name, error=str(e))

        # Final merge of any remaining events (force write, ignore rate limit)
        if save_to_sheet:
            from .local_event_cache import force_merge_to_sheets, get_cache_stats
            stats = get_cache_stats()
            print(f"\nLocal cache: {stats['event_count']} events from {stats['venues_fetched']} venues")
            force_merge_to_sheets()

        # Batch update venue metadata (last_event_fetch, event_count, event_source)
        print("Updating venue metadata...")
        _batch_update_venue_metadata(results, city)

    else:
        # Sequential execution (original behavior - no local cache needed)
        all_events = []
        for i, venue in enumerate(venues_to_fetch, 1):
            venue_name = venue.get("name", "")
            print(f"\n[{i}/{total}] {venue_name}")

            result = fetch_venue_events(
                venue=venue,
                force_refresh=force_refresh,
                city=city,
                state_code=state_code,
            )
            results[venue_name] = result

            if result.events:
                all_events.extend(result.events)

        # Save to sheet (sequential mode - write per venue is OK)
        if save_to_sheet and all_events:
            print(f"\nSaving {len(all_events)} events to Google Sheet...")
            for venue_name, result in results.items():
                if result.events:
                    append_venue_events(result.events, venue_name)

    # Summary
    fetched = sum(1 for r in results.values() if r.events)
    skipped = sum(1 for r in results.values() if r.skipped)
    errors = sum(1 for r in results.values() if r.error)
    total_events = sum(len(r.events) for r in results.values())

    print(f"\n{'='*50}")
    print(f"Fetch complete: {fetched} venues, {total_events} events")
    print(f"Skipped (cached): {skipped}")
    if errors:
        print(f"Errors: {errors}")

    return results


def fetch_events_by_category(
    category: str,
    city: str = "NYC",
    force_refresh: bool = False,
) -> dict[str, FetchResult]:
    """
    Fetch events for all venues in a category.

    Args:
        category: Venue category to fetch
        city: City name
        force_refresh: If True, fetch regardless of cache

    Returns:
        Dict mapping venue_name -> FetchResult
    """
    from .cache import read_cached_venues

    # Get venues in this category
    all_venues = read_cached_venues(city)
    category_venues = [
        v for v in all_venues
        if v.get("category", "").lower() == category.lower()
    ]

    if not category_venues:
        print(f"No venues found in category: {category}")
        return {}

    print(f"Found {len(category_venues)} venues in category: {category}")
    return fetch_events_for_venues(category_venues, force_refresh, city)


if __name__ == "__main__":
    # Test with a single venue
    test_venue = {
        "name": "Beacon Theatre",
        "category": "concert halls",
        "website": "",
        "city": "NYC",
    }

    print("Testing event fetcher with Beacon Theatre...")
    result = fetch_venue_events(test_venue, force_refresh=True)
    print(f"\nResult: {len(result.events)} events")
    for event in result.events[:5]:
        print(f"  - {event.get('date_str')}: {event.get('name')}")
