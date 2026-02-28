"""Main orchestrator for fetching events from venues.

Coordinates between Ticketmaster and website scraping based on
venue category and characteristics.
"""

import html as html_lib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from .source_strategy import determine_fetch_strategy, select_best_source
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
_SHARED_FEED_CACHE: dict[str, list[dict]] = {}
_SHARED_FEED_INFLIGHT: dict[str, threading.Event] = {}
_SHARED_FEED_LOCK = threading.Lock()


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


def _is_timeout_error(exc: Exception) -> bool:
    """Return True when error text indicates provider timeout/deadline."""
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


def _is_cloudflare_response(response) -> bool:
    """Return True if the response is a Cloudflare block or challenge page."""
    if "cloudflare" in response.headers.get("server", "").lower():
        return True
    if "cf-ray" in response.headers:
        return True
    body = response.text[:2000].lower()
    if "just a moment" in body and "cloudflare" in body:
        return True
    return False


def _fetch_raw_html(url: str) -> str:
    """Fetch raw HTML for structured extraction and iframe discovery.

    Retries with curl_cffi impersonating Chrome when a Cloudflare 403 is detected.
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    timeout = int(_settings.EVENT_FETCHER_HTML_TIMEOUT_SEC)
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    if response.status_code == 403 and _is_cloudflare_response(response):
        time.sleep(1.0)  # Brief pause before retry — avoids burst on CF-protected sites
        from curl_cffi import requests as cf_requests
        cf_resp = cf_requests.get(url, impersonate="chrome124", timeout=timeout, allow_redirects=True)
        cf_resp.raise_for_status()
        return cf_resp.text
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

    Uses BeautifulSoup for robust handling of malformed markup.
    """
    if not html:
        return ""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Drop elements that contribute no readable content
    for tag in soup(["script", "style", "noscript", "head", "meta", "link"]):
        tag.decompose()

    # Get text with newlines between block elements for readability
    text = soup.get_text(separator="\n")

    # Collapse multiple whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


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
                    "description": _strip_tags(str(item.get("description", "")))[:200],
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
    generic_tokens = {
        "new",
        "york",
        "nyc",
        "city",
        "park",
        "parks",
        "playground",
        "beach",
        "garden",
        "gardens",
        "historic",
        "house",
        "houses",
        "center",
        "centre",
        "community",
        "field",
        "plaza",
        "square",
        "north",
        "south",
        "east",
        "west",
        "street",
        "avenue",
        "road",
        "drive",
        "lane",
        "trail",
        "riverside",
        "central",
        "brooklyn",
        "manhattan",
        "bronx",
        "queens",
        "staten",
        "island",
        "the",
        "venue",
    }
    text = " ".join(
        [
            venue.get("name", ""),
            venue.get("address", ""),
            venue.get("neighborhood", ""),
            venue.get("city", ""),
        ]
    ).lower()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text)
        if token not in generic_tokens
    }
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
            str(event.get("park_names", "")),
            str(event.get("location", "")),
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


def _is_shared_multi_location_feed(url: str) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    domain = parsed.netloc.lower().replace("www.", "")
    path = (parsed.path or "").rstrip("/").lower()
    if domain.endswith("nycgovparks.org") and path in ("/events", "/events/volunteer"):
        return True
    return False


def _is_nyc_parks_shared_feed(url: str) -> bool:
    """True when URL is the generic NYC Parks shared events listing."""
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    domain = parsed.netloc.lower().replace("www.", "")
    path = (parsed.path or "").rstrip("/").lower()
    return domain.endswith("nycgovparks.org") and path in ("/events", "/events/volunteer")


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities for cleaner event descriptions."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", str(text))
    decoded = html_lib.unescape(no_tags)
    return re.sub(r"\s+", " ", decoded).strip()


def _parse_nyc_parks_datetime(date_str: str, time_str: str) -> datetime | None:
    """Parse NYC Parks feed date/time into timezone-aware datetime."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    time_str = str(time_str or "").strip()
    if time_str:
        for fmt in ("%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", fmt)
                return dt.replace(tzinfo=ZoneInfo("America/New_York"))
            except ValueError:
                continue
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=ZoneInfo("America/New_York"))
    except ValueError:
        return None


def _fetch_nyc_parks_open_data_events() -> list[dict]:
    """
    Fetch upcoming NYC Parks events from the official open-data JSON feed.

    Uses the dataset linked from NYC Open Data:
    https://data.cityofnewyork.us/City-Government/NYC-Parks-Events-Listing-Event-Listing/fudw-fgrp/about_data
    """
    import requests

    feed_url = str(
        getattr(
            _settings,
            "NYC_PARKS_OPEN_DATA_FEED_URL",
            "https://www.nycgovparks.org/xml/events_300_rss.json",
        )
    ).strip()
    if not feed_url:
        return []

    timeout = int(getattr(_settings, "EVENT_FETCHER_HTML_TIMEOUT_SEC", 20))
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    try:
        response = requests.get(feed_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        record_failure(
            "event_fetcher.nyc_parks_open_data",
            str(e),
            feed_url=feed_url,
        )
        return []

    if not isinstance(payload, list):
        return []

    events: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    one_year_out = today + timedelta(days=366)

    for row in payload:
        if not isinstance(row, dict):
            continue

        title = str(row.get("title", "")).strip()
        if not title:
            continue

        start_date = str(row.get("startdate", "")).strip()
        if not start_date:
            continue

        dt = _parse_nyc_parks_datetime(start_date, str(row.get("starttime", "")))
        event_date = dt.date() if dt else None
        if event_date and (event_date < today or event_date > one_year_out):
            continue

        park_names = str(row.get("parknames", "")).strip()
        park_id = str(row.get("parkids", "")).strip()
        location = str(row.get("location", "")).strip()
        categories = str(row.get("categories", "")).strip()
        link = str(row.get("link", "")).strip()
        if link and not re.match(r"^https?://", link, re.IGNORECASE):
            link = f"https://www.nycgovparks.org{link}" if link.startswith("/") else f"http://{link}"

        date_str = start_date[:10]
        dedupe_key = (title.lower(), date_str, park_names.lower())
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        description = _strip_tags(str(row.get("description", "") or row.get("snippet", "")))
        event_type = "event"
        if categories:
            event_type = categories.split("|", 1)[0].strip() or "event"

        events.append(
            {
                "name": title,
                "datetime": dt,
                "date_str": date_str,
                "venue_name": park_names or location,
                "park_id": park_id,
                "park_names": park_names,
                "location": location,
                "event_type": event_type,
                "categories": categories,
                "url": link,
                "source": "nyc_parks_open_data",
                "matched_artist": "",
                "travel_minutes": None,
                "description": description,
                "event_source_url": feed_url,
                "extraction_method": "nyc_parks_open_data_json",
                "validation_confidence": 0.98,
            }
        )

    log_event(
        "event_fetch_nyc_parks_open_data",
        feed_url=feed_url,
        event_count=len(events),
    )
    return events


def _normalize_venue_name_key(name: str) -> str:
    """Normalize venue names for dedupe/upsert matching."""
    raw = str(name or "").lower().strip()
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"\s+(nyc|ny|club|venue|theater|theatre)$", "", raw)
    raw = re.sub(r"^the\s+", "", raw)
    return raw.strip()


def _park_name_from_open_data_event(event: dict) -> str:
    """Extract best park/venue name from NYC Parks open-data event."""
    park_names = str(event.get("park_names", "")).strip()
    if park_names:
        return park_names

    location = str(event.get("location", "")).strip()
    if location:
        match = re.search(r"\(in\s+([^)]+)\)", location, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _event_park_name_key(event: dict) -> str:
    """Best-effort normalized park name key from NYC Parks event rows."""
    for field in ("park_names", "venue_name", "location"):
        raw = str(event.get(field, "") or "").strip()
        if not raw:
            continue
        if field == "location":
            match = re.search(r"\(in\s+([^)]+)\)", raw, flags=re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
        key = _normalize_venue_name_key(raw)
        if key:
            return key
    return ""


def _is_nyc_parks_open_data_event(event: dict) -> bool:
    """Detect events produced by the NYC Parks open-data ingestion path."""
    source = str(event.get("source", "") or "").strip().lower()
    method = str(event.get("extraction_method", "") or "").strip().lower()
    return source == "nyc_parks_open_data" or method == "nyc_parks_open_data_json"


def sync_nyc_parks_venues_from_open_data(city: str = "NYC") -> tuple[int, int]:
    """
    Ensure NYC Parks venues from open data exist in Venue Scout cache.

    Returns:
        (added_count, updated_count)
    """
    if str(city or "").strip().lower() != "nyc":
        return 0, 0

    events = _fetch_nyc_parks_open_data_events()
    if not events:
        return 0, 0

    from .cache import read_cached_venues, append_venues_to_cache, update_venues_batch

    existing = read_cached_venues(city)
    existing_by_key = {_normalize_venue_name_key(v.get("name", "")): v for v in existing}

    shared_events_url = "https://www.nycgovparks.org/events"
    website_default = "https://www.nycgovparks.org"
    category_default = "nyc parks with free events"

    to_add: list[dict] = []
    to_update: list[dict] = []
    added_keys: set[str] = set()

    for event in events:
        park_name = _park_name_from_open_data_event(event)
        key = _normalize_venue_name_key(park_name)
        if not key:
            continue

        if key in existing_by_key:
            current = existing_by_key[key]
            patch: dict = {"name": current.get("name", park_name), "city": city}
            changed = False

            if not str(current.get("events_url", "")).strip():
                patch["events_url"] = shared_events_url
                changed = True
            if not str(current.get("website", "")).strip():
                patch["website"] = website_default
                changed = True
            if not str(current.get("website_status", "")).strip():
                patch["website_status"] = "verified"
                changed = True
            if not str(current.get("preferred_event_source", "")).strip():
                patch["preferred_event_source"] = "scrape"
                changed = True

            if changed:
                to_update.append(patch)
            continue

        if key in added_keys:
            continue
        added_keys.add(key)
        to_add.append(
            {
                "name": park_name,
                "address": city,
                "city": city,
                "neighborhood": "",
                "website": website_default,
                "events_url": shared_events_url,
                "category": category_default,
                "description": "Auto-synced NYC Parks venue from open-data events feed.",
                "source": "nyc_parks_open_data",
                "address_verified": "",
                "website_status": "verified",
                "website_attempts": 0,
                "preferred_event_source": "scrape",
                "api_endpoint": "",
                "ticketmaster_venue_id": "",
                "last_event_fetch": "",
                "event_count": 0,
                "event_source": "",
            }
        )

    if to_add:
        append_venues_to_cache(to_add, city, "nyc_parks_open_data")
        # Refresh existing map so update lookups stay accurate after append.
        existing = read_cached_venues(city)
        existing_by_key = {_normalize_venue_name_key(v.get("name", "")): v for v in existing}

    updated_count = 0
    if to_update:
        updated_count = update_venues_batch(to_update, city)

    log_event(
        "nyc_parks_venues_synced",
        city=city,
        added_count=len(to_add),
        updated_count=updated_count,
        source_event_count=len(events),
    )
    return len(to_add), updated_count


def _canonical_feed_key(url: str) -> str:
    """Normalize feed URL into a stable key for per-run cache reuse."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.lower()
    host = parsed.netloc.lower().replace("www.", "")
    path = (parsed.path or "/").rstrip("/") or "/"
    path = path.lower()
    query = f"?{parsed.query.lower()}" if parsed.query else ""
    return f"{host}{path}{query}"


def _clone_events(events: list[dict]) -> list[dict]:
    """Return shallow copies so per-venue scoring mutations don't leak across venues."""
    return [dict(event) for event in events]


def _get_or_fetch_shared_feed_events(
    feed_url: str,
    fetch_once_fn,
    venue_name: str,
) -> tuple[list[dict], bool]:
    """
    Fetch shared feed once per run and reuse for all venues with same feed URL.

    Returns:
        (events, from_cache)
    """
    key = _canonical_feed_key(feed_url)
    if not key:
        return fetch_once_fn(), False

    while True:
        with _SHARED_FEED_LOCK:
            cached = _SHARED_FEED_CACHE.get(key)
            if cached is not None:
                increment("event_fetcher.shared_feed.cache_hit")
                log_event(
                    "event_fetch_shared_feed_cache_hit",
                    venue_name=venue_name,
                    feed_key=key,
                )
                return _clone_events(cached), True

            inflight = _SHARED_FEED_INFLIGHT.get(key)
            if inflight is None:
                inflight = threading.Event()
                _SHARED_FEED_INFLIGHT[key] = inflight
                is_fetcher = True
            else:
                is_fetcher = False

        if is_fetcher:
            break
        inflight.wait(timeout=120)

    fetched: list[dict] = []
    try:
        fetched = fetch_once_fn() or []
        with _SHARED_FEED_LOCK:
            _SHARED_FEED_CACHE[key] = _clone_events(fetched)
        increment("event_fetcher.shared_feed.cache_miss")
        log_event(
            "event_fetch_shared_feed_cached",
            venue_name=venue_name,
            feed_key=key,
            event_count=len(fetched),
        )
        return _clone_events(fetched), False
    finally:
        with _SHARED_FEED_LOCK:
            marker = _SHARED_FEED_INFLIGHT.pop(key, None)
            if marker:
                marker.set()


def _filter_events_for_venue(events: list[dict], venue: dict) -> list[dict]:
    """Filter out low-relevance events from shared feeds."""
    if not events:
        return []

    if any(_is_nyc_parks_open_data_event(event) for event in events):
        venue_key = _normalize_venue_name_key(venue.get("name", ""))
        exact = []
        for event in events:
            park_key = _event_park_name_key(event)
            if park_key and venue_key and park_key == venue_key:
                event["relevance_score"] = 100
                exact.append(event)
        if exact:
            return exact
        # If park name is missing in feed row, do not spray it to arbitrary park venues.
        return []

    shared_feed = any(
        _is_shared_multi_location_feed(event.get("event_source_url", "") or event.get("url", ""))
        for event in events
    )
    threshold = 35 if shared_feed else 20
    filtered = []
    for event in events:
        score = _score_event_relevance(event, venue)
        event["relevance_score"] = score
        # Keep high-confidence matches; allow unknown location pages to pass at lower score if tiny set.
        if score >= threshold:
            filtered.append(event)

    if not filtered:
        if shared_feed:
            # Shared citywide feeds should not "spray" events to venue pages.
            return []
        # Avoid dropping all events for tiny pages with sparse metadata.
        if len(events) <= 2:
            return events
        # For larger mixed feeds, keep only stronger candidates.
        ranked = sorted(events, key=lambda e: e.get("relevance_score", 0), reverse=True)
        if ranked and ranked[0].get("relevance_score", 0) >= 12:
            return [event for event in ranked[: min(5, len(ranked))]]
        return []
    return filtered


def _fetch_from_feed(
    feed_url: str,
    feed_type: str,
    venue_name: str,
) -> list[dict]:
    """Fetch and normalize events from an iCal or RSS feed."""
    from .feed_finder import fetch_and_parse_feed
    from datetime import date, timedelta

    raw_events = fetch_and_parse_feed(feed_url, feed_type)
    if not raw_events:
        return []

    today = date.today()
    cutoff = today + timedelta(days=365)
    events = []

    for e in raw_events:
        date_raw = e.get("date_str", "")
        try:
            event_date = datetime.strptime(date_raw, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Skip past events and events beyond 1 year (e.g. permanent exhibitions)
        if event_date < today or event_date > cutoff:
            continue

        time_raw = e.get("time", "")
        dt = None
        if time_raw:
            try:
                dt = datetime.strptime(f"{date_raw} {time_raw}", "%Y-%m-%d %H:%M")
            except ValueError:
                pass

        events.append(_normalize_event_schema(
            {
                "name": e.get("title", ""),
                "datetime": dt,
                "date_str": date_raw,
                "end_date": e.get("end_date", ""),
                "venue_name": venue_name,
                "event_type": "",
                "url": e.get("event_url", ""),
                "source": f"feed_{feed_type}",
                "matched_artist": "",
                "travel_minutes": None,
                "description": e.get("description", ""),
                "event_source_url": feed_url,
                "extraction_method": f"feed_{feed_type}",
                "relevance_score": None,
                "validation_confidence": 1.0,
            },
            venue_name,
            f"feed_{feed_type}",
        ))

    return events


def _fetch_from_website(
    url: str,
    venue_name: str,
    events_url: str = "",
    venue_context: dict | None = None,
    default_event_venue_name: str | None = None,
) -> list[dict]:
    """
    Fetch events by scraping venue website.

    Uses structured extraction first (JSON-LD), then LLM extraction.
    Falls back to iframe pages when embedded event widgets are used.
    If events_url is provided, fetches from that instead of the homepage.

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
    fallback_venue_name = venue_name if default_event_venue_name is None else default_event_venue_name

    if not fetch_url:
        return []

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
            if raw_html:
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

        # Try JSON-LD extraction first
        if raw_html:
            structured_events = _extract_events_from_jsonld(raw_html, fallback_venue_name, fetch_url)
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
                    iframe_events = _extract_events_from_jsonld(iframe_html, fallback_venue_name, iframe_url)
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
        events = _parse_events_with_llm(content, venue_name, fetch_url, default_venue_name=fallback_venue_name)
        if events:
            # Mark extraction method based on source used
            if used_playwright:
                extraction_method = "llm_parse_playwright"
            else:
                extraction_method = "llm_parse_raw_html"
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
                    events = _parse_events_with_llm(
                        pw_text,
                        venue_name,
                        fetch_url,
                        default_venue_name=fallback_venue_name,
                    )
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
                    iframe_html = _fetch_raw_html(iframe_url)
                    iframe_content = _strip_html_for_llm(iframe_html)

                    iframe_events = _parse_events_with_llm(
                        iframe_content,
                        venue_name,
                        iframe_url,
                        default_venue_name=fallback_venue_name,
                    )
                    if iframe_events:
                        for event in iframe_events:
                            event["event_source_url"] = iframe_url
                            event["extraction_method"] = "llm_iframe_raw_html"
                        return iframe_events
                except Exception:
                    continue

        # Try Google Calendar iframe extraction as last resort
        if raw_html:
            gcal_events = _extract_google_calendar_events(raw_html, fallback_venue_name)
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


def _build_event_extraction_prompt(content: str, today_iso: str, chunk_label: str = "") -> str:
    """Build LLM prompt for event extraction from page text."""
    chunk_prefix = f"{chunk_label}\n" if chunk_label else ""
    return f"""{chunk_prefix}Extract upcoming events from this webpage content. Today's date is {today_iso}.

WEBPAGE CONTENT:
{content}

Extract each event and return as a JSON array. For each event include:
- name: Event/show name
- date_str: Start date in YYYY-MM-DD format (infer year if not shown, assume current/next year)
- end_date: End date in YYYY-MM-DD format for multi-day events like exhibitions, festivals, or runs (empty string if single-day)
- time: Time if available (e.g. "7:00 PM")
- venue_name: Venue/location shown for the event (empty string if missing)
- event_type: Type of event (concert, comedy, theater, exhibition, festival, reading, etc.)
- url: Event URL if available, otherwise empty string
- description: Brief description if available

Only include events whose start date OR end date is on or after today ({today_iso}).
If this chunk has no events, return: []

Return ONLY the JSON array, no other text."""


def _extract_json_array_payload(response: str) -> list[dict]:
    """Extract first valid JSON array from a model response."""
    text = str(response or "").strip()
    if not text:
        return []

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates = fenced + re.findall(r"\[[\s\S]*\]", text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return [row for row in obj if isinstance(row, dict)]
        except Exception:
            continue
    return []


def _split_content_for_llm(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> list[str]:
    """Split long content into bounded chunks, preferring natural boundaries."""
    text = str(content or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len and len(chunks) < max_chunks:
        end = min(text_len, start + chunk_size)
        if end < text_len:
            boundary = text.rfind("\n\n", start + int(chunk_size * 0.55), end)
            if boundary == -1:
                boundary = text.rfind(". ", start + int(chunk_size * 0.55), end)
                if boundary != -1:
                    boundary += 1
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - max(0, chunk_overlap))
    return chunks


def _looks_eventish_chunk(text: str) -> bool:
    """Heuristic to prioritize chunks likely to contain event listings."""
    value = str(text or "")
    if not value:
        return False
    patterns = (
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
        r"\b20\d{2}-\d{2}-\d{2}\b",
        r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b",
        r"\b(rsvp|ticket|register|event|workshop|talk|lecture|reading)\b",
    )
    lowered = value.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def _invoke_event_llm_with_fallback(
    prompt: str,
    parse_timeout_sec: int,
    parse_max_retries: int,
    primary_model: str,
    timeout_fallback_models: list[str],
) -> str:
    """Call event extraction LLM with timeout-based fallback model chain."""
    from utils.llm import generate_content

    try:
        return generate_content(
            prompt,
            max_retries=max(1, parse_max_retries),
            timeout_sec=max(5, parse_timeout_sec),
            model_name=primary_model,
        )
    except Exception as primary_exc:
        if not _is_timeout_error(primary_exc):
            raise

        fallback_error: Exception = primary_exc
        for fallback_model in timeout_fallback_models:
            try:
                print(f"    Primary parse model timed out; retrying with {fallback_model}...")
                return generate_content(
                    prompt,
                    max_retries=1,
                    timeout_sec=max(5, parse_timeout_sec),
                    model_name=fallback_model,
                )
            except Exception as exc:
                fallback_error = exc
        raise fallback_error


def _normalize_llm_event_rows(
    rows: list[dict],
    venue_name: str,
    source_url: str,
    fallback_venue_name: str,
    extraction_method: str,
) -> list[dict]:
    """Convert extracted rows into canonical event objects."""
    out: list[dict] = []
    today = datetime.now(ZoneInfo("America/New_York")).date()

    for row in rows:
        name = str(row.get("name", "") or "").strip()
        if not name:
            continue
        date_raw = str(row.get("date_str", "") or "").strip()
        time_raw = str(row.get("time", "") or "").strip()
        dt = None

        if date_raw and time_raw:
            dt = _parse_date_flexible(f"{date_raw} {time_raw}")
        if dt is None and date_raw:
            dt = _parse_date_flexible(date_raw)

        normalized_date = date_raw
        if dt is not None:
            normalized_date = dt.strftime("%Y-%m-%d")
        elif date_raw:
            parsed = _parse_date_flexible(date_raw)
            if parsed:
                dt = parsed
                normalized_date = parsed.strftime("%Y-%m-%d")

        if dt is not None and dt.date() < today:
            continue
        if normalized_date:
            try:
                parsed_date = datetime.strptime(normalized_date, "%Y-%m-%d").date()
                if parsed_date < today:
                    continue
            except Exception:
                pass

        parsed_venue_name = str(row.get("venue_name", "") or "").strip()
        out.append(
            {
                "name": name,
                "datetime": dt,
                "date_str": normalized_date,
                "end_date": str(row.get("end_date", "") or "").strip(),
                "venue_name": parsed_venue_name or fallback_venue_name or venue_name,
                "event_type": str(row.get("event_type", "") or "").strip(),
                "url": str(row.get("url", "") or "").strip() or source_url,
                "source": "scrape",
                "matched_artist": "",
                "travel_minutes": None,
                "description": _strip_tags(str(row.get("description", "") or "")),
                "event_source_url": source_url,
                "extraction_method": extraction_method,
                "validation_confidence": 0.6,
            }
        )

    return out


def _dedupe_llm_events(events: list[dict]) -> list[dict]:
    """Deduplicate LLM extracted events by normalized name/date/url."""
    best: dict[tuple[str, str, str], dict] = {}
    for event in events:
        key = (
            str(event.get("name", "")).strip().lower(),
            str(event.get("date_str", "")).strip(),
            str(event.get("url", "")).strip().lower(),
        )
        existing = best.get(key)
        if existing is None:
            best[key] = event
            continue
        current_desc = str(event.get("description", "") or "")
        existing_desc = str(existing.get("description", "") or "")
        if len(current_desc) > len(existing_desc):
            best[key] = event
    return list(best.values())


def _parse_events_with_llm(
    content: str,
    venue_name: str,
    source_url: str,
    default_venue_name: str | None = None,
) -> list[dict]:
    """
    Use LLM to extract events from webpage content.

    Returns:
        List of event dicts
    """
    try:
        parse_timeout_sec = int(getattr(_settings, "EVENT_PARSE_LLM_TIMEOUT_SEC", 12))
        parse_max_retries = int(getattr(_settings, "EVENT_PARSE_LLM_MAX_RETRIES", 1))
        primary_model = str(getattr(_settings, "GEMINI_MODEL", "gemini-3-flash-preview")).strip()
        timeout_fallback_models = _fallback_model_list(
            getattr(
                _settings,
                "EVENT_PARSE_LLM_TIMEOUT_FALLBACK_MODELS",
                ("gemini-2.5-flash", "gemini-2.0-flash"),
            ),
            default=("gemini-2.5-flash", "gemini-2.0-flash"),
        )
        timeout_fallback_models = [m for m in timeout_fallback_models if m != primary_model]

        chunk_threshold = int(getattr(_settings, "EVENT_PARSE_LLM_CHUNK_THRESHOLD", 7000))
        chunk_size = int(getattr(_settings, "EVENT_PARSE_LLM_CHUNK_SIZE", 5000))
        chunk_overlap = int(getattr(_settings, "EVENT_PARSE_LLM_CHUNK_OVERLAP", 350))
        max_chunks = int(getattr(_settings, "EVENT_PARSE_LLM_MAX_CHUNKS", 8))
        today_iso = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        fallback_venue_name = venue_name if default_venue_name is None else default_venue_name

        def _parse_block(block: str, chunk_label: str, extraction_method: str) -> list[dict]:
            prompt = _build_event_extraction_prompt(block, today_iso=today_iso, chunk_label=chunk_label)
            response = _invoke_event_llm_with_fallback(
                prompt=prompt,
                parse_timeout_sec=parse_timeout_sec,
                parse_max_retries=parse_max_retries,
                primary_model=primary_model,
                timeout_fallback_models=timeout_fallback_models,
            )
            payload = _extract_json_array_payload(response)
            if not payload:
                return []
            return _normalize_llm_event_rows(
                rows=payload,
                venue_name=venue_name,
                source_url=source_url,
                fallback_venue_name=fallback_venue_name,
                extraction_method=extraction_method,
            )

        text = str(content or "").strip()
        if not text:
            return []

        if len(text) < chunk_threshold:
            events = _parse_block(text, chunk_label="", extraction_method="llm_parse")
            if events:
                return _dedupe_llm_events(events)

        chunks = _split_content_for_llm(
            content=text,
            chunk_size=max(1200, chunk_size),
            chunk_overlap=max(0, chunk_overlap),
            max_chunks=max(1, max_chunks),
        )
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        eventish = [chunk for chunk in chunks if _looks_eventish_chunk(chunk)]
        if eventish:
            chunks = eventish[:max_chunks]

        if not chunks:
            return []

        print(f"    Using chunked LLM extraction: {len(chunks)} chunk(s)")
        collected: list[dict] = []
        for idx, chunk in enumerate(chunks, start=1):
            try:
                chunk_label = f"CONTENT CHUNK {idx}/{len(chunks)}"
                chunk_events = _parse_block(
                    block=chunk,
                    chunk_label=chunk_label,
                    extraction_method="llm_parse_chunked",
                )
                if chunk_events:
                    collected.extend(chunk_events)
            except Exception as exc:
                print(f"    Chunk {idx} parse error: {exc}")

        return _dedupe_llm_events(collected)

    except Exception as e:
        print(f"    LLM parsing error: {e}")
        return []


_NYC_METRO_STATES = frozenset({"NY", "NJ", "CT"})


def _normalize_tm_events(events: list[dict], venue_name: str) -> list[dict]:
    """Convert Ticketmaster events to our standard format."""
    normalized = []
    skipped_non_nyc = 0
    for e in events:
        # Drop events outside the NYC metro area using TM's structured venue_state field
        venue_state = str(e.get("venue_state", "") or "").strip().upper()
        if venue_state and venue_state not in _NYC_METRO_STATES:
            skipped_non_nyc += 1
            continue

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
            "description": _strip_tags(e.get("info", "") or ""),  # TM uses 'info' field
            "artists": e.get("artists", []),
            "price_min": e.get("price_min"),
            "price_max": e.get("price_max"),
            "event_source_url": e.get("url", ""),
            "extraction_method": "ticketmaster_api",
            "validation_confidence": 0.95,
        })

    if skipped_non_nyc:
        print(f"  TM geo filter: dropped {skipped_non_nyc} non-NYC-metro events")
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

    # Feed-based fetching (iCal/RSS) — preferred over HTML scraping
    feed_url = venue.get("feed_url", "")
    feed_type = venue.get("feed_type", "")
    if feed_url and feed_type in ("ical", "rss"):
        feed_events = _fetch_from_feed(feed_url, feed_type, venue_name)
        if feed_events:
            feed_source = f"feed_{feed_type}"
            result.events = feed_events
            result.source_used = feed_source
            increment("event_fetcher.fetch_venue.success")
            log_event(
                "event_fetch_success",
                venue_name=venue_name,
                city=city,
                strategy=feed_source,
                source_used=feed_source,
                event_count=len(feed_events),
            )
            if not skip_metadata_update:
                mark_venue_fetched(venue_name, city, len(feed_events), feed_source)
            print(f"    Result: {len(feed_events)} events (source: {feed_source})")
            return result
        print(f"    Feed returned no events, falling back to normal strategy")

    # Determine strategy
    strategy = determine_fetch_strategy(
        venue_name, category, website, preferred_source, ticketmaster_venue_id
    )
    print(f"    Strategy: {strategy}")

    if strategy == "skip":
        result.skipped = True
        result.skip_reason = "no_fetchable_source"
        increment("event_fetcher.fetch_venue.skip_no_source")
        return result

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

            # Use shared-feed cache or website scraping if no known API or API failed
            if not scrape_events:
                used_nyc_parks_open_data = False
                shared_feed = _is_shared_multi_location_feed(events_url)
                if shared_feed and events_url:
                    if _is_nyc_parks_shared_feed(events_url):
                        used_nyc_parks_open_data = True
                        def _fetch_nyc_parks_shared_once() -> list[dict]:
                            fetched = _fetch_nyc_parks_open_data_events()
                            return _normalize_event_batch(fetched, "", "nyc_parks_open_data") if fetched else []

                        scrape_events, from_cache = _get_or_fetch_shared_feed_events(
                            events_url,
                            _fetch_nyc_parks_shared_once,
                            venue_name=venue_name,
                        )
                        if scrape_events:
                            website_source = (
                                "nyc_parks_open_data_cached" if from_cache else "nyc_parks_open_data"
                            )
                            print(
                                "    NYC Parks open-data feed "
                                f"({'cache' if from_cache else 'fresh'}): {len(scrape_events)} events"
                            )
                            increment("event_fetcher.source.nyc_parks_open_data.success")
                    else:
                        def _fetch_shared_once() -> list[dict]:
                            fetched = _fetch_from_website(
                                website,
                                venue_name,
                                events_url,
                                venue_context=venue,
                                default_event_venue_name="",
                            )
                            return _normalize_event_batch(fetched, "", "scrape") if fetched else []

                        scrape_events, from_cache = _get_or_fetch_shared_feed_events(
                            events_url,
                            _fetch_shared_once,
                            venue_name=venue_name,
                        )
                        if scrape_events:
                            website_source = "scrape_shared_cached" if from_cache else "scrape_shared"
                            print(
                                f"    Shared feed ({'cache' if from_cache else 'fresh'}): {len(scrape_events)} events"
                            )
                            increment("event_fetcher.source.scrape.success")
                else:
                    scrape_events = _fetch_from_website(
                        website,
                        venue_name,
                        events_url,
                        venue_context=venue,
                    )
                if scrape_events:
                    if website_source not in (
                        "scrape_shared",
                        "scrape_shared_cached",
                        "nyc_parks_open_data",
                        "nyc_parks_open_data_cached",
                    ):
                        scrape_events = _normalize_event_batch(scrape_events, venue_name, "scrape")
                    print(f"    Scrape: {len(scrape_events)} events")
                    if not website_source:
                        website_source = "scrape"
                    if website_source == "scrape":
                        increment("event_fetcher.source.scrape.success")
                else:
                    if used_nyc_parks_open_data:
                        result.source_errors["nyc_parks_open_data"] = "open_data_feed_returned_no_events_or_failed"
                        increment("event_fetcher.source.nyc_parks_open_data.empty")
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

    # Reset per-run shared-feed cache so each run fetches fresh once, then reuses.
    with _SHARED_FEED_LOCK:
        _SHARED_FEED_CACHE.clear()
        _SHARED_FEED_INFLIGHT.clear()

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

        # Update venue metadata FIRST (so timestamps are saved even if merge fails)
        print("Updating venue metadata...")
        _batch_update_venue_metadata(results, city)
        print(f"Updated metadata for {len(results)} venues")

        # Final merge of any remaining events (force write, ignore rate limit)
        if save_to_sheet:
            from .local_event_cache import force_merge_to_sheets, get_cache_stats
            stats = get_cache_stats()
            print(f"\nLocal cache: {stats['event_count']} events from {stats['venues_fetched']} venues")
            try:
                force_merge_to_sheets()
            except Exception as e:
                print(f"WARNING: Event merge failed but venue metadata was saved: {e}")
                print("Events are preserved in local cache for retry")

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
