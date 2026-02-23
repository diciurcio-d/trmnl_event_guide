"""API endpoint detection for venue event pages.

Uses Playwright to intercept XHR/Fetch requests and identify
API endpoints that return event data.
"""

import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()

# URL patterns that likely contain event data
EVENT_API_PATTERNS = [
    r'/api/events',
    r'/api/v\d+/events',
    r'/events\.json',
    r'/calendar\.json',
    r'/schedule\.json',
    r'/shows\.json',
    r'/performances',
    r'/listings',
    r'/upcoming',
    r'/whatson',
    r'/get-?events',
    r'/event-?list',
    r'/calendar/events',
    r'/feed/events',
    r'\.ics$',  # iCal feeds
    r'/rss',
    r'/feed',
]

# Keywords in JSON responses that suggest event data
EVENT_KEYWORDS = [
    'event', 'events', 'show', 'shows', 'performance', 'performances',
    'concert', 'concerts', 'date', 'datetime', 'startDate', 'start_date',
    'title', 'name', 'venue', 'artist', 'acts', 'lineup',
    'ticketUrl', 'ticket_url', 'buyTickets',
]

# Domains to skip (third-party trackers, analytics, etc.)
SKIP_DOMAINS = frozenset([
    'google-analytics.com', 'googletagmanager.com', 'googleapis.com',
    'facebook.com', 'facebook.net', 'fbcdn.net',
    'twitter.com', 'twimg.com',
    'doubleclick.net', 'googlesyndication.com',
    'cloudflare.com', 'jsdelivr.net', 'unpkg.com',
    'sentry.io', 'bugsnag.com', 'newrelic.com',
    'hotjar.com', 'fullstory.com', 'segment.com',
    'intercom.io', 'zendesk.com', 'drift.com',
    'stripe.com', 'braintree.com',
    'fonts.googleapis.com', 'fonts.gstatic.com',
])


def _should_skip_url(url: str) -> bool:
    """Check if URL should be skipped (trackers, analytics, etc.)."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Skip known tracker domains
        for skip_domain in SKIP_DOMAINS:
            if skip_domain in domain:
                return True

        # Skip static assets
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.gif', '.svg', '.woff', '.woff2', '.ttf']):
            return True

        return False
    except Exception:
        return True


def _matches_event_pattern(url: str) -> bool:
    """Check if URL matches common event API patterns."""
    url_lower = url.lower()
    for pattern in EVENT_API_PATTERNS:
        if re.search(pattern, url_lower):
            return True
    return False


def _looks_like_event_data(data: Any) -> tuple[bool, int]:
    """
    Check if JSON data looks like event data.

    Returns:
        Tuple of (is_event_data, confidence_score)
        confidence_score is 0-100
    """
    if not data:
        return False, 0

    # Convert to string for keyword search
    data_str = json.dumps(data).lower() if not isinstance(data, str) else data.lower()

    # Count event-related keywords
    keyword_count = sum(1 for kw in EVENT_KEYWORDS if kw.lower() in data_str)

    # Check for array of objects (typical event list structure)
    is_array = isinstance(data, list)
    has_objects = is_array and len(data) > 0 and isinstance(data[0], dict)

    # Check for date-like values
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'T\d{2}:\d{2}',       # ISO time
    ]
    has_dates = any(re.search(p, data_str) for p in date_patterns)

    # Calculate confidence score
    score = 0
    if keyword_count >= 3:
        score += 30
    elif keyword_count >= 1:
        score += 15

    if has_objects:
        score += 25
    elif is_array:
        score += 10

    if has_dates:
        score += 25

    # Check for typical event object structure
    if has_objects:
        first_obj = data[0]
        if any(k in first_obj for k in ['name', 'title', 'event']):
            score += 10
        if any(k in first_obj for k in ['date', 'datetime', 'start', 'startDate']):
            score += 10

    return score >= 40, score


def _extract_events_from_api_response(
    data: Any,
    venue_name: str,
    source_url: str,
) -> list[dict]:
    """
    Extract events from API response data.

    Handles various common API response formats including:
    - Direct array of events
    - Nested under 'events', 'data', 'results', etc.
    - Airtable-style 'records[].fields' structure
    """
    events = []

    # Handle different response structures
    event_list = None

    if isinstance(data, list):
        event_list = data
    elif isinstance(data, dict):
        # Handle Airtable-style 'records' structure
        if 'records' in data and isinstance(data['records'], list):
            # Flatten Airtable records: extract 'fields' from each record
            event_list = []
            for record in data['records']:
                if isinstance(record, dict) and 'fields' in record:
                    fields = record['fields']
                    # Add record ID if present
                    if 'id' in record:
                        fields['_record_id'] = record['id']
                    event_list.append(fields)

        # Try other common nested keys
        if not event_list:
            for key in ['events', 'data', 'results', 'items', 'shows', 'performances', 'calendar']:
                if key in data and isinstance(data[key], list):
                    event_list = data[key]
                    break

        # Maybe the dict itself is a single event
        if not event_list and any(k in data for k in ['name', 'title', 'date', 'startDate']):
            event_list = [data]

    if not event_list:
        return []

    # Helper to get field with multiple possible names (case-insensitive)
    def get_field(item: dict, *field_names) -> Any:
        """Get field value trying multiple possible names."""
        # First try exact matches
        for name in field_names:
            if name in item:
                return item[name]

        # Then try case-insensitive and space-tolerant matches
        item_lower = {k.lower().replace(' ', '_').replace('-', '_'): v for k, v in item.items()}
        for name in field_names:
            key = name.lower().replace(' ', '_').replace('-', '_')
            if key in item_lower:
                return item_lower[key]

        return None

    # Parse each event
    for item in event_list:
        if not isinstance(item, dict):
            continue

        # Extract name (with various field name conventions)
        name = get_field(item,
            'name', 'title', 'eventName', 'event_name', 'show',
            'Event name', 'Show name', 'Event title', 'Show title',
            'Name', 'Title', 'event', 'Event',
            'Event', 'Show', 'Performance'  # Airtable-style
        ) or ''

        if not name:
            continue

        # Extract date
        date_str = ''
        dt = None

        raw_date = get_field(item,
            'date', 'startDate', 'start_date', 'datetime', 'dateTime',
            'eventDate', 'event_date', 'start', 'begins',
            'Date', 'Start date', 'Event date', 'Show date',
            'Start Date', 'Event Date', 'Show Date',
            'datestring', 'dateString', 'date_string',  # Airtable-style
            'Event start date and time', 'google_start_time'
        )

        if raw_date:
            if isinstance(raw_date, str):
                # Try to parse various formats
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
                            '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y']:
                    try:
                        dt = datetime.strptime(raw_date[:19], fmt)
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                        date_str = dt.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue

                # If couldn't parse, try to extract date portion
                if not date_str:
                    match = re.search(r'(\d{4}-\d{2}-\d{2})', raw_date)
                    if match:
                        date_str = match.group(1)

        # Extract time
        time_str = get_field(item,
            'time', 'startTime', 'start_time', 'Time', 'Start time', 'Door time'
        ) or ''

        # Extract URL
        url = get_field(item,
            'url', 'link', 'ticketUrl', 'ticket_url', 'eventUrl',
            'URL', 'Link', 'Ticket URL', 'Tickets', 'Event URL',
            'Ticket URL', 'Stellar URL', 'Fever URL'  # Airtable-style
        ) or ''

        # Extract event type
        event_type = get_field(item,
            'type', 'eventType', 'event_type', 'category',
            'Type', 'Category', 'Event type', 'Genre'
        ) or 'event'

        # Extract artists if available
        artists = []
        artist_field = get_field(item, 'artist', 'artists', 'Artist', 'Artists', 'Performer', 'Performers')
        if artist_field:
            if isinstance(artist_field, str):
                artists = [artist_field]
            elif isinstance(artist_field, list):
                artists = [x.get('name', x) if isinstance(x, dict) else str(x) for x in artist_field]

        events.append({
            'name': str(name),
            'datetime': dt,
            'date_str': date_str,
            'venue_name': venue_name,
            'event_type': str(event_type),
            'url': str(url) if url else source_url,
            'source': 'api',
            'matched_artist': '',
            'travel_minutes': None,
            'artists': artists,
        })

    return events


def detect_api_endpoints(
    url: str,
    wait_time: float = None,
) -> list[dict]:
    """
    Load a URL and intercept XHR/Fetch requests to find API endpoints.

    Args:
        url: The venue events page URL to analyze
        wait_time: Seconds to wait for API calls (default from settings)

    Returns:
        List of detected API endpoints with their responses:
        [{'url': str, 'data': Any, 'confidence': int}, ...]
    """
    if wait_time is None:
        wait_time = getattr(_settings, 'VENUE_API_DETECTION_WAIT', 5)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright not installed. Run: pip install playwright && playwright install")
        return []

    detected_endpoints = []

    def handle_response(response):
        """Callback for intercepted responses."""
        try:
            req_url = response.url

            # Skip irrelevant URLs
            if _should_skip_url(req_url):
                return

            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type and 'text/json' not in content_type:
                # Still check if URL matches event patterns
                if not _matches_event_pattern(req_url):
                    return

            # Try to parse JSON response
            try:
                data = response.json()
            except Exception:
                return

            # Check if it looks like event data
            is_event_data, confidence = _looks_like_event_data(data)

            if is_event_data or _matches_event_pattern(req_url):
                # Boost confidence if URL matches pattern
                if _matches_event_pattern(req_url):
                    confidence = min(100, confidence + 20)

                detected_endpoints.append({
                    'url': req_url,
                    'data': data,
                    'confidence': confidence,
                })

        except Exception as e:
            pass  # Silently skip failed responses

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )
            page = context.new_page()

            # Listen for responses
            page.on('response', handle_response)

            # Navigate to page
            print(f"    Loading page for API detection: {url[:60]}...")
            goto_timeout = int(_settings.VENUE_API_DETECTOR_GOTO_TIMEOUT_MS)
            page.goto(url, wait_until='networkidle', timeout=goto_timeout)

            # Wait a bit more for lazy-loaded content
            page.wait_for_timeout(int(wait_time * 1000))

            # Try scrolling to trigger more API calls
            page.evaluate('window.scrollTo(0, document.body.scrollHeight / 2)')
            scroll_wait_ms = int(_settings.VENUE_API_DETECTOR_SCROLL_WAIT_MS)
            page.wait_for_timeout(scroll_wait_ms)

            browser.close()

    except Exception as e:
        print(f"    API detection error: {e}")
        return []

    # Sort by confidence
    detected_endpoints.sort(key=lambda x: x['confidence'], reverse=True)

    return detected_endpoints


def fetch_from_api(
    api_url: str,
    venue_name: str,
) -> list[dict]:
    """
    Fetch events directly from a known API endpoint.

    Args:
        api_url: The API endpoint URL
        venue_name: Name of the venue

    Returns:
        List of event dicts
    """
    import requests

    try:
        print(f"    Fetching from API: {api_url[:60]}...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
        }

        timeout = int(_settings.VENUE_API_DETECTOR_FETCH_TIMEOUT_SEC)
        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        events = _extract_events_from_api_response(data, venue_name, api_url)

        return events

    except Exception as e:
        print(f"    API fetch error: {e}")
        return []


def detect_and_fetch(
    page_url: str,
    venue_name: str,
    min_confidence: int = 50,
) -> tuple[list[dict], str | None]:
    """
    Detect API endpoints and fetch events in one step.

    Args:
        page_url: The venue events page URL
        venue_name: Name of the venue
        min_confidence: Minimum confidence score to use endpoint

    Returns:
        Tuple of (events list, api_endpoint_url or None)
    """
    # Detect endpoints
    endpoints = detect_api_endpoints(page_url)

    if not endpoints:
        print(f"    No API endpoints detected")
        return [], None

    # Use the highest confidence endpoint
    best = endpoints[0]

    if best['confidence'] < min_confidence:
        print(f"    Best endpoint confidence too low: {best['confidence']}")
        return [], None

    print(f"    Found API endpoint (confidence: {best['confidence']})")

    # Extract events from the already-fetched data
    events = _extract_events_from_api_response(
        best['data'],
        venue_name,
        best['url']
    )

    if events:
        print(f"    Extracted {len(events)} events from API")
        return events, best['url']

    return [], None


if __name__ == "__main__":
    # Test with a known venue
    test_url = "https://www.caveat.nyc/events"
    print(f"Testing API detection on: {test_url}")

    endpoints = detect_api_endpoints(test_url)

    print(f"\nDetected {len(endpoints)} potential API endpoints:")
    for ep in endpoints[:5]:
        print(f"  Confidence {ep['confidence']}: {ep['url'][:80]}")

    if endpoints:
        events, api_url = detect_and_fetch(test_url, "Caveat NYC")
        print(f"\nExtracted {len(events)} events")
        for event in events[:3]:
            print(f"  - {event.get('date_str')}: {event.get('name')}")
