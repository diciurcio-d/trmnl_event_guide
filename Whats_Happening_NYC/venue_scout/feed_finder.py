"""
Feed discovery and parsing for venue event feeds.

Uses curl_cffi to bypass Cloudflare's basic TLS fingerprinting, then:
  1. Scans fetched HTML for <link rel="alternate"> feed declarations
  2. Scans for embedded Google Calendar iframes
  3. Fingerprints CMS and probes one known feed path as fallback

Parses discovered iCal and RSS/Atom feeds into a normalized event dict:
  {title, date_str, end_date, time, description, event_url}

Usage:
    python -m venue_scout.feed_finder <url> [url2 ...]
"""

import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from urllib.parse import urljoin, urlparse

from curl_cffi import requests as cf_requests


# ---------------------------------------------------------------------------
# HTTP helpers (all requests go through here so the delay is enforced)
# ---------------------------------------------------------------------------

_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_IMPERSONATE = "chrome124"
_TIMEOUT = 15
_DELAY = 0.1  # seconds between any outbound request — be a good citizen


def _get(url: str, extra_headers: dict | None = None) -> tuple[str, str, int]:
    """GET with curl_cffi impersonating Chrome.

    Returns (body, final_url, status_code).
    """
    time.sleep(_DELAY)
    headers = {**_BROWSER_HEADERS, **(extra_headers or {})}
    resp = cf_requests.get(
        url,
        impersonate=_IMPERSONATE,
        headers=headers,
        timeout=_TIMEOUT,
        allow_redirects=True,
    )
    return resp.text, str(resp.url), resp.status_code


def _head(url: str) -> tuple[int, str]:
    """HEAD a URL and return (status_code, content_type)."""
    time.sleep(_DELAY)
    try:
        resp = cf_requests.head(
            url,
            impersonate=_IMPERSONATE,
            headers=_BROWSER_HEADERS,
            timeout=8,
            allow_redirects=True,
        )
        ct = resp.headers.get("content-type", "").lower()
        return resp.status_code, ct
    except Exception:
        return 0, ""


# ---------------------------------------------------------------------------
# HTML scanners
# ---------------------------------------------------------------------------

def _is_cloudflare_block(html: str, status: int) -> bool:
    if status == 403:
        lower = html[:3000].lower()
        return "cloudflare" in lower or "cf-ray" in lower
    lower = html[:3000].lower()
    return "just a moment" in lower and "cloudflare" in lower


def _parse_link_tags(html: str) -> list[dict]:
    """Extract all <link> tags from HTML as attribute dicts."""
    tags = []
    for tag_match in re.finditer(r"<link\s([^>]+?)\/?>", html, re.IGNORECASE | re.DOTALL):
        raw = tag_match.group(1)
        attrs: dict[str, str] = {}
        for attr in re.finditer(r'([\w-]+)\s*=\s*["\']([^"\']*)["\']', raw):
            attrs[attr.group(1).lower()] = attr.group(2)
        tags.append(attrs)
    return tags


def _scan_link_tags(html: str, base_url: str) -> dict[str, str]:
    """Return feed URLs declared in <link rel="alternate"> tags."""
    feeds: dict[str, str] = {}
    for tag in _parse_link_tags(html):
        if tag.get("rel", "").lower() != "alternate":
            continue
        mime = tag.get("type", "").lower()
        href = tag.get("href", "").strip()
        if not href:
            continue
        url = urljoin(base_url, href)
        if "rss" in mime or "atom" in mime:
            feeds.setdefault("rss", url)
        elif "calendar" in mime or mime == "text/calendar":
            feeds.setdefault("ical", url)
    return feeds


def _scan_google_calendar(html: str) -> str | None:
    """Extract a Google Calendar embed URL from an iframe src."""
    match = re.search(
        r'calendar\.google\.com/calendar/(?:embed|r)\?[^"\']*src=([^&"\'>\s]+)',
        html,
        re.IGNORECASE,
    )
    if match:
        cal_id = match.group(1)
        return f"https://calendar.google.com/calendar/embed?src={cal_id}"
    return None


def _scan_ics_links(html: str, base_url: str) -> str | None:
    """Find any href ending in .ics."""
    match = re.search(r'href=["\']([^"\']+\.ics)["\']', html, re.IGNORECASE)
    if match:
        return urljoin(base_url, match.group(1))
    return None


# ---------------------------------------------------------------------------
# CMS fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint_cms(html: str) -> str:
    if "tribe-bar-date" in html or "tribe_events" in html or "tribe_filterbar" in html:
        return "wordpress+tribe"
    if "squarespace.com" in html:
        return "squarespace"
    if "wix.com" in html:
        return "wix"
    if "webflow.com" in html:
        return "webflow"
    if "wp-content" in html or "wp-json" in html:
        return "wordpress"
    return "unknown"


def _cms_feed_candidates(cms: str, root: str) -> list[tuple[str, str]]:
    """Return (feed_url, feed_type) candidates for a given CMS."""
    if cms == "wordpress+tribe":
        return [
            (f"{root}/events/?ical=1", "ical"),
            (f"{root}/events/feed/", "rss"),
        ]
    if cms == "wordpress":
        return [(f"{root}/feed/", "rss")]
    if cms == "squarespace":
        return [
            (f"{root}/events?format=ical", "ical"),
        ]
    return []


def _is_feed_content_type(ct: str, feed_type: str) -> bool:
    if feed_type == "ical":
        return "calendar" in ct or "ics" in ct
    if feed_type == "rss":
        return "rss" in ct or "xml" in ct or "atom" in ct
    return False


# ---------------------------------------------------------------------------
# Feed discovery
# ---------------------------------------------------------------------------

def find_feeds(url: str) -> dict:
    """
    Discover event feeds for a venue URL.

    Returns:
        {
          url, final_url, status, cloudflare_blocked, cms,
          feeds: {rss, ical, google_calendar},
          error,
        }
    """
    result: dict = {
        "url": url,
        "final_url": url,
        "status": 0,
        "cloudflare_blocked": False,
        "cms": "unknown",
        "feeds": {},
        "error": None,
    }

    try:
        html, final_url, status = _get(url)
    except Exception as e:
        result["error"] = str(e)
        return result

    result["final_url"] = final_url
    result["status"] = status

    if _is_cloudflare_block(html, status):
        result["cloudflare_blocked"] = True
        result["error"] = f"Cloudflare block (HTTP {status})"
        return result

    if status not in (200, 301, 302, 303, 307, 308):
        result["error"] = f"HTTP {status}"
        return result

    root = f"{urlparse(final_url).scheme}://{urlparse(final_url).netloc}"

    # Step 1: <link rel="alternate"> tags
    result["feeds"].update(_scan_link_tags(html, final_url))

    # Step 2: Google Calendar embed iframe
    gcal = _scan_google_calendar(html)
    if gcal:
        result["feeds"]["google_calendar"] = gcal

    # Step 3: bare .ics href
    if "ical" not in result["feeds"]:
        ics = _scan_ics_links(html, final_url)
        if ics:
            result["feeds"]["ical"] = ics

    # Step 4: CMS fingerprint + HEAD probe
    # Always probe CMS-specific feeds — generic /feed/ isn't events-specific.
    cms = _fingerprint_cms(html)
    result["cms"] = cms

    for candidate_url, feed_type in _cms_feed_candidates(cms, root):
        if feed_type in result["feeds"]:
            continue
        status_code, ct = _head(candidate_url)
        if status_code == 200 and _is_feed_content_type(ct, feed_type):
            result["feeds"][feed_type] = candidate_url

    return result


# ---------------------------------------------------------------------------
# iCal parser
# ---------------------------------------------------------------------------

def _ical_unfold(text: str) -> str:
    """Unfold iCal line continuations (RFC 5545 §3.1)."""
    return re.sub(r"\r?\n[ \t]", "", text)


def _ical_unescape(text: str) -> str:
    """Unescape iCal text values."""
    return (
        text.replace("\\n", "\n")
            .replace("\\N", "\n")
            .replace("\\,", ",")
            .replace("\\;", ";")
            .replace("\\\\", "\\")
    )


def _parse_ical_dt(raw: str) -> tuple[date | None, str]:
    """Parse an iCal DTSTART/DTEND value.

    Returns (date, time_str) where time_str is "HH:MM" or "".
    """
    # Strip TZID= or VALUE= prefix: e.g. TZID=America/New_York:20260226T190000
    value = raw.split(":")[-1].strip()
    try:
        if "T" in value:
            dt = datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
            return dt.date(), dt.strftime("%H:%M")
        else:
            return datetime.strptime(value[:8], "%Y%m%d").date(), ""
    except ValueError:
        return None, ""


def parse_ical(text: str) -> list[dict]:
    """Parse an iCal/ICS feed and return normalized event dicts.

    Each dict has: title, date_str, end_date, time, description, event_url
    """
    text = _ical_unfold(text)
    events: list[dict] = []

    for block in re.finditer(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.DOTALL):
        raw = block.group(1)

        def get(field: str) -> str:
            m = re.search(rf"^{field}(?:;[^:\r\n]+)?:(.*)", raw, re.MULTILINE)
            return _ical_unescape(m.group(1).strip()) if m else ""

        title = get("SUMMARY")
        description = get("DESCRIPTION")
        event_url = get("URL")
        dtstart_raw = get("DTSTART")
        dtend_raw = get("DTEND")

        if not title or not dtstart_raw:
            continue

        start_date, start_time = _parse_ical_dt(dtstart_raw)
        if not start_date:
            continue

        end_date, _ = _parse_ical_dt(dtend_raw) if dtend_raw else (None, "")

        # For all-day multi-day events iCal DTEND is exclusive — subtract one day
        if end_date and not start_time and end_date > start_date:
            from datetime import timedelta
            end_date = end_date - timedelta(days=1)

        # Don't emit end_date if it's the same as start_date
        end_date_str = end_date.strftime("%Y-%m-%d") if end_date and end_date != start_date else ""

        events.append({
            "title": title,
            "date_str": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date_str,
            "time": start_time,
            "description": _strip_html(description),
            "event_url": event_url,
        })

    return events


# ---------------------------------------------------------------------------
# RSS / Atom parser
# ---------------------------------------------------------------------------

# Atom namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"
# Common event extension namespaces (The Events Calendar, etc.)
_TRIBE_NS = "http://theeventscalendar.com/"


def _parse_rss_date(raw: str) -> tuple[date | None, str]:
    """Parse an RSS pubDate or Atom updated/published string."""
    if not raw:
        return None, ""
    raw = raw.strip()
    # Try RFC 2822 (RSS): "Thu, 26 Feb 2026 19:00:00 +0000"
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",   # Atom ISO 8601
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.date(), dt.strftime("%H:%M")
        except ValueError:
            continue
    return None, ""


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    import html as html_lib
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_lib.unescape(no_tags)).strip()


def parse_rss(text: str) -> list[dict]:
    """Parse an RSS 2.0 or Atom feed and return normalized event dicts.

    Each dict has: title, date_str, end_date, time, description, event_url
    """
    events: list[dict] = []

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return events

    # Detect Atom vs RSS
    tag = root.tag.lower()
    is_atom = "atom" in tag or root.tag == f"{{{_ATOM_NS}}}feed"

    if is_atom:
        ns = {"a": _ATOM_NS}
        items = root.findall("a:entry", ns) or root.findall("entry")
    else:
        # RSS 2.0: root is <rss>, items are under channel/item
        channel = root.find("channel") or root
        items = channel.findall("item")

    for item in items:
        def text_of(tag_name: str, ns_uri: str = "") -> str:
            el = item.find(f"{{{ns_uri}}}{tag_name}" if ns_uri else tag_name)
            return (el.text or "").strip() if el is not None else ""

        if is_atom:
            title = text_of("title", _ATOM_NS) or text_of("title")
            date_raw = text_of("published", _ATOM_NS) or text_of("updated", _ATOM_NS)
            description = _strip_html(text_of("summary", _ATOM_NS) or text_of("content", _ATOM_NS))
            link_el = item.find(f"{{{_ATOM_NS}}}link") or item.find("link")
            event_url = (link_el.get("href", "") if link_el is not None else "") or text_of("id", _ATOM_NS)
        else:
            title = text_of("title")
            date_raw = text_of("pubDate") or text_of("date")
            description = _strip_html(text_of("description"))
            event_url = text_of("link") or text_of("guid")

        if not title:
            continue

        start_date, start_time = _parse_rss_date(date_raw)
        if not start_date:
            continue

        events.append({
            "title": title,
            "date_str": start_date.strftime("%Y-%m-%d"),
            "end_date": "",  # RSS rarely carries end dates
            "time": start_time,
            "description": description[:500],
            "event_url": event_url,
        })

    return events


# ---------------------------------------------------------------------------
# Unified fetch + parse
# ---------------------------------------------------------------------------

def fetch_and_parse_feed(feed_url: str, feed_type: str) -> list[dict]:
    """Fetch a feed URL and return parsed event dicts.

    Args:
        feed_url:  The iCal or RSS URL.
        feed_type: "ical" or "rss".
    """
    accept = "text/calendar, application/ics, */*" if feed_type == "ical" else "application/rss+xml, application/atom+xml, */*"
    try:
        body, _, status = _get(feed_url, extra_headers={"Accept": accept})
    except Exception as e:
        print(f"  Error fetching {feed_url}: {e}")
        return []

    if status != 200:
        print(f"  Feed returned HTTP {status}: {feed_url}")
        return []

    if feed_type == "ical":
        return parse_ical(body)
    else:
        return parse_rss(body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(r: dict, show_events: bool = True) -> None:
    print(f"\n{'─' * 60}")
    print(f"URL:    {r['url']}")
    if r["final_url"] != r["url"]:
        print(f"Resolved: {r['final_url']}")
    print(f"Status: {r['status']}  |  CMS: {r['cms']}")

    if r["cloudflare_blocked"]:
        print("⚠️  Cloudflare blocked — curl_cffi not enough for this site")
        return
    if r["error"]:
        print(f"Error:  {r['error']}")
        return

    if not r["feeds"]:
        print("No feeds found")
        return

    for feed_type, feed_url in r["feeds"].items():
        print(f"  [{feed_type}] {feed_url}")

    if not show_events:
        return

    today = date.today()
    for feed_type in ("ical", "rss"):
        feed_url = r["feeds"].get(feed_type)
        if not feed_url:
            continue
        events = fetch_and_parse_feed(feed_url, feed_type)
        upcoming = [e for e in events if e["date_str"] >= today.strftime("%Y-%m-%d")]
        upcoming.sort(key=lambda e: (e["date_str"], e["time"]))
        print(f"\n  Upcoming events from {feed_type.upper()} feed ({len(upcoming)} found):")
        for e in upcoming[:15]:
            t = f" {e['time']}" if e["time"] else ""
            end = f" → {e['end_date']}" if e["end_date"] else ""
            print(f"    {e['date_str']}{end}{t}  {e['title']}")
        break  # one feed is enough for CLI preview


if __name__ == "__main__":
    urls = sys.argv[1:]
    if not urls:
        print("Usage: python -m venue_scout.feed_finder <url> [url2 ...]")
        sys.exit(1)

    for url in urls:
        result = find_feeds(url)
        _print_result(result, show_events=True)
