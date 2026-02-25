"""Website validation and discovery for venues.

Finds official venue websites and filters out aggregator sites.
"""

import re
import sys
import time
import os
import json
import importlib.util
from urllib.parse import urljoin, urlparse, quote_plus
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from venue_scout.observability import increment, log_event, record_failure


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()

_DEFAULT_IFRAME_MEDIA_HOST_PATTERNS = (
    "instagram.com",
    "cdninstagram.com",
    "fbcdn.net",
    "facebook.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "tiktok.com",
    "soundcloud.com",
    "spotify.com",
    "wistia.com",
)
_DEFAULT_IFRAME_MEDIA_EXTENSIONS = (".mp4", ".m3u8", ".mp3", ".mov", ".webm")
_IFRAME_MEDIA_HOST_PATTERNS = tuple(
    str(x).lower().strip()
    for x in getattr(_settings, "WEBSITE_VALIDATOR_IFRAME_MEDIA_HOST_PATTERNS", _DEFAULT_IFRAME_MEDIA_HOST_PATTERNS)
    if str(x).strip()
)
_IFRAME_MEDIA_EXTENSIONS = tuple(
    str(x).lower().strip()
    for x in getattr(_settings, "WEBSITE_VALIDATOR_IFRAME_MEDIA_PATH_EXTENSIONS", _DEFAULT_IFRAME_MEDIA_EXTENSIONS)
    if str(x).strip()
)

# Aggregator domains to reject - these are not official venue websites
AGGREGATOR_DOMAINS = frozenset([
    # Booking/listing sites
    "classpass.com",
    "mindbodyonline.com",
    "vagaro.com",
    "wellnessliving.com",
    "booksy.com",
    "schedulicity.com",
    # Review sites
    "yelp.com",
    "tripadvisor.com",
    "foursquare.com",
    "trustpilot.com",
    # Maps
    "google.com/maps",
    "maps.google.com",
    "goo.gl/maps",
    "mapquest.com",
    # Event aggregators
    "eventbrite.com",
    "ticketmaster.com",
    "stubhub.com",
    "seatgeek.com",
    "vividseats.com",
    "axs.com",
    "dice.fm",
    "songkick.com",
    "bandsintown.com",
    "jambase.com",
    "ticketweb.com",
    "ticketfly.com",
    "etix.com",
    "showclix.com",
    # Venue rental sites
    "peerspace.com",
    "tagvenue.com",
    "giggster.com",
    "splacer.co",
    "venuebook.com",
    "thevendry.com",
    "cvent.com",
    "weddingwire.com",
    "theknot.com",
    "zola.com",
    # Social media
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "tiktok.com",
    "youtube.com",
    "linkedin.com",
    "x.com",
    # Generic directories
    "yellowpages.com",
    "whitepages.com",
    "manta.com",
    "bbb.org",
    "chamberofcommerce.com",
    # NYC-specific aggregators
    "nycgo.com",
    "timeout.com",
    "thrillist.com",
    "secretnyc.co",
    "6sqft.com",
    "untappedcities.com",
    "nyctrivialist.com",
    "nycinsiderguide.com",
    "iloveny.com",
    "newyork.com",
    "nyc.com",
    "newyorkcity.com",
    # Museum/venue directories
    "whichmuseum.com",
    "museumary.com",
    "museumsnyc.org",
    "museumhack.com",
    "nycmuseums.org",
    "atlasobscura.com",
    "roadtrippers.com",
    # Entertainment guides
    "broadwayworld.com",
    "playbill.com",
    "theatermania.com",
    "nytix.com",
    "broadwaydirect.com",
    "broadway.com",
    "newyorktheatreguide.com",
    # Ticketing/promotion platforms (not venue sites)
    "livenation.com",
    "aegpresents.com",
    # Food/nightlife aggregators
    "opentable.com",
    "resy.com",
    "eater.com",
    "infatuation.com",
    "zagat.com",
    "grubstreet.com",
    # Fitness directories
    "gympass.com",
    "fitnessblender.com",
    "fitnessinternational.com",
    # Wikipedia (not official)
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    # Other listing sites
    "findagrave.com",
    "legacy.com",
    "citysearch.com",
    "judysbook.com",
    "merchantcircle.com",
    "hotfrog.com",
    "brownpapertickets.com",
])

# Patterns that suggest aggregator URLs
AGGREGATOR_PATTERNS = [
    r"/listings?/",
    r"/venue/",
    r"/place/",
    r"/business/",
    r"/biz/",
    r"/location/",
    r"/explore/locations/",
    r"/studios/",
    r"/rooms/",
]

EVENT_PATH_KEYWORDS = (
    "event",
    "events",
    "calendar",
    "shows",
    "show",
    "exhibition",
    "exhibitions",
    "upcoming",
    "upcoming-exhibitions",
    "shows.html",
    "calendar.html",
    "events.html",
    "happenings",
    "program",
    "programming",
    "public-program",
    "public-programs",
    "public_program",
    "public_programs",
    "whats-on",
    "whats_on",
    "tickets",
)

_SHARED_EVENTS_FEED_PATTERNS = (
    ("nycgovparks.org", "/events"),
    ("nycgovparks.org", "/events/volunteer"),
)

_GENERIC_VENUE_TOKENS = frozenset(
    {
        "the",
        "new",
        "york",
        "nyc",
        "city",
        "park",
        "parks",
        "center",
        "centre",
        "community",
        "house",
        "hall",
        "museum",
        "garden",
        "gardens",
        "playground",
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
        "brooklyn",
        "manhattan",
        "bronx",
        "queens",
        "staten",
        "island",
    }
)


DEAD_SITE_PATTERNS = [
    r"domain is for sale",
    r"this domain may be for sale",
    r"parkingcrew",
    r"sedo",
    r"buy this domain",
    r"site can'?t be reached",
    r"account suspended",
]


def _int_env(name: str, default: int) -> int:
    """Read integer env var with safe fallback."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _is_transient_provider_error(error_text: str) -> bool:
    """True when an error likely came from temporary provider load or network jitter."""
    if not error_text:
        return False
    text = error_text.lower()
    patterns = (
        "503",
        "overloaded",
        "unavailable",
        "timed out",
        "timeout",
        "read timeout",
        "connection reset",
        "temporary failure",
        "too many requests",
        "429",
    )
    return any(p in text for p in patterns)


def _float_env(name: str, default: float) -> float:
    """Read float env var with safe fallback."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _load_jina_api_key() -> str:
    """Load Jina API key from config file (best effort)."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    if not config_path.exists():
        return ""
    try:
        with open(config_path) as f:
            config = json.load(f)
        return str(config.get("jina", {}).get("api_key", "")).strip()
    except Exception:
        return ""


def _fetch_text_via_jina_with_retries(url: str, timeout: int, retries: int, retry_delay: float) -> tuple[str, str]:
    """Fetch URL content via Jina with bounded retries."""
    # Check if Jina is disabled via env var or setting
    skip_jina = os.environ.get("WEBSITE_VALIDATOR_SKIP_JINA", "").lower() in ("1", "true", "yes")
    if not skip_jina:
        skip_jina = bool(getattr(_settings, "WEBSITE_VALIDATOR_SKIP_JINA", False))
    if skip_jina:
        return "", "jina_skipped"

    from utils.jina_reader import fetch_page_text_jina

    attempts = max(1, retries)
    last_error = ""
    for attempt in range(attempts):
        try:
            return fetch_page_text_jina(url, timeout=timeout), ""
        except Exception as e:
            last_error = str(e)
            if attempt < attempts - 1 and retry_delay > 0:
                time.sleep(retry_delay)
    return "", last_error


def _search_event_candidates_jina(
    homepage_url: str,
    venue_name: str,
    timeout: int,
    max_results: int,
) -> list[str]:
    """Use Jina Search to discover same-domain event/calendar URLs."""
    # Check if Jina is disabled
    skip_jina = os.environ.get("WEBSITE_VALIDATOR_SKIP_JINA", "").lower() in ("1", "true", "yes")
    if not skip_jina:
        skip_jina = bool(getattr(_settings, "WEBSITE_VALIDATOR_SKIP_JINA", False))
    if skip_jina:
        return []

    api_key = _load_jina_api_key()
    if not api_key:
        return []

    domain = extract_domain(homepage_url)
    if not domain:
        return []

    query = quote_plus(
        f"site:{domain} {venue_name} events calendar programs \"public programs\" shows exhibitions upcoming happenings"
    )
    search_url = f"https://s.jina.ai/{query}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    candidates: list[str] = []
    seen = set()
    rows = data.get("data", []) if isinstance(data, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pieces = [
            row.get("url", ""),
            row.get("title", ""),
            row.get("description", ""),
            row.get("content", ""),
        ]
        for piece in pieces:
            if not piece:
                continue
            for match in re.findall(r'https?://[^\s<>"\')]+', str(piece)):
                norm = _normalize_url(match)
                if not norm or norm in seen:
                    continue
                if not _same_site(norm, homepage_url):
                    continue
                path_query = (urlparse(norm).path + "?" + urlparse(norm).query).lower()
                if any(token in path_query for token in EVENT_PATH_KEYWORDS):
                    seen.add(norm)
                    candidates.append(norm)
                    if len(candidates) >= max_results:
                        return candidates
    return candidates


def _events_url_hint_score(url: str, homepage_url: str, venue_name: str = "") -> int:
    """Path/query-only confidence used for blocked-content fallback picks."""
    parsed = urlparse(_normalize_url(url))
    path_query = (parsed.path + "?" + parsed.query).lower()
    score = 0
    if _normalize_url(url).rstrip("/") == _normalize_url(homepage_url).rstrip("/"):
        return 0
    for keyword in EVENT_PATH_KEYWORDS:
        if keyword in path_query:
            score += 1
    if any(token in path_query for token in ("/events", "/calendar", "/shows", "/exhibitions", "upcoming")):
        score += 2
    if "iframe" in path_query:
        score -= 1
    if _is_known_shared_events_feed(url):
        score -= 2
        if _url_contains_venue_tokens(url, venue_name):
            score += 2
    return score


def _venue_name_tokens(venue_name: str) -> set[str]:
    """Get venue-identifying tokens and remove generic words."""
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", str(venue_name or "").lower())
        if token not in _GENERIC_VENUE_TOKENS
    }
    return tokens


def _is_known_shared_events_feed(url: str) -> bool:
    """True when URL is known to be a citywide/shared feed endpoint."""
    norm = _normalize_url(url)
    parsed = urlparse(norm)
    domain = parsed.netloc.lower().replace("www.", "")
    path = (parsed.path or "").rstrip("/").lower()
    if not path:
        path = "/"
    for shared_domain, shared_path in _SHARED_EVENTS_FEED_PATTERNS:
        if domain.endswith(shared_domain) and path == shared_path:
            return True
    return False


def _url_contains_venue_tokens(url: str, venue_name: str) -> bool:
    """Heuristic: URL path/query contains venue-specific token(s)."""
    if not venue_name:
        return False
    parsed = urlparse(_normalize_url(url))
    path_query = f"{parsed.path}?{parsed.query}".lower()
    tokens = _venue_name_tokens(venue_name)
    return any(token in path_query for token in tokens)


def verify_official_website(url: str, venue_name: str, city: str = "NYC") -> tuple[bool, str]:
    """
    Use LLM to verify if a URL is the official venue website.

    Args:
        url: The URL to verify
        venue_name: Name of the venue
        city: City for context

    Returns:
        Tuple of (is_official, reason):
        - is_official: True if this is the official venue website
        - reason: Explanation of the decision
    """
    from utils.llm import generate_content

    domain = extract_domain(url)

    prompt = f"""Verify if this URL is the OFFICIAL website for the venue, or if it's a listing/aggregator site.

Venue: {venue_name}
City: {city}
URL: {url}
Domain: {domain}

An OFFICIAL website is:
- Owned and operated by the venue itself
- Has the venue's own domain (often contains venue name)
- Contains the venue's event calendar, about page, contact info

An AGGREGATOR/LISTING site is:
- A directory that lists many venues (like Yelp, TripAdvisor, TimeOut)
- A ticketing platform (like Eventbrite, Ticketmaster)
- A review or recommendation site
- A museum/venue directory
- A tourism or city guide site
- A third-party site with venue info but not run by the venue

Respond with ONLY one line in this format:
OFFICIAL: [reason]
or
AGGREGATOR: [reason]

Be strict - if uncertain, say AGGREGATOR."""

    try:
        increment("website_validator.verify.calls")
        response = generate_content(prompt).strip()

        if response.upper().startswith("OFFICIAL"):
            reason = response.split(":", 1)[1].strip() if ":" in response else "Verified as official"
            log_event("website_verify_official", venue_name=venue_name, domain=domain)
            return True, reason
        else:
            reason = response.split(":", 1)[1].strip() if ":" in response else "Detected as aggregator"
            log_event("website_verify_rejected", venue_name=venue_name, domain=domain, reason=reason)
            return False, reason

    except Exception as e:
        record_failure("website_validator.verify", str(e), venue_name=venue_name, url=url, city=city)
        return False, f"Verification error: {e}"


def _llm_pick_events_page(
    venue_name: str,
    homepage_url: str,
    candidate_snippets: list[tuple[str, str]],
) -> str | None:
    """
    Ask the LLM to pick the best upcoming-events page from fetched candidates.

    Args:
        venue_name: Human-readable venue name (for context).
        homepage_url: The venue's homepage (so the LLM can say HOMEPAGE).
        candidate_snippets: List of (url, content_snippet) pairs already fetched.

    Returns:
        The chosen URL, or None if the LLM cannot identify a good events page.
    """
    from utils.llm import generate_content

    if not candidate_snippets:
        return None

    parts = []
    for i, (url, snippet) in enumerate(candidate_snippets, 1):
        parts.append(f"--- Candidate {i}: {url} ---\n{snippet[:2000].strip()}\n")

    candidates_text = "\n".join(parts)

    prompt = f"""You are helping find the EVENTS / CALENDAR page for a venue.

Venue: {venue_name}
Homepage: {homepage_url}

Below are {len(candidate_snippets)} candidate pages with content snippets.
Pick the page that BEST shows UPCOMING events, shows, performances, or exhibitions.

Prefer pages that:
- List multiple UPCOMING events with future dates
- Have ticket links, booking options, or reservation info
- Use words like "upcoming", "on now", "schedule", or show date/time grids
- Have clean, short paths — strongly prefer /events over /exhibitions, /calendar over /calendar/film
- Have clean paths without query-string filters (e.g. /events beats /events?category=film)

Avoid:
- Past/archived events, history pages, "exhibitions archive"
- Permanent collection pages (no specific dates)
- General about/contact/host pages
- Homepages unless events are clearly listed inline
- URLs with query-string filters that narrow results to a single category
- Sub-category pages when a broader listing page is available (e.g. avoid /events/film if /events exists)

If events are shown on the homepage, return: HOMEPAGE
If no candidate clearly shows upcoming events, return: NONE
Otherwise return ONLY the exact URL of the best events page. No explanation.

{candidates_text}"""

    try:
        increment("website_validator.llm_pick_events_page.calls")
        response = generate_content(prompt).strip()

        if response.upper() == "HOMEPAGE":
            log_event("events_page_llm_picked_homepage", venue_name=venue_name, homepage_url=homepage_url)
            return homepage_url

        if response.upper() == "NONE":
            log_event("events_page_llm_returned_none", venue_name=venue_name, homepage_url=homepage_url)
            return None

        match = re.search(r'https?://[^\s<>"]+', response)
        if match:
            chosen = _normalize_url(match.group())
            log_event("events_page_llm_picked", venue_name=venue_name, homepage_url=homepage_url, events_url=chosen)
            return chosen

        return None

    except Exception as e:
        record_failure(
            "website_validator.find_events_page",
            f"llm_pick_failed:{e}",
            venue_name=venue_name,
            homepage_url=homepage_url,
        )
        return None


def find_events_page(homepage_url: str, venue_name: str) -> tuple[str | None, bool]:
    """
    Find the events/calendar page URL for a venue by reading the homepage.

    Fetches the homepage, probes candidate URLs, then uses an LLM to pick
    the best events/calendar page.

    Args:
        homepage_url: The venue's homepage URL
        venue_name: Name of the venue

    Returns:
        Tuple of (events_url, cloudflare_detected):
        - events_url: The verified events page URL, or None if not found
        - cloudflare_detected: True if any fetch encountered a Cloudflare challenge
    """
    if not homepage_url:
        return None

    try:
        increment("website_validator.find_events_page.calls")
        homepage_url = _normalize_url(homepage_url)
        budget_sec = _int_env(
            "WEBSITE_VALIDATOR_EVENTS_BUDGET_SEC",
            int(_settings.WEBSITE_VALIDATOR_EVENTS_BUDGET_SEC),
        )
        budget_extension_sec = _int_env(
            "WEBSITE_VALIDATOR_EVENTS_BUDGET_EXTENSION_SEC",
            int(_settings.WEBSITE_VALIDATOR_EVENTS_BUDGET_EXTENSION_SEC),
        )
        max_budget_sec = _int_env(
            "WEBSITE_VALIDATOR_MAX_EVENTS_BUDGET_SEC",
            int(_settings.WEBSITE_VALIDATOR_MAX_EVENTS_BUDGET_SEC),
        )
        max_candidates = _int_env(
            "WEBSITE_VALIDATOR_MAX_EVENT_CANDIDATES",
            int(_settings.WEBSITE_VALIDATOR_MAX_EVENT_CANDIDATES),
        )
        max_iframe_candidates = _int_env(
            "WEBSITE_VALIDATOR_MAX_IFRAME_CANDIDATES",
            int(_settings.WEBSITE_VALIDATOR_MAX_IFRAME_CANDIDATES),
        )
        jina_timeout = _int_env(
            "WEBSITE_VALIDATOR_JINA_TIMEOUT_SEC",
            int(_settings.WEBSITE_VALIDATOR_JINA_TIMEOUT_SEC),
        )
        jina_retries = _int_env(
            "WEBSITE_VALIDATOR_JINA_RETRIES",
            int(_settings.WEBSITE_VALIDATOR_JINA_RETRIES),
        )
        jina_retry_delay = _float_env(
            "WEBSITE_VALIDATOR_JINA_RETRY_DELAY_SEC",
            float(_settings.WEBSITE_VALIDATOR_JINA_RETRY_DELAY_SEC),
        )
        jina_search_timeout = _int_env(
            "WEBSITE_VALIDATOR_JINA_SEARCH_TIMEOUT_SEC",
            int(_settings.WEBSITE_VALIDATOR_JINA_SEARCH_TIMEOUT_SEC),
        )
        jina_search_max_results = _int_env(
            "WEBSITE_VALIDATOR_JINA_SEARCH_MAX_RESULTS",
            int(_settings.WEBSITE_VALIDATOR_JINA_SEARCH_MAX_RESULTS),
        )
        started = time.monotonic()
        current_budget_sec = budget_sec
        transient_errors = 0
        blocked_errors = 0
        blocked_candidates: dict[str, int] = {}
        _cf_flag: list[bool] = []  # Appended to by _fetch_html whenever a CF retry fires
        print(f"    Fetching homepage to find events page...")
        homepage_content = ""
        homepage_jina_error = ""

        def _capture_probe_error(reason: str, candidate: str, stage: str) -> bool:
            nonlocal transient_errors, blocked_errors
            lowered = (reason or "").lower()
            if _is_transient_provider_error(reason):
                transient_errors += 1
            is_blocked = any(token in lowered for token in ("403", "forbidden", "access denied", "blocked"))
            if is_blocked:
                blocked_errors += 1
            record_failure(
                "website_validator.find_events_page",
                reason,
                venue_name=venue_name,
                homepage_url=homepage_url,
                candidate_url=candidate,
                stage=stage,
            )
            return is_blocked

        def _has_budget() -> bool:
            nonlocal current_budget_sec
            elapsed = time.monotonic() - started
            if elapsed <= current_budget_sec:
                return True
            if transient_errors > 0 and current_budget_sec < max_budget_sec:
                prev = current_budget_sec
                current_budget_sec = min(max_budget_sec, current_budget_sec + budget_extension_sec)
                log_event(
                    "events_probe_budget_extended",
                    venue_name=venue_name,
                    homepage_url=homepage_url,
                    from_sec=prev,
                    to_sec=current_budget_sec,
                    transient_errors=transient_errors,
                )
                return True
            record_failure(
                "website_validator.find_events_page",
                "events_probe_budget_exceeded",
                venue_name=venue_name,
                homepage_url=homepage_url,
                budget_sec=current_budget_sec,
                transient_errors=transient_errors,
            )
            return False

        def _probe_candidate(candidate: str) -> None:
            """Fetch candidate content and append (url, snippet) to candidate_snippets.
            Updates blocked_candidates when we get a 403 but no content."""
            if not _candidate_preflight_ok(candidate):
                return

            blocked_signal = False
            content, jina_error = _fetch_text_via_jina_with_retries(
                candidate,
                timeout=jina_timeout,
                retries=jina_retries,
                retry_delay=jina_retry_delay,
            )
            if jina_error and jina_error != "jina_skipped":
                blocked_signal = _capture_probe_error(
                    f"jina_candidate_fetch_failed:{jina_error}",
                    candidate,
                    "candidate_jina_fetch",
                ) or blocked_signal

            if not content or len(content) < 100:
                try:
                    html_fallback = _fetch_html(candidate, _cf_flag=_cf_flag)
                    content = _html_to_text(html_fallback)
                except Exception as e:
                    blocked_signal = _capture_probe_error(
                        f"html_candidate_fetch_failed:{e}",
                        candidate,
                        "candidate_html_fetch",
                    ) or blocked_signal

                if not content or len(content) < 100:
                    if blocked_signal:
                        hint = _events_url_hint_score(candidate, homepage_url, venue_name)
                        if hint >= 2:
                            blocked_candidates[candidate] = max(hint, blocked_candidates.get(candidate, 0))
                    return

            candidate_snippets.append((candidate, content[:3000]))

            # Discover and collect iframe event widgets (e.g. embedded Ticketmaster/Eventbrite calendars).
            candidate_html = ""
            try:
                candidate_html = _fetch_html(candidate, _cf_flag=_cf_flag)
            except Exception as e:
                _capture_probe_error(
                    f"html_candidate_fetch_failed:{e}",
                    candidate,
                    "candidate_iframe_html_fetch",
                )

            iframe_urls = _extract_iframe_links_html(candidate_html, candidate)
            for iframe_url in iframe_urls[:max_iframe_candidates]:
                if not _has_budget():
                    return
                if not _candidate_preflight_ok(iframe_url):
                    continue

                iframe_blocked_signal = False
                iframe_content, iframe_error = _fetch_text_via_jina_with_retries(
                    iframe_url,
                    timeout=jina_timeout,
                    retries=jina_retries,
                    retry_delay=jina_retry_delay,
                )
                if iframe_error and iframe_error != "jina_skipped":
                    iframe_blocked_signal = _capture_probe_error(
                        f"jina_iframe_fetch_failed:{iframe_error}",
                        iframe_url,
                        "iframe_jina_fetch",
                    ) or iframe_blocked_signal
                if not iframe_content or len(iframe_content) < 100:
                    try:
                        iframe_html = _fetch_html(iframe_url, _cf_flag=_cf_flag)
                        iframe_content = _html_to_text(iframe_html)
                    except Exception as e:
                        iframe_blocked_signal = _capture_probe_error(
                            f"html_iframe_fetch_failed:{e}",
                            iframe_url,
                            "iframe_html_fetch",
                        ) or iframe_blocked_signal
                if not iframe_content or len(iframe_content) < 100:
                    if iframe_blocked_signal:
                        hint = _events_url_hint_score(iframe_url, homepage_url, venue_name)
                        if hint >= 2:
                            blocked_candidates[iframe_url] = max(hint, blocked_candidates.get(iframe_url, 0))
                    continue

                candidate_snippets.append((iframe_url, iframe_content[:3000]))

        try:
            homepage_content, homepage_jina_error = _fetch_text_via_jina_with_retries(
                homepage_url,
                timeout=jina_timeout,
                retries=jina_retries,
                retry_delay=jina_retry_delay,
            )
        except Exception as e:
            homepage_jina_error = str(e)
        if homepage_jina_error == "jina_skipped":
            homepage_jina_error = ""
        if homepage_jina_error:
            _capture_probe_error(
                f"jina_homepage_fetch_failed:{homepage_jina_error}",
                homepage_url,
                "homepage_jina_fetch",
            )

        homepage_html = ""
        homepage_html_error = ""
        try:
            homepage_html = _fetch_html(homepage_url, _cf_flag=_cf_flag)
        except Exception as e:
            homepage_html_error = str(e)
            _capture_probe_error(
                f"html_homepage_fetch_failed:{homepage_html_error}",
                homepage_url,
                "homepage_html_fetch",
            )

        if not homepage_content and homepage_html:
            homepage_content = _html_to_text(homepage_html)

        homepage_short = not homepage_content or len(homepage_content) < 100
        if homepage_short:
            record_failure(
                "website_validator.find_events_page",
                "homepage_content_too_short",
                venue_name=venue_name,
                homepage_url=homepage_url,
            )

        candidates = _build_events_candidates(homepage_url, homepage_content, homepage_html)
        if not candidates:
            candidates = [homepage_url]
        candidate_set = set(candidates)

        homepage_blocked_or_transient = (
            _is_transient_provider_error(homepage_jina_error)
            or _is_transient_provider_error(homepage_html_error)
            or any(token in (homepage_jina_error + " " + homepage_html_error).lower() for token in ("403", "forbidden"))
        )
        if homepage_short or homepage_blocked_or_transient:
            search_candidates = _search_event_candidates_jina(
                homepage_url=homepage_url,
                venue_name=venue_name,
                timeout=jina_search_timeout,
                max_results=jina_search_max_results,
            )
            if search_candidates:
                log_event(
                    "events_page_search_candidates_added",
                    venue_name=venue_name,
                    homepage_url=homepage_url,
                    count=len(search_candidates),
                )
            ordered = []
            for url in search_candidates + candidates:
                norm = _normalize_url(url)
                if norm and norm not in ordered:
                    ordered.append(norm)
            candidates = ordered
            candidate_set = set(candidates)

        # (url, content_snippet) pairs collected while probing — fed to LLM for final decision.
        candidate_snippets: list[tuple[str, str]] = []
        timed_out = False

        for candidate in candidates[:max_candidates]:
            if not _has_budget():
                timed_out = True
                break
            print(f"    Verifying events page: {candidate}")
            _probe_candidate(candidate)

        should_try_search_fallback = (
            not candidate_snippets
            and not timed_out
            and (blocked_errors > 0 or transient_errors > 0)
        )
        if should_try_search_fallback:
            extra_candidates = _search_event_candidates_jina(
                homepage_url=homepage_url,
                venue_name=venue_name,
                timeout=jina_search_timeout,
                max_results=jina_search_max_results,
            )
            new_candidates = []
            for url in extra_candidates:
                norm = _normalize_url(url)
                if not norm or norm in candidate_set:
                    continue
                candidate_set.add(norm)
                new_candidates.append(norm)
            if new_candidates:
                log_event(
                    "events_page_search_fallback_probe",
                    venue_name=venue_name,
                    homepage_url=homepage_url,
                    count=len(new_candidates),
                )
                for candidate in new_candidates[:max_candidates]:
                    if not _has_budget():
                        timed_out = True
                        break
                    print(f"    Verifying events page: {candidate}")
                    _probe_candidate(candidate)

        # LLM picks the best events page from all collected snippets.
        if candidate_snippets:
            chosen_url = _llm_pick_events_page(venue_name, homepage_url, candidate_snippets)
            if chosen_url:
                # If the LLM picks the homepage but a plausible blocked event URL exists,
                # prefer the blocked URL so downstream fetchers target the listing path.
                is_homepage_best = _normalize_url(chosen_url).rstrip("/") == homepage_url.rstrip("/")
                if is_homepage_best and blocked_candidates:
                    fallback_url, fallback_hint = max(blocked_candidates.items(), key=lambda item: item[1])
                    if fallback_hint >= 3:
                        log_event(
                            "events_page_blocked_preferred_over_homepage",
                            venue_name=venue_name,
                            homepage_url=homepage_url,
                            events_url=fallback_url,
                            hint_score=fallback_hint,
                        )
                        return fallback_url, bool(_cf_flag)
                print(f"    ✓ Events page confirmed: {chosen_url}")
                log_event("events_page_verified", venue_name=venue_name, events_url=chosen_url)
                return chosen_url, bool(_cf_flag)

        if blocked_candidates:
            fallback_url, fallback_hint = max(blocked_candidates.items(), key=lambda item: item[1])
            log_event(
                "events_page_blocked_fallback",
                venue_name=venue_name,
                homepage_url=homepage_url,
                events_url=fallback_url,
                hint_score=fallback_hint,
                blocked_errors=blocked_errors,
            )
            return fallback_url, bool(_cf_flag)

        record_failure(
            "website_validator.find_events_page",
            "reachable_no_events_page_budget_limited" if timed_out else "reachable_no_events_page",
            venue_name=venue_name,
            homepage_url=homepage_url,
            blocked_errors=blocked_errors,
            transient_errors=transient_errors,
            budget_sec=current_budget_sec,
        )
        return None, bool(_cf_flag)

    except Exception as e:
        print(f"    Error finding events page: {e}")
        record_failure(
            "website_validator.find_events_page",
            str(e),
            venue_name=venue_name,
            homepage_url=homepage_url,
        )
        return None, False


def is_aggregator_url(url: str) -> bool:
    """
    Check if a URL is from an aggregator site.

    Args:
        url: The URL to check

    Returns:
        True if this is an aggregator URL
    """
    if not url:
        return False

    # Extract the domain from the URL for precise matching
    url_domain = extract_domain(url)

    # Check domain - must match as full domain or subdomain
    for agg_domain in AGGREGATOR_DOMAINS:
        # Match if: url_domain == agg_domain OR url_domain ends with .agg_domain
        if url_domain == agg_domain or url_domain.endswith('.' + agg_domain):
            return True

    # Note: We don't check AGGREGATOR_PATTERNS against all URLs anymore.
    # Patterns like /venue/ and /location/ are common on legitimate venue sites.
    # Instead, we rely on the domain blocklist and LLM verification.

    return False


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    if not url:
        return ""

    # Remove protocol
    url = re.sub(r'^https?://', '', url.lower())
    # Remove www
    url = re.sub(r'^www\.', '', url)
    # Get domain only
    domain = url.split('/')[0]
    return domain


def _normalize_url(url: str) -> str:
    """Normalize URL so candidate matching/fetching is more stable."""
    if not url:
        return ""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def _same_site(url_a: str, url_b: str) -> bool:
    """Return True if domains match exactly or as subdomains."""
    a = extract_domain(url_a)
    b = extract_domain(url_b)
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _fetch_html(url: str, timeout: int | None = None, _cf_flag: list | None = None) -> str:
    """Fetch raw HTML directly from site. Retries with curl_cffi on Cloudflare 403.

    Args:
        _cf_flag: Optional mutable list; if provided, True is appended when a CF retry fires.
                  Callers can use this to detect and record Cloudflare protection.
    """
    default_timeout = int(_settings.WEBSITE_VALIDATOR_HTML_TIMEOUT_SEC)
    timeout = _int_env("WEBSITE_VALIDATOR_HTML_TIMEOUT_SEC", timeout or default_timeout)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    normalized = _normalize_url(url)
    response = requests.get(normalized, headers=headers, timeout=timeout, allow_redirects=True)
    if response.status_code == 403 and _is_cloudflare_response(response):
        if _cf_flag is not None:
            _cf_flag.append(True)
        time.sleep(1.0)  # Brief pause before retry — avoids burst on CF-protected sites
        from curl_cffi import requests as cf_requests
        cf_resp = cf_requests.get(normalized, impersonate="chrome124", timeout=timeout, allow_redirects=True)
        cf_resp.raise_for_status()
        return cf_resp.text
    response.raise_for_status()
    return response.text


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    """Extract hyperlink URLs from raw HTML."""
    if not html:
        return []
    links = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
        href = match.group(1).strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        links.append(urljoin(base_url, href))
    return links


def _extract_iframe_links_html(html: str, base_url: str) -> list[str]:
    """Extract iframe source URLs from raw HTML."""
    if not html:
        return []
    links = []
    for match in re.finditer(r'<iframe[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE):
        src = match.group(1).strip()
        if not src or src.startswith(("javascript:", "data:", "#")):
            continue
        resolved = urljoin(base_url, src)
        if _is_media_iframe_url(resolved):
            continue
        links.append(resolved)
    return links


def _is_media_iframe_url(url: str) -> bool:
    """Return True when iframe URL is a media player/embed rather than event page content."""
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()

    if any(pattern in host for pattern in _IFRAME_MEDIA_HOST_PATTERNS):
        return True
    if any(path.endswith(ext) for ext in _IFRAME_MEDIA_EXTENSIONS):
        return True
    # Common media embed markers.
    media_tokens = ("player", "autoplay", "playlist", "clip", "reel", "video")
    if any(token in path for token in media_tokens) and any(pattern in host for pattern in ("instagram", "youtube", "vimeo", "facebook", "tiktok")):
        return True
    if "_nc_" in query and "fbcdn" in host:
        return True
    return False


def _extract_links_from_text(text: str, base_url: str) -> list[str]:
    """Extract absolute and relative URLs from markdown-like page text."""
    if not text:
        return []
    links = []

    # Markdown links: [label](url)
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target = match.group(1).strip()
        if target and not target.startswith(("mailto:", "tel:", "#")):
            links.append(urljoin(base_url, target))

    # Raw absolute URLs
    for match in re.finditer(r'https?://[^\s<>"\')]+', text):
        links.append(match.group(0).strip())

    return links



def _html_to_text(html: str) -> str:
    """Very lightweight HTML->text conversion for heuristic scoring fallback."""
    if not html:
        return ""
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_cloudflare_response(response: requests.Response) -> bool:
    """Return True if the HTTP response is a Cloudflare block or challenge page."""
    server = response.headers.get("server", "").lower()
    if "cloudflare" in server:
        return True
    if "cf-ray" in response.headers:
        return True
    body = response.text[:2000].lower()
    if "just a moment" in body and "cloudflare" in body:
        return True
    return False


def _is_dead_site(url: str) -> tuple[bool, str]:
    """Check whether a site appears dead/parked/unreachable."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    timeout = _int_env(
        "WEBSITE_VALIDATOR_HEALTH_TIMEOUT_SEC",
        int(_settings.WEBSITE_VALIDATOR_HEALTH_TIMEOUT_SEC),
    )

    # Fast health pass with HEAD first.
    try:
        head = requests.head(_normalize_url(url), headers=headers, timeout=timeout, allow_redirects=True)
        status = head.status_code
        if status in (404, 410):
            return True, f"http_status_{status}"
        # 5xx often indicates unavailable/broken endpoint for our purposes.
        if status >= 500:
            return True, f"http_status_{status}"
        if status in (200, 204, 301, 302, 303, 307, 308):
            return False, ""
    except requests.exceptions.RequestException:
        # Fall back to GET for sites that don't support HEAD or fail on it.
        pass

    try:
        response = requests.get(_normalize_url(url), headers=headers, timeout=timeout, allow_redirects=True)
        status = response.status_code
        if status in (404, 410):
            return True, f"http_status_{status}"
        if status >= 500:
            return True, f"http_status_{status}"
        if status == 403 and _is_cloudflare_response(response):
            return False, "cloudflare_blocked"
        body = response.text[:10000].lower()
        for pattern in DEAD_SITE_PATTERNS:
            if re.search(pattern, body):
                return True, f"dead_site_pattern:{pattern}"
        return False, ""
    except requests.exceptions.RequestException as e:
        return True, f"unreachable:{e}"


def _candidate_preflight_ok(url: str) -> bool:
    """Fast, cheap preflight check before expensive content probes."""
    timeout = _int_env(
        "WEBSITE_VALIDATOR_CANDIDATE_PREFLIGHT_TIMEOUT_SEC",
        int(_settings.WEBSITE_VALIDATOR_CANDIDATE_PREFLIGHT_TIMEOUT_SEC),
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    normalized = _normalize_url(url)

    # HEAD first for speed.
    try:
        head = requests.head(normalized, headers=headers, timeout=timeout, allow_redirects=True)
        status = head.status_code
        if status in (404, 410):
            return False
        if status >= 500:
            return False
        if status in (200, 204, 301, 302, 303, 307, 308):
            return True
    except requests.exceptions.RequestException:
        pass

    # Fallback GET for sites that don't support HEAD correctly.
    try:
        resp = requests.get(normalized, headers=headers, timeout=timeout, allow_redirects=True)
        status = resp.status_code
        if status in (404, 410):
            return False
        if status >= 500:
            return False
        return status < 500
    except requests.exceptions.RequestException:
        return False


def _candidate_event_paths(homepage_url: str) -> list[str]:
    """Common event-path candidates on venue websites."""
    base = _normalize_url(homepage_url).rstrip("/")
    return [
        f"{base}/events",
        f"{base}/events.html",
        f"{base}/calendar",
        f"{base}/calendar.html",
        f"{base}/calendar/exhibitions",
        f"{base}/pages/calendar",
        f"{base}/shows",
        f"{base}/shows.html",
        f"{base}/exhibitions",
        f"{base}/exhibitions/upcoming",
        f"{base}/upcoming-exhibitions",
        f"{base}/upcoming",
        f"{base}/happenings",
        f"{base}/whatson",
        f"{base}/whats-on",
        f"{base}/program",
        f"{base}/programs",
        f"{base}/public-programs",
        f"{base}/public-programming",
        f"{base}/programming",
    ]


def _candidate_domain_variants(url: str) -> list[str]:
    """Generate lightweight hostname variants (www/non-www and optional leading 'the')."""
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    scheme = parsed.scheme or "https"
    host = parsed.netloc.lower()
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""

    if not host:
        return [normalized]

    root = host[4:] if host.startswith("www.") else host
    hosts = [host]

    if root != host:
        hosts.append(root)
    else:
        hosts.append(f"www.{root}")

    parts = root.split(".")
    if len(parts) >= 2:
        label = parts[0]
        suffix = ".".join(parts[1:])
        label_variants = {label}
        if label.startswith("the") and len(label) > 3:
            label_variants.add(label[3:])
        else:
            label_variants.add(f"the{label}")
        for lv in label_variants:
            hosts.append(f"{lv}.{suffix}")
            hosts.append(f"www.{lv}.{suffix}")

    variants = []
    seen = set()
    for candidate_host in hosts:
        candidate = _normalize_url(f"{scheme}://{candidate_host}{path}{query}")
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    return variants[:8]


def _resolve_reachable_url_variant(url: str) -> tuple[str, str]:
    """
    Resolve to a reachable URL, trying hostname variants only if needed.

    Returns:
        (resolved_url, failure_reason)
        - resolved_url is non-empty when reachable.
        - failure_reason populated only when no variant is reachable.
    """
    variants = _candidate_domain_variants(url)
    first_reason = ""
    for candidate in variants:
        dead, reason = _is_dead_site(candidate)
        if not dead:
            return candidate, reason  # reason may be "cloudflare_blocked" or ""
        if not first_reason:
            first_reason = reason
    return "", first_reason or "unreachable"


def _build_events_candidates(homepage_url: str, homepage_text: str, homepage_html: str) -> list[str]:
    """Assemble likely event URLs from page text/html plus common paths."""
    text_links = _extract_links_from_text(homepage_text, homepage_url)
    html_links = _extract_links_from_html(homepage_html, homepage_url)
    candidates = []
    candidates.extend(text_links)
    candidates.extend(html_links)
    candidates.extend(_candidate_event_paths(homepage_url))
    candidates.append(homepage_url)

    filtered = []
    seen = set()
    common_candidates = {_normalize_url(u) for u in _candidate_event_paths(homepage_url)}
    discovered_candidates = {_normalize_url(u) for u in (text_links + html_links)}
    static_ext = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".woff", ".woff2")

    for url in candidates:
        norm = _normalize_url(url)
        if not norm or norm in seen:
            continue
        if not _same_site(norm, homepage_url):
            continue

        parsed = urlparse(norm)
        path = parsed.path.lower()
        query = parsed.query.lower()
        path_query = f"{path}?{query}" if query else path
        is_common_candidate = norm in common_candidates
        has_keyword = any(keyword in path_query for keyword in EVENT_PATH_KEYWORDS)
        is_homepage = norm.rstrip("/") == _normalize_url(homepage_url).rstrip("/")
        segments = [segment for segment in path.split("/") if segment]

        if path.endswith(static_ext):
            continue
        # Skip deep event detail pages (we want listing endpoints).
        if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path):
            continue
        # Skip individual event/exhibition detail pages identified by a numeric ID.
        if re.search(r"/\d{4,}$", path) and not is_common_candidate:
            continue
        # Skip event/exhibition detail pages: /exhibitions/slug, /events/slug, /shows/slug, etc.
        # These are individual item pages; we want the parent listing (e.g. /exhibitions, /events).
        # Keep known sub-listing segments like /upcoming, /film, /current that are valid listing views.
        _EVENTS_PARENT_SEGS = frozenset({
            "exhibitions", "exhibition", "events", "shows", "show",
            "performances", "performance", "talks", "talk",
            "calendar", "whats-on", "whatson", "happenings",
        })
        _SUBLISTING_SEGS = frozenset({
            "upcoming", "current", "past", "history", "archive", "all",
            "film", "films", "programs", "series", "special",
            "exhibitions", "shows", "performances",
        })
        if (
            len(segments) >= 2
            and segments[0] in _EVENTS_PARENT_SEGS
            and segments[1] not in _SUBLISTING_SEGS
            and not is_common_candidate
        ):
            continue
        if len(segments) > 4 and not is_common_candidate:
            continue
        if len(path) > 80 and not is_common_candidate:
            continue

        if has_keyword or is_common_candidate or is_homepage:
            seen.add(norm)
            filtered.append(norm)

    # Prefer likely listing pages over detail pages.
    def _priority(candidate: str) -> tuple[int, int]:
        parsed = urlparse(candidate)
        path = parsed.path.lower()
        query = parsed.query.lower()
        path_query = f"{path}?{query}" if query else path
        seg_count = len([s for s in path.split("/") if s])
        score = 0
        if candidate in discovered_candidates:
            score -= 3
        if candidate in common_candidates:
            score += 3
        # Prefer events/program listing pages first; exhibitions/upcoming remain valid but lower priority.
        if "/events" in path_query:
            score -= 4
        if any(k in path_query for k in ("/program", "/programs", "public-program")):
            score -= 3
        if any(k in path_query for k in ("/calendar", "/shows", "/happenings")):
            score -= 2
        if any(k in path_query for k in ("/exhibitions", "exhibition", "/upcoming", "upcoming")):
            score -= 1
        if any(k in path_query for k in ("calendarid=", "calendar=")):
            score -= 1
        # Penalise archive/history pages — they have past events, not upcoming ones.
        if any(k in path_query for k in ("/history", "/archive", "/past", "/previous")):
            score += 4
        # Penalise URLs with query strings — prefer clean root paths over filtered variants.
        if parsed.query:
            score += 2
        return (score, seg_count)

    filtered.sort(key=_priority)

    # Guarantee the top common candidates (generic event paths) are always included
    # even when many discovered links push them past the slice limit.
    guaranteed = [c for c in filtered if c in common_candidates][:3]
    top = filtered[:12]
    for c in guaranteed:
        if c not in top:
            top.append(c)
    return top


def search_venue_website(venue_name: str, city: str = "NYC") -> tuple[str | None, bool]:
    """
    Search for a venue's official website.

    Args:
        venue_name: Name of the venue
        city: City for context

    Returns:
        Tuple of (url, is_closed):
        - url: Official website URL if found, None otherwise
        - is_closed: True if venue is confirmed permanently closed
    """
    from utils.llm import generate_content

    prompt = f"""Find the official website for this venue:

Venue: {venue_name}
City: {city}

Rules:
1. If the venue is PERMANENTLY CLOSED (no longer operating), return "CLOSED"
2. If open, return ONLY the official venue website URL
3. Do NOT return aggregator sites like Yelp, TripAdvisor, Eventbrite, ClassPass, Google Maps, Facebook, Instagram, Peerspace, Wikipedia, etc.
4. The website should be the venue's own domain (e.g., venuename.com)
5. If you cannot find an official website but venue may still exist, return "NONE"

Return ONLY one of: the URL, "CLOSED", or "NONE". Nothing else."""

    try:
        response_orig = generate_content(prompt).strip()
        response = response_orig.upper()

        # Check if venue is closed
        if "CLOSED" in response:
            return None, True

        # Check if no website found
        if response == "NONE":
            return None, False

        # Extract URL from response (use original case)
        url_match = re.search(r'https?://[^\s<>"]+', response_orig)
        if url_match:
            url = url_match.group()
            # Verify it's not an aggregator
            if is_aggregator_url(url):
                return None, False
            return url, False

        # Maybe just a domain was returned
        if '.' in response_orig and ' ' not in response_orig:
            url = response_orig if response_orig.startswith('http') else f'https://{response_orig}'
            if not is_aggregator_url(url):
                return url, False

        return None, False

    except Exception as e:
        print(f"    Error searching for website: {e}")
        return None, False


def search_and_verify_website(
    venue_name: str,
    city: str = "NYC",
    rejected_domains: list[str] | None = None,
) -> tuple[str | None, bool, str, str]:
    """
    Search for official venue website and verify it with LLM.

    Args:
        venue_name: Name of the venue
        city: City for context
        rejected_domains: List of domains already rejected (to avoid re-suggesting)

    Returns:
        Tuple of (url, is_closed, rejection_reason, outcome):
        - url: Verified official URL if found, None otherwise
        - is_closed: True if venue is confirmed permanently closed
        - rejection_reason: If URL was rejected, why (empty string if accepted or not found)
        - outcome: One of "verified", "aggregator_rejected", "search_failed", "ambiguous", "closed"
    """
    from utils.llm import generate_content

    rejected_domains = rejected_domains or []
    rejected_list = ", ".join(rejected_domains) if rejected_domains else "none"

    prompt = f"""Find the official website for this venue:

Venue: {venue_name}
City: {city}
Already rejected domains (do NOT suggest these): {rejected_list}

Rules:
1. If the venue is PERMANENTLY CLOSED (no longer operating), return "CLOSED"
2. Return ONLY the official venue website URL - the venue's own domain
3. Do NOT return:
   - Aggregator sites (Yelp, TripAdvisor, TimeOut, etc.)
   - Ticketing sites (Eventbrite, Ticketmaster, etc.)
   - Social media (Facebook, Instagram, etc.)
   - Museum/venue directories (whichmuseum.com, atlasobscura.com, etc.)
   - Tourism guides (nycgo.com, iloveny.com, etc.)
   - Any domain in the rejected list above
4. The website should be owned by the venue itself
5. If you cannot find an official website, return "NONE"

Return ONLY one of: the URL, "CLOSED", or "NONE". Nothing else."""

    try:
        increment("website_validator.search.calls")
        response_orig = generate_content(prompt).strip()
        response = response_orig.upper()

        if "CLOSED" in response:
            return None, True, "", "closed"

        if response == "NONE":
            return None, False, "search returned none", "search_failed"

        # Extract URL from response
        url_match = re.search(r'https?://[^\s<>"]+', response_orig)
        if url_match:
            url = url_match.group()
            domain = extract_domain(url)

            # Check blocklist
            if is_aggregator_url(url):
                return None, False, f"Blocklist: {domain}", "aggregator_rejected"

            # Check if already rejected
            if domain in rejected_domains:
                return None, False, f"Already rejected: {domain}", "ambiguous"

            # LLM verification
            is_official, reason = verify_official_website(url, venue_name, city)
            if is_official:
                return url, False, "", "verified"
            else:
                return None, False, f"LLM rejected: {reason}", "aggregator_rejected"

        # Maybe just a domain was returned
        if '.' in response_orig and ' ' not in response_orig:
            url = response_orig if response_orig.startswith('http') else f'https://{response_orig}'
            domain = extract_domain(url)

            if not is_aggregator_url(url) and domain not in rejected_domains:
                is_official, reason = verify_official_website(url, venue_name, city)
                if is_official:
                    return url, False, "", "verified"
                else:
                    return None, False, f"LLM rejected: {reason}", "aggregator_rejected"

        return None, False, "response was not URL/NONE/CLOSED", "ambiguous"

    except Exception as e:
        record_failure("website_validator.search", str(e), venue_name=venue_name, city=city)
        return None, False, f"Error: {e}", "search_failed"


def validate_venue_website(
    venue: dict,
    city: str = "NYC",
    max_attempts: int | None = None,
    verify_with_llm: bool = True,
    find_events: bool = True,
    refresh_events_url_if_no_events: bool = False,
) -> dict:
    """
    Validate and potentially discover a venue's website.

    Uses a multi-step process:
    1. Check current URL against blocklist
    2. If passes blocklist, verify with LLM that it's the official site
    3. If rejected, search for the correct official website
    4. Repeat search up to max_attempts times, tracking rejected domains
    5. Once verified, find the events/calendar page URL

    Args:
        venue: Venue dict with name, website, etc.
        city: City for search context
        max_attempts: Max search attempts before giving up (defaults to settings value)
        verify_with_llm: If True, use LLM to verify URLs aren't aggregators (default True)
        find_events: If True, also find and store the events page URL (default True)
        refresh_events_url_if_no_events: If True, re-run find_events_page for verified venues
            that have event_count == 0, even if events_url is already set. Useful for
            correcting bad events_url values on venues that have never returned events.

    Returns:
        Updated venue dict with:
        - website: validated/discovered URL or ""
        - events_url: URL to events/calendar page if found
        - website_status: "verified", "reachable_no_events_page", "aggregator_rejected",
                         "search_failed", "ambiguous", "closed", "dead_site", "unreachable",
                         "ticketmaster_skip"
        - website_attempts: number of search attempts made
    """
    venue = dict(venue)  # Don't modify original
    name = venue.get("name", "")
    current_url = venue.get("website", "")
    attempts = venue.get("website_attempts", 0)
    if not max_attempts or max_attempts < 1:
        max_attempts = int(_settings.WEBSITE_VALIDATOR_MAX_ATTEMPTS)
    attempt_delay = _float_env(
        "WEBSITE_VALIDATOR_ATTEMPT_DELAY_SEC",
        float(_settings.WEBSITE_VALIDATOR_ATTEMPT_DELAY_SEC),
    )
    rejected_domains = []  # Track rejected domains to avoid re-suggesting

    tm_id = str(venue.get("ticketmaster_venue_id", "") or "").strip()
    if tm_id and tm_id.lower() != "not_found":
        venue["website_status"] = "ticketmaster_skip"
        venue["validation_reason"] = f"skipped website validation: has ticketmaster_venue_id={tm_id}"
        log_event("website_validate_skipped_ticketmaster", venue_name=name, ticketmaster_venue_id=tm_id)
        return venue

    increment("website_validator.validate.calls")
    log_event("website_validate_start", venue_name=name, city=city, current_url=current_url)

    # Check if already validated
    status = venue.get("website_status", "")
    if status in ("verified", "reachable_no_events_page", "search_failed", "closed", "ambiguous", "dead_site", "unreachable"):
        # Re-run find_events_page if: no events_url yet, OR venue is verified with zero
        # events and the caller has opted in to refreshing stale event URLs.
        has_no_events_url = not venue.get("events_url")
        zero_event_venue = (
            refresh_events_url_if_no_events
            and (venue.get("event_count") or 0) == 0
        )
        if status in ("verified", "reachable_no_events_page") and find_events and (has_no_events_url or zero_event_venue):
            events_url, cf_detected = find_events_page(venue.get("website", ""), name)
            if cf_detected:
                venue["cloudflare_protected"] = "yes"
            if events_url:
                venue["events_url"] = events_url
                venue["website_status"] = "verified"
            else:
                venue["website_status"] = "reachable_no_events_page"
            if venue.get("website_status") == "verified" and venue.get("website") and not venue.get("feed_url"):
                from .feed_finder import find_feeds
                feed_result = find_feeds(venue["website"])
                for _feed_type in ("ical", "rss"):
                    _feed_url = feed_result.get("feeds", {}).get(_feed_type)
                    if _feed_url:
                        venue["feed_url"] = _feed_url
                        venue["feed_type"] = _feed_type
                        break
            return venue

        # Allow retrying prior search failures while attempts remain.
        if status == "search_failed" and attempts < max_attempts:
            venue["website_status"] = ""
            venue["validation_reason"] = ""
        else:
            # Opportunistically discover feeds for verified venues that don't have one yet
            if status == "verified" and venue.get("website") and not venue.get("feed_url"):
                from .feed_finder import find_feeds
                feed_result = find_feeds(venue["website"])
                for _feed_type in ("ical", "rss"):
                    _feed_url = feed_result.get("feeds", {}).get(_feed_type)
                    if _feed_url:
                        venue["feed_url"] = _feed_url
                        venue["feed_type"] = _feed_type
                        break
            return venue

    # Check current website against blocklist first
    if current_url:
        resolved_current, health_reason = _resolve_reachable_url_variant(current_url)
        if not resolved_current:
            status = "unreachable" if health_reason.startswith("unreachable:") else "dead_site"
            venue["website_status"] = status
            venue["validation_reason"] = health_reason
            venue["website"] = ""
            current_url = ""
        else:
            if health_reason == "cloudflare_blocked":
                venue["cloudflare_protected"] = "yes"
            if _normalize_url(resolved_current) != _normalize_url(current_url):
                log_event(
                    "website_variant_selected",
                    venue_name=name,
                    original_url=current_url,
                    resolved_url=resolved_current,
                )
            current_url = resolved_current
            venue["website"] = current_url

        if current_url and is_aggregator_url(current_url):
            print(f"    Rejecting (blocklist): {extract_domain(current_url)}")
            rejected_domains.append(extract_domain(current_url))
            venue["website"] = ""
            venue["website_status"] = "aggregator_rejected"
            current_url = ""
        elif current_url and verify_with_llm:
            # LLM verification for URLs that pass blocklist
            print(f"    Verifying: {extract_domain(current_url)}")
            is_official, reason = verify_official_website(current_url, name, city)
            if is_official:
                print(f"    ✓ Verified: {reason}")
                venue["website_status"] = "verified"
                venue["validation_reason"] = reason
                # Find events page
                if find_events:
                    events_url, cf_detected = find_events_page(current_url, name)
                    if cf_detected:
                        venue["cloudflare_protected"] = "yes"
                    if events_url:
                        venue["events_url"] = events_url
                        venue["website_status"] = "verified"
                    else:
                        venue["website_status"] = "reachable_no_events_page"
                log_event("website_validate_success", venue_name=name, website=current_url)
                if venue.get("website_status") == "verified" and venue.get("website") and not venue.get("feed_url"):
                    from .feed_finder import find_feeds
                    feed_result = find_feeds(venue["website"])
                    for _feed_type in ("ical", "rss"):
                        _feed_url = feed_result.get("feeds", {}).get(_feed_type)
                        if _feed_url:
                            venue["feed_url"] = _feed_url
                            venue["feed_type"] = _feed_type
                            break
                return venue
            else:
                print(f"    ✗ Rejected: {reason}")
                rejected_domains.append(extract_domain(current_url))
                venue["website"] = ""
                venue["website_status"] = "aggregator_rejected"
                venue["validation_reason"] = reason
                current_url = ""
        elif current_url:
            # No LLM verification - trust blocklist
            venue["website_status"] = "verified"
            if find_events:
                events_url, cf_detected = find_events_page(current_url, name)
                if cf_detected:
                    venue["cloudflare_protected"] = "yes"
                if events_url:
                    venue["events_url"] = events_url
            if venue.get("website") and not venue.get("feed_url"):
                from .feed_finder import find_feeds
                feed_result = find_feeds(venue["website"])
                for _feed_type in ("ical", "rss"):
                    _feed_url = feed_result.get("feeds", {}).get(_feed_type)
                    if _feed_url:
                        venue["feed_url"] = _feed_url
                        venue["feed_type"] = _feed_type
                        break
            return venue

    # Search for official website (try multiple times if needed)
    while attempts < max_attempts:
        attempts += 1
        print(f"    Searching (attempt {attempts}/{max_attempts})...")

        found_url, is_closed, rejection_reason, outcome = search_and_verify_website(
            name, city, rejected_domains
        )

        if is_closed:
            print(f"    Venue is permanently closed")
            venue["website_status"] = "closed"
            venue["website_attempts"] = attempts
            venue["validation_reason"] = "venue reported as permanently closed"
            log_event("website_validate_closed", venue_name=name, attempts=attempts)
            return venue

        if found_url:
            resolved_url, health_reason = _resolve_reachable_url_variant(found_url)
            if not resolved_url:
                dead_status = "unreachable" if health_reason.startswith("unreachable:") else "dead_site"
                domain = extract_domain(found_url)
                rejected_domains.append(domain)
                rejection_reason = f"{dead_status}:{health_reason}"
                print(f"    ✗ Rejected discovered site {domain}: {health_reason}")
                venue["validation_reason"] = rejection_reason
                venue["website_status"] = dead_status
                record_failure(
                    "website_validator.validate",
                    rejection_reason,
                    venue_name=name,
                    attempt=attempts,
                    outcome=dead_status,
                )
                time.sleep(attempt_delay)
                continue
            if health_reason == "cloudflare_blocked":
                venue["cloudflare_protected"] = "yes"
            if _normalize_url(resolved_url) != _normalize_url(found_url):
                log_event(
                    "website_variant_selected",
                    venue_name=name,
                    original_url=found_url,
                    resolved_url=resolved_url,
                )
            found_url = resolved_url

            print(f"    ✓ Found official website: {found_url}")
            venue["website"] = found_url
            venue["website_status"] = "verified"
            venue["website_attempts"] = attempts
            venue["validation_reason"] = "verified through search and LLM checks"
            # Find events page
            if find_events:
                events_url, cf_detected = find_events_page(found_url, name)
                if cf_detected:
                    venue["cloudflare_protected"] = "yes"
                if events_url:
                    venue["events_url"] = events_url
                    venue["website_status"] = "verified"
                else:
                    venue["website_status"] = "reachable_no_events_page"
            log_event("website_validate_success", venue_name=name, website=found_url, attempts=attempts)
            if venue.get("website_status") == "verified" and venue.get("website") and not venue.get("feed_url"):
                from .feed_finder import find_feeds
                feed_result = find_feeds(venue["website"])
                for _feed_type in ("ical", "rss"):
                    _feed_url = feed_result.get("feeds", {}).get(_feed_type)
                    if _feed_url:
                        venue["feed_url"] = _feed_url
                        venue["feed_type"] = _feed_type
                        break
            return venue

        if rejection_reason:
            # Extract domain from rejection reason if possible
            domain_match = re.search(r'([a-z0-9.-]+\.[a-z]{2,})', rejection_reason.lower())
            if domain_match:
                rejected_domains.append(domain_match.group(1))
            print(f"    ✗ {rejection_reason}")
            venue["validation_reason"] = rejection_reason
            venue["website_status"] = outcome or "ambiguous"
            record_failure(
                "website_validator.validate",
                rejection_reason,
                venue_name=name,
                attempt=attempts,
                outcome=outcome,
            )

        # Small delay between attempts
        time.sleep(attempt_delay)

    # Exhausted all attempts
    if not venue.get("website_status"):
        venue["website_status"] = "search_failed"
    venue["website_attempts"] = attempts
    venue["validation_reason"] = venue.get("validation_reason", f"no official site found in {attempts} attempts")
    print(f"    Could not find official website after {attempts} attempts")
    log_event(
        "website_validate_exhausted",
        venue_name=name,
        attempts=attempts,
        status=venue["website_status"],
        reason=venue.get("validation_reason", ""),
    )
    return venue


def validate_venues_batch(
    venues: list[dict],
    city: str = "NYC",
    max_attempts: int | None = None,
    delay: float | None = None,
) -> list[dict]:
    """
    Validate websites for a batch of venues.

    Args:
        venues: List of venue dicts
        city: City for search context
        max_attempts: Max search attempts per venue
        delay: Delay between searches (rate limiting)

    Returns:
        List of updated venue dicts
    """
    results = []
    if not max_attempts or max_attempts < 1:
        max_attempts = int(_settings.WEBSITE_VALIDATOR_MAX_ATTEMPTS)
    if delay is None:
        delay = float(_settings.WEBSITE_VALIDATOR_BATCH_DELAY_SEC)

    for i, venue in enumerate(venues, 1):
        name = venue.get("name", "")[:40]
        print(f"[{i}/{len(venues)}] {name}")

        updated = validate_venue_website(venue, city, max_attempts)
        results.append(updated)

        # Rate limit if we did a search
        if updated.get("website_attempts", 0) > venue.get("website_attempts", 0):
            time.sleep(delay)

    return results


def clean_aggregator_urls(venues: list[dict]) -> tuple[list[dict], int]:
    """
    Remove aggregator URLs from venues without searching for replacements.

    Args:
        venues: List of venue dicts

    Returns:
        Tuple of (updated venues, count of URLs removed)
    """
    removed = 0
    results = []

    for venue in venues:
        venue = dict(venue)
        url = venue.get("website", "")

        if url and is_aggregator_url(url):
            venue["website"] = ""
            venue["website_status"] = "aggregator_rejected"
            removed += 1

        results.append(venue)

    return results, removed


def get_validation_summary(venues: list[dict]) -> dict:
    """Get summary of website validation status."""
    by_status = {}
    with_website = 0
    without_website = 0

    for venue in venues:
        status = venue.get("website_status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        if venue.get("website"):
            with_website += 1
        else:
            without_website += 1

    return {
        "total": len(venues),
        "with_website": with_website,
        "without_website": without_website,
        "by_status": by_status,
    }


if __name__ == "__main__":
    # Test with some example URLs
    test_urls = [
        "https://www.beacontheatre.com",
        "https://www.yelp.com/biz/beacon-theatre-new-york",
        "https://classpass.com/studios/tabata-ultimate-fitness",
        "https://www.peerspace.com/pages/listings/123",
        "https://www.caveat.nyc",
        "https://goo.gl/maps/abc123",
        "https://www.eventbrite.com/venue/beacon-theatre",
        "",
    ]

    print("Testing aggregator detection:")
    print("=" * 60)
    for url in test_urls:
        is_agg = is_aggregator_url(url)
        status = "AGGREGATOR" if is_agg else "OK"
        print(f"{status:12} {url[:50]}")
