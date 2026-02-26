"""Core venue discovery logic using web search and LLM extraction."""

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import generate_content
from .state import Venue
from .website_validator import is_aggregator_url


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()
_last_search_time = 0

# Domains to skip when scraping - these are rental platforms, not venue directories
RENTAL_PLATFORM_DOMAINS = frozenset([
    "peerspace.com",
    "giggster.com",
    "splacer.co",
    "tagvenue.com",
    "venuebook.com",
    "thevendry.com",
    "storefront.com",
    "liquidspace.com",
    "breather.com",
    "deskpass.com",
])

# Patterns that indicate a rental listing, not a real venue
RENTAL_LISTING_PATTERNS = [
    r'\d+\s*sqft',  # "5000 sqft"
    r'\d+\s*sq\s*ft',  # "5000 sq ft"
    r'with\s+(free\s+)?equipment',  # "with free equipment"
    r'meeting\s+room',  # "meeting room"
    r'seminar\s+room',  # "seminar room"
    r'conference\s+room',  # "conference room"
    r'podcast\s+studio',  # "podcast studio"
    r'recording\s+studio',  # "recording studio" (generic)
    r'photo\s+studio',  # "photo studio"
    r'film\s+studio',  # "film studio"
    r'production\s+space',  # "production space"
    r'event\s+space\s+(for|in)',  # "event space for/in"
    r'loft\s+(with|space|studio)',  # "loft with/space/studio"
    r'brownstone',  # "brownstone"
    r'townhouse',  # "townhouse"
    r'apartment',  # "apartment"
    r'modern\s+(brooklyn|manhattan|nyc)',  # "modern brooklyn"
    r'bright\s+(versatile|modern|spacious)',  # "bright versatile"
    r'skyline\s+view',  # "skyline view"
    r'rooftop\s+access',  # "rooftop access"
    r'flexible\s+(space|room)',  # "flexible space"
    r'multi-?use',  # "multi-use" or "multiuse"
    r'content\s+creation',  # "content creation"
]


def is_rental_platform_url(url: str) -> bool:
    """Check if URL is from a rental platform."""
    if not url:
        return False
    url_lower = url.lower()
    for domain in RENTAL_PLATFORM_DOMAINS:
        if domain in url_lower:
            return True
    return False


def is_rental_listing_name(name: str) -> bool:
    """Check if venue name looks like a rental listing, not a real venue."""
    if not name:
        return True

    name_lower = name.lower()

    # Check against patterns
    for pattern in RENTAL_LISTING_PATTERNS:
        if re.search(pattern, name_lower):
            return True

    # Names that are too descriptive (contain multiple adjectives or descriptions)
    # Real venues have proper names, not descriptions
    descriptive_words = ['bright', 'modern', 'cozy', 'spacious', 'elegant',
                         'stylish', 'unique', 'versatile', 'artsy', 'chic',
                         'upscale', 'industrial', 'rustic', 'airy', 'sunlit']
    word_count = sum(1 for word in descriptive_words if word in name_lower)
    if word_count >= 2:
        return True

    # Names that start with adjectives AND contain rental-type words
    words = name_lower.split()
    if words and words[0] in descriptive_words:
        rental_words = ['loft', 'space', 'studio', 'room', 'house', 'home',
                        'apartment', 'brownstone', 'townhouse']
        if any(w in name_lower for w in rental_words):
            return True

    # Names with numbers followed by generic rental descriptors
    # "23B Manhattan Studio" is rental, but "54 Below" or "230 Fifth" are real venues
    if re.match(r'^\d+[a-z]?\s+', name_lower):
        # Only reject if followed by rental-type words
        rental_indicators = ['studio', 'space', 'loft', 'room', 'floor',
                             'sqft', 'sq ft', 'apartment', 'suite']
        if any(ind in name_lower for ind in rental_indicators):
            return True

    # Names containing "studio" with location modifiers (rental pattern)
    if re.search(r'(manhattan|brooklyn|nyc|midtown|chelsea|tribeca|soho)\s+studio', name_lower):
        return True

    # Names with "+ equipment" or "free equipment"
    if re.search(r'\+\s*(free\s+)?equipment', name_lower):
        return True

    # Names that are clearly rental descriptions
    if re.search(r'(for rent|available for|book now|hourly)', name_lower):
        return True

    # Names with class/workshop/seminar patterns (but not if it's a theater name)
    if re.search(r'(class|workshop|seminar)[/\s]', name_lower):
        if not re.search(r'(theater|theatre|playhouse|stage)', name_lower):
            return True

    # Names with "media studio" or "wellness" rental patterns
    if re.search(r'(eco-?wellness|media\s+studio)', name_lower):
        return True

    return False


VENUE_EXTRACTION_PROMPT = """You are extracting venue information from a webpage about {city} {category}.

Extract venues that are ESTABLISHED BUSINESSES with their own identity. For each venue, provide:
- name: The venue name (must be a proper business name, not a description)
- address: Full address if available, otherwise just "{city}" or neighborhood
- neighborhood: Neighborhood/area within the city (e.g., "East Village", "Williamsburg")
- website: Venue website URL if available, otherwise empty string
- description: Brief description (max 100 chars)
- venue_type: The PRIMARY type of venue based on what events they host. Choose ONE from:
  theater, comedy_club, music_venue, museum, gallery, cinema, dance_venue,
  sports_arena, bookstore, library, cultural_center, bar_with_events,
  event_space, park, outdoor_venue, variety_venue, other

DO NOT EXTRACT:
- Rental space listings (Peerspace, Giggster, Splacer, etc.) - these have descriptive names like "Modern Loft with Skyline Views" or "Bright Studio Space"
- Generic meeting rooms or seminar rooms
- Private homes, townhouses, apartments, or brownstones for rent
- Podcast/recording studios that are just rental spaces
- Dance companies, theater troupes, or performance groups (extract venues, not performers)
- Geographic locations (neighborhoods, bridges, beaches) without an actual venue
- Names that include dimensions (sqft), equipment lists, or descriptive phrases

GOOD venue names: "The Comedy Cellar", "Beacon Theatre", "Blue Note Jazz Club", "MoMA"
BAD venue names: "Bright Modern Brooklyn Brownstone", "5000 sqft Loft with Equipment", "Podcast Studio in Chelsea"

Return ONLY a valid JSON array. No markdown, no explanation. Example:
[
  {{"name": "The Comedy Cellar", "address": "117 MacDougal St, New York, NY 10012", "neighborhood": "Greenwich Village", "website": "https://comedycellar.com", "description": "Legendary comedy club featuring top comedians nightly", "venue_type": "comedy_club"}}
]

If no venues found, return: []

PAGE CONTENT:
{content}
"""


def _rate_limit():
    """Enforce rate limiting between searches."""
    global _last_search_time
    now = time.time()
    elapsed = now - _last_search_time
    delay = _settings.VENUE_SEARCH_DELAY
    if elapsed < delay:
        sleep_time = delay - elapsed
        print(f"    Rate limiting: sleeping {sleep_time:.1f}s...", flush=True)
        time.sleep(sleep_time)
    _last_search_time = time.time()


def search_web(query: str) -> list[dict]:
    """
    Search the web and return results.

    Returns list of dicts with 'title', 'url', 'snippet' keys.
    NOTE: Requires a search provider to be configured.
    """
    raise NotImplementedError(
        "search_web() requires a web search provider. "
        "Configure a search API (e.g. Google Custom Search) and implement here."
    )


def extract_venues_from_content(
    content: str,
    city: str,
    category: str,
) -> list[Venue]:
    """Use LLM to extract venues from page content."""
    # Truncate content if too long
    max_chars = _settings.SCRAPER_MAX_CONTENT_CHARS
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... [truncated]"

    prompt = VENUE_EXTRACTION_PROMPT.format(
        city=city,
        category=category,
        content=content,
    )

    response_text = generate_content(prompt).strip()

    # Parse JSON response
    venues = []
    try:
        raw_venues = json.loads(response_text)
        if isinstance(raw_venues, list):
            for raw in raw_venues:
                name = raw.get("name", "")
                if name:
                    # Skip rental listings
                    if is_rental_listing_name(name):
                        continue
                    # Use LLM-determined venue_type if available, else fall back to search category
                    venue_type = raw.get("venue_type", "").replace("_", " ") or category
                    # Check and clear aggregator URLs
                    website = raw.get("website", "")
                    website_status = ""
                    if website:
                        if is_aggregator_url(website):
                            website = ""  # Clear aggregator URL
                            website_status = "aggregator_rejected"
                        else:
                            website_status = "verified"
                    venues.append(Venue(
                        name=raw.get("name", "")[:100],
                        address=raw.get("address", city),
                        city=city,
                        neighborhood=raw.get("neighborhood", ""),
                        website=website,
                        category=venue_type,
                        description=raw.get("description", "")[:100],
                        source=f"web_search:{category}",
                        address_verified="",
                        website_status=website_status,
                        website_attempts=0,
                        preferred_event_source="",
                        api_endpoint="",
                        ticketmaster_venue_id="",
                    ))
    except json.JSONDecodeError:
        # Try to find JSON array in response
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            try:
                raw_venues = json.loads(match.group())
                if isinstance(raw_venues, list):
                    for raw in raw_venues:
                        name = raw.get("name", "")
                        if name:
                            # Skip rental listings
                            if is_rental_listing_name(name):
                                continue
                            venue_type = raw.get("venue_type", "").replace("_", " ") or category
                            # Check and clear aggregator URLs
                            website = raw.get("website", "")
                            website_status = ""
                            if website:
                                if is_aggregator_url(website):
                                    website = ""
                                    website_status = "aggregator_rejected"
                                else:
                                    website_status = "verified"
                            venues.append(Venue(
                                name=raw.get("name", "")[:100],
                                address=raw.get("address", city),
                                city=city,
                                neighborhood=raw.get("neighborhood", ""),
                                website=website,
                                category=venue_type,
                                description=raw.get("description", "")[:100],
                                source=f"web_search:{category}",
                                address_verified="",
                                website_status=website_status,
                                website_attempts=0,
                                preferred_event_source="",
                                api_endpoint="",
                                ticketmaster_venue_id="",
                            ))
            except json.JSONDecodeError:
                pass

    return venues


def search_venues_by_category(city: str, category: str) -> list[Venue]:
    """
    Search for venues in a city by category.

    1. Web search for "{city} {category}"
    2. Fetch top results with direct HTTP
    3. Extract venues with LLM
    """
    print(f"  Searching: {city} {category}...", flush=True)

    # Search the web
    query = f"{city} {category} list"
    results = search_web(query)

    if not results:
        print(f"    No search results found", flush=True)
        return []

    print(f"    Found {len(results)} search results", flush=True)

    # Fetch and process top 3 results
    all_venues = []

    for i, result in enumerate(results[:3]):
        url = result.get("url")
        if not url:
            continue

        # Skip rental platforms - they list spaces, not established venues
        if is_rental_platform_url(url):
            print(f"    Skipping rental platform: {url[:50]}...", flush=True)
            continue

        # Skip aggregator sites
        if is_aggregator_url(url):
            print(f"    Skipping aggregator: {url[:50]}...", flush=True)
            continue

        print(f"    Fetching: {url[:60]}...", flush=True)

        try:
            import requests
            from bs4 import BeautifulSoup
            timeout = int(_settings.VENUE_DISCOVERY_SEARCH_TIMEOUT_SEC)
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.get_text(separator="\n", strip=True)[:_settings.SCRAPER_MAX_CONTENT_CHARS]

            # Extract venues from this page
            venues = extract_venues_from_content(content, city, category)
            print(f"    Extracted {len(venues)} venues", flush=True)

            all_venues.extend(venues)

        except Exception as e:
            print(f"    Error fetching {url}: {e}", flush=True)
            continue

    return all_venues


def deduplicate_venues(venues: list[Venue]) -> list[Venue]:
    """Remove duplicate venues based on name similarity."""
    seen_names = set()
    unique_venues = []

    for venue in venues:
        # Normalize name for comparison
        name_key = venue["name"].lower().strip()
        # Remove common suffixes/prefixes
        name_key = re.sub(r'\s+(nyc|ny|club|venue|theater|theatre)$', '', name_key)
        name_key = re.sub(r'^the\s+', '', name_key)

        if name_key not in seen_names:
            seen_names.add(name_key)
            unique_venues.append(venue)

    return unique_venues


def discover_venues(
    city: str,
    categories: list[str] | None = None,
    force: bool = False,
) -> list[Venue]:
    """
    Discover venues in a city across all categories.

    Writes to cache incrementally after each category.
    Skips categories that have already been searched (unless force=True).

    Args:
        city: City to search (e.g., "NYC", "Los Angeles")
        categories: Optional list of categories to search
                   (defaults to VENUE_CATEGORIES_MVP)
        force: If True, re-search even if category was already searched

    Returns:
        List of all venues for the city (from cache)
    """
    from .cache import (
        is_category_searched,
        append_venues_to_cache,
        read_cached_venues,
        load_seed_venues,
    )

    if categories is None:
        categories = _settings.VENUE_CATEGORIES_MVP

    # Check which categories need to be searched
    if force:
        to_search = categories
        skipped = []
    else:
        to_search = []
        skipped = []
        for cat in categories:
            if is_category_searched(city, cat):
                skipped.append(cat)
            else:
                to_search.append(cat)

    print(f"\n{'='*60}", flush=True)
    print(f"Discovering venues in {city}", flush=True)
    print(f"Categories to search: {len(to_search)}/{len(categories)}", flush=True)
    if skipped:
        print(f"Skipping (already cached): {len(skipped)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    errors = []
    new_venue_count = 0

    for i, category in enumerate(to_search, 1):
        print(f"[{i}/{len(to_search)}] Searching: {city} {category}...", flush=True)
        try:
            venues = search_venues_by_category(city, category)
            new_venue_count += len(venues)

            # Write to cache immediately
            append_venues_to_cache(venues, city, category)
            print(f"  -> Found {len(venues)} venues for '{category}'\n", flush=True)

        except Exception as e:
            error_msg = f"Error searching {category}: {e}"
            print(f"  -> {error_msg}\n", flush=True)
            errors.append(error_msg)

    # Load any seed venues that aren't already in cache
    seed_count = load_seed_venues(city)

    # Enrich any venues with bad addresses
    from .enrich_addresses import enrich_venues_in_cache
    enriched_count = enrich_venues_in_cache(city)

    # Read all venues from cache (includes previously cached + new + seeds)
    all_venues = read_cached_venues(city)

    print(f"\n{'='*60}", flush=True)
    print(f"Discovery complete!", flush=True)
    print(f"New venues found this run: {new_venue_count}", flush=True)
    if seed_count:
        print(f"Seed venues added: {seed_count}", flush=True)
    if enriched_count:
        print(f"Addresses enriched: {enriched_count}", flush=True)
    print(f"Total venues for {city}: {len(all_venues)}", flush=True)
    if errors:
        print(f"Errors: {len(errors)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return all_venues
