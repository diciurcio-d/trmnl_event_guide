"""
Venue discovery using Gemini with Google Search Grounding.

Two-phase prioritized approach:
1. Phase 1: Get the most notable/important venues first
2. Phase 2: Fill remaining with comprehensive sub-query discovery

Replaces the Jina-based discovery with faster, more reliable Gemini Grounding.
"""

import importlib.util
import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from .state import Venue


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


def _load_config():
    """Load API config."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path) as f:
        return json.load(f)


_settings = _load_settings()
_config = _load_config()

# Initialize Gemini client
_client = genai.Client(api_key=_config["gemini"]["api_key"])

# Models
GROUNDING_MODEL = "gemini-2.5-pro"  # For web search grounding (5 RPM, 1500/day free)
FAST_MODEL = "gemini-3-flash-preview"  # For non-grounding tasks

# Rate limiting
GROUNDING_RPM = 5
GROUNDING_DELAY = 60 / GROUNDING_RPM + 1  # ~13 seconds between grounding calls
_last_grounding_call = 0

# Limits
MAX_VENUES_PER_CATEGORY = 250


def _rate_limit_grounding():
    """Enforce rate limiting for grounding calls."""
    global _last_grounding_call
    now = time.time()
    elapsed = now - _last_grounding_call
    if elapsed < GROUNDING_DELAY:
        wait = GROUNDING_DELAY - elapsed
        print(f"    [rate limit: waiting {wait:.1f}s]", flush=True)
        time.sleep(wait)
    _last_grounding_call = time.time()


def _extract_json_from_response(text: str) -> list | None:
    """Extract JSON array from response text."""
    # Try markdown code block first
    json_match = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON array
    array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    # Try whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _extract_number(text: str) -> int | None:
    """Extract a number from text, handling commas in large numbers."""
    text_normalized = re.sub(r'(\d),(\d)', r'\1\2', text)

    patterns = [
        r'approximately\s+([\d,]+)',
        r'around\s+([\d,]+)',
        r'about\s+([\d,]+)',
        r'roughly\s+([\d,]+)',
        r'over\s+([\d,]+)',
        r'more\s+than\s+([\d,]+)',
        r'estimated\s+([\d,]+)',
        r'([\d,]+)\s*[-–]\s*([\d,]+)',  # Range
        r'there\s+are\s+([\d,]+)',
        r'([\d,]+)\s+(?:art\s+)?(?:galleries|venues|clubs|theaters|bookstores|museums)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_normalized.lower())
        if match:
            if match.lastindex == 2:
                return int(match.group(2).replace(',', ''))
            return int(match.group(1).replace(',', ''))

    # Fallback: find largest number
    numbers = re.findall(r'\b(\d{2,})\b', text_normalized)
    if numbers:
        return max(int(n) for n in numbers)

    return None


def _get_estimated_count(category: str, city: str) -> int:
    """Ask Gemini to estimate how many venues exist."""
    _rate_limit_grounding()

    prompt = f"How many {category} exist in {city}? Include all types. Respond with just the number."

    try:
        response = _client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        count = _extract_number(response.text)
        print(f"  Estimated: {count}", flush=True)
        return count or 30
    except Exception as e:
        print(f"  Error estimating count: {e}", flush=True)
        return 30


def _get_most_important_venues(category: str, city: str, limit: int = 75) -> list[dict]:
    """Phase 1: Get the most notable/important venues first."""
    _rate_limit_grounding()

    prompt = f"""Search for the most important, notable, and well-known {category} in {city}.
Focus on famous, established, landmark venues frequently mentioned in "best of" lists.
Return up to {limit} venues as JSON array ordered by importance.
Include: name, address, neighborhood.
Return ONLY JSON."""

    try:
        response = _client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return _extract_json_from_response(response.text) or []
    except Exception as e:
        print(f"    Error: {e}", flush=True)
        return []


def _get_subqueries(category: str, estimated_count: int, city: str) -> list[str]:
    """Ask Gemini (fast model, no grounding) to generate sub-queries."""
    num_subqueries = max(3, min(15, (estimated_count // 40) + 1))

    prompt = f"""I need to find ALL {category} in {city} (approximately {estimated_count} total).
I already have the most famous ones.
Generate {num_subqueries} search queries to find the REMAINING venues.
Divide by: neighborhood, specialty, size, type.
Return as JSON array of search query strings.
Return ONLY the JSON array."""

    try:
        response = _client.models.generate_content(model=FAST_MODEL, contents=prompt)
        result = _extract_json_from_response(response.text)
        if result and isinstance(result, list):
            return [str(q) for q in result]
    except Exception as e:
        print(f"    Error generating sub-queries: {e}", flush=True)

    return [f"{category} in {city}"]


def _get_venues_with_grounding(query: str, exclude_names: set = None, city: str = "NYC") -> list[dict]:
    """Get venues using grounding, with rate limiting."""
    _rate_limit_grounding()
    exclude_names = exclude_names or set()

    if exclude_names:
        exclude_str = ", ".join(sorted(exclude_names)[:25])
        prompt = f"""Search for {query}.
EXCLUDE these: {exclude_str}
Return up to 40 DIFFERENT venues as JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""
    else:
        prompt = f"""Search for {query}.
Return up to 40 venues as JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""

    try:
        response = _client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return _extract_json_from_response(response.text) or []
    except Exception as e:
        print(f"    Error: {e}", flush=True)
        return []


def discover_category(category: str, city: str = "NYC") -> list[Venue]:
    """
    Discover venues for a single category using prioritized approach.

    Returns list of Venue dicts with category populated.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"DISCOVERING: {category}", flush=True)
    print(f"{'='*60}", flush=True)

    all_venues: list[Venue] = []
    seen_names: set[str] = set()

    def add_venues(raw_venues: list[dict]) -> int:
        """Add venues, dedupe, return count of new ones."""
        new_count = 0
        for v in raw_venues:
            if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
                return new_count
            name = v.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                venue = Venue(
                    name=v.get('name', '')[:100],
                    address=v.get('address', city),
                    city=city,
                    neighborhood=v.get('neighborhood', ''),
                    website='',
                    events_url='',
                    category=category,
                    description='',
                    source=f'gemini_grounding:{category}',
                    address_verified='',
                    website_status='',
                    website_attempts=0,
                    preferred_event_source='',
                    api_endpoint='',
                    ticketmaster_venue_id='',
                    last_event_fetch='',
                    event_count=0,
                    event_source='',
                )
                all_venues.append(venue)
                new_count += 1
        return new_count

    # Phase 1: Most important venues
    print("\n--- PHASE 1: Most Important Venues ---", flush=True)
    important = _get_most_important_venues(category, city, limit=75)
    new_count = add_venues(important)
    print(f"  Found {new_count} notable venues", flush=True)

    # Get more notable if we got a good batch
    if len(all_venues) >= 50 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
        print("  Getting more notable venues...", flush=True)
        more = _get_venues_with_grounding(f"more notable {category}", seen_names, city)
        new_count = add_venues(more)
        print(f"  Found {new_count} more", flush=True)

    # Phase 2: Comprehensive discovery if needed
    if len(all_venues) < MAX_VENUES_PER_CATEGORY:
        print("\n--- PHASE 2: Comprehensive Discovery ---", flush=True)
        estimated = _get_estimated_count(category, city)

        if estimated > len(all_venues):
            subqueries = _get_subqueries(category, estimated, city)
            print(f"  Generated {len(subqueries)} sub-queries", flush=True)

            for i, sq in enumerate(subqueries, 1):
                if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
                    print(f"\n  Reached {MAX_VENUES_PER_CATEGORY} cap.", flush=True)
                    break

                print(f"\n  [{i}/{len(subqueries)}] '{sq[:45]}...'", flush=True)
                venues = _get_venues_with_grounding(sq, seen_names, city)
                new_count = add_venues(venues)
                print(f"    Found {new_count} new (total: {len(all_venues)})", flush=True)

                # Second batch if we got many
                if new_count >= 20 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
                    venues = _get_venues_with_grounding(sq, seen_names, city)
                    new_count = add_venues(venues)
                    print(f"    Batch 2: {new_count} new (total: {len(all_venues)})", flush=True)

    print(f"\nCATEGORY TOTAL: {len(all_venues)} venues", flush=True)
    return all_venues


def discover_venues(
    city: str,
    categories: list[str] | None = None,
    force: bool = False,
) -> list[Venue]:
    """
    Discover venues in a city across specified categories.

    Uses Gemini with Google Search Grounding for fast, reliable discovery.
    Prioritizes notable venues first, then fills with comprehensive discovery.

    Args:
        city: City to search (e.g., "NYC")
        categories: List of categories to search (defaults to VENUE_CATEGORIES_ALL)
        force: If True, re-search even if category was already searched

    Returns:
        List of all discovered venues
    """
    from .cache import (
        is_category_searched,
        append_venues_to_cache,
        read_cached_venues,
        load_seed_venues,
    )

    if categories is None:
        categories = _settings.VENUE_CATEGORIES_ALL

    # Check which categories need to be searched
    if force:
        to_search = categories
    else:
        to_search = [cat for cat in categories if not is_category_searched(city, cat)]

    print(f"\n{'='*60}", flush=True)
    print(f"Gemini Grounding Venue Discovery - {city}", flush=True)
    print(f"Categories to search: {len(to_search)}/{len(categories)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_new_venues = []
    errors = []

    for i, category in enumerate(to_search, 1):
        print(f"\n[{i}/{len(to_search)}] Category: {category}", flush=True)
        try:
            venues = discover_category(category, city)
            all_new_venues.extend(venues)

            # Write to cache immediately
            append_venues_to_cache(venues, city, category)
            print(f"  -> Cached {len(venues)} venues", flush=True)

        except Exception as e:
            error_msg = f"Error discovering {category}: {e}"
            print(f"  -> {error_msg}", flush=True)
            errors.append(error_msg)

    # Load seed venues
    seed_count = load_seed_venues(city)

    # Read all cached venues
    all_venues = read_cached_venues(city)

    print(f"\n{'='*60}", flush=True)
    print(f"Discovery complete!", flush=True)
    print(f"New venues this run: {len(all_new_venues)}", flush=True)
    if seed_count:
        print(f"Seed venues added: {seed_count}", flush=True)
    print(f"Total venues for {city}: {len(all_venues)}", flush=True)
    if errors:
        print(f"Errors: {len(errors)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return all_venues
