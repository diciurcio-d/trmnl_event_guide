#!/usr/bin/env python3
"""
Prioritized venue discovery - get most important venues first.

Two-phase approach:
1. Phase 1: Get the most notable/important venues (top 50-100)
2. Phase 2: Fill remaining cap with comprehensive sub-queries
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from google.genai import types

config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
with open(config_path) as f:
    config = json.load(f)

client = genai.Client(api_key=config["gemini"]["api_key"])

GROUNDING_MODEL = "gemini-2.5-pro"
FAST_MODEL = "gemini-3-flash-preview"

MAX_VENUES_PER_CATEGORY = 250
GROUNDING_RPM = 5
GROUNDING_DELAY = 60 / GROUNDING_RPM + 1

_last_grounding_call = 0


def rate_limit_grounding():
    global _last_grounding_call
    now = time.time()
    elapsed = now - _last_grounding_call
    if elapsed < GROUNDING_DELAY:
        wait = GROUNDING_DELAY - elapsed
        print(f"    [rate limit: waiting {wait:.1f}s]", flush=True)
        time.sleep(wait)
    _last_grounding_call = time.time()


def extract_json_from_response(text: str) -> list | None:
    json_match = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def extract_number(text: str) -> int | None:
    text_normalized = re.sub(r'(\d),(\d)', r'\1\2', text)

    patterns = [
        r'approximately\s+([\d,]+)',
        r'around\s+([\d,]+)',
        r'about\s+([\d,]+)',
        r'roughly\s+([\d,]+)',
        r'over\s+([\d,]+)',
        r'more\s+than\s+([\d,]+)',
        r'estimated\s+([\d,]+)',
        r'([\d,]+)\s*[-–]\s*([\d,]+)',
        r'there\s+are\s+([\d,]+)',
        r'([\d,]+)\s+(?:art\s+)?(?:galleries|venues|clubs|theaters|bookstores|museums)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_normalized.lower())
        if match:
            if match.lastindex == 2:
                return int(match.group(2).replace(',', ''))
            return int(match.group(1).replace(',', ''))

    numbers = re.findall(r'\b(\d{2,})\b', text_normalized)
    if numbers:
        return max(int(n) for n in numbers)

    return None


def get_estimated_count(category: str, city: str = "NYC") -> int:
    rate_limit_grounding()

    prompt = f"""How many {category} exist in {city}?
Include all types. Respond with just the number."""

    response = client.models.generate_content(
        model=GROUNDING_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    count = extract_number(response.text)
    print(f"  Estimated: {count}")
    return count or 30


def get_most_important_venues(category: str, city: str = "NYC", limit: int = 75) -> list:
    """Phase 1: Get the most notable/important venues first."""
    rate_limit_grounding()

    prompt = f"""Search for the most important, notable, and well-known {category} in {city}.

Focus on:
- The most famous and established venues
- Venues that are considered landmarks or institutions
- Venues frequently mentioned in "best of" lists
- Venues with the highest reputation and recognition

Return up to {limit} venues as a JSON array, ordered by importance/notability.
Include: name, address, neighborhood.
Return ONLY JSON."""

    try:
        response = client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return extract_json_from_response(response.text) or []
    except Exception as e:
        print(f"    Error: {e}")
        return []


def get_subqueries(category: str, estimated_count: int, city: str = "NYC") -> list[str]:
    """Generate sub-queries for comprehensive coverage."""
    num_subqueries = max(3, min(20, (estimated_count // 40) + 1))

    prompt = f"""I need to find ALL {category} in {city} (approximately {estimated_count} total).
I already have the most famous/notable ones.

Generate {num_subqueries} search queries to find the REMAINING venues -
the lesser-known, neighborhood, emerging, or specialty ones.

Divide by: neighborhood, specialty, size, type, etc.
Each query should target a DIFFERENT subset.

Return as a JSON array of search query strings.
Return ONLY the JSON array."""

    response = client.models.generate_content(
        model=FAST_MODEL,
        contents=prompt,
    )

    result = extract_json_from_response(response.text)
    if result and isinstance(result, list):
        return [str(q) for q in result]
    return [f"{category} in {city}"]


def get_venues_with_grounding(query: str, exclude_names: set = None, city: str = "NYC") -> list:
    rate_limit_grounding()
    exclude_names = exclude_names or set()

    if exclude_names:
        exclude_str = ", ".join(sorted(exclude_names)[:30])
        prompt = f"""Search for {query}.
EXCLUDE these (already found): {exclude_str}
Return up to 40 DIFFERENT venues as JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""
    else:
        prompt = f"""Search for {query}.
Return up to 40 venues as JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""

    try:
        response = client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return extract_json_from_response(response.text) or []
    except Exception as e:
        print(f"    Error: {e}")
        return []


def discover_venues_prioritized(category: str, city: str = "NYC") -> list:
    """Prioritized venue discovery - important venues first."""
    print(f"\n{'='*60}")
    print(f"DISCOVERING: {category}")
    print(f"{'='*60}")

    all_venues = []
    seen_names = set()

    def add_venues(venues: list) -> int:
        new_count = 0
        for v in venues:
            if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
                return new_count
            name = v.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                all_venues.append(v)
                new_count += 1
        return new_count

    # Phase 1: Get most important venues first
    print("\n--- PHASE 1: Most Important Venues ---")
    important = get_most_important_venues(category, city, limit=75)
    new_count = add_venues(important)
    print(f"  Found {new_count} notable venues")

    # If we already hit the cap or have enough, try one more batch
    if len(all_venues) >= 50 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
        print("  Getting more notable venues...")
        more_important = get_venues_with_grounding(
            f"more notable and important {category}", seen_names, city
        )
        new_count = add_venues(more_important)
        print(f"  Found {new_count} more notable venues")

    if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
        print(f"\n  Reached cap with notable venues only.")
    else:
        # Phase 2: Get estimated count and fill with comprehensive discovery
        print("\n--- PHASE 2: Comprehensive Discovery ---")
        estimated = get_estimated_count(category, city)

        if estimated > len(all_venues) and len(all_venues) < MAX_VENUES_PER_CATEGORY:
            print(f"  Generating sub-queries for remaining venues...")
            subqueries = get_subqueries(category, estimated, city)
            print(f"  Generated {len(subqueries)} sub-queries")

            for i, sq in enumerate(subqueries, 1):
                if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
                    print(f"\n  Reached {MAX_VENUES_PER_CATEGORY} venue cap.")
                    break

                print(f"\n  [{i}/{len(subqueries)}] '{sq[:50]}...'")

                venues = get_venues_with_grounding(sq, seen_names, city)
                new_count = add_venues(venues)
                print(f"    Found {new_count} new (total: {len(all_venues)})")

                # Second batch if we got many
                if new_count >= 20 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
                    venues = get_venues_with_grounding(sq, seen_names, city)
                    new_count = add_venues(venues)
                    print(f"    Batch 2: {new_count} new (total: {len(all_venues)})")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_venues)} unique venues")
    print(f"{'='*60}")

    return all_venues


def test_art_galleries():
    """Test with art galleries."""
    venues = discover_venues_prioritized("art galleries", "NYC")

    print("\n--- FIRST 30 VENUES (should be most important) ---")
    for i, v in enumerate(venues[:30], 1):
        print(f"  {i}. {v.get('name')} ({v.get('neighborhood', '')})")

    print(f"\n--- LAST 10 VENUES (filled in later) ---")
    for i, v in enumerate(venues[-10:], len(venues) - 9):
        print(f"  {i}. {v.get('name')} ({v.get('neighborhood', '')})")

    return venues


def test_bookstores():
    """Test with bookstores."""
    venues = discover_venues_prioritized("independent bookstores that host events", "NYC")

    print("\n--- VENUES (ordered by importance) ---")
    for i, v in enumerate(venues[:20], 1):
        print(f"  {i}. {v.get('name')} ({v.get('neighborhood', '')})")
    if len(venues) > 20:
        print(f"  ... and {len(venues) - 20} more")

    # Check for notable bookstores at top
    top_10_names = [v.get('name', '').lower() for v in venues[:10]]
    notable = ['strand', 'mcnally jackson', 'books are magic', 'housing works']
    found_notable = [n for n in notable if any(n in name for name in top_10_names)]
    print(f"\nNotable bookstores in top 10: {found_notable}")

    return venues


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "galleries":
            test_art_galleries()
        elif test == "bookstores":
            test_bookstores()
        else:
            print(f"Usage: python test_prioritized.py [galleries|bookstores]")
    else:
        # Default: test bookstores (faster)
        test_bookstores()
