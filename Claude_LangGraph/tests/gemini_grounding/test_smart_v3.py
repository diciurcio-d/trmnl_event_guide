#!/usr/bin/env python3
"""
Smart venue discovery V3
- Gemini 2.5 Pro for web grounding (1500/day free)
- Gemini 3 Flash Preview for non-grounding tasks
- Rate limiting (respect RPM limits)
- Cap venues per category at 250
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

# Models
GROUNDING_MODEL = "gemini-2.5-pro"  # For web search grounding (1500/day free)
FAST_MODEL = "gemini-3-flash-preview"  # For non-grounding tasks

# Limits
MAX_VENUES_PER_CATEGORY = 250
GROUNDING_RPM = 5  # Gemini 2.5 Pro limit
GROUNDING_DELAY = 60 / GROUNDING_RPM + 1  # ~13 seconds between grounding calls

_last_grounding_call = 0


def rate_limit_grounding():
    """Enforce rate limiting for grounding calls."""
    global _last_grounding_call
    now = time.time()
    elapsed = now - _last_grounding_call
    if elapsed < GROUNDING_DELAY:
        wait = GROUNDING_DELAY - elapsed
        print(f"    [rate limit: waiting {wait:.1f}s]", flush=True)
        time.sleep(wait)
    _last_grounding_call = time.time()


def extract_json_from_response(text: str) -> list | None:
    """Extract JSON array from response text."""
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
    """Extract a number from text, handling commas in large numbers."""
    # First, normalize text - remove commas from numbers
    text_normalized = re.sub(r'(\d),(\d)', r'\1\2', text)

    patterns = [
        r'approximately\s+([\d,]+)',
        r'around\s+([\d,]+)',
        r'about\s+([\d,]+)',
        r'roughly\s+([\d,]+)',
        r'over\s+([\d,]+)',
        r'more\s+than\s+([\d,]+)',
        r'estimated\s+([\d,]+)',
        r'([\d,]+)\s*[-–]\s*([\d,]+)',  # Range like "100-150"
        r'there\s+are\s+([\d,]+)',
        r'([\d,]+)\s+(?:art\s+)?(?:galleries|venues|clubs|theaters|bookstores|museums)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_normalized.lower())
        if match:
            if match.lastindex == 2:
                return int(match.group(2).replace(',', ''))
            return int(match.group(1).replace(',', ''))

    # Fallback: find largest number in text
    numbers = re.findall(r'\b(\d{2,})\b', text_normalized)
    if numbers:
        return max(int(n) for n in numbers)

    return None


def get_estimated_count(category: str, city: str = "NYC") -> int:
    """Ask Gemini (with grounding) to estimate count."""
    rate_limit_grounding()

    prompt = f"""How many {category} exist in {city}?
Include all types - well-known AND lesser-known.
Respond with just the number."""

    response = client.models.generate_content(
        model=GROUNDING_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    count = extract_number(response.text)
    print(f"  Estimated: {count} ({response.text[:100].strip()})")
    return count or 30


def get_subqueries(category: str, estimated_count: int, city: str = "NYC") -> list[str]:
    """Ask Gemini (fast, no grounding) to generate sub-queries."""
    num_subqueries = max(2, (estimated_count // 35) + 1)

    prompt = f"""I need to find ALL {category} in {city} (approximately {estimated_count} total).

Generate {num_subqueries} search queries that together cover ALL of them.
Each query should target a DIFFERENT subset with minimal overlap.
Divide by: neighborhood, style, type, size, specialty, etc.

Return as a JSON array of search query strings.
Return ONLY the JSON array, no other text."""

    response = client.models.generate_content(
        model=FAST_MODEL,
        contents=prompt,
    )

    result = extract_json_from_response(response.text)
    if result and isinstance(result, list):
        return [str(q) for q in result]
    return [f"{category} in {city}"]


def get_venues_with_grounding(query: str, exclude_names: set = None, city: str = "NYC") -> list:
    """Get venues using grounding, with rate limiting."""
    rate_limit_grounding()

    exclude_names = exclude_names or set()

    if exclude_names:
        exclude_str = ", ".join(sorted(exclude_names)[:25])
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


def discover_venues(category: str, city: str = "NYC") -> list:
    """Smart venue discovery with rate limiting and 250 cap."""
    print(f"\n{'='*60}")
    print(f"DISCOVERING: {category}")
    print(f"{'='*60}")

    # Step 1: Get estimated count
    print("\nStep 1: Estimating count...")
    estimated = get_estimated_count(category, city)

    all_venues = []
    seen_names = set()

    def add_venues(venues: list) -> int:
        """Add venues, dedupe, return count of new ones."""
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

    # Step 2: Strategy based on count
    if estimated <= 50:
        print(f"\nStep 2: Count ({estimated}) <= 50, direct query...")
        venues = get_venues_with_grounding(f"all {category} in {city}")
        new_count = add_venues(venues)
        print(f"  Found {new_count} venues")

        # Try for more if we got close to estimate
        if new_count >= estimated - 5 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
            print("  Trying for more...")
            more = get_venues_with_grounding(f"more {category} in {city}", seen_names)
            new_count = add_venues(more)
            print(f"  Found {new_count} additional venues")

    else:
        print(f"\nStep 2: Count ({estimated}) > 50, generating sub-queries...")
        subqueries = get_subqueries(category, estimated, city)
        print(f"  Generated {len(subqueries)} sub-queries")

        print("\nStep 3: Executing sub-queries...")
        for i, sq in enumerate(subqueries, 1):
            if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
                print(f"  Reached {MAX_VENUES_PER_CATEGORY} venue cap, stopping.")
                break

            print(f"\n  [{i}/{len(subqueries)}] '{sq[:50]}...'")

            # First batch
            venues = get_venues_with_grounding(sq)
            new_count = add_venues(venues)
            print(f"    Batch 1: {len(venues)} returned, {new_count} new (total: {len(all_venues)})")

            # Second batch if we got many and still under cap
            if new_count >= 20 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
                venues = get_venues_with_grounding(sq, seen_names)
                new_count = add_venues(venues)
                print(f"    Batch 2: {len(venues)} returned, {new_count} new (total: {len(all_venues)})")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_venues)} unique venues (cap: {MAX_VENUES_PER_CATEGORY})")
    print(f"{'='*60}")

    return all_venues


def test_bookstores():
    """Test with bookstores."""
    venues = discover_venues("independent bookstores that host events", "NYC")

    print("\nVenues found:")
    for v in sorted(venues, key=lambda x: x.get('name', '')):
        print(f"  - {v.get('name')}")

    names_lower = [v.get('name', '').lower() for v in venues]
    if any('yu and me' in n or 'yu & me' in n for n in names_lower):
        print("\n✓ Found Yu and Me Books!")
    else:
        print("\n✗ Yu and Me Books not found")

    return venues


def test_art_galleries():
    """Test with art galleries (large category)."""
    venues = discover_venues("art galleries", "NYC")

    print("\nSample venues:")
    for v in sorted(venues, key=lambda x: x.get('name', ''))[:30]:
        print(f"  - {v.get('name')} ({v.get('neighborhood', '')})")
    if len(venues) > 30:
        print(f"  ... and {len(venues) - 30} more")

    return venues


def test_comedy_clubs():
    """Test with comedy clubs."""
    venues = discover_venues("comedy clubs and comedy venues", "NYC")

    print("\nVenues found:")
    for v in sorted(venues, key=lambda x: x.get('name', '')):
        print(f"  - {v.get('name')}")

    return venues


if __name__ == "__main__":
    import sys

    # Allow selecting which test to run
    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "bookstores":
            test_bookstores()
        elif test == "galleries":
            test_art_galleries()
        elif test == "comedy":
            test_comedy_clubs()
        else:
            print(f"Unknown test: {test}")
            print("Usage: python test_smart_v3.py [bookstores|galleries|comedy]")
    else:
        # Run all tests
        print("Running all tests (this will take a while due to rate limiting)...")

        bookstores = test_bookstores()
        comedy = test_comedy_clubs()
        # Skip galleries for now - it's very large

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Bookstores: {len(bookstores)} venues")
        print(f"Comedy clubs: {len(comedy)} venues")
