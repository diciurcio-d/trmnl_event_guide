#!/usr/bin/env python3
"""
Smart venue discovery V2 - test with larger categories and better prompts.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from google.genai import types

config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
with open(config_path) as f:
    config = json.load(f)

client = genai.Client(api_key=config["gemini"]["api_key"])

MAX_PER_QUERY = 50


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
    """Extract a number from text."""
    patterns = [
        r'approximately\s+(\d+)',
        r'around\s+(\d+)',
        r'about\s+(\d+)',
        r'roughly\s+(\d+)',
        r'over\s+(\d+)',
        r'more\s+than\s+(\d+)',
        r'(\d+)\s*[-–]\s*(\d+)',  # Range like "100-150"
        r'there\s+are\s+(\d+)',
        r'\b(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            # If it's a range, take the higher number
            if match.lastindex == 2:
                return int(match.group(2))
            return int(match.group(1))
    return None


def get_estimated_count_v2(category: str, city: str = "NYC") -> int:
    """Ask Gemini to estimate count with better prompting."""
    prompt = f"""How many {category} exist in {city}?
Include all types - well-known AND lesser-known ones.
Give me your best estimate. Respond with just the number."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    count = extract_number(response.text)
    print(f"  Estimated count: {count}")
    print(f"  Raw response: {response.text[:200]}")
    return count or 0


def get_subqueries_v2(category: str, estimated_count: int, city: str = "NYC") -> list[str]:
    """Ask Gemini to generate sub-queries that partition the category."""
    num_subqueries = max(2, (estimated_count // 35) + 1)  # Aim for ~35 per sub-query

    prompt = f"""I need to find ALL {category} in {city} (approximately {estimated_count} total).

Generate {num_subqueries} search queries that together would find ALL of them.
Each query should target a DIFFERENT subset with NO overlap.
Consider dividing by: neighborhood, style, size, specialty, etc.

Return as a JSON array of search query strings.
Example: ["large art galleries in Chelsea NYC", "small artist-run galleries Brooklyn NYC"]
Return ONLY the JSON array."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    result = extract_json_from_response(response.text)
    if result and isinstance(result, list):
        print(f"  Generated sub-queries:")
        for q in result:
            print(f"    - {q}")
        return [str(q) for q in result]
    return [f"{category} in {city}"]


def get_venues_batch(query: str, exclude_names: set = None, city: str = "NYC") -> list:
    """Get a batch of venues, optionally excluding already-found ones."""
    exclude_names = exclude_names or set()

    if exclude_names:
        exclude_str = ", ".join(sorted(exclude_names)[:30])  # Limit to avoid too-long prompt
        prompt = f"""Search for {query}.
EXCLUDE these venues (already found): {exclude_str}
Return up to 30 DIFFERENT venues as a JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""
    else:
        prompt = f"""Search for {query}.
Return up to 40 venues as a JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    return extract_json_from_response(response.text) or []


def discover_venues_v2(category: str, city: str = "NYC") -> list:
    """Smart venue discovery V2."""
    print(f"\n{'='*60}")
    print(f"DISCOVERING: {category}")
    print(f"{'='*60}")

    # Step 1: Get estimated count
    print("\nStep 1: Estimating count...")
    estimated = get_estimated_count_v2(category, city)

    if estimated == 0:
        estimated = 30

    all_venues = []
    seen_names = set()

    # Step 2: Strategy based on count
    if estimated <= MAX_PER_QUERY:
        print(f"\nStep 2: Count ({estimated}) <= {MAX_PER_QUERY}, direct query...")
        venues = get_venues_batch(f"all {category} in {city}")
        for v in venues:
            name = v.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                all_venues.append(v)
        print(f"  Found {len(all_venues)} venues")

        # If we got close to the estimate, try one more batch
        if len(all_venues) >= estimated - 5:
            print("  Trying for more...")
            more = get_venues_batch(f"more {category} in {city}", seen_names)
            new_count = 0
            for v in more:
                name = v.get('name', '').lower().strip()
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_venues.append(v)
                    new_count += 1
            print(f"  Found {new_count} additional venues")

    else:
        print(f"\nStep 2: Count ({estimated}) > {MAX_PER_QUERY}, using sub-queries...")
        subqueries = get_subqueries_v2(category, estimated, city)

        for sq in subqueries:
            print(f"\n  Querying: '{sq[:50]}...'")

            # First batch
            venues = get_venues_batch(sq)
            new_count = 0
            for v in venues:
                name = v.get('name', '').lower().strip()
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_venues.append(v)
                    new_count += 1
            print(f"    Batch 1: {len(venues)} returned, {new_count} new")

            # Second batch if we got many
            if new_count >= 20:
                venues = get_venues_batch(sq, seen_names)
                new_count = 0
                for v in venues:
                    name = v.get('name', '').lower().strip()
                    if name and name not in seen_names:
                        seen_names.add(name)
                        all_venues.append(v)
                        new_count += 1
                print(f"    Batch 2: {len(venues)} returned, {new_count} new")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_venues)} unique venues")
    print(f"{'='*60}")

    return all_venues


def test_art_galleries():
    """Test with art galleries (should be >100)."""
    venues = discover_venues_v2("art galleries", "NYC")

    print("\nSample venues:")
    for v in sorted(venues, key=lambda x: x.get('name', ''))[:20]:
        print(f"  - {v.get('name')} ({v.get('neighborhood', '')})")
    if len(venues) > 20:
        print(f"  ... and {len(venues) - 20} more")

    return venues


def test_bookstores_broader():
    """Test bookstores with broader query."""
    venues = discover_venues_v2("independent bookstores", "NYC")

    print("\nVenues found:")
    for v in sorted(venues, key=lambda x: x.get('name', '')):
        print(f"  - {v.get('name')}")

    # Check for Yu and Me
    names_lower = [v.get('name', '').lower() for v in venues]
    if any('yu and me' in n or 'yu & me' in n for n in names_lower):
        print("\n✓ Found Yu and Me Books!")
    else:
        print("\n✗ Yu and Me Books not found")

    return venues


def test_music_venues():
    """Test with live music venues."""
    venues = discover_venues_v2("live music venues and concert halls", "NYC")

    print("\nSample venues:")
    for v in sorted(venues, key=lambda x: x.get('name', ''))[:25]:
        print(f"  - {v.get('name')}")
    if len(venues) > 25:
        print(f"  ... and {len(venues) - 25} more")

    return venues


if __name__ == "__main__":
    # Test 1: Art galleries (definitely >50)
    galleries = test_art_galleries()

    # Test 2: Bookstores (broader search)
    bookstores = test_bookstores_broader()

    # Test 3: Music venues
    music = test_music_venues()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Art galleries: {len(galleries)} venues")
    print(f"Bookstores: {len(bookstores)} venues")
    print(f"Music venues: {len(music)} venues")
