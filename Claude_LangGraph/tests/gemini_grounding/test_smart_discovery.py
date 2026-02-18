#!/usr/bin/env python3
"""
Smart venue discovery with Gemini Grounding.

Strategy:
1. Ask Gemini for estimated count
2. If count <= 50: Request full list directly
3. If count > 50: Ask Gemini to generate sub-queries that each return <50 results
4. For each sub-query, paginate if needed (first 20, next 20, etc.)
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

MAX_PER_QUERY = 50  # Assume Gemini can return at most 50 items per query


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
    # Look for patterns like "approximately 30" or "around 50" or "there are 75"
    patterns = [
        r'approximately\s+(\d+)',
        r'around\s+(\d+)',
        r'about\s+(\d+)',
        r'roughly\s+(\d+)',
        r'there\s+are\s+(\d+)',
        r'(\d+)\s+(?:bookstores|venues|clubs|theaters|museums)',
        r'\b(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return None


def get_estimated_count(category: str, city: str = "NYC") -> int:
    """Ask Gemini to estimate how many venues exist."""
    prompt = f"""How many {category} are there in {city}?
Give me your best estimate as a single number. Just respond with the number."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    count = extract_number(response.text)
    print(f"  Estimated count: {count} (raw: {response.text[:100]})")
    return count or 0


def get_subqueries(category: str, estimated_count: int, city: str = "NYC") -> list[str]:
    """Ask Gemini to generate sub-queries that each return <50 results."""
    num_subqueries = (estimated_count // 40) + 1  # Aim for ~40 per sub-query to be safe

    prompt = f"""I need to search for all {category} in {city}.
There are approximately {estimated_count} total.

Generate {num_subqueries} distinct search sub-queries that together would cover ALL {category} in {city}.
Each sub-query should target a different subset (by neighborhood, style, type, etc.)
and return roughly {estimated_count // num_subqueries} venues each.

Return as a JSON array of strings, e.g.: ["query 1", "query 2", "query 3"]
Return ONLY the JSON array."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    result = extract_json_from_response(response.text)
    if result and isinstance(result, list):
        return [str(q) for q in result]
    return [category]  # Fallback to original category


def get_venues_full_list(category: str, expected_count: int, city: str = "NYC") -> list:
    """Request the full list when count <= 50."""
    prompt = f"""Search for ALL {category} in {city}.
There are approximately {expected_count} total. Please find all of them.

Return a JSON array with:
- name: Venue name
- address: Full address or "{city}"
- neighborhood: Neighborhood

Return ONLY the JSON array, no other text."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    return extract_json_from_response(response.text) or []


def get_venues_paginated(query: str, city: str = "NYC", page_size: int = 25) -> list:
    """Get venues using pagination."""
    all_venues = []
    seen_names = set()
    page = 1
    max_pages = 4  # Safety limit

    while page <= max_pages:
        if page == 1:
            prompt = f"""Search for {query} in {city}.
Return the first {page_size} venues as a JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""
        else:
            exclude_list = ", ".join(sorted(seen_names))
            prompt = f"""Search for MORE {query} in {city}.
EXCLUDE these (already found): {exclude_list}
Return {page_size} DIFFERENT venues as a JSON array.
Include: name, address, neighborhood.
Return ONLY JSON."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        venues = extract_json_from_response(response.text) or []
        new_count = 0
        for v in venues:
            name = v.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                all_venues.append(v)
                new_count += 1

        print(f"    Page {page}: {len(venues)} returned, {new_count} new")

        # Stop if we got few new results
        if new_count < 5:
            break

        page += 1

    return all_venues


def discover_venues_smart(category: str, city: str = "NYC") -> list:
    """Smart venue discovery with automatic strategy selection."""
    print(f"\n{'='*60}")
    print(f"SMART DISCOVERY: {category}")
    print(f"{'='*60}")

    # Step 1: Get estimated count
    print("\nStep 1: Estimating count...")
    estimated = get_estimated_count(category, city)

    if estimated == 0:
        print("  Could not estimate count, using default strategy")
        estimated = 30

    # Step 2: Choose strategy based on count
    if estimated <= MAX_PER_QUERY:
        print(f"\nStep 2: Count ({estimated}) <= {MAX_PER_QUERY}, requesting full list...")
        venues = get_venues_full_list(category, estimated, city)
        print(f"  Got {len(venues)} venues")

    else:
        print(f"\nStep 2: Count ({estimated}) > {MAX_PER_QUERY}, generating sub-queries...")
        subqueries = get_subqueries(category, estimated, city)
        print(f"  Generated {len(subqueries)} sub-queries:")
        for sq in subqueries:
            print(f"    - {sq}")

        # Step 3: Execute each sub-query with pagination
        print("\nStep 3: Executing sub-queries with pagination...")
        all_venues = []
        seen_names = set()

        for sq in subqueries:
            print(f"\n  Sub-query: '{sq}'")
            venues = get_venues_paginated(sq, city, page_size=25)

            new_count = 0
            for v in venues:
                name = v.get('name', '').lower().strip()
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_venues.append(v)
                    new_count += 1

            print(f"    Total from this sub-query: {len(venues)}, new unique: {new_count}")

        venues = all_venues

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULT: {len(venues)} unique venues found")
    print(f"{'='*60}")

    return venues


def test_bookstores():
    """Test with bookstores."""
    venues = discover_venues_smart("bookstores that host author events and readings", "NYC")

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


def test_jazz_clubs():
    """Test with jazz clubs (should be >50)."""
    venues = discover_venues_smart("jazz clubs and jazz venues", "NYC")

    print("\nVenues found:")
    for v in sorted(venues, key=lambda x: x.get('name', ''))[:30]:
        print(f"  - {v.get('name')}")
    if len(venues) > 30:
        print(f"  ... and {len(venues) - 30} more")

    return venues


def test_comedy_clubs():
    """Test with comedy clubs."""
    venues = discover_venues_smart("comedy clubs and comedy venues", "NYC")

    print("\nVenues found:")
    for v in sorted(venues, key=lambda x: x.get('name', '')):
        print(f"  - {v.get('name')}")

    return venues


if __name__ == "__main__":
    print("SMART VENUE DISCOVERY TEST")
    print("=" * 60)

    # Test 1: Bookstores (likely <50)
    bookstores = test_bookstores()

    # Test 2: Jazz clubs (likely >50, needs sub-queries)
    jazz = test_jazz_clubs()

    # Test 3: Comedy clubs
    comedy = test_comedy_clubs()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Bookstores: {len(bookstores)} venues")
    print(f"Jazz clubs: {len(jazz)} venues")
    print(f"Comedy clubs: {len(comedy)} venues")
