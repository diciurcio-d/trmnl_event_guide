#!/usr/bin/env python3
"""
Test comprehensive venue discovery with Gemini Grounding.

Approach 1: Ask for count first, then request full list
Approach 2: Use multiple specific sub-queries
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
    # Look for patterns like "41 Broadway theaters" or "there are 41"
    match = re.search(r'(\d+)\s*(broadway|theater|venue|club|bookstore)', text.lower())
    if match:
        return int(match.group(1))
    # Just find any number
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return int(match.group(1))
    return None


def ask_count(category: str, city: str = "NYC") -> int | None:
    """Ask Gemini how many venues exist in a category."""
    prompt = f"How many {category} are there in {city}? Give me just the number."

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    return extract_number(response.text)


def get_venues_with_target(category: str, target_count: int, city: str = "NYC") -> list:
    """Get venues, telling Gemini how many we expect."""
    prompt = f"""Search for ALL {category} in {city}.

There are approximately {target_count} {category} in {city}.
Please find and list ALL of them.

Return a JSON array with:
- name: Venue name
- address: Full address or "{city}"
- neighborhood: Neighborhood in {city}

Return ONLY the JSON array, no other text."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    return extract_json_from_response(response.text) or []


def get_venues_paginated(category: str, city: str = "NYC", page_size: int = 25) -> list:
    """Get venues in multiple requests to get more results."""
    all_venues = []
    seen_names = set()

    # First batch
    prompt = f"""Search for {category} in {city}.
Return the first {page_size} venues as a JSON array.
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
    for v in venues:
        name = v.get('name', '').lower().strip()
        if name and name not in seen_names:
            seen_names.add(name)
            all_venues.append(v)

    print(f"  Batch 1: {len(venues)} venues")

    if len(venues) >= page_size - 5:  # If we got close to page_size, try another batch
        exclude_list = ", ".join(sorted(seen_names)[:20])
        prompt = f"""Search for MORE {category} in {city}.
EXCLUDE these venues (already found): {exclude_list}
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

        print(f"  Batch 2: {len(venues)} venues ({new_count} new)")

    return all_venues


def get_venues_with_subqueries(category: str, subqueries: list[str], city: str = "NYC") -> list:
    """Use multiple specific sub-queries to get comprehensive results."""
    all_venues = []
    seen_names = set()

    for query in subqueries:
        prompt = f"""Search for {query} in {city}.
Return as JSON array with: name, address, neighborhood.
Return ONLY JSON, no other text."""

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

        print(f"  '{query}': {len(venues)} found, {new_count} new")

    return all_venues


def test_broadway_theaters():
    """Test comprehensive Broadway theater discovery."""
    print("=" * 60)
    print("TEST: Broadway Theaters - Count First Approach")
    print("=" * 60)

    # Step 1: Ask how many
    print("\nStep 1: Asking for count...")
    count = ask_count("Broadway theaters")
    print(f"Gemini says there are {count} Broadway theaters")

    # Step 2: Get full list with target
    print(f"\nStep 2: Requesting all {count} theaters...")
    venues = get_venues_with_target("Broadway theaters", count or 41)
    print(f"Got {len(venues)} venues")

    for v in sorted(venues, key=lambda x: x.get('name', '')):
        print(f"  - {v.get('name')}")

    return venues


def test_broadway_paginated():
    """Test paginated approach."""
    print("\n" + "=" * 60)
    print("TEST: Broadway Theaters - Paginated Approach")
    print("=" * 60)

    venues = get_venues_paginated("Broadway theaters", page_size=25)
    print(f"\nTotal: {len(venues)} unique venues")

    return venues


def test_bookstores_subqueries():
    """Test sub-query approach for bookstores."""
    print("\n" + "=" * 60)
    print("TEST: Bookstores - Sub-query Approach")
    print("=" * 60)

    subqueries = [
        "independent bookstores with author events",
        "bookstores that host readings and signings",
        "literary bookstores and book shops",
        "used bookstores with events",
        "specialty bookstores (mystery, romance, children's)",
    ]

    venues = get_venues_with_subqueries("bookstores", subqueries)
    print(f"\nTotal unique bookstores: {len(venues)}")

    for v in sorted(venues, key=lambda x: x.get('name', '')):
        print(f"  - {v.get('name')}")

    return venues


def test_comedy_clubs_subqueries():
    """Test sub-query approach for comedy clubs."""
    print("\n" + "=" * 60)
    print("TEST: Comedy Clubs - Sub-query Approach")
    print("=" * 60)

    subqueries = [
        "stand-up comedy clubs",
        "improv comedy theaters",
        "comedy venues in Manhattan",
        "comedy clubs in Brooklyn",
        "open mic comedy nights venues",
    ]

    venues = get_venues_with_subqueries("comedy venues", subqueries)
    print(f"\nTotal unique comedy venues: {len(venues)}")

    return venues


if __name__ == "__main__":
    # Test 1: Count first approach
    broadway = test_broadway_theaters()

    # Test 2: Paginated approach
    broadway_paged = test_broadway_paginated()

    # Test 3: Sub-queries for bookstores
    bookstores = test_bookstores_subqueries()

    # Test 4: Sub-queries for comedy
    comedy = test_comedy_clubs_subqueries()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Broadway (count-first): {len(broadway)} venues")
    print(f"Broadway (paginated): {len(broadway_paged)} venues")
    print(f"Bookstores (sub-queries): {len(bookstores)} venues")
    print(f"Comedy (sub-queries): {len(comedy)} venues")
