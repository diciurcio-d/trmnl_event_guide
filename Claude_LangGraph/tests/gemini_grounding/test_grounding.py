#!/usr/bin/env python3
"""
Test Gemini API with Google Search Grounding for venue discovery.

This tests whether we can replace Jina Search + Jina Reader + Gemini extraction
with a single Gemini Grounding call.
"""

import json
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from google.genai import types

# Load config
config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Initialize Gemini client
client = genai.Client(api_key=config["gemini"]["api_key"])


def test_basic_grounding():
    """Test basic grounded search."""
    print("=" * 60)
    print("TEST 1: Basic Grounded Search")
    print("=" * 60)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What are the top 5 comedy clubs in NYC?",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    print("\nResponse:")
    print(response.text)

    # Check grounding metadata
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            meta = candidate.grounding_metadata
            print("\n--- Grounding Metadata ---")
            if hasattr(meta, 'search_entry_point'):
                print(f"Search query used: {meta.search_entry_point}")
            if hasattr(meta, 'grounding_chunks') and meta.grounding_chunks:
                print(f"Sources used: {len(meta.grounding_chunks)}")
                for i, chunk in enumerate(meta.grounding_chunks[:3]):
                    if hasattr(chunk, 'web'):
                        print(f"  {i+1}. {chunk.web.title}: {chunk.web.uri}")

    return response


def test_venue_extraction_json():
    """Test extracting venues as structured JSON."""
    print("\n" + "=" * 60)
    print("TEST 2: Venue Extraction as JSON")
    print("=" * 60)

    prompt = """Search for comedy clubs in NYC and return a JSON array of venues.

For each venue, include:
- name: The venue name
- address: Full address if available
- neighborhood: NYC neighborhood (e.g., "Greenwich Village")
- website: Venue website URL if found
- description: Brief description (max 50 words)

Return ONLY valid JSON, no markdown or explanation. Example format:
[{"name": "Example Club", "address": "123 Main St", "neighborhood": "Midtown", "website": "https://example.com", "description": "A comedy venue"}]
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    print("\nRaw response:")
    print(response.text[:2000])

    # Try to parse as JSON
    try:
        # Clean up response (remove markdown if present)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        venues = json.loads(text)
        print(f"\n--- Parsed {len(venues)} venues ---")
        for v in venues[:5]:
            print(f"  - {v.get('name')}: {v.get('neighborhood', 'N/A')}")
        return venues
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON: {e}")
        return None


def test_specific_category(category: str, city: str = "NYC"):
    """Test venue extraction for a specific category."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Category Search - {category}")
    print("=" * 60)

    prompt = f"""Search for {category} in {city} and return a JSON array of venues.

For each venue found, include:
- name: The venue name (must be an actual business name)
- address: Full address if available, otherwise "{city}"
- neighborhood: Neighborhood within the city
- website: Direct venue website URL (not aggregator sites like Yelp/TripAdvisor)
- description: Brief description (max 50 words)
- venue_type: Primary type (e.g., "comedy_club", "music_venue", "theater")

IMPORTANT:
- Only include established venues with their own identity
- Do NOT include rental spaces, private event spaces, or generic descriptions
- Return at least 10 venues if possible
- Return ONLY valid JSON array, no markdown or explanation
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    # Parse response
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])  # Remove first and last lines

    try:
        venues = json.loads(text)
        print(f"\nFound {len(venues)} venues:")
        for v in venues:
            name = v.get('name', 'Unknown')
            hood = v.get('neighborhood', '')
            website = v.get('website', '')[:40] if v.get('website') else ''
            print(f"  - {name} ({hood}) {website}")
        return venues
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON: {e}")
        print(f"Raw text: {text[:500]}")
        return None


def test_comparison_with_current(category: str = "comedy clubs standup"):
    """Compare grounding results with current sheet data."""
    print("\n" + "=" * 60)
    print(f"TEST 4: Compare with Current Sheet - {category}")
    print("=" * 60)

    # Load current venues from sheet
    from venue_scout.cache import read_cached_venues
    current_venues = read_cached_venues()
    current_in_cat = [v for v in current_venues if v.get('source') == f'web_search:{category}']
    current_names = {v['name'].lower().strip() for v in current_in_cat}

    print(f"\nCurrent sheet has {len(current_in_cat)} venues for '{category}':")
    for v in sorted(current_in_cat, key=lambda x: x['name']):
        print(f"  - {v['name']}")

    # Get grounded results
    print(f"\nSearching with Gemini Grounding...")
    grounded_venues = test_specific_category(category)

    if grounded_venues:
        grounded_names = {v['name'].lower().strip() for v in grounded_venues}

        overlap = current_names & grounded_names
        only_current = current_names - grounded_names
        only_grounded = grounded_names - current_names

        print(f"\n--- Comparison ---")
        print(f"Overlap: {len(overlap)} venues")
        print(f"Only in current sheet: {len(only_current)}")
        print(f"Only in grounded search: {len(only_grounded)}")

        if only_grounded:
            print(f"\nNew venues from grounding:")
            for name in sorted(only_grounded):
                print(f"  + {name}")


def main():
    print("Gemini Grounding Tests for Venue Discovery")
    print("=" * 60)

    # Test 1: Basic grounding
    test_basic_grounding()

    # Test 2: JSON extraction
    test_venue_extraction_json()

    # Test 3: Specific category
    test_specific_category("jazz clubs")

    # Test 4: Compare with current data
    test_comparison_with_current("comedy clubs standup")


if __name__ == "__main__":
    main()
