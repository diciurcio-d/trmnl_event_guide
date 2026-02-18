#!/usr/bin/env python3
"""
Test Gemini API with Google Search Grounding - V2 with better JSON parsing.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google import genai
from google.genai import types

# Load config
config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
with open(config_path) as f:
    config = json.load(f)

client = genai.Client(api_key=config["gemini"]["api_key"])


def extract_json_from_response(text: str) -> list | None:
    """Extract JSON array from response text, handling markdown and preamble."""
    # Try to find JSON array in the text
    # First, look for ```json ... ``` blocks
    json_match = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON array
    array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    # Try the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def search_venues(category: str, city: str = "NYC") -> list:
    """Search for venues using Gemini with grounding."""

    prompt = f"""Search for {category} in {city}.

Return a JSON array of venues. For each venue include:
- name: Venue name (actual business name only)
- address: Full address or "{city}" if unknown
- neighborhood: NYC neighborhood
- website: Direct venue website (NOT Google redirect URLs)
- description: Brief description under 50 words

Return 10-20 venues. Return ONLY the JSON array, no other text.
Example: [{{"name": "Example", "address": "123 St", "neighborhood": "Midtown", "website": "https://example.com", "description": "A venue"}}]"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    venues = extract_json_from_response(response.text)
    return venues or []


def compare_with_sheet(category: str):
    """Compare grounding results with current sheet data."""
    from venue_scout.cache import read_cached_venues

    # Load current venues
    current_venues = read_cached_venues()
    current_in_cat = [v for v in current_venues if v.get('source') == f'web_search:{category}']
    current_names = {v['name'].lower().strip() for v in current_in_cat}

    print(f"\n{'='*60}")
    print(f"Category: {category}")
    print(f"{'='*60}")
    print(f"\nCurrent sheet: {len(current_in_cat)} venues")

    # Get grounded results
    print("Searching with Gemini Grounding...")
    grounded = search_venues(category)

    print(f"Grounding found: {len(grounded)} venues")

    if grounded:
        grounded_names = {v['name'].lower().strip() for v in grounded}

        overlap = current_names & grounded_names
        only_current = current_names - grounded_names
        only_grounded = grounded_names - current_names

        print(f"\n--- Results ---")
        print(f"Overlap: {len(overlap)}")
        print(f"Only in sheet: {len(only_current)}")
        print(f"Only in grounding: {len(only_grounded)}")

        if overlap:
            print(f"\nMatched venues:")
            for name in sorted(overlap)[:5]:
                print(f"  ✓ {name}")

        if only_grounded:
            print(f"\nNEW venues from grounding:")
            for name in sorted(only_grounded):
                print(f"  + {name}")

        if only_current:
            print(f"\nMissing from grounding (in sheet only):")
            for name in sorted(only_current)[:5]:
                print(f"  - {name}")
            if len(only_current) > 5:
                print(f"  ... and {len(only_current) - 5} more")

        return {
            'category': category,
            'current_count': len(current_in_cat),
            'grounded_count': len(grounded),
            'overlap': len(overlap),
            'new_from_grounding': len(only_grounded),
            'grounded_venues': grounded
        }

    return None


def test_multiple_categories():
    """Test several categories and summarize."""
    categories = [
        "comedy clubs standup",
        "jazz clubs",
        "live music venues",
        "bookstores with author events",
        "broadway theaters",
    ]

    results = []
    for cat in categories:
        result = compare_with_sheet(cat)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<30} {'Sheet':>6} {'Grnd':>6} {'Ovlp':>6} {'New':>5}")
    print("-" * 60)

    for r in results:
        print(f"{r['category'][:28]:<30} {r['current_count']:>6} {r['grounded_count']:>6} {r['overlap']:>6} {r['new_from_grounding']:>5}")


if __name__ == "__main__":
    test_multiple_categories()
