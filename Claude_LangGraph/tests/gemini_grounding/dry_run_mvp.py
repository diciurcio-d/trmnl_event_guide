#!/usr/bin/env python3
"""
MVP dry run - 7 categories including one large category.
Exports results to a new Google Sheet for review.
"""

import json
import re
import sys
import time
from datetime import datetime
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
OUTPUT_DIR = Path(__file__).parent.parent.parent / "venue_scout" / "outputs" / "gemini_grounding"

MAX_VENUES_PER_CATEGORY = 250
GROUNDING_DELAY = 13  # ~5 RPM for Gemini 2.5 Pro

_last_grounding_call = 0

# MVP categories - 7 random including 1 large (art galleries)
MVP_TEST_CATEGORIES = [
    "art galleries",              # Large category (~1500)
    "jazz clubs",                 # Medium
    "broadway theaters",          # Well-defined
    "bookstores with author events",  # Small-medium
    "comedy clubs standup",       # Medium
    "outdoor concert venues amphitheaters",  # Small
    "museums",                    # Medium-large
]


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
    prompt = f"How many {category} exist in {city}? Include all types. Respond with just the number."
    response = client.models.generate_content(
        model=GROUNDING_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    count = extract_number(response.text)
    print(f"  Estimated: {count}", flush=True)
    return count or 30


def get_most_important_venues(category: str, city: str = "NYC", limit: int = 75) -> list:
    rate_limit_grounding()
    prompt = f"""Search for the most important, notable, and well-known {category} in {city}.
Focus on famous, established, landmark venues.
Return up to {limit} venues as JSON array ordered by importance.
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
        print(f"    Error: {e}", flush=True)
        return []


def get_subqueries(category: str, estimated_count: int, city: str = "NYC") -> list[str]:
    num_subqueries = max(3, min(15, (estimated_count // 40) + 1))
    prompt = f"""I need to find ALL {category} in {city} (approximately {estimated_count} total).
I already have the most famous ones.
Generate {num_subqueries} search queries to find the REMAINING venues.
Divide by: neighborhood, specialty, size, type.
Return as JSON array of search query strings.
Return ONLY the JSON array."""

    response = client.models.generate_content(model=FAST_MODEL, contents=prompt)
    result = extract_json_from_response(response.text)
    if result and isinstance(result, list):
        return [str(q) for q in result]
    return [f"{category} in {city}"]


def get_venues_with_grounding(query: str, exclude_names: set = None, city: str = "NYC") -> list:
    rate_limit_grounding()
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
        response = client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return extract_json_from_response(response.text) or []
    except Exception as e:
        print(f"    Error: {e}", flush=True)
        return []


def discover_venues_prioritized(category: str, city: str = "NYC") -> list:
    """Prioritized venue discovery."""
    print(f"\n{'='*60}", flush=True)
    print(f"DISCOVERING: {category}", flush=True)
    print(f"{'='*60}", flush=True)

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
                # Add category to venue
                v['category'] = category
                all_venues.append(v)
                new_count += 1
        return new_count

    # Phase 1: Most important venues
    print("\n--- PHASE 1: Most Important Venues ---", flush=True)
    important = get_most_important_venues(category, city, limit=75)
    new_count = add_venues(important)
    print(f"  Found {new_count} notable venues", flush=True)

    if len(all_venues) >= 50 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
        print("  Getting more notable venues...", flush=True)
        more = get_venues_with_grounding(f"more notable {category}", seen_names, city)
        new_count = add_venues(more)
        print(f"  Found {new_count} more", flush=True)

    if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
        print(f"  Reached cap with notable venues.", flush=True)
    else:
        # Phase 2: Comprehensive discovery
        print("\n--- PHASE 2: Comprehensive Discovery ---", flush=True)
        estimated = get_estimated_count(category, city)

        if estimated > len(all_venues) and len(all_venues) < MAX_VENUES_PER_CATEGORY:
            subqueries = get_subqueries(category, estimated, city)
            print(f"  Generated {len(subqueries)} sub-queries", flush=True)

            for i, sq in enumerate(subqueries, 1):
                if len(all_venues) >= MAX_VENUES_PER_CATEGORY:
                    print(f"\n  Reached {MAX_VENUES_PER_CATEGORY} cap.", flush=True)
                    break

                print(f"\n  [{i}/{len(subqueries)}] '{sq[:45]}...'", flush=True)
                venues = get_venues_with_grounding(sq, seen_names, city)
                new_count = add_venues(venues)
                print(f"    Found {new_count} new (total: {len(all_venues)})", flush=True)

                if new_count >= 20 and len(all_venues) < MAX_VENUES_PER_CATEGORY:
                    venues = get_venues_with_grounding(sq, seen_names, city)
                    new_count = add_venues(venues)
                    print(f"    Batch 2: {new_count} new (total: {len(all_venues)})", flush=True)

    print(f"\nCATEGORY TOTAL: {len(all_venues)} venues", flush=True)
    return all_venues


def export_to_sheet(all_venues: list, sheet_name: str):
    """Export venues to a new Google Sheet."""
    from venue_scout.cache import _get_sheets_service

    service = _get_sheets_service()
    if not service:
        print("ERROR: Could not get Sheets service")
        return None

    # Create new spreadsheet
    spreadsheet = service.spreadsheets().create(
        body={
            'properties': {'title': sheet_name},
            'sheets': [{'properties': {'title': 'Venues'}}]
        }
    ).execute()

    spreadsheet_id = spreadsheet['spreadsheetId']
    print(f"\nCreated spreadsheet: {sheet_name}")
    print(f"ID: {spreadsheet_id}")

    # Prepare data
    headers = ['name', 'address', 'neighborhood', 'category']
    rows = [headers]
    for v in all_venues:
        rows.append([
            v.get('name', ''),
            v.get('address', ''),
            v.get('neighborhood', ''),
            v.get('category', ''),
        ])

    # Write data
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range='Venues!A1',
        valueInputOption='RAW',
        body={'values': rows}
    ).execute()

    print(f"Wrote {len(all_venues)} venues to sheet")
    print(f"URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")

    return spreadsheet_id


def main():
    print("=" * 60)
    print("MVP DRY RUN - 7 Categories")
    print("=" * 60)
    print(f"\nCategories to discover:")
    for cat in MVP_TEST_CATEGORIES:
        print(f"  - {cat}")

    all_venues = []

    for category in MVP_TEST_CATEGORIES:
        venues = discover_venues_prioritized(category, "NYC")
        all_venues.extend(venues)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    from collections import Counter
    by_category = Counter(v.get('category') for v in all_venues)
    for cat, count in by_category.most_common():
        print(f"  {cat}: {count}")

    print(f"\nTOTAL: {len(all_venues)} venues")

    # Save to JSON backup first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"mvp_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_venues, f, indent=2)
    print(f"\nSaved backup to: {json_path}")

    # Export to sheet
    sheet_name = f"Venue Discovery MVP Test {timestamp}"
    export_to_sheet(all_venues, sheet_name)


if __name__ == "__main__":
    main()
