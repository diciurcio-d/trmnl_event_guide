#!/usr/bin/env python3
"""Export venues to JSON for the frontend."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.cache import read_cached_venues
from venue_scout.paths import VENUES_EXPORT_FILE, ensure_data_dir


def export_venues(city: str, output_path: str | None = None):
    """Export venues for a city to JSON."""
    venues = read_cached_venues(city)

    if not output_path:
        ensure_data_dir()
        output_path = VENUES_EXPORT_FILE

    # Group by category
    by_category = {}
    for v in venues:
        cat = v.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "name": v.get("name", ""),
            "address": v.get("address", ""),
            "neighborhood": v.get("neighborhood", ""),
            "website": v.get("website", ""),
            "description": v.get("description", ""),
        })

    # Sort venues within each category
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["name"].lower())

    data = {
        "city": city,
        "total": len(venues),
        "categories": by_category,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(venues)} venues in {len(by_category)} categories to {output_path}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "NYC"
    export_venues(city)
