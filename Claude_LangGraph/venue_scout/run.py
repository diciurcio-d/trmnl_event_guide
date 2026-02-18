#!/usr/bin/env python3
"""
Venue Scout - CLI entry point.

Discovers venues in a city where events happen.

Usage:
    python venue_scout/run.py NYC                    # MVP categories (7)
    python venue_scout/run.py NYC --full             # All categories (29)
    python venue_scout/run.py NYC --group music      # Just music venues
    python venue_scout/run.py NYC --force            # Ignore cache
    python venue_scout/run.py "Los Angeles" --full   # Full scan of LA
"""

import argparse
import importlib.util
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from venue_scout.discovery import discover_venues
from venue_scout.cache import (
    read_cached_venues,
    get_cache_summary,
    get_searched_categories,
)


def _load_settings():
    """Load settings module."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


def main():
    parser = argparse.ArgumentParser(
        description="Discover venues in a city where events happen"
    )
    parser.add_argument(
        "city",
        nargs="?",
        default=None,
        help="City to search (e.g., 'NYC', 'Los Angeles', 'Chicago')"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh, ignore cache"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Search all venue categories (29 total, takes longer)"
    )
    parser.add_argument(
        "--group",
        choices=list(_settings.VENUE_CATEGORIES.keys()),
        help="Search only venues in a specific category group"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to search"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available category groups and exit"
    )

    args = parser.parse_args()

    # List categories mode
    if args.list_categories:
        print("\n📋 Available Venue Category Groups:\n")
        for group_name, cats in _settings.VENUE_CATEGORIES.items():
            print(f"  {group_name} ({len(cats)} categories):")
            for cat in cats:
                print(f"    • {cat}")
            print()
        print(f"Total: {len(_settings.VENUE_CATEGORIES_ALL)} categories")
        print(f"MVP subset: {len(_settings.VENUE_CATEGORIES_MVP)} categories")
        return

    city = args.city
    if not city:
        parser.error("city is required unless using --list-categories")

    # Determine which categories to search
    if args.categories:
        categories = args.categories
    elif args.group:
        categories = _settings.VENUE_CATEGORIES[args.group]
    elif args.full:
        categories = _settings.VENUE_CATEGORIES_ALL
    else:
        categories = _settings.VENUE_CATEGORIES_MVP

    print(f"\n🎭 Venue Scout - {city}")
    print("=" * 50)
    print(f"📋 Categories requested: {len(categories)}")

    # Show what's already cached
    already_searched = get_searched_categories(city)
    if already_searched and not args.force:
        print(f"✅ Already searched: {len(already_searched)} categories")

    # Discover venues (skips already-searched categories, caches incrementally)
    venues = discover_venues(city, categories, force=args.force)

    # Print results
    if venues:
        print(f"\n📍 Found {len(venues)} venues in {city}")
        print("-" * 50)

        # Group by category
        by_category = {}
        for venue in venues:
            cat = venue.get("category", "Other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(venue)

        for category, cat_venues in sorted(by_category.items()):
            print(f"\n🏷️  {category.title()} ({len(cat_venues)})")
            for venue in sorted(cat_venues, key=lambda v: v["name"]):
                name = venue["name"]
                neighborhood = venue.get("neighborhood", "")
                website = venue.get("website", "")

                if neighborhood:
                    print(f"   • {name} ({neighborhood})")
                else:
                    print(f"   • {name}")

                if website:
                    print(f"     {website}")
    else:
        print("\n❌ No venues found")

    # Print cache summary
    print("\n" + "=" * 50)
    summary = get_cache_summary()
    print(f"📊 Cache: {summary['total_venues']} total venues")
    if summary['venues_by_city']:
        for city_name, count in summary['venues_by_city'].items():
            print(f"   • {city_name}: {count} venues")


if __name__ == "__main__":
    main()
