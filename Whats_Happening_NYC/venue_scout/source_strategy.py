"""Source selection strategy for venue event fetching.

Determines whether to use Ticketmaster, website scraping, or both
based on venue category and characteristics.
"""

# Categories that typically don't have Ticketmaster listings
SCRAPE_ONLY_CATEGORIES = frozenset([
    "bookstores with author events",
    "libraries with public events",
    "museums",
    "history museums",
    "science museums",
    "art galleries chelsea",
    "art galleries brooklyn",
    "photography galleries",
    "cultural centers",
    "community centers with events",
    "lecture halls universities",
    "workshop class spaces",
    "podcast recording venues",
    "parks with events concerts",
    "nyc parks with free events",
    "outdoor movies parks nyc",
    "summer concerts parks nyc",
    "shakespeare in the park nyc",
    "parks with programming activities",
    "outdoor event spaces gardens",
    "public plazas with programming",
])

# Keywords in venue names that suggest scrape-only
SCRAPE_ONLY_KEYWORDS = frozenset([
    "library",
    "museum",
    "gallery",
    "bookstore",
    "books",
    "botanical garden",  # More specific - avoid matching "Madison Square Garden"
    "public garden",
    "plaza",
    "cultural center",
    "community center",
    "university",
    "college",
])

# Known major venues that should always use Ticketmaster
KNOWN_TM_VENUES = frozenset([
    "madison square garden",
    "msg",
    "barclays center",
    "radio city music hall",
    "beacon theatre",
    "beacon theater",
    "carnegie hall",
    "terminal 5",
    "brooklyn steel",
    "webster hall",
    "irving plaza",
    "hammerstein ballroom",
    "forest hills stadium",
    "the rooftop at pier 17",
    "summerstage",
    "central park summerstage",
])

# Categories that are primarily on Ticketmaster
TICKETMASTER_CATEGORIES = frozenset([
    "concert halls",
    "live music venues",
    "rock music venues indie",
    "jazz clubs",
    "classical music venues",
    "opera houses",
    "broadway theaters",
    "off-broadway theaters",
    "sports arenas stadiums",
    "outdoor concert venues amphitheaters",
])


def should_skip_ticketmaster(category: str, venue_name: str = "") -> bool:
    """
    Determine if Ticketmaster should be skipped for this venue.

    Args:
        category: Venue category
        venue_name: Venue name (optional, for keyword matching)

    Returns:
        True if Ticketmaster should be skipped
    """
    cat_lower = category.lower().strip()
    name_lower = venue_name.lower().strip()

    # Never skip known Ticketmaster venues
    if name_lower in KNOWN_TM_VENUES:
        return False

    # Check if name contains a known TM venue
    for tm_venue in KNOWN_TM_VENUES:
        if tm_venue in name_lower or name_lower in tm_venue:
            return False

    # Check category
    if cat_lower in SCRAPE_ONLY_CATEGORIES:
        return True

    # Check keywords in venue name
    for keyword in SCRAPE_ONLY_KEYWORDS:
        if keyword in name_lower:
            return True

    return False


def determine_fetch_strategy(
    venue_name: str,
    category: str,
    website: str = "",
    preferred_source: str = "",
) -> str:
    """
    Determine the optimal fetch strategy for a venue.

    Args:
        venue_name: Name of the venue
        category: Venue category
        website: Venue website URL
        preferred_source: Previously determined preferred source

    Returns:
        One of: "ticketmaster_only", "scrape_only", "both", "scrape_first"
    """
    # If we've already determined the best source, use it
    if preferred_source:
        if preferred_source == "ticketmaster":
            return "ticketmaster_only"
        elif preferred_source == "scrape":
            return "scrape_only"

    # Check if venue category suggests scraping only
    if should_skip_ticketmaster(category, venue_name):
        return "scrape_only"

    # Check if venue category is primarily on Ticketmaster
    cat_lower = category.lower().strip()
    if cat_lower in TICKETMASTER_CATEGORIES:
        if website:
            return "both"  # Try both for first fetch to compare
        return "ticketmaster_only"

    # For venues with websites that are in mixed categories, try both
    if website:
        return "both"

    # Default: try Ticketmaster first
    return "ticketmaster_only"


def get_source_priority(strategy: str) -> list[str]:
    """
    Get ordered list of sources to try for a strategy.

    Args:
        strategy: One of the strategy strings

    Returns:
        List of sources in priority order
    """
    if strategy == "ticketmaster_only":
        return ["ticketmaster"]
    elif strategy == "scrape_only":
        return ["scrape"]
    elif strategy == "both":
        return ["ticketmaster", "scrape"]
    elif strategy == "scrape_first":
        return ["scrape", "ticketmaster"]
    else:
        return ["ticketmaster", "scrape"]


def select_best_source(
    tm_results: list | None,
    scrape_results: list | None,
) -> tuple[str, list]:
    """
    Compare results from both sources and select the best one.

    Args:
        tm_results: Events from Ticketmaster (or None if not tried)
        scrape_results: Events from scraping (or None if not tried)

    Returns:
        Tuple of (best_source, events_list)
    """
    tm_count = len(tm_results) if tm_results else 0
    scrape_count = len(scrape_results) if scrape_results else 0

    # If only one source has results, use it
    if tm_count > 0 and scrape_count == 0:
        return "ticketmaster", tm_results
    if scrape_count > 0 and tm_count == 0:
        return "scrape", scrape_results

    # If neither has results
    if tm_count == 0 and scrape_count == 0:
        return "", []

    # Both have results - prefer the one with more
    # Slight preference for Ticketmaster as data is more structured
    if tm_count >= scrape_count:
        return "ticketmaster", tm_results
    else:
        return "scrape", scrape_results
