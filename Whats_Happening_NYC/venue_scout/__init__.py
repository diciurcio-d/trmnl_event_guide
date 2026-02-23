"""Venue Scout - discovers venues where events happen in a given city."""

from .state import Venue, VenueScoutState
from .discovery import discover_venues, search_venues_by_category
from .cache import (
    read_cached_venues,
    write_venues_to_cache,
    append_venues_to_cache,
    is_city_fresh,
    is_category_searched,
    mark_city_updated,
    mark_category_searched,
    get_searched_categories,
)
from .event_cache import (
    is_venue_events_fresh,
    mark_venue_fetched,
    get_stale_venues,
    clear_venue_cache,
)
from .event_fetcher import (
    fetch_venue_events,
    fetch_events_for_venues,
    fetch_events_by_category,
    FetchResult,
)
from .concert_matcher import (
    get_user_artists,
    match_events_to_artists,
    highlight_matched_events,
)
from .venue_events_sheet import (
    read_venue_events_from_sheet,
    write_venue_events_to_sheet,
    get_matched_events,
)

__all__ = [
    # Core types
    "Venue",
    "VenueScoutState",
    "FetchResult",
    # Venue discovery
    "discover_venues",
    "search_venues_by_category",
    # Venue cache
    "read_cached_venues",
    "write_venues_to_cache",
    "append_venues_to_cache",
    "is_city_fresh",
    "is_category_searched",
    "mark_city_updated",
    "mark_category_searched",
    "get_searched_categories",
    # Event cache
    "is_venue_events_fresh",
    "mark_venue_fetched",
    "get_stale_venues",
    "clear_venue_cache",
    # Event fetching
    "fetch_venue_events",
    "fetch_events_for_venues",
    "fetch_events_by_category",
    # Concert matching
    "get_user_artists",
    "match_events_to_artists",
    "highlight_matched_events",
    # Event storage
    "read_venue_events_from_sheet",
    "write_venue_events_to_sheet",
    "get_matched_events",
]
