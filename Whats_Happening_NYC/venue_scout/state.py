"""Data models for venue scouting."""

from typing import TypedDict


class Venue(TypedDict):
    """A venue where events happen."""
    name: str
    address: str
    lat: float | None
    lng: float | None
    city: str
    neighborhood: str
    website: str
    events_url: str  # Direct URL to events/calendar page
    category: str
    description: str
    source: str  # How discovered (e.g., "web_search:comedy clubs")
    address_verified: str  # "yes", "not_found", or "" (not yet checked)
    # Website validation fields
    website_status: str      # "verified", "aggregator_rejected", "not_found", ""
    website_attempts: int    # Number of search attempts made
    # Event fetching fields
    preferred_event_source: str  # "ticketmaster", "scrape", "both", or ""
    api_endpoint: str            # Detected API URL if any
    ticketmaster_venue_id: str   # Ticketmaster venue ID if found
    # Website protection
    cloudflare_protected: str    # "yes" if site is behind Cloudflare; "" otherwise
    # Feed discovery fields
    feed_url: str   # Discovered iCal or RSS feed URL; "" if none found
    feed_type: str  # "ical", "rss", or ""
    # Event tracking fields (stored in Venues sheet, replaces JSON metadata)
    last_event_fetch: str        # ISO timestamp of last event fetch
    event_count: int             # Number of events found in last fetch
    event_source: str            # Source used: "ticketmaster", "api", "scrape", etc.


class VenueScoutState(TypedDict):
    """State for venue discovery process."""
    city: str
    venues: list[Venue]
    categories_searched: list[str]
    errors: list[str]
