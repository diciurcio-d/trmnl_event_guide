"""State definitions for the Event Analyzer Agent."""

from typing import TypedDict, Annotated, Literal
from datetime import datetime


class EventSource(TypedDict, total=False):
    """A configured event source with scraping metadata."""
    # Required fields
    name: str
    url: str
    method: Literal["api", "playwright"]
    enabled: bool

    # Location and type defaults
    default_location: str
    default_event_type: str

    # Playwright settings
    wait_seconds: int  # How long to wait after page load
    cloudflare_protected: bool  # Whether to check for Cloudflare challenge

    # API settings (for API sources)
    api_events_path: str  # JSON path to events array (e.g., "events" or "data.items")

    # Parsing hints for LLM
    parsing_hints: str  # Instructions to help LLM parse this source


class Event(TypedDict):
    """A single event."""
    name: str
    datetime: datetime | None
    date_str: str
    type: str
    sold_out: bool
    source: str
    location: str
    description: str
    has_specific_time: bool
    url: str
    travel_minutes: int | None  # Transit time from user's home


class Concert(TypedDict):
    """A concert from Ticketmaster."""
    artist: str
    artist_source: str  # subscription, liked_songs, or both
    artist_ytmusic_id: str
    artist_tm_id: str
    liked_songs: int
    event_id: str
    event_name: str
    date: str
    time: str
    venue: str
    city: str
    state: str
    url: str
    price_min: float | None
    price_max: float | None
    status: str
    travel_minutes: int | None  # Transit time from user's home


class AgentState(TypedDict):
    """State for the Event Analyzer Agent."""

    # Configured event sources
    sources: list[EventSource]

    # Aggregated events from all sources
    events: list[Event]

    # Concerts from Ticketmaster (based on user's music taste)
    concerts: list[Concert]

    # Conversation messages
    messages: Annotated[list, "add_messages"]

    # URL currently being analyzed for addition
    pending_source: str | None

    # Whether sources have been confirmed by user
    sources_confirmed: bool

    # Whether events have been fetched
    events_fetched: bool

    # Whether concerts have been fetched
    concerts_fetched: bool

    # Cache settings
    force_update: bool  # If True, bypass cache on next fetch
    cache_threshold_days: int  # Days before cache is considered stale


# Default sources with enhanced metadata
DEFAULT_SOURCES: list[EventSource] = [
    {
        "name": "Caveat NYC",
        "url": "https://www.caveat.nyc/api/events/upcoming",
        "method": "api",
        "enabled": True,
        "default_location": "Caveat NYC, 21 A Clinton St",
        "default_event_type": "Comedy/Performance",
        "api_events_path": "records",
        "api_format": "airtable",
        "parsing_hints": "Airtable API. Records have 'fields' with 'Event', 'Event start date and time', 'Short description'.",
    },
    {
        "name": "Riverside Park",
        "url": "https://riversideparknyc.org/wp-json/tribe/events/v1/events",
        "method": "api",
        "enabled": True,
        "default_location": "Riverside Park, NYC",
        "default_event_type": "Outdoor Event",
        "api_events_path": "events",
        "parsing_hints": "WordPress Tribe Events API. Events in 'events' array with 'title', 'start_date', 'venue'.",
    },
    {
        "name": "AMNH",
        "url": "https://www.amnh.org/calendar",
        "method": "playwright",
        "enabled": True,
        "default_location": "American Museum of Natural History",
        "default_event_type": "Museum Event",
        "wait_seconds": 5,
        "cloudflare_protected": True,
        "parsing_hints": "Look for event cards with type (Festival, Planetarium Program, etc.), title, and date. Dates are like 'Monday, February 3, 2026'.",
    },
    {
        "name": "The Met",
        "url": "https://www.metmuseum.org/events",
        "method": "playwright",
        "enabled": True,
        "default_location": "The Met Fifth Avenue",
        "default_event_type": "Museum Event",
        "wait_seconds": 5,
        "cloudflare_protected": True,
        "parsing_hints": "Events are grouped by day (e.g., 'Saturday, February 1'). Each event has a time like '10:30 AM' and title. Some at 'The Met Cloisters'.",
    },
    {
        "name": "Asia Society",
        "url": "https://asiasociety.org/new-york/events",
        "method": "playwright",
        "enabled": True,
        "default_location": "Asia Society, 725 Park Ave",
        "default_event_type": "Cultural Event",
        "wait_seconds": 4,
        "cloudflare_protected": False,
        "parsing_hints": "Events marked as IN-PERSON or VIRTUAL. Dates like 'Sat 15 Feb 2026'. Times in format '6:30-8 pm'.",
    },
    {
        "name": "Strand Books",
        "url": "https://www.strandbooks.com/events.html",
        "method": "playwright",
        "enabled": True,
        "default_location": "Strand Bookstore, 828 Broadway",
        "default_event_type": "Book Event",
        "wait_seconds": 3,
        "cloudflare_protected": False,
        "parsing_hints": "Calendar format with month headers (e.g., 'February 2026'). Days shown as 'MON', 'TUE' etc. with date number. Look for 'SOLD OUT' prefix on sold out events.",
    },
    {
        "name": "NY Historical Society",
        "url": "https://www.nyhistory.org/programs?genres=talks&subgenres=",
        "method": "playwright",
        "enabled": True,
        "default_location": "NY Historical Society, 170 Central Park West",
        "default_event_type": "Talk/Lecture",
        "wait_seconds": 5,
        "cloudflare_protected": False,
        "parsing_hints": "Events have 'TALKS' label. Dates like 'Date: Thursday, February 6, 6:30–8 pm'. Look for 'Sold Out' text.",
    },
    {
        "name": "Open House NY",
        "url": "https://ohny.org/calendar/",
        "method": "playwright",
        "enabled": True,
        "default_location": "Various NYC Locations",
        "default_event_type": "Architecture Tour",
        "wait_seconds": 5,
        "cloudflare_protected": False,
        "parsing_hints": "Calendar with 'Upcoming' section. Dates like 'February 8th, 2026'. Times like '10:00AM-12:00PM'. Some marked 'Members Only'.",
    },
    {
        "name": "Transit Techies",
        "url": "https://www.transittechies.com/events",
        "method": "playwright",
        "enabled": True,
        "default_location": "Various NYC Locations",
        "default_event_type": "Tech Meetup",
        "wait_seconds": 3,
        "cloudflare_protected": False,
        "parsing_hints": "Events about transit/transportation tech. Dates like 'Wednesday, January 28, 2026'. Times in format '6:30 PM 8:30 PM' (start and end). Each event has venue name and address. Look for speaker names and event descriptions.",
    },
    {
        "name": "92NY",
        "url": "https://www.92ny.org/whats-on/events?hierarchicalMenu%5BEventMenu.lvl0%5D%5B0%5D=Talks",
        "method": "playwright",
        "enabled": False,  # Disabled: Incapsula bot protection blocks all automated access
        "default_location": "92NY, 1395 Lexington Ave",
        "default_event_type": "Talk/Lecture",
        "wait_seconds": 8,
        "cloudflare_protected": True,
        "parsing_hints": "Cultural talks and lectures. Look for event titles, dates, and times. Events may show 'Sold Out' or ticket availability. Times typically in 12-hour format like '7:30 PM'.",
    },
]
