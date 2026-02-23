"""Shared filesystem paths for the venue_scout package."""

from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"

# Data files
SEED_VENUES_FILE = DATA_DIR / "seed_venues.json"
VENUE_CACHE_METADATA_FILE = DATA_DIR / "venue_cache_metadata.json"
LOCAL_EVENTS_CACHE_FILE = DATA_DIR / "local_events_cache.json"
PLACES_CACHE_FILE = DATA_DIR / "places_cache.json"
TRAVEL_CACHE_FILE = DATA_DIR / "travel_cache.json"
VENUES_EXPORT_FILE = DATA_DIR / "venues.json"
SEMANTIC_EVENTS_INDEX_FILE = DATA_DIR / "semantic_events.faiss"
SEMANTIC_EVENTS_METADATA_FILE = DATA_DIR / "semantic_events_metadata.json"

# Output files
WEBSITE_VALIDATOR_OUTPUTS_DIR = OUTPUTS_DIR / "website_validator"


def ensure_data_dir() -> None:
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_outputs_dir() -> None:
    """Ensure outputs directory exists."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_website_validator_outputs_dir() -> None:
    """Ensure website validator outputs directory exists."""
    WEBSITE_VALIDATOR_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
