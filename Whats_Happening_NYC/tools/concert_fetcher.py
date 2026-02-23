"""Tool for fetching concerts based on user's YouTube Music preferences."""

import sys
from pathlib import Path
from langchain_core.tools import tool

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "concert_finder"))

from cache.concert_cache import (
    is_concerts_cache_fresh,
    read_cached_concerts,
    get_concerts_cache_summary,
)


@tool
def fetch_concerts(force_update: bool = False, cache_threshold_days: int = 7) -> dict:
    """
    Fetch concerts for the user's YouTube Music artists.

    Uses cached data if fresh, otherwise fetches from Ticketmaster API.

    Args:
        force_update: If True, bypass cache and fetch fresh data
        cache_threshold_days: Days before cache is considered stale

    Returns:
        Dict with concerts list and metadata
    """
    # Check cache first
    if not force_update and is_concerts_cache_fresh(threshold_days=cache_threshold_days):
        summary = get_concerts_cache_summary()
        concerts = read_cached_concerts()
        return {
            "source": "cache",
            "concerts": concerts,
            "total": len(concerts),
            "cache_status": "fresh",
            "last_updated": summary.get("last_updated", "unknown"),
        }

    # Need to fetch fresh data
    try:
        from find_concerts import find_concerts_for_liked_artists
        import settings

        result = find_concerts_for_liked_artists(
            num_songs=settings.YTMUSIC_NUM_SONGS,
            min_songs_per_artist=settings.YTMUSIC_MIN_SONGS_PER_ARTIST,
            max_artists=settings.YTMUSIC_MAX_ARTISTS,
            months_ahead=settings.TICKETMASTER_MONTHS_AHEAD,
            dma_id=settings.TICKETMASTER_DMA_ID,
            use_attraction_id=settings.TICKETMASTER_USE_ATTRACTION_ID,
            force_refresh=True,
        )

        if "error" in result:
            # Try to use stale cache as fallback
            concerts = read_cached_concerts()
            if concerts:
                return {
                    "source": "stale_cache",
                    "concerts": concerts,
                    "total": len(concerts),
                    "cache_status": "stale",
                    "error": result["error"],
                }
            return {
                "source": "error",
                "concerts": [],
                "total": 0,
                "cache_status": "error",
                "error": result["error"],
            }

        concerts = result.get("concerts_found", [])
        return {
            "source": "fetched",
            "concerts": concerts,
            "total": len(concerts),
            "cache_status": "fresh",
            "artists_checked": len(result.get("artists_checked", [])),
        }

    except Exception as e:
        # Try stale cache as fallback
        concerts = read_cached_concerts()
        if concerts:
            return {
                "source": "stale_cache",
                "concerts": concerts,
                "total": len(concerts),
                "cache_status": "stale",
                "error": str(e),
            }
        return {
            "source": "error",
            "concerts": [],
            "total": 0,
            "cache_status": "error",
            "error": str(e),
        }
