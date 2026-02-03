"""Find upcoming concerts for artists from your YouTube Music liked songs."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from ytmusic_client import YTMusicClient
from ticketmaster_client import TicketmasterClient
import settings

# Add parent directory to path for cache imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from cache.concert_cache import (
    is_concerts_cache_fresh,
    read_cached_concerts,
    write_concerts_to_cache,
    get_concerts_cache_summary,
)

# JSON cache for detailed results
RESULTS_CACHE = Path(__file__).parent / ".concert_results.json"


def find_concerts_for_liked_artists(
    num_songs: int,
    min_songs_per_artist: int,
    max_artists: int,
    months_ahead: int,
    dma_id: str,
    use_attraction_id: bool,
    force_refresh: bool = False,
) -> dict:
    """
    Find concerts for artists from your YouTube Music liked songs.

    Args:
        num_songs: Number of liked songs to analyze
        min_songs_per_artist: Minimum liked songs to consider an artist
        max_artists: Maximum number of artists to search for
        months_ahead: How far ahead to search for concerts
        dma_id: Ticketmaster DMA ID (345 = NYC metro area)
        use_attraction_id: Use attraction ID lookup for more precise matches
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        Dict with artists checked and concerts found
    """
    # Check cache first
    if not force_refresh and is_concerts_cache_fresh(threshold_days=7):
        summary = get_concerts_cache_summary()
        cached_concerts = read_cached_concerts()
        print(f"Using cached concert data ({summary['total_concerts']} concerts, last updated: {summary['last_updated'][:10]})")
        return {
            "source": "cache",
            "concerts_found": cached_concerts,
            "cache_summary": summary,
        }

    print("Fetching fresh concert data...")

    # Initialize clients
    ytmusic = YTMusicClient()
    ticketmaster = TicketmasterClient()

    if not ytmusic.is_authenticated():
        return {"error": "YouTube Music not authenticated. Run 'ytmusicapi browser' first."}

    # Get artists from subscriptions + liked songs
    artists = ytmusic.get_combined_artists(
        num_songs=num_songs,
        min_songs=min_songs_per_artist,
        max_artists=max_artists,
    )

    if not artists:
        return {"error": "No artists found in liked songs"}

    print(f"\nFound {len(artists)} artists. Searching for concerts...\n")

    # Search for concerts for each artist
    results = {
        "search_date": datetime.now().isoformat(),
        "source": "youtube_music_liked_songs",
        "songs_analyzed": num_songs,
        "dma_id": dma_id,
        "months_ahead": months_ahead,
        "artists_checked": [],
        "concerts_found": [],
    }

    # Search sequentially with rate limiting (Ticketmaster allows 5 req/sec)
    for i, artist in enumerate(artists):
        artist_name = artist["name"]
        song_count = artist.get("liked_song_count", 0)
        source = artist.get("source", "unknown")
        tm_attraction_id = None

        try:
            if use_attraction_id:
                # Try to get attraction ID for more precise results
                tm_attraction_id = ticketmaster.get_attraction_id(artist_name)
                time.sleep(settings.REQUEST_DELAY_SECONDS)

                if tm_attraction_id:
                    events = ticketmaster.get_events_by_attraction_id(
                        tm_attraction_id, months_ahead=months_ahead, dma_id=dma_id
                    )
                else:
                    events = ticketmaster.search_artist_events(
                        artist_name, months_ahead=months_ahead, dma_id=dma_id
                    )
            else:
                events = ticketmaster.search_artist_events(
                    artist_name, months_ahead=months_ahead, dma_id=dma_id
                )

            time.sleep(settings.REQUEST_DELAY_SECONDS)

        except Exception as e:
            print(f"  {i+1:3}. ✗ {artist_name}: error - {e}")
            events = []

        # Store attraction ID on artist dict for use in concert records
        artist["tm_attraction_id"] = tm_attraction_id

        results["artists_checked"].append({
            "name": artist_name,
            "source": source,
            "ytmusic_id": artist.get("id"),
            "tm_attraction_id": tm_attraction_id,
            "liked_songs": song_count,
            "concerts_found": len(events),
        })

        if events:
            source_tag = "★" if source in ("subscription", "both") else " "
            print(f"  {i+1:3}. {source_tag} ✓ {artist_name} ({song_count} songs): {len(events)} concert(s)")
            for event in events:
                concert = {
                    "artist": artist_name,
                    "artist_source": source,
                    "artist_ytmusic_id": artist.get("id"),
                    "artist_tm_id": artist.get("tm_attraction_id"),
                    "liked_songs": song_count,
                    "event_id": event["id"],
                    "event_name": event["name"],
                    "date": event["date"],
                    "time": event["time"],
                    "venue": event["venue_name"],
                    "city": event["venue_city"],
                    "state": event["venue_state"],
                    "url": event["url"],
                    "price_min": event["price_min"],
                    "price_max": event["price_max"],
                    "status": event["status"],
                }
                results["concerts_found"].append(concert)
        else:
            source_tag = "★" if source in ("subscription", "both") else " "
            print(f"  {i+1:3}. {source_tag} · {artist_name} ({song_count} songs): no concerts")

    # Sort concerts by date
    results["concerts_found"].sort(key=lambda x: x["date"])

    # Deduplicate (same event might appear for multiple artists)
    seen = set()
    unique_concerts = []
    for concert in results["concerts_found"]:
        key = (concert["date"], concert["venue"], concert["event_name"])
        if key not in seen:
            seen.add(key)
            unique_concerts.append(concert)
    results["concerts_found"] = unique_concerts

    # Cache results to JSON (detailed)
    RESULTS_CACHE.write_text(json.dumps(results, indent=2))

    # Cache results to CSV (for LangGraph integration)
    write_concerts_to_cache(results["concerts_found"])
    print(f"\nCached {len(results['concerts_found'])} concerts to CSV.")

    return results


def print_concert_summary(results: dict):
    """Print a summary of found concerts."""
    concerts = results.get("concerts_found", [])

    if not concerts:
        print("\nNo concerts found for your liked artists in the specified area/timeframe.")
        return

    print(f"\n{'='*60}")
    print(f"  CONCERTS FOR YOUR YOUTUBE MUSIC ARTISTS")
    print(f"  {len(concerts)} shows found")
    print(f"{'='*60}\n")

    # Group by month
    current_month = None
    for concert in concerts:
        date = datetime.strptime(concert["date"], "%Y-%m-%d")
        month_str = date.strftime("%B %Y")

        if month_str != current_month:
            current_month = month_str
            print(f"\n--- {month_str} ---\n")

        day_str = date.strftime("%a %b %d")
        time_str = concert["time"][:5] if concert["time"] else "TBA"

        price_str = ""
        if concert["price_min"]:
            price_str = f" | ${concert['price_min']:.0f}-${concert['price_max']:.0f}"

        status = ""
        if concert["status"] == "onsale":
            status = " [ON SALE]"
        elif concert["status"] == "offsale":
            status = " [OFF SALE]"

        print(f"  {day_str} @ {time_str}{status}")
        print(f"  {concert['artist']} ({concert['liked_songs']} liked songs)")
        print(f"  {concert['event_name']}")
        print(f"  {concert['venue']}, {concert['city']}{price_str}")
        print(f"  {concert['url']}\n")


def main():
    """Run the concert finder."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find concerts for artists from your YouTube Music liked songs"
    )
    parser.add_argument(
        "-n", "--num-songs",
        type=int,
        help=f"Number of liked songs to analyze (default: {settings.YTMUSIC_NUM_SONGS})"
    )
    parser.add_argument(
        "-a", "--max-artists",
        type=int,
        help=f"Maximum artists to search for (default: {settings.YTMUSIC_MAX_ARTISTS})"
    )
    parser.add_argument(
        "--min-songs",
        type=int,
        help=f"Minimum liked songs to include an artist (default: {settings.YTMUSIC_MIN_SONGS_PER_ARTIST})"
    )
    parser.add_argument(
        "-m", "--months",
        type=int,
        help=f"Months ahead to search (default: {settings.TICKETMASTER_MONTHS_AHEAD})"
    )
    parser.add_argument(
        "--dma",
        help=f"Ticketmaster DMA ID (default: {settings.TICKETMASTER_DMA_ID} for NYC)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh, bypass cache"
    )

    args = parser.parse_args()

    results = find_concerts_for_liked_artists(
        num_songs=args.num_songs or settings.YTMUSIC_NUM_SONGS,
        min_songs_per_artist=args.min_songs or settings.YTMUSIC_MIN_SONGS_PER_ARTIST,
        max_artists=args.max_artists or settings.YTMUSIC_MAX_ARTISTS,
        months_ahead=args.months or settings.TICKETMASTER_MONTHS_AHEAD,
        dma_id=args.dma or settings.TICKETMASTER_DMA_ID,
        use_attraction_id=settings.TICKETMASTER_USE_ATTRACTION_ID,
        force_refresh=args.force,
    )

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_concert_summary(results)


if __name__ == "__main__":
    main()
