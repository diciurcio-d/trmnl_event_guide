"""Match venue events to user's YouTube Music artists.

Integrates with YTMusicClient to get user's artists and matches
them against event names/artists.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _normalize_artist_name(name: str) -> str:
    """Normalize artist name for matching."""
    name = name.lower().strip()
    # Remove common suffixes
    name = re.sub(r'\s+(live|tour|concert|presents?)$', '', name)
    # Remove "the" prefix
    name = re.sub(r'^the\s+', '', name)
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name)
    return name


def get_user_artists(
    num_songs: int = 1000,
    min_songs: int = 2,
    max_artists: int = 100,
) -> list[dict]:
    """
    Get user's artists from YouTube Music.

    Args:
        num_songs: Number of liked songs to analyze
        min_songs: Minimum liked songs to include an artist
        max_artists: Maximum artists to return

    Returns:
        List of artist dicts with name, id, liked_song_count
    """
    try:
        from concert_finder.ytmusic_client import YTMusicClient, ARTISTS_CACHE
    except ImportError:
        print("Could not import YTMusicClient")
        return []

    # Try to load from cache first
    if ARTISTS_CACHE.exists():
        import json
        try:
            with open(ARTISTS_CACHE) as f:
                cache = json.load(f)
            if cache.get("artists"):
                print(f"Loaded {len(cache['artists'])} artists from cache")
                return cache["artists"][:max_artists]
        except (json.JSONDecodeError, IOError):
            pass

    # Fetch from YTMusic
    client = YTMusicClient()
    if not client.is_authenticated():
        print("Not authenticated with YouTube Music")
        return []

    artists = client.get_top_artists(
        limit=num_songs,
        min_songs=min_songs,
        max_artists=max_artists,
    )
    return artists


def get_artist_names(artists: list[dict] | None = None) -> set[str]:
    """
    Get normalized set of artist names.

    Args:
        artists: List of artist dicts (will fetch if None)

    Returns:
        Set of normalized artist names
    """
    if artists is None:
        artists = get_user_artists()

    names = set()
    for artist in artists:
        name = artist.get("name", "")
        if name:
            names.add(_normalize_artist_name(name))
    return names


def match_event_to_artists(
    event: dict,
    artist_names: set[str],
) -> str | None:
    """
    Check if an event matches any of the user's artists.

    Args:
        event: Event dict with name, artists fields
        artist_names: Set of normalized artist names

    Returns:
        Matched artist name if found, None otherwise
    """
    # Check explicit artists field (from Ticketmaster)
    event_artists = event.get("artists", [])
    for artist in event_artists:
        if isinstance(artist, str):
            norm = _normalize_artist_name(artist)
            if norm in artist_names:
                return artist

    # Check event name for artist mentions
    event_name = event.get("name", "")
    event_name_norm = _normalize_artist_name(event_name)

    for artist_name in artist_names:
        # Check if artist name appears in event name
        if artist_name in event_name_norm:
            # Avoid false positives for very short names
            if len(artist_name) > 3:
                return artist_name

        # Check for word boundary match
        pattern = r'\b' + re.escape(artist_name) + r'\b'
        if re.search(pattern, event_name_norm):
            return artist_name

    return None


def match_events_to_artists(
    events: list[dict],
    artists: list[dict] | None = None,
) -> list[dict]:
    """
    Match a list of events to user's artists.

    Args:
        events: List of event dicts
        artists: List of artist dicts (will fetch if None)

    Returns:
        Events with matched_artist field populated
    """
    artist_names = get_artist_names(artists)

    if not artist_names:
        print("No artists to match against")
        return events

    matched_count = 0
    for event in events:
        matched = match_event_to_artists(event, artist_names)
        if matched:
            event["matched_artist"] = matched
            matched_count += 1
        else:
            event["matched_artist"] = ""

    print(f"Matched {matched_count}/{len(events)} events to user's artists")
    return events


def highlight_matched_events(
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Separate matched events from others.

    Args:
        events: List of event dicts with matched_artist field

    Returns:
        Tuple of (matched_events, other_events)
    """
    matched = []
    other = []

    for event in events:
        if event.get("matched_artist"):
            matched.append(event)
        else:
            other.append(event)

    return matched, other


def get_matched_events_summary(events: list[dict]) -> dict:
    """
    Get summary of matched events by artist.

    Returns:
        Dict mapping artist name -> list of events
    """
    by_artist = {}
    for event in events:
        artist = event.get("matched_artist")
        if artist:
            if artist not in by_artist:
                by_artist[artist] = []
            by_artist[artist].append(event)

    return by_artist


if __name__ == "__main__":
    print("Testing concert matcher...")

    # Get user's artists
    artists = get_user_artists(num_songs=500, min_songs=2, max_artists=50)
    print(f"\nFound {len(artists)} artists")

    if artists:
        print("\nTop 10 artists:")
        for artist in artists[:10]:
            print(f"  - {artist['name']} ({artist.get('liked_song_count', 0)} songs)")

    # Test matching with sample events
    test_events = [
        {"name": "Taylor Swift | The Eras Tour", "artists": ["Taylor Swift"]},
        {"name": "Jazz Night at Blue Note", "artists": []},
        {"name": "Random Comedy Show", "artists": []},
    ]

    matched_events = match_events_to_artists(test_events, artists)

    print("\nTest event matching:")
    for event in matched_events:
        matched = event.get("matched_artist", "")
        status = f"MATCHED: {matched}" if matched else "no match"
        print(f"  - {event['name']}: {status}")
