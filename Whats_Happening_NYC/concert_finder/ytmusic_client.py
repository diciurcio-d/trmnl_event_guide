"""YouTube Music client to fetch user's liked songs and extract artists."""

import json
from pathlib import Path
from collections import Counter

# Browser auth path
BROWSER_AUTH_PATH = Path(__file__).parent.parent / "config" / "browser.json"
ARTISTS_CACHE = Path(__file__).parent / ".artists_cache.json"


class YTMusicClient:
    """Client for YouTube Music to get liked songs and extract artists."""

    def __init__(self, auth_path: str | Path | None = None):
        self.auth_path = Path(auth_path) if auth_path else BROWSER_AUTH_PATH
        self.ytmusic = None
        self._init_client()

    def _init_client(self):
        """Initialize the YTMusic client with browser auth."""
        try:
            from ytmusicapi import YTMusic
        except ImportError:
            raise ImportError(
                "ytmusicapi not installed. Run: pip install ytmusicapi"
            )

        if not self.auth_path.exists():
            print(f"Auth file not found at {self.auth_path}")
            print("Run 'ytmusicapi browser' to set up authentication.")
            return

        try:
            self.ytmusic = YTMusic(str(self.auth_path))
        except Exception as e:
            print(f"Failed to initialize YTMusic client: {e}")

    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self.ytmusic is not None

    def get_liked_songs(self, limit: int = 500) -> list[dict]:
        """
        Get user's liked songs.

        Args:
            limit: Maximum number of songs to fetch

        Returns:
            List of song dicts with title, artists, album, etc.
        """
        if not self.ytmusic:
            print("Not authenticated. Run setup first.")
            return []

        try:
            response = self.ytmusic.get_liked_songs(limit=limit)
            tracks = response.get("tracks", [])

            songs = []
            for track in tracks:
                if not track:
                    continue

                # Extract artist info
                artists = []
                for artist in track.get("artists", []):
                    if artist and artist.get("name"):
                        artists.append({
                            "name": artist["name"],
                            "id": artist.get("id"),
                        })

                songs.append({
                    "title": track.get("title", "Unknown"),
                    "artists": artists,
                    "album": track.get("album", {}).get("name") if track.get("album") else None,
                    "video_id": track.get("videoId"),
                    "duration": track.get("duration"),
                    "is_available": track.get("isAvailable", True),
                })

            return songs

        except Exception as e:
            print(f"Error fetching liked songs: {e}")
            return []

    def extract_unique_artists(
        self,
        songs: list[dict],
        min_song_count: int = 1,
    ) -> list[dict]:
        """
        Extract unique artists from songs, ranked by number of liked songs.

        Args:
            songs: List of song dicts from get_liked_songs()
            min_song_count: Minimum number of songs to include an artist

        Returns:
            List of artist dicts sorted by song count (descending)
        """
        artist_songs = Counter()
        artist_info = {}

        for song in songs:
            for artist in song.get("artists", []):
                name = artist.get("name")
                if not name:
                    continue

                # Normalize artist name (basic cleaning)
                name_clean = name.strip()

                artist_songs[name_clean] += 1
                if name_clean not in artist_info:
                    artist_info[name_clean] = {
                        "name": name_clean,
                        "id": artist.get("id"),
                    }

        # Filter and sort
        artists = []
        for name, count in artist_songs.most_common():
            if count >= min_song_count:
                info = artist_info[name]
                info["liked_song_count"] = count
                artists.append(info)

        return artists

    def get_subscriptions(self) -> list[dict]:
        """
        Get user's artist subscriptions.

        Returns:
            List of subscribed artist dicts with name, id, etc.
        """
        if not self.ytmusic:
            print("Not authenticated. Run setup first.")
            return []

        try:
            subs = self.ytmusic.get_library_subscriptions(limit=500)

            artists = []
            for sub in subs:
                artists.append({
                    "name": sub.get("artist", "Unknown"),
                    "id": sub.get("browseId"),
                    "subscribers": sub.get("subscribers"),
                    "source": "subscription",
                    "liked_song_count": 0,  # Will be updated if also in liked songs
                })

            return artists

        except Exception as e:
            print(f"Error fetching subscriptions: {e}")
            return []

    def get_top_artists(
        self,
        limit: int,
        min_songs: int,
        max_artists: int,
    ) -> list[dict]:
        """
        Get top artists based on liked songs.

        Args:
            limit: Number of liked songs to analyze
            min_songs: Minimum liked songs to include an artist
            max_artists: Maximum number of artists to return

        Returns:
            List of top artists sorted by liked song count
        """
        print(f"Fetching up to {limit} liked songs...")
        songs = self.get_liked_songs(limit=limit)

        if not songs:
            return []

        print(f"Found {len(songs)} liked songs. Extracting artists...")
        artists = self.extract_unique_artists(songs, min_song_count=min_songs)

        # Mark source
        for artist in artists:
            artist["source"] = "liked_songs"

        # Cache results
        cache_data = {
            "total_songs": len(songs),
            "total_artists": len(artists),
            "artists": artists[:max_artists],
        }
        ARTISTS_CACHE.write_text(json.dumps(cache_data, indent=2))

        return artists[:max_artists]

    def get_combined_artists(
        self,
        num_songs: int,
        min_songs: int,
        max_artists: int,
    ) -> list[dict]:
        """
        Get artists from both subscriptions and liked songs.

        Subscriptions are included first, then top artists by liked songs
        fill the remaining slots up to max_artists.

        Args:
            num_songs: Number of liked songs to analyze
            min_songs: Minimum liked songs to include an artist from likes
            max_artists: Total maximum number of artists to return

        Returns:
            Combined list of artists with source marked
        """
        # Get subscriptions first
        print("Fetching artist subscriptions...")
        subscriptions = self.get_subscriptions()
        print(f"Found {len(subscriptions)} subscriptions.")

        # Get liked songs and extract artists
        print(f"\nFetching up to {num_songs} liked songs...")
        songs = self.get_liked_songs(limit=num_songs)

        if not songs:
            print("No liked songs found.")
            return subscriptions[:max_artists]

        print(f"Found {len(songs)} liked songs. Extracting artists...")
        liked_artists = self.extract_unique_artists(songs, min_song_count=min_songs)

        # Build a lookup of liked song counts by artist name (case-insensitive)
        liked_counts = {a["name"].lower(): a["liked_song_count"] for a in liked_artists}
        liked_ids = {a["name"].lower(): a["id"] for a in liked_artists}

        # Update subscriptions with liked song counts
        sub_names = set()
        for sub in subscriptions:
            name_lower = sub["name"].lower()
            sub_names.add(name_lower)
            if name_lower in liked_counts:
                sub["liked_song_count"] = liked_counts[name_lower]
                sub["source"] = "both"
                # Use liked songs ID if subscription ID is missing
                if not sub.get("id") and liked_ids.get(name_lower):
                    sub["id"] = liked_ids[name_lower]

        # Calculate how many liked artists to add
        remaining_slots = max_artists - len(subscriptions)

        # Add top liked artists that aren't already subscribed
        combined = list(subscriptions)
        added = 0
        for artist in liked_artists:
            if added >= remaining_slots:
                break
            if artist["name"].lower() not in sub_names:
                artist["source"] = "liked_songs"
                combined.append(artist)
                added += 1

        print(f"\nCombined: {len(subscriptions)} subscriptions + {added} top liked artists = {len(combined)} total")

        # Cache results
        cache_data = {
            "total_songs": len(songs),
            "subscriptions": len(subscriptions),
            "liked_artists_added": added,
            "total_artists": len(combined),
            "artists": combined,
        }
        ARTISTS_CACHE.write_text(json.dumps(cache_data, indent=2))

        return combined


def main():
    """Test the YouTube Music client."""
    client = YTMusicClient()

    if not client.is_authenticated():
        print("\nNot authenticated. Run 'ytmusicapi browser' to set up.")
        return

    print("\nFetching your liked songs and extracting top artists...\n")

    artists = client.get_top_artists(limit=500, min_songs=2, max_artists=30)

    if artists:
        print(f"\nYour top {len(artists)} artists (by liked songs):\n")
        for i, artist in enumerate(artists, 1):
            print(f"  {i:2}. {artist['name']} ({artist['liked_song_count']} songs)")
    else:
        print("No artists found. Make sure you have liked songs on YouTube Music.")


if __name__ == "__main__":
    main()
