"""Ticketmaster Discovery API client to find concerts."""

import time
from datetime import datetime, timedelta
import requests

from config import get_ticketmaster_key

TICKETMASTER_API_BASE = "https://app.ticketmaster.com/discovery/v2"


class TicketmasterClient:
    """Client for Ticketmaster Discovery API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_ticketmaster_key()

        if not self.api_key:
            raise ValueError(
                "Ticketmaster API key required. Set TICKETMASTER_API_KEY env var "
                "or pass api_key to constructor."
            )

    def _request_with_retry(self, url: str, params: dict, max_retries: int = 3) -> requests.Response | None:
        """Make a request with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    time.sleep(wait_time)
                    continue

                return response

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        return None

    def search_events(
        self,
        keyword: str | None = None,
        artist_name: str | None = None,
        city: str | None = None,
        state_code: str | None = None,
        country_code: str = "US",
        dma_id: str | None = None,  # Designated Market Area (NYC = 345)
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        classification_name: str = "Music",
        size: int = 20,
    ) -> list[dict]:
        """
        Search for events on Ticketmaster.

        Args:
            keyword: General search term
            artist_name: Artist/attraction name to search for
            city: City name
            state_code: State code (e.g., "NY")
            country_code: Country code (default "US")
            dma_id: DMA ID for metro area (NYC = 345)
            start_date: Start of date range
            end_date: End of date range
            classification_name: Event category (Music, Sports, etc.)
            size: Number of results (max 200)

        Returns:
            List of event dicts
        """
        params = {
            "apikey": self.api_key,
            "countryCode": country_code,
            "size": size,
            "sort": "date,asc",
        }

        if keyword:
            params["keyword"] = keyword
        if artist_name:
            params["keyword"] = artist_name
        if city:
            params["city"] = city
        if state_code:
            params["stateCode"] = state_code
        if dma_id:
            params["dmaId"] = dma_id
        if classification_name:
            params["classificationName"] = classification_name

        # Date range
        if start_date:
            params["startDateTime"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_date:
            params["endDateTime"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = self._request_with_retry(
            f"{TICKETMASTER_API_BASE}/events.json",
            params=params,
        )

        if not response or response.status_code != 200:
            return []

        data = response.json()
        events_data = data.get("_embedded", {}).get("events", [])

        events = []
        for event in events_data:
            # Parse venue
            venues = event.get("_embedded", {}).get("venues", [])
            venue = venues[0] if venues else {}

            # Parse date/time
            dates = event.get("dates", {}).get("start", {})
            date_str = dates.get("localDate", "")
            time_str = dates.get("localTime", "")

            # Parse price range
            price_ranges = event.get("priceRanges", [])
            price_min = price_ranges[0].get("min") if price_ranges else None
            price_max = price_ranges[0].get("max") if price_ranges else None

            # Get attractions (artists)
            attractions = event.get("_embedded", {}).get("attractions", [])
            artist_names = [a.get("name") for a in attractions]

            events.append({
                "id": event.get("id"),
                "name": event.get("name"),
                "date": date_str,
                "time": time_str,
                "datetime_str": f"{date_str} {time_str}".strip(),
                "venue_name": venue.get("name"),
                "venue_city": venue.get("city", {}).get("name"),
                "venue_state": venue.get("state", {}).get("stateCode"),
                "venue_address": venue.get("address", {}).get("line1"),
                "artists": artist_names,
                "url": event.get("url"),
                "price_min": price_min,
                "price_max": price_max,
                "image_url": event.get("images", [{}])[0].get("url"),
                "status": event.get("dates", {}).get("status", {}).get("code"),
            })

        return events

    def search_artist_events(
        self,
        artist_name: str,
        months_ahead: int = 3,
        dma_id: str | None = "345",  # NYC metro area
    ) -> list[dict]:
        """
        Search for upcoming events by a specific artist.

        Args:
            artist_name: Name of the artist
            months_ahead: How many months to look ahead
            dma_id: DMA ID (345 = NYC metro)

        Returns:
            List of events for this artist
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(days=months_ahead * 30)

        return self.search_events(
            artist_name=artist_name,
            dma_id=dma_id,
            start_date=start_date,
            end_date=end_date,
        )

    def get_attraction_id(self, artist_name: str) -> str | None:
        """
        Search for an artist/attraction ID.

        This can be used for more precise event searches.
        """
        params = {
            "apikey": self.api_key,
            "keyword": artist_name,
            "size": 1,
        }

        response = self._request_with_retry(
            f"{TICKETMASTER_API_BASE}/attractions.json",
            params=params,
        )

        if not response or response.status_code != 200:
            return None

        data = response.json()
        attractions = data.get("_embedded", {}).get("attractions", [])

        if attractions:
            return attractions[0].get("id")
        return None

    def get_events_by_attraction_id(
        self,
        attraction_id: str,
        months_ahead: int = 3,
        dma_id: str | None = "345",
    ) -> list[dict]:
        """
        Get events for a specific attraction ID (more precise than keyword search).
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(days=months_ahead * 30)

        params = {
            "apikey": self.api_key,
            "attractionId": attraction_id,
            "size": 50,
            "sort": "date,asc",
            "startDateTime": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDateTime": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        if dma_id:
            params["dmaId"] = dma_id

        response = self._request_with_retry(
            f"{TICKETMASTER_API_BASE}/events.json",
            params=params,
        )

        if not response or response.status_code != 200:
            return []

        data = response.json()
        events_data = data.get("_embedded", {}).get("events", [])

        # Use same parsing as search_events
        events = []
        for event in events_data:
            venues = event.get("_embedded", {}).get("venues", [])
            venue = venues[0] if venues else {}
            dates = event.get("dates", {}).get("start", {})
            price_ranges = event.get("priceRanges", [])
            attractions = event.get("_embedded", {}).get("attractions", [])

            events.append({
                "id": event.get("id"),
                "name": event.get("name"),
                "date": dates.get("localDate", ""),
                "time": dates.get("localTime", ""),
                "venue_name": venue.get("name"),
                "venue_city": venue.get("city", {}).get("name"),
                "venue_state": venue.get("state", {}).get("stateCode"),
                "artists": [a.get("name") for a in attractions],
                "url": event.get("url"),
                "price_min": price_ranges[0].get("min") if price_ranges else None,
                "price_max": price_ranges[0].get("max") if price_ranges else None,
                "status": event.get("dates", {}).get("status", {}).get("code"),
            })

        return events


if __name__ == "__main__":
    client = TicketmasterClient()

    print("Searching for concerts in NYC area...\n")

    # Example: Search for a specific artist
    artist = "Tyler the Creator"
    events = client.search_artist_events(artist, months_ahead=6)

    if events:
        print(f"Found {len(events)} events for '{artist}':\n")
        for event in events:
            price_str = ""
            if event["price_min"]:
                price_str = f" (${event['price_min']}-${event['price_max']})"

            print(f"  {event['date']} - {event['name']}")
            print(f"    {event['venue_name']}, {event['venue_city']}")
            print(f"    {event['url']}{price_str}\n")
    else:
        print(f"No upcoming events found for '{artist}' in NYC area.")
