"""Google Maps Distance Matrix API for travel time estimation."""

import json
import requests
from pathlib import Path


def _load_config() -> dict:
    """Load config from file."""
    config_path = Path(__file__).parent.parent / "concert_finder" / "config.json"
    with open(config_path) as f:
        return json.load(f)


def get_travel_time(
    origin: str,
    destination: str,
    mode: str = "transit",
) -> dict:
    """
    Get travel time between two locations using Google Distance Matrix API.

    Args:
        origin: Starting address or coordinates
        destination: Destination address or coordinates
        mode: Travel mode - "transit", "driving", "walking", "bicycling"

    Returns:
        Dict with:
        - duration_minutes: Travel time in minutes
        - duration_text: Human-readable duration (e.g., "45 mins")
        - distance_km: Distance in kilometers
        - distance_text: Human-readable distance (e.g., "12.5 km")
        - status: "OK" or error message
    """
    config = _load_config()
    api_key = config.get("google_cloud", {}).get("api_key")

    if not api_key:
        return {"status": "error", "error": "Google Cloud API key not configured"}

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "mode": mode,
        "key": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "OK":
            return {"status": "error", "error": data.get("error_message", data["status"])}

        element = data["rows"][0]["elements"][0]

        if element["status"] != "OK":
            return {"status": "error", "error": element["status"]}

        return {
            "status": "OK",
            "duration_minutes": element["duration"]["value"] // 60,
            "duration_text": element["duration"]["text"],
            "distance_km": element["distance"]["value"] / 1000,
            "distance_text": element["distance"]["text"],
        }

    except requests.RequestException as e:
        return {"status": "error", "error": str(e)}


def get_user_home_address() -> str | None:
    """Get the user's home address from config."""
    config = _load_config()
    address = config.get("user", {}).get("home_address", "")
    return address if address else None


def get_max_travel_minutes() -> int:
    """Get the user's max travel time preference."""
    config = _load_config()
    return config.get("user", {}).get("max_travel_minutes", 60)


def is_within_travel_limit(
    destination: str,
    max_minutes: int | None = None,
    mode: str = "transit",
) -> tuple[bool, dict]:
    """
    Check if a destination is within the user's travel time limit.

    Args:
        destination: Destination address
        max_minutes: Override max travel time (uses config default if None)
        mode: Travel mode

    Returns:
        Tuple of (is_within_limit, travel_info_dict)
    """
    home = get_user_home_address()
    if not home:
        return True, {"status": "no_home_address"}

    if max_minutes is None:
        max_minutes = get_max_travel_minutes()

    travel_info = get_travel_time(home, destination, mode)

    if travel_info["status"] != "OK":
        # If we can't calculate, don't filter out
        return True, travel_info

    is_within = travel_info["duration_minutes"] <= max_minutes
    return is_within, travel_info


_travel_time_cache: dict[str, int | None] = {}


def get_travel_time_cached(destination: str, mode: str = "transit") -> int | None:
    """
    Get travel time with caching to avoid repeated API calls for same location.

    Args:
        destination: Destination address
        mode: Travel mode

    Returns:
        Travel time in minutes, or None if unavailable
    """
    home = get_user_home_address()
    if not home:
        return None

    cache_key = f"{destination}|{mode}"
    if cache_key in _travel_time_cache:
        return _travel_time_cache[cache_key]

    result = get_travel_time(home, destination, mode)
    if result["status"] == "OK":
        minutes = result["duration_minutes"]
        _travel_time_cache[cache_key] = minutes
        return minutes
    else:
        _travel_time_cache[cache_key] = None
        return None


def enrich_events_with_travel_time(events: list[dict], mode: str = "transit") -> list[dict]:
    """
    Add travel_minutes field to a list of events.

    Caches travel times by location to minimize API calls.

    Args:
        events: List of event dicts with 'location' field
        mode: Travel mode for distance calculation

    Returns:
        Same events list with travel_minutes added
    """
    home = get_user_home_address()
    if not home:
        print("    No home address configured, skipping travel time calculation")
        for event in events:
            event["travel_minutes"] = None
        return events

    # Group events by unique location
    unique_locations = set(e.get("location", "") for e in events if e.get("location"))
    print(f"    Calculating travel times for {len(unique_locations)} unique locations...")

    # Calculate travel time for each unique location
    location_times: dict[str, int | None] = {}
    for location in unique_locations:
        if not location:
            continue
        travel_mins = get_travel_time_cached(location, mode)
        location_times[location] = travel_mins

    # Apply to events
    for event in events:
        location = event.get("location", "")
        event["travel_minutes"] = location_times.get(location)

    return events


def test_distance_api():
    """Test the Distance Matrix API."""
    print("Testing Distance Matrix API...")

    home = get_user_home_address()
    if not home:
        print("No home address configured. Set 'user.home_address' in config.json")
        print("\nTesting with sample addresses...")
        home = "Times Square, New York, NY"

    destinations = [
        "Strand Bookstore, 828 Broadway, New York, NY",
        "American Museum of Natural History, New York, NY",
        "Barclays Center, Brooklyn, NY",
    ]

    for dest in destinations:
        print(f"\nFrom: {home}")
        print(f"To: {dest}")
        result = get_travel_time(home, dest)
        if result["status"] == "OK":
            print(f"  Transit: {result['duration_text']} ({result['distance_text']})")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_distance_api()
