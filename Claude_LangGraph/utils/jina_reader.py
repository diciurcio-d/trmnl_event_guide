"""Jina Reader API for converting webpages to LLM-friendly text."""

import json
import importlib.util
import time
import requests
from pathlib import Path


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()
_last_request_time = 0


def _get_api_key() -> str:
    """Load Jina API key from config."""
    config_path = Path(__file__).parent.parent / "concert_finder" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config["jina"]["api_key"]


def _rate_limit():
    """Enforce rate limiting between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    delay = _settings.JINA_REQUEST_DELAY
    if elapsed < delay:
        sleep_time = delay - elapsed
        print(f"    Rate limiting: sleeping {sleep_time:.1f}s...", flush=True)
        time.sleep(sleep_time)
    _last_request_time = time.time()


def fetch_page_text_jina(url: str) -> str:
    """
    Fetch page content using Jina Reader API.

    Converts any webpage to clean, LLM-friendly markdown text.
    Automatically handles JavaScript rendering, removes ads/navigation.

    Args:
        url: The webpage URL to fetch

    Returns:
        Clean markdown text of the page content

    Raises:
        Exception if the request fails
    """
    _rate_limit()

    api_key = _get_api_key()

    # Jina Reader API endpoint
    jina_url = f"https://r.jina.ai/{url}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/plain",
    }

    response = requests.get(jina_url, headers=headers, timeout=60)
    response.raise_for_status()

    return response.text


def test_jina_reader(url: str = "https://www.caveat.nyc") -> None:
    """Test the Jina reader with a sample URL."""
    print(f"Testing Jina Reader with: {url}")
    print("-" * 50)

    try:
        text = fetch_page_text_jina(url)
        print(f"Success! Got {len(text)} characters")
        print("\nFirst 1000 characters:")
        print(text[:1000])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.strandbooks.com/events.html"
    test_jina_reader(url)
