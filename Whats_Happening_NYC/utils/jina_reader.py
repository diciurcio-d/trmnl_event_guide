"""Jina Reader API for converting webpages to LLM-friendly text."""

import json
import importlib.util
import re
import time
import os
import requests
from pathlib import Path
from urllib.parse import urlparse


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
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config["jina"]["api_key"]


def _rate_limit():
    """Enforce rate limiting between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    delay = float(os.getenv("JINA_REQUEST_DELAY_SEC", str(_settings.JINA_REQUEST_DELAY)))
    if elapsed < delay:
        sleep_time = delay - elapsed
        print(f"    Rate limiting: sleeping {sleep_time:.1f}s...", flush=True)
        time.sleep(sleep_time)
    _last_request_time = time.time()


def fetch_page_text_jina(url: str, timeout: int | None = None) -> str:
    """
    Fetch page content using Jina Reader API.

    Converts any webpage to clean, LLM-friendly markdown text.
    Automatically handles JavaScript rendering, removes ads/navigation.

    Args:
        url: The webpage URL to fetch
        timeout: Request timeout in seconds (defaults to settings.JINA_TIMEOUT_SEC)

    Returns:
        Clean markdown text of the page content

    Raises:
        Exception if the request fails
    """
    _rate_limit()

    api_key = _get_api_key()

    normalized = _normalize_web_url(url)
    candidates = [normalized]
    # Fallback: try https if input was http
    if normalized.startswith("http://"):
        candidates.append("https://" + normalized[len("http://"):])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/plain",
    }

    default_timeout = int(_settings.JINA_TIMEOUT_SEC)
    request_timeout = (
        timeout
        if timeout and timeout > 0
        else int(os.getenv("JINA_TIMEOUT_SEC", str(default_timeout)))
    )

    last_error = None
    for candidate in candidates:
        jina_url = f"https://r.jina.ai/{candidate}"
        try:
            response = requests.get(jina_url, headers=headers, timeout=request_timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            last_error = e
            continue

    raise Exception(f"Jina fetch failed for {url}: {last_error}")


def _normalize_web_url(url: str) -> str:
    """Normalize URL to reduce fetch failures across readers/providers."""
    if not url:
        return ""

    url = url.strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""

    rebuilt = f"{scheme}://{netloc}{path}"
    if query:
        rebuilt += f"?{query}"
    return rebuilt


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
