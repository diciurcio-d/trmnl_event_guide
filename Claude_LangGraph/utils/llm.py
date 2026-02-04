"""Shared LLM utilities."""

import json
import os
from pathlib import Path
from google import genai

_client = None


def _load_api_key_from_config() -> str | None:
    """Load Gemini API key from config file."""
    config_path = Path(__file__).parent.parent / "concert_finder" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("gemini", {}).get("api_key")
    return None


def get_gemini_model():
    """Get Gemini client (singleton to avoid repeated setup)."""
    global _client

    if _client is not None:
        return _client

    # Try config file first, then environment variables
    api_key = (
        _load_api_key_from_config()
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Add to config.json or set GOOGLE_API_KEY environment variable."
        )

    _client = genai.Client(api_key=api_key)
    return _client


def generate_content(prompt: str, max_retries: int = 3) -> str:
    """Generate content using Gemini with retry logic for transient errors."""
    import time

    client = get_gemini_model()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            # Retry on overloaded/unavailable errors
            if "503" in error_str or "overloaded" in error_str.lower() or "unavailable" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"    Model overloaded, retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
            raise

    raise Exception("Max retries exceeded")
