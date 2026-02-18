"""Shared LLM utilities."""

import importlib.util
import json
import os
from pathlib import Path
from google import genai

_client = None


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()


def _load_api_key_from_config() -> str | None:
    """Load Gemini API key from config file."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
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


def generate_content(prompt: str, max_retries: int | None = None) -> str:
    """Generate content using Gemini with retry logic for transient errors."""
    import time

    client = get_gemini_model()
    retry_limit = max_retries if max_retries is not None else int(getattr(_settings, "GEMINI_MAX_RETRIES", 3))
    retry_limit = max(1, int(retry_limit))
    model_name = str(getattr(_settings, "GEMINI_MODEL", "gemini-3-flash-preview"))
    temperature = float(getattr(_settings, "GEMINI_TEMPERATURE", 0.0))
    seed = int(getattr(_settings, "GEMINI_SEED", 20260213))
    initial_delay = float(getattr(_settings, "GEMINI_RETRY_INITIAL_DELAY_SEC", 1.0))
    backoff_multiplier = float(getattr(_settings, "GEMINI_RETRY_BACKOFF_MULTIPLIER", 2.0))
    max_retry_delay = float(getattr(_settings, "GEMINI_RETRY_MAX_DELAY_SEC", 8.0))
    retryable_tokens = (
        "503",
        "overloaded",
        "unavailable",
        "timeout",
        "timed out",
        "deadline exceeded",
        "429",
        "rate limit",
        "resource exhausted",
    )

    for attempt in range(retry_limit):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "seed": seed,
                },
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            lowered = error_str.lower()
            # Retry on transient provider/load/network failures.
            if any(token in lowered for token in retryable_tokens):
                if attempt < retry_limit - 1:
                    wait_time = initial_delay * (backoff_multiplier ** attempt)
                    wait_time = min(wait_time, max_retry_delay)
                    print(f"    Gemini transient error, retrying in {wait_time:.1f}s...", flush=True)
                    time.sleep(wait_time)
                    continue
            raise

    raise Exception("Max retries exceeded")
