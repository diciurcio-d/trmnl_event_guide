"""Configuration loader for concert finder."""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.json"


def load_config() -> dict:
    """Load configuration from config.json or environment variables."""
    config = {"ticketmaster_api_key": None}

    if CONFIG_PATH.exists():
        try:
            file_config = json.loads(CONFIG_PATH.read_text())
            if "ticketmaster" in file_config:
                config["ticketmaster_api_key"] = file_config["ticketmaster"].get("api_key")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error reading config.json: {e}")

    # Environment variable overrides config file
    if os.environ.get("TICKETMASTER_API_KEY"):
        config["ticketmaster_api_key"] = os.environ["TICKETMASTER_API_KEY"]

    return config


def get_ticketmaster_key() -> str | None:
    """Get Ticketmaster API key."""
    return load_config()["ticketmaster_api_key"]
