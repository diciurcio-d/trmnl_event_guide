"""Push events to TRMNL via webhook API."""

import importlib.util
import json
import os
import requests
from pathlib import Path

from event_selector import get_trmnl_events


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()

# Configuration
TRMNL_WEBHOOK_URL = os.environ.get("TRMNL_WEBHOOK_URL")
TEMPLATE_PATH = Path(__file__).parent / "template.html"


def get_markup() -> str:
    """
    Get the TRMNL markup template.

    Returns the Liquid template that TRMNL will render with merge variables.
    """
    return '''<div class="layout layout--col gap--space-between">
  <div class="grid grid--cols-3">
    <div class="row">
      {% for event in events %}
      <div class="item col--span-1">
        <div class="content">
          <div class="row">
            <span class="label label--underline">{{ event.day }}</span>
            <span class="label">{{ event.date }}</span>
            {% if event.time %}
              <span class="value value--small">{{ event.time }}</span>
            {% else %}
              <span class="label label--small">See site</span>
            {% endif %}
          </div>
          <span class="title title--small">{{ event.name }}</span>
          <div class="divider"></div>
          {% if event.description %}
            <span class="label label--small">{{ event.description }}</span>
          {% endif %}
          <span class="label label--small label--gray">{{ event.source }}</span>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
</div>
<div class="title_bar">
  <img class="image" src="https://usetrmnl.com/images/plugins/trmnl--render.svg" />
  <span class="title">NYC Events</span>
</div>'''


def push_to_trmnl(webhook_url: str | None = None) -> dict:
    """
    Push current events to TRMNL via webhook.

    Args:
        webhook_url: TRMNL webhook URL. If None, uses TRMNL_WEBHOOK_URL env var.

    Returns:
        Response dict with status and message
    """
    url = webhook_url or TRMNL_WEBHOOK_URL

    if not url:
        return {
            "success": False,
            "error": "No webhook URL. Set TRMNL_WEBHOOK_URL env var or pass webhook_url.",
        }

    # Get events
    events = get_trmnl_events(count=6, days=6)

    if not events:
        return {
            "success": False,
            "error": "No events found in the next 6 days.",
        }

    # Build payload
    payload = {
        "merge_variables": {
            "events": events,
        }
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=int(_settings.TRMNL_WEBHOOK_TIMEOUT_SEC),
        )

        if response.status_code == 200:
            return {
                "success": True,
                "message": f"Pushed {len(events)} events to TRMNL",
                "events": events,
            }
        else:
            return {
                "success": False,
                "error": f"TRMNL API error: {response.status_code} - {response.text}",
            }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
        }


def print_events(events: list[dict]):
    """Print events in a readable format."""
    print(f"\nSelected {len(events)} events:\n")
    for i, e in enumerate(events, 1):
        print(f"  {i}. {e['day']} {e['date']} @ {e['time']}")
        print(f"     {e['name']}")
        print(f"     ({e['source']})\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--push":
        # Push to TRMNL
        url = sys.argv[2] if len(sys.argv) > 2 else None
        result = push_to_trmnl(url)

        if result["success"]:
            print(f"Success: {result['message']}")
            print_events(result["events"])
        else:
            print(f"Error: {result['error']}")
            sys.exit(1)
    else:
        # Just show events
        events = get_trmnl_events(count=6, days=6)
        print_events(events)

        print("\nTo push to TRMNL:")
        print("  python trmnl_push.py --push <webhook_url>")
        print("  or set TRMNL_WEBHOOK_URL environment variable")
