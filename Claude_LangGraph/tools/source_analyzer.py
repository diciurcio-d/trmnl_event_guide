"""Tool to analyze a URL and create a source configuration."""

import re
import json
import requests
from urllib.parse import urlparse
from langchain_core.tools import tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import generate_content


SOURCE_CONFIG_PROMPT = """Analyze this webpage content and create a configuration for scraping events from it.

URL: {url}
Domain: {domain}

Based on the page content, determine:
1. A short, friendly name for this source (e.g., "Brooklyn Museum", "NYC Parks")
2. The default location for events (address or venue name)
3. The default event type (e.g., "Museum Event", "Concert", "Workshop")
4. Parsing hints that describe:
   - How dates are formatted on this page
   - How times are shown
   - Any special markers (like "SOLD OUT")
   - The structure of event listings

Return ONLY valid JSON with this structure:
{{
  "name": "Source Name",
  "default_location": "Venue Address",
  "default_event_type": "Event Type",
  "parsing_hints": "Detailed hints about date/time formats and page structure"
}}

PAGE CONTENT (first 8000 chars):
{content}
"""


@tool
def analyze_source(url: str) -> dict:
    """
    Analyze a URL and generate a complete source configuration.

    This tool checks the URL, determines the scraping method (API or Playwright),
    and uses an LLM to generate parsing hints for the generic scraper.

    Args:
        url: The URL to analyze

    Returns:
        dict with complete source configuration ready to add to sources list
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    result = {
        "url": url,
        "method": "playwright",  # Default to playwright as fallback
        "enabled": True,
        "wait_seconds": 3,
        "cloudflare_protected": False,
        "reason": "",
        "status_code": None,
        "content_type": None,
    }

    try:
        # First, try a simple GET request
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        result["status_code"] = response.status_code
        result["content_type"] = response.headers.get("Content-Type", "")
        content = response.text

        # Check for JSON API
        if "application/json" in result["content_type"]:
            result["method"] = "api"
            result["reason"] = "URL returns JSON data directly - can use requests library"

            # Try to determine the events path
            try:
                data = response.json()
                if isinstance(data, dict):
                    for key in ["events", "data", "items", "results"]:
                        if key in data and isinstance(data[key], list):
                            result["api_events_path"] = key
                            break
            except:
                pass
        else:
            # Check for common API patterns in URL
            api_patterns = ["/api/", "/wp-json/", "/rest/", "/v1/", "/v2/", ".json"]
            if any(pattern in url.lower() for pattern in api_patterns):
                try:
                    response.json()
                    result["method"] = "api"
                    result["reason"] = "URL appears to be an API endpoint and returns valid JSON"
                except:
                    pass

        # Check for bot protection indicators
        if response.status_code == 403:
            result["method"] = "playwright"
            result["cloudflare_protected"] = True
            result["wait_seconds"] = 5
            result["reason"] = "Site returned 403 Forbidden - likely has bot protection"

        # Check for Cloudflare or other challenges
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in [
            "cloudflare",
            "checking your browser",
            "just a moment",
            "challenge-platform",
        ]):
            result["method"] = "playwright"
            result["cloudflare_protected"] = True
            result["wait_seconds"] = 5
            result["reason"] = "Site has Cloudflare or similar bot protection"

        # Now use LLM to generate source configuration
        print(f"    Analyzing page structure with LLM...", flush=True)

        prompt = SOURCE_CONFIG_PROMPT.format(
            url=url,
            domain=domain,
            content=content[:8000],
        )

        response_text = generate_content(prompt).strip()

        # Parse the LLM response
        try:
            config = json.loads(response_text)
            result["name"] = config.get("name", domain)
            result["default_location"] = config.get("default_location", "See event details")
            result["default_event_type"] = config.get("default_event_type", "Event")
            result["parsing_hints"] = config.get("parsing_hints", "Extract event names, dates, times, and locations.")
        except json.JSONDecodeError:
            # Try to find JSON in response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    config = json.loads(match.group())
                    result["name"] = config.get("name", domain)
                    result["default_location"] = config.get("default_location", "See event details")
                    result["default_event_type"] = config.get("default_event_type", "Event")
                    result["parsing_hints"] = config.get("parsing_hints", "Extract event names, dates, times, and locations.")
                except:
                    # Fallback to defaults
                    result["name"] = domain.replace("www.", "").split(".")[0].title()
                    result["default_location"] = "See event details"
                    result["default_event_type"] = "Event"
                    result["parsing_hints"] = "Extract event names, dates, times, and locations."
            else:
                result["name"] = domain.replace("www.", "").split(".")[0].title()
                result["default_location"] = "See event details"
                result["default_event_type"] = "Event"
                result["parsing_hints"] = "Extract event names, dates, times, and locations."

        if not result["reason"]:
            result["reason"] = "Standard webpage - using Playwright for reliable extraction"

        return result

    except requests.exceptions.Timeout:
        result["reason"] = "Request timed out - site may be slow, using Playwright"
        result["name"] = domain.replace("www.", "").split(".")[0].title()
        result["default_location"] = "See event details"
        result["default_event_type"] = "Event"
        result["parsing_hints"] = "Extract event names, dates, times, and locations."
        return result

    except requests.exceptions.RequestException as e:
        result["reason"] = f"Request failed ({str(e)}) - using Playwright as fallback"
        result["name"] = domain.replace("www.", "").split(".")[0].title()
        result["default_location"] = "See event details"
        result["default_event_type"] = "Event"
        result["parsing_hints"] = "Extract event names, dates, times, and locations."
        return result
