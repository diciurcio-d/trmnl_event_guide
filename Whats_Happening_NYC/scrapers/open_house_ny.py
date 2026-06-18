"""Custom scraper for Open House New York (ohny.org).

Fetches tour and event listings directly from OHNY's public Builder.io content API.
Bypasses slow browser engines and heavy client-side Javascript execution.
"""

import re
import html as html_lib
import requests
from datetime import datetime
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

def parse_ohny_date(date_str: str) -> datetime | None:
    """Parse various OHNY date formats into timezone-aware New York datetime."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    ny_tz = ZoneInfo("America/New_York")
    
    # Format 1: Javascript Date string e.g. "Sat Jun 20 2026 12:00:00 GMT-0400 (Eastern Daylight Time)"
    if "GMT" in date_str:
        prefix = date_str.split(" GMT")[0].strip()
        try:
            dt = datetime.strptime(prefix, "%a %b %d %Y %H:%M:%S")
            return dt.replace(tzinfo=ny_tz)
        except ValueError:
            pass
            
    # Format 2: Locale string e.g. "11/17/2025, 6:30 PM"
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y, %I:%M %p")
        return dt.replace(tzinfo=ny_tz)
    except ValueError:
        pass

    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y, %H:%M")
        return dt.replace(tzinfo=ny_tz)
    except ValueError:
        pass

    # Try standard ISO or other fallback formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[:19], fmt)
            return dt.replace(tzinfo=ny_tz)
        except ValueError:
            continue
            
    return None

def fetch_open_house_ny_events(venue_name: str = "Open House NY") -> list[dict]:
    """
    Fetch upcoming events for Open House New York directly from Builder.io CDN API.
    
    Args:
        venue_name: Canonical name of the venue to set in normalized events.
        
    Returns:
        List of normalized event dictionaries.
    """
    api_key = "fc1101199ae44b6eb081e8c3d1d68c26"
    url = f"https://cdn.builder.io/api/v3/content/activity?apiKey={api_key}&limit=100"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    try:
        print(f"      [OHNY Scraper] Fetching events from Builder.io Content API...")
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"      [OHNY Scraper] API returned status {res.status_code}: {res.text[:200]}")
            return []
            
        payload = res.json()
        results = payload.get("results", [])
        print(f"      [OHNY Scraper] Found {len(results)} total activities in Builder.io payload.")
        
        events = []
        ny_tz = ZoneInfo("America/New_York")
        now = datetime.now(ny_tz)
        
        for item in results:
            data = item.get("data", {})
            if not data:
                continue
                
            name = data.get("title") or item.get("name") or ""
            name = str(name).strip()
            if not name:
                continue
                
            # Date/Time parsing
            event_date_str = data.get("eventDate")
            if not event_date_str:
                continue
                
            dt_local = parse_ohny_date(event_date_str)
            if not dt_local:
                # If we can't parse the date, skip it
                continue
                
            # Filter for upcoming events only
            if dt_local < now:
                continue
                
            date_str = dt_local.strftime("%Y-%m-%d")
            
            # URL Construction
            url_path = data.get("url") or (f"/activity/{data.get('slug')}" if data.get("slug") else "")
            url_path = str(url_path).strip()
            if url_path:
                full_url = urljoin("https://ohny.org", url_path)
            else:
                full_url = "https://ohny.org/calendar/"
                
            # Location/Address Construction
            borough = str(data.get("borough") or "").strip()
            neighborhood = str(data.get("neighborhood") or "").strip()
            
            location_parts = []
            if neighborhood and neighborhood.lower() != "none":
                location_parts.append(neighborhood)
            if borough and borough.lower() != "none":
                location_parts.append(borough)
                
            location = ", ".join(location_parts) if location_parts else "Various NYC Locations"
            
            # Description Cleaning
            excerpt = str(data.get("excerpt") or "").strip()
            description = re.sub(r"<[^>]+>", " ", excerpt)
            description = html_lib.unescape(description)
            description = re.sub(r"\s+", " ", description).strip()
            # Truncate description slightly for readability if needed
            if len(description) > 300:
                description = description[:297] + "..."
                
            # Event type / Format
            format_val = data.get("format")
            if isinstance(format_val, list):
                event_type = ", ".join(str(f).strip() for f in format_val if f)
            else:
                event_type = str(format_val or "").strip()
            
            if not event_type:
                event_type = "Architecture Tour"
            
            # Smart pricing / free heuristics
            title_lower = name.lower()
            desc_lower = description.lower()
            is_free = True
            price = ""
            if any(k in title_lower or k in desc_lower for k in ["benefit", "gala", "members only", "members-only", "ticketed", "admission"]):
                is_free = False
                
            # Create standard schema event dictionary
            event_dict = {
                "name": name[:70],
                "datetime": dt_local,
                "date_str": date_str,
                "venue_name": venue_name,
                "address": location,  # In generic scraper, 'location' and 'address' are used
                "event_type": event_type,
                "url": full_url,
                "source": "open_house_ny",
                "matched_artist": "",
                "travel_minutes": None,
                "description": description[:100],  # Keep standard brief description
                "is_free": is_free,
                "price": price,
                "event_source_url": full_url,
                "extraction_method": "ohny_builder_api",
                "relevance_score": None,
                "validation_confidence": 1.0,
            }
            events.append(event_dict)
            
        # De-duplicate events by URL, date_str, and normalized name
        seen_keys = set()
        unique_events = []
        for ev in events:
            key = (ev["url"], ev["date_str"], ev["name"].lower())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_events.append(ev)
                
        print(f"      [OHNY Scraper] Parsed and filtered {len(unique_events)} upcoming unique events.")
        return unique_events
        
    except Exception as e:
        print(f"      [OHNY Scraper] Exception during Open House NY fetch: {e}")
        return []
