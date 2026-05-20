"""Custom scraper for the 92nd Street Y (92ny.org).

Fetches event listings directly from 92NY's public Algolia search API index.
Bypasses Tessitura session redirects and other anti-scraping measures.
"""

import re
import html as html_lib
import requests
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

def fetch_92ny_events(venue_name: str = "92nd Street Y") -> list[dict]:
    """
    Fetch upcoming events for 92nd Street Y directly from Algolia.
    
    Args:
        venue_name: Canonical name of the venue to set in normalized events.
        
    Returns:
        List of normalized event dictionaries.
    """
    app_id = "YJQESJQHDI"
    api_key = "6d28dbd16e5728d3fe7fb4c1d80095f0"
    index_name = "92NY_events_prod"

    url = f"https://{app_id}-dsn.algolia.net/1/indexes/{index_name}/query"

    headers = {
        "X-Algolia-API-Key": api_key,
        "X-Algolia-Application-Id": app_id,
        "Content-Type": "application/json"
    }

    # Fetch up to 250 items to get all current events
    payload = {
        "params": "query=&hitsPerPage=250&filters=ObjectType:Event"
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"      [92NY Scraper] Algolia API returned status {resp.status_code}: {resp.text}")
            return []
        
        data = resp.json()
        hits = data.get("hits", [])
        print(f"      [92NY Scraper] Found {len(hits)} total events in Algolia index.")
        
        events = []
        now = datetime.now(timezone.utc)
        
        for hit in hits:
            # 1. Clean and unescape Title
            title_html = hit.get("Title") or hit.get("title") or ""
            name = re.sub(r'<[^>]+>', '', title_html)
            name = html_lib.unescape(name).strip()
            
            # 2. Date and Time parsing
            first_date_ts = hit.get("FirstDate")
            if not first_date_ts:
                continue
                
            try:
                # FirstDate is a Unix timestamp in local NY time represented as if it were UTC.
                # Interpret as UTC first, then replace timezone metadata to America/New_York local event time.
                dt_utc = datetime.fromtimestamp(first_date_ts, tz=timezone.utc)
                dt_local = dt_utc.replace(tzinfo=ZoneInfo("America/New_York"))
            except Exception as e:
                print(f"      [92NY Scraper] Error parsing timestamp {first_date_ts}: {e}")
                continue
                
            # Filter for upcoming events only
            if dt_local.astimezone(timezone.utc) < now:
                continue
                
            # Standard YYYY-MM-DD string
            date_str = dt_local.strftime("%Y-%m-%d")
            
            # 3. URL Construction
            relative_url = hit.get("URL") or hit.get("url") or ""
            if not relative_url:
                continue
            if relative_url.startswith("http"):
                full_url = relative_url
            else:
                full_url = urljoin("https://www.92ny.org", relative_url)
                
            # 4. Description Cleaning
            short_desc = hit.get("ShortDesc") or hit.get("description") or ""
            description = re.sub(r'<[^>]+>', '', short_desc)
            description = html_lib.unescape(description).strip()
            
            # 5. Pricing Extraction
            lowest_price = str(hit.get("LowestPrice") or "").strip()
            if not lowest_price and hit.get("InSessionClasses"):
                for item in hit.get("InSessionClasses", []):
                    p = str(item.get("price") or "").strip()
                    if p:
                        lowest_price = p
                        break
                        
            is_free = False
            if lowest_price:
                lowest_price_lower = lowest_price.lower()
                if "free" in lowest_price_lower or lowest_price_lower == "$0" or lowest_price_lower == "$0.00":
                    is_free = True
            else:
                lowest_price = ""
                
            # Create standard schema event dictionary
            event_dict = {
                "name": name,
                "datetime": dt_local,
                "date_str": date_str,
                "venue_name": venue_name,
                "address": "1395 Lexington Ave, New York, NY 10128",
                "event_type": "Talks/Concerts/Art",
                "url": full_url,
                "source": "92ny_algolia",
                "matched_artist": "",
                "travel_minutes": None,
                "description": description,
                "is_free": is_free,
                "price": lowest_price,
                "event_source_url": full_url,
                "extraction_method": "92ny_algolia",
                "relevance_score": None,
                "validation_confidence": 1.0,
            }
            events.append(event_dict)
            
        print(f"      [92NY Scraper] Parsed and filtered {len(events)} upcoming events.")
        return events
        
    except Exception as e:
        print(f"      [92NY Scraper] Exception during 92nd Street Y Algolia fetch: {e}")
        return []
