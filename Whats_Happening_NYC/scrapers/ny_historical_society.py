"""Custom scraper for the New-York Historical Society (nyhistory.org).

Queries the museum's public Prismic CMS search API directly.
Fetches up to 100 structured event listings in a single, fast request.
"""

import re
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def fetch_ny_historical_society_events(venue_name: str = "NY Historical Society") -> list[dict]:
    """
    Fetch upcoming events for the New-York Historical Society directly from their Prismic CMS API.
    
    Args:
        venue_name: Canonical name of the venue to set in normalized events.
        
    Returns:
        List of normalized event dictionaries.
    """
    api_base = "https://nyhs-prod.cdn.prismic.io/api/v2"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    try:
        # Step 1: Fetch master database reference (ref)
        print("      [NYHS Scraper] Fetching Prismic API metadata for active ref...")
        meta_res = requests.get(api_base, headers=headers, timeout=12)
        if meta_res.status_code != 200:
            print(f"      [NYHS Scraper] Metadata API returned status {meta_res.status_code}")
            return []
            
        api_meta = meta_res.json()
        master_ref = None
        for r in api_meta.get("refs", []):
            if r.get("isMasterRef"):
                master_ref = r.get("ref")
                break
        if not master_ref and api_meta.get("refs"):
            master_ref = api_meta["refs"][0].get("ref")
            
        if not master_ref:
            print("      [NYHS Scraper] Could not find active database reference (ref).")
            return []
            
        # Step 2: Query for documents of type 'program'
        query_url = f"{api_base}/documents/search?ref={master_ref}&q=[[at(document.type,%20%22program%22)]]&pageSize=100"
        print(f"      [NYHS Scraper] Querying structured programs from Prismic...")
        res = requests.get(query_url, headers=headers, timeout=12)
        if res.status_code != 200:
            print(f"      [NYHS Scraper] Query API returned status {res.status_code}")
            return []
            
        payload = res.json()
        results = payload.get("results", [])
        print(f"      [NYHS Scraper] Found {len(results)} total program documents in payload.")
        
        events = []
        ny_tz = ZoneInfo("America/New_York")
        now = datetime.now(ny_tz)
        
        for doc in results:
            uid = doc.get("uid")
            slugs = doc.get("slugs", [""])
            slug = uid or slugs[0]
            
            data = doc.get("data", {})
            if not data:
                continue
                
            # Title Parsing
            title_list = data.get("title", [])
            title = title_list[0].get("text", "").strip() if title_list else ""
            if not title:
                title = str(data.get("meta_title") or "").strip()
            if not title:
                title = slug.replace("-", " ").title() if slug else "Unknown Event"
                
            # Event Date and Time Parsing
            dt_str = data.get("event_main_datetime")
            if not dt_str:
                continue
                
            try:
                # ISO offset timezone format: "2026-06-02T18:00:00+0000"
                dt_utc = datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                dt_local = dt_utc.astimezone(ny_tz)
            except Exception:
                continue
                
            # Filter for upcoming events only
            if dt_local < now:
                continue
                
            date_str = dt_local.strftime("%Y-%m-%d")
            
            # URL Construction
            url_path = data.get("external_page_url")
            if url_path:
                full_url = str(url_path).strip()
            else:
                full_url = f"https://www.nyhistory.org/programs/{slug}"
                
            # Location/Address Parsing
            loc_list = data.get("location", [])
            location = loc_list[0].get("text", "").strip() if loc_list else ""
            if not location:
                location = "NY Historical Society, 170 Central Park West, New York, NY 10024"
                
            # Description Parsing
            desc_list = data.get("excerpt_rich", [])
            if not desc_list:
                desc_list = data.get("event_details", [])
            description = desc_list[0].get("text", "").strip() if desc_list else ""
            if not description:
                description = str(data.get("meta_description") or "").strip()
            # Clean up spacing
            description = re.sub(r"\s+", " ", description)
            if len(description) > 300:
                description = description[:297] + "..."
                
            # Price / Free Extraction Heuristics
            price_str = str(data.get("price") or "").strip()
            is_paid = data.get("is_paid_ticketing", True)
            is_free = not is_paid
            if "free" in price_str.lower():
                is_free = True
                
            # Map to canonical schema
            event_dict = {
                "name": title[:70],
                "datetime": dt_local,
                "date_str": date_str,
                "venue_name": venue_name,
                "address": location,
                "event_type": "Talk/Lecture",
                "url": full_url,
                "source": "nyhistory_scraper",
                "matched_artist": "",
                "travel_minutes": None,
                "description": description[:100],
                "is_free": is_free,
                "price": price_str,
                "event_source_url": full_url,
                "extraction_method": "nyhistory_prismic_api",
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
                
        # Sort chronologically
        unique_events.sort(key=lambda x: x["datetime"])
        
        print(f"      [NYHS Scraper] Parsed and filtered {len(unique_events)} upcoming unique events.")
        return unique_events
        
    except Exception as e:
        print(f"      [NYHS Scraper] Exception during NY Historical Society fetch: {e}")
        return []
