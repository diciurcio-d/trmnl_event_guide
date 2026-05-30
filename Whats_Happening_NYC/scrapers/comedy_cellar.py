"""Custom scraper for the Comedy Cellar (comedycellar.com) and associated rooms.

Fetches show listings directly from the Comedy Cellar's public lineup REST API,
parses showtimes, locations, comedian lineups with bios, deep reservation links,
and dynamically routes them in-memory to their corresponding separate database venues.
"""

import re
import time
import random
import json
import requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

def fetch_comedy_cellar_events(venue_name: str) -> list[dict]:
    """
    Fetch upcoming events for Comedy Cellar, Village Underground, or Fat Black Pussycat.
    Queries the central New York lineup REST API, maps shows to correct venues in-memory,
    and returns a filtered list of unique upcoming events.
    
    Args:
        venue_name: Requested venue name to filter for (e.g. 'Comedy Cellar', 'Village Underground', etc.)
        
    Returns:
        List of normalized event dictionaries.
    """
    all_events = []
    ny_tz = ZoneInfo("America/New_York")
    now_local = datetime.now(ny_tz)
    now_utc = datetime.now(timezone.utc)
    
    url = "https://www.comedycellar.com/lineup/api/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    
    # 1. Determine which target venue name was requested to filter results at the end
    is_village_request = "village" in venue_name.lower()
    is_pussycat_request = "fat black" in venue_name.lower() or "pussycat" in venue_name.lower()
    
    if is_village_request:
        canonical_target_venue = "Village Underground"
    elif is_pussycat_request:
        canonical_target_venue = "Fat Black Pussycat"
    else:
        canonical_target_venue = "Comedy Cellar"
        
    print(f"      [Comedy Cellar Scraper] Fetching unified lineups. Will filter for '{canonical_target_venue}'")
    
    # 2. Query today's payload first to discover all upcoming valid date strings
    payload = {
        "date": "today",
        "venue": "newyork",
        "type": "lineup"
    }
    body = f"action=cc_get_shows&json={json.dumps(payload)}"
    
    try:
        r = requests.post(url, headers=headers, data=body, timeout=15)
        if r.status_code != 200:
            print(f"      [Comedy Cellar Scraper] API initial lookup failed with status: {r.status_code}")
            return []
            
        res_json = r.json()
        dates_dict = res_json.get("dates", {})
        if not dates_dict:
            print("      [Comedy Cellar Scraper] No dates found in initial API response.")
            return []
            
        # Chronological sort of the date keys (which are YYYY-MM-DD strings)
        date_keys = sorted(list(dates_dict.keys()))
        print(f"      [Comedy Cellar Scraper] Discovered {len(date_keys)} upcoming show dates.")
        
        # 3. Iterate over the dates to parse show lineups (limiting to up to 32 days, though API usually has 28)
        for idx, date_str in enumerate(date_keys):
            # Parse date string to verify
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
                
            # Skip past dates (just in case)
            if event_date < now_local.date():
                continue
                
            # Polite randomized delay between queries
            if idx > 0:
                delay = random.uniform(2.0, 4.5)
                print(f"      [Comedy Cellar Scraper] Sleeping {delay:.2f} seconds to remain polite...")
                time.sleep(delay)
                
            print(f"      [Comedy Cellar Scraper] Fetching shows for {date_str} ({idx+1}/{len(date_keys)})...")
            
            # Fetch lineup for this date
            date_payload = {
                "date": date_str,
                "venue": "newyork",
                "type": "lineup"
            }
            date_body = f"action=cc_get_shows&json={json.dumps(date_payload)}"
            
            try:
                date_resp = requests.post(url, headers=headers, data=date_body, timeout=15)
                if date_resp.status_code != 200:
                    print(f"      [Comedy Cellar Scraper] Failed to fetch shows for {date_str}: status {date_resp.status_code}")
                    continue
                    
                date_data = date_resp.json()
                show_html = date_data.get("show", {}).get("html", "")
                if not show_html:
                    continue
                    
                soup = BeautifulSoup(show_html, "html.parser")
                
                # Locate all set headers representing shows
                headers_el = soup.find_all(class_="set-header")
                
                for h in headers_el:
                    show_div = h.parent
                    if not show_div:
                        continue
                        
                    # A. Showtime Parsing
                    time_el = h.find(class_="bold")
                    if not time_el:
                        continue
                    # Strip any mobile class text or 'show' keyword
                    time_text = re.sub(r'\s*show\s*', '', time_el.get_text(), flags=re.IGNORECASE).strip()
                    
                    # Parse time like "7:00 pm" or "11:30 pm"
                    time_match = re.match(r"^(\d+):(\d+)\s*(am|pm)$", time_text.lower())
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2))
                        am_pm = time_match.group(3)
                        if am_pm == "pm" and hour != 12:
                            hour += 12
                        elif am_pm == "am" and hour == 12:
                            hour = 0
                    else:
                        # Fallback default time
                        hour, minute = 19, 0
                        
                    # Create timezone-aware datetime
                    dt_local = datetime(
                        event_date.year, event_date.month, event_date.day,
                        hour=hour, minute=minute, tzinfo=ny_tz
                    )
                    
                    # Skip past shows
                    if dt_local.astimezone(timezone.utc) < now_utc:
                        continue
                        
                    # B. Show Title
                    title_el = h.find(class_="title")
                    title_text = title_el.get_text().strip() if title_el else ""
                    
                    # C. In-Memory Room Routing & Physical Details
                    title_lower = title_text.lower()
                    if "village underground" in title_lower:
                        event_venue = "Village Underground"
                        event_address = "130 W 3rd St, New York, NY 10012, USA"
                    elif any(k in title_lower for k in ["fat black pussycat", "fbpc", "hot soup"]):
                        event_venue = "Fat Black Pussycat"
                        event_address = "130 W 3rd St, New York, NY 10012, USA"
                    else:
                        event_venue = "Comedy Cellar"
                        event_address = "117 MacDougal St, New York, NY 10012, USA"
                        
                    # Premium cleanup: convert room names/generic titles into beautiful descriptive show names
                    event_name = title_text
                    if event_name in ["MacDougal Street", "Village Underground", "Fat Black Pussycat (Bar)", "Fat Black Pussycat (Lounge)", "Fat Black Pussycat"]:
                        event_name = f"{event_venue} Lineup"
                        
                    # D. Lineup & Biography parsing
                    lineup_div = show_div.find(class_="lineup")
                    comedians = []
                    href = "/new-york-line-up/" # Default fallback deep link
                    
                    if lineup_div:
                        # Reservation link
                        reserve_el = lineup_div.find(class_="make-reservation")
                        a_tag = reserve_el.find("a") if reserve_el else None
                        if a_tag:
                            href_val = a_tag.get("href", "")
                            if href_val:
                                href = href_val
                                
                        # Extract comedians
                        set_contents = lineup_div.find_all(class_="set-content")
                        for sc in set_contents:
                            name_el = sc.find(class_="name")
                            if name_el:
                                name = name_el.get_text().strip()
                                sc_text = sc.get_text().strip()
                                # Subtract the comedian name to get the bio
                                bio = sc_text.replace(name, "").strip()
                                # Clean up formatting and website links
                                bio = bio.replace("> Website", "").strip()
                                bio = re.sub(r'\s+', ' ', bio)
                                comedians.append(f"{name} ({bio})" if bio else name)
                                
                    full_url = href
                    if not full_url.startswith("http"):
                        full_url = f"https://www.comedycellar.com{full_url}"
                        
                    # Create description text
                    if comedians:
                        rich_description = f"Room: {title_text}. Stand-up line-up: " + ", ".join(comedians)
                    else:
                        rich_description = f"Show at {event_venue} - {title_text}."
                        
                    # E. Dynamic Pricing Estimates
                    # Sunday (6) through Thursday (3). Friday (4) and Saturday (5).
                    if dt_local.weekday() in [4, 5]:
                        price_str = "$20 - $25 cover + 2-drink min"
                    else:
                        price_str = "$14 - $20 cover + 2-drink min"
                        
                    all_events.append({
                        "name": event_name,
                        "datetime": dt_local,
                        "date_str": date_str,
                        "venue_name": event_venue,
                        "address": event_address,
                        "event_type": "comedy",
                        "url": full_url,
                        "source": "comedy_cellar_scraper",
                        "matched_artist": "",
                        "travel_minutes": None,
                        "description": rich_description,
                        "is_free": False,
                        "price": price_str,
                        "event_source_url": full_url,
                        "extraction_method": "comedy_cellar_scraper",
                        "relevance_score": None,
                        "validation_confidence": 1.0,
                    })
            except Exception as date_err:
                print(f"      [Comedy Cellar Scraper] Error parsing date {date_str}: {date_err}")
                continue
                
    except Exception as e:
        print(f"      [Comedy Cellar Scraper] Exception during Comedy Cellar fetch: {e}")
        return []
        
    # Filter in-memory to return only the requested venue's events
    filtered_events = [ev for ev in all_events if ev["venue_name"] == canonical_target_venue]
    
    # De-duplicate events by URL, date_str, and normalized name
    seen_keys = set()
    unique_events = []
    for ev in filtered_events:
        key = (ev["url"], ev["date_str"], ev["name"].lower(), ev["datetime"].strftime("%H:%M"))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_events.append(ev)
            
    print(f"      [Comedy Cellar Scraper] Finished crawling. Total crawled: {len(all_events)}, Filtered for '{canonical_target_venue}': {len(filtered_events)}, Unique: {len(unique_events)}")
    return unique_events
