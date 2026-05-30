"""Custom scraper for Smoke Jazz Club (smokejazz.com).

Fetches event listings from Smoke's official ticketing calendar (tickets.smokejazz.com).
Leverages Vue.js Pinia state JSON parsing for exact metadata extraction with a BS4 HTML-based parsing fallback.

NOTE: Smoke Jazz Club's booking lookahead is narrow (typically only 4-8 days), meaning events should be refreshed extra frequently (e.g., every 2-3 days) to avoid stale "zero-event" listings.
"""

import re
import html as html_lib
import json
import requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

def parse_html_paragraph_prices(desc_element) -> str:
    """
    Helper to search for paragraphs containing '$' inside a description element to find price categories.
    """
    if not desc_element:
        return ""
    p_tags = desc_element.find_all("p")
    price_lines = []
    for p in p_tags:
        text = p.get_text().strip()
        if "$" in text:
            # Clean up the line
            clean_text = re.sub(r'\s+', ' ', text)
            price_lines.append(clean_text)
            
    if price_lines:
        return " | ".join(price_lines)
    return ""

def parse_html_performance_time(time_str: str) -> tuple[int, int] | None:
    """
    Helper to parse a performance time string like '6PM DINNER SHOW' or '8:30 PM' into (hour, minute) 24-hr format.
    """
    t = time_str.strip().upper()
    match = re.search(r"(\d+)(?::(\d+))?\s*(AM|PM)", t)
    if not match:
        return None
    hour_str, min_str, am_pm = match.groups()
    hour = int(hour_str)
    minute = int(min_str) if min_str else 0
    if am_pm == "PM" and hour != 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0
    return hour, minute

def fetch_smoke_jazz_events(venue_name: str = "Smoke Jazz Club") -> list[dict]:
    """
    Fetch upcoming performances for Smoke Jazz Club.
    Attempts high-fidelity Pinia Vue.js state JSON extraction first, with a robust card-based BeautifulSoup HTML fallback.
    
    Args:
        venue_name: Canonical venue name to set.
        
    Returns:
        List of normalized event dictionaries.
    """
    events = []
    ny_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    
    url = "https://tickets.smokejazz.com/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"      [Smoke Jazz Scraper] Fetching {url}...")
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"      [Smoke Jazz Scraper] Failed to fetch tickets page: status {r.status_code}")
            return []
            
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Strategy A: Extract embedded Vue Pinia JSON state
        state_parsed = False
        performances = []
        
        for s in soup.find_all("script"):
            if s.string and "window.__pinia" in s.string:
                m = re.search(r"window\.__pinia\s*=\s*(.*?);?$", s.string.strip(), re.DOTALL)
                if m:
                    try:
                        state = json.loads(m.group(1))
                        perfs_data = state.get("performancePaginate", {}).get("performances", [])
                        if perfs_data:
                            performances = perfs_data
                            state_parsed = True
                            print(f"      [Smoke Jazz Scraper] Successfully extracted {len(performances)} performances from Pinia state!")
                            break
                    except Exception as pinia_err:
                        print(f"      [Smoke Jazz Scraper] Error decoding Pinia state JSON: {pinia_err}")
                        
        if state_parsed and performances:
            # Parse using structured Pinia state
            for p in performances:
                # 1. Date & Time
                date_str = p.get("date", "") # YYYY-MM-DD
                time_str = p.get("time", "") # HH:MM
                if not date_str or not time_str:
                    continue
                    
                try:
                    dt_local = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=ny_tz)
                except Exception as e:
                    print(f"      [Smoke Jazz Scraper] Error parsing datetime {date_str} {time_str}: {e}")
                    continue
                    
                # Filter for upcoming events
                if dt_local.astimezone(timezone.utc) < now_utc:
                    continue
                    
                # 2. Show Info
                show = p.get("show") or {}
                show_id = p.get("show_id") or show.get("id") or ""
                name = (show.get("name") or "Jazz Performance").strip()
                
                # 3. URL Construction
                full_url = f"https://tickets.smokejazz.com/shows/{show_id}/?date={date_str}"
                
                # 4. Description Cleaning
                desc_html = show.get("description") or ""
                description = re.sub(r'<[^>]+>', '', desc_html)
                description = html_lib.unescape(description).strip()
                description = re.sub(r'\s+', ' ', description)
                
                # 5. Pricing Range
                prices = show.get("price_per_person") or []
                if prices:
                    price_str = " - ".join([f"${str(pr)}" for pr in prices])
                else:
                    price_str = ""
                    
                # Append performance tag if any (e.g. 6PM Dinner Show)
                button_text = (p.get("button_text") or "").strip()
                if button_text:
                    description = f"Performance: {button_text}. {description}"
                    
                events.append({
                    "name": name,
                    "datetime": dt_local,
                    "date_str": date_str,
                    "venue_name": "Smoke Jazz Club",
                    "address": "2751 Broadway, New York, NY 10025",
                    "event_type": "music",
                    "url": full_url,
                    "source": "smoke_jazz_scraper",
                    "matched_artist": "",
                    "travel_minutes": None,
                    "description": description,
                    "is_free": False,
                    "price": price_str,
                    "event_source_url": full_url,
                    "extraction_method": "smoke_jazz_scraper",
                    "relevance_score": None,
                    "validation_confidence": 1.0,
                })
        else:
            # Strategy B Fallback: BeautifulSoup HTML card parsing
            print("      [Smoke Jazz Scraper] Falling back to manual HTML card parsing...")
            cards = soup.find_all(class_=lambda c: c and "show-card" in c)
            print(f"      [Smoke Jazz Scraper] Found {len(cards)} show cards in HTML.")
            
            for card in cards:
                card_id = card.get("id", "")
                match = re.search(r"show-(\d+)-(\d{4}-\d{2}-\d{2})", card_id)
                if not match:
                    continue
                    
                show_id = match.group(1)
                date_str = match.group(2)
                
                # Title
                name_el = card.find(id=f"show-{show_id}-name")
                name = name_el.get_text().strip() if name_el else "Jazz Performance"
                
                # URL
                full_url = f"https://tickets.smokejazz.com/shows/{show_id}/?date={date_str}"
                
                # Description
                desc_el = card.find(id=f"show-{show_id}-description")
                description = ""
                price_str = ""
                if desc_el:
                    description = desc_el.get_text().strip()
                    description = re.sub(r'\s+', ' ', description)
                    price_str = parse_html_paragraph_prices(desc_el)
                    
                # Performance buttons (to get the separate times)
                perf_buttons = card.find_all("button", class_=lambda c: c and "performance-button" in c)
                if not perf_buttons:
                    perf_buttons = card.find_all(class_="performance-button")
                    
                # If no performance buttons found, we can default to 7:00 PM
                time_buttons = [b.get_text().strip() for b in perf_buttons if b.get_text().strip()]
                if not time_buttons:
                    time_buttons = ["7:00 PM"]
                    
                for t_btn in time_buttons:
                    parsed_time = parse_html_performance_time(t_btn)
                    if parsed_time:
                        hour, minute = parsed_time
                    else:
                        hour, minute = 19, 0
                        
                    try:
                        # Parse event date and apply local timezone
                        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        dt_local = datetime(event_date.year, event_date.month, event_date.day, hour, minute, tzinfo=ny_tz)
                    except Exception as e:
                        print(f"      [Smoke Jazz Scraper] Error constructing date for card show {show_id}: {e}")
                        continue
                        
                    # Filter for upcoming events
                    if dt_local.astimezone(timezone.utc) < now_utc:
                        continue
                        
                    event_desc = description
                    if t_btn:
                        event_desc = f"Performance: {t_btn}. {event_desc}"
                        
                    events.append({
                        "name": name,
                        "datetime": dt_local,
                        "date_str": date_str,
                        "venue_name": "Smoke Jazz Club",
                        "address": "2751 Broadway, New York, NY 10025",
                        "event_type": "music",
                        "url": full_url,
                        "source": "smoke_jazz_scraper",
                        "matched_artist": "",
                        "travel_minutes": None,
                        "description": event_desc,
                        "is_free": False,
                        "price": price_str,
                        "event_source_url": full_url,
                        "extraction_method": "smoke_jazz_scraper",
                        "relevance_score": None,
                        "validation_confidence": 1.0,
                    })
                    
    except Exception as e:
        print(f"      [Smoke Jazz Scraper] Exception during Smoke Jazz fetch: {e}")
        return []
        
    # De-duplicate events by URL, date_str, and normalized name & time
    seen_keys = set()
    unique_events = []
    for ev in events:
        key = (ev["url"], ev["date_str"], ev["name"].lower(), ev["datetime"].strftime("%H:%M"))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_events.append(ev)
            
    print(f"      [Smoke Jazz Scraper] Crawling complete. Unique upcoming performances parsed: {len(unique_events)}")
    return unique_events
