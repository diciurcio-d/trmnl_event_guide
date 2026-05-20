"""Custom scraper for The Metropolitan Museum of Art (The Met) and The Met Cloisters (metmuseum.org).

Queries the Met's official events calendar with from/to bounds and pageSize/page parameters,
and dynamically routes events to their correct corresponding venue listings based on location tags.
Supports randomized polite delay pacing when fetching multiple pages.
"""

import re
import time
import random
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

def parse_the_met_date_header(date_str: str, current_date: datetime) -> datetime | None:
    """
    Parses a date header like 'Thursday, May 21' or 'Jun 5' or 'June 12' into a date object.
    Uses current_date (local NY time) to infer the year.
    """
    s = date_str.strip().replace("\xa0", " ")
    
    # Match pattern for month name (abbreviated or full) followed by day number
    pattern = r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\s+(\d+)"
    match = re.search(pattern, s, re.IGNORECASE)
    if not match:
        return None
        
    month_name, day_str = match.groups()
    day_num = int(day_str)
    
    months = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12
    }
    
    m_lower = month_name.lower()
    if m_lower not in months:
        return None
    month_num = months[m_lower]
    
    year = current_date.year
    # Handle year rollover: if event month is early (e.g., January) and current month is late (e.g., Nov/Dec)
    if month_num < current_date.month and current_date.month >= 11:
        year += 1
        
    try:
        return datetime(year, month_num, day_num)
    except ValueError:
        return None

def parse_the_met_time(time_str: str) -> tuple[int, int] | None:
    """
    Parses a time string like '10:30 AM' or '6:00 PM' into (hour, minute) in 24-hour format.
    """
    t = time_str.strip().lower().replace("\xa0", " ")
    
    # Handle time ranges: take the first part
    t = re.split(r'[\-\u2013\u2014]', t)[0].strip()
    
    # Clean up periods if any
    t = t.replace(".", "").strip()
    
    match = re.match(r"^(\d+):(\d+)\s*(am|pm)$", t)
    if not match:
        # Try hour-only like "11 am"
        match_hour = re.match(r"^(\d+)\s*(am|pm)$", t)
        if not match_hour:
            return None
        hour_str, am_pm = match_hour.groups()
        hour = int(hour_str)
        minute = 0
    else:
        hour_str, min_str, am_pm = match.groups()
        hour = int(hour_str)
        minute = int(min_str)
        
    if am_pm == "pm" and hour != 12:
        hour += 12
    elif am_pm == "am" and hour == 12:
        hour = 0
        
    return hour, minute

def fetch_the_met_events(venue_name: str) -> list[dict]:
    """
    Fetch upcoming events for either 'The Metropolitan Museum of Art' or 'The Met Cloisters'
    covering a 32-day lookahead.
    Queries the central Met events calendar dynamically, parses event locations in-memory,
    and assigns them to their corresponding physical venue with correct addresses.
    Supports randomized polite delays when fetching multiple pages.
    
    Args:
        venue_name: Canonical name of the venue ('The Metropolitan Museum of Art' or 'The Met Cloisters')
        
    Returns:
        List of normalized event dictionaries.
    """
    events = []
    ny_tz = ZoneInfo("America/New_York")
    now = datetime.now(ny_tz)
    
    today_str = now.strftime("%Y-%m-%d")
    end_date = now + timedelta(days=32)
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Check which target venue name was requested to filter results
    is_cloisters_request = "cloisters" in venue_name.lower()
    canonical_target_venue = "The Met Cloisters" if is_cloisters_request else "The Metropolitan Museum of Art"
    
    print(f"      [The Met Scraper] Starting fetch for '{canonical_target_venue}' (Lookahead: {today_str} to {end_date_str})")
    
    page = 1
    max_pages = 8
    all_raw_events = []
    
    while page <= max_pages:
        if page > 1:
            # Randomized polite delay between 2.0 and 4.5 seconds as requested by USER
            delay = random.uniform(2.0, 4.5)
            print(f"      [The Met Scraper] Sleeping {delay:.2f} seconds to remain polite (randomized pacing)...")
            time.sleep(delay)
            
        params = {
            "from": today_str,
            "to": end_date_str,
            "pageSize": "20",
            "page": str(page)
        }
        
        url = "https://www.metmuseum.org/events"
        print(f"      [The Met Scraper] Page {page}: Requesting {url} with params {params}...")
        
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"      [The Met Scraper] Page {page} returned bad status code: {r.status_code}")
                break
                
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Find date group containers
            groups = soup.find_all(class_=lambda c: c and "groupContainer" in c)
            if not groups:
                print(f"      [The Met Scraper] Page {page} has no date group containers. Ending search.")
                break
                
            page_raw_count = 0
            page_cards_count = 0
            
            for grp in groups:
                # Group date header (e.g., 'Thursday, May 21')
                date_el = grp.find(class_=lambda c: c and "date" in c)
                date_text = date_el.get_text().strip() if date_el else ""
                if not date_text:
                    continue
                    
                event_date = parse_the_met_date_header(date_text, now)
                if not event_date:
                    continue
                    
                # Event cards in this date group
                cards = grp.find_all(class_=lambda c: c and "eventCard" in c)
                page_cards_count += len(cards)
                for card in cards:
                    # Title & link
                    title_el = card.find(class_=lambda c: c and "title" in c)
                    if not title_el:
                        continue
                    a_tag = title_el.find("a")
                    if not a_tag:
                        continue
                        
                    title = a_tag.get_text().strip().replace("\xa0", " ")
                    title = re.sub(r'\s+', ' ', title)
                    href = a_tag.get("href")
                    
                    # Resolve to absolute URL
                    full_url = href
                    if href.startswith("/"):
                        full_url = f"https://www.metmuseum.org{href}"
                    elif href.startswith("engage.metmuseum.org") or href.startswith("metmuseum.org"):
                        full_url = f"https://{href}"
                        
                    # Description
                    desc_el = card.find(class_=lambda c: c and "description" in c)
                    description = desc_el.get_text().strip().replace("\xa0", " ") if desc_el else ""
                    description = re.sub(r'\s+', ' ', description)
                    
                    # Metadata blocks
                    meta_divs = card.find_all(class_=lambda c: c and "timeAndPlace" in c)
                    time_str = ""
                    location_str = ""
                    price_str = ""
                    
                    for div in meta_divs:
                        spans = div.find_all("span")
                        span_texts = [s.get_text().strip() for s in spans if s.get_text().strip()]
                        if len(span_texts) == 2:
                            # Usually [time, location]
                            time_str = span_texts[0]
                            location_str = span_texts[1]
                        elif len(span_texts) == 1:
                            # Usually [price]
                            price_str = span_texts[0]
                            
                    # Clean/normalize values
                    if not location_str:
                        location_str = "The Met Fifth Avenue"
                        
                    # Determine physical venue assignment
                    is_cloisters = "cloisters" in location_str.lower()
                    
                    event_venue = "The Met Cloisters" if is_cloisters else "The Metropolitan Museum of Art"
                    event_address = "99 Margaret Corbin Dr, New York, NY 10040" if is_cloisters else "1000 5th Ave, New York, NY 10028"
                    
                    # Parse event hours and minute (defaulting to Standard Met Opening 10:00 AM)
                    hour = 10
                    minute = 0
                    if time_str:
                        time_parsed = parse_the_met_time(time_str)
                        if time_parsed:
                            hour, minute = time_parsed
                            
                    dt_local = event_date.replace(hour=hour, minute=minute, tzinfo=ZoneInfo("America/New_York"))
                    
                    # Skip past events
                    if dt_local < now:
                        continue
                        
                    date_str = dt_local.strftime("%Y-%m-%d")
                    
                    # Infer Category taxonomy from deep link keywords
                    category_str = "Museum Event"
                    url_lower = full_url.lower()
                    if "talks" in url_lower or "lecture" in url_lower or "academic" in url_lower:
                        category_str = "Talk"
                    elif "performance" in url_lower or "music" in url_lower or "dance" in url_lower:
                        category_str = "Performance"
                    elif "workshop" in url_lower or "class" in url_lower or "studio" in url_lower:
                        category_str = "Workshop"
                    elif "family" in url_lower or "storytime" in url_lower:
                        category_str = "Family"
                    elif "membership" in url_lower:
                        category_str = "Member Event"
                    elif "exhibition" in url_lower:
                        category_str = "Exhibition"
                        
                    # Clean up prices
                    is_free = False
                    if price_str.lower() in ["free", "free with museum admission", "gratuito con la entrada al museo"]:
                        is_free = True
                        
                    # Compose standard description field
                    rich_description = f"Category: {category_str}. Location: {location_str}."
                    if price_str:
                        rich_description += f" Price: {price_str}."
                    if time_str:
                        rich_description += f" Scheduled Time: {time_str}."
                    if description:
                        rich_description += f" Description: {description}"
                        
                    all_raw_events.append({
                        "name": title,
                        "datetime": dt_local,
                        "date_str": date_str,
                        "venue_name": event_venue,
                        "address": event_address,
                        "event_type": category_str.lower(),
                        "url": full_url,
                        "source": "the_met_scraper",
                        "matched_artist": "",
                        "travel_minutes": None,
                        "description": rich_description,
                        "is_free": is_free,
                        "price": price_str,
                        "event_source_url": full_url,
                        "extraction_method": "the_met_scraper",
                        "relevance_score": None,
                        "validation_confidence": 1.0,
                    })
                    page_raw_count += 1
                    
            print(f"      [The Met Scraper] Page {page} completed. Found {page_cards_count} event cards, parsed {page_raw_count} raw items.")
            if page_cards_count == 0:
                print("      [The Met Scraper] No event cards found on this page. Ending pagination loop.")
                break
                
            page += 1
            
        except Exception as err:
            print(f"      [The Met Scraper] Error during Page {page} fetch/parse: {err}")
            break
            
    # Filter the events in-memory to only include the requested venue name
    filtered_events = [ev for ev in all_raw_events if ev["venue_name"] == canonical_target_venue]
    
    # De-duplicate events by URL, date_str, and normalized name
    seen_keys = set()
    unique_events = []
    for ev in filtered_events:
        key = (ev["url"], ev["date_str"], ev["name"].lower())
        if key not in seen_keys:
            seen_keys.add(key)
            unique_events.append(ev)
            
    print(f"      [The Met Scraper] Search finished. Crawled total: {len(all_raw_events)}, Filtered for '{canonical_target_venue}': {len(filtered_events)}, Unique: {len(unique_events)}")
    return unique_events
