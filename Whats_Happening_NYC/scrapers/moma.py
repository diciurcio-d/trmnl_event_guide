"""Custom scraper for the Museum of Modern Art (MoMA) (moma.org).

Fetches event listings from the public MoMA calendar using Playwright
to bypass Cloudflare protection and robustly parses the server-rendered HTML calendar.
"""

import re
import time
import random
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright

def parse_moma_date_header(date_header_str: str, current_date: datetime) -> datetime | None:
    """
    Parses a MoMA date header like 'Tuesday, May 19' or 'Fri, Dec 25' into a date object.
    Uses current_date (local NY time) to infer the year.
    """
    s = date_header_str.strip().replace("\xa0", " ")
    
    # Match any month name followed by a day number
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

def parse_moma_time(time_str: str) -> tuple[int, int] | None:
    """
    Parses a time string like '11:30 a.m.' or '1:30 p.m.' or '4:00 p.m.' or '11 a.m.'
    into (hour, minute) in 24-hour format.
    """
    t = time_str.strip().lower().replace("\xa0", " ")
    
    # Handle time ranges: take the first part (start time)
    t = re.split(r'[\-\u2013\u2014]', t)[0].strip()
    
    # Clean up periods in a.m. / p.m.
    t = t.replace(".", "").strip()
    
    # Match patterns like "11:30 am", "11 am", "11:30am", "11am"
    match = re.match(r"^(\d+)(?::(\d+))?\s*(am|pm)$", t)
    if not match:
        return None
        
    hour_str, min_str, am_pm = match.groups()
    hour = int(hour_str)
    minute = int(min_str) if min_str else 0
    
    if am_pm == "pm" and hour != 12:
        hour += 12
    elif am_pm == "am" and hour == 12:
        hour = 0
        
    return hour, minute

def parse_moma_html(html_content: str, current_time: datetime, venue_name: str = "Museum of Modern Art") -> list[dict]:
    """
    Parses calendar events from rendered HTML page.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    h2_tags = soup.find_all("h2")
    
    events_found = []
    # Match strings starting with weekday names (abbreviated or full)
    date_regex = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+[A-Z][a-z]+", re.IGNORECASE)
    
    matched_headers_count = 0
    parsed_date_headers_count = 0
    day_containers_found = 0
    total_lis_scanned = 0
    
    for h2 in h2_tags:
        text = h2.get_text().strip().replace("\xa0", " ")
        if not date_regex.match(text):
            continue
            
        matched_headers_count += 1
        event_date = parse_moma_date_header(text, current_time)
        if not event_date:
            continue
            
        parsed_date_headers_count += 1
        # Try to locate parent day container containing sibling elements/ul
        day_container = h2.find_parent("div", class_=lambda c: c and "@480/layout/flex:row" in c)
        if not day_container:
            day_container = h2.find_parent("div")
            
        if day_container:
            day_containers_found += 1
            
        ul = day_container.find("ul") if day_container else None
        if not ul:
            continue
            
        lis = ul.find_all("li", recursive=False)
        for li in lis:
            total_lis_scanned += 1
            a = li.find("a", href=lambda h: h and "/calendar/events/" in h)
            if not a:
                continue
                
            href = a.get("href")
            full_url = f"https://www.moma.org{href}"
            
            # Find the title paragraph
            title_p = li.find("p", class_=lambda c: c and "typography" in c)
            title = ""
            if title_p:
                title = title_p.get_text().strip().replace("\xa0", " ")
                title = re.sub(r'\s+', ' ', title)
            if not title:
                title = a.get_text().strip().replace("\xa0", " ")
                title = re.sub(r'\s+', ' ', title)
                
            meta_p_tags = li.find_all("p")
            time_str = ""
            location_str = "MoMA"
            category_str = "Museum Event"
            
            for p_tag in meta_p_tags:
                p_text = p_tag.get_text().strip().replace("\xa0", " ")
                # Check for time pattern
                if re.search(r'\d+:\d+\s*(a\.m\.|p\.m\.)', p_text, re.IGNORECASE) or re.search(r'\d+\s*(a\.m\.|p\.m\.)', p_text, re.IGNORECASE):
                    time_str = p_text
                # Check for location markers
                elif "Floor" in p_text or "Garden" in p_text or "Theater" in p_text or p_text.startswith("MoMA"):
                    location_str = p_text
                # Check for museum categories
                elif p_text in ["Film", "Performance", "Exhibition", "Gallery Session", "Member Event", "Workshop", "Family", "Talk"]:
                    category_str = p_text
            
            # Determine hour and minute (defaulting to 10:30 AM MoMA standard opening if unstated)
            hour = 10
            minute = 30
            if time_str:
                time_parsed = parse_moma_time(time_str)
                if time_parsed:
                    hour, minute = time_parsed
            
            dt_local = event_date.replace(hour=hour, minute=minute, tzinfo=ZoneInfo("America/New_York"))
            
            # Skip past events
            if dt_local < current_time:
                continue
                
            date_str = dt_local.strftime("%Y-%m-%d")
            
            # Construct standard schema description
            description = f"Category: {category_str}. Location: {location_str}."
            if time_str:
                description += f" Scheduled Time: {time_str}."
                
            events_found.append({
                "name": title,
                "datetime": dt_local,
                "date_str": date_str,
                "venue_name": venue_name,
                "address": "11 W 53rd St, New York, NY 10019",
                "event_type": category_str,
                "url": full_url,
                "source": "moma_scraper",
                "matched_artist": "",
                "travel_minutes": None,
                "description": description,
                "is_free": False,
                "price": "",
                "event_source_url": full_url,
                "extraction_method": "moma_scraper",
                "relevance_score": None,
                "validation_confidence": 1.0,
            })
            
    print(f"      [MoMA Scraper Parse Diagnostic] Total h2 tags: {len(h2_tags)}, Date regex matched: {matched_headers_count}, Date headers parsed: {parsed_date_headers_count}, Day containers found: {day_containers_found}, Total list items scanned: {total_lis_scanned}, Events parsed: {len(events_found)}")
    return events_found

def fetch_moma_events(venue_name: str = "Museum of Modern Art") -> list[dict]:
    """
    Fetch upcoming events for Museum of Modern Art (MoMA) covering a 32-day lookahead.
    Makes 4 separate, isolated page fetches spaced 8 days apart to secure complete calendar coverage,
    running each request in a clean browser process with randomized polite sleep gaps in between.
    
    Args:
        venue_name: Canonical name of the venue.
        
    Returns:
        List of normalized event dictionaries.
    """
    events = []
    ny_tz = ZoneInfo("America/New_York")
    now = datetime.now(ny_tz)
    
    for i in range(4):
        offset_days = i * 8
        target_date = now + timedelta(days=offset_days)
        date_str = target_date.strftime("%Y-%m-%d")
        url = f"https://www.moma.org/calendar/?date={date_str}"
        
        if i > 0:
            # Randomized polite delay between 2.5 and 5.5 seconds
            delay = random.uniform(2.5, 5.5)
            print(f"      [MoMA Scraper] Sleeping {delay:.2f} seconds to remain polite (randomized pacing)...")
            time.sleep(delay)
            
        print(f"      [MoMA Scraper] Page {i+1}/4: Fetching {url} in isolated browser...")
        
        try:
            with sync_playwright() as p:
                # Launch a completely clean browser instance to clear SPA routing context and session caches
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )
                page = context.new_page()
                
                # Navigation with domcontentloaded to handle heavy tracking script delay
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                print("      [MoMA Scraper] Reached domcontentloaded. Pausing 6 seconds for client-side rendering...")
                page.wait_for_timeout(6000)
                
                html_content = page.content()
                page_events = parse_moma_html(html_content, now, venue_name)
                print(f"      [MoMA Scraper] Parsed {len(page_events)} upcoming events from Page {i+1}.")
                events.extend(page_events)
                
                browser.close()
        except Exception as page_err:
            print(f"      [MoMA Scraper] Error during Page {i+1} fetch or parse: {page_err}")
            
    # De-duplicate events by URL, date_str, and normalized name
    seen_keys = set()
    unique_events = []
    for ev in events:
        key = (ev["url"], ev["date_str"], ev["name"].lower())
        if key not in seen_keys:
            seen_keys.add(key)
            unique_events.append(ev)
            
    print(f"      [MoMA Scraper] Crawling complete. Total parsed: {len(events)}, Unique upcoming events: {len(unique_events)}")
    return unique_events
