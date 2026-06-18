"""Custom scraper for Municipal Art Society (MAS) Tours (mas.org).

Fetches event listings directly from their public Events + Tours catalog.
Bypasses heavy browser automation and generic LLM parsing.
"""

import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

def parse_mas_datetime(date_str: str, year_context: int) -> datetime | None:
    """Parse MAS date strings like 'Saturday, June 6 at 11:00 AM' into datetime objects."""
    s = str(date_str).strip()
    s = re.sub(r'\s+', ' ', s)
    ny_tz = ZoneInfo("America/New_York")
    
    # Match patterns like: "June 6 at 11:00 AM" or "June 6 at 11:00AM"
    match = re.search(
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\s+(\d+)\s+at\s+(\d+:\d+)\s*(AM|PM)',
        s,
        re.IGNORECASE
    )
    if not match:
        return None
        
    month_name, day_str, time_str, am_pm = match.groups()
    day_num = int(day_str)
    
    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12
    }
    
    m_num = months.get(month_name.lower())
    if not m_num:
        return None
        
    time_parts = time_str.split(":")
    hour = int(time_parts[0])
    minute = int(time_parts[1])
    
    if am_pm.upper() == "PM" and hour != 12:
        hour += 12
    elif am_pm.upper() == "AM" and hour == 12:
        hour = 0
        
    try:
        dt = datetime(year_context, m_num, day_num, hour, minute)
        return dt.replace(tzinfo=ny_tz)
    except ValueError:
        return None

def fetch_mas_tours_events(venue_name: str = "MAS Tours") -> list[dict]:
    """
    Fetch upcoming tours and events for the Municipal Art Society.
    
    Args:
        venue_name: Canonical name of the venue to set in normalized events.
        
    Returns:
        List of normalized event dictionaries.
    """
    url = "https://www.mas.org/events"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    try:
        print(f"      [MAS Scraper] Fetching events from {url}...")
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"      [MAS Scraper] Server returned status {res.status_code}")
            return []
            
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.find_all(class_="event-item")
        print(f"      [MAS Scraper] Found {len(items)} event items on the page.")
        
        events = []
        ny_tz = ZoneInfo("America/New_York")
        now = datetime.now(ny_tz)
        
        for item in items:
            # 1. Title and URL
            title_div = item.find(class_="item-title")
            title = ""
            href = ""
            if title_div:
                a = title_div.find("a")
                if a:
                    title = a.get_text(strip=True)
                    href = str(a.get("href") or "").strip()
            
            if not title:
                continue
                
            # 2. Date / Time Parsing (with Year Context from preceding section-header)
            date_div = item.find(class_="event_date")
            date_str = date_div.get_text(strip=True) if date_div else ""
            
            parent_section = item.find_previous(class_="section-header")
            year_context = now.year
            if parent_section:
                header_text = parent_section.get_text(strip=True)
                year_match = re.search(r'\b(202\d)\b', header_text)
                if year_match:
                    year_context = int(year_match.group(1))
                    
            dt_local = parse_mas_datetime(date_str, year_context)
            if not dt_local:
                # Fallback to current year context if parsing fails or header is missing
                dt_local = parse_mas_datetime(date_str, now.year)
                
            if not dt_local:
                continue
                
            # Filter for upcoming events only
            if dt_local < now:
                continue
                
            date_str_formatted = dt_local.strftime("%Y-%m-%d")
            
            # 3. Prices
            price_div = item.find(class_="event_prices")
            price_text = ""
            if price_div:
                price_text = price_div.get_text(" ", strip=True)
                price_text = re.sub(r'\s+', ' ', price_text).strip()
                
            is_free = True
            if "member" in price_text.lower() or "$" in price_text:
                is_free = False
                if "free" in price_text.lower():
                    is_free = True
                    
            # 4. Host/Docent and Description Construction
            host_div = item.find(class_="small_title_or_label")
            host = host_div.get_text(strip=True) if host_div else ""
            
            desc_parts = []
            if host:
                desc_parts.append(host)
            if price_text:
                desc_parts.append(price_text)
            description = " | ".join(desc_parts)
            
            # Map to canonical schema
            event_dict = {
                "name": title[:70],
                "datetime": dt_local,
                "date_str": date_str_formatted,
                "venue_name": venue_name,
                "address": "Various NYC Locations",
                "event_type": "Walking Tour",
                "url": href or url,
                "source": "mas_tours",
                "matched_artist": "",
                "travel_minutes": None,
                "description": description[:100],
                "is_free": is_free,
                "price": price_text,
                "event_source_url": url,
                "extraction_method": "mas_tours_html_parser",
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
        
        print(f"      [MAS Scraper] Parsed and filtered {len(unique_events)} upcoming unique events.")
        return unique_events
        
    except Exception as e:
        print(f"      [MAS Scraper] Exception during MAS Tours fetch: {e}")
        return []
