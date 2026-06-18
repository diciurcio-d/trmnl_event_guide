"""
NYC event newsletters — curates and emails highlights for specific categories.

Supports three newsletter types:
1. Talks & Tours (educational events, lectures, book launches, science, geopolitics)
2. Music Matches (concerts matching the user's liked YouTube Music library)
3. General Highlights (weekly picks excluding music matches and talks/tours)

Sends via Gmail SMTP using credentials from config.json.
"""

from __future__ import annotations

import json
import re
import smtplib
import textwrap
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("America/New_York")
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.json"
_SETTINGS_PATH = Path(__file__).parent.parent / "settings.py"


def _load_settings():
    import importlib.util
    spec = importlib.util.spec_from_file_location("settings", _SETTINGS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_settings = _load_settings()

# Events to select per day of week (0=Mon … 6=Sun)
_PICKS_BY_DOW = {0: 2, 1: 2, 2: 2, 3: 3, 4: 4, 5: 7, 6: 6}

# Max candidate events sent to LLM per day (keeps prompt size reasonable)
_CANDIDATES_PER_DAY = 40


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Classification Helpers
# ---------------------------------------------------------------------------

def _is_music_event(event: dict) -> bool:
    """Check if the event is a musical performance, concert, or gig."""
    category = str(event.get("category", "") or "").lower().strip()
    event_type = str(event.get("event_type", "") or "").lower().strip()
    
    music_cats = {"concert", "live music", "jazz club", "classical music", "music venue", "opera", "music"}
    if category in music_cats or event_type in music_cats:
        return True
        
    name = str(event.get("name", "") or "").lower()
    desc = str(event.get("description", "") or "").lower()
    
    keywords = [
        r"\bmusic\b", r"\bconcert\b", r"\bjazz\b", r"\bperformance\b", 
        r"\brecital\b", r"\bopera\b", r"\bsymphony\b", r"\blive band\b", 
        r"\bgig\b", r"\bsinger\b", r"\bsongwriter\b", r"\borchestra\b"
    ]
    
    for kw in keywords:
        if re.search(kw, name) or re.search(kw, desc):
            return True
            
    return False


def _is_sports_event(event: dict) -> bool:
    """Check if the event is a sports game, match viewing, or watch party."""
    category = str(event.get("category", "") or "").lower().strip()
    event_type = str(event.get("event_type", "") or "").lower().strip()
    
    sports_cats = {"sports", "sports_recreation", "stadium", "arena"}
    if category in sports_cats or event_type in sports_cats:
        return True
        
    name = str(event.get("name", "") or "").lower()
    desc = str(event.get("description", "") or "").lower()
    
    keywords = [
        r"\bmatch viewing\b", r"\bviewing party\b", r"\bwatch party\b", 
        r"\bwatching party\b", r"\bworld cup\b", r"\bsuper bowl\b", 
        r"\bgame viewing\b", r"\bcoppa italia\b", r"\buefa\b"
    ]
    
    for kw in keywords:
        if re.search(kw, name) or re.search(kw, desc):
            return True
            
    return False


def _is_performing_arts_or_comedy(event: dict) -> bool:
    """Check if the event is a comedy show, play, musical, or other performing arts show."""
    category = str(event.get("category", "") or "").lower().strip()
    event_type = str(event.get("event_type", "") or "").lower().strip()
    
    perf_cats = {
        "comedy", "standup", "stand-up", "improv", "theater", "theatre", 
        "broadway", "dance", "ballet", "magic show", "variety show", "cabaret"
    }
    if category in perf_cats or event_type in perf_cats:
        return True
        
    name = str(event.get("name", "") or "").lower()
    desc = str(event.get("description", "") or "").lower()
    
    keywords = [
        r"\bcomedy\b", r"\bstandup\b", r"\bstand-up\b", r"\bcomedian\b",
        r"\bimprov\b", r"\bplay\b", r"\bmusical\b", r"\btheater\b", r"\btheatre\b"
    ]
    
    for kw in keywords:
        if re.search(kw, name) or re.search(kw, desc):
            return True
            
    return False


def _is_talk_or_tour(event: dict) -> bool:
    """Check if the event is a talk, lecture, book launch, history lesson, or tour."""
    name = str(event.get("name", "") or "").lower()
    desc = str(event.get("description", "") or "").lower()

    # Exclude English language/conversation classes
    english_class_keywords = [
        r"\benglish conversation class",
        r"\benglish class",
        r"\besl\b",
        r"\blearn english\b",
        r"\bspeak english\b"
    ]
    if any(re.search(kw, name) or re.search(kw, desc) for kw in english_class_keywords):
        return False

    category = str(event.get("category", "") or "").lower().strip()
    event_type = str(event.get("event_type", "") or "").lower().strip()
    
    educational_cats = {"walking tour", "architecture tour", "book event", "talk/lecture", "museum event", "educational"}
    if category in educational_cats or event_type in educational_cats:
        return True
        
    # Exclude music, comedy, and general performing arts events
    # unless they are explicitly educational/historical/literary talks or tours.
    if _is_music_event(event) or _is_performing_arts_or_comedy(event):
        name = str(event.get("name", "") or "").lower()
        desc = str(event.get("description", "") or "").lower()
        educational_keywords = [
            r"\bwalking tour\b", r"\bguided tour\b", r"\bhistoric tour\b", r"\barchitecture tour\b",
            r"\bmuseum tour\b", r"\bgallery tour\b", r"\bexhibition tour\b", r"\bneighborhood tour\b",
            r"\blecture\b", r"\btalk\b", r"\bpanel discussion\b", r"\bsymposium\b",
            r"\bconversation\b", r"\bdiscussion\b", r"\bbook launch\b", r"\bbook event\b", r"\bq&a\b"
        ]
        if not any(re.search(kw, name) or re.search(kw, desc) for kw in educational_keywords):
            return False
        
    name = str(event.get("name", "") or "").lower()
    desc = str(event.get("description", "") or "").lower()
    
    keywords = [
        r"\btalk\b", r"\blecture\b", r"\bgeopolitics\b", r"\bscience lecture\b", 
        r"\bbook launch\b", r"\bhistory lesson\b", r"\bpanel discussion\b", 
        r"\bsymposium\b", r"\breading\b", r"\bauthor\b", r"\bpodcast\b", 
        r"\btour\b", r"\bwalk\b", r"\bseminar\b", r"\bq&a\b", r"\bconversation\b"
    ]
    
    for kw in keywords:
        if re.search(kw, name) or re.search(kw, desc):
            return True
            
    return False


# ---------------------------------------------------------------------------
# Event collection
# ---------------------------------------------------------------------------

def _parse_event_dt(event: dict) -> datetime | None:
    raw = event.get("datetime")
    if raw and isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=_TZ)
    if raw and isinstance(raw, str) and raw not in ("", "None"):
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=_TZ)
        except ValueError:
            pass
    date_str = str(event.get("date_str", "") or "").strip()
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_TZ)
        except ValueError:
            pass
    return None


def get_upcoming_events(days: int = 7) -> list[dict]:
    """Return all events from the sheet that fall within the next `days` days."""
    from venue_scout.venue_events_sheet import read_venue_events_from_sheet
    events = read_venue_events_from_sheet()
    now = datetime.now(_TZ)
    cutoff = now + timedelta(days=days)
    upcoming = []
    for event in events:
        dt = _parse_event_dt(event)
        if dt is None:
            continue
        if now <= dt <= cutoff:
            upcoming.append(event)
    return upcoming


def _group_by_day(events: list[dict]) -> dict[str, list[dict]]:
    """Group events into {YYYY-MM-DD: [events]} dicts, sorted by date."""
    groups: dict[str, list[dict]] = {}
    for event in events:
        dt = _parse_event_dt(event)
        if dt is None:
            continue
        key = dt.date().isoformat()
        groups.setdefault(key, []).append(event)
    return dict(sorted(groups.items()))


def _score_event(event: dict) -> int:
    """Simple heuristic score to surface interesting candidates."""
    score = 0
    if event.get("description") and len(str(event["description"])) > 60:
        score += 2
    if event.get("url"):
        score += 1
    if event.get("event_type") and event["event_type"] not in ("", "unknown"):
        score += 1
    if event.get("venue_name"):
        score += 1
    dt = _parse_event_dt(event)
    if dt and dt.hour >= 17:   # evening events
        score += 1
    return score


# ---------------------------------------------------------------------------
# LLM curation
# ---------------------------------------------------------------------------

def _event_summary_for_llm(event: dict, idx: int) -> dict:
    dt = _parse_event_dt(event)
    is_free = event.get("is_free")
    price = str(event.get("price", "") or "").strip()
    if is_free is True:
        price_str = "Free"
    elif price:
        price_str = price
    else:
        price_str = ""
    return {
        "index": idx,
        "name": event.get("name", ""),
        "venue": event.get("venue_name", ""),
        "time": dt.strftime("%-I:%M %p") if dt and dt.hour != 0 else "time TBD",
        "type": event.get("event_type", ""),
        "description": str(event.get("description", "") or "")[:300],
        "url": event.get("url") or event.get("event_source_url", ""),
        "price": price_str,
    }


def curate_newsletter(grouped: dict[str, list[dict]], newsletter_type: str = "general") -> list[dict]:
    """
    Ask Gemini to pick highlights for the entire week in a single LLM call.
    """
    import re
    from utils.llm import generate_content

    model = str(getattr(_settings, "NEWSLETTER_MODEL", "gemini-2.5-flash"))
    timeout = int(getattr(_settings, "NEWSLETTER_TIMEOUT_SEC", 60))

    # Build per-day metadata and a flat global candidate list
    day_meta = []          # [{date, label, n_picks, start_idx, end_idx}]
    all_candidates = []    # flat list of event dicts in global index order

    for date_str, events in grouped.items():
        dt_day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_TZ)
        dow = dt_day.weekday()
        n_picks = _PICKS_BY_DOW.get(dow, 2)
        label = dt_day.strftime("%A, %B %-d")

        candidates = sorted(events, key=_score_event, reverse=True)[:_CANDIDATES_PER_DAY]
        start_idx = len(all_candidates)
        all_candidates.extend(candidates)
        end_idx = len(all_candidates)

        day_meta.append({
            "date": date_str,
            "label": label,
            "n_picks": n_picks,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "candidates": candidates,
        })

    # Build per-day sections for the prompt using global indices
    days_payload = []
    for d in day_meta:
        summaries = [
            _event_summary_for_llm(e, d["start_idx"] + i)
            for i, e in enumerate(d["candidates"])
        ]
        days_payload.append({
            "date": d["date"],
            "label": d["label"],
            "picks_needed": d["n_picks"],
            "candidates": summaries,
        })

    if newsletter_type == "talks":
        topic_desc = "NYC Talks & Tours newsletter"
        preferences = """- Upcoming educational events, lectures, talks, tours, book launches, history lessons, science presentations, geopolitics panels, and podcast host appearances.
- A mix of topics (science, history, books, politics, etc.) spread across the week."""
    else:
        topic_desc = "NYC weekly events highlights newsletter"
        preferences = """- A mix of types (comedy, theatre, food, art, sports, outdoor, film, etc.) spread across the week.
- Do NOT include any music concerts or academic lectures/talks/tours (these are covered in other newsletters)."""

    prompt = f"""You are curating a {topic_desc} for the coming week. Pick highlights across all days in a single pass so you can ensure variety — no venue or event should appear more than once across the whole newsletter.

For each day, select exactly the number of events specified in "picks_needed". Prefer:
- Specific/unique events over generic ones
- Evening events for weekdays
{preferences}

Return strict JSON — a list of day objects, one per day, in the same order as the input. Each day object has:
- "date": (string, YYYY-MM-DD, from input)
- "picks": list of exactly picks_needed objects, each with:
  - "index": (int, the global index from the candidate list)
  - "name": (string, event name — you may shorten if very long)
  - "venue": (string)
  - "time": (string, e.g. "8:00 PM" or "time TBD")
  - "description": (string, 1-2 punchy sentences you write, max 180 chars — make it sound exciting)
  - "url": (string, from input — copy exactly)

Days and candidates:
{json.dumps(days_payload, ensure_ascii=False)}
"""

    prompt_chars = len(prompt)
    prompt_tokens_approx = prompt_chars // 4
    print(f"\n--- Single-call newsletter prompt ({prompt_chars:,} chars ≈ {prompt_tokens_approx:,} tokens) ---")
    print(prompt[:2000])
    if prompt_chars > 2000:
        print(f"  ... [{prompt_chars - 2000:,} more chars] ...")
    print(f"--- End prompt ---\n")

    # Build the fallback buckets (top-N-by-score, no LLM)
    def _fallback_buckets() -> list[dict]:
        return [
            {
                "date": d["date"],
                "label": d["label"],
                "picks": [
                    {
                        "name": e.get("name", ""),
                        "venue": e.get("venue_name", ""),
                        "time": _event_summary_for_llm(e, 0)["time"],
                        "description": str(e.get("description", "") or "")[:180],
                        "url": e.get("url") or e.get("event_source_url", ""),
                    }
                    for e in d["candidates"][:d["n_picks"]]
                ],
            }
            for d in day_meta
        ]

    try:
        raw = generate_content(
            prompt,
            max_retries=2,
            timeout_sec=timeout,
            model_name=model,
        )
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            raise ValueError("no JSON array in response")
        week_data = json.loads(match.group())
    except Exception as exc:
        print(f"  LLM curation failed: {exc} — using top-N by score for all days")
        return _fallback_buckets()

    # Map LLM output back to day_meta order, resolving URLs from global index
    date_to_meta = {d["date"]: d for d in day_meta}
    day_buckets = []

    for day_result in week_data:
        date_str = day_result.get("date", "")
        meta = date_to_meta.get(date_str)
        if meta is None:
            continue

        picks = day_result.get("picks", [])
        # Resolve URL and price from global all_candidates if LLM omitted them
        for pick in picks:
            idx = pick.get("index")
            if isinstance(idx, int) and 0 <= idx < len(all_candidates):
                src = all_candidates[idx]
                src_url = src.get("url") or src.get("event_source_url", "")
                if not pick.get("url") and src_url:
                    pick["url"] = src_url
                # Carry price/is_free from source event if not set by LLM
                if "price" not in pick:
                    is_free = src.get("is_free")
                    price = str(src.get("price", "") or "").strip()
                    pick["price"] = "Free" if is_free is True else price

        picks = picks[:meta["n_picks"]]
        day_buckets.append({
            "date": date_str,
            "label": meta["label"],
            "picks": picks,
        })

    # If LLM returned fewer days than expected, fill gaps with fallback
    returned_dates = {b["date"] for b in day_buckets}
    for d in day_meta:
        if d["date"] not in returned_dates:
            print(f"  LLM missing day {d['date']} — using top-{d['n_picks']} by score")
            day_buckets.append({
                "date": d["date"],
                "label": d["label"],
                "picks": [
                    {
                        "name": e.get("name", ""),
                        "venue": e.get("venue_name", ""),
                        "time": _event_summary_for_llm(e, 0)["time"],
                        "description": str(e.get("description", "") or "")[:180],
                        "url": e.get("url") or e.get("event_source_url", ""),
                    }
                    for e in d["candidates"][:d["n_picks"]]
                ],
            })

    # Sort final output by date (preserves week order)
    day_buckets.sort(key=lambda b: b["date"])
    return day_buckets


def curate_music_newsletter(events: list[dict]) -> list[dict]:
    """
    Curate the music newsletter. Since it's limited to matched library artists,
    we group them by day and pass them to the LLM to write exciting descriptions
    for each one (no need to enforce a specific picks-per-day limit).
    """
    import re
    from utils.llm import generate_content

    if not events:
        return []

    # Group by date
    grouped = _group_by_day(events)
    sorted_dates = sorted(grouped.keys())

    day_buckets = []
    for date_str in sorted_dates:
        day_events = grouped[date_str]
        dt_day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_TZ)
        label = dt_day.strftime("%A, %B %-d")

        summaries = [
            _event_summary_for_llm(e, idx)
            for idx, e in enumerate(day_events)
        ]

        prompt = f"""You are writing exciting descriptions for upcoming music performances in NYC that match the user's favorite artists.

Date: {label}

For each performance, write 1-2 punchy sentences describing the artist and event (max 180 chars). Make it sound exciting!
Return strict JSON as a list of objects, one per performance, in the same order as the input. Each object must have:
- "name": (string, event name)
- "venue": (string)
- "time": (string, e.g. "8:00 PM" or "time TBD")
- "description": (string, the punchy description you wrote)
- "url": (string, copy exactly)

Performances:
{json.dumps(summaries, ensure_ascii=False)}
"""

        model = str(getattr(_settings, "NEWSLETTER_MODEL", "gemini-2.5-flash"))
        timeout = int(getattr(_settings, "NEWSLETTER_TIMEOUT_SEC", 60))

        picks = []
        try:
            raw = generate_content(prompt, max_retries=2, timeout_sec=timeout, model_name=model)
            match = re.search(r"\[[\s\S]*\]", raw)
            if not match:
                raise ValueError("no JSON array in response")
            picks = json.loads(match.group())
            # Ensure URL, matched_artist, and price are preserved
            for i, p in enumerate(picks):
                if i < len(day_events):
                    src = day_events[i]
                    p["url"] = src.get("url") or src.get("event_source_url", "")
                    p["matched_artist"] = src.get("matched_artist", "")
                    p["price"] = "Free" if src.get("is_free") is True else (src.get("price", "") or "")
        except Exception as exc:
            print(f"  LLM curation failed for {date_str}: {exc} — using fallbacks")
            picks = []
            for e in day_events:
                picks.append({
                    "name": e.get("name", ""),
                    "venue": e.get("venue_name", ""),
                    "time": _event_summary_for_llm(e, 0)["time"],
                    "description": str(e.get("description", "") or "")[:180],
                    "url": e.get("url") or e.get("event_source_url", ""),
                    "matched_artist": e.get("matched_artist", ""),
                    "price": "Free" if e.get("is_free") is True else (e.get("price", "") or "")
                })

        day_buckets.append({
            "date": date_str,
            "label": label,
            "picks": picks
        })

    return day_buckets


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 0; background: #f4f4f4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #1a1a1a; }}
  .wrapper {{ max-width: 620px; margin: 0 auto; background: #ffffff; }}
  .header {{ background: #111; padding: 28px 32px 22px; }}
  .header h1 {{ margin: 0; color: #fff; font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
  .header p {{ margin: 6px 0 0; color: #aaa; font-size: 13px; }}
  .body {{ padding: 24px 32px 8px; }}
  .day-section {{ margin-bottom: 32px; }}
  .day-label {{ font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #fff; background: #222; padding: 8px 12px; margin-bottom: 16px; border-radius: 3px; }}
  .event {{ margin-bottom: 18px; }}
  .event-name {{ font-size: 15px; font-weight: 600; margin: 0 0 2px; }}
  .event-name a {{ color: #111; text-decoration: none; }}
  .event-name a:hover {{ text-decoration: underline; }}
  .event-meta {{ font-size: 12px; color: #777; margin: 0 0 4px; }}
  .event-desc {{ font-size: 13px; color: #444; margin: 0; line-height: 1.5; }}
  .footer {{ background: #f8f8f8; padding: 18px 32px; border-top: 1px solid #eee; }}
  .footer p {{ margin: 0; font-size: 11px; color: #aaa; }}
  @media (max-width: 480px) {{
    .header, .body, .footer {{ padding-left: 20px; padding-right: 20px; }}
  }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>{title}</h1>
    <p>{date_range}</p>
  </div>
  <div class="body">
    {day_sections}
  </div>
  <div class="footer">
    <p>You're receiving this because you're awesome. &nbsp;·&nbsp; <a href="https://whats-happening-nyc-996955494439.us-central1.run.app" style="color:#aaa;">Browse all events</a></p>
  </div>
</div>
</body>
</html>
"""


def _render_day_section(day: dict) -> str:
    label = day.get("label", "")
    date_str = day.get("date", "")
    picks = day.get("picks", [])

    try:
        dt_day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_TZ)
        short_date = dt_day.strftime("%a, %b %-d")
    except (ValueError, TypeError):
        short_date = ""

    events_html = ""
    for pick in picks:
        name = pick.get("name", "Untitled Event")
        url = pick.get("url", "")
        venue = pick.get("venue", "")
        time_str = pick.get("time", "")
        desc = pick.get("description", "")
        matched_artist = pick.get("matched_artist", "")

        if url:
            name_html = f'<a href="{url}">{name}</a>'
        else:
            name_html = name

        if matched_artist:
            artist_tag = f'<span style="background-color: #fff3cd; color: #856404; font-size: 11px; font-weight: 700; text-transform: uppercase; padding: 2px 6px; border-radius: 10px; margin-left: 8px; vertical-align: middle;">⭐ Match: {matched_artist}</span>'
            name_html = f"{name_html} {artist_tag}"

        price_str = str(pick.get("price", "") or "").strip()
        when = f"{short_date} · {time_str}" if time_str and time_str != "time TBD" else (short_date or time_str)
        meta_parts = [p for p in [venue, when, price_str] if p]
        meta = " · ".join(meta_parts)

        events_html += f"""\
    <div class="event">
      <p class="event-name">{name_html}</p>
      {f'<p class="event-meta">{meta}</p>' if meta else ''}
      {f'<p class="event-desc">{desc}</p>' if desc else ''}
    </div>
"""

    return f"""\
  <div class="day-section">
    <div class="day-label">{label}</div>
{events_html}  </div>
"""


def render_html(day_buckets: list[dict], date_range: str, title: str) -> str:
    day_sections = "\n".join(_render_day_section(d) for d in day_buckets)
    return _HTML_TEMPLATE.format(title=title, date_range=date_range, day_sections=day_sections)


# ---------------------------------------------------------------------------
# Email sending
# ---------------------------------------------------------------------------

def send_newsletter(html: str, subject: str, config: dict) -> None:
    gmail = config.get("gmail", {})
    sender = gmail.get("sender", "")
    password = gmail.get("app_password", "")
    recipient = gmail.get("recipient", sender)

    if not sender or not password:
        raise ValueError("Gmail credentials missing from config — set config.gmail.sender and config.gmail.app_password")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

    print(f"  Newsletter sent to {recipient}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(newsletter_type: str = "all", dry_run: bool = False, force: bool = False) -> None:
    if newsletter_type == "all":
        print("=== Curating and sending ALL three newsletters ===")
        print()
        for t in ["general", "talks", "music"]:
            try:
                main(newsletter_type=t, dry_run=dry_run, force=force)
            except Exception as e:
                print(f"Error curating/sending {t} newsletter: {e}")
            print()
        return

    import sys
    from venue_scout.concert_matcher import match_events_to_artists, get_user_artists

    config = _load_config()
    now = datetime.now(_TZ)

    if newsletter_type == "music":
        # Check bi-weekly schedule: run only on even ISO week numbers
        # (Allows --dry-run or --force to bypass this check)
        current_week = now.isocalendar().week
        if not dry_run and not force and (current_week % 2 != 0):
            print(f"Skipping music newsletter: configured to run bi-weekly on even weeks (current week {current_week} is odd).")
            print("Use --force to run anyway.")
            return
    
    if newsletter_type == "talks":
        days_range = 7
        end = now + timedelta(days=days_range)
        date_range = f"{now.strftime('%B %-d')} – {end.strftime('%B %-d, %Y')}"
        subject = f"What's Happening NYC · Talks & Tours · {now.strftime('%B %-d')}"
        title = "What's Happening NYC · Talks & Tours"
    elif newsletter_type == "music":
        days_range = 30
        end = now + timedelta(days=days_range)
        date_range = f"{now.strftime('%B %-d')} – {end.strftime('%B %-d, %Y')}"
        subject = f"What's Happening NYC · Music Matches · {now.strftime('%B %-d')}"
        title = "What's Happening NYC · Concert Matches"
    else:  # general
        days_range = 7
        end = now + timedelta(days=days_range)
        date_range = f"{now.strftime('%B %-d')} – {end.strftime('%B %-d, %Y')}"
        subject = f"What's Happening NYC · Weekly Highlights · {now.strftime('%B %-d')}"
        title = "What's Happening NYC · Weekly Highlights"

    print(f"=== NYC Newsletter ({newsletter_type}) — {date_range} ===")
    print()

    print(f"Loading upcoming events for the next {days_range} days...")
    events = get_upcoming_events(days=days_range)
    print(f"  {len(events)} total upcoming events")

    if not events:
        print("No upcoming events found. Exiting.")
        sys.exit(0)

    # Step 2: Apply first-stage filtering based on type
    if newsletter_type == "talks":
        filtered_events = [e for e in events if _is_talk_or_tour(e)]
        print(f"  Filtered to {len(filtered_events)} talks/tours")
    elif newsletter_type == "music":
        music_events = [e for e in events if _is_music_event(e) and not _is_sports_event(e)]
        print(f"  Filtered to {len(music_events)} music events. Matching against YT Music library...")
        artists = get_user_artists()
        if not artists:
            print("ERROR: No YouTube Music liked artists found. YouTube Music may not be authenticated.")
            sys.exit(1)
        matched_music_events = match_events_to_artists(music_events, artists)
        filtered_events = [e for e in matched_music_events if e.get("matched_artist")]
        print(f"  Found {len(filtered_events)} matched artist concerts")
    else:  # general
        # Get matched artists to exclude them
        artists = get_user_artists()
        if not artists:
            print("WARNING: YouTube Music not authenticated. Music match exclusions will be skipped for the general newsletter.")
            artist_names = set()
        else:
            artist_names = {a.get("name", "").lower() for a in artists if a.get("name")}
        
        filtered_events = []
        for e in events:
            # Exclude talks/tours
            if _is_talk_or_tour(e):
                continue
            # Exclude matched music events
            if _is_music_event(e):
                from venue_scout.concert_matcher import match_event_to_artists
                if match_event_to_artists(e, artist_names):
                    continue
            filtered_events.append(e)
        print(f"  Filtered to {len(filtered_events)} general events (excluding talks/tours and matched music)")

    if not filtered_events:
        print("No events found matching criteria. Exiting.")
        sys.exit(0)

    # Step 3: Curation
    if newsletter_type == "music":
        day_buckets = curate_music_newsletter(filtered_events)
    else:
        grouped = _group_by_day(filtered_events)
        print(f"  Spread across {len(grouped)} days")
        print()
        
        model = str(getattr(_settings, "NEWSLETTER_MODEL", "gemini-2.5-flash"))
        timeout = int(getattr(_settings, "NEWSLETTER_TIMEOUT_SEC", 60))
        print(f"Curating picks with {model} (single call, timeout {timeout}s)...")
        day_buckets = curate_newsletter(grouped, newsletter_type)

    total_picks = sum(len(d["picks"]) for d in day_buckets)
    print(f"  Selected {total_picks} events across {len(day_buckets)} days")
    print()

    html = render_html(day_buckets, date_range, title)

    if dry_run:
        out_path = Path(f"/tmp/newsletter_preview_{newsletter_type}.html")
        out_path.write_text(html)
        print(f"Dry run — preview written to {out_path}")
        print("Open it in a browser to review before sending.")
        return

    print(f"Sending newsletter: '{subject}'")
    send_newsletter(html, subject, config)
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Write HTML to /tmp instead of sending")
    parser.add_argument("--type", "-t", choices=["talks", "music", "general", "all"], default="all", help="Type of newsletter to send")
    parser.add_argument("--force", "-f", action="store_true", help="Force send even if bi-weekly schedule would skip")
    args = parser.parse_args()
    main(newsletter_type=args.type, dry_run=args.dry_run, force=args.force)
