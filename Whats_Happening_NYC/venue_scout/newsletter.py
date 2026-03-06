"""
Weekly NYC event newsletter — curates and emails highlights for the next 7 days.

Sends via Gmail SMTP using credentials from config.json.
"""

from __future__ import annotations

import json
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
    return {
        "index": idx,
        "name": event.get("name", ""),
        "venue": event.get("venue_name", ""),
        "time": dt.strftime("%-I:%M %p") if dt and dt.hour != 0 else "time TBD",
        "type": event.get("event_type", ""),
        "description": str(event.get("description", "") or "")[:300],
        "url": event.get("url") or event.get("event_source_url", ""),
    }


def curate_newsletter(grouped: dict[str, list[dict]]) -> list[dict]:
    """
    Ask Gemini to pick highlights for each day.

    Venues and event names already chosen on earlier days are passed as
    exclusions to each subsequent day's prompt so nothing repeats.

    Returns a list of day-buckets:
      [{"date": "2026-03-07", "label": "Saturday, March 7",
        "picks": [{"name":..., "venue":..., "time":..., "description":..., "url":...}]}]
    """
    import re
    from utils.llm import generate_content

    model = str(getattr(_settings, "NEWSLETTER_MODEL", "gemini-2.5-flash"))
    timeout = int(getattr(_settings, "NEWSLETTER_TIMEOUT_SEC", 60))

    day_buckets = []
    used_venues: set[str] = set()   # normalised venue names already picked
    used_events: set[str] = set()   # normalised event names already picked

    for date_str, events in grouped.items():
        dt_day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_TZ)
        dow = dt_day.weekday()
        n_picks = _PICKS_BY_DOW.get(dow, 2)
        label = dt_day.strftime("%A, %B %-d")

        # Sort by score and take top candidates
        candidates = sorted(events, key=_score_event, reverse=True)[:_CANDIDATES_PER_DAY]
        summaries = [_event_summary_for_llm(e, i) for i, e in enumerate(candidates)]

        exclusion_note = ""
        if used_venues or used_events:
            parts = []
            if used_venues:
                parts.append("venues: " + ", ".join(sorted(used_venues)))
            if used_events:
                parts.append("events: " + ", ".join(sorted(used_events)))
            exclusion_note = (
                "\nALREADY FEATURED in earlier days (do NOT pick these again):\n"
                + "\n".join(parts)
                + "\n"
            )

        prompt = f"""You are curating a NYC events newsletter. For {label}, pick exactly {n_picks} events that together offer variety and genuine interest.

Prefer: specific/unique events over generic ones, evening events for weekdays, a mix of types (music, art, food, comedy, sports, outdoor, family, etc.).
{exclusion_note}
Return strict JSON — a list of {n_picks} objects, each with keys:
- "index": (int, from the input)
- "name": (string, event name — you may shorten if very long)
- "venue": (string)
- "time": (string, e.g. "8:00 PM" or "time TBD")
- "description": (string, 1-2 punchy sentences you write, max 180 chars — make it sound exciting)
- "url": (string, from input)

Events:
{json.dumps(summaries, ensure_ascii=False)}
"""
        print(f"\n--- Prompt for {label} ---")
        print(prompt)
        print(f"--- End prompt ---\n")

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
            picks = json.loads(match.group())
            # Resolve index back to original event URL if LLM omitted it
            for pick in picks:
                idx = pick.get("index")
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    src_url = candidates[idx].get("url") or candidates[idx].get("event_source_url", "")
                    if not pick.get("url") and src_url:
                        pick["url"] = src_url
        except Exception as exc:
            print(f"  LLM curation failed for {label}: {exc} — using top {n_picks} by score")
            picks = [
                {
                    "name": e.get("name", ""),
                    "venue": e.get("venue_name", ""),
                    "time": _event_summary_for_llm(e, 0)["time"],
                    "description": str(e.get("description", "") or "")[:180],
                    "url": e.get("url") or e.get("event_source_url", ""),
                }
                for e in candidates[:n_picks]
            ]

        picks = picks[:n_picks]

        # Record what was picked so future days can avoid repeating them
        for pick in picks:
            v = str(pick.get("venue", "")).strip().lower()
            n = str(pick.get("name", "")).strip().lower()
            if v:
                used_venues.add(v)
            if n:
                used_events.add(n)

        day_buckets.append({
            "date": date_str,
            "label": label,
            "picks": picks,
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
<title>What's Happening NYC</title>
<style>
  body {{ margin: 0; padding: 0; background: #f4f4f4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #1a1a1a; }}
  .wrapper {{ max-width: 620px; margin: 0 auto; background: #ffffff; }}
  .header {{ background: #111; padding: 28px 32px 22px; }}
  .header h1 {{ margin: 0; color: #fff; font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
  .header p {{ margin: 6px 0 0; color: #aaa; font-size: 13px; }}
  .body {{ padding: 24px 32px 8px; }}
  .day-section {{ margin-bottom: 28px; }}
  .day-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #888; border-bottom: 1px solid #eee; padding-bottom: 6px; margin-bottom: 14px; }}
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
    <h1>What's Happening NYC</h1>
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
    picks = day.get("picks", [])
    events_html = ""
    for pick in picks:
        name = pick.get("name", "Untitled Event")
        url = pick.get("url", "")
        venue = pick.get("venue", "")
        time_str = pick.get("time", "")
        desc = pick.get("description", "")

        if url:
            name_html = f'<a href="{url}">{name}</a>'
        else:
            name_html = name

        meta_parts = [p for p in [venue, time_str] if p]
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


def render_html(day_buckets: list[dict], date_range: str) -> str:
    day_sections = "\n".join(_render_day_section(d) for d in day_buckets)
    return _HTML_TEMPLATE.format(date_range=date_range, day_sections=day_sections)


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

def main(dry_run: bool = False) -> None:
    import sys
    config = _load_config()
    now = datetime.now(_TZ)
    end = now + timedelta(days=7)
    date_range = f"{now.strftime('%B %-d')} – {end.strftime('%B %-d, %Y')}"
    subject = f"What's Happening NYC · {now.strftime('%B %-d')}"

    print(f"=== NYC Newsletter — {date_range} ===")
    print()

    print("Loading upcoming events from sheet...")
    events = get_upcoming_events(days=7)
    print(f"  {len(events)} events in the next 7 days")

    if not events:
        print("No upcoming events found. Exiting.")
        sys.exit(0)

    grouped = _group_by_day(events)
    print(f"  Spread across {len(grouped)} days")
    print()

    model = str(getattr(_settings, "NEWSLETTER_MODEL", "gemini-2.5-flash"))
    timeout = int(getattr(_settings, "NEWSLETTER_TIMEOUT_SEC", 60))
    print(f"Curating picks with {model} (timeout {timeout}s per day)...")
    day_buckets = curate_newsletter(grouped)
    total_picks = sum(len(d["picks"]) for d in day_buckets)
    print(f"  Selected {total_picks} events across {len(day_buckets)} days")
    print()

    html = render_html(day_buckets, date_range)

    if dry_run:
        out_path = Path("/tmp/newsletter_preview.html")
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
    args = parser.parse_args()
    main(dry_run=args.dry_run)
