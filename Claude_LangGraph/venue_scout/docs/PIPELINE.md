# Venue Scout Pipeline

The venue scouting process consists of four main stages:

## Stage 1: Discovery

Discover venues in a city using Gemini with Google Search Grounding.

```bash
# Run discovery for all categories
python -m venue_scout.cli discover --city NYC

# Or run the discovery script directly
python tests/gemini_grounding/run_remaining.py
```

**Output:** Venues added to "Venue Scout Cache" Google Sheet with basic info (name, address, category).

---

## Stage 2: Validation

Validate discovered venues - find official websites and events pages.

```bash
# RECOMMENDED: Use parallel workers + skip-jina for speed
python -m venue_scout.cli --workers 5 validate-websites --skip-jina

# Limit to test subset
python -m venue_scout.cli --workers 5 validate-websites --skip-jina --limit 50
```

**What it does:**
- Searches for official venue websites (not aggregators like Yelp)
- Discovers events/calendar page URLs
- Marks closed venues
- Sets `website_status`: verified, closed, search_failed, etc.

**Output:** Venues updated with `website`, `events_url`, `website_status`.

---

## Stage 3: Cleaning

Clean and enrich verified venues before event crawling.

```bash
# Run cleaning on verified venues
python -m venue_scout.cli clean-venues --limit 50
```

**What it does:**
1. **Verify addresses** - Validate with Google Places API, get formatted addresses
2. **Remove duplicates** - Dedupe by normalized name, keep best-scored entry
3. **Add descriptions** - Generate venue descriptions using Gemini

**Output:** Venues updated in "Venue Scout Cache" sheet with clean, enriched data.

---

## Stage 4: Event Crawling

Fetch upcoming events from verified venues.

```bash
# Fetch events for all verified venues
python -m venue_scout.cli --workers 5 all

# Fetch for specific categories
python -m venue_scout.cli --workers 5 categories "music venue" "comedy clubs"

# Export to Google Sheets
python -m venue_scout.cli --workers 5 --export all
```

**What it does:**
- Uses Ticketmaster API for venues with TM IDs (fast)
- Uses direct API for venues with discovered endpoints (fast)
- Falls back to web scraping for other venues
- Matches events to user's YouTube Music artists

**Output:** Events cached locally and optionally exported to Google Sheets.

---

## Quick Reference

| Stage | Command | Time (5k venues) |
|-------|---------|------------------|
| Discovery | `discover --city NYC` | ~2-4 hours |
| Validation | `--workers 5 validate-websites --skip-jina` | ~4-8 hours |
| Cleaning | `clean-venues` | ~1-2 hours |
| Crawling | `--workers 5 all` | ~2-4 hours |

---

## Data Flow

```
Discovery (Gemini Grounding)
    ↓
Venue Scout Cache (Google Sheet)
    ↓
Validation (website + events_url)
    ↓
Cleaning (addresses, dedup, descriptions)
    ↓
Event Crawling (TM API / scraping)
    ↓
Events Cache + Google Sheets export
```

All venue data is stored in a single "Venue Scout Cache" sheet throughout the pipeline.
