# Venue Scout CLI

Command-line interface for fetching and managing venue events.

## Basic Usage

```bash
python -m venue_scout.cli [OPTIONS] COMMAND [ARGS]
```

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--city CITY` | | City to search (default: NYC) |
| `--force` | `-f` | Force refresh, ignore cache |
| `--export` | `-e` | Export results to Google Sheet |
| `--match` | `-m` | Match events to YouTube Music artists |
| `--workers N` | `-w N` | Number of parallel workers (default: 1) |
| `--resume` | `-r` | Resume interrupted fetch from local cache |

---

## Commands

### `all` - Fetch Events for ALL Venues

Fetch events for all verified venues in the database.

```bash
# Fetch all venues (sequential)
python -m venue_scout.cli all

# Fetch all venues with 10 parallel workers
python -m venue_scout.cli --workers 10 all

# Fetch all, export to Google Sheets, and match to your artists
python -m venue_scout.cli --workers 10 --export --match all

# Force refresh all venues (ignore cache)
python -m venue_scout.cli --force --workers 10 --export all
```

**What it does:**
- Fetches events for all 1,470+ verified venues
- Uses Ticketmaster API for venues with TM IDs (fast)
- Uses direct API for venues with discovered endpoints (fast)
- Falls back to web scraping for other venues
- Shows breakdown of sources before starting

---

### `venues` - Fetch Events for Specific Venues

Fetch events for one or more venues by name.

```bash
# Single venue
python -m venue_scout.cli venues "BAM"

# Multiple venues
python -m venue_scout.cli venues "BAM" "Irving Plaza" "Webster Hall"

# With options
python -m venue_scout.cli --force --workers 5 venues "BAM" "Irving Plaza"

# Match to your YouTube Music artists
python -m venue_scout.cli --match venues "Bowery Ballroom" "Mercury Lounge"

# Export to Google Sheets
python -m venue_scout.cli --export venues "Carnegie Hall"
```

---

### `categories` - Fetch Events by Venue Category

Fetch events for all venues in specified categories.

```bash
# Single category
python -m venue_scout.cli categories "music venue"

# Multiple categories
python -m venue_scout.cli categories "music venue" "comedy clubs" "jazz clubs"

# Parallel with 10 workers
python -m venue_scout.cli --force --workers 10 categories "broadway theaters"

# Match and export
python -m venue_scout.cli --match --export categories "rock music venues"
```

**Available categories:**
- `music venue`, `rock music venues`, `jazz clubs`, `live music bars`
- `broadway theaters`, `off broadway theaters`
- `comedy clubs`
- `concert halls`, `performing arts centers`
- `dance venues ballet contemporary`
- `art galleries`
- `bookstores with author events`
- `trivia night bars`
- And more...

---

### `status` - Show Cache Status

Display information about cached venue events.

```bash
python -m venue_scout.cli status
```

**Output includes:**
- Total venues fetched
- Fresh vs stale venues
- Total events cached
- Breakdown by city and source

---

### `matched` - Show Matched Events

Display events that match your YouTube Music artists.

```bash
python -m venue_scout.cli matched
```

**Output:**
- Events grouped by matched artist
- Date, venue, and ticket URL for each event

---

### `clear` - Clear Cache

Clear cached venue event data.

```bash
# Clear all cache
python -m venue_scout.cli clear

# Clear specific venue
python -m venue_scout.cli clear --venue "BAM"

# Clear specific city
python -m venue_scout.cli --city NYC clear
```

---

### `export` - Export to Google Sheets

Export all cached events to Google Sheets.

```bash
python -m venue_scout.cli export
```

---

### `validate-websites` - Validate Venue Websites

Validate and discover official venue websites using LLM search.

```bash
# RECOMMENDED: Use parallel workers for 10x+ speedup
python -m venue_scout.cli --workers 5 validate-websites

# Skip Jina Reader if it's rate-limited (uses direct HTML fetch)
python -m venue_scout.cli --workers 5 validate-websites --skip-jina

# Limit to first 50 venues
python -m venue_scout.cli --workers 5 validate-websites --limit 50

# Sequential (SLOW - ~1 venue/min, avoid unless debugging)
python -m venue_scout.cli validate-websites
```

**⚠️ Always use `--workers 5` or higher** - Sequential mode is ~10x slower.

| Workers | 1000 Venues | 3000 Venues |
|---------|-------------|-------------|
| 1 (default) | ~16 hours | ~50 hours |
| 5 | ~1.5 hours | ~4 hours |
| 10 | ~45 min | ~2 hours |

**What it does:**
- Finds official websites for venues without URLs
- Discovers events/calendar pages during validation
- Replaces aggregator URLs (Yelp, TripAdvisor) with official sites
- Marks permanently closed venues
- Retries up to 3 times for venues with search failures

**Note:** Do NOT use `tests/run_website_validator.py` - it runs sequentially. Always use the CLI.

---

### `find-events-pages` - Discover Events Pages

Find events/calendar pages for venues that have websites but no events_url.

```bash
# RECOMMENDED: Use parallel workers
python -m venue_scout.cli --workers 5 find-events-pages

# Limit to 100 venues
python -m venue_scout.cli --workers 5 find-events-pages --limit 100
```

**⚠️ Always use `--workers 5` or higher** for faster processing.

**What it does:**
- Scans verified venues that have `website` but no `events_url`
- Checks common paths: `/events`, `/calendar`, `/shows`, `/exhibitions`
- Uses LLM to verify event page content
- Updates venue records with discovered events_url

---

### `scan-apis` - Scan for API Endpoints

Scan venue websites to detect API endpoints for direct event fetching.

```bash
# Scan all venues
python -m venue_scout.cli scan-apis

# Limit to 100 venues
python -m venue_scout.cli scan-apis --limit 100

# Scan specific category only
python -m venue_scout.cli scan-apis --category "music venue"
```

**What it does:**
- Uses Playwright to intercept XHR/Fetch requests
- Detects JSON API endpoints returning event data
- Saves discovered endpoints for future direct API fetching
- Venues with APIs skip slower web scraping

---

### `scan-ticketmaster` - Check Ticketmaster Presence

Scan venues to check if they're listed on Ticketmaster.

```bash
# Scan all ticketed-category venues
python -m venue_scout.cli scan-ticketmaster

# Limit to 50 venues
python -m venue_scout.cli scan-ticketmaster --limit 50
```

**What it does:**
- Checks music venues, theaters, comedy clubs against Ticketmaster API
- Saves Ticketmaster venue ID for direct event fetching
- Marks venues not on Ticketmaster to avoid re-checking
- Venues on Ticketmaster skip web scraping entirely

---

## Examples

### Quick Start

```bash
# Fetch events for popular music venues
python -m venue_scout.cli --force --workers 5 venues \
  "BAM" "Irving Plaza" "Webster Hall" "Bowery Ballroom"

# See which events match your music taste
python -m venue_scout.cli matched
```

### Full Venue Scan

```bash
# 1. First, scan for Ticketmaster presence (one-time)
python -m venue_scout.cli scan-ticketmaster

# 2. Scan for API endpoints (one-time)
python -m venue_scout.cli scan-apis --limit 200

# 3. Fetch all music venue events with parallelization
python -m venue_scout.cli --force --workers 10 --export categories "music venue"

# 4. View matched events
python -m venue_scout.cli matched
```

### Maintenance

```bash
# Check cache status
python -m venue_scout.cli status

# Force refresh stale venues
python -m venue_scout.cli --force categories "comedy clubs"

# Clear and re-fetch specific venue
python -m venue_scout.cli clear --venue "BAM"
python -m venue_scout.cli --force venues "BAM"
```

---

## Event Sources

The CLI uses multiple sources to fetch events:

| Source | Used For | Speed |
|--------|----------|-------|
| **Ticketmaster API** | Venues with TM IDs | Fast |
| **Direct API** | Venues with discovered API endpoints | Fast |
| **Web Scraping** | Venues without API/TM | Slower |

Priority order:
1. If venue has `ticketmaster_venue_id` → Use Ticketmaster API
2. If venue has `api_endpoint` → Use direct API
3. Otherwise → Scrape website with Jina Reader + LLM parsing

---

## Parallelization

Use `--workers N` to speed up fetching:

| Workers | 100 Venues | 1000 Venues |
|---------|------------|-------------|
| 1 | ~5 min | ~50 min |
| 5 | ~1 min | ~10 min |
| 10 | ~30 sec | ~5 min |

**Note:** Rate limits may affect very high parallelization. Recommended: 5-10 workers.

---

## Data Storage

- **Venue data**: Google Sheets (cached locally)
- **Event cache**: `venue_events_metadata.json`
- **Ticketmaster IDs**: Stored in venue records
- **API endpoints**: Stored in venue records
