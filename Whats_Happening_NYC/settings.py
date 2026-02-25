"""Central settings for concert finder and event scraping."""

# YouTube Music settings
YTMUSIC_NUM_SONGS = 5000          # Number of liked songs to analyze
YTMUSIC_MIN_SONGS_PER_ARTIST = 3  # Minimum liked songs to include an artist
YTMUSIC_MAX_ARTISTS = 100         # Maximum artists to search for

# Ticketmaster settings
TICKETMASTER_MONTHS_AHEAD = 6     # How far ahead to search for concerts
TICKETMASTER_DMA_ID = "345"       # DMA ID (345 = NYC metro area)
TICKETMASTER_USE_ATTRACTION_ID = True  # Use attraction ID for precise matches

# =============================================================================
# API RATE LIMITS & DELAYS
# =============================================================================
# Each API has different rate limits. We set conservative delays to stay safe.
#
# GEMINI API (gemini-3-flash-preview):
#   - Free tier: 10-15 RPM (requests/min), 250K TPM (tokens/min), 100-1000 RPD (requests/day)
#   - Tier 1 (paid): 150-300 RPM, 1M TPM, 1500 RPD
#   - Tier 2: 500-1500 RPM, 2M TPM, 10000 RPD
#   - Docs: https://ai.google.dev/gemini-api/docs/rate-limits
#
# GOOGLE PLACES API (New):
#   - No hard QPS limit documented, but ~100 QPS is safe
#   - Free tier: ~300 requests/day for Text Search
#   - Paid: Per-request pricing, no strict daily limit
#   - Docs: https://developers.google.com/maps/documentation/places/web-service/usage-and-billing
#
# TICKETMASTER DISCOVERY API:
#   - 5 requests/second (5 QPS), 5000 requests/day
#   - Docs: https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/
#
# JINA READER API:
#   - Free tier: 200 RPM, varies by plan
#   - Can return 402 when rate limited
#   - Docs: https://jina.ai/reader/
#
# =============================================================================

# Ticketmaster rate limiting
TICKETMASTER_REQUEST_DELAY = 0.25  # 4 QPS (limit is 5 QPS)

# Jina Reader rate limiting
JINA_REQUEST_DELAY = 1.5           # ~40 RPM to avoid burst failures

# Google Places API rate limiting
VENUE_ENRICH_PLACES_DELAY_SEC = 0.1  # 10 QPS (limit is ~100 QPS)

# Gemini API rate limiting
VENUE_DESCRIPTION_DELAY_SEC = 0.5    # 2 QPS / 120 RPM (free tier limit is 10-15 RPM, paid is 150+ RPM)

# LLM (Gemini) settings
GEMINI_MODEL = "gemini-2.5-flash-lite"   # Fast/low-cost default model for classification/extraction tasks.
QUERY_RANKING_MODEL = "gemini-2.5-flash-lite"  # Primary model for chat query ranking.
QUERY_DATE_TOOL_MODEL = "gemini-2.5-flash-lite"  # Model for date-intent/tool extraction in chat queries.
GEMINI_TEMPERATURE = 0.0                 # Deterministic output for repeatable validation/ranking behavior.
GEMINI_SEED = 20260213                   # Fixed seed to reduce run-to-run variation in LLM responses.
GEMINI_MAX_RETRIES = 4                   # Total Gemini attempts per prompt when transient provider errors occur.
GEMINI_RETRY_INITIAL_DELAY_SEC = 1.5     # First retry sleep for transient Gemini failures (503/overload/timeouts).
GEMINI_RETRY_BACKOFF_MULTIPLIER = 2.0    # Multiplier applied to each subsequent Gemini retry delay.
GEMINI_RETRY_MAX_DELAY_SEC = 12.0        # Cap to prevent unbounded Gemini backoff sleeps.
GEMINI_HTTP_TIMEOUT_SEC = 30             # Per-request HTTP timeout for Gemini API calls.
QUERY_DATE_TOOL_TIMEOUT_SEC = 10         # Date-tool timeout for interactive event queries.
QUERY_LLM_TIMEOUT_SEC = 12               # Ranking LLM timeout for interactive event queries.
QUERY_LLM_MAX_RETRIES = 1                # Single primary ranking attempt before timeout-model fallback.
QUERY_LLM_TIMEOUT_FALLBACK_MODELS = (    # Timeout fallback order for query ranking.
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)
QUERY_SEMANTIC_SKIP_BROAD = True         # Skip semantic retrieval for broad/vague queries.
QUERY_SEMANTIC_SPECIFICITY_THRESHOLD = 2  # Minimum specificity score to enable semantic retrieval.
QUERY_BROAD_LEXICAL_TOP_K = 120          # Candidate cap for broad-query lexical pre-rank.
EVENT_PARSE_LLM_TIMEOUT_SEC = 12         # Scrape parsing timeout for interactive event extraction.
EVENT_PARSE_LLM_MAX_RETRIES = 1          # Single primary parse attempt before timeout-model fallback chain.
EVENT_PARSE_LLM_TIMEOUT_FALLBACK_MODELS = (  # Timeout fallback order for event-page parsing.
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)
EVENT_PARSE_LLM_CHUNK_THRESHOLD = 7000   # Use chunked extraction for long pages to reduce timeout risk.
EVENT_PARSE_LLM_CHUNK_SIZE = 5000        # Approx chars per LLM extraction chunk.
EVENT_PARSE_LLM_CHUNK_OVERLAP = 350      # Overlap between chunks to avoid boundary misses.
EVENT_PARSE_LLM_MAX_CHUNKS = 8           # Hard cap on extraction chunks per page.

# Semantic retrieval settings (query -> embeddings -> FAISS -> top-N -> LLM ranking)
SEMANTIC_RETRIEVAL_ENABLED = True
SEMANTIC_EMBEDDING_MODEL = "gemini-embedding-001"
SEMANTIC_TOP_K = 60
SEMANTIC_EMBED_BATCH_SIZE = 64
# Gemini embedding throughput:
# - ~3000 embedding requests/minute (practically enforced via contents-per-minute)
# - up to ~1,000,000 tokens/minute on paid tier
SEMANTIC_EMBED_TOKENS_PER_MINUTE = 1_000_000
SEMANTIC_EMBED_CONTENTS_PER_MINUTE = 2800
SEMANTIC_EMBED_REQUESTS_PER_MINUTE = 1200
SEMANTIC_EMBED_RATE_LIMIT_HEADROOM = 0.9
SEMANTIC_EMBED_MAX_TOKENS_PER_CALL = 12000
LLM_EVENT_CONTEXT_LIMIT = 100

# Jina Reader settings
JINA_TIMEOUT_SEC = 12                    # Max wait per Jina fetch before falling back or marking as failed.

# HTTP timeout settings
GOOGLE_DISTANCE_MATRIX_BATCH_TIMEOUT_SEC = 20  # Batch Distance Matrix request timeout.
GOOGLE_DISTANCE_MATRIX_SINGLE_TIMEOUT_SEC = 10  # Single origin->destination Distance Matrix timeout.
TICKETMASTER_REQUEST_TIMEOUT_SEC = 15     # Ticketmaster Discovery API request timeout.
TRMNL_WEBHOOK_TIMEOUT_SEC = 10            # TRMNL webhook POST timeout.
VENUE_DISCOVERY_SEARCH_TIMEOUT_SEC = 15   # Jina Search API timeout during venue discovery.
VENUE_ENRICH_PLACES_TIMEOUT_SEC = 10      # Google Places lookup timeout while enriching addresses.
SOURCE_ANALYZER_FETCH_TIMEOUT_SEC = 10    # Initial page fetch timeout in source analyzer tool.

# Scraper/API detector timeouts
GENERIC_SCRAPER_PLAYWRIGHT_GOTO_TIMEOUT_MS = 45000  # Playwright navigation timeout for generic scraper.
GENERIC_SCRAPER_API_TIMEOUT_SEC = 20      # Direct API JSON fetch timeout in generic scraper.
VENUE_API_DETECTOR_GOTO_TIMEOUT_MS = 30000  # Playwright navigation timeout for API endpoint detection.
VENUE_API_DETECTOR_FETCH_TIMEOUT_SEC = 15  # Direct fetch timeout when calling detected API endpoints.
VENUE_API_DETECTOR_SCROLL_WAIT_MS = 1000   # Pause after scroll to trigger lazy-loaded API requests.

# Website validator settings
WEBSITE_VALIDATOR_MAX_ATTEMPTS = 3       # How many search/verify loops to try per venue before giving up.
WEBSITE_VALIDATOR_ATTEMPT_DELAY_SEC = 0.5  # Pause between attempts to reduce provider throttling bursts.
WEBSITE_VALIDATOR_CLI_DELAY_SEC = 0.4    # Sequential CLI pause between venues during validation runs.
WEBSITE_VALIDATOR_BATCH_DELAY_SEC = 1.0  # Batch helper pacing when validating lists sequentially.
WEBSITE_VALIDATOR_FIND_EVENTS_DURING_VALIDATION = True  # Run events-page discovery in the same validate-websites pass to avoid a second crawl.
WEBSITE_FIND_EVENTS_CLI_DELAY_SEC = 1.5  # Sequential CLI pause between events-page discovery checks.
WEBSITE_VALIDATOR_EVENTS_BUDGET_SEC = 20 # Base per-venue events-page probe budget.
WEBSITE_VALIDATOR_EVENTS_BUDGET_EXTENSION_SEC = 8  # Extra budget granted when probes fail due to transient provider issues.
WEBSITE_VALIDATOR_MAX_EVENTS_BUDGET_SEC = 35  # Hard cap for adaptive budget extensions (keeps long-tail bounded).
WEBSITE_VALIDATOR_MAX_EVENT_CANDIDATES = 8  # Max candidate event URLs to probe per venue.
WEBSITE_VALIDATOR_MAX_IFRAME_CANDIDATES = 1  # Max iframe URLs to probe for embedded event feeds.
WEBSITE_VALIDATOR_CANDIDATE_PREFLIGHT_TIMEOUT_SEC = 2  # Fast URL preflight timeout before costly Jina probes.
WEBSITE_VALIDATOR_JINA_TIMEOUT_SEC = 6   # Per-call Jina timeout for event-page content probes.
WEBSITE_VALIDATOR_JINA_RETRIES = 2       # Attempts per Jina read before treating it as failed.
WEBSITE_VALIDATOR_JINA_RETRY_DELAY_SEC = 0.35  # Delay between Jina retries to reduce burst failures.
WEBSITE_VALIDATOR_JINA_SEARCH_TIMEOUT_SEC = 8  # Timeout for Jina Search fallback used to find non-standard event URLs.
WEBSITE_VALIDATOR_JINA_SEARCH_MAX_RESULTS = 8  # Max same-domain URLs accepted from Jina Search fallback.
WEBSITE_VALIDATOR_SKIP_JINA = True             # Default to direct HTML fetch; avoid Jina dependency unless explicitly re-enabled.
WEBSITE_VALIDATOR_IFRAME_MEDIA_HOST_PATTERNS = (  # iframe hosts treated as media embeds, not event listing pages.
    "instagram.com",
    "cdninstagram.com",
    "fbcdn.net",
    "facebook.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "tiktok.com",
    "soundcloud.com",
    "spotify.com",
    "wistia.com",
)
WEBSITE_VALIDATOR_IFRAME_MEDIA_PATH_EXTENSIONS = (".mp4", ".m3u8", ".mp3", ".mov", ".webm")
WEBSITE_VALIDATOR_HTML_TIMEOUT_SEC = 12  # Direct HTML fetch timeout used as Jina fallback/probing.
WEBSITE_VALIDATOR_HEALTH_TIMEOUT_SEC = 6 # HEAD/GET timeout when checking if site is dead/unreachable.

# Event fetcher settings
EVENT_FETCHER_SKIP_JINA = True    # Default to raw HTML parsing; avoid Jina dependency unless explicitly re-enabled.
EVENT_FETCHER_PLAYWRIGHT_FALLBACK = True  # Use Playwright when raw HTML fetch fails (403), empty content, or JS calendar detected
EVENT_FETCHER_PLAYWRIGHT_TIMEOUT_MS = 12000  # Playwright navigation timeout (faster: domcontentloaded, not networkidle)
EVENT_FETCHER_PLAYWRIGHT_WAIT_MS = 1500  # Wait for JS to render after domcontentloaded (ms)

# Event scraping settings
SCRAPER_MAX_CONTENT_CHARS = 25000  # Max chars to send to LLM for parsing
SCRAPER_PLAYWRIGHT_WAIT = 3        # Default seconds to wait for Playwright pages
SCRAPER_CLOUDFLARE_WAIT = 2        # Seconds between Cloudflare challenge checks
SCRAPER_CLOUDFLARE_MAX_CHECKS = 15 # Max Cloudflare challenge checks

# Cache settings
CACHE_THRESHOLD_DAYS = 7          # Days before cached data is considered stale

# User preferences (can be overridden in config.json)
DEFAULT_MAX_TRAVEL_MINUTES = 60   # Default max travel time filter
DEFAULT_HOME_ADDRESS = ""         # User's home address for distance calculations

# Venue scouting settings
VENUE_SEARCH_DELAY = 2.0          # Delay between web searches (seconds)
VENUE_CACHE_THRESHOLD_DAYS = 30   # Venues change less often than events

# Comprehensive venue categories organized by event type
VENUE_CATEGORIES = {
    # Performing Arts
    "performing_arts": [
        "broadway theaters",
        "off-broadway theaters",
        "dance venues ballet contemporary",
        "immersive theater venues",
        "burlesque drag show venues",
        "circus aerial performance venues",
        "magic show venues",
        "comedy clubs standup",
        "improv comedy theaters",
        "variety show venues cabaret",
    ],
    # Visual Arts & Film
    "visual_arts_film": [
        "museums",
        "history museums",
        "science museums",
        "art galleries chelsea",
        "art galleries brooklyn",
        "movie theaters repertory cinema",
        "photography galleries",
    ],
    # Music
    "music": [
        "concert halls",
        "live music venues",
        "jazz clubs",
        "rock music venues indie",
        "classical music venues",
        "opera houses",
    ],
    # Sports & Recreation
    "sports_recreation": [
        "sports arenas stadiums",
        "ice skating rinks",
    ],
    # Knowledge & Literature
    "knowledge_literature": [
        "bookstores with author events",
        "libraries with public events",
        "lecture halls universities",
    ],
    # Social & Community
    "social_community": [
        "cultural centers",
        "community centers with events",
        "event spaces party venues",
        "rooftop bars with events",
        "bars with live events comedy",
    ],
    # Outdoor & Parks
    "outdoor": [
        "outdoor concert venues amphitheaters",
        "outdoor event spaces gardens",
        "public plazas with programming",
        "parades and parade routes",
    ],
    # Parks - NYC Parks Dept hosts many events
    "parks": [
        "NYC parks with free events",
        "outdoor movies parks NYC",
        "summer concerts parks NYC",
        "Shakespeare in the Park NYC",
        "parks with programming activities",
    ],
    # Discovery queries - broader searches to catch notable venues
    "discovery": [
        "best live music venues",
        "best concert venues",
        "best nightclubs",
        "warehouse party venues brooklyn",
        "alternative performance spaces",
        "unique event venues",
        "best comedy shows",
        "best things to do at night",
    ],
}

# Flattened list for simple iteration
VENUE_CATEGORIES_ALL = [
    cat for cats in VENUE_CATEGORIES.values() for cat in cats
]

# MVP subset for quick testing
VENUE_CATEGORIES_MVP = [
    "comedy clubs standup",
    "live music venues",
    "broadway theaters",
    "museums",
    "bookstores with author events",
    "cultural centers",
    "parks with events concerts",
]

# Venue event fetching settings
VENUE_EVENT_CACHE_DAYS = 7        # Days before venue events are stale
VENUE_API_DETECTION_WAIT = 3     # Seconds to wait for API detection with Playwright
VENUE_FETCH_DELAY = 0.5          # Delay between venue fetches (seconds)
EVENT_FETCHER_HTML_TIMEOUT_SEC = 20  # Raw HTML timeout when extracting JSON-LD/iframes from venue pages.
NYC_PARKS_OPEN_DATA_FEED_URL = "https://www.nycgovparks.org/xml/events_300_rss.json"  # NYC Parks open-data upcoming events feed.
