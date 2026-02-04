"""Central settings for concert finder and event scraping."""

# YouTube Music settings
YTMUSIC_NUM_SONGS = 5000          # Number of liked songs to analyze
YTMUSIC_MIN_SONGS_PER_ARTIST = 3  # Minimum liked songs to include an artist
YTMUSIC_MAX_ARTISTS = 100         # Maximum artists to search for

# Ticketmaster settings
TICKETMASTER_MONTHS_AHEAD = 6     # How far ahead to search for concerts
TICKETMASTER_DMA_ID = "345"       # DMA ID (345 = NYC metro area)
TICKETMASTER_USE_ATTRACTION_ID = True  # Use attraction ID for precise matches

# Rate limiting
TICKETMASTER_REQUEST_DELAY = 0.25  # Delay between Ticketmaster requests (seconds)
JINA_REQUEST_DELAY = 1.5           # Delay between Jina requests (20 req/min limit, but processing adds buffer)

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
