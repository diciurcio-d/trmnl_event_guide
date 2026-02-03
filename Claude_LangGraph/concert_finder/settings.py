"""Central settings for concert finder."""

# YouTube Music settings
YTMUSIC_NUM_SONGS = 5000          # Number of liked songs to analyze
YTMUSIC_MIN_SONGS_PER_ARTIST = 3  # Minimum liked songs to include an artist
YTMUSIC_MAX_ARTISTS = 100          # Maximum artists to search for

# Ticketmaster settings
TICKETMASTER_MONTHS_AHEAD = 6     # How far ahead to search for concerts
TICKETMASTER_DMA_ID = "345"       # DMA ID (345 = NYC metro area)
TICKETMASTER_USE_ATTRACTION_ID = True  # Use attraction ID for precise matches

# Rate limiting
REQUEST_DELAY_SECONDS = 0.25      # Delay between Ticketmaster requests
