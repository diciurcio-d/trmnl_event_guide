# Venue Scout Layout

This package keeps runtime code at the top level and groups non-code artifacts into dedicated folders.

## Structure

- `venue_scout/*.py`
  - Runtime modules (discovery, validation, event fetching, API/server entrypoints).
- `venue_scout/data/`
  - Durable JSON files and local caches.
  - Includes: `seed_venues.json`, `venues.json`, `local_events_cache.json`, `places_cache.json`, `travel_cache.json`, `venue_cache_metadata.json`.
- `venue_scout/outputs/website_validator/`
  - Validation run outputs and benchmark JSON artifacts.
- `venue_scout/docs/`
  - Operational docs and CLI usage docs.
- `venue_scout/paths.py`
  - Central path constants used by runtime modules.

## Notes

- New file paths should be defined in `venue_scout/paths.py` and imported where needed.
- This avoids file sprawl in the package root and makes cache/output locations explicit.
