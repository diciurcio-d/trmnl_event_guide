# Venue Scout MVP Boundary

This folder is the production path for the NYC MVP.

## Scope
- Primary runtime: `venue_scout/server.py` + `venue_scout/index.html`
- Primary data pipeline: `venue_scout` CLI + API modules
- Canonical storage: Google Sheets (venues + venue events)

## Non-Goals (for now)
- Full multi-city rollout
- Replacing Google Sheets
- Folding all prototype code from repository root into this flow

## LangGraph Integration (later)
- LangGraph should consume Venue Scout API outputs as a client layer.
- Venue discovery/fetch/cache contracts live here and remain the system of record.
