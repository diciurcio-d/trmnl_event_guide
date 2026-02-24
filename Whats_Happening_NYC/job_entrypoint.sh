#!/bin/bash
set -e

# Write secrets from env vars to /tmp/config/ (same as entrypoint.sh)
mkdir -p /tmp/config
[ -n "$APP_CONFIG_JSON" ]      && printf '%s' "$APP_CONFIG_JSON"      > /tmp/config/config.json
[ -n "$SHEETS_CONFIG_JSON" ]   && printf '%s' "$SHEETS_CONFIG_JSON"   > /tmp/config/sheets_config.json
[ -n "$SERVICE_ACCOUNT_JSON" ] && printf '%s' "$SERVICE_ACCOUNT_JSON" > /tmp/config/service_account.json

# Symlink /app/config -> /tmp/config/
[ -d /app/config ] && rmdir /app/config 2>/dev/null || true
[ ! -L /app/config ] && ln -s /tmp/config /app/config

exec python -m venue_scout.job
