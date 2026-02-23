#!/bin/bash
set -e

# Cloud Run mounts secret volumes as read-only, making the directory immutable.
# Instead we write all configs to /tmp/config/ (always writable) and
# symlink /app/config -> /tmp/config so existing code paths work unchanged.

mkdir -p /tmp/config

[ -n "$APP_CONFIG_JSON" ]      && printf '%s' "$APP_CONFIG_JSON"      > /tmp/config/config.json
[ -n "$SHEETS_CONFIG_JSON" ]   && printf '%s' "$SHEETS_CONFIG_JSON"   > /tmp/config/sheets_config.json
[ -n "$SERVICE_ACCOUNT_JSON" ] && printf '%s' "$SERVICE_ACCOUNT_JSON" > /tmp/config/service_account.json

# Symlink /app/config -> /tmp/config (remove the empty dir the image created first)
[ -d /app/config ] && rmdir /app/config 2>/dev/null || true
[ ! -L /app/config ] && ln -s /tmp/config /app/config

exec gunicorn \
    --bind "0.0.0.0:$PORT" \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --log-level info \
    venue_scout.server:app
