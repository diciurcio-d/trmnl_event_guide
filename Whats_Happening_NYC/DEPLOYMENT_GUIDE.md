# What's Happening NYC — Deployment & Update Guide

## Overview

The app runs on **Google Cloud Run** as a containerized Flask app.
Live URL: `https://whats-happening-nyc-996955494439.us-central1.run.app`

The key resources involved:

| Resource | Purpose |
|---|---|
| **Artifact Registry** | Stores Docker images |
| **Cloud Run** | Runs the container (scales to zero when idle) |
| **Secret Manager** | Stores API keys and config files securely |
| **Cloud Storage** | Stores the FAISS index + data files (mounted at `/app/venue_scout/data`) |

---

## How to Update the App (Code Changes)

### 1. Make your code changes locally

Edit files under `Whats_Happening_NYC/`. The main files you'll touch:

- `venue_scout/server.py` — Flask routes, rate limiting, default address
- `venue_scout/index.html` — Frontend UI
- `venue_scout/event_fetcher.py` — Event scraping logic
- `config/config.json` — App settings (address, API keys, etc.)

### 2. Build and push the image

**Use `gcloud builds submit` — do not use `docker build` + `docker push` locally.**

Building locally on an Apple Silicon Mac produces an arm64 image, which Cloud Run rejects. Pushing large layers (Playwright/Chrome ~300MB) from Docker Desktop is also unreliable and slow. `gcloud builds submit` builds natively on amd64 GCP hardware and pushes directly to Artifact Registry, avoiding both problems.

From inside `Whats_Happening_NYC/`:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/gen-lang-client-0046008897/whats-happening-nyc/app:latest \
  --project gen-lang-client-0046008897 \
  .
```

This uploads your source, builds on GCP, and pushes the image automatically. Takes ~3-5 minutes.

### 3. Deploy to Cloud Run

```bash
gcloud run deploy whats-happening-nyc \
  --image=us-central1-docker.pkg.dev/gen-lang-client-0046008897/whats-happening-nyc/app:latest \
  --region=us-central1
```

Cloud Run will spin up the new revision with zero downtime and shift all traffic to it.

---

## How to Update Secrets

Secrets are stored in **Secret Manager** and injected into the container as environment variables at startup. The `entrypoint.sh` script writes them to `/tmp/config/` and symlinks `/app/config` there.

The three secrets are:

| Secret Name | Env Var | File it becomes |
|---|---|---|
| `app-config` | `APP_CONFIG_JSON` | `/app/config/config.json` |
| `sheets-config` | `SHEETS_CONFIG_JSON` | `/app/config/sheets_config.json` |
| `service-account-key` | `SERVICE_ACCOUNT_JSON` | `/app/config/service_account.json` |

### Updating a secret

If you change `config/config.json` (e.g., to update the default address or an API key):

```bash
gcloud secrets versions add app-config \
  --data-file=config/config.json \
  --project=gen-lang-client-0046008897
```

Replace `app-config` with `sheets-config` or `service-account-key` as needed.

**Then redeploy** so Cloud Run picks up the new secret version:

```bash
gcloud run deploy whats-happening-nyc \
  --image=us-central1-docker.pkg.dev/gen-lang-client-0046008897/whats-happening-nyc/app:latest \
  --region=us-central1
```

> Secret changes don't hot-reload — you must redeploy. The deploy is instant (reuses the cached image).

### Viewing current secret contents

```bash
gcloud secrets versions access latest --secret=app-config \
  --project=gen-lang-client-0046008897
```

### If you rotate the service account key

1. Generate a new key in GCP IAM → Service Accounts → `whats-happening-app@...` → Keys
2. Download it to `config/service_account.json`
3. Upload the new version:
   ```bash
   gcloud secrets versions add service-account-key \
     --data-file=config/service_account.json
   ```
4. Redeploy

---

## Updating Data Files (FAISS Index, Caches)

The `venue_scout/data/` directory is served from Cloud Storage bucket `whats-happening-nyc-data`, mounted at `/app/venue_scout/data` via GCS FUSE.

To update a file:

```bash
gsutil cp venue_scout/data/venues.faiss gs://whats-happening-nyc-data/venues.faiss
```

Changes take effect immediately — no redeploy needed. The running container reads from GCS on each access.

To see what's in the bucket:

```bash
gsutil ls -lh gs://whats-happening-nyc-data/
```

---

## Checking App Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=whats-happening-nyc" \
  --limit=50 \
  --format="table(timestamp, textPayload)" \
  --project=gen-lang-client-0046008897
```

Or open Cloud Console → Cloud Run → `whats-happening-nyc` → Logs.

---

## Checking Current Deployment

```bash
# See all revisions and which one is live
gcloud run revisions list \
  --service=whats-happening-nyc \
  --region=us-central1

# See the current service config
gcloud run services describe whats-happening-nyc \
  --region=us-central1
```

---

## Environment Variables (Cloud Run Config)

These are set on the Cloud Run service and don't require a redeploy to view, but do require one to change:

| Env Var | Value |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | `/app/config/service_account.json` |
| `APP_CONFIG_JSON` | *(secret: app-config)* |
| `SHEETS_CONFIG_JSON` | *(secret: sheets-config)* |
| `SERVICE_ACCOUNT_JSON` | *(secret: service-account-key)* |

To see the full service config including env vars:

```bash
gcloud run services describe whats-happening-nyc \
  --region=us-central1 \
  --format=yaml
```

---

## Nightly Event Refresh Job

A Cloud Run Job (`daily-event-refresh`) runs every day at 8:00 AM UTC. It:
1. Selects the 100 venues with the oldest `last_event_fetch` timestamp
2. Fetches their events and writes new ones to the Venue Events Google Sheet
3. Rebuilds the FAISS semantic index incrementally
4. Appends a row to the **"Run Log"** tab in the Venue Events sheet

### Check recent runs

```bash
gcloud run jobs executions list \
  --job=daily-event-refresh \
  --region=us-central1 \
  --project=gen-lang-client-0046008897 \
  --limit=7
```

### Check a run's summary stats

```bash
gcloud logging read \
  'resource.type="cloud_run_job" AND textPayload:"Fetch summary"' \
  --limit=5 \
  --format="table(timestamp, textPayload)" \
  --project=gen-lang-client-0046008897
```

### Trigger a manual run

```bash
gcloud run jobs execute daily-event-refresh \
  --region=us-central1 \
  --project=gen-lang-client-0046008897
```

### Run Log sheet

Each successful run appends a row to the **"Run Log"** tab in the Venue Events spreadsheet with:
`date · started_at · duration_min · venues_processed · venues_with_events · errors · events_before · events_added · events_removed · events_after · status`

`events_removed` = events archived as past or dropped (computed as `before + added − after`).

---

## Newsletter

`venue_scout/newsletter.py` generates a weekly HTML email digest of ~25–30 upcoming events across the next 7 days, with more picks on weekends. It uses Gemini to curate highlights and sends via Gmail SMTP.

### Scheduled jobs

The three newsletters run automatically at **11:00 AM ET** via Cloud Scheduler triggering a Cloud Run Job (`newsletter-weekly`) with arguments:

1. **General Highlights:** Weekly on Thursdays (`0 11 * * 4`)
2. **Talks & Tours:** Weekly on Wednesdays (`0 11 * * 3`)
3. **Music Matches:** Once every two weeks on Tuesdays (`0 11 * * 2`). Note: The scheduler runs weekly, but the script automatically skips odd ISO week numbers unless the `--force` flag is passed.

#### Cloud Run Jobs
- **Job Name:** `newsletter-weekly`
- **Region:** `us-central1`
- **Project:** `gen-lang-client-0046008897`

#### Cloud Scheduler Configuration
Create or update the triggers using HTTP POST requests targeting the Cloud Run Job Run API with arguments in the JSON body:

```bash
# 1. General Highlights (Weekly on Thursdays)
gcloud scheduler jobs create http newsletter-general-thursday \
  --schedule="0 11 * * 4" \
  --time-zone="America/New_York" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/gen-lang-client-0046008897/jobs/newsletter-weekly:run" \
  --http-method=POST \
  --oauth-service-account-email="service-account-key@gen-lang-client-0046008897.iam.gserviceaccount.com" \
  --message-body='{"overrides":{"containerOverrides":[{"args":["--type","general"]}]}}' \
  --headers="Content-Type=application/json" \
  --project=gen-lang-client-0046008897

# 2. Talks & Tours (Weekly on Wednesdays)
gcloud scheduler jobs create http newsletter-talks-wednesday \
  --schedule="0 11 * * 3" \
  --time-zone="America/New_York" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/gen-lang-client-0046008897/jobs/newsletter-weekly:run" \
  --http-method=POST \
  --oauth-service-account-email="service-account-key@gen-lang-client-0046008897.iam.gserviceaccount.com" \
  --message-body='{"overrides":{"containerOverrides":[{"args":["--type","talks"]}]}}' \
  --headers="Content-Type=application/json" \
  --project=gen-lang-client-0046008897

# 3. Music Matches (Bi-weekly on Tuesdays)
gcloud scheduler jobs create http newsletter-music-tuesday \
  --schedule="0 11 * * 2" \
  --time-zone="America/New_York" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/gen-lang-client-0046008897/jobs/newsletter-weekly:run" \
  --http-method=POST \
  --oauth-service-account-email="service-account-key@gen-lang-client-0046008897.iam.gserviceaccount.com" \
  --message-body='{"overrides":{"containerOverrides":[{"args":["--type","music"]}]}}' \
  --headers="Content-Type=application/json" \
  --project=gen-lang-client-0046008897
```

#### Trigger a manual run of a specific newsletter:
```bash
gcloud run jobs execute newsletter-weekly \
  --args="--type,music" \
  --region=us-central1 \
  --project=gen-lang-client-0046008897
```

### Run locally

```bash
# From Whats_Happening_NYC/

# Preview — writes HTML to /tmp/newsletter_preview.html, does not send
python3 -m venue_scout.newsletter --dry-run

# Send to the configured recipient
python3 -m venue_scout.newsletter
```

### Gmail credentials

Stored in `config.json` under the `gmail` key (pushed to Secret Manager as `app-config`):

```json
"gmail": {
  "sender": "diciurcio.david@gmail.com",
  "app_password": "...",
  "recipient": "diciurcio.david@gmail.com"
}
```

To update: edit `config/config.json` then run:

```bash
gcloud secrets versions add app-config \
  --data-file=config/config.json \
  --project=gen-lang-client-0046008897
```

No redeploy needed — the newsletter runs locally or as a job, not via the web server.

### Changing the curation model

Edit `settings.py`:

```python
NEWSLETTER_MODEL = "gemini-2.5-pro"   # richer picks, slower
NEWSLETTER_TIMEOUT_SEC = 180          # give Pro more time for the single full-week call
```

The newsletter uses a single LLM call for the whole week (all days' candidates in one prompt, ~30K tokens). The model logs the prompt size and which model it used at the start of each run.

### Event distribution per day

| Day | Picks |
|-----|-------|
| Mon–Wed | 2 |
| Thu | 3 |
| Fri | 4 |
| Sat | 7 |
| Sun | 6 |

---

## Local Development

Run the app locally (uses OAuth token instead of service account):

```bash
cd Whats_Happening_NYC
python -m flask --app venue_scout.server run --port 5000
```

Make sure `config/config.json`, `config/sheets_config.json`, and `config/google_token.json` exist locally.

---

## GCP Project Reference

- **Project ID**: `gen-lang-client-0046008897`
- **Region**: `us-central1`
- **Artifact Registry repo**: `whats-happening-nyc`
- **Cloud Run service**: `whats-happening-nyc`
- **GCS bucket**: `whats-happening-nyc-data`
- **Service account**: `whats-happening-app@gen-lang-client-0046008897.iam.gserviceaccount.com`

---

## Quick Reference: Full Redeploy

```bash
# From Whats_Happening_NYC/
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/gen-lang-client-0046008897/whats-happening-nyc/app:latest \
  --project gen-lang-client-0046008897 \
  . \
  && gcloud run deploy whats-happening-nyc \
       --image=us-central1-docker.pkg.dev/gen-lang-client-0046008897/whats-happening-nyc/app:latest \
       --region=us-central1
```

## Quick Reference: Config-only Update (no code change)

```bash
gcloud secrets versions add app-config \
  --data-file=config/config.json \
  && gcloud run deploy whats-happening-nyc \
       --image=us-central1-docker.pkg.dev/gen-lang-client-0046008897/whats-happening-nyc/app:latest \
       --region=us-central1
```
