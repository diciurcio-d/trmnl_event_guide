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
