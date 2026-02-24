# Use Python 3.11 slim as base — small footprint, stable
FROM python:3.11-slim

# Install system deps needed by Playwright and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker can cache this layer.
# If requirements.txt doesn't change, pip install is skipped on rebuilds.
COPY requirements.txt .

# Install Python dependencies + gunicorn (our production server)
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Install Playwright and download Chromium with all its system dependencies.
# --with-deps handles OS-level packages (libglib, libnss, etc.) automatically.
RUN playwright install --with-deps chromium

# Copy application code (secrets and data/ are NOT copied — see .dockerignore)
COPY . .

# Make entrypoints executable
RUN chmod +x /app/entrypoint.sh /app/job_entrypoint.sh

# Create directories that Cloud Run will mount over at runtime:
#   - venue_scout/data/  -> GCS bucket (persistent caches, FAISS index)
#   - config/            -> Secret Manager (API keys, service account)
# These just need to exist as mount points in the image.
RUN mkdir -p venue_scout/data venue_scout/outputs/website_validator config

# Run as non-root for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run injects PORT at runtime (default 8080)
ENV PORT=8080
EXPOSE 8080

# gunicorn flags:
#   --workers 1       Single worker process (Cloud Run scales by adding containers, not processes)
#   --threads 8       Multiple threads to handle concurrent requests within that worker
#   --timeout 120     Long timeout for Playwright scraping and LLM calls
#   exec              Replaces the shell so signals (SIGTERM) reach gunicorn directly
CMD ["/app/entrypoint.sh"]
