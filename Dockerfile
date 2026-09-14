FROM python:3.12-slim

WORKDIR /app

# System deps: none required beyond what pip installs (no native PDF libs
# needed -- pypdf is pure-Python).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/


# Persistent checkpoint DB lives here when CHECKPOINT_BACKEND=sqlite;
# mount a volume at this path to survive container restarts.
RUN mkdir -p /data
ENV CHECKPOINT_DB_PATH=/data/checkpoints.sqlite

# No network port is exposed -- this is a CLI tool, not a web server.
# Run with `docker run --env-file .env myimage "your question"`.
ENTRYPOINT ["python", "-m", "app"]
