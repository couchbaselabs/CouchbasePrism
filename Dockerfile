# PRISM - governed retrieval demo/reference architecture.
#
# Docker-only prerequisite: this image bundles Python and every dependency,
# so nothing besides Docker needs to be installed on the host.
#
# config.yaml (real secrets) is NEVER copied into this image - it's built
# from this repo's public checkout and published to a public registry, so
# baking in a deployer's Couchbase/AI Data Plane/AWS/OpenAI credentials
# would leak them to everyone who pulls the image. It's mounted at
# container RUNTIME instead (see docker-compose.yml / install.sh) - copy
# config.example.yaml to config.yaml, fill it in, mount it.
FROM python:3.12-slim

WORKDIR /app

# System deps for pymupdf's PDF parsing - most platforms get a prebuilt
# wheel and skip the compile step entirely; libs stay in case one doesn't.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so a code-only change doesn't invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The system itself, corpus-agnostic
COPY prism/ ./prism/
COPY app/ ./app/
# Sample corpus (ftsprism, 3M's SEC filings, ~31MB) - the Setup tab's S3
# upload step and eval/run_benchmark both read from here; without it, the
# "stage sample PDFs" step in a fresh deployment has nothing to stage.
COPY eval/ ./eval/
COPY design/ ./design/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY manage.py VERSION README.md LICENSE config.example.yaml ./
COPY .streamlit/ ./.streamlit/

# Runs as a non-root user - this is a POC/demo/learning tool, not a product
# where security is paramount (no auth gate, no TLS), but "don't run the
# container as root" is basic hygiene worth keeping regardless.
RUN useradd --create-home --uid 1000 prism \
    && chown -R prism:prism /app
USER prism

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# --server.address=0.0.0.0 - the default (localhost) would be unreachable
# from outside the container. --server.headless - no browser to open inside
# a container, and no need to check for a newer Streamlit version on start.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
