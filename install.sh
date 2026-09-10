#!/usr/bin/env bash
# PRISM one-line installer - matches couchbase-fhir-ce's install pattern:
#
#   curl -sSL https://raw.githubusercontent.com/couchbaselabs/CouchbasePrism/main/config.example.yaml -o config.yaml
#   $EDITOR config.yaml   # fill in Capella, AI Data Plane, AWS, OpenAI, domains
#   curl -sSL https://raw.githubusercontent.com/couchbaselabs/CouchbasePrism/main/install.sh | bash -s -- ./config.yaml
#
# PRISM is a single container (no haproxy/multi-service split the way
# FHIR CE has), so this is a plain `docker run`, not a generated
# docker-compose.yml - a cloned checkout still has docker-compose.yml for
# local dev (`docker compose up --build`), but a bare curl-only install has
# no Dockerfile alongside it to build from, so `docker run` against the
# published image is the honest one-liner here.
set -euo pipefail

IMAGE="${PRISM_IMAGE:-ghcr.io/couchbaselabs/couchbaseprism}"
VERSION="${PRISM_VERSION:-latest}"
PORT="${PRISM_PORT:-8501}"
CONTAINER_NAME="${PRISM_CONTAINER_NAME:-prism}"
CONFIG_PATH="${1:-./config.yaml}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required - https://docs.docker.com/get-docker/" >&2
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "No config file at $CONFIG_PATH." >&2
    echo "Create one from the template first:" >&2
    echo "  curl -sSL https://raw.githubusercontent.com/couchbaselabs/CouchbasePrism/main/config.example.yaml -o config.yaml" >&2
    echo "  \$EDITOR config.yaml   # fill in Capella, AI Data Plane, AWS, OpenAI, domains" >&2
    exit 1
fi

CONFIG_ABS_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

echo "Pulling ${IMAGE}:${VERSION} ..."
docker pull "${IMAGE}:${VERSION}"

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "Replacing existing '$CONTAINER_NAME' container ..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting PRISM ..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${PORT}:8501" \
    -v "${CONFIG_ABS_PATH}:/app/config.yaml:ro" \
    "${IMAGE}:${VERSION}"

echo
echo "PRISM is starting at http://localhost:${PORT}"
echo "  logs:   docker logs -f ${CONTAINER_NAME}"
echo "  stop:   docker stop ${CONTAINER_NAME}"
echo "  update: re-run this script"
