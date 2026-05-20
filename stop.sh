#!/bin/bash
# Stop FLUX.1-dev Studio

set -e

cd "$(dirname "$0")"

echo "Stopping FLUX.1-dev services..."
docker compose down

echo "Done!"
