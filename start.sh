#!/bin/bash
# Start FLUX.1-dev Studio

set -e

cd "$(dirname "$0")"

echo "Starting FLUX.1-dev inference service..."
docker compose up -d --build flux-inference

echo "Waiting for inference service to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8630/v1/health/ready > /dev/null 2>&1; then
        echo "Inference service is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

echo "Starting FLUX.1-dev app..."
docker compose up -d --build app

echo ""
echo "FLUX.1-dev Studio is running!"
echo "Access at: http://localhost:8631/flux"
echo "Inference API: http://localhost:8630"
