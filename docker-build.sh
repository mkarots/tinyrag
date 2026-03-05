#!/bin/bash
# Build and publish Docker image for raglet

set -e

IMAGE_NAME="mkarots/raglet"
VERSION="${1:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${VERSION}"

# Build image
docker build -t "${IMAGE_NAME}:${VERSION}" .

# Tag as latest if version is not 'latest'
if [ "$VERSION" != "latest" ]; then
    docker tag "${IMAGE_NAME}:${VERSION}" "${IMAGE_NAME}:latest"
fi

echo "✓ Build complete"
echo ""
echo "To publish to Docker Hub:"
echo "  docker login"
echo "  docker push ${IMAGE_NAME}:${VERSION}"
if [ "$VERSION" != "latest" ]; then
    echo "  docker push ${IMAGE_NAME}:latest"
fi
