FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY raglet/ ./raglet/

# Install raglet and dependencies
RUN uv pip install --system --no-cache -e .

# Set entrypoint
ENTRYPOINT ["raglet"]

# Default command
CMD ["--help"]
