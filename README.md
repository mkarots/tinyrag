# tinyrag

**Portable memory for small text corpora.**

Create searchable `.tinyrag` files from your documents - no infrastructure, no servers, just `pip install tinyrag`.

## Quick Start

```python
from tinyrag import TinyRAG

# Create from files
rag = TinyRAG.from_files(["doc.txt", "notes.md"])

# Get chunks
chunks = rag.get_all_chunks()
```

## Installation

```bash
pip install tinyrag
```

For development:

```bash
pip install -e ".[dev,all]"
```

## Features

- ✅ Extract text from .txt and .md files
- ✅ Intelligent chunking with sentence awareness
- ✅ Portable `.tinyrag` file format (coming soon)
- ✅ Vector search (coming soon)
- ✅ Zero infrastructure - works offline

## Development

### Setup

```bash
make install-dev
```

### Run Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-e2e          # E2E tests only
```

### Linting & Formatting

```bash
make lint      # Check code style
make format     # Format code
make type-check # Type checking
```

### CI/CD

```bash
make ci  # Run full CI pipeline
```

## Architecture

tinyrag follows SOLID principles with clear separation of concerns:

- **core/** - Domain models and orchestrator
- **processing/** - Document extraction and chunking
- **embeddings/** - Embedding generation (coming soon)
- **vector_store/** - Vector storage and search (coming soon)
- **storage/** - File serialization (coming soon)
- **config/** - Configuration system

## Status

🚧 **Early Development** - Milestone 1 (Foundation) in progress

See [FINAL_PLAN.md](FINAL_PLAN.md) for implementation roadmap.

## License

MIT
