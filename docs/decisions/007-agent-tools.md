# Decision 007: Agent Tools Integration

**Date:** December 2024  
**Status:** Accepted

## Context

Agents need programmatic access to tinyrag functionality.

## Decision

Expose as Anthropic tools: `create_tinyrag`, `search_tinyrag`, `get_tinyrag_info`.

## Rationale

- Agents need programmatic access
- Maintains "zero infrastructure" (library, not API)
- Portable files work well for agent workflows
- Can be adapted to other frameworks

## Consequences

- Three tool functions
- Tool schemas (JSON)
- Zero infrastructure (agents use library directly)
- Portable files enable agent workflows

## Tools

1. `create_tinyrag` - Create .tinyrag file from documents
2. `search_tinyrag` - Search a .tinyrag file
3. `get_tinyrag_info` - Get metadata about file
