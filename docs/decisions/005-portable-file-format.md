# Decision 005: Portable .sqlite File Format

**Date:** February 2026
**Status:** Accepted (Updated)

## Context

Need portable knowledge bases that can be saved, shared, and loaded anywhere.  
**Critical requirement:** Must support incremental updates for agent memory use case.

## Decision

Single `.sqlite` file as **SQLite database** containing chunks, embeddings, and metadata.  
FAISS index is **rebuilt on load** (fast for workspace-scale).

**File Extension:** `.sqlite` (honest about format, works directly with sqlite3 CLI)

## Rationale

- ✅ **Portable** (single file, git commit, share)
- ✅ **Incremental updates** (INSERT/UPDATE operations, no archive recreation)
- ✅ **Production-ready** (works in Docker, multiple namespaced files)
- ✅ **Fast saves** (no archive extraction/recreation)
- ✅ **Fast loads** (rebuild index ~10-50ms for 1000 chunks)
- ✅ **Standard format** (SQLite, Python stdlib)
- ✅ **Queryable** (can query chunks by source, date, etc.)
- ✅ **Inspectable** (can use sqlite3 CLI directly)

## Consequences

- ✅ Single file (portable)
- ✅ Incremental updates supported
- ✅ Fast saves (SQL INSERT/UPDATE)
- ⚠️ Index rebuilt on load (but fast for workspace-scale)
- ⚠️ Less human-readable (but can inspect with sqlite3 CLI)
- ✅ Production-ready for context layer router pattern
- ✅ Honest file extension (.sqlite makes format clear)

## Format

```
my_knowledge.sqlite (SQLite database)
├── metadata table      # Config, version, creation date
├── chunks table        # Chunk objects (text, source, index, metadata)
├── embeddings table    # Embeddings (stored as BLOB)
└── (FAISS index rebuilt on load)
```

## Schema

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    index INTEGER NOT NULL,
    metadata TEXT,  -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,  -- NumPy array as bytes
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);
```

## Update History

**Original Decision (March 2026):** Zip archive format (`.raglet` file)  
**Updated (March 2026):** Changed to SQLite (`.sqlite` file) for incremental updates support

**Reason for Change:**
- Agent memory use case requires incremental updates
- Zip archives can't efficiently update files inside
- SQLite supports true incremental updates (INSERT/UPDATE)
- Rebuilding FAISS index on load is fast enough (~10-50ms)
- Production use case (context layer router) benefits from SQLite
- `.sqlite` extension is honest about format and works with sqlite3 CLI directly
