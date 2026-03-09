# Decision 010: Workspace-Scale Limits and Graceful Degradation

**Date:** 2026-03-07
**Updated:** 2026-03-09 (benchmark data, chunk_size=256, MPS fp16)
**Status:** Accepted

## Context

raglet is described as "workspace-scale" in its design principles, but that term was never given concrete numbers. This left two problems:

1. Users have no clear signal when they are about to leave the safe operating envelope
2. The library silently degrades — search gets slow, RAM spikes, and in the worst case the process OOMs — with no explanation

### Measured performance (2026-03-09)

Benchmarks on Apple Silicon MPS fp16 with `all-MiniLM-L6-v2`, chunk_size=256 tokens.
Full results: [`benchmarks/SCALE_REPORT.md`](../../benchmarks/SCALE_REPORT.md)

| Text size | Chunks  | Build     | Search p50 | Save     | Load    |
|----------:|--------:|----------:|-----------:|---------:|--------:|
|    102 KB |     140 |     1.2s  |     4.8 ms |   4.2 ms |  5.2 ms |
|      1 MB |   1,390 |     3.6s  |     3.7 ms |  12.2 ms |  9.8 ms |
|      5 MB |   6,949 |    17.5s  |     6.0 ms |  64.7 ms | 27.4 ms |
|     10 MB |  13,899 |    34.6s  |     6.3 ms |   122 ms | 51.4 ms |
|     20 MB |  27,797 |    1m10s  |     7.1 ms |   263 ms |  106 ms |
|     50 MB |  69,494 |    3m02s  |     9.2 ms |   655 ms |  290 ms |
|    100 MB | 138,989 |    6m00s  |    10.4 ms |  1227 ms |  574 ms |

Embedding throughput plateaus at ~395 chunks/sec (MPS fp16). CPU throughput is ~106 chunks/sec (~10-20x slower).

Key observations:
- **Search stays under 11 ms** up to 139K chunks (100 MB) — FAISS brute-force is still fast at this scale
- **Build time scales linearly** with text size — it is entirely embedding-bound
- **Incremental appends** (`add_chunks`, `add_file`) only embed new content, so a 100 KB append to a 100 MB raglet takes ~1s, not 6 minutes
- **Save/load scale linearly** with chunk count but stay under 1.3s even at 139K chunks

The scalability analysis (`SCALABILITY_ANALYSIS.md`) estimates theoretical limits at extreme scale:
- At 200,000 chunks (~600 MB RAM, <50 ms search): comfortable on a 16 GB laptop
- At 500,000 chunks (~3 GB RAM, ~150 ms search): noticeably degraded
- At 1,000,000 chunks (~6 GB RAM, ~400 ms search): unusable without a GPU and large RAM

The two most dangerous failure modes are silent: `IndexFlatIP` degrades linearly with chunk count and Python heap grows with chunk objects. Neither raises an error — they just get slow, then OOM.

Note: the original `np.vstack` O(n²) bottleneck in `add_chunks` was fixed in ADR 009 (deferred materialisation). Incremental ingestion is now O(1) per append.

## Decision

Introduce **explicit, configurable workspace-scale limits** with a two-tier (soft/hard) enforcement model.

**Default limits:**

| Limit | Soft (warn) | Hard (error) |
|---|---|---|
| Total chunks | 200,000 | 500,000 |
| Total source text | 300 MB | 750 MB |
| Single file size | 10 MB | 50 MB |

These limits are conservative. The benchmarks show raglet handles 139K chunks (100 MB) comfortably on MPS — search stays under 11 ms, save under 1.3s, load under 600 ms. The soft limit at 200,000 gives a ~1.4x margin above the largest measured size, and the hard limit at 500,000 is where RAM and search latency begin to degrade meaningfully per the scalability analysis.

**New components:**

- `WorkspaceLimitsConfig` — configurable limit values, lives in `RAGletConfig`
- `WorkspaceLimitError` — raised at hard limits, carries `suggestion` string
- `WorkspaceLimitChecker` — single-responsibility enforcer, injected into the pipeline

**Enforcement points:** pre-scan (source size), per-file (individual file), post-chunk (total chunks).

**Bypass:** Users can raise or disable limits explicitly via `WorkspaceLimitsConfig`. This is allowed but requires deliberate opt-in.

See `docs/proposals/WORKSPACE_SCALE_DESIGN.md` for full architecture.

## Rationale

**Two tiers rather than one hard stop:** A hard stop at 200k would be frustrating for users who are at 210k and just need to push through. A soft warning lets them decide; a hard stop at 500k prevents genuinely bad outcomes (OOM, minute-long search latency).

**Enforce at ingestion, not at search:** Search latency is a runtime concern and varies by machine. Chunk count is known at ingestion time and is the root cause of all downstream problems. Checking at ingestion gives the user actionable feedback before they've committed to a slow or broken state.

**Configurable rather than hardcoded:** Different machines have different RAM. A 64 GB workstation has a meaningfully different limit than a 8 GB laptop. Making limits configurable lets the library serve both without lying to either.

**Build time is the real user-facing constraint, not search:** Benchmarks show search stays under 11 ms even at 139K chunks (100 MB). The pain point is build time — 35s for 10 MB, 6 minutes for 100 MB on MPS. The "build once, append cheaply" pattern (documented in `docs/USAGE_PATTERNS.md`) is the recommended workflow: pay the build cost once, then append incrementally.

**Error messages must include suggestions:** A limit error with no guidance is a dead end. Every `WorkspaceLimitError` includes at least three concrete alternatives (split by topic, filter at ingestion, escalate to a vector database). This is a first-class design requirement for the error type, not an afterthought.

**`enabled=True` default:** Limits are on by default because the target user is someone who has not read the internals. Power users who have read the internals can disable limits. This is the right default for the stated audience.

## Consequences

- `RAGletConfig` gains a `limits: WorkspaceLimitsConfig` field (non-breaking, backward compatible)
- `from_files()` and `from_directory()` will raise `WorkspaceLimitError` for datasets that exceed the hard limits
- The `WorkspaceLimitError` message explicitly names vector databases as alternatives — this is intentional and honest
- Users with legitimate large-scale use cases are directed to appropriate tools rather than working around a reluctant library
- `CLAUDE.md` Scope Boundaries section should be updated to include these numbers

## Alternatives Considered

**No limits at all (status quo):** Rejected. Silent OOM and minute-long search are worse than a clear error. The library's stated principles require honesty about what it can do.

**Single hard limit only:** Rejected. Too blunt — users who are slightly over the limit deserve a warning before a crash.

**Dynamic limits based on available RAM:** Considered but rejected for this iteration. Detecting available RAM reliably across OS/container/VM environments is harder than it looks, and the benefit does not justify the complexity. A configurable static limit achieves most of the same goal.

**Warn-only, never error:** Rejected. A soft warning that is never enforced trains users to ignore it. The hard limit exists precisely so the warning is credible.
