# Stage 3: LSM Tree Storage — Phased Implementation Plan

*Parent document: [Scaling Roadmap](./2026-03-20-scaling-roadmap.md), Stage 3.*
*Status: Phases 1-4 complete. Phase 5 pending.*

---

## Overview

Stage 3 replaces full-world snapshots with incremental persistence via an
LSM tree over archetype pages. The goal: persistence cost proportional to the
**mutation rate**, not the **world size**.

This plan breaks Stage 3 into five phases with clear deliverables, testing
criteria, and dependency ordering. Each phase produces a working, testable
artifact. No phase requires the next to be useful.

**Target crate**: `minkowski-lsm` (new workspace member).
**Estimated scope**: 5–8 KLOC across all phases.

---

## Prerequisites (Stage 2.5) — Deferred

Stage 2.5 (SlabPool-only allocation) was deferred. The LSM crate operates on a
separate I/O path that reads page data from World's existing allocator via
`storage::dirty_pages` per-column bitsets. The LSM sorted-run format is
allocator-agnostic — page bytes are copied into the sorted run during flush
regardless of whether the source column is SlabPool-backed or malloc-backed.

Stage 2.5 remains a prerequisite for:
- O_DIRECT / io_uring I/O (Stage 4 RSM)
- Zero-copy page flush (avoiding the memcpy from column → sorted run)
- Page-aligned dirty tracking without per-column bitset overhead

These are Stage 4+ concerns, not Stage 3.

---

## Phase 1: SortedRun Format and FlushWriter

> **Status: Complete.** Implemented in PR #159. 159 unit + 38 integration tests passing.

**Goal**: Define the on-disk format and write dirty pages from L0 → L1.

This is the foundation — every subsequent phase reads/writes SortedRun files.
Get the format right first; optimize later.

### Deliverables

| Component | Description |
|---|---|
| `SortedRun` | Immutable file format: header + sorted page images + sparse index + footer |
| `SortedRunReader` | Memory-mapped reader with page lookup by `(ArchetypeId, page_index)` |
| `FlushWriter` | Iterates `DirtyPageTracker` across all archetypes, writes dirty pages to a new L1 sorted run |

### SortedRun file format (proposed)

```
┌──────────────────────────────────────────────┐
│  Header (64 bytes, cache-line aligned)       │
│  - magic: [u8; 8] "MKLSM01\0"              │
│  - version: u32                              │
│  - archetype_count: u32                      │
│  - page_count: u64                           │
│  - sequence_lo: u64 (WAL sequence range)     │
│  - sequence_hi: u64                          │
│  - crc32: u32 (header checksum)              │
│  - reserved: [u8; 20]                        │
├──────────────────────────────────────────────┤
│  Page images (sorted by archetype_id, then   │
│  page_index within archetype)                │
│  - Each page: raw column bytes, PAGE_SIZE    │
│    rows × item_layout.size(), 64-byte aligned│
├──────────────────────────────────────────────┤
│  Sparse index                                │
│  - Array of (ArchetypeId, page_index, offset)│
│  - Binary-searchable                         │
├──────────────────────────────────────────────┤
│  Footer (64 bytes)                           │
│  - sparse_index_offset: u64                  │
│  - bloom_filter_offset: u64 (0 = absent)     │
│  - total_crc32: u32 (full-file checksum)     │
│  - reserved: [u8; 44]                        │
└──────────────────────────────────────────────┘
```

### Key decisions

- **Page size**: Use `DirtyPageTracker::PAGE_SIZE` (256 rows). Physical byte
  size varies per component layout. This matches the existing dirty tracking
  granularity — no impedance mismatch.
- **Column-per-file vs. all-columns-per-file**: One sorted run contains all
  dirty columns for all dirty archetypes in a single flush. Simpler to
  implement, simpler manifest, atomic flush semantics. Revisit if compaction
  shows this is a bottleneck.
- **CRC32 per page**: Each page image carries a CRC32 for integrity. Verified
  on recovery reads via `CrcProof`.

### Integration points

- `DirtyPageTracker` (existing) → `FlushWriter` iterates `dirty_pages()` per column
- `SlabPool` (existing) → page data is already mmap-backed and aligned
- `CodecRegistry` (existing) → column type metadata stored in sorted run header
  for schema validation on recovery

### Testing

- Unit: round-trip write → read for various component types and archetype sizes
- Unit: flush with no dirty pages produces no file
- Unit: CRC32 validation catches corruption (flip a byte, verify rejection)
- Property: for any set of mutations, flush + read-back yields identical page bytes
- Integration: `World` → mutate → flush → verify sorted run contents match world state
- Miri: reader/writer operate on `Vec<u8>` buffers (no mmap) under `cfg(miri)`

### Risks

- **Schema in sorted runs**: sorted runs must record which component type each
  page belongs to. If archetypes change between flush and recovery (component
  added/removed), the reader must handle schema mismatch. Defer migration to
  Phase 4; Phase 1 assumes stable schemas between flush and read.

---

## Phase 2: LsmManifest and Multi-Level Structure

> **Status: Complete.** Implemented in PRs #160, #163–#166. Manifest with type-safe operations, CRC32 frames, crash recovery, compaction commit atomicity.

**Goal**: Track sorted runs across levels (L1, L2, L3) with a persistent manifest.

### Deliverables

| Component | Description |
|---|---|
| `LsmManifest` | Tracks sorted runs per level: file paths, sequence ranges, archetype coverage |
| `ManifestLog` | Append-only log of manifest changes (add run, remove run, level promotion) |
| `LevelPolicy` | Configuration: max runs per level, size thresholds for promotion |

### Design

The manifest is a small, append-only log (not a sorted run). On startup, replay
the manifest log to reconstruct the in-memory manifest state. This is the same
pattern as RocksDB's MANIFEST file.

```rust
pub struct LsmManifest {
    /// Sorted runs at each level, ordered by sequence range.
    levels: [Vec<SortedRunMeta>; NUM_LEVELS],  // NUM_LEVELS = 4 (L0–L3)
    /// Next sequence number for flush.
    next_sequence: u64,
}

pub struct SortedRunMeta {
    path: PathBuf,
    level: u8,
    sequence_range: (u64, u64),
    archetype_coverage: FixedBitSet,  // which archetypes have pages in this run
    page_count: u64,
    size_bytes: u64,
}
```

### Manifest log format

```
[ManifestEntry::AddRun { level, meta }]
[ManifestEntry::RemoveRun { level, path }]
[ManifestEntry::PromoteRun { from_level, to_level, path }]
```

Each entry is length-prefixed with CRC32, same frame format as the WAL.

### Testing

- Unit: manifest add/remove/promote operations maintain invariants
- Unit: manifest log replay reconstructs identical manifest state
- Unit: corrupt manifest entry detected and skipped (CRC32 mismatch)
- Integration: multiple flushes → manifest tracks all runs with correct sequence ordering

### Risks

- **Manifest size**: with frequent flushes the manifest log grows. Periodic
  manifest checkpoint (write full state, truncate log) needed if this becomes
  a problem. Defer to Phase 3 (compaction naturally reduces run count).

---

## Phase 3: Compactor

> **Status: Complete.** Implemented in PRs #167–#170. Merge kernel with emit-list construction, entity-level dedup, observer hooks, compact_one public API.

**Goal**: Merge sorted runs within and across levels to bound read amplification
and disk usage.

### Deliverables

| Component | Description |
|---|---|
| `Compactor` | Merges sorted runs: L1→L2 (minor), L2→L3 (major) |
| `MergeIterator` | K-way merge over sorted run readers, deduplicating by `(archetype_id, page_index)` — latest sequence wins |
| `CompactionScheduler` | Decides when and what to compact based on `LevelPolicy` |

### Compaction strategy

**Archetype-parallel compaction.** Sorted runs are sorted by
`(archetype_id, page_index)`. Compaction for different archetypes is fully
independent — no cross-archetype merging is ever needed. The `Compactor` can
process archetypes in parallel via `rayon`.

**Minor compaction (L1 → L2)**:
- Triggered when L1 run count exceeds threshold (e.g., 4 runs)
- Merge all L1 runs for a given archetype into a single L2 run
- Delete merged L1 runs, update manifest

**Major compaction (L2 → L3)**:
- Triggered when L2 total size exceeds threshold
- Merge all L2 runs into a single L3 baseline run
- This is the "full compacted baseline" used for recovery

**Tombstone handling**:
- Despawned entities leave "gaps" in archetype pages (swap-remove fills them)
- The sorted run stores the page image as-is — the gap is just overwritten data
  from the swap. No explicit tombstones needed at the page level.
- Archetype-level tombstone: if an archetype is completely emptied, compaction
  drops it from the output run. The manifest's `archetype_coverage` bitset
  reflects this.

### Scheduling model

Two modes, user-selectable:

1. **Background thread** (default): compaction runs on a dedicated thread,
   rate-limited to avoid I/O spikes. Suitable for server workloads.
2. **Cooperative**: `compact_step()` performs a bounded amount of work per call.
   Game loops call this during idle time. Bounded by byte budget per step.

```rust
pub enum CompactionMode {
    Background { io_budget_bytes_per_sec: usize },
    Cooperative { bytes_per_step: usize },
}
```

### Testing

- Unit: `MergeIterator` correctly deduplicates pages (latest sequence wins)
- Unit: compaction of overlapping runs produces correct merged output
- Unit: empty archetype dropped from compacted run
- Stress: rapid mutation + flush + compact cycle doesn't lose data
- Property: for any sequence of mutations, `compact(flush(mutations))` produces
  a sorted run whose pages match the current world state
- Benchmark: compaction throughput (MB/s) for various archetype sizes

### Risks

- **Compaction during mutation**: if the world is being mutated while compaction
  reads L1 runs, there's no conflict — L1 runs are immutable files. New dirty
  pages go to a new L1 run. But the manifest must be updated atomically
  (add new compacted run + remove old runs in one manifest log entry).
- **Disk space amplification**: during compaction, old and new runs coexist
  briefly. Worst case: 2× disk usage during major compaction. Acceptable
  given the alternative (full-world snapshots).

---

## Phase 4: BlockedBloomFilter

> **Status: Complete.** Implemented in PR #174. Cache-line-blocked bloom filter, zero-copy BloomView, write_bloom_section dedup, checked_mul overflow protection.

**Goal**: Accelerate recovery-time page lookups across LSM levels.

### Deliverables

| Component | Description |
|---|---|
| `BlockedBloomFilter` | Cache-line-blocked bloom filter: one probe = one cache miss |
| Integration | One filter per `SortedRun`, serialized in the footer, rebuilt on compaction |

### Design (from roadmap)

```rust
#[repr(C, align(64))]
struct Block {
    words: [u64; 8],  // 512 bits per cache line
}

pub struct BlockedBloomFilter {
    blocks: &[Block],  // SlabPool-allocated, contiguous
    num_blocks: u32,
    seed: u64,
}
```

- **Hash**: Enhanced double-hashing with per-word entropy via `splitmix64`.
  `h1 = splitmix64(key ^ seed)` selects the block; `h2 = splitmix64(h1)` and
  `h3 = splitmix64(h2)` generate per-word bit positions as
  `bit_i = ((h2 >> (i*6)) ^ (h3 >> ((7-i)*6))) & 0x3F`. XOR-combining two
  independent 6-bit slices at mirrored offsets gives each word full 0–63 range
  and 12 bits of effective input entropy per position.
- **Key**: `(ArchetypeId, page_index)` packed as `u64`
- **Sizing**: 10 bits/key → ~0.84% FPR
- **SIMD**: `#[repr(align(64))]` enables AVX2 `vpand`+`vpcmpeq` auto-vectorization

### Integration

- `FlushWriter` builds a filter for each new sorted run
- `Compactor` rebuilds filters when merging runs (bulk insert)
- `SortedRunReader::contains_page()` checks the bloom filter before binary-searching
  the sparse index
- `LsmRecovery` (Phase 5) probes filters top-down (L1 → L2 → L3) to skip levels

### Testing

- Unit: insert N keys, verify zero false negatives
- Unit: FPR within expected bounds (measure over 100K random probes)
- Unit: serialization round-trip preserves filter state
- Unit: SIMD path produces same results as scalar path (cross-validate)
- Benchmark: probe throughput (ops/sec) for various filter sizes
- Miri: scalar-only path under `cfg(miri)`, no alignment-dependent UB

### Risks

- **Filter sizing at flush time**: the `FlushWriter` must know the expected key
  count upfront to size the filter. This is known — it's the count of dirty pages
  being flushed. No risk here.
- **False positives during recovery**: a false positive causes one unnecessary
  I/O (read a sorted run that doesn't contain the page). At 0.84% FPR this is
  negligible. No mitigation needed.

---

## Phase 5: LsmRecovery and Durable Integration

> **Status: Not started.** This is the remaining Stage 3 work.

**Goal**: Replace full-world snapshot recovery with incremental LSM recovery.
Wire the LSM into the existing `Durable<S>` strategy.

### Deliverables

| Component | Description |
|---|---|
| `LsmRecovery` | Restores `World` from L3 baseline + L2 delta + L1 delta + WAL tail |
| `Durable<S>` update | Flush dirty pages instead of (or in addition to) full snapshots |
| `AutoCheckpoint` update | Trigger LSM flush instead of snapshot when WAL threshold exceeded |
| Migration path | Support loading old v2 snapshots for upgrade from Stage 2 |

### Recovery sequence

```
1. Load LsmManifest from manifest log
2. Load L3 baseline (if exists) → reconstruct World archetypes and columns
3. Apply L2 sorted runs (merge pages on top of L3 baseline)
4. Apply L1 sorted runs (merge pages on top of L2 state)
5. Replay WAL tail from last flush sequence number
6. World is now fully recovered
```

At each level, `BlockedBloomFilter` skips runs that don't contain the target page.

### Durable strategy changes

```
Before (Stage 2):
  commit → WAL write → [threshold] → full snapshot

After (Stage 3):
  commit → WAL write → [threshold] → LSM flush (dirty pages only)
                                    → [L1 threshold] → minor compaction
                                    → [L2 threshold] → major compaction
```

The full snapshot path remains available as a fallback (`Durable::force_snapshot()`),
but `AutoCheckpoint` defaults to LSM flush.

### Migration from Stage 2

On first startup with LSM enabled:
1. If a v2 snapshot exists and no LSM manifest exists, load the snapshot
   (existing `Snapshot::load` path)
2. Immediately flush the entire world as an L3 baseline sorted run
3. Create the initial manifest
4. Subsequent checkpoints use LSM flush

This is a one-time migration. The v2 snapshot code remains in `minkowski-persist`
but is no longer the default checkpoint path.

### Testing

- Integration: mutate → flush → crash-simulate → recover → verify world state matches
- Integration: recovery from L3-only, L3+L1, L3+L2+L1, and L3+L2+L1+WAL tail
- Integration: migration from v2 snapshot → LSM baseline
- Stress: rapid mutation + crash at random points → recovery always succeeds
- Fuzz: random mutation sequences → flush at random intervals → recover → compare
  to in-memory reference world
- Loom: concurrent flush + mutation doesn't corrupt state (if background compaction
  is enabled)

### Risks

- **WAL sequence alignment**: the LSM flush must record which WAL sequence number
  it covers so recovery knows where to start WAL replay. The `FlushWriter` must
  capture the current WAL sequence atomically with the dirty page snapshot.
  Use the existing `Durable<S>` commit sequence number.
- **Partial flush on crash**: if the process crashes mid-flush, the sorted run
  file may be incomplete. The manifest log hasn't been updated yet (manifest
  update is the commit point). Recovery ignores orphaned sorted run files not
  referenced by the manifest.

---

## Phase Dependency Graph

```
Phase 1 ──────▶ Phase 2 ──────▶ Phase 3
SortedRun       Manifest        Compactor
FlushWriter                     MergeIterator
    │                               │
    │               ┌───────────────┘
    ▼               ▼
Phase 4 ──────▶ Phase 5
BloomFilter     LsmRecovery
                Durable integration
```

- **Phases 1→2→3** are strictly sequential (each builds on the prior)
- **Phase 4** depends only on Phase 1 (needs `SortedRun` format) and can be
  developed in parallel with Phases 2–3
- **Phase 5** depends on all prior phases

---

## Milestone Summary

| Phase | Deliverable | Key Metric | Est. LOC |
|---|---|---|---|
| 1 | Dirty page flush to disk | Round-trip correctness for all component types | ~1,200 |
| 2 | Multi-level manifest | Manifest replay correctness, crash safety | ~800 |
| 3 | Compaction pipeline | Space amplification ≤ 2×, throughput ≥ 500 MB/s | ~1,500 |
| 4 | Bloom filter | FPR ≤ 1%, probe ≤ 1 cache miss | ~600 |
| 5 | Recovery + integration | Recovery time O(delta), migration from v2 snapshots | ~1,500 |
| **Total** | | | **~5,600** |

---

## Open Decisions (resolve before Phase 1)

1. **Physical page size for I/O**: `DirtyPageTracker` uses 256-row logical pages.
   Should the sorted run write these as-is, or batch multiple logical pages into
   a larger I/O page (e.g., 64 KiB) for NVMe alignment? Recommendation: write
   logical pages as-is in Phase 1, add I/O batching in Phase 3 if benchmarks
   show it matters.

2. **Crate boundary**: should `BlockedBloomFilter` live in `minkowski-lsm` or in
   `minkowski` core (for reuse by indexes)? Recommendation: `minkowski-lsm`
   initially, extract later if needed.

3. **Compaction thread model**: the `CompactionScheduler` needs a way to run in
   background without blocking the main ECS tick. Options: dedicated thread
   (simple), rayon task (integrates with existing parallelism), or cooperative
   polling. Recommendation: start with dedicated thread, add cooperative mode
   in Phase 3 as an alternative.

4. **WAL segment cleanup**: after a successful L3 major compaction, WAL segments
   older than the L3 sequence range can be deleted. This extends the existing
   WAL compaction (which only deletes after snapshot). Implement in Phase 5
   alongside recovery.

---

## References

- [Scaling Roadmap — Stage 3](./2026-03-20-scaling-roadmap.md#stage-3-lsm-tree-storage)
- [RocksDB MANIFEST](https://github.com/facebook/rocksdb/wiki/MANIFEST)
- [Blocked Bloom Filters (Putze et al.)](https://algo2.iti.kit.edu/documents/cacheefficientbloomfilters-jea.pdf)
- [TigerBeetle Storage Engine](https://tigerbeetle.com/blog/a-]database-without-dynamic-memory-allocation)
