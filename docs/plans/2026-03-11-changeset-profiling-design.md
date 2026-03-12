# EnumChangeSet Performance Investigation

## Problem

QueryWriter is 14x slower than QueryMut for the same Position += Velocity workload
(150 µs vs 11 µs over 10K entities). The theoretical minimum overhead for buffered
writes is 3-5x (clone + arena copy + apply copy back). The gap suggests hidden
inefficiencies.

## Baseline Numbers (2026-03-11)

| Benchmark | Time (10K entities) | Per entity |
|---|---|---|
| `query_mut_chunk_10k` | 1.7 µs | 0.17 ns |
| `query_mut_10k` | 11 µs | 1.1 ns |
| `query_writer_10k` | 150 µs | 15 ns |
| `dynamic_for_each_10k` | 293 µs | 29 ns |
| `simple_insert/batch` | 2.9 ms | 290 ns |
| `simple_insert/changeset` | 4.4 ms | 440 ns |

## Investigation Order

1. **QueryWriter** (14x gap, tightest hot loop, biggest relative overhead)
2. **DynamicCtx** (2x over QueryWriter — collect+write pattern + HashMap lookups)
3. **Bulk spawn** (50% over direct — arena + apply vs BlobVec push)

Findings from earlier stages cascade to later ones since they share Arena and apply().

## Approach

### Phase 1: Flamegraph Capture

Standalone profiling binary (`examples/examples/profile_changeset.rs`), not criterion.
Criterion's measurement harness pollutes flamegraphs with timing/statistics overhead.

The binary runs:
- **QueryMut integration** (baseline): 10K entities, 1000 iterations
- **QueryWriter integration** (subject): 10K entities, 1000 iterations

Both in the same binary, separated by a clear marker (e.g. `std::hint::black_box`
boundary) so they appear as distinct subtrees in the flamegraph.

Profile with `samply record` which captures to Firefox Profiler format, giving
interactive call tree navigation. `perf` available as fallback.

```bash
cargo build -p minkowski-examples --example profile_changeset --release
samply record target/release/examples/profile_changeset
```

### Phase 2: Targeted Microbenchmarks

Based on flamegraph findings, add criterion benchmarks isolating specific costs.
Candidates (to be confirmed by profiling):

- Arena::alloc in isolation (N allocs of various sizes)
- WritableRef::modify clone+closure+insert_raw cycle
- apply() loop: entity location lookup + memcpy + tick marking
- Vec<Mutation> push overhead and memory layout
- DropEntry bookkeeping cost (needs_drop check, Vec push)

New file: `crates/minkowski-bench/benches/changeset_micro.rs`

### Phase 3: Optimize

Fix what the profile reveals. Possible categories:

- **Inlining failures**: Arena::alloc, insert_raw, or modify not inlined into the
  hot loop. Fix with `#[inline]` or restructuring.
- **Unnecessary work per entity**: DropEntry tracking for Copy types, bounds checks,
  alignment calculations that could be hoisted.
- **Memory layout**: Vec<Mutation> enum size causing cache pressure, arena growth
  pattern suboptimal for known-size workloads.
- **Apply loop**: per-mutation entity location lookup could be batched or sorted.

Each optimization gets a before/after benchmark comparison.

### Future: PGO Pass

After algorithmic improvements, a profile-guided optimization pass
(`-Cprofile-generate` / `-Cprofile-use`) to see if LLVM can further improve
inlining and code layout decisions with runtime profile data.

## Success Criteria

- Flamegraphs captured and bottlenecks identified
- QueryWriter overhead reduced toward the 3-5x theoretical floor
- No correctness regressions (all existing tests pass)
- Findings documented for future optimization work

## Non-Goals

- Changing QueryWriter's semantic model (buffered writes are load-bearing for
  transactional conflict detection and WAL durability)
- Optimizing the WAL serialization path (separate investigation)
- Changing the benchmarked workload (Position += Velocity is a standard ECS
  microbenchmark, keeps results comparable)
