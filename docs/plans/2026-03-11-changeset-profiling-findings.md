# EnumChangeSet Profiling Findings

## Summary

The 14x QueryWriter-vs-QueryMut overhead splits roughly 46% recording / 35% apply /
15% allocator / 2.5% lifecycle. The dominant root cause is **function call overhead
from non-inlined hot-path functions** — `Arena::alloc` and `record_insert` are called
10K times per iteration but never inlined, spending 33% of their self-time on
prologue/epilogue instructions. The apply phase is dominated by per-entity entity
location lookups and column pointer math.

## Baseline Numbers

| Benchmark | Time | Per entity |
|---|---|---|
| `reducer/query_mut_chunk_10k` | 1.7 µs | 0.17 ns |
| `reducer/query_mut_10k` | 11 µs | 1.1 ns |
| `reducer/query_writer_10k` | 150 µs | 15 ns |
| `changeset/record_10k_inserts` | 241 µs | 24.1 ns |
| `changeset/apply_10k_overwrites` | 100 µs | 10.0 ns |
| `changeset/record_apply_10k` | 340 µs | 34.0 ns |
| `changeset/new_drop_empty` | 11 ns | — |

## Profiling Breakdown (perf, self-time)

### Recording phase: 46% of total

| Symbol | Self % | Notes |
|---|---|---|
| `Arena::alloc` | 12.5% | NOT inlined — 33% of self-time on push/pop |
| `memcpy` (inside Arena::alloc) | 16.8% | 12-byte copy per entity |
| `record_insert` | 11.8% | NOT inlined — Vec push + capacity check |
| `register_query_writer::{{closure}}` | 2.6% | Archetype iteration + WritableRef setup |

**Root cause:** `Arena::alloc` and `record_insert` have zero `#[inline]` annotations.
Each of the 10K entities per iteration incurs full function-call overhead (register
save/restore, stack frame setup). For functions that do ~20 instructions of actual
work, the call/return overhead dominates.

### Apply phase: 35% of total

| Symbol | Self % | Notes |
|---|---|---|
| `EnumChangeSet::apply` | 30.0% | Per-mutation entity lookup, memcpy, tick mark |
| `memcpy` (inside apply) | 3.6% | Copy from arena back into BlobVec column |

**Root cause:** Each of the 10K `Mutation::Insert` records triggers:
1. Entity location lookup (`entity_locations[entity.index()]`)
2. Component ID match to find the right column
3. `BlobVec::get_ptr_mut(row, tick)` — pointer arithmetic + tick mark
4. `copy_nonoverlapping` — 12-byte write

The entity lookup and column matching happen per-mutation. For the overwrite-only
workload that QueryWriter produces (same archetype, same component, just different
values), this is redundant — all entities are in the same archetype.

### Allocator overhead: 15% of total

Unresolved libc symbols (malloc/free/realloc) totaling ~15%. These come from:
- `Vec<Mutation>` growing during recording (no pre-allocation)
- Arena growing (doubling strategy, but starts at 0)
- EnumChangeSet drop (Vec + Arena deallocation)

### Lifecycle: 2.5% of total

| Symbol | Self % | Notes |
|---|---|---|
| `drop_in_place<EnumChangeSet>` | 2.2% | Vec + Arena dealloc |
| `drain_orphans` | 0.15% | Optimistic abort cleanup |
| `Transact::transact` | 0.13% | Begin/commit wrapper |

Negligible. Transaction overhead is not a factor.

## Optimization Candidates

### Candidate 1: `#[inline]` on Arena::alloc and record_insert

- **Current cost:** ~24% of total (12.5% alloc + 11.8% record_insert)
- **Theoretical minimum:** near zero (20 instructions inlined into loop body)
- **Approach:** Add `#[inline]` to `Arena::alloc`, `Arena::get`, `record_insert`,
  and `insert_raw`. LLVM can then eliminate call/return overhead and hoist the
  capacity check out of the inner loop.
- **Expected improvement:** 20-30% reduction in QueryWriter time
- **Risk:** Low. These are small, non-recursive functions.

### Candidate 2: Pre-allocate Vec<Mutation> and Arena capacity

- **Current cost:** ~15% (allocator overhead from growing)
- **Approach:** Add `EnumChangeSet::with_capacity(n)` that pre-sizes both
  `Vec<Mutation>` and `Arena` based on expected mutation count. QueryWriter
  knows the archetype entity count before iterating.
- **Expected improvement:** 10-15% reduction
- **Risk:** Low. Ergonomic addition, no semantic change.

### Candidate 3: Batch overwrite path in apply()

- **Current cost:** 30% self-time in apply
- **Approach:** When all mutations are `Insert` for the same component on entities
  in the same archetype (which is exactly what QueryWriter produces), skip
  per-mutation entity lookup and instead:
  1. Sort mutations by entity row within the archetype
  2. Memcpy in batch, mark column changed once
- **Expected improvement:** 15-25% reduction in apply time
- **Risk:** Medium. Requires detecting the "uniform overwrite" pattern and adding
  a fast path. Must not regress the general case.

### Candidate 4: Inline WritableRef::modify → set → insert_raw chain

- **Current cost:** Currently hidden inside the closure (2.6% visible)
- **Approach:** `#[inline]` on `WritableRef::modify`, `WritableRef::set`, and
  `EnumChangeSet::insert_raw`. The full chain from user closure to arena write
  should be a single inlined sequence.
- **Expected improvement:** Included in Candidate 1's estimate
- **Risk:** Low.

## Recommendation

**Phase 3 implementation order:**

1. **Candidate 1 + 4 together** (inlining) — lowest risk, highest expected impact,
   applies to all changeset paths. Measure before/after with existing benchmarks.

2. **Candidate 2** (pre-allocation) — straightforward, independent of inlining.
   Can be done in parallel.

3. **Candidate 3** (batch overwrite) — highest complexity, consider only after
   1+2 are measured. The inlining improvements may reduce apply overhead enough
   that batch optimization isn't worth the complexity.

## Appendix: Mutation Enum Size

`Mutation` is 48 bytes. 10K mutations = 480KB, which fits in L2 but not L1.
The `Spawn` variant contains a `Vec<(ComponentId, usize, Layout)>` (24 bytes for
the Vec header alone), which inflates all variants to 48 bytes due to enum sizing.
If Spawn were separated or boxed, Insert/Remove variants could shrink to ~32 bytes
(Entity + ComponentId + offset + Layout = 8+8+8+16 = 40, rounded with discriminant).
This is a potential future optimization but unlikely to be the bottleneck after
inlining fixes.
