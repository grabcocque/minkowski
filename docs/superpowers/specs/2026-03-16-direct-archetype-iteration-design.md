# Direct Archetype Iteration for Scan-Only Batched Methods

**Version**: v1.3.3
**Date**: 2026-03-16
**Status**: Approved

## Problem

`for_each_batched` on scan-only plans (including eliminated inner joins):
1. Collects entities into ScratchBuffer via `compiled_for_each_raw` (~22 us)
2. Sorts by (archetype_id, row) (~2 us for 10K)
3. Walks archetype runs with `init_fetch`/`fetch` (~5 us)

Steps 1-2 are wasted work for scan-only plans. The scan already walks
archetypes sequentially -- entities are inherently archetype-grouped.
Collecting them into a Vec just to sort and rediscover the grouping is
a 5x overhead (27 us vs 5 us theoretical).

## Solution: Direct Archetype Iteration

For scan-only plans without custom predicates, bypass the ScratchBuffer
entirely. Walk archetypes directly using the stored `required`/`changed`
bitsets and call `init_fetch`/`fetch` inline.

### New Fields on `QueryPlanResult`

```rust
/// Component requirements for the direct archetype iteration fast path.
/// `Some` when the plan is scan-only with no custom predicates.
/// `None` when the plan has joins or custom filter closures.
scan_required: Option<FixedBitSet>,
scan_changed: FixedBitSet,
```

Set during `build()`: if `self.joins.is_empty()` and `filter_preds` has
no custom predicates (only index/spatial predicates that were absorbed),
store the `left_required`/`left_changed` bitsets. Otherwise `None`.

Actually simpler: `filter_preds` at this point in build() contains the
*remaining* predicates that weren't absorbed by indexes. If
`filter_preds.is_empty()` after Phase 1 classification, there are no
custom filters. Combined with `self.joins.is_empty()` (after Phase 4b
elimination), we have a pure scan.

### Fast Path in `for_each_batched_inner`

```rust
// Fast path: scan-only plan with no custom predicates.
// Walk archetypes directly — no ScratchBuffer, no sort.
if let Some(ref required) = self.scan_required {
    let changed = &self.scan_changed;
    let tick = self.last_read_tick;
    for arch in &world.archetypes.archetypes {
        if arch.is_empty() || !required.is_subset(&arch.component_ids) {
            continue;
        }
        if !passes_change_filter(arch, changed, tick) {
            continue;
        }
        // Validate Q's required components.
        if !Q::required_ids(&world.components).is_subset(&arch.component_ids) {
            return Err(PlanExecError::ComponentMismatch {
                query: std::any::type_name::<Q>(),
                archetype_id: arch.id,
            });
        }
        let fetch = Q::init_fetch(arch, &world.components);
        for (row, &entity) in arch.entities.iter().enumerate() {
            let item = unsafe { Q::fetch(&fetch, row) };
            callback(entity, item);
        }
    }
    return Ok(());
}
// ... existing ScratchBuffer path for join plans / filtered plans ...
```

### Fast Path in `for_each_join_chunk`

Same check. For scan-only plans, yield full archetype slices:

```rust
if let Some(ref required) = self.scan_required {
    // ... archetype walk ...
    let fetch = Q::init_fetch(arch, &world.components);
    let slice = unsafe { Q::as_slice(&fetch, arch.len()) };
    self.row_indices.clear();
    self.row_indices.extend(0..arch.len());
    callback(&arch.entities, &self.row_indices, slice);
}
```

### What triggers the fast path

| Plan type | `scan_required` | Path |
|---|---|---|
| Pure scan, no predicates | `Some(bitset)` | Direct archetype |
| Scan + index predicates (absorbed) | `Some(bitset)` | Direct archetype |
| Scan + custom predicates | `None` | ScratchBuffer |
| Eliminated inner join, no predicates | `Some(bitset)` | Direct archetype |
| Left join (not eliminated) | `None` | ScratchBuffer + sort |

### Mutable columns

`for_each_batched` (not `_inner`) already marks mutable columns with
`mark_changed` before calling inner. The direct path doesn't change this.

## Performance Predictions

| Benchmark | Before | After |
|---|---|---|
| `join/for_each_batched_10k` | 27 us | ~5-6 us |
| `join/for_each_chunk_10k` | 35 us | ~5-6 us |
| `join/manual_query_10k` | 4.8 us | 4.8 us |

## Test Plan

4 tests:

1. `direct_iter_batched_scan_only` -- verify for_each_batched on pure scan
   uses direct path (same results, faster)
2. `direct_iter_batched_with_custom_predicate_uses_scratch` -- verify
   custom predicates fall back to ScratchBuffer path
3. `direct_iter_chunk_scan_only` -- verify for_each_join_chunk yields
   full archetype slices with sequential rows
4. `direct_iter_batched_eliminated_join` -- verify eliminated inner join
   takes the direct path

## Files Modified

| File | Change |
|---|---|
| `crates/minkowski/src/planner.rs` | `scan_required`/`scan_changed` fields, fast paths in `for_each_batched_inner` and `for_each_join_chunk`, 4 tests |
