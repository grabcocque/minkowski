# Allocation-Free Query Execution Engine

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current closure-dispatch execution engine with a compiled
query pipeline that matches `world.query().for_each()` speed on scan paths
and uses plan-owned scratch buffers for joins.

**Architecture:** Two execution paths determined at build time. Scan-only plans
compile to monomorphic `QueryIter` wrappers — same machine code as hand-written
iteration. Join/gather plans use a single flat scratch buffer with sorted
intersection — no `HashSet`, no per-node `Vec<Entity>`, no global allocator
hits during execution.

**Scope:** Eq, Range, inner/left joins. No `Changed<T>` integration (subscription
model unchanged). No SpatialIndex predicates. Custom predicates as `dyn Fn`
fallback.

---

## Context

PR #94 added a Volcano query planner with three layers: logical plan tree,
vectorized plan tree, and an executable plan tree. The execution engine uses
`Arc<dyn Fn>` closures and allocates `Vec<Entity>` per tree node on every
`execute()` call — acceptable for debugging but unsuitable for hot-path use.

The goal is to replace the execution layer while keeping the planning layer
(cost model, EXPLAIN, index selection, join ordering, subscriptions)
unchanged.

## Constraints

- **Hot-path speed**: scan execution must match `world.query().for_each_chunk()`
  — identical machine code after inlining.
- **Zero allocation on scan path**: no `Vec`, no `Box`, no `Arc` during
  iteration. Pre-computed state only.
- **Pool-compatible allocation on join path**: scratch buffers allocated from
  `SharedPool` when available, reused across calls.
- **Compile-once, execute-many**: `build()` does all expensive work. `execute()`
  / `for_each()` is the fast path.
- **`&mut self` execute**: plan-owned scratch enables reuse, prevents concurrent
  execution (which would be unsound with shared scratch).

## Execution Model

### Two Paths

The plan shape is known at `build()` time. The builder selects one of:

```
ScanBuilder::build()
  ├─ no joins, no index gathers → CompiledScan
  └─ has joins or index gathers → CompiledEntityPlan
```

**CompiledScan** — the fast path. A type-erased closure that constructs a
`QueryIter<Q>` and iterates via `for_each_chunk`. The generic `Q: WorldQuery`
is captured at `build()` time while still in scope, then stored as a trait
object. Eq/Range predicates are fused into the iteration as typed slice filters.
No entity buffers, no intermediate collections.

**CompiledEntityPlan** — the join/gather path. Index lookups produce entity
handles. Joins intersect entity sets via sorted intersection. All intermediate
results live in a single plan-owned scratch buffer. The caller receives
`&[Entity]` borrowed from the scratch.

### Why Two Paths

A scan-only plan iterates archetype columns contiguously — yielding `&[T]`
slices that LLVM can auto-vectorize. This is the 9x performance advantage of
`for_each_chunk` over `for_each` (documented in perf-shakedown baselines).
Routing scans through an entity buffer would sacrifice this entirely.

Joins inherently produce cross-archetype entity sets — contiguous slice access
is impossible. The entity buffer is unavoidable but localized to the uncommon
case.

## Plan Compilation

At `build()` time, three things are captured while the generic `Q` is in scope:

### 1. Scan Closure

A `Box<dyn FnMut>` that constructs and iterates a `QueryIter<Q>`. The
monomorphic iteration code is baked in at compile time. Stored type-erased
on `QueryPlanResult`.

### 2. Filter Closures

For each Eq/Range predicate, a typed filter compiled from the predicate's
value. These operate on `&[T]` slices from `for_each_chunk`:

- **Eq**: `|scores: &[Score], entities: &[Entity], out: &mut F| { ... }`
  with a tight `==` comparison loop — SIMD-friendly.
- **Range**: same pattern with `>= lo && < hi`.
- **Custom**: `dyn Fn(&World, Entity) -> bool` per-entity fallback. Explicitly
  the slow path.

Filters are composed in selectivity order (most selective first) and fused into
the scan closure. The compiled scan doesn't produce entities and then filter —
it filters during iteration and calls the user's callback only for matching
entities. No intermediate entity buffer at all.

### 3. Entity-Mode Fallback

If the plan has joins or index gathers, a separate entity-collecting scan
closure is compiled. This one writes to the scratch buffer. Joins and gathers
operate on the scratch buffer only.

## Scratch Buffer

For the entity-mode path, the plan owns a reusable buffer:

```rust
struct ScratchBuffer {
    entities: Vec<Entity>,  // allocated from SharedPool at build() time
}
```

**Sizing**: pre-allocated to `estimated_rows` from the cost model, capped at
64K entities. If execution exceeds the estimate, the buffer grows once from
the pool — amortized across subsequent executions.

**Reuse**: `execute()` clears the buffer (len = 0, no dealloc) and refills.
Capacity persists. After a few executions the buffer stabilizes — no further
allocations.

**Multi-node plans**: a join plan uses a single flat buffer with index ranges:

```
[ left entities | right entities | join output ]
     0..left_len   left_len..right_end   right_end..total
```

Three slices into one allocation. `clear()` resets all ranges.

## Join Execution

Joins only exist in the entity-mode path.

**Inner join (hash join path)**: sorted intersection, no HashSet.

1. Left child writes entities to `scratch[0..left_len]`
2. Right child writes entities to `scratch[left_len..right_end]`
3. Sort left slice via `sort_unstable` (in-place, no allocation)
4. Iterate right, binary search into left
5. Matching entities written to `scratch[right_end..]`

O(n log n) sort + O(m log n) probe. Cache-friendly, predictable.

**Inner join (nested-loop path)**: both sides small (<64 rows). Quadratic
scan, still into scratch buffer.

**Left join**: all left-side entities preserved unconditionally.

**Input reordering**: for inner joins, smaller side becomes the sorted (build)
side. Left joins preserve semantic left side.

## API Surface

### Building Plans (unchanged)

```rust
let mut planner = QueryPlanner::new(&world);
planner.add_btree_index(&score_idx, &world);
let mut plan = planner
    .scan::<(&Score, &Pos)>()
    .filter(Predicate::eq(Score(42)))
    .build();
```

### Executing — Scan Path (new)

```rust
// Per-entity callback, inlined by LLVM
plan.for_each(&mut world, |entity, score: &Score, pos: &Pos| {
    // hot path — same speed as world.query().for_each()
});

// Per-chunk callback for manual SIMD
plan.for_each_chunk(&mut world, |entities, scores: &[Score], positions: &[Pos]| {
    // called per archetype chunk
});
```

### Executing — Entity Path (changed)

```rust
// Returns slice borrowed from plan's scratch buffer
let entities: &[Entity] = plan.execute(&mut world);
for &e in entities {
    let score = world.get::<Score>(e).unwrap();
}
```

### EXPLAIN / Cost / Warnings (unchanged)

```rust
println!("{}", plan.explain());
println!("{}", plan.logical_cost());
for w in plan.warnings() { ... }
```

### Signature Changes

| Method | Before | After |
|--------|--------|-------|
| `execute` | `&self, &World` → `Vec<Entity>` | `&mut self, &mut World` → `&[Entity]` |
| `for_each` | — (new) | `&mut self, &mut World, callback` |
| `for_each_chunk` | — (new) | `&mut self, &mut World, callback` |

## What Changes vs What Stays

### Removed

- `ExecNode` enum and all variants
- `ClosureNode` enum
- `lower_to_executable()` function
- `ScanFn`, `FilterFn`, `IndexLookupFn` type aliases
- `ProbeSet` helper
- `scan_matching_entities()` helper
- All `#[cfg(test)]` ExecNode introspection methods

### Added

- `CompiledScan` — type-erased scan closure with fused filters
- `CompiledEntityPlan` — scratch buffer + entity-mode execution
- `ScratchBuffer` — pool-aware reusable entity buffer
- `for_each()` / `for_each_chunk()` on `QueryPlanResult`

### Unchanged

- `QueryPlanner`, `ScanBuilder`, `Predicate`, `JoinKind`
- `PlanNode`, `VecExecNode`, cost model, `lower_to_vectorized()`
- `Indexed<T>`, `SubscriptionBuilder`, `SubscriptionPlan`
- `TablePlanner`, `HasBTreeIndex`, `HasHashIndex`
- `CardinalityConstraint`, `PlanWarning`, `SubscriptionError`
- `IndexDescriptor` (execution closures stay for index gather path)
- EXPLAIN output format
- All planning tests (index selection, join ordering, warnings, cost)

## Testing Strategy

- Existing 54 planning tests stay — they test PlanNode/VecExecNode/cost, not execution.
- Replace 31 execution tests (`execute_*`, `exec_tree_*`) with equivalent tests against new API.
- New tests for:
  - `for_each` / `for_each_chunk` yield correct entities and components
  - Scratch buffer reuse (execute twice, verify no reallocation)
  - Typed Eq/Range filters on scan path
  - Join via sorted intersection matches old HashSet results
  - Left join preserves all left entities
  - Pool-backed scratch buffer respects budget
  - Custom predicate fallback still works
  - Empty world / empty results
  - Despawned entity filtering
