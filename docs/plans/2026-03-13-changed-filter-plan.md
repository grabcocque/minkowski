# Changed<T> Filtering for Query Executor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Changed<T>` archetype-level filtering to the query executor so planned queries can skip unchanged archetypes, consistent with the existing `world.query()` path.

**Architecture:** New `WorldQuery::changed_ids()` trait method extracts a `FixedBitSet` of changed-filtered component IDs at `scan::<Q>()` time. The bitset is captured in scan/collector closures alongside `required`. A per-plan `Tick` field tracks the last-read tick. `for_each`/`execute` advance the tick after completion; `for_each_raw` reads but doesn't advance.

**Tech Stack:** Rust, fixedbitset, existing Tick/BlobVec infrastructure.

**Spec:** `docs/plans/2026-03-12-changed-filter-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/minkowski/src/query/fetch.rs` | Modify | Add `changed_ids` to `WorldQuery` trait + all impls + tuple macro |
| `crates/minkowski/src/planner.rs` | Modify | Closure signatures, scan capture, tick state, join collectors |

---

### Task 1: Add `changed_ids` to `WorldQuery` trait and all implementations

**Files:**
- Modify: `crates/minkowski/src/query/fetch.rs:31-86` (trait definition)
- Modify: `crates/minkowski/src/query/fetch.rs:112-300` (concrete impls)
- Modify: `crates/minkowski/src/query/fetch.rs:303-368` (tuple macro)

- [ ] **Step 1: Write failing tests for `changed_ids`**

Add tests in the existing `#[cfg(test)] mod tests` block at the bottom of `fetch.rs`:

Note: The fetch.rs test module already defines `Pos`. Add `Vel` alongside it:

```rust
#[derive(Debug, PartialEq, Clone, Copy)]
struct Vel {
    dx: f32,
    dy: f32,
}
```

Then add the tests:

```rust
#[test]
fn changed_ids_for_ref_is_empty() {
    let mut reg = ComponentRegistry::new();
    reg.register::<Pos>();
    let bits = <&Pos as WorldQuery>::changed_ids(&reg);
    assert!(bits.is_clear());
}

#[test]
fn changed_ids_for_mut_ref_is_empty() {
    let mut reg = ComponentRegistry::new();
    reg.register::<Pos>();
    let bits = <&mut Pos as WorldQuery>::changed_ids(&reg);
    assert!(bits.is_clear());
}

#[test]
fn changed_ids_for_entity_is_empty() {
    let reg = ComponentRegistry::new();
    let bits = <Entity as WorldQuery>::changed_ids(&reg);
    assert!(bits.is_clear());
}

#[test]
fn changed_ids_for_option_is_empty() {
    let mut reg = ComponentRegistry::new();
    reg.register::<Pos>();
    let bits = <Option<&Pos> as WorldQuery>::changed_ids(&reg);
    assert!(bits.is_clear());
}

#[test]
fn changed_ids_for_changed_contains_component() {
    let mut reg = ComponentRegistry::new();
    reg.register::<Pos>();
    let bits = <Changed<Pos> as WorldQuery>::changed_ids(&reg);
    let comp_id = reg.id::<Pos>().unwrap();
    assert!(bits.contains(comp_id));
    assert_eq!(bits.count_ones(..), 1);
}

#[test]
fn changed_ids_tuple_unions_terms() {
    let mut reg = ComponentRegistry::new();
    reg.register::<Pos>();
    reg.register::<Vel>();
    let bits = <(Changed<Pos>, &Vel)>::changed_ids(&reg);
    let pos_id = reg.id::<Pos>().unwrap();
    // Changed<Pos> contributes Pos; &Vel contributes nothing
    assert!(bits.contains(pos_id));
    assert_eq!(bits.count_ones(..), 1);
}

#[test]
fn changed_ids_tuple_multiple_changed() {
    let mut reg = ComponentRegistry::new();
    reg.register::<Pos>();
    reg.register::<Vel>();
    let bits = <(Changed<Pos>, Changed<Vel>)>::changed_ids(&reg);
    let pos_id = reg.id::<Pos>().unwrap();
    let vel_id = reg.id::<Vel>().unwrap();
    assert!(bits.contains(pos_id));
    assert!(bits.contains(vel_id));
    assert_eq!(bits.count_ones(..), 2);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- changed_ids`
Expected: FAIL — `changed_ids` method doesn't exist yet.

- [ ] **Step 3: Add `changed_ids` default method to `WorldQuery` trait**

In the trait definition (around line 73, after `mutable_ids`), add:

```rust
/// Returns ComponentIds that have `Changed<T>` filtering.
/// Used by the query planner to capture archetype-level change
/// filters at scan compilation time.
fn changed_ids(_registry: &ComponentRegistry) -> FixedBitSet {
    FixedBitSet::new()
}
```

The default returns empty — correct for `&T`, `&mut T`, `Entity`, `Option<&T>`.

- [ ] **Step 4: Add `changed_ids` impl for `Changed<T>`**

In the `Changed<T>` WorldQuery impl (around line 266), add:

```rust
fn changed_ids(registry: &ComponentRegistry) -> FixedBitSet {
    <&T>::required_ids(registry)
}
```

This reuses `<&T>::required_ids` which returns a bitset with T's component ID — exactly what we need. Same pattern as `Changed<T>::required_ids` already does on line 275-277.

- [ ] **Step 5: Add `changed_ids` to the tuple macro**

In `impl_world_query_tuple!` (around line 303), add after the `mutable_ids` method:

```rust
fn changed_ids(registry: &ComponentRegistry) -> FixedBitSet {
    let mut bits = FixedBitSet::new();
    $(
        let sub = $name::changed_ids(registry);
        bits.grow(sub.len());
        bits.union_with(&sub);
    )*
    bits
}
```

Same union pattern as `required_ids`, `accessed_ids`, and `mutable_ids`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib -- changed_ids`
Expected: All 7 tests PASS.

- [ ] **Step 7: Run full test suite to verify no regressions**

Run: `cargo test -p minkowski --lib`
Expected: All existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski/src/query/fetch.rs
git commit -m "feat: add WorldQuery::changed_ids() for planner change detection"
```

---

### Task 2: Thread `Tick` through closure types and add plan-instance tick state

This task updates the internal planner types to carry a `Tick` parameter and adds the
`last_read_tick` field to `QueryPlanResult`. No behavioral change yet — closures receive
the tick but don't use it.

**Files:**
- Modify: `crates/minkowski/src/planner.rs:612-621` (`QueryPlanResult` struct)
- Modify: `crates/minkowski/src/planner.rs:692-769` (`execute`, `for_each`, `for_each_raw`)
- Modify: `crates/minkowski/src/planner.rs:798-814` (`Debug` impl)
- Modify: `crates/minkowski/src/planner.rs:1100-1155` (type aliases, `JoinExec`, `collect_matching_entities`)
- Modify: `crates/minkowski/src/planner.rs:1370-1485` (build Phase 7-9)
- Modify: `crates/minkowski/src/planner.rs:1643-1726` (`scan`, `scan_with_estimate`)

- [ ] **Step 1: Update type aliases**

Change the three type aliases (around line 1106-1127):

```rust
// Before
type CompiledForEach = Box<dyn FnMut(&World, &mut dyn FnMut(Entity))>;
type CompiledForEachRaw = Box<dyn FnMut(&World, &mut dyn FnMut(Entity))>;
type EntityCollector = Box<dyn FnMut(&World, &mut ScratchBuffer)>;

// After
type CompiledForEach = Box<dyn FnMut(&World, Tick, &mut dyn FnMut(Entity))>;
type CompiledForEachRaw = Box<dyn FnMut(&World, Tick, &mut dyn FnMut(Entity))>;
type EntityCollector = Box<dyn FnMut(&World, Tick, &mut ScratchBuffer)>;
```

- [ ] **Step 2: Update `collect_matching_entities` to accept `Tick`**

Change the helper function signature (line 1147):

```rust
fn collect_matching_entities(
    world: &World,
    required: &FixedBitSet,
    _tick: Tick,
    scratch: &mut ScratchBuffer,
) {
```

The `_tick` parameter is unused for now — it will be used in Task 3.

- [ ] **Step 3: Add `last_read_tick` field to `QueryPlanResult`**

Add to the struct (around line 612):

```rust
pub struct QueryPlanResult {
    root: PlanNode,
    vec_root: VecExecNode,
    join_exec: Option<JoinExec>,
    compiled_for_each: Option<CompiledForEach>,
    compiled_for_each_raw: Option<CompiledForEachRaw>,
    scratch: Option<ScratchBuffer>,
    opts: VectorizeOpts,
    warnings: Vec<PlanWarning>,
    last_read_tick: Tick,
}
```

Add to the `Debug` impl (around line 798):

```rust
.field("last_read_tick", &self.last_read_tick)
```

Add to the construction in `build()` (around line 1475):

```rust
last_read_tick: Tick::default(),
```

Import `Tick` at the top of the file. `Tick` is `pub(crate)` and lives in `crate::tick::Tick`.

- [ ] **Step 4: Update `for_each` to pass tick and advance**

```rust
pub fn for_each(&mut self, world: &mut World, mut callback: impl FnMut(Entity)) {
    let compiled = self.compiled_for_each.as_mut().expect(
        "for_each() is only available for scan-only plans (no joins). \
             For plans with joins, use execute() which returns &[Entity].",
    );
    let tick = self.last_read_tick;
    compiled(&*world, tick, &mut callback);
    self.last_read_tick = world.next_tick();
}
```

- [ ] **Step 5: Update `for_each_raw` to pass tick (no advancement)**

```rust
pub fn for_each_raw(&mut self, world: &World, mut callback: impl FnMut(Entity)) {
    let compiled = self.compiled_for_each_raw.as_mut().expect(
        "for_each_raw() is only available for scan-only plans (no joins). \
             For plans with joins, use execute() which returns &[Entity].",
    );
    let tick = self.last_read_tick;
    compiled(world, tick, &mut callback);
}
```

- [ ] **Step 6: Update `execute` to pass tick and advance**

```rust
pub fn execute(&mut self, world: &mut World) -> &[Entity] {
    let scratch = self
        .scratch
        .as_mut()
        .expect("execute() requires a plan with a scratch buffer");
    scratch.clear();
    let tick = self.last_read_tick;

    if let Some(join) = &mut self.join_exec {
        (join.left_collector)(&*world, tick, scratch);
        for step in &mut join.steps {
            let left_len = scratch.len();
            (step.right_collector)(&*world, tick, scratch);
            match step.join_kind {
                JoinKind::Inner => {
                    let match_count = scratch.sorted_intersection(left_len).len();
                    if match_count > 0 {
                        let total = scratch.entities.len();
                        scratch.entities.copy_within(total - match_count.., 0);
                    }
                    scratch.entities.truncate(match_count);
                }
                JoinKind::Left => {
                    scratch.entities.truncate(left_len);
                }
            }
        }
        self.last_read_tick = world.next_tick();
        scratch.as_slice()
    } else if let Some(compiled) = &mut self.compiled_for_each {
        compiled(&*world, tick, &mut |entity: Entity| {
            scratch.push(entity);
        });
        self.last_read_tick = world.next_tick();
        scratch.as_slice()
    } else {
        panic!(
            "execute() called on a plan with no join executor and no compiled scan — \
             this indicates a bug in plan compilation"
        );
    }
}
```

- [ ] **Step 7: Update scan closure factories to accept `Tick`**

In `scan::<Q>()` (around line 1654), update the closure signatures. The `_tick`
parameter is captured but unused for now:

```rust
compile_for_each: Some(Box::new(move || {
    let required = required_for_each;
    Box::new(move |world: &World, _tick: Tick, callback: &mut dyn FnMut(Entity)| {
        for arch in &world.archetypes.archetypes {
            if !arch.is_empty() && required.is_subset(&arch.component_ids) {
                for &entity in &arch.entities {
                    callback(entity);
                }
            }
        }
    })
})),
compile_for_each_raw: Some(Box::new(move || {
    let required = required_for_each_raw;
    Box::new(move |world: &World, _tick: Tick, callback: &mut dyn FnMut(Entity)| {
        for arch in &world.archetypes.archetypes {
            if !arch.is_empty() && required.is_subset(&arch.component_ids) {
                for &entity in &arch.entities {
                    callback(entity);
                }
            }
        }
    })
})),
```

Do the same for `scan_with_estimate::<Q>()` (around line 1700).

- [ ] **Step 8: Update build Phase 7 (join collectors) to pass tick**

In the left collector closure (around line 1381):

```rust
let left_collector: EntityCollector =
    Box::new(move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
        if left_filters.is_empty() {
            collect_matching_entities(world, &left_required, tick, scratch);
        } else {
            for arch in &world.archetypes.archetypes {
                if !arch.is_empty() && left_required.is_subset(&arch.component_ids) {
                    for &entity in &arch.entities {
                        if left_filters.iter().all(|f| f(world, entity)) {
                            scratch.push(entity);
                        }
                    }
                }
            }
        }
    });
```

In the right collector closure (around line 1403):

```rust
right_collector: Box::new(
    move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
        collect_matching_entities(world, &right_required, tick, scratch);
    },
),
```

- [ ] **Step 9: Update build Phase 8 (filter fusion) to pass tick**

The compiled scan closures with filters (around line 1434):

```rust
Box::new(move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
    scan_fn(world, tick, &mut |entity: Entity| {
        if all_filter_fns.iter().all(|f| f(world, entity)) {
            callback(entity);
        }
    });
})
```

Same for `compiled_for_each_raw` variant (around line 1454).

- [ ] **Step 10: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests PASS. No behavioral change — tick is threaded through but `_tick`
parameters are unused in the archetype loops.

- [ ] **Step 11: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "refactor: thread Tick through planner closure types"
```

---

### Task 3: Wire `changed_ids` into scan closures and join collectors

This task adds the actual archetype filtering. After this, `Changed<T>` works end-to-end
in the query executor.

**Files:**
- Modify: `crates/minkowski/src/planner.rs:1147-1155` (`collect_matching_entities`)
- Modify: `crates/minkowski/src/planner.rs:1160-1182` (`ScanBuilder`, `JoinSpec`)
- Modify: `crates/minkowski/src/planner.rs:1196-1210` (`join::<Q>()`)
- Modify: `crates/minkowski/src/planner.rs:1370-1419` (build Phase 7)
- Modify: `crates/minkowski/src/planner.rs:1643-1726` (`scan`, `scan_with_estimate`)

- [ ] **Step 1: Write failing tests for Changed<T> in the planner**

First, add the `Changed` import to the planner test module:

```rust
use crate::Changed;
```

Then add to the planner test module (after the existing `for_each_raw_with_filter` test):

```rust
#[test]
fn for_each_changed_skips_stale_archetypes() {
    let mut world = World::new();
    // Spawn entities with Score
    for i in 0..5u32 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // First call: everything is "changed" (last_read_tick starts at 0)
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 5);

    // Second call: nothing changed since last for_each
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 0);
}

#[test]
fn for_each_changed_detects_mutation() {
    let mut world = World::new();
    // Put entities in DIFFERENT archetypes so column-level change
    // detection can distinguish them.
    let e = world.spawn((Score(1),));            // archetype: (Score,)
    world.spawn((Score(2), Team(0)));             // archetype: (Score, Team)

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // Consume initial changes
    plan.for_each(&mut world, |_| {});

    // Mutate only the (Score,) archetype's column
    let _ = world.get_mut::<Score>(e);

    // Should see only entities in the changed archetype (1 entity)
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 1);
}

#[test]
fn for_each_raw_changed_reads_tick_but_does_not_advance() {
    let mut world = World::new();
    for i in 0..3u32 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // for_each_raw reads tick (0) — everything is "changed"
    let mut count = 0;
    plan.for_each_raw(&world, |_| count += 1);
    assert_eq!(count, 3);

    // for_each_raw again — tick NOT advanced, so still sees everything
    let mut count = 0;
    plan.for_each_raw(&world, |_| count += 1);
    assert_eq!(count, 3);
}

#[test]
fn execute_changed_skips_stale_archetypes() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // First call: all changed
    assert_eq!(plan.execute(&mut world).len(), 5);

    // Second call: nothing changed
    assert_eq!(plan.execute(&mut world).len(), 0);
}

#[test]
fn for_each_no_changed_pays_zero_cost() {
    // Queries without Changed<T> must still work normally
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(&Score,)>().build();

    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 5);

    // Call again — no Changed<T>, so should still see everything
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 5);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib -- for_each_changed`
Expected: FAIL — `Changed<Score>` scan doesn't filter.

- [ ] **Step 3: Add `left_changed` field to `ScanBuilder`**

```rust
pub struct ScanBuilder<'w> {
    planner: &'w QueryPlanner<'w>,
    query_name: &'static str,
    estimated_rows: usize,
    predicates: Vec<Predicate>,
    joins: Vec<JoinSpec>,
    compile_for_each: Option<Box<dyn FnOnce() -> CompiledForEach>>,
    compile_for_each_raw: Option<Box<dyn FnOnce() -> CompiledForEachRaw>>,
    left_required: Option<FixedBitSet>,
    left_changed: Option<FixedBitSet>,
}
```

- [ ] **Step 4: Add `right_changed` field to `JoinSpec`**

```rust
struct JoinSpec {
    right_query_name: &'static str,
    right_estimated_rows: usize,
    join_kind: JoinKind,
    right_required: FixedBitSet,
    right_changed: FixedBitSet,
}
```

- [ ] **Step 5: Capture `changed_ids` in `scan::<Q>()` and `scan_with_estimate::<Q>()`**

In `scan::<Q>()` (around line 1643), add capture and thread through closures:

```rust
pub fn scan<Q: crate::query::fetch::WorldQuery + 'static>(&'w self) -> ScanBuilder<'w> {
    let required = Q::required_ids(self.components);
    let changed = Q::changed_ids(self.components);
    let required_for_each = required.clone();
    let changed_for_each = changed.clone();
    let required_for_each_raw = required.clone();
    let changed_for_each_raw = changed.clone();
    let left_required = required.clone();
    let left_changed = changed.clone();
    ScanBuilder {
        planner: self,
        query_name: std::any::type_name::<Q>(),
        estimated_rows: self.total_entities,
        predicates: Vec::new(),
        joins: Vec::new(),
        compile_for_each: Some(Box::new(move || {
            let required = required_for_each;
            let changed = changed_for_each;
            Box::new(move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                for arch in &world.archetypes.archetypes {
                    if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                        continue;
                    }
                    if !changed.is_clear()
                        && !changed.ones().all(|bit| {
                            arch.column_index(bit).map_or(false, |col| {
                                arch.columns[col].changed_tick.is_newer_than(tick)
                            })
                        })
                    {
                        continue;
                    }
                    for &entity in &arch.entities {
                        callback(entity);
                    }
                }
            })
        })),
        compile_for_each_raw: Some(Box::new(move || {
            let required = required_for_each_raw;
            let changed = changed_for_each_raw;
            Box::new(move |world: &World, tick: Tick, callback: &mut dyn FnMut(Entity)| {
                for arch in &world.archetypes.archetypes {
                    if arch.is_empty() || !required.is_subset(&arch.component_ids) {
                        continue;
                    }
                    if !changed.is_clear()
                        && !changed.ones().all(|bit| {
                            arch.column_index(bit).map_or(false, |col| {
                                arch.columns[col].changed_tick.is_newer_than(tick)
                            })
                        })
                    {
                        continue;
                    }
                    for &entity in &arch.entities {
                        callback(entity);
                    }
                }
            })
        })),
        left_required: Some(left_required),
        left_changed: Some(left_changed),
    }
}
```

Apply the identical pattern to `scan_with_estimate::<Q>()`.

- [ ] **Step 6: Capture `changed_ids` in `join::<Q>()`**

In `join::<Q>()` (around line 1196):

```rust
pub fn join<Q: crate::query::fetch::WorldQuery + 'static>(
    mut self,
    join_kind: JoinKind,
) -> Self {
    let right_rows = self.planner.total_entities;
    let required = Q::required_ids(self.planner.components);
    let changed = Q::changed_ids(self.planner.components);
    self.joins.push(JoinSpec {
        right_query_name: std::any::type_name::<Q>(),
        right_estimated_rows: right_rows,
        join_kind,
        right_required: required,
        right_changed: changed,
    });
    self
}
```

- [ ] **Step 7: Update `collect_matching_entities` to filter by changed ticks**

```rust
fn collect_matching_entities(
    world: &World,
    required: &FixedBitSet,
    changed: &FixedBitSet,
    tick: Tick,
    scratch: &mut ScratchBuffer,
) {
    for arch in &world.archetypes.archetypes {
        if arch.is_empty() || !required.is_subset(&arch.component_ids) {
            continue;
        }
        if !changed.is_clear()
            && !changed.ones().all(|bit| {
                arch.column_index(bit)
                    .map_or(false, |col| arch.columns[col].changed_tick.is_newer_than(tick))
            })
        {
            continue;
        }
        for &entity in &arch.entities {
            scratch.push(entity);
        }
    }
}
```

- [ ] **Step 8: Update build Phase 7 — left collector with `left_changed`**

In build Phase 7, extract `left_changed` alongside `left_required`:

```rust
let left_required = self
    .left_required
    .clone()
    .expect("join plan requires left_required bitset");
let left_changed = self
    .left_changed
    .clone()
    .unwrap_or_default();
```

Update the left collector closure:

```rust
let left_collector: EntityCollector =
    Box::new(move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
        if left_filters.is_empty() {
            collect_matching_entities(world, &left_required, &left_changed, tick, scratch);
        } else {
            for arch in &world.archetypes.archetypes {
                if arch.is_empty() || !left_required.is_subset(&arch.component_ids) {
                    continue;
                }
                if !left_changed.is_clear()
                    && !left_changed.ones().all(|bit| {
                        arch.column_index(bit).map_or(false, |col| {
                            arch.columns[col].changed_tick.is_newer_than(tick)
                        })
                    })
                {
                    continue;
                }
                for &entity in &arch.entities {
                    if left_filters.iter().all(|f| f(world, entity)) {
                        scratch.push(entity);
                    }
                }
            }
        }
    });
```

Update right collector closures to use `right_changed`:

```rust
let steps: Vec<JoinStep> = self
    .joins
    .iter()
    .map(|join| {
        let right_required = join.right_required.clone();
        let right_changed = join.right_changed.clone();
        JoinStep {
            right_collector: Box::new(
                move |world: &World, tick: Tick, scratch: &mut ScratchBuffer| {
                    collect_matching_entities(
                        world,
                        &right_required,
                        &right_changed,
                        tick,
                        scratch,
                    );
                },
            ),
            join_kind: join.join_kind,
        }
    })
    .collect();
```

- [ ] **Step 9: Run tests**

Run: `cargo test -p minkowski --lib -- for_each_changed`
Run: `cargo test -p minkowski --lib -- execute_changed`
Run: `cargo test -p minkowski --lib -- for_each_no_changed`
Expected: All PASS.

- [ ] **Step 10: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests PASS.

- [ ] **Step 11: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "feat: wire Changed<T> filtering into query executor scan closures"
```

---

### Task 4: Join integration tests and edge cases

**Files:**
- Modify: `crates/minkowski/src/planner.rs` (test module)

- [ ] **Step 1: Write join-specific Changed<T> tests**

```rust
#[test]
fn execute_join_changed_left_only() {
    let mut world = World::new();
    // All entities have both Score and Team
    for i in 0..5u32 {
        world.spawn((Score(i), Team(i % 2)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(Changed<Score>, &Score)>()
        .join::<(&Team,)>(JoinKind::Inner)
        .build();

    // First call: all changed
    assert_eq!(plan.execute(&mut world).len(), 5);

    // Second call: nothing changed
    assert_eq!(plan.execute(&mut world).len(), 0);
}

#[test]
fn execute_join_changed_right_only() {
    let mut world = World::new();
    for i in 0..5u32 {
        world.spawn((Score(i), Team(i % 2)));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner
        .scan::<(&Score,)>()
        .join::<(Changed<Team>, &Team)>(JoinKind::Inner)
        .build();

    // First call: all changed
    assert_eq!(plan.execute(&mut world).len(), 5);

    // Second call: right side not changed but left has no Changed filter —
    // left still yields all 5. Right yields 0 (Changed<Team> stale).
    // Inner join: intersection of 5 and 0 = 0
    assert_eq!(plan.execute(&mut world).len(), 0);
}

#[test]
fn for_each_raw_then_for_each_advances() {
    let mut world = World::new();
    for i in 0..3u32 {
        world.spawn((Score(i),));
    }

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // for_each_raw: sees everything, doesn't advance tick
    let mut count = 0;
    plan.for_each_raw(&world, |_| count += 1);
    assert_eq!(count, 3);

    // for_each: also sees everything (tick still at 0), and advances
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 3);

    // for_each again: tick advanced, nothing changed
    let mut count = 0;
    plan.for_each(&mut world, |_| count += 1);
    assert_eq!(count, 0);
}

#[test]
fn for_each_changed_multiple_archetypes_partial() {
    let mut world = World::new();
    // Archetype 1: (Score,)
    let e1 = world.spawn((Score(1),));
    // Archetype 2: (Score, Team)
    let _e2 = world.spawn((Score(2), Team(0)));

    let planner = QueryPlanner::new(&world);
    let mut plan = planner.scan::<(Changed<Score>, &Score)>().build();

    // Consume initial changes
    plan.for_each(&mut world, |_| {});

    // Mutate only archetype 1
    let _ = world.get_mut::<Score>(e1);

    // Only archetype 1 should be returned
    let mut found = Vec::new();
    plan.for_each(&mut world, |entity| found.push(entity));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], e1);
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p minkowski --lib -- execute_join_changed`
Run: `cargo test -p minkowski --lib -- for_each_raw_then_for_each`
Run: `cargo test -p minkowski --lib -- for_each_changed_multiple`
Expected: All PASS.

- [ ] **Step 3: Run full test suite + clippy**

Run: `cargo test -p minkowski --lib`
Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: All PASS, no warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/minkowski/src/planner.rs
git commit -m "test: Changed<T> join integration and edge case tests"
```

---

### Task 5: Update planner example and documentation

**Files:**
- Modify: `examples/examples/planner.rs`
- Modify: `CLAUDE.md` (Query Planner section if needed)

- [ ] **Step 1: Add Changed<T> section to the planner example**

Add a new section demonstrating `Changed<T>` with `for_each`. Show the pattern: first
call sees everything, mutate, second call sees only changes, third call sees nothing.

Use existing example components (look at what the planner example already imports) and
`world.get_mut` for mutation.

- [ ] **Step 2: Run the example**

Run: `cargo run -p minkowski-examples --example planner --release`
Expected: New section output showing changed/unchanged counts.

- [ ] **Step 3: Commit**

```bash
git add examples/examples/planner.rs
git commit -m "docs: add Changed<T> section to planner example"
```
