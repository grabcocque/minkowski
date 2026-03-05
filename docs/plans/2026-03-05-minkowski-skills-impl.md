# Minkowski Skills & Example Rewrites Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a comprehensive Claude Code skill + 8 slash commands for Minkowski users, and rewrite all 7 non-reducer examples to dogfood the reducer API.

**Architecture:** Pure markdown for skills/commands (no compilation). Rust rewrites for examples (must compile and run). Skills/commands ship first since they're independent; examples ordered simplest-to-most-complex.

**Tech Stack:** Markdown (skills/commands), Rust (examples), minkowski crate API (reducers, transactions, queries).

---

## Task 1: Auto-Triggering Skill

**Files:**
- Create: `.claude/skills/minkowski-guide.md`

**Context:** This is the core deliverable — a comprehensive skill that fires whenever Claude detects Minkowski usage. It contains decision frameworks, strong defaults, pitfall alerts, and example pointers. It does NOT duplicate CLAUDE.md — it adds the opinionated "what to use when" layer. Target length: 300-400 lines of markdown.

**Reference:** Read the design doc at `docs/plans/2026-03-05-minkowski-skills-design.md` for the full spec of what this skill contains (sections 1-5: quick reference, decision flowcharts, strong defaults, pitfall alerts, example pointers).

**Step 1: Create the skill file**

Write `.claude/skills/minkowski-guide.md` with YAML frontmatter and all 5 content sections. The frontmatter `description` field is what triggers the skill — it must mention: Minkowski, ECS, World, Entity, Component, query, reducer, transaction, persistence, spatial index, data modeling, concurrency model, performance optimization.

Structure:

```markdown
---
description: >
  [trigger description covering all Minkowski usage patterns]
---

# Minkowski ECS Guide

## Quick Reference
[one paragraph per subsystem: World, queries, transactions, reducers, persistence, spatial indexing]

## Decision Flowcharts
[if/then trees for each of the 8 areas from the design doc]

## Strong Defaults
[prescriptive recommendations organized by topic]

## Pitfall Alerts
[patterns to flag proactively, with explanations]

## Examples
[pointers to example files for each pattern]

## References
[pointer to CLAUDE.md for architecture details]
```

The skill should be self-contained enough that Claude can give correct guidance without reading any other files. But for deep dives, it points to examples and CLAUDE.md.

**Important details for the skill content:**

For the Quick Reference section, cover these subsystems:
- **World** — sole mutable entry point. `spawn/despawn/insert/remove/get/get_mut/query`. Entity = u64 (index + generation). Components are `'static + Send + Sync`.
- **Queries** — tuple composition: `&T` read, `&mut T` write, `Entity` ID, `Option<&T>` optional, `Changed<T>` filter. `for_each_chunk` for SIMD, `par_for_each` for rayon. Query cache auto-maintained.
- **Table** — `#[derive(Table)]` for fixed schemas. `query_table/query_table_mut` skip archetype matching. Named field access via generated `Ref<'w>`/`Mut<'w>` types.
- **Transactions** — `Sequential` (zero overhead), `Optimistic` (tick validation), `Pessimistic` (column locks). `Tx` doesn't hold `&mut World` — split-phase design enables parallel reads. `ReadOnlyWorldQuery` bound on shared-ref queries.
- **Reducers** — `ReducerRegistry` type-erases closures with Access metadata. Entity handles (`EntityMut<C>`), query handles (`QueryMut<Q>`, `QueryRef<Q>`, `QueryWriter<Q>`), spawner (`Spawner<B>`), dynamic (`DynamicCtx`). `call()` for transactional, `run()` for scheduled.
- **Persistence** — `Durable<S, W>` wraps any strategy with WAL logging. `Snapshot` for point-in-time saves. `CodecRegistry` for component serialization.
- **Spatial Indexing** — `SpatialIndex` trait with `rebuild` + optional `update`. External to World. Stale entries via `is_alive()`. Grid for uniform, tree for clustered.

For the Decision Flowcharts section, use clear if/then prose (not graphviz — this is a skill file read by Claude, not rendered as HTML):
- Data modeling decisions
- Query pattern decisions
- Mutation strategy decisions
- Concurrency model decisions
- Reducer type decisions
- Persistence decisions
- Spatial index decisions
- Performance optimization decisions

For the Strong Defaults section, organize by topic with bullet points. These are the prescriptive recommendations that apply to most users:
- "Start with tuple queries, graduate to Table when schema is fixed"
- "Start with Sequential, add Optimistic when concurrency is needed"
- "Use QueryWriter over QueryMut when the reducer needs to be durable"
- "Prefer `for_each_chunk` for numeric tight loops (SIMD)"
- "Every mutation should go through a reducer — direct World methods for setup only"
- "Register all systems in ReducerRegistry even single-threaded — free conflict detection"
- "One concept = one component. Copy types when possible."
- "Use sparse components for rare optional data"
- "CommandBuffer for structural changes during iteration"
- "EnumChangeSet for reversible/serializable mutations"
- "Always check `is_alive()` on entity references from spatial indexes"

For the Pitfall Alerts section, format as: pattern -> explanation -> recommendation:
- `world.insert()` in hot loop -> archetype migration is O(components) per entity -> batch structural changes or use remove-less designs
- `Changed<T>` misunderstanding -> "since last query observed this column", not per-frame -> first call always matches everything
- `query_raw()` vs `query()` -> `query_raw` skips cache and ticks, used by transactions -> don't use it for normal iteration
- Missing `is_alive()` on spatial index results -> despawned entities leave stale entries -> always validate
- `&mut T` in transaction query -> `tx.query(&world)` requires `ReadOnlyWorldQuery` -> use `tx.write()` for mutations
- `QueryMut` for durable reducer -> direct mutation incompatible with WAL -> use `QueryWriter` instead
- Dynamic reducer undeclared access -> panics in all builds (assert!, not debug_assert!) -> declare all accessed types in builder
- Forgetting `sync_reserved()` after snapshot restore -> entity ID overlap -> always call it

For the Examples section, point to each example with what it demonstrates:
- `boids.rs` — query reducers, spatial grid, SIMD vectorization via `for_each_chunk`
- `life.rs` — QueryWriter, Table derive, undo/redo via changeset reversal
- `nbody.rs` — query reducers, Barnes-Hut quadtree, parallel force computation
- `scheduler.rs` — ReducerRegistry for access metadata, conflict detection, batch scheduling
- `transaction.rs` — three strategies (Sequential/Optimistic/Pessimistic), raw Tx + reducer comparison
- `battle.rs` — EntityMut reducers, rayon parallel dispatch, tunable conflict rates
- `persist.rs` — QueryWriter + Durable, WAL, snapshots, recovery
- `reducer.rs` — comprehensive reducer reference (all handle types + dynamic + structural mutations)

**Step 2: Verify the skill appears in Claude Code**

Run: `ls -la .claude/skills/minkowski-guide.md`
Expected: File exists with reasonable size (300-400 lines).

**Step 3: Commit**

```bash
git add .claude/skills/minkowski-guide.md
git commit -m "feat: add minkowski-guide auto-triggering skill"
```

---

## Task 2: Eight Slash Commands

**Files:**
- Create: `.claude/commands/minkowski/model.md`
- Create: `.claude/commands/minkowski/query.md`
- Create: `.claude/commands/minkowski/mutate.md`
- Create: `.claude/commands/minkowski/concurrency.md`
- Create: `.claude/commands/minkowski/reducer.md`
- Create: `.claude/commands/minkowski/persist.md`
- Create: `.claude/commands/minkowski/index.md`
- Create: `.claude/commands/minkowski/optimize.md`

**Context:** Each command follows the same 3-step structure: Assess (examine user's code), Recommend (clear recommendation), Implement (write code or point to example). The design doc at `docs/plans/2026-03-05-minkowski-skills-design.md` § "Slash Commands" specifies the strong defaults and Socratic questions for each.

All commands share this YAML frontmatter pattern:

```yaml
---
description: [what this command helps with]
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---
```

No `args` — these commands assess the user's current code context automatically.

**Step 1: Create the directory**

```bash
mkdir -p .claude/commands/minkowski
```

**Step 2: Create all 8 command files**

Each command file should:
1. Start with YAML frontmatter (description + allowed-tools)
2. Instruct Claude to assess the user's current codebase (search for relevant imports, component types, existing patterns)
3. List the strong defaults as prescriptive guidance
4. List the Socratic questions for ambiguous decisions
5. Point to relevant example files
6. Reference the minkowski-guide skill for broader context

Here's the template each command should follow (adapt content per the design doc):

```markdown
---
description: [one-line description]
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

Help the user with [topic] in their Minkowski project.

## Step 1: Assess

Search the user's codebase for:
- [relevant patterns to look for]
- [existing code that reveals their situation]

## Step 2: Recommend

**Strong defaults:**
- [prescriptive recommendations from the design doc]

**Ask if unclear:**
- [Socratic questions from the design doc]

## Step 3: Implement

Help write the code or point to the relevant example:
- [example pointer]
- [code pattern]
```

Specific content for each command — draw from the design doc's slash command section:

**model.md**: Search for component struct definitions, spawn calls, Table derives. Recommend one-concept-per-component, Copy types, Table for fixed schemas, sparse for rare data.

**query.md**: Search for `world.query` calls, for_each/for_each_chunk/par_for_each usage. Recommend &T/&mut T/Entity/Option/Changed patterns, chunk iteration for SIMD.

**mutate.md**: Search for world.insert/remove in loops, CommandBuffer usage, EnumChangeSet usage. Recommend CommandBuffer for structural changes during iteration, EnumChangeSet for undo, reducer handles for transactions.

**concurrency.md**: Search for Optimistic/Pessimistic/Sequential imports, rayon usage, thread::spawn. Recommend Sequential first, Optimistic for concurrency, Pessimistic for write-heavy, Durable for persistence.

**reducer.md**: Search for ReducerRegistry, register_entity/register_query/etc calls, DynamicCtx. Recommend EntityMut for single-entity, QueryMut for bulk direct, QueryWriter for bulk buffered/durable, Dynamic for runtime-conditional.

**persist.md**: Search for minkowski_persist imports, Wal/Snapshot/CodecRegistry/Durable. Recommend WAL for crash safety, snapshots for fast restore, sync_reserved after restore.

**index.md**: Search for SpatialIndex impl, grid/tree structs, neighbor queries. Recommend grid for uniform, tree for clustered, is_alive checks, rebuild vs incremental.

**optimize.md**: Search for repr(align), for_each_chunk, par_for_each, spawn/insert in loops. Recommend align(16) for SIMD, for_each_chunk for vectorization, minimize migrations.

**Step 3: Verify commands appear**

Run: `ls .claude/commands/minkowski/`
Expected: 8 `.md` files.

**Step 4: Commit**

```bash
git add .claude/commands/minkowski/
git commit -m "feat: add 8 /minkowski: slash commands for guided decision-making"
```

---

## Task 3: Rewrite `scheduler.rs`

**Files:**
- Modify: `examples/examples/scheduler.rs`

**Context:** Simplest rewrite. Currently uses manual `Access::of::<Q>()` for 6 systems with function pointers. Rewrite to register all 6 systems as reducers in a `ReducerRegistry`, then use `query_reducer_access()` / `reducer_access()` instead of manual Access construction. The conflict detection and greedy graph coloring logic stays the same.

**Current structure (224 lines):**
- 6 system functions taking `&mut World` (lines 36-73)
- `SystemEntry` struct with `name`, `access`, `run` function pointer (line 76)
- `build_batches()` greedy coloring (lines 82-104)
- `conflict_matrix()` prints NxN conflicts (lines 106-109)
- `main()`: spawn entities, build SystemEntry vec with manual `Access::of`, print conflicts, run batches

**New structure:**
- Same 6 system closures, but registered via `registry.register_query` / `registry.register_query_ref`
- `ReducerRegistry` replaces the manual `SystemEntry` vec
- Use `registry.query_reducer_access(id)` to get Access metadata
- Conflict matrix and batch building operate on `QueryReducerId` handles
- Frame loop calls `registry.run()` per system

**Step 1: Rewrite the example**

Key changes:
1. Replace `use minkowski::{Access, World}` with `use minkowski::{Access, QueryMut, QueryRef, QueryReducerId, ReducerRegistry, World}`
2. Remove `SystemEntry` struct and the 6 free functions
3. Register systems as query reducers: `register_query` for mutable systems, `register_query_ref` for read-only
4. Build batches using `Vec<QueryReducerId>` + `registry.query_reducer_access(id)`
5. Execute via `registry.run(&mut world, id, args)` in batch order
6. Keep the conflict matrix printing and greedy coloring algorithm

The 6 systems become closures:

```rust
// Movement: reads Vel, writes Pos
let movement = registry.register_query::<(&mut Pos, &Vel), f32, _>(
    &mut world, "movement",
    |mut query: QueryMut<(&mut Pos, &Vel)>, dt: f32| {
        query.for_each(|(pos, vel)| {
            pos.x += vel.dx * dt;
            pos.y += vel.dy * dt;
        });
    },
);

// Gravity: writes Vel
let gravity = registry.register_query::<(&mut Vel,), f32, _>(
    &mut world, "gravity",
    |mut query: QueryMut<(&mut Vel,)>, dt: f32| {
        query.for_each(|(vel,)| {
            vel.dy -= 9.81 * dt;
        });
    },
);

// Damping: writes Vel
let damping = registry.register_query::<(&mut Vel,), f32, _>(
    &mut world, "damping",
    |mut query: QueryMut<(&mut Vel,)>, dt: f32| {
        query.for_each(|(vel,)| {
            vel.dx *= 0.99;
            vel.dy *= 0.99;
        });
    },
);

// Health regen: writes Health
let regen = registry.register_query::<(&mut Health,), (), _>(
    &mut world, "health_regen",
    |mut query: QueryMut<(&mut Health,)>, ()| {
        query.for_each(|(health,)| {
            health.0 = (health.0 + 1).min(100);
        });
    },
);

// Logger: reads Pos, Vel (read-only)
let logger = registry.register_query_ref::<(&Pos, &Vel), (), _>(
    &mut world, "logger",
    |mut query: QueryRef<(&Pos, &Vel)>, ()| {
        let count = query.count();
        println!("  [logger] {} entities", count);
    },
);

// Health logger: reads Health (read-only)
let health_logger = registry.register_query_ref::<(&Health,), (), _>(
    &mut world, "health_logger",
    |mut query: QueryRef<(&Health,)>, ()| {
        let count = query.count();
        println!("  [health_logger] {} entities", count);
    },
);
```

Collect IDs into a vec for batching. The batch builder uses `registry.query_reducer_access(id)` instead of stored `access` fields.

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Expected: No warnings.

Run: `cargo run -p minkowski-examples --example scheduler --release`
Expected: Prints conflict matrix, batches, and frame execution. Output structure should match original (system names, conflict pairs, batch assignments, per-frame logs).

**Step 3: Commit**

```bash
git add examples/examples/scheduler.rs
git commit -m "refactor(examples): scheduler uses ReducerRegistry for access metadata"
```

---

## Task 4: Rewrite `transaction.rs`

**Files:**
- Modify: `examples/examples/transaction.rs`

**Context:** Currently demonstrates 3 transaction strategies using raw `Tx` API. Rewrite to have two parts: (1) brief raw Tx demo showing the building blocks, (2) same logic via `EntityMut` reducers dispatched through `registry.call()`. This shows how reducers abstract the transaction boilerplate.

**Current structure (228 lines):**
- Components: `Pos { x, y }`, `Vel { dx, dy }`, `Health(u32)`
- `spawn_world()` creates 100 entities
- Sequential: `tx.query()` + direct sequential mutation
- Optimistic: `transact()` closure with `tx.query()` + `tx.write()`
- Pessimistic: Same API, different strategy
- Helper functions: `avg_pos()`, `avg_health()`

**New structure:**
- Keep Part 1: raw Tx demo (abbreviated — just Sequential + one Optimistic example to show the primitives)
- Add Part 2: register `move_entities` as a query reducer, `decay_health` as an entity reducer
- Dispatch same operations via `registry.call()` / `registry.run()` across all 3 strategies
- Show that the reducer API produces identical results with less boilerplate

**Step 1: Rewrite the example**

Key changes:
1. Add imports: `EntityMut, Optimistic, Pessimistic, QueryMut, ReducerRegistry, ReducerId, QueryReducerId`
2. Keep `spawn_world()`, `avg_pos()`, `avg_health()` helpers
3. Part 1 (raw Tx): abbreviated demo of Sequential begin/commit + Optimistic transact closure. ~40 lines.
4. Part 2 (reducers): register `move_entities` as `register_query::<(&mut Pos, &Vel), f32, _>` and `decay_health` as `register_query::<(&mut Health,), u32, _>`. Dispatch via `registry.run()` for all three strategies (Sequential is just direct run; Optimistic and Pessimistic go through `call()` for entity reducers). Show conflict demo using Access metadata.

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Run: `cargo run -p minkowski-examples --example transaction --release`
Expected: Both parts produce correct output — movement updates and health decay visible in avg stats.

**Step 3: Commit**

```bash
git add examples/examples/transaction.rs
git commit -m "refactor(examples): transaction shows raw Tx + reducer comparison"
```

---

## Task 5: Rewrite `persist.rs`

**Files:**
- Modify: `examples/examples/persist.rs`

**Context:** Currently uses raw `Durable` transactions with `tx.query()` + `tx.write()`. Rewrite to use a `QueryWriter` reducer — the motivating use case for QueryWriter (buffered writes compatible with WAL). Snapshot save/restore stays unchanged.

**Current structure (131 lines):**
- Components: `Pos { x, y }`, `Vel { dx, dy }` (Serialize/Deserialize)
- Phase 1: Create world with 100 entities
- Phase 2: Save snapshot + create WAL
- Phase 3: 10 durable transaction frames with `durable.transact()`
- Phase 4: Recover from snapshot + WAL replay

**New structure:**
- Same 4 phases, but Phase 3 uses a `QueryWriter` reducer registered in `ReducerRegistry`
- Register `apply_velocity` as `register_query_writer::<(&mut Pos, &Vel), f32, _>`
- Dispatch via `registry.call(&durable, &mut world, id, dt)`
- The Durable wrapper automatically logs the changeset to WAL on commit

**Step 1: Rewrite the example**

Key changes:
1. Add imports: `QueryWriter, ReducerRegistry, WritableRef`
2. Register `apply_velocity` QueryWriter reducer: reads Vel, buffers Pos writes via `WritableRef::modify()`
3. Replace raw `durable.transact()` with `registry.call(&durable, &mut world, writer_id, dt)`
4. Keep snapshot save/load and WAL replay exactly as-is
5. Keep recovery verification (comparing entity states before/after)

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Run: `cargo run -p minkowski-examples --example persist --release`
Expected: Same output — 10 frames of movement, snapshot save, WAL recovery, state matches.

**Step 3: Commit**

```bash
git add examples/examples/persist.rs
git commit -m "refactor(examples): persist uses QueryWriter reducer for durable mutations"
```

---

## Task 6: Rewrite `battle.rs`

**Files:**
- Modify: `examples/examples/battle.rs`

**Context:** Currently uses split-phase `Tx` API with rayon for parallel reads. Rewrite attack and heal as `EntityMut` reducers dispatched via `ReducerRegistry`. Rayon dispatches `registry.call()` in parallel. The tunable conflict rate parameter stays.

**Current structure (501 lines):**
- Components: `Health(u32)`, `Team(u8)`, `Damage(u32)`, `Healing(u32)`
- `compute_combat()` / `compute_healing()` — snapshot-based parallel reads
- `apply_effects()` — applies damage/healing to Health
- Split-phase: `strategy.begin()` → parallel reads via rayon → sequential writes → `try_commit()`
- Two conflict modes: low (disjoint Damage/Healing) and high (both write Health)
- Two strategies: Optimistic and Pessimistic

**New structure:**
- Register `attack` and `heal` as `EntityMut<(Health,)>` reducers (both read and write Health)
- For the low-conflict mode: register `apply_damage` writing `Damage` and `apply_healing` writing `Healing` as disjoint reducers
- Rayon dispatches `registry.call()` in parallel — the retry loop is handled internally by the strategy
- Keep `FrameStats` tracking (retries, successes, conflicts)
- The conflict rate parameter controls whether reducers write overlapping or disjoint components

The key simplification: instead of manual `begin/read/write/try_commit`, each entity operation becomes a single `registry.call()`. The registry + strategy handle retry internally. Rayon just calls `registry.call()` from multiple threads.

**Step 1: Rewrite the example**

This is a significant rewrite (~500 lines). Key structure:
1. Keep component definitions and `spawn_arena()`
2. Register entity reducers for attack and heal
3. Frame function: collect entity pairs via snapshot, dispatch via `rayon::par_iter` calling `registry.call()` per entity
4. Track retries/conflicts via `AtomicU32` counters (same as current `FrameStats`)
5. Keep both Optimistic and Pessimistic scenarios
6. Keep the 4-scenario matrix (low/high conflict × optimistic/pessimistic)

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Run: `cargo run -p minkowski-examples --example battle --release`
Expected: 4 scenarios execute, stats printed per frame. Conflict rates should roughly match original behavior.

**Step 3: Commit**

```bash
git add examples/examples/battle.rs
git commit -m "refactor(examples): battle uses EntityMut reducers with rayon dispatch"
```

---

## Task 7: Rewrite `boids.rs`

**Files:**
- Modify: `examples/examples/boids.rs`

**Context:** Currently uses direct `world.query()` + `world.get_mut()` with snapshot-based parallel force computation. Rewrite to use query reducers for each boid rule, keep spatial grid external.

**Current structure (430 lines):**
- Components: `Position(Vec2)`, `Velocity(Vec2)`, `Acceleration(Vec2)`
- `Vec2` helper struct with arithmetic
- `SpatialGrid` implementing `SpatialIndex`
- `BoidParams` configuration
- Frame loop: zero accel → rebuild grid → parallel force compute → apply forces → integrate → spawn/despawn churn

**New structure:**
- Keep `Vec2`, `SpatialGrid`, `BoidParams` unchanged
- Register 4 query reducers:
  - `zero_accel`: `QueryMut<(&mut Acceleration,)>` — zeros out
  - `apply_boid_forces`: `QueryMut<(&mut Acceleration, &Position, &Velocity)>` — reads grid snapshot, applies separation/alignment/cohesion (the grid snapshot is captured by the closure via shared reference)
  - `integrate`: `QueryMut<(&mut Position, &mut Velocity, &Acceleration)>` — Euler step with velocity clamping
  - `spawn_despawn_churn`: needs `CommandBuffer` or a `Spawner` + dynamic with `can_despawn()` — for the entity churn demo

The spatial grid stays external (called between frames). The grid snapshot can be captured by the force closure since it's rebuilt each frame before the reducer runs.

Note: The parallel force computation currently uses rayon with a snapshot. With query reducers, `QueryMut::for_each()` is sequential (it has `&mut World`). For the parallel path, keep the snapshot approach outside the reducer for the force computation, then apply accumulated forces via a reducer. Or accept sequential execution (boids at 5K entities is fast enough).

**Decision for implementer:** Prefer simplicity — make `apply_boid_forces` a single `QueryMut` that reads the grid snapshot and applies all 3 rules. The snapshot is built before the reducer runs. If the parallel path was important for the demo, keep it as a separate parallel computation step followed by a force-application reducer.

**Step 1: Rewrite the example**

Key changes:
1. Add imports: `QueryMut, QueryReducerId, ReducerRegistry`
2. Keep `Vec2`, `SpatialGrid`, `BoidParams` exactly as-is
3. Register query reducers for each simulation step
4. Frame loop: `grid.rebuild()` → `registry.run()` for each reducer in order
5. Keep stats logging and spawn/despawn churn (churn can remain as direct World methods since it's setup-like, or use a Spawner reducer)

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Run: `cargo run -p minkowski-examples --example boids --release`
Expected: 5K entities, 1K frames. Stats should show similar performance to original.

**Step 3: Commit**

```bash
git add examples/examples/boids.rs
git commit -m "refactor(examples): boids uses query reducers for simulation steps"
```

---

## Task 8: Rewrite `nbody.rs`

**Files:**
- Modify: `examples/examples/nbody.rs`

**Context:** Currently uses direct query + `get_mut` with snapshot-based parallel force computation. Rewrite to use query reducers, keep Barnes-Hut tree external.

**Current structure (582 lines):**
- Components: `Position(Vec2)`, `Velocity(Vec2)`, `Mass(f32)`
- `Vec2`, `Rect` (AABB), `QuadNode`, `BarnesHutTree` (implements `SpatialIndex`)
- Frame loop: rebuild tree → snapshot → parallel force compute → apply forces → integrate → spawn/despawn churn

**New structure:**
- Keep `Vec2`, `Rect`, `QuadNode`, `BarnesHutTree` unchanged
- Register query reducers:
  - `calculate_forces`: `QueryMut<(&mut Velocity, &Position, &Mass)>` — traverses tree, accumulates forces. Tree captured by closure.
  - `integrate`: `QueryMut<(&mut Position, &Velocity)>` — Verlet/Euler step with toroidal wrapping

Same parallel consideration as boids: the snapshot-based parallel force computation can remain outside the reducer for performance, with a force-application reducer afterwards. Or go sequential if simpler.

**Step 1: Rewrite the example**

Same pattern as boids:
1. Add imports
2. Keep tree and helper structs
3. Register reducers
4. Frame loop: `tree.rebuild()` → `registry.run()` per reducer
5. Keep churn and stats

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Run: `cargo run -p minkowski-examples --example nbody --release`
Expected: 2K entities, 1K frames. Stats should show similar performance.

**Step 3: Commit**

```bash
git add examples/examples/nbody.rs
git commit -m "refactor(examples): nbody uses query reducers for force and integration steps"
```

---

## Task 9: Rewrite `life.rs`

**Files:**
- Modify: `examples/examples/life.rs`

**Context:** Currently builds `EnumChangeSet` manually per generation. Rewrite the step function to use a `QueryWriter` reducer — reads cell state via `WritableRef`, counts neighbors, buffers birth/death writes through `modify()`. Table derive and undo/redo stay as-is.

**Current structure (265 lines):**
- Components: `CellState(bool)`, `NeighborCount(u8)`, `Cell` (`#[derive(Table)]`)
- Helper functions: `snapshot_states()`, `count_neighbors()`, `write_neighbor_counts()`, `apply_rules()`, `apply_updates()`, `alive_count()`
- Main loop: snapshot → count neighbors → write counts → apply rules via EnumChangeSet → push undo
- Rewind: apply reverses. Replay: re-simulate.

**New structure:**
- Register a `QueryWriter` reducer for the `apply_rules` step: reads `CellState` + `NeighborCount`, buffers writes to `CellState`
- The changeset from `registry.call()` captures undo automatically (the Optimistic strategy's changeset can be extracted for undo)
- Table derive and query_table usage stay for neighbor counting (that's what Table is good for)
- Keep the undo/redo stack

**Important consideration:** QueryWriter goes through `strategy.transact()`, which produces a changeset. For undo, we need to capture the reverse changeset. Check whether `registry.call()` returns the reverse — if not, the undo approach may need to stay as manual EnumChangeSet for this demo, with the neighbor counting and integration steps using reducers.

**Alternative if QueryWriter undo is complex:** Keep the EnumChangeSet-based undo as-is (it's demonstrating a specific pattern), but convert the `write_neighbor_counts` step into a `QueryMut` reducer and the `apply_rules` step into a `QueryWriter` reducer. The undo stack captures changesets from the QueryWriter's commit.

**Step 1: Rewrite the example**

Key changes:
1. Add imports: `Optimistic, QueryMut, QueryWriter, ReducerRegistry, WritableRef`
2. Keep Table derive, `Cell`, `CellState`, `NeighborCount`
3. Keep `snapshot_states()`, `count_neighbors()`, `alive_count()` helpers
4. Register `write_neighbor_counts` as `QueryMut` (writes NeighborCount based on snapshot)
5. Register `apply_rules` as `QueryWriter` (reads NeighborCount + CellState, buffers CellState writes)
6. For undo: if changeset capture from registry.call() is available, use it. Otherwise keep manual EnumChangeSet for the apply_rules step and note this is demonstrating changeset undo.
7. Keep rewind and replay phases

**Step 2: Verify it compiles and runs**

Run: `cargo clippy -p minkowski-examples -- -D warnings 2>&1 | head -30`
Run: `cargo run -p minkowski-examples --example life --release`
Expected: 64x64 grid, 500 generations. Rewind and replay produce same alive counts.

**Step 3: Commit**

```bash
git add examples/examples/life.rs
git commit -m "refactor(examples): life uses QueryWriter reducer for rule application"
```

---

## Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Context:** Update example descriptions in the Build & Test Commands section and any stale test counts.

**Step 1: Update example descriptions**

Update each `cargo run` comment to reflect the new reducer-based structure:
- `boids` — "Boids flocking with query reducers + spatial grid"
- `life` — "Game of Life with QueryWriter, Table, undo/redo"
- `nbody` — "Barnes-Hut N-body with query reducers"
- `scheduler` — "ReducerRegistry-based conflict detection + batch scheduling"
- `transaction` — "Transaction strategies: raw Tx + reducer comparison"
- `battle` — "Multi-threaded EntityMut reducers with tunable conflict"
- `persist` — "Durable QueryWriter reducer: WAL + snapshots"

**Step 2: Update test count**

Run: `cargo test -p minkowski --lib 2>&1 | tail -3`
Update the test count in the `cargo test` comment if it changed.

**Step 3: Verify and commit**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md example descriptions for reducer-based examples"
```
