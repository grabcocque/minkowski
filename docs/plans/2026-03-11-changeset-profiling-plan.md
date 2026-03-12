# EnumChangeSet Profiling Investigation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Profile QueryWriter vs QueryMut to find where the 14x overhead lives, then add targeted microbenchmarks to isolate individual costs.

**Architecture:** Standalone profiling binary for flamegraph capture (samply), separate from criterion. Criterion microbenchmarks for isolated cost measurement of individual operations identified by profiling.

**Tech Stack:** samply (flamegraph profiler), criterion (microbenchmarks), minkowski ECS

---

### Task 1: Create the profiling binary

**Files:**
- Create: `examples/examples/profile_changeset.rs`

The binary must exercise the exact same workload as the reducer benchmarks
(Position += Velocity over 10K entities) but in a tight loop without criterion
overhead. Two clearly separated phases: QueryMut baseline, then QueryWriter subject.

**Step 1: Write the profiling binary**

```rust
//! Profiling harness for QueryWriter vs QueryMut overhead comparison.
//!
//! NOT a benchmark — this binary is designed for flamegraph capture with samply.
//! Criterion's measurement harness pollutes profiles with statistics overhead.
//!
//! Usage:
//!   cargo build -p minkowski-examples --example profile_changeset --release
//!   samply record target/release/examples/profile_changeset
//!
//! The profiler will show two distinct call subtrees under main():
//!   run_query_mut()   — baseline (direct mutation)
//!   run_query_writer() — subject (buffered via EnumChangeSet)
//!
//! Compare time spent in each to identify where the overhead lives.

use minkowski::{Optimistic, QueryMut, QueryWriter, ReducerRegistry, World};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

const ENTITY_COUNT: usize = 10_000;
const ITERATIONS: usize = 1_000;

fn setup_world() -> World {
    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
            Velocity {
                dx: 1.0,
                dy: 0.0,
                dz: 0.0,
            },
        ));
    }
    world
}

/// Baseline: direct mutation via QueryMut (scheduled reducer, no changeset).
#[inline(never)]
fn run_query_mut(world: &mut World, registry: &mut ReducerRegistry) {
    let id = registry
        .register_query::<(&mut Position, &Velocity), (), _>(
            world,
            "integrate_mut",
            |mut query: QueryMut<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(pos, vel)| {
                    pos.x += vel.dx;
                    pos.y += vel.dy;
                    pos.z += vel.dz;
                });
            },
        )
        .unwrap();

    for _ in 0..ITERATIONS {
        registry.run(world, id, ()).unwrap();
    }
}

/// Subject: buffered mutation via QueryWriter (transactional, uses EnumChangeSet).
#[inline(never)]
fn run_query_writer(world: &mut World, registry: &mut ReducerRegistry) {
    let strategy = Optimistic::new(world);
    let id = registry
        .register_query_writer::<(&mut Position, &Velocity), (), _>(
            world,
            "integrate_writer",
            |mut query: QueryWriter<'_, (&mut Position, &Velocity)>, (): ()| {
                query.for_each(|(mut pos, vel)| {
                    pos.modify(|p| {
                        p.x += vel.dx;
                        p.y += vel.dy;
                        p.z += vel.dz;
                    });
                });
            },
        )
        .unwrap();

    for _ in 0..ITERATIONS {
        registry.call(&strategy, world, id, ()).unwrap();
    }
}

fn main() {
    // Phase 1: QueryMut baseline
    let mut world = setup_world();
    let mut registry = ReducerRegistry::new();
    run_query_mut(&mut world, &mut registry);
    std::hint::black_box(&world);

    // Phase 2: QueryWriter subject
    let mut world = setup_world();
    let mut registry = ReducerRegistry::new();
    run_query_writer(&mut world, &mut registry);
    std::hint::black_box(&world);
}
```

**Step 2: Build and verify the binary runs**

Run: `cargo build -p minkowski-examples --example profile_changeset --release && target/release/examples/profile_changeset`

Expected: exits cleanly in ~2 seconds, no output.

**Step 3: Commit**

```bash
git add examples/examples/profile_changeset.rs
git commit -m "Add profiling harness for QueryWriter vs QueryMut comparison"
```

---

### Task 2: Capture flamegraphs

This task is interactive — the operator captures profiles and interprets them.

**Step 1: Profile with samply**

Run:
```bash
samply record target/release/examples/profile_changeset
```

This opens Firefox Profiler in the browser. Look at the call tree view.

**Step 2: Compare the two phases**

In the profiler, find the two `#[inline(never)]` functions:
- `run_query_mut` — baseline. Note total self-time.
- `run_query_writer` — subject. Note total self-time and drill into children.

**Step 3: Identify the top costs in run_query_writer**

Expand the `run_query_writer` call tree and note what percentage of time is spent in:

1. **The for_each iteration** (archetype scanning, fetch setup)
2. **WritableRef::modify** → clone + closure + set
3. **insert_raw** → ManuallyDrop + record_insert → Arena::alloc
4. **Mutation vec push** (growing the Vec<Mutation>)
5. **transact()** overhead (begin/commit/tick validation)
6. **changeset.apply()** (apply loop: entity lookup + memcpy + tick mark)
7. **EnumChangeSet::new() / drop** (allocation + deallocation per call)

Record these as percentages. The top 2-3 costs become the microbenchmark targets.

**Step 4: Document findings**

Create a findings section in the design doc or a separate file:
```
docs/plans/2026-03-11-changeset-profiling-findings.md
```

Format:
```markdown
# Profiling Findings

## QueryMut baseline
Total: X ms for 1000 iterations = Y µs/iter

## QueryWriter breakdown
Total: X ms for 1000 iterations = Y µs/iter

| Cost center | % of total | Absolute |
|---|---|---|
| ... | ...% | ... µs/iter |
```

---

### Task 3: Add changeset microbenchmarks

**Files:**
- Create: `crates/minkowski-bench/benches/changeset_micro.rs`
- Modify: `crates/minkowski-bench/Cargo.toml` (add `[[bench]]` entry)

Based on profiling findings from Task 2, write targeted criterion benchmarks.
The candidates below cover the most likely bottlenecks — include only those
confirmed by profiling, skip any that showed negligible cost.

**Step 1: Add the bench entry to Cargo.toml**

Add to `crates/minkowski-bench/Cargo.toml`:
```toml
[[bench]]
name = "changeset_micro"
harness = false
```

**Step 2: Write the microbenchmark file**

The following benchmarks isolate individual costs. Include the ones relevant
to profiling findings.

```rust
use criterion::{Criterion, criterion_group, criterion_main};
use minkowski::{EnumChangeSet, World};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

const N: usize = 10_000;

/// Measure: build a changeset with 10K insert mutations (arena alloc + vec push).
/// This isolates the recording cost from the apply cost.
fn bench_record_inserts(c: &mut Criterion) {
    let mut world = World::new();
    // Spawn entities first so they exist for insert
    let entities: Vec<_> = (0..N)
        .map(|i| {
            world.spawn((Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },))
        })
        .collect();
    let vel_id = world.register_component::<Velocity>();

    c.bench_function("changeset/record_10k_inserts", |b| {
        b.iter(|| {
            let mut cs = EnumChangeSet::new();
            for &entity in &entities {
                cs.insert_raw::<Velocity>(
                    entity,
                    vel_id,
                    Velocity {
                        dx: 1.0,
                        dy: 0.0,
                        dz: 0.0,
                    },
                );
            }
            cs
        });
    });
}

/// Measure: apply a pre-built changeset of 10K insert-overwrites.
/// Isolates the apply loop cost (entity lookup + memcpy + tick mark).
fn bench_apply_overwrites(c: &mut Criterion) {
    let mut world = World::new();
    for i in 0..N {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
            Velocity {
                dx: 1.0,
                dy: 0.0,
                dz: 0.0,
            },
        ));
    }

    c.bench_function("changeset/apply_10k_overwrites", |b| {
        b.iter_batched(
            || {
                // Setup: record 10K overwrites (Position already exists on each entity)
                let mut cs = EnumChangeSet::new();
                // Query current entities to get their IDs
                let mut targets = Vec::with_capacity(N);
                world
                    .query::<(minkowski::Entity,)>()
                    .for_each(|(entity,)| {
                        targets.push(entity);
                    });
                let pos_id = world.register_component::<Position>();
                for entity in targets {
                    cs.insert_raw::<Position>(
                        entity,
                        pos_id,
                        Position {
                            x: 99.0,
                            y: 99.0,
                            z: 99.0,
                        },
                    );
                }
                cs
            },
            |cs| {
                cs.apply(&mut world).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Measure: EnumChangeSet new + drop cycle (allocation overhead per transaction).
fn bench_changeset_lifecycle(c: &mut Criterion) {
    c.bench_function("changeset/new_drop_empty", |b| {
        b.iter(|| {
            let cs = EnumChangeSet::new();
            std::hint::black_box(cs);
        });
    });
}

/// Measure: full QueryWriter round-trip WITHOUT the transact() wrapper.
/// This would require internal access, so instead we measure the equivalent:
/// record N inserts + apply, which is what QueryWriter does minus archetype scanning.
fn bench_record_and_apply(c: &mut Criterion) {
    let mut world = World::new();
    for i in 0..N {
        world.spawn((
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
            Velocity {
                dx: 1.0,
                dy: 0.0,
                dz: 0.0,
            },
        ));
    }
    let pos_id = world.register_component::<Position>();

    c.bench_function("changeset/record_apply_10k", |b| {
        b.iter(|| {
            let mut cs = EnumChangeSet::new();
            let mut targets = Vec::with_capacity(N);
            world
                .query::<(minkowski::Entity, &Position, &Velocity)>()
                .for_each(|(entity, pos, vel)| {
                    targets.push((
                        entity,
                        Position {
                            x: pos.x + vel.dx,
                            y: pos.y + vel.dy,
                            z: pos.z + vel.dz,
                        },
                    ));
                });
            for (entity, pos) in targets {
                cs.insert_raw::<Position>(entity, pos_id, pos);
            }
            cs.apply(&mut world).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_record_inserts,
    bench_apply_overwrites,
    bench_changeset_lifecycle,
    bench_record_and_apply,
);
criterion_main!(benches);
```

**Step 3: Run the microbenchmarks**

Run: `cargo bench -p minkowski-bench -- changeset`

Expected: all four benchmarks report timings. Compare:
- `record_10k_inserts` — how much time is just recording?
- `apply_10k_overwrites` — how much time is just applying?
- `record_apply_10k` — sum vs parts shows overhead of query + collect pattern
- `new_drop_empty` — per-transaction allocation cost

**Step 4: Commit**

```bash
git add crates/minkowski-bench/benches/changeset_micro.rs crates/minkowski-bench/Cargo.toml
git commit -m "Add changeset microbenchmarks for profiling investigation"
```

---

### Task 4: Analyze and document results

**Files:**
- Create: `docs/plans/2026-03-11-changeset-profiling-findings.md`

**Step 1: Compile all measurements**

Gather:
1. Flamegraph cost breakdown from Task 2
2. Microbenchmark numbers from Task 3
3. Existing reducer benchmark numbers for context

**Step 2: Write findings document**

Structure:
```markdown
# EnumChangeSet Profiling Findings

## Summary
[One paragraph: where does the 14x overhead live?]

## Breakdown

### Recording phase (Arena + Vec<Mutation>)
- Time: X µs for 10K inserts
- % of total QueryWriter time: Y%
- Dominant cost: [arena alloc / vec push / alignment / ...]

### Apply phase (entity lookup + memcpy + tick)
- Time: X µs for 10K overwrites
- % of total QueryWriter time: Y%
- Dominant cost: [entity_locations lookup / BlobVec::get_ptr_mut / ...]

### Transaction overhead (begin + commit + tick validation)
- Time: X µs per call
- % of total QueryWriter time: Y%

### Per-transaction fixed cost (EnumChangeSet new/drop)
- Time: X ns per cycle

## Optimization Candidates

### [Candidate 1]: [name]
- Current cost: X µs
- Theoretical minimum: Y µs
- Approach: [description]
- Expected improvement: Z%

### [Candidate 2]: [name]
...

## Recommendation
[Which optimizations to pursue in Phase 3, in what order]
```

**Step 3: Commit**

```bash
git add docs/plans/2026-03-11-changeset-profiling-findings.md
git commit -m "Document changeset profiling findings"
```

---

## Notes

- **Task 2 is interactive**: the operator must look at the flamegraph and interpret
  it. The profiling binary from Task 1 makes this possible, but human judgment
  determines which microbenchmarks to prioritize in Task 3.

- **Task 3 benchmarks may need adjustment**: the exact benchmarks depend on what
  Task 2 reveals. The plan includes the most likely candidates. If profiling shows
  an unexpected bottleneck (e.g., `Optimistic::begin()` tick capture, or the
  `needs_drop` check), add a benchmark for that instead.

- **`insert_raw` is `pub(crate)`**: the microbenchmarks live in `minkowski-bench`
  (external crate) and cannot call `insert_raw` directly. Use the public
  `cs.insert::<T>(&mut world, entity, value)` API instead, which adds one
  `register_component` call but is functionally equivalent for profiling. If the
  registration cost is significant, pre-register and note the delta.

- **Position is `Copy`**: no DropEntry overhead for Position inserts. To benchmark
  DropEntry cost separately, use a non-Copy type (e.g., `String` or a wrapper with
  a custom Drop).

- **Mutation enum is 48 bytes**: 10K mutations = 480KB. This fits in L2 cache but
  not L1. If apply() shows cache misses, consider whether a more compact mutation
  representation would help.
