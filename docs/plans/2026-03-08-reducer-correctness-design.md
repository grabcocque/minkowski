# Reducer Correctness Guidance — Design

**Goal:** Document four behavioral properties that well-written reducers should satisfy, with determinism as the primary focus.

**Approach:** Documentation + doc comments. No API changes, no new types, no runtime checks.

---

## Motivation

Reducers are user-supplied closures that mutate world state. The type system constrains *what* they can access (via typed handles and Access bitsets), but it cannot constrain *how* they compute. Four behavioral properties matter for correctness, and all four are beyond Rust's type system today:

| Property | Risk if violated | Detectability |
|---|---|---|
| Determinism | Silent replication divergence, irreproducible bugs | Silent |
| No unwinding | Transaction abort, scheduler disruption | Loud (panic) |
| Termination | Deadlock (holds locks/transactions), scheduler stall | Loud (hang) |
| No host APIs | Blocks scheduler, breaks determinism, holds locks | Varies |

Determinism is the most dangerous because it's the quiet one — a non-deterministic reducer produces correct-looking results that diverge across replicas or replay runs.

## Deliverables

### 1. `docs/reducer-correctness.md`

User-facing guide with four sections.

**Determinism (primary):**
- A deterministic reducer's output is a pure function of its handle state and args. Same world state + same args = same mutations.
- What breaks determinism: RNG, `SystemTime`/`Instant`, `HashMap` iteration order, file/network I/O, global mutable state (`static mut`, thread-locals), pointer-derived values, `thread::current().id()`.
- What's already safe: all handle operations (`EntityMut::get`, `QueryWriter::for_each`, `Spawner::reserve`, `DynamicCtx::read`), `BTreeMap`/`BTreeSet` iteration, args.
- Iteration order: currently deterministic (archetypes in creation order, rows in insertion order) given identical world state. Not a stability guarantee.
- Replication: non-deterministic reducers break WAL replay on replicas. If using `Durable<S>`, reducers must be deterministic or replication diverges silently.

**No unwinding:**
- Transactional reducers (EntityMut, Spawner, QueryWriter, DynamicCtx): panic-safe. `Tx::Drop` reclaims entity IDs via `OrphanQueue`, changeset is discarded, world state is untouched.
- Scheduled reducers (QueryMut, QueryRef): panic with `&mut World` held. World remains consistent (no partial column writes), but the panic propagates to the caller.
- Guidance: prefer transactional reducers when panic risk is non-trivial.

**Termination:**
- Reducers should complete in bounded time.
- Under `Pessimistic`, the reducer holds column locks for its entire execution — non-termination deadlocks the system.
- Under `Optimistic`, long-running reducers increase conflict window and retry probability.
- Guidance: avoid unbounded loops. If iteration is needed, use the handle's `for_each` (bounded by entity count).

**No host APIs:**
- Overlaps with determinism — I/O, network, system clock break both.
- Additional concern: blocking I/O holds locks/transactions open, starving other reducers.
- Guidance: perform I/O outside reducers, pass results in via args. Read files before dispatch, write results after commit.

### 2. CLAUDE.md addition

New bullet under Key Conventions:

> **Reducer determinism rule**: reducers must be pure functions of their handle state and args. No RNG, system time, HashMap iteration, I/O, or global mutable state. When writing example reducers, use deterministic alternatives (BTreeMap, args-provided seeds, pre-computed values).

### 3. Doc comments on iteration methods

Add to `QueryWriter::for_each`, `QueryMut::for_each`, `QueryRef::for_each`:

> Iteration visits archetypes in creation order and rows within each archetype in insertion order. This is deterministic given identical world state but is not a stability guarantee — do not rely on specific ordering across engine versions.

## Non-goals

- No marker traits (`PureReducer`, `Infallible<F>`). Cost exceeds value at current scale.
- No runtime checks (e.g., intercepting `SystemTime::now()` calls).
- No `catch_unwind` wrappers. Panic propagation is intentional — reducer bugs must be visible.
- No WASM sandboxing. Reducers run in-process with full Rust capabilities.

## Scope

This is guidance, not enforcement. The doc helps users write correct reducers. The CLAUDE.md rule ensures AI-written reducers follow the same standard. The doc comments make iteration order behavior discoverable at the point of use.
