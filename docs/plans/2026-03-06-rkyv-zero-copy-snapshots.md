# rkyv Zero-Copy Snapshots Design

## Problem

Snapshot load deserializes every component value through bincode (`deserialize` → typed value → `ptr::copy_nonoverlapping` → BlobVec), allocating a `Vec<u8>` per component per entity. For a world with 100K entities and 5 components each, that's 500K allocations on load. WAL replay has the same cost per record.

rkyv enables zero-copy deserialization: the archived representation *is* the in-memory representation, so loading a snapshot can mmap the file and read component data directly — no per-value allocation, no intermediate typed values. This matters most for snapshot load (bulk) and less for WAL append (incremental, already fast).

**Why now:** This is the last planned feature before the roadmap shifts to post-1.0 stretch goals (replication & sync). The `WireFormat` trait was designed with rkyv as the intended second implementation, but the simpler path is to replace bincode entirely rather than maintain two serialization backends. Doing it now means replication can build on the faster format from the start.

## Current State

### Serialization architecture (all in `crates/minkowski-persist/`)

| File | Role | serde/bincode coupling |
|---|---|---|
| `format.rs` | `WireFormat` trait + `Bincode` impl | Trait is format-agnostic. `Bincode` impl calls `bincode::serialize/deserialize`. |
| `record.rs` | `WalRecord`, `SnapshotData`, etc. | All types derive `serde::{Serialize, Deserialize}`. |
| `codec.rs` | `CodecRegistry` — type-erased per-component serialization | **Deeply coupled.** `register<T>` captures `bincode::serialize`/`bincode::deserialize` closures. `insert_sparse_fn` calls `bincode::deserialize` directly. Trait bound is `T: Serialize + DeserializeOwned`. |
| `wal.rs` | `Wal<W: WireFormat>` — append-only log | Uses `WireFormat` for record-level ser/de. Component-level ser/de goes through `CodecRegistry`. |
| `snapshot.rs` | `Snapshot<W: WireFormat>` — full world save/load | Same split: `WireFormat` for the envelope, `CodecRegistry` for component bytes. |
| `durable.rs` | `Durable<S, W>` — WAL wrapper for transactions | Delegates to `Wal::append`, no direct ser/de. |

### Key observations

1. **Two-level serialization.** Component values are serialized individually through `CodecRegistry` into `Vec<u8>` blobs. These blobs are then embedded in record/snapshot structs that are serialized as a whole through `WireFormat`. rkyv handles both levels.

2. **BlobVec is 64-byte aligned.** Columns allocate with `max(component_align, 64)`. rkyv archived types must be aligned to at most the component's natural alignment — the 64-byte column alignment is separate from the component layout and is not a constraint on serialization format.

3. **Component data is opaque bytes.** The codec converts `T` → `Vec<u8>` (serialize) and `&[u8]` → `Vec<u8>` (deserialize, returning raw memory bytes). The snapshot/WAL never sees typed component data — only `(ComponentId, Vec<u8>)` pairs.

4. **No external users.** The persist crate is pre-1.0 with no backward-compatibility contract. The only consumers are the persist example and the persist crate's own tests. This makes a full replacement feasible.

### Verified-before-design: what exists vs what needs creation

**Exists:**
- `WireFormat` trait with `serialize_record`/`deserialize_record`/`serialize_snapshot`/`deserialize_snapshot` — `format.rs:4-11`
- `CodecRegistry::register<T: Component + Serialize + DeserializeOwned>` — `codec.rs:69`
- `CodecRegistry::serialize(id, ptr, &mut Vec<u8>)` / `deserialize(id, &[u8]) -> Vec<u8>` — `codec.rs:142-162`
- All record types (`WalRecord`, `SnapshotData`, etc.) with `#[derive(Serialize, Deserialize)]` — `record.rs`
- `BlobVec::MIN_COLUMN_ALIGN = 64` — `blob_vec.rs:25`
- `World::archetype_column_ptr(arch_idx, comp_id, row)` — `world.rs:1030`

**Needs creation:**
- rkyv derive annotations replacing serde on all record types
- `CodecRegistry` rewritten with rkyv bounds instead of serde bounds
- Zero-copy snapshot load path via mmap + archived access

**Removed (no longer needed):**
- `WireFormat` trait — only one format, no need for the abstraction
- `Bincode` struct — removed
- `<W: WireFormat>` type parameter on `Wal`, `Snapshot`, `Durable` — removed

## Proposed Design

### Guiding principle: one format, no abstraction overhead

Replace bincode with rkyv as the sole serialization backend. Remove the `WireFormat` trait and the `<W>` type parameter from `Wal`, `Snapshot`, and `Durable`. The persist crate has no external users and no backward-compatibility contract — this is the time to make the switch clean.

### Why not keep bincode alongside rkyv?

The dual-format approach (bincode default, rkyv opt-in behind a feature flag) was initially considered but rejected after analysis:

- **No use cases become impossible.** rkyv handles all types that bincode handles — `String`, `Vec`, `HashMap`, custom structs. The derives are different syntax (`Archive + rkyv::Serialize + rkyv::Deserialize` vs `serde::Serialize + serde::Deserialize`) but equivalent capability.
- **rkyv complexity is internal.** Users never see `rancor` errors — the persist crate wraps everything in `CodecError`/`WalError`/`SnapshotError`.
- **Dual-format means dual maintenance.** Two serialization paths, a `CodecFormat` trait, `CodecSupport<T>` bounds, default type parameters, feature flags, conditional compilation — all to support a bincode path that nobody needs.
- **The `WireFormat` trait loses its reason to exist.** It was designed for "bincode now, rkyv later." Now it's later. One format means the trait is pure indirection.

### Approach: full replacement

#### 1. Record types switch from serde to rkyv

```rust
// record.rs
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub enum SerializedMutation {
    Spawn {
        entity: u64,
        components: Vec<(u32, Vec<u8>)>,  // ComponentId is u32
    },
    Despawn { entity: u64 },
    Insert {
        entity: u64,
        component_id: u32,
        data: Vec<u8>,
    },
    Remove {
        entity: u64,
        component_id: u32,
    },
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct WalRecord {
    pub seq: u64,
    pub mutations: Vec<SerializedMutation>,
}

// ... same for SnapshotData, ArchetypeData, ColumnData, etc.
```

The `Vec<u8>` fields in `SerializedMutation::Insert { data: Vec<u8> }` archive as `ArchivedVec<u8>` — a pointer+length into the archive buffer. True zero-copy for the component bytes.

#### 2. Remove `WireFormat` trait, inline rkyv calls

```rust
// format.rs — becomes a thin module with serialize/deserialize helpers

use crate::record::{SnapshotData, WalRecord};

#[derive(Debug)]
pub struct FormatError(pub String);

impl std::fmt::Display for FormatError { ... }
impl std::error::Error for FormatError {}

pub fn serialize_record(record: &WalRecord) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(record)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn deserialize_record(bytes: &[u8]) -> Result<WalRecord, FormatError> {
    rkyv::from_bytes::<WalRecord, rkyv::rancor::Error>(bytes)
        .map_err(|e| FormatError(e.to_string()))
}

pub fn serialize_snapshot(snapshot: &SnapshotData) -> Result<Vec<u8>, FormatError> {
    rkyv::to_bytes::<rkyv::rancor::Error>(snapshot)
        .map(|v| v.to_vec())
        .map_err(|e| FormatError(e.to_string()))
}

pub fn deserialize_snapshot(bytes: &[u8]) -> Result<SnapshotData, FormatError> {
    rkyv::from_bytes::<SnapshotData, rkyv::rancor::Error>(bytes)
        .map_err(|e| FormatError(e.to_string()))
}
```

#### 3. `Wal`, `Snapshot`, `Durable` lose their type parameter

```rust
// Before: pub struct Wal<W: WireFormat> { format: W, ... }
// After:
pub struct Wal {
    file: File,
    next_seq: u64,
}

// Before: pub struct Snapshot<W: WireFormat> { format: W }
// After:
pub struct Snapshot;

// Before: pub struct Durable<S: Transact, W: WireFormat> { inner: S, wal: Mutex<Wal<W>>, ... }
// After:
pub struct Durable<S: Transact> {
    inner: S,
    wal: Mutex<Wal>,
    codecs: CodecRegistry,
}
```

Construction simplifies accordingly:
```rust
// Before: Wal::create(&path, Bincode)
// After:
Wal::create(&path)

// Before: Snapshot::new(Bincode)
// After:
Snapshot::new()  // or just Snapshot (unit struct)
```

#### 4. CodecRegistry switches to rkyv bounds

```rust
// codec.rs
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

impl CodecRegistry {
    pub fn register<T>(&mut self, world: &mut World)
    where
        T: Component + Archive,
        T::Archived: RkyvDeserialize<T, rkyv::de::Pool>,
        for<'a> T: RkyvSerialize<rkyv::ser::Serializer<
            rkyv::util::AlignedVec,
            rkyv::ser::allocator::ArenaHandle<'a>,
            rkyv::ser::sharing::Share,
        >>,
    {
        let comp_id = world.register_component::<T>();
        // Capture rkyv closures instead of bincode closures
        let serialize_fn: SerializeFn = |ptr, out| {
            let value = unsafe { &*ptr.cast::<T>() };
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(value)
                .map_err(|e| CodecError::Serialize(e.to_string()))?;
            out.extend_from_slice(&bytes);
            Ok(())
        };
        let deserialize_fn: DeserializeFn = |bytes| {
            let value: T = rkyv::from_bytes::<T, rkyv::rancor::Error>(bytes)
                .map_err(|e| CodecError::Deserialize(e.to_string()))?;
            let layout = Layout::new::<T>();
            let mut buf = vec![0u8; layout.size()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &value as *const T as *const u8,
                    buf.as_mut_ptr(),
                    layout.size(),
                );
            }
            std::mem::forget(value);
            Ok(buf)
        };
        // ... same pattern for insert_sparse_fn
    }
}
```

**Note on rkyv bounds verbosity.** The `where` clause on `register` is noisy because rkyv 0.8's serializer is parameterized on allocator/sharing strategy. In practice, users never write this — they derive `Archive + rkyv::Serialize + rkyv::Deserialize` and the compiler infers the bounds. We can also define a type alias:

```rust
pub trait Persistable: Component + Archive
where
    Self::Archived: RkyvDeserialize<Self, rkyv::de::Pool>,
{ /* blanket or manual impls */ }
```

This would let the public API read `register<T: Persistable>` — clean and self-documenting.

#### 5. Zero-copy snapshot load

```rust
impl Snapshot {
    /// Load a snapshot with zero-copy component data.
    ///
    /// mmaps the snapshot file. Archived component bytes are copied directly
    /// into BlobVec columns without intermediate typed deserialization.
    /// The mmap is dropped after load completes.
    pub fn load_zero_copy(
        path: &Path,
        codecs: &CodecRegistry,
    ) -> Result<(World, u64), SnapshotError> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Skip the 8-byte length prefix
        let archived = rkyv::access::<ArchivedSnapshotData, rkyv::rancor::Error>(&mmap[8..])?;

        let mut world = World::new();
        // Register component types from archived schema...
        // For each archived archetype:
        //   For each column:
        //     For each row:
        //       archived_value.as_slice() -> raw bytes
        //       -> BlobVec::push(raw_bytes)  // direct copy, no deserialization
        // Restore allocator state...
        Ok((world, archived.wal_seq.to_native()))
    }
}
```

**Critical insight:** For rkyv, the serialized bytes of a `#[repr(C)]` component type *are* the in-memory bytes (assuming matching endianness). So `archived_column.values[row].as_slice()` gives exactly the bytes that should go into the BlobVec — no deserialization step. This is where the 500K-allocation saving lives.

**Alignment caveat:** rkyv aligns archived values to their natural alignment within the archive buffer, not to the BlobVec's 64-byte column alignment. This is fine — `BlobVec::push` handles column alignment. We're copying bytes *into* BlobVec, not mapping BlobVec *onto* the archive.

**Non-repr(C) components:** rkyv still works — the archived representation may differ from Rust's layout, so the zero-copy path falls back to `rkyv::from_bytes` per value. The `load_zero_copy` method can detect this: if `size_of::<T::Archived>() == size_of::<T>()` and the type is `repr(C)`, use direct copy; otherwise deserialize normally. This check can be deferred to a future optimization — the initial implementation can always go through `from_bytes` per value, which is already faster than bincode.

### API Surface

**Removed public types:**
- `WireFormat` trait
- `Bincode` struct

**Changed public types:**
- `Wal` — no longer generic (`Wal` instead of `Wal<W>`)
- `Snapshot` — no longer generic (`Snapshot` instead of `Snapshot<W>`)
- `Durable<S>` — one fewer type parameter (`Durable<S>` instead of `Durable<S, W>`)
- `CodecRegistry::register` bound — `T: Persistable` instead of `T: Serialize + DeserializeOwned`

**New public types/items:**
- `Persistable` — trait alias for rkyv bounds on persistent components
- `Snapshot::load_zero_copy` — mmap-based bulk load
- `FormatError` — replaces `WireFormat::Error`

**Unchanged:**
- `CodecRegistry` struct shape (still `HashMap<ComponentId, ComponentCodec>`)
- `CodecRegistry::serialize` / `deserialize` signatures (still `ptr -> Vec<u8>` / `&[u8] -> Vec<u8>`)
- `Snapshot::save` / `Snapshot::load` — same signatures, rkyv under the hood
- `Wal::append` / `Wal::replay` — same signatures
- All core ECS types (`World`, `BlobVec`, etc.) — **zero changes to the core crate**

### Internal Architecture

```
User component: #[derive(Archive, rkyv::Serialize, rkyv::Deserialize)]

CodecRegistry
  register::<T>() captures rkyv::to_bytes / rkyv::from_bytes closures
  serialize(id, ptr) -> Vec<u8>  (rkyv bytes)
  deserialize(id, &[u8]) -> Vec<u8>  (raw memory bytes)

Snapshot
  save() -> CodecRegistry serializes each component value via rkyv
         -> rkyv::to_bytes serializes the SnapshotData envelope
         -> write to file

  load() -> read file
         -> rkyv::from_bytes deserializes SnapshotData (owned, allocating)
         -> CodecRegistry deserializes each component value
         -> standard restore_world path

  load_zero_copy() -> mmap file
                   -> rkyv::access gives &ArchivedSnapshotData (zero-copy)
                   -> copy archived component bytes directly to BlobVec
                   -> drop mmap
```

### Data Flow

**Snapshot save:**
```
World -> for each archetype, for each entity, for each component:
  archetype_column_ptr(arch, comp, row)
    -> CodecRegistry::serialize(comp_id, ptr) -> Vec<u8>
      -> embedded in SnapshotData::archetypes[i].columns[j].values[row]
  -> rkyv::to_bytes(data) -> Vec<u8>
    -> write [len: u64 LE][payload] to file
```

**Snapshot load (allocating, standard path):**
```
File -> read bytes -> rkyv::from_bytes -> owned SnapshotData
  -> register component types from schema
  -> for each archetype:
      for each entity: alloc_entity()
      for each column, for each row:
        CodecRegistry::deserialize(comp_id, &bytes) -> Vec<u8>
        -> record_spawn via EnumChangeSet
  -> changeset.apply(&mut world)
  -> restore_allocator_state
```

**Snapshot load_zero_copy:**
```
File -> mmap -> rkyv::access -> &ArchivedSnapshotData (zero alloc)
  -> register component types from schema
  -> for each archived archetype:
      alloc_entity() for each entity
      for each column:
        for each row:
          archived_value.as_slice() -> raw bytes
          -> BlobVec::push(raw_bytes)  // direct copy, no deserialization
  -> restore_allocator_state
  -> drop mmap
```

### Dependencies

```toml
# crates/minkowski-persist/Cargo.toml
[dependencies]
minkowski = { path = "../minkowski" }
rkyv = { version = "0.8", features = ["alloc"] }
memmap2 = "0.9"
parking_lot = "0.12"
fixedbitset = "0.5"

# Removed: serde, bincode
```

Clean dependency swap. `rkyv` replaces `serde` + `bincode`. `memmap2` is new (for zero-copy load). No feature flags — rkyv is the only path.

### Component requirements

Persistent component types must derive rkyv traits:

```rust
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, Copy, Archive, Serialize, Deserialize, PartialEq, Debug)]
struct Pos { x: f32, y: f32 }
```

For the zero-copy load path to give maximum benefit, components should also be `#[repr(C)]` so that the archived representation matches the in-memory layout byte-for-byte. Without `#[repr(C)]`, rkyv still works correctly — the archived representation may differ, requiring a per-value deserialization on load instead of a direct byte copy.

**Practical impact:** Most ECS components are small, `Copy`, POD structs (`Pos`, `Vel`, `Health`). Adding `#[derive(Archive, Serialize, Deserialize)]` is the same effort as the current `#[derive(serde::Serialize, serde::Deserialize)]`. Components with heap fields (`String`, `Vec`) work fine — rkyv's `ArchivedString`/`ArchivedVec` handle them, just without zero-copy for those specific fields.

## Alternatives Considered

### Alternative A: Dual-format (bincode default, rkyv opt-in)

**Approach:** Keep bincode as default, add rkyv behind a feature flag. `WireFormat` trait stays, `CodecRegistry` gains a `CodecFormat` type parameter with default `BincodeCodec`.

**Pros:**
- No changes for users who don't want rkyv
- Gradual migration path

**Cons:**
- Two serialization paths to maintain and test
- `CodecFormat` trait + `CodecSupport<T>` bounds + default type parameter + feature flag = significant complexity
- The `WireFormat` trait becomes pure indirection for users who do switch
- No existing users to protect — the backward compatibility is protecting nobody

**Rejected because:** The persist crate is pre-1.0 with no external consumers. The dual-format complexity is engineering debt with no payoff. One clean format is simpler to maintain, test, and document than two formats with an abstraction layer between them.

### Alternative B: Zero-copy via mmap-backed BlobVec (in-place operation)

**Approach:** Instead of copying archived bytes into BlobVec, make BlobVec's backing memory *be* the mmap.

**Pros:**
- True zero-copy — not even a memcpy on load
- Minimal memory footprint (shared pages with OS page cache)

**Cons:**
- BlobVec must be 64-byte aligned, archive data may not be
- BlobVec needs to grow (push/remove) — can't grow an mmap region in place
- Mutation invalidates mmap pages (copy-on-write), losing the sharing benefit
- Column `changed_tick` is not in the archive — needs separate tracking
- Fundamentally changes BlobVec's ownership model (allocated vs mapped)

**Rejected because:** The complexity is enormous for marginal benefit over "copy archived bytes into BlobVec." The copy is a single memcpy per column per archetype — fast enough. Could be revisited if profiling shows snapshot load is still a bottleneck.

### Alternative C: Keep serde trait bounds, use rkyv-serde bridge

**Approach:** Use `rkyv`'s serde compatibility layer so components keep `#[derive(serde::Serialize, serde::Deserialize)]` and rkyv wraps them.

**Pros:**
- Zero changes to user component definitions
- Familiar serde ecosystem

**Cons:**
- Loses the zero-copy benefit entirely — the serde bridge serializes through serde, not rkyv's native format
- Adds complexity (rkyv + serde + bridge layer) for no performance gain
- Defeats the purpose of the migration

**Rejected because:** If we're not getting zero-copy, there's no reason to switch from bincode at all. The serde bridge is the worst of both worlds.

## Semantic Review

### 1. Can this be called with the wrong World?

No new World-crossing risk. `Snapshot::load` and `load_zero_copy` create a fresh World internally. `CodecRegistry` stores ComponentIds — schema-based restore ensures IDs match. Same as current design.

### 2. Can Drop observe inconsistent state?

The mmap in `load_zero_copy` is a local variable dropped at the end of the method. All component bytes are copied into BlobVec columns before the mmap drops. No reference to the mmap escapes. If the function panics mid-restore, the partially-constructed World drops normally — BlobVec drop functions handle cleanup. No new Drop hazard.

### 3. Can two threads reach this through `&self`?

`Snapshot::load_zero_copy` creates a fresh mmap and World internally — no shared mutable state. `CodecRegistry` is not Sync (contains HashMap), same as before. `Durable`'s `Wal` is behind `Mutex`, unchanged. No new concurrency concern.

### 4. Does dedup/merge/collapse preserve the strongest invariant?

No dedup/merge in this design. The migration is a clean swap — bincode code is removed, rkyv code takes its place. No coexistence means no invariant to preserve between formats.

### 5. What happens if this is abandoned halfway through?

- **load_zero_copy panics mid-restore:** mmap dropped (unmapped), partially-constructed World dropped (BlobVec cleanup runs). No leaked resources.
- **Implementation abandoned partway:** Steps are ordered so each one leaves the crate in a compilable, testable state. Step 1 (record types) is independent of step 2 (format module), etc.

### 6. Can a type bound be violated by a legal generic instantiation?

The `Persistable` trait (or raw rkyv bounds on `register`) ensures only types with proper rkyv derives can be registered. A type without `Archive` can't be registered — caught at compile time. The bounds are structural, not behavioral.

### 7. Does the API surface permit any operation not covered by the Access bitset?

This design doesn't touch Access/reducer/transaction. Snapshot and Wal operate on `&World` / `&mut World` directly — outside the scheduler's jurisdiction. No new bypass paths.

## Implementation Plan

### Step 1: Swap dependencies
**Files:** `crates/minkowski-persist/Cargo.toml`
- Replace `serde = { version = "1", features = ["derive"] }` and `bincode = "1"` with `rkyv = { version = "0.8", features = ["alloc"] }` and `memmap2 = "0.9"`
- `cargo check -p minkowski-persist` will fail — expected, drives the rest of the steps

### Step 2: Migrate record types from serde to rkyv
**Files:** `crates/minkowski-persist/src/record.rs`
- Replace `use serde::{Deserialize, Serialize}` with `use rkyv::{Archive, Deserialize, Serialize}`
- Replace `#[derive(Serialize, Deserialize, ...)]` with `#[derive(Archive, Serialize, Deserialize, ...)]` on all types
- Note: `ComponentId` is `pub type ComponentId = u32` — rkyv handles primitive type aliases natively

### Step 3: Replace WireFormat trait with direct rkyv functions
**Files:** `crates/minkowski-persist/src/format.rs`
- Remove `WireFormat` trait and `Bincode` struct
- Add `FormatError` wrapper and free functions: `serialize_record`, `deserialize_record`, `serialize_snapshot`, `deserialize_snapshot`
- Add round-trip tests using the new functions

### Step 4: Rewrite CodecRegistry with rkyv bounds
**Files:** `crates/minkowski-persist/src/codec.rs`
- Replace `use serde::{de::DeserializeOwned, Serialize}` with rkyv imports
- Change `register<T>` bound from `T: Component + Serialize + DeserializeOwned` to rkyv bounds (or `T: Persistable`)
- Replace `bincode::serialize`/`bincode::deserialize` in captured closures with `rkyv::to_bytes`/`rkyv::from_bytes`
- Replace `bincode::deserialize` in `insert_sparse_fn` with rkyv equivalent
- Update all tests (change `#[derive(Serialize, Deserialize)]` to `#[derive(Archive, rkyv::Serialize, rkyv::Deserialize)]` on test types)

### Step 5: Simplify Wal, Snapshot, Durable — remove type parameter
**Files:** `crates/minkowski-persist/src/wal.rs`, `crates/minkowski-persist/src/snapshot.rs`, `crates/minkowski-persist/src/durable.rs`, `crates/minkowski-persist/src/lib.rs`
- Remove `<W: WireFormat>` from `Wal`, `Snapshot`, `Durable`
- Replace `self.format.serialize_record(...)` with `format::serialize_record(...)`
- Construction: `Wal::create(&path)` instead of `Wal::create(&path, Bincode)`
- Update all tests
- Update `lib.rs` re-exports (remove `Bincode`, `WireFormat`)
- Run `cargo test -p minkowski-persist` — all existing tests should pass with new syntax

### Step 6: Implement zero-copy snapshot load
**Files:** `crates/minkowski-persist/src/snapshot.rs`
- Add `Snapshot::load_zero_copy(path, codecs)` method
- Implement mmap → `rkyv::access` → direct BlobVec copy path
- Tests: save then `load_zero_copy`, verify component values match
- Tests: empty world, multiple archetypes, sparse components, non-persisted component gaps
- Test: compare `load` vs `load_zero_copy` results are identical

### Step 7: Update persist example and examples Cargo.toml
**Files:** `examples/examples/persist.rs`, `examples/Cargo.toml`
- Change component derives from serde to rkyv
- Remove `Bincode` usage, use simplified `Wal::create`/`Snapshot::new` API
- Add a `load_zero_copy` demonstration
- Verify: `cargo run -p minkowski-examples --example persist --release`

### Step 8: Documentation and cleanup
**Files:** `README.md`, `CLAUDE.md`, `docs/adr/013-rkyv-zero-copy-snapshots.md`
- Write ADR-013 documenting the decision to replace bincode entirely
- Update README roadmap (move rkyv from roadmap to features)
- Update CLAUDE.md: deps table (rkyv replaces serde/bincode), persist-related text
- Update `.claude/skills/minkowski-guide.md` persistence section
- Update `.claude/commands/minkowski/persist.md` if it references serde/bincode
