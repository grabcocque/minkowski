# WAL Checkpoint Markers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add checkpoint markers to the WAL that protect users from unbounded WAL growth via a callback-driven snapshot mechanism.

**Architecture:** New `WalEntry::Checkpoint` variant in the WAL stream. `Wal` tracks bytes since last checkpoint and exposes `checkpoint_needed()`. `CheckpointHandler` trait with `AutoCheckpoint` default impl. `Durable` orchestrates the callback after successful transact.

**Tech Stack:** Rust, rkyv (existing), parking_lot Mutex (existing in Durable)

---

### Task 1: WalEntry::Checkpoint variant

Add the new variant to the WAL entry enum so it can be serialized into the segment stream.

**Files:**
- Modify: `crates/minkowski-persist/src/record.rs:92-96`

**Step 1: Write the failing test**

Add to `record.rs` test module:

```rust
#[test]
fn wal_entry_checkpoint_variant() {
    let checkpoint = WalEntry::Checkpoint { snapshot_seq: 42 };
    assert!(matches!(
        checkpoint,
        WalEntry::Checkpoint { snapshot_seq: 42 }
    ));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- wal_entry_checkpoint_variant`
Expected: compilation error (variant doesn't exist)

**Step 3: Add the variant**

In `record.rs`, add `Checkpoint` to `WalEntry`:

```rust
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub enum WalEntry {
    Schema(WalSchema),
    Mutations(WalRecord),
    Checkpoint { snapshot_seq: u64 },
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p minkowski-persist -- wal_entry_checkpoint_variant`
Expected: PASS

**Step 5: Run all tests to check nothing broke**

Run: `cargo test -p minkowski-persist`
Expected: all pass — existing code uses `match` on `WalEntry` in several places (`wal.rs`, `replication.rs`). These will fail with "non-exhaustive patterns." Fix each match arm:

In `wal.rs` `replay_from()` (~line 347): add `WalEntry::Checkpoint { .. } => {}` (skip during replay).

In `wal.rs` `scan_active_segment()` (~line 473): add `WalEntry::Checkpoint { .. } => {}` (skip during scan — we'll handle checkpoint recovery in Task 4).

In `wal.rs` `open()` earlier-segment scan (~line 265): add `WalEntry::Checkpoint { .. } => {}`.

In `replication.rs` `WalCursor::open()` (~line 54): add `Some((WalEntry::Checkpoint { .. }, next_pos)) => { pos = next_pos; }` (skip, advance past).

In `replication.rs` `WalCursor::next_batch()` (~line 87): add `Some((WalEntry::Checkpoint { .. }, next_pos)) => { self.pos = next_pos; }` (skip, advance past).

**Step 6: Run all tests again**

Run: `cargo test -p minkowski-persist`
Expected: all pass

**Step 7: Commit**

```bash
git add crates/minkowski-persist/src/record.rs crates/minkowski-persist/src/wal.rs crates/minkowski-persist/src/replication.rs
git commit -m "feat(persist): add WalEntry::Checkpoint variant"
```

---

### Task 2: WalConfig + Wal checkpoint state

Add `max_bytes_between_checkpoints` to `WalConfig` and checkpoint tracking fields to `Wal`.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Step 1: Write the failing tests**

Add to `wal.rs` test module:

```rust
#[test]
fn wal_config_checkpoint_default_disabled() {
    let config = WalConfig::default();
    assert!(config.max_bytes_between_checkpoints.is_none());
}

#[test]
fn checkpoint_needed_when_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();
    assert!(!wal.checkpoint_needed());
    assert_eq!(wal.last_checkpoint_seq(), None);
}

#[test]
fn checkpoint_needed_after_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(128),
    };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    assert!(!wal.checkpoint_needed());

    // Write enough records to exceed 128 bytes
    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    assert!(wal.checkpoint_needed());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- checkpoint_needed wal_config_checkpoint`
Expected: compilation errors

**Step 3: Implement**

Update `WalConfig`:

```rust
#[derive(Debug, Clone)]
pub struct WalConfig {
    pub max_segment_bytes: usize,
    pub max_bytes_between_checkpoints: Option<usize>,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: None,
        }
    }
}
```

Add fields to `Wal` struct:

```rust
pub struct Wal {
    // ... existing fields ...
    last_checkpoint_seq: Option<u64>,
    bytes_since_checkpoint: u64,
}
```

Initialize both to `None` / `0` in `create()` and `open()`.

Update `append()` to increment `bytes_since_checkpoint` alongside `active_bytes`:

```rust
let frame_bytes = 4 + payload.len() as u64;
self.active_bytes += frame_bytes;
self.bytes_since_checkpoint += frame_bytes;
```

Add methods:

```rust
pub fn checkpoint_needed(&self) -> bool {
    match self.config.max_bytes_between_checkpoints {
        Some(max) => self.bytes_since_checkpoint >= max as u64,
        None => false,
    }
}

pub fn last_checkpoint_seq(&self) -> Option<u64> {
    self.last_checkpoint_seq
}
```

Update `small_config()` in tests to include the new field:

```rust
fn small_config() -> WalConfig {
    WalConfig {
        max_segment_bytes: 128,
        max_bytes_between_checkpoints: None,
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski-persist -- checkpoint_needed wal_config_checkpoint`
Expected: PASS

**Step 5: Run all tests**

Run: `cargo test -p minkowski-persist`
Expected: all pass

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "feat(persist): checkpoint state tracking in Wal"
```

---

### Task 3: acknowledge_snapshot

Implement `Wal::acknowledge_snapshot()` which writes a `WalEntry::Checkpoint` frame and resets the byte counter.

**Files:**
- Modify: `crates/minkowski-persist/src/wal.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn acknowledge_snapshot_writes_checkpoint_and_resets() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(128),
    };
    let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

    // Write enough to trigger checkpoint_needed
    for i in 0..10 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }
    assert!(wal.checkpoint_needed());

    let seq = wal.next_seq();
    wal.acknowledge_snapshot(seq).unwrap();

    assert_eq!(wal.last_checkpoint_seq(), Some(seq));
    assert!(!wal.checkpoint_needed());
}

#[test]
fn acknowledge_snapshot_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(1024),
    };

    {
        let mut wal = Wal::create(&wal_dir, &codecs, config.clone()).unwrap();
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: 1.0, y: 2.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);

        wal.acknowledge_snapshot(wal.next_seq()).unwrap();
    }

    let wal2 = Wal::open(&wal_dir, &codecs, config).unwrap();
    assert_eq!(wal2.last_checkpoint_seq(), Some(1));
    assert!(!wal2.checkpoint_needed());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- acknowledge_snapshot`
Expected: compilation error

**Step 3: Implement acknowledge_snapshot**

```rust
/// Record that a snapshot was taken at the given seq.
/// Writes a Checkpoint entry to the WAL stream and resets the byte counter.
pub fn acknowledge_snapshot(&mut self, seq: u64) -> Result<(), WalError> {
    let entry = WalEntry::Checkpoint { snapshot_seq: seq };
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
        .map_err(|e| WalError::Format(e.to_string()))?;

    {
        let mut writer = BufWriter::new(&self.active_file);
        let len: u32 = payload.len().try_into().map_err(|_| {
            WalError::Format("checkpoint entry too large".into())
        })?;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&payload)?;
        writer.flush()?;
    }

    self.active_bytes += 4 + payload.len() as u64;
    self.last_checkpoint_seq = Some(seq);
    self.bytes_since_checkpoint = 0;
    Ok(())
}
```

**Step 4: Update scan_active_segment to recover checkpoint state**

In `scan_active_segment()`, update the `Checkpoint` match arm:

```rust
WalEntry::Checkpoint { snapshot_seq } => {
    self.last_checkpoint_seq = Some(snapshot_seq);
    self.bytes_since_checkpoint = 0;
}
```

And after the loop, `bytes_since_checkpoint` will naturally be the bytes written after the last checkpoint because we reset it to 0 on each checkpoint and the subsequent mutation entries increment it... but wait, `scan_active_segment` doesn't currently call `append`, so `bytes_since_checkpoint` won't be incremented during the scan. We need to track bytes manually:

Update `scan_active_segment` to track bytes after the last checkpoint:

```rust
fn scan_active_segment(&mut self) -> Result<(u64, bool), WalError> {
    let mut last_seq = 0u64;
    let mut has_mutations = false;
    let mut pos: u64 = 0;
    let mut bytes_after_checkpoint: u64 = 0;

    while let Some((entry, next_pos)) = self.read_next_entry(pos)? {
        let frame_bytes = next_pos - pos;
        match entry {
            WalEntry::Mutations(record) => {
                last_seq = record.seq;
                has_mutations = true;
                bytes_after_checkpoint += frame_bytes;
            }
            WalEntry::Checkpoint { snapshot_seq } => {
                self.last_checkpoint_seq = Some(snapshot_seq);
                bytes_after_checkpoint = 0;
            }
            WalEntry::Schema(_) => {}
        }
        pos = next_pos;
    }

    self.bytes_since_checkpoint = bytes_after_checkpoint;
    Ok((last_seq, has_mutations))
}
```

**Step 5: Run tests**

Run: `cargo test -p minkowski-persist -- acknowledge_snapshot`
Expected: PASS

**Step 6: Run all tests**

Run: `cargo test -p minkowski-persist`
Expected: all pass

**Step 7: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs
git commit -m "feat(persist): acknowledge_snapshot writes checkpoint + recovers on open"
```

---

### Task 4: WalCursor skips checkpoint entries + replay_from skips them

Verify that cursors and replay correctly skip `Checkpoint` entries. These match arms were added in Task 1, but we need a test that proves a WAL with checkpoint entries round-trips correctly.

**Files:**
- Modify: `crates/minkowski-persist/src/replication.rs` (test only)
- Modify: `crates/minkowski-persist/src/wal.rs` (test only)

**Step 1: Write the tests**

In `replication.rs` test module:

```rust
#[test]
fn cursor_skips_checkpoint_entries() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let mut wal = Wal::create(&wal_dir, &codecs, WalConfig::default()).unwrap();

    // Write 3 records, checkpoint, then 2 more
    for i in 0..3 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }
    wal.acknowledge_snapshot(wal.next_seq()).unwrap();
    for i in 3..5 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    drop(wal);

    let mut cursor = WalCursor::open(&wal_dir, 0).unwrap();
    let batch = cursor.next_batch(100).unwrap();
    // Should see all 5 mutation records, no checkpoint in batch
    assert_eq!(batch.records.len(), 5);
    assert_eq!(batch.records[0].seq, 0);
    assert_eq!(batch.records[4].seq, 4);
}
```

In `wal.rs` test module:

```rust
#[test]
fn replay_skips_checkpoint_entries() {
    let dir = tempfile::tempdir().unwrap();
    let wal_dir = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<Pos>("pos", &mut world);

    let mut wal = Wal::create(&wal_dir, &codecs, default_config()).unwrap();

    for i in 0..3 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }
    wal.acknowledge_snapshot(wal.next_seq()).unwrap();
    for i in 3..5 {
        let e = world.alloc_entity();
        let mut cs = EnumChangeSet::new();
        cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
        wal.append(&cs, &codecs).unwrap();
        cs.apply(&mut world);
    }

    let mut world2 = World::new();
    codecs.register_one(world.component_id::<Pos>().unwrap(), &mut world2);
    let last = wal.replay(&mut world2, &codecs).unwrap();
    assert_eq!(last, 4);
    assert_eq!(world2.query::<(&Pos,)>().count(), 5);
}
```

**Step 2: Run tests**

Run: `cargo test -p minkowski-persist -- replay_skips_checkpoint cursor_skips_checkpoint`
Expected: PASS (match arms already added in Task 1)

**Step 3: Commit**

```bash
git add crates/minkowski-persist/src/wal.rs crates/minkowski-persist/src/replication.rs
git commit -m "test(persist): verify checkpoint entries skipped in replay + cursor"
```

---

### Task 5: CheckpointHandler trait + AutoCheckpoint

Add the trait and default implementation.

**Files:**
- Create: `crates/minkowski-persist/src/checkpoint.rs`
- Modify: `crates/minkowski-persist/src/lib.rs`

**Step 1: Write the failing test**

In the new `checkpoint.rs` file, include an inline test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecRegistry;
    use crate::wal::{Wal, WalConfig};
    use minkowski::World;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Clone, Copy, Archive, Serialize, Deserialize)]
    #[repr(C)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[test]
    fn auto_checkpoint_creates_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let wal_dir = dir.path().join("test.wal");
        let snap_dir = dir.path().join("snaps");
        std::fs::create_dir_all(&snap_dir).unwrap();

        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<Pos>("pos", &mut world);

        let config = WalConfig {
            max_segment_bytes: 64 * 1024 * 1024,
            max_bytes_between_checkpoints: Some(128),
        };
        let mut wal = Wal::create(&wal_dir, &codecs, config).unwrap();

        for i in 0..10 {
            let e = world.alloc_entity();
            let mut cs = minkowski::EnumChangeSet::new();
            cs.spawn_bundle(&mut world, e, (Pos { x: i as f32, y: 0.0 },));
            wal.append(&cs, &codecs).unwrap();
            cs.apply(&mut world);
        }

        assert!(wal.checkpoint_needed());

        let mut handler = AutoCheckpoint::new(&snap_dir);
        handler.on_checkpoint_needed(&mut world, &mut wal, &codecs);

        assert!(!wal.checkpoint_needed());
        assert!(wal.last_checkpoint_seq().is_some());

        // Verify snapshot file was created
        let snaps: Vec<_> = std::fs::read_dir(&snap_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "snap").unwrap_or(false))
            .collect();
        assert_eq!(snaps.len(), 1);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p minkowski-persist -- auto_checkpoint_creates_snapshot`
Expected: compilation error (module doesn't exist)

**Step 3: Implement**

Create `crates/minkowski-persist/src/checkpoint.rs`:

```rust
use std::path::{Path, PathBuf};

use minkowski::World;

use crate::codec::CodecRegistry;
use crate::snapshot::Snapshot;
use crate::wal::Wal;

/// Callback invoked by [`Durable`](crate::Durable) when the WAL exceeds
/// `max_bytes_between_checkpoints` without a snapshot acknowledgment.
pub trait CheckpointHandler: Send {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    );
}

/// Default checkpoint handler: saves a snapshot and acknowledges it.
///
/// Snapshots are written to `snap_dir/checkpoint-{seq:06}.snap`.
pub struct AutoCheckpoint {
    snap_dir: PathBuf,
}

impl AutoCheckpoint {
    pub fn new(snap_dir: &Path) -> Self {
        Self {
            snap_dir: snap_dir.to_path_buf(),
        }
    }
}

impl CheckpointHandler for AutoCheckpoint {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    ) {
        let seq = wal.next_seq();
        let path = self.snap_dir.join(format!("checkpoint-{seq:06}.snap"));
        let snap = Snapshot::new();
        snap.save(&path, world, codecs, seq)
            .expect("checkpoint snapshot failed");
        wal.acknowledge_snapshot(seq)
            .expect("checkpoint acknowledge failed");
    }
}
```

Add to `lib.rs`:

```rust
pub mod checkpoint;
pub use checkpoint::{AutoCheckpoint, CheckpointHandler};
```

**Step 4: Run test**

Run: `cargo test -p minkowski-persist -- auto_checkpoint_creates_snapshot`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/minkowski-persist/src/checkpoint.rs crates/minkowski-persist/src/lib.rs
git commit -m "feat(persist): CheckpointHandler trait + AutoCheckpoint default"
```

---

### Task 6: Durable integration

Add optional checkpoint handler to `Durable` and fire it after successful transact.

**Files:**
- Modify: `crates/minkowski-persist/src/durable.rs`
- Modify: `crates/minkowski-persist/src/lib.rs` (if needed)

**Step 1: Write the failing test**

Add to `durable.rs` test module:

```rust
use crate::checkpoint::CheckpointHandler;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

struct CountingHandler {
    count: Arc<AtomicU32>,
}

impl CheckpointHandler for CountingHandler {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    ) {
        self.count.fetch_add(1, Ordering::SeqCst);
        wal.acknowledge_snapshot(wal.next_seq()).unwrap();
    }
}

#[test]
fn durable_fires_checkpoint_handler() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(64), // very small to trigger quickly
    };
    let wal = Wal::create(&wal_path, &codecs, config).unwrap();
    let strategy = Optimistic::new(&world);

    let count = Arc::new(AtomicU32::new(0));
    let handler = CountingHandler { count: count.clone() };
    let durable = Durable::with_checkpoint(strategy, wal, codecs, handler);

    let access = Access::of::<(&mut Pos,)>(&mut world);
    let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

    // Transact enough times to trigger checkpoint
    for _ in 0..20 {
        durable
            .transact(&mut world, &access, |tx, world| {
                tx.write::<Pos>(world, e, Pos { x: 10.0, y: 20.0 });
            })
            .unwrap();
    }

    assert!(count.load(Ordering::SeqCst) >= 1, "handler should have fired");
}

#[test]
fn durable_no_handler_backward_compat() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs.register::<Pos>(&mut world);

    let config = WalConfig {
        max_segment_bytes: 64 * 1024 * 1024,
        max_bytes_between_checkpoints: Some(64),
    };
    let wal = Wal::create(&wal_path, &codecs, config).unwrap();
    let strategy = Optimistic::new(&world);
    let durable = Durable::new(strategy, wal, codecs);

    let access = Access::of::<(&mut Pos,)>(&mut world);
    let e = world.spawn((Pos { x: 1.0, y: 2.0 },));

    // Should work fine without handler, even if threshold is exceeded
    for _ in 0..20 {
        durable
            .transact(&mut world, &access, |tx, world| {
                tx.write::<Pos>(world, e, Pos { x: 10.0, y: 20.0 });
            })
            .unwrap();
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-persist -- durable_fires_checkpoint durable_no_handler`
Expected: compilation error (with_checkpoint doesn't exist)

**Step 3: Implement**

Update `Durable` struct:

```rust
use crate::checkpoint::CheckpointHandler;

pub struct Durable<S: Transact> {
    inner: S,
    wal: Mutex<Wal>,
    codecs: CodecRegistry,
    checkpoint_handler: Option<Mutex<Box<dyn CheckpointHandler>>>,
}
```

Note: `checkpoint_handler` is behind its own `Mutex` because `on_checkpoint_needed` takes `&mut self`, and we already hold the WAL lock. Using `Mutex<Box<dyn CheckpointHandler>>` avoids needing both `&mut` on Durable.

Update constructors:

```rust
impl<S: Transact> Durable<S> {
    pub fn new(strategy: S, wal: Wal, codecs: CodecRegistry) -> Self {
        Self {
            inner: strategy,
            wal: Mutex::new(wal),
            codecs,
            checkpoint_handler: None,
        }
    }

    pub fn with_checkpoint(
        strategy: S,
        wal: Wal,
        codecs: CodecRegistry,
        handler: impl CheckpointHandler + 'static,
    ) -> Self {
        Self {
            inner: strategy,
            wal: Mutex::new(wal),
            codecs,
            checkpoint_handler: Some(Mutex::new(Box::new(handler))),
        }
    }
}
```

Update `transact()` — after `forward.apply(world)`, add:

```rust
// Check if checkpoint is needed
if let Some(ref handler_mutex) = self.checkpoint_handler {
    let mut wal = self.wal.lock();
    if wal.checkpoint_needed() {
        let mut handler = handler_mutex.lock();
        handler.on_checkpoint_needed(world, &mut wal, &self.codecs);
    }
}
```

Note: this acquires the WAL lock a second time (first for append, then for checkpoint check). This is fine — the append lock is released after `writer.flush()`, and the checkpoint lock is a separate acquisition.

**Step 4: Run tests**

Run: `cargo test -p minkowski-persist -- durable_fires_checkpoint durable_no_handler`
Expected: PASS

**Step 5: Run all tests**

Run: `cargo test -p minkowski-persist`
Expected: all pass

**Step 6: Commit**

```bash
git add crates/minkowski-persist/src/durable.rs
git commit -m "feat(persist): Durable checkpoint handler integration"
```

---

### Task 7: Update lib.rs exports + clippy + full test suite

Ensure all new types are exported and everything is clean.

**Files:**
- Modify: `crates/minkowski-persist/src/lib.rs`

**Step 1: Verify exports**

Ensure `lib.rs` has:

```rust
pub mod checkpoint;
pub use checkpoint::{AutoCheckpoint, CheckpointHandler};
```

(This was added in Task 5, just verify.)

**Step 2: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean

**Step 3: Run full test suite**

Run: `cargo test -p minkowski-persist && cargo test -p minkowski --lib`
Expected: all pass

**Step 4: Run examples**

Run: `cargo run -p minkowski-examples --example persist --release`
Run: `cargo run -p minkowski-examples --example replicate --release`
Expected: both run successfully (they don't use checkpoints, just verify no regressions)

**Step 5: Commit if any fixes were needed**

```bash
git add -A
git commit -m "chore(persist): checkpoint exports + cleanup"
```
