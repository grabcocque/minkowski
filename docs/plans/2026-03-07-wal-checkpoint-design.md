# WAL Checkpoint Markers Design

**Goal:** Protect users from unbounded WAL growth by providing a callback-driven checkpoint mechanism that triggers when the WAL exceeds a configurable byte threshold without a snapshot.

**Architecture:** A new `WalEntry::Checkpoint` variant is written to the WAL stream when a snapshot is acknowledged. `Durable` checks `wal.checkpoint_needed()` after each successful transact and fires an optional `CheckpointHandler` callback. A default `AutoCheckpoint` impl does the obvious thing (save snapshot + acknowledge). Users can override with custom behavior.

**Guiding principle:** Mechanism, not policy. The engine provides the trigger and a sensible default. The user controls thresholds, snapshot location, rotation, and any custom behavior via trait implementation.

---

## WalEntry::Checkpoint

New variant in the WAL entry enum:

```rust
pub enum WalEntry {
    Schema(WalSchema),
    Mutations(WalRecord),
    Checkpoint { snapshot_seq: u64 },
}
```

Written to the WAL stream by `Wal::acknowledge_snapshot(seq)`. Self-describing — discovered during `open()` active segment scan. No separate metadata file.

## WalConfig addition

```rust
pub struct WalConfig {
    pub max_segment_bytes: usize,                      // existing, default 64MB
    pub max_bytes_between_checkpoints: Option<usize>,  // None = disabled
}
```

`None` means no checkpoint enforcement (opt-in safety net).

## Wal state + methods

New fields on `Wal`:

```rust
last_checkpoint_seq: Option<u64>,
bytes_since_checkpoint: u64,
```

New methods:

- **`acknowledge_snapshot(seq)`** — writes `WalEntry::Checkpoint { snapshot_seq }` to the active segment, resets `bytes_since_checkpoint` to 0, stores `last_checkpoint_seq`
- **`checkpoint_needed()`** — returns true when `bytes_since_checkpoint >= max_bytes_between_checkpoints`
- **`last_checkpoint_seq()`** — returns the last acknowledged snapshot seq

`append()` increments `bytes_since_checkpoint` after each write. No callback — just bookkeeping.

`open()` picks up `Checkpoint` entries during active segment scan. `bytes_since_checkpoint` is set to bytes written after the last checkpoint in the active segment. Conservative if the checkpoint was in a sealed segment (may trigger an early checkpoint after restart — safe).

## CheckpointHandler trait

```rust
pub trait CheckpointHandler {
    fn on_checkpoint_needed(
        &mut self,
        world: &mut World,
        wal: &mut Wal,
        codecs: &CodecRegistry,
    );
}
```

Called synchronously from `Durable::transact()` after successful WAL append + world apply, when `wal.checkpoint_needed()` returns true.

## AutoCheckpoint default

```rust
pub struct AutoCheckpoint {
    snap_dir: PathBuf,
}
```

Implements `CheckpointHandler`: saves a snapshot to `snap_dir/checkpoint-{seq:06}.snap`, calls `wal.acknowledge_snapshot(seq)`. Does the obvious, correct thing by default.

## Durable integration

`Durable<S>` gains an optional handler:

```rust
pub struct Durable<S: Transact> {
    inner: S,
    wal: Mutex<Wal>,
    codecs: CodecRegistry,
    checkpoint_handler: Option<Box<dyn CheckpointHandler + Send>>,
}
```

- `Durable::new(strategy, wal, codecs)` — no handler (backward compatible)
- `Durable::with_checkpoint(strategy, wal, codecs, handler)` — with handler

`transact()` checks `wal.checkpoint_needed()` after successful commit. If true and handler exists, calls `handler.on_checkpoint_needed(world, &mut wal, &self.codecs)`.

## Edge cases

- **No checkpoint configured**: everything works as today
- **Checkpoint in sealed segment**: on `open()`, bytes_since_checkpoint is conservatively set to active segment's mutation bytes (may trigger early — safe)
- **Handler panics**: propagates (same policy as WAL write failure)
- **WalCursor sees checkpoint entries**: skips them (not mutations, not schema)
- **replay_from with checkpoints**: checkpoint entries in the stream are skipped during replay. Future optimization: use `last_checkpoint_seq` to skip segments

## Testing

- `acknowledge_snapshot` writes entry and resets counter
- `checkpoint_needed` threshold detection
- `open` recovers `last_checkpoint_seq` from active segment
- `Durable::transact` fires handler when threshold crossed
- `AutoCheckpoint` creates snapshot and resets counter
- `WalCursor` skips checkpoint entries
- `replay_from` works with checkpoint entries in stream
- No handler set — backward compatible, no enforcement
