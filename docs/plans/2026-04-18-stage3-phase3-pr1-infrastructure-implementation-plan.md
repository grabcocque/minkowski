# Phase 3 PR 1: Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Phase 3's pure-infrastructure prerequisites as a single squash-merged PR. No behavioral changes to existing callers; enables PR 2 (LsmCheckpoint + snapshot cut) and PR 3 (compactor).

**Architecture:** Three tightly coupled type-system / module changes: (a) move zero-copy primitives from `minkowski-persist::codec` into `minkowski-lsm::codec` and flip the crate dependency direction; (b) convert `LsmManifest` to a const-generic `LsmManifest<const N: usize = 4>` with `DefaultManifest` alias; (c) change `SortedRunReader::validate_page_crc` to return `Result<CrcProof, LsmError>`, enabling the existing `raw_copy_size` fast path in recovery. Plus a regime-model module doc on `manifest.rs`.

**Scope note:** `recover_world_from_lsm` (the helper that reconstructs a `World` by iterating sorted-run pages) is **deferred to PR 2**, where it pairs naturally with `LsmCheckpoint`'s first real use. PR 1 only lands the `CrcProof`-returning signature change on `validate_page_crc`; PR 2 lands the helper that consumes it. Rationale: the page-to-archetype plumbing (reconstruct archetype schema from slots, spawn entities from entity-ID pages, fill columns from component pages) is non-trivial and has no value without a caller — PR 2 provides both.

**Tech Stack:** Rust edition 2024; `rkyv` 0.8 (moved into lsm with codec); `thiserror` 2 (already in lsm via error.rs); `crc32fast` 1 (already in lsm); clippy::pedantic workspace-wide.

---

## File Structure

### Files moved
- `crates/minkowski-persist/src/codec.rs` → `crates/minkowski-lsm/src/codec.rs` (681 lines, unchanged content except visibility of `CrcProof`)

### Files created
- `docs/plans/2026-04-18-stage3-phase3-pr1-infrastructure-implementation-plan.md` (this file)

### Files modified
- `crates/minkowski-lsm/Cargo.toml` — add `rkyv`, `thiserror` deps
- `crates/minkowski-persist/Cargo.toml` — add `minkowski-lsm` dep
- `crates/minkowski-lsm/src/lib.rs` — add `pub mod codec;` + re-exports
- `crates/minkowski-persist/src/lib.rs` — remove `pub mod codec;`, add re-exports from `minkowski_lsm::codec`
- `crates/minkowski-persist/src/wal.rs` — rewrite `use crate::codec::*` → `use minkowski_lsm::codec::*`
- `crates/minkowski-persist/src/checkpoint.rs` — same import rewrite
- `crates/minkowski-persist/src/durable.rs` — same
- `crates/minkowski-persist/src/replication.rs` — same
- `crates/minkowski-persist/src/snapshot.rs` — same (will be deleted in PR 2, but must compile here)
- `crates/minkowski-persist/src/index.rs` — test-only import rewrite
- `crates/minkowski-persist/src/record.rs` — no direct codec import, but may need `minkowski_lsm::codec::CodecError` if it references it
- `crates/minkowski-persist/src/blob.rs` — check for any codec imports (rkyv usage unrelated)
- `crates/minkowski-lsm/src/manifest.rs` — const-generic `LsmManifest<const N: usize = 4>`, regime-model module doc, `DefaultManifest` alias, `NUM_LEVELS` removed
- `crates/minkowski-lsm/src/types.rs` — add `MAX_LEVELS = 32` constant; `Level::new` checks against `MAX_LEVELS` (not `NUM_LEVELS` which no longer exists)
- `crates/minkowski-lsm/src/manifest_log.rs` — propagate const generic through `ManifestLog::recover`, `apply_entry`, bounds checks
- `crates/minkowski-lsm/src/manifest_ops.rs` — propagate const generic through `flush_and_record`
- `crates/minkowski-lsm/src/writer.rs` — `FlushWriter<const N: usize = 4>`, propagation
- `crates/minkowski-lsm/src/reader.rs` — `SortedRunReader::validate_page_crc` returns `Result<CrcProof, LsmError>`; import `CrcProof` from the new `crate::codec`

### Tests
- `crates/minkowski-lsm/src/manifest.rs` (inline) — new test: `lsm_manifest_alternate_level_count_compiles_and_works`
- `crates/minkowski-lsm/src/reader.rs` (inline) — new test: `validate_page_crc_returns_proof_token`
- `crates/minkowski-persist/*` tests — unchanged, must continue to pass

---

## Task 1: Add rkyv + thiserror deps to minkowski-lsm

**Files:**
- Modify: `crates/minkowski-lsm/Cargo.toml`

- [ ] **Step 1: Edit Cargo.toml to add dependencies**

Replace the `[dependencies]` section with:

```toml
[dependencies]
minkowski = { path = "../minkowski" }
crc32fast = "1"
memmap2 = "0.9"
rkyv = { version = "0.8", features = ["alloc", "bytecheck"] }
thiserror = "2"
```

- [ ] **Step 2: Verify lsm still builds standalone**

Run: `cargo check -p minkowski-lsm`
Expected: clean build, no warnings.

- [ ] **Step 3: Commit**

```bash
git add crates/minkowski-lsm/Cargo.toml
git commit -m "chore(lsm): add rkyv + thiserror deps for codec module move"
```

---

## Task 2: Move codec.rs from persist to lsm

**Files:**
- Move: `crates/minkowski-persist/src/codec.rs` → `crates/minkowski-lsm/src/codec.rs`
- Modify: `crates/minkowski-lsm/src/lib.rs` (add `pub mod codec;`)
- Modify: `crates/minkowski-lsm/src/codec.rs` (widen `CrcProof` visibility)

- [ ] **Step 1: Move the file with git**

Run:
```bash
git mv crates/minkowski-persist/src/codec.rs crates/minkowski-lsm/src/codec.rs
```

- [ ] **Step 2: Add `pub mod codec;` to lsm/lib.rs**

Open `crates/minkowski-lsm/src/lib.rs`. Add near the top, after existing `pub mod` declarations:

```rust
pub mod codec;
```

- [ ] **Step 3: Widen CrcProof visibility from `pub(crate)` to `pub`**

Open `crates/minkowski-lsm/src/codec.rs`. Find the line:

```rust
pub(crate) struct CrcProof(());
```

Replace with:

```rust
/// A proof token returned by [`CrcProof::verify`] after successful CRC32
/// validation of a byte payload. Unforgeable: the only public constructor
/// is [`CrcProof::verify`], which runs the actual checksum.
///
/// Used by [`CodecRegistry::decode`] to gate the `raw_copy_size` fast path
/// (direct memcpy, skipping rkyv bytecheck). Producers: WAL frame reader
/// ([`minkowski_persist::wal::read_next_frame`]), LSM page validator
/// ([`SortedRunReader::validate_page_crc`]).
pub struct CrcProof(());
```

- [ ] **Step 4: Verify lsm still builds**

Run: `cargo check -p minkowski-lsm`
Expected: compiles cleanly. `cargo check` for persist will fail at this point — that is expected and fixed in Task 3.

- [ ] **Step 5: Commit (broken intermediate state acknowledged)**

```bash
git add crates/minkowski-lsm/src/codec.rs \
        crates/minkowski-lsm/src/lib.rs \
        crates/minkowski-persist/src/codec.rs
git commit -m "refactor(lsm): move codec module from persist to lsm

Move CrcProof, CodecRegistry, CodecError into minkowski-lsm::codec.
CrcProof visibility widens from pub(crate) to pub since it now crosses
the crate boundary. minkowski-persist does not build after this commit;
Task 3 restores it by flipping the dep direction."
```

---

## Task 3: Update persist to depend on lsm; rewrite imports

**Files:**
- Modify: `crates/minkowski-persist/Cargo.toml`
- Modify: `crates/minkowski-persist/src/lib.rs`
- Modify: `crates/minkowski-persist/src/wal.rs`
- Modify: `crates/minkowski-persist/src/checkpoint.rs`
- Modify: `crates/minkowski-persist/src/durable.rs`
- Modify: `crates/minkowski-persist/src/replication.rs`
- Modify: `crates/minkowski-persist/src/snapshot.rs` (will die in PR 2 but must compile now)
- Modify: `crates/minkowski-persist/src/index.rs` (test imports)

- [ ] **Step 1: Add minkowski-lsm dep to persist**

Open `crates/minkowski-persist/Cargo.toml`. In `[dependencies]`, add after `minkowski = { path = "../minkowski" }`:

```toml
minkowski-lsm = { path = "../minkowski-lsm" }
```

- [ ] **Step 2: Rewrite lib.rs re-exports**

Open `crates/minkowski-persist/src/lib.rs`. Find:

```rust
pub mod codec;
```

Delete that line. Find:

```rust
pub use codec::{CodecError, CodecRegistry};
```

Replace with:

```rust
pub use minkowski_lsm::codec::{CodecError, CodecRegistry, CrcProof};
```

- [ ] **Step 3: Rewrite imports in wal.rs**

Open `crates/minkowski-persist/src/wal.rs`. Find line 9:

```rust
use crate::codec::{CodecError, CodecRegistry, CrcProof};
```

Replace with:

```rust
use minkowski_lsm::codec::{CodecError, CodecRegistry, CrcProof};
```

Find line 1161 (inside `#[cfg(test)] mod tests`):

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

- [ ] **Step 4: Rewrite imports in checkpoint.rs**

Open `crates/minkowski-persist/src/checkpoint.rs`. Find line 7:

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

Find line 90 (test scope):

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

- [ ] **Step 5: Rewrite imports in durable.rs**

Open `crates/minkowski-persist/src/durable.rs`. Find line 6:

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

Find line 135 (test scope):

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

- [ ] **Step 6: Rewrite imports in replication.rs**

Open `crates/minkowski-persist/src/replication.rs`. Find line 16:

```rust
use crate::codec::{CodecError, CodecRegistry};
```

Replace with:

```rust
use minkowski_lsm::codec::{CodecError, CodecRegistry};
```

Find line 89 (test scope):

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

- [ ] **Step 7: Rewrite imports in snapshot.rs**

Open `crates/minkowski-persist/src/snapshot.rs`. Find line 8:

```rust
use crate::codec::{CodecError, CodecRegistry, CrcProof};
```

Replace with:

```rust
use minkowski_lsm::codec::{CodecError, CodecRegistry, CrcProof};
```

Find line 482 (test scope):

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

- [ ] **Step 8: Rewrite test imports in index.rs**

Open `crates/minkowski-persist/src/index.rs`. Find line 496 (test scope):

```rust
use crate::codec::CodecRegistry;
```

Replace with:

```rust
use minkowski_lsm::codec::CodecRegistry;
```

- [ ] **Step 9: Check for any other `crate::codec` references**

Run:

```bash
grep -rn "crate::codec\|use codec::\|self::codec" crates/minkowski-persist/src/ 2>/dev/null
```

Expected output: none. If any remain, rewrite them following the pattern from Steps 3–8.

- [ ] **Step 10: Build the full workspace**

Run: `cargo check --workspace`
Expected: clean build across all crates.

- [ ] **Step 11: Run full workspace test suite**

Run: `cargo test --workspace`
Expected: all tests pass. No new failures from the move; same test count as before.

- [ ] **Step 12: Clippy check**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 13: Commit**

```bash
git add crates/minkowski-persist/
git commit -m "refactor(persist): migrate codec imports to minkowski-lsm

Add minkowski-lsm as a dependency of minkowski-persist. Rewrite all
use crate::codec::* as use minkowski_lsm::codec::*. Re-export from
persist::lib for downstream convenience. No behavioral changes;
pure import rewrite."
```

---

## Task 4: Const-generic LsmManifest — failing test

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs` (add test)

- [ ] **Step 1: Add failing test at the bottom of the existing `mod tests`**

Open `crates/minkowski-lsm/src/manifest.rs`. Find the existing `#[cfg(test)] mod tests { ... }` block. Add this test before the closing `}` of the mod:

```rust
    #[test]
    fn lsm_manifest_alternate_level_count_compiles_and_works() {
        // Default N=4 path still works.
        let m4: LsmManifest<4> = LsmManifest::new();
        assert_eq!(m4.total_runs(), 0);

        // Alternate N=7 manifest constructs and is distinct at the type level.
        let m7: LsmManifest<7> = LsmManifest::new();
        assert_eq!(m7.total_runs(), 0);

        // DefaultManifest alias resolves to N=4.
        let md: DefaultManifest = LsmManifest::new();
        assert_eq!(md.total_runs(), 0);
    }
```

- [ ] **Step 2: Run the test, confirm it fails to compile**

Run: `cargo test -p minkowski-lsm lsm_manifest_alternate_level_count_compiles_and_works 2>&1 | head -40`
Expected: compilation errors pointing at `LsmManifest<4>`, `LsmManifest<7>`, `DefaultManifest` — none of these types exist yet.

---

## Task 5: Const-generic LsmManifest — introduce parameter

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs`
- Modify: `crates/minkowski-lsm/src/types.rs`

- [ ] **Step 1: Replace NUM_LEVELS constant with MAX_LEVELS sanity bound**

Open `crates/minkowski-lsm/src/types.rs`. Find:

```rust
use crate::manifest::NUM_LEVELS;
```

Replace with:

```rust
/// Maximum level count accepted by [`Level::new`]. A generous upper bound:
/// the default manifest uses 4 levels, TigerBeetle-style configurations
/// use 7. `Level::new` rejects anything ≥ `MAX_LEVELS`; per-manifest
/// bounds are enforced at the manifest boundary.
pub const MAX_LEVELS: usize = 32;
```

Find `Level::new`:

```rust
pub fn new(level: u8) -> Option<Self> {
    if (level as usize) < NUM_LEVELS {
        Some(Self(level))
    } else {
        None
    }
}
```

Replace with:

```rust
pub fn new(level: u8) -> Option<Self> {
    if (level as usize) < MAX_LEVELS {
        Some(Self(level))
    } else {
        None
    }
}
```

- [ ] **Step 2: Update manifest.rs with const generic + regime doc**

Open `crates/minkowski-lsm/src/manifest.rs`. Replace the entire file content starting from the top through the `impl LsmManifest { pub fn new() ...` definition. Be precise:

Find:

```rust
use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

/// Number of LSM levels (L0 through L3).
pub const NUM_LEVELS: usize = 4;

/// In-memory manifest tracking all sorted runs across levels.
pub struct LsmManifest {
    levels: [Vec<SortedRunMeta>; NUM_LEVELS],
    next_sequence: u64,
}
```

Replace with:

```rust
//! # LsmManifest
//!
//! In-memory index of sorted runs across LSM levels.
//!
//! ## Level Count
//!
//! Defaults to 4 levels. This fits the expected minkowski regime:
//! on-disk data up to ~100× RAM, with reads served from the in-memory
//! World rather than from level traversal (the in-memory World IS the
//! merged view).
//!
//! At T=10 size ratio: 4 levels covers ~0.1× to 100× RAM on disk.
//!
//! For ledger-style workloads (TigerBeetle territory, ever-growing
//! history), construct [`LsmManifest<7>`] instead. Merge logic is
//! level-count-agnostic; only bounds checks and manifest serialization
//! care about `N`.

use std::path::{Path, PathBuf};

use crate::error::LsmError;
use crate::types::{Level, PageCount, SeqNo, SeqRange, SizeBytes};

/// In-memory manifest tracking all sorted runs across `N` levels.
///
/// `N` is a const generic with default 4. Use [`DefaultManifest`] as
/// the conventional alias.
pub struct LsmManifest<const N: usize = 4> {
    levels: [Vec<SortedRunMeta>; N],
    next_sequence: u64,
}

/// Conventional alias for the default 4-level manifest.
pub type DefaultManifest = LsmManifest<4>;
```

- [ ] **Step 3: Update impl block**

In `manifest.rs`, find:

```rust
impl LsmManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            levels: std::array::from_fn(|_| Vec::new()),
            next_sequence: 0,
        }
    }
```

Replace the `impl LsmManifest {` line and (only that line) with:

```rust
impl<const N: usize> LsmManifest<N> {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            levels: std::array::from_fn(|_| Vec::new()),
            next_sequence: 0,
        }
    }
```

- [ ] **Step 4: Fix `add_run` / `remove_run` / `promote_run` / `runs_at_level` bounds handling**

Any method that uses `level.as_index()` as an array index into `self.levels` must guard against `level.as_index() >= N` since `Level::new` now accepts up to `MAX_LEVELS = 32`.

In `manifest.rs`, find `add_run`:

```rust
pub fn add_run(&mut self, level: Level, meta: SortedRunMeta) {
    self.levels[level.as_index()].push(meta);
}
```

Replace with:

```rust
pub fn add_run(&mut self, level: Level, meta: SortedRunMeta) -> Result<(), LsmError> {
    if level.as_index() >= N {
        return Err(LsmError::Format(format!(
            "level {} out of range for {}-level manifest",
            level,
            N
        )));
    }
    self.levels[level.as_index()].push(meta);
    Ok(())
}
```

Similarly update `remove_run` (returns `Option<SortedRunMeta>` currently) to return `Option<SortedRunMeta>` with an early `None` if out of range:

Find:

```rust
pub fn remove_run(&mut self, level: Level, path: &Path) -> Option<SortedRunMeta> {
    let runs = &mut self.levels[level.as_index()];
    // ...
}
```

Change the first two lines to:

```rust
pub fn remove_run(&mut self, level: Level, path: &Path) -> Option<SortedRunMeta> {
    if level.as_index() >= N {
        return None;
    }
    let runs = &mut self.levels[level.as_index()];
    // ...
}
```

For `promote_run`:

```rust
pub fn promote_run(&mut self, from: Level, to: Level, path: &Path) -> Result<(), LsmError> {
    // existing body
}
```

Add at the top of the body:

```rust
if from.as_index() >= N || to.as_index() >= N {
    return Err(LsmError::Format(format!(
        "level out of range for {}-level manifest: from={}, to={}",
        N, from, to
    )));
}
```

For `runs_at_level`:

```rust
pub fn runs_at_level(&self, level: Level) -> &[SortedRunMeta] {
    &self.levels[level.as_index()]
}
```

Replace with:

```rust
pub fn runs_at_level(&self, level: Level) -> &[SortedRunMeta] {
    if level.as_index() >= N {
        return &[];
    }
    &self.levels[level.as_index()]
}
```

- [ ] **Step 5: Fix existing tests that expect `add_run` to not return Result**

In `manifest.rs` `mod tests`, find all `m.add_run(Level::Lx, ...)` calls and change them to `m.add_run(Level::Lx, ...).unwrap()`.

- [ ] **Step 6: Fix `for lvl in 0..NUM_LEVELS` loop**

In `manifest.rs` `mod tests`, find:

```rust
for lvl in 0..NUM_LEVELS {
    assert!(m.runs_at_level(Level::new(lvl as u8).unwrap()).is_empty());
}
```

Replace with:

```rust
for lvl in 0..4 {
    assert!(m.runs_at_level(Level::new(lvl as u8).unwrap()).is_empty());
}
```

(The test is checking the default `LsmManifest<4>`, so hardcoding 4 here matches intent.)

- [ ] **Step 7: Run the previously failing test**

Run: `cargo test -p minkowski-lsm lsm_manifest_alternate_level_count_compiles_and_works`
Expected: PASS.

- [ ] **Step 8: Do NOT commit yet** — Task 6 finishes the propagation.

---

## Task 6: Propagate const generic through ManifestLog, flush_and_record, FlushWriter

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest_log.rs`
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs`
- Modify: `crates/minkowski-lsm/src/writer.rs`
- Modify all call sites referencing `LsmManifest` without a type parameter

- [ ] **Step 1: Grep for all `LsmManifest` call sites**

Run:

```bash
grep -rn "LsmManifest" crates/minkowski-lsm/ crates/minkowski-persist/ 2>/dev/null
```

Expected output: a list of ~15–25 lines. Each will either (a) already work as `LsmManifest<4>` (if inference happens) or (b) need an explicit `<const N: usize>` parameter added to a surrounding signature.

- [ ] **Step 2: Update `ManifestLog::recover` signature**

Open `crates/minkowski-lsm/src/manifest_log.rs`. Find the `recover` signature:

```rust
pub fn recover(path: &Path) -> Result<(LsmManifest, Self), LsmError> {
```

Replace with:

```rust
pub fn recover<const N: usize>(path: &Path) -> Result<(LsmManifest<N>, Self), LsmError> {
```

- [ ] **Step 3: Update `apply_entry` to take `&mut LsmManifest<N>`**

In `manifest_log.rs`, find `fn apply_entry(manifest: &mut LsmManifest, entry: &ManifestEntry)`. Change to:

```rust
fn apply_entry<const N: usize>(
    manifest: &mut LsmManifest<N>,
    entry: &ManifestEntry,
) -> Result<(), LsmError> {
    // existing body, with these changes:
    //   - manifest.add_run(level, meta.clone()) becomes manifest.add_run(level, meta.clone())? (propagate error)
    //   - manifest.promote_run(...)? stays as-is (already Result)
```

Specifically in the `AddRun`, `AddRunAndSequence` arms, change:

```rust
manifest.add_run(*level, meta.clone());
```

To:

```rust
manifest.add_run(*level, meta.clone())?;
```

- [ ] **Step 4: Update `replay_frames` helper**

In `manifest_log.rs`, find `replay_frames` and add the const generic propagation:

```rust
fn replay_frames<const N: usize>(
    file: &File,
    path: &Path,
    manifest: &mut LsmManifest<N>,
    start_pos: u64,
) -> Result<u64, LsmError> {
```

- [ ] **Step 5: Update `flush_and_record` signature**

Open `crates/minkowski-lsm/src/manifest_ops.rs`. Find `flush_and_record`:

```rust
pub fn flush_and_record(
    world: &World,
    seq_range: SeqRange,
    manifest: &mut LsmManifest,
    log: &mut ManifestLog,
    run_dir: &Path,
) -> Result<(), LsmError> {
```

Replace with:

```rust
pub fn flush_and_record<const N: usize>(
    world: &World,
    seq_range: SeqRange,
    manifest: &mut LsmManifest<N>,
    log: &mut ManifestLog,
    run_dir: &Path,
) -> Result<(), LsmError> {
```

Inside the body, find `manifest.add_run(Level::L0, meta);` (or similar) and add `?`:

```rust
manifest.add_run(Level::L0, meta)?;
```

- [ ] **Step 6: Update FlushWriter (if it holds LsmManifest)**

Open `crates/minkowski-lsm/src/writer.rs`. Check whether `FlushWriter` references `LsmManifest` in any field or method. If yes, propagate `const N: usize` through its definition. If `FlushWriter` does NOT hold `LsmManifest` (it likely doesn't — it writes pages, not manifests), no change is needed here. Verify by running:

```bash
grep -n "LsmManifest" crates/minkowski-lsm/src/writer.rs
```

If no matches, skip this step.

- [ ] **Step 7: Update tests in manifest_log.rs to use explicit type params**

In `manifest_log.rs`, tests call `ManifestLog::recover(...)`. After the generic is introduced, Rust inference will pick N=4 only if the binding is typed. If the test binds as `let (mut manifest, ...)`, inference uses the default `N=4`. That's usually fine. If any test needs to be explicit, write `ManifestLog::recover::<4>(...)`.

Run tests after every change (see Step 10) to catch missing annotations.

- [ ] **Step 8: Update tests in manifest_ops.rs**

Similarly, `flush_and_record(...)` calls should infer `N=4` from the `LsmManifest<4>` (default) type of the `manifest` argument. If any callsite fails inference, add `flush_and_record::<4>(...)`.

- [ ] **Step 9: Update manifest_integration.rs**

Open `crates/minkowski-lsm/tests/manifest_integration.rs`. Bindings like `let (mut manifest, mut log) = ManifestLog::recover(&log_path).unwrap();` should infer `N=4` and keep working. Verify by running (Step 10).

- [ ] **Step 10: Build and test**

```bash
cargo check -p minkowski-lsm
cargo check --workspace
cargo test -p minkowski-lsm
cargo test --workspace
```

Expected: all pass. 118 tests in minkowski-lsm + one new (`lsm_manifest_alternate_level_count_compiles_and_works`) = 119.

- [ ] **Step 11: Clippy**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add crates/minkowski-lsm/src/manifest.rs \
        crates/minkowski-lsm/src/types.rs \
        crates/minkowski-lsm/src/manifest_log.rs \
        crates/minkowski-lsm/src/manifest_ops.rs \
        crates/minkowski-lsm/src/writer.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "feat(lsm): LsmManifest<const N: usize = 4> with DefaultManifest alias

Convert LsmManifest into a const-generic type parameterized by level
count. Default N=4 remains the conventional choice; N=7 (TigerBeetle
regime) is available via LsmManifest<7>. Merge logic is N-agnostic;
only bounds checks and serialization care about N. Level::new checks
against a generous MAX_LEVELS=32 sanity bound; per-manifest bounds
are enforced at add_run/remove_run/promote_run/runs_at_level.

Add module-level regime-model documentation on manifest.rs explaining
why N=4 fits the expected minkowski workload (on-disk up to ~100× RAM,
reads served from the in-memory World, not from level traversal)."
```

---

## Task 7: `validate_page_crc` returns `CrcProof` — failing test

**Files:**
- Modify: `crates/minkowski-lsm/src/reader.rs`

- [ ] **Step 1: Add failing test**

Open `crates/minkowski-lsm/src/reader.rs`. Inside the existing `#[cfg(test)] mod tests` block, add:

```rust
    #[test]
    fn validate_page_crc_returns_proof_token() {
        // This test fails to compile until validate_page_crc returns
        // Result<CrcProof, LsmError> instead of Result<(), LsmError>.
        use crate::codec::CrcProof;

        let (reader, _td) = build_single_page_reader();
        let page = reader.pages().next().unwrap();
        let proof: CrcProof = reader.validate_page_crc(&page).unwrap();
        // Consume the proof: no-op Drop, but ensures type is correct.
        let _ = proof;
    }
```

If `build_single_page_reader` doesn't already exist, add this helper inside `mod tests`:

```rust
    fn build_single_page_reader() -> (SortedRunReader, tempfile::TempDir) {
        // Reuses the same setup pattern as validate_page_crc_succeeds.
        // Copy the body of validate_page_crc_succeeds up to the point
        // of constructing the reader.
        todo!("copy from validate_page_crc_succeeds setup")
    }
```

Actually, don't use `todo!` — the plan forbids placeholders. Instead, inline the setup in the new test by copying the pattern from the existing `validate_page_crc_succeeds` test (around `reader.rs:437-447`). If the existing test body is short, duplicate it. If long, factor it out into a helper.

Open `reader.rs` and read `validate_page_crc_succeeds` starting at line ~437. Copy the setup (everything before the existing `reader.validate_page_crc(&page).unwrap();` call). Use that setup in the new test.

- [ ] **Step 2: Run test, confirm it fails**

```bash
cargo test -p minkowski-lsm validate_page_crc_returns_proof_token 2>&1 | head -20
```

Expected: compile error — either "cannot find type `CrcProof` in this scope" (if the import line is missing), or "mismatched types" where `()` is returned but `CrcProof` is annotated. Either is a valid fail for TDD purposes.

---

## Task 8: `validate_page_crc` — implement change

**Files:**
- Modify: `crates/minkowski-lsm/src/reader.rs`

- [ ] **Step 1: Import CrcProof**

At the top of `crates/minkowski-lsm/src/reader.rs`, add:

```rust
use crate::codec::CrcProof;
```

- [ ] **Step 2: Change `validate_page_crc` signature and body**

Find the current `validate_page_crc` (around line 278):

```rust
pub fn validate_page_crc(&self, page: &PageRef<'_>) -> Result<(), LsmError> {
    let item_size = self.item_size_for_slot(page.header().slot)?;
    let actual_len = page.header().row_count as usize * item_size;
    let computed = crc32fast::hash(&page.data()[..actual_len]);

    if computed != page.header().page_crc32 {
        return Err(LsmError::Crc {
            offset: page.file_offset(),
            expected: page.header().page_crc32,
            actual: computed,
        });
    }
    Ok(())
}
```

Replace with:

```rust
/// Validate the CRC of a specific page and return a [`CrcProof`] on success.
///
/// The returned token feeds into [`CodecRegistry::decode`]'s `raw_copy_size`
/// fast path (direct memcpy, skipping rkyv bytecheck). The CRC covers
/// `row_count * item_size` bytes — the actual data, not zero-padding.
pub fn validate_page_crc(&self, page: &PageRef<'_>) -> Result<CrcProof, LsmError> {
    let item_size = self.item_size_for_slot(page.header().slot)?;
    let actual_len = page.header().row_count as usize * item_size;
    let payload = &page.data()[..actual_len];

    CrcProof::verify(payload, page.header().page_crc32).ok_or_else(|| LsmError::Crc {
        offset: page.file_offset(),
        expected: page.header().page_crc32,
        actual: crc32fast::hash(payload),
    })
}
```

This reuses `CrcProof::verify` rather than computing the CRC twice on the success path. On the error path, we recompute to populate the `actual` field (same cost as the old code).

- [ ] **Step 3: Update existing caller in the same test module**

Find `validate_page_crc_succeeds` and any other tests that call `validate_page_crc(...).unwrap()` and discard the result. They continue to work because the returned `CrcProof` is ignored. No changes needed unless a test asserts on `Ok(())` explicitly — in that case, change `Ok(())` to `Ok(_)` or extract the proof.

Grep to audit:

```bash
grep -n "validate_page_crc" crates/minkowski-lsm/src/reader.rs
```

Expected: 3 existing test callsites + 1 new one. Verify each still compiles in Step 5.

- [ ] **Step 4: Update any external callers**

Grep the workspace:

```bash
grep -rn "validate_page_crc" crates/ 2>/dev/null
```

Expected: only in `reader.rs` and the PR 1 test. No external callers today. If any appear, adjust as needed.

- [ ] **Step 5: Run tests**

```bash
cargo test -p minkowski-lsm
```

Expected: `validate_page_crc_returns_proof_token` now PASSES. All existing tests still pass. 120 tests total.

- [ ] **Step 6: Clippy**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski-lsm/src/reader.rs
git commit -m "feat(lsm): validate_page_crc returns CrcProof on success

SortedRunReader::validate_page_crc now returns Result<CrcProof, LsmError>.
The returned proof token feeds CodecRegistry::decode's raw_copy_size
fast path during recovery, preserving zero-copy page restoration.

No callers today consume the proof — this prepares the API for
recover_world_from_lsm in a subsequent commit."
```

---

## Task 9: Final verification + push

**Files:** none modified; verification only

- [ ] **Step 1: Full workspace build**

```bash
cargo check --workspace
```

Expected: clean.

- [ ] **Step 2: Full workspace test**

```bash
cargo test --workspace
```

Expected: all tests pass. `minkowski-lsm` has 120 tests (118 prior + `lsm_manifest_alternate_level_count_compiles_and_works` + `validate_page_crc_returns_proof_token`). `minkowski-persist` test count unchanged.

- [ ] **Step 3: Clippy pedantic**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 4: Format check**

```bash
cargo fmt --all --check
```

Expected: clean.

- [ ] **Step 5: Create branch and push**

Start a new branch based on main:

```bash
git checkout -b lsm/phase3-pr1-infrastructure main
# Cherry-pick commits from the working branch if needed, OR
# if the commits were already on a transient branch, rebase them.
```

If the PR 1 commits were made on an existing branch (e.g., still on `lsm/phase3-prep-enum-tags` which has PR #167 squashed into main), cherry-pick the new commits onto a fresh branch off main:

```bash
git log --oneline main..HEAD | tail -n +1  # see commits to pick
git checkout -b lsm/phase3-pr1-infrastructure main
git cherry-pick <each-sha>
```

Then push:

```bash
git push -u origin lsm/phase3-pr1-infrastructure
```

- [ ] **Step 6: Open PR**

```bash
gh pr create --title "refactor(lsm): Phase 3 PR 1 — infrastructure" --body "$(cat <<'EOF'
## Summary

Phase 3 infrastructure PR — prerequisite for PR 2 (LsmCheckpoint + snapshot cut) and PR 3 (compactor). No behavioral changes; pure refactor + type-system extension.

Design: `docs/plans/2026-04-18-stage3-phase3-compactor-design.md` §PR 1.

### Changes

- Move `CrcProof` + `CodecRegistry` + `CodecError` from `minkowski-persist::codec` → `minkowski-lsm::codec`
- Flip crate layering: `minkowski-persist` now depends on `minkowski-lsm`
- Convert `LsmManifest` → `LsmManifest<const N: usize = 4>` with `DefaultManifest` alias
- Propagate const generic through `ManifestLog::recover`, `flush_and_record`, `apply_entry`, `replay_frames`
- `SortedRunReader::validate_page_crc` returns `Result<CrcProof, LsmError>` (was `Result<(), LsmError>`)
- Module-level regime-model documentation on `manifest.rs`

Note: `recover_world_from_lsm` helper is deferred to PR 2, where it lands alongside `LsmCheckpoint`'s first use.

## Test Plan
- [x] `cargo test --workspace` — all tests pass (120 in lsm, persist unchanged)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` — clean
- [x] `cargo fmt --all --check` — clean
- [x] New tests: const-generic manifest alternate-N compiles and works; validate_page_crc returns proof token

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: Wait for CI to go green (7/7)**

Monitor via `gh pr checks`. All gates must pass: Format, Clippy, Test, ThreadSanitizer, Loom, CI, Claude Code Review.

- [ ] **Step 8: User squash-merges**

This is an explicit user action; the plan ends here. Post-merge cleanup (branch deletion, memory updates, etc.) is handled outside this plan.

---

## Known Risks

- **Zero-copy regression risk**: `validate_page_crc`'s return-type change is mechanical. In PR 1 there are no production callers of `validate_page_crc` that consume the proof — the only post-change consumers are tests. PR 2 is where the zero-copy path is actually exercised end-to-end (via `recover_world_from_lsm`), and that's where regression becomes possible. Plan-level mitigation for PR 1: the signature change compiles and tests demonstrate the proof is producible. End-to-end zero-copy verification lives in PR 2.

- **const generic inference failures in tests**: introducing `LsmManifest<const N: usize = 4>` may cause inference failures in test bindings that previously worked without annotations. The plan's test-update steps assume inference picks N=4 from the default; if a specific test requires `LsmManifest<4>` to be explicit, the task steps flag it.

- **`replicate.rs` example**: PR 1 does not touch `examples/replicate.rs`. Snapshot-based APIs are still in use there. PR 1 compiles as-is; PR 2 stubs or rewrites the example.
