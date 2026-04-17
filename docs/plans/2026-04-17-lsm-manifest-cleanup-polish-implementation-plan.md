# LSM Manifest Cleanup Polish (PR B2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply six independent cleanup items queued from PR A + PR B1 reviews — tighten `writer::flush` signature, add `SeqNo` arithmetic tombstones, introduce `PageCount(NonZeroU64)` newtype, switch `archetype_coverage` to `Box<[u16]>`, add `L3` encode/decode round-trip test, and fix tag-constant coupling in regression tests.

**Architecture:** Incremental type-safety and test-hardening polish on the post-PR-B1 manifest subsystem. Each task produces a focused commit touching 1-3 files; no cross-item coupling. No wire format changes (PageCount wraps u64 at codec boundaries). Each task ends with green `cargo test -p minkowski-lsm` + `cargo clippy --workspace --all-targets -- -D warnings`.

**Tech Stack:** Rust 2024 edition, `minkowski-lsm` workspace crate, existing `LsmError`/`SeqNo`/`SeqRange`/`Level` types from PR A, new `static_assertions` dev-dependency (Task 4).

**Spec:** `project_lsm_phase2_type_safety.md` in `~/.claude/projects/-home-lewdwig-git-minkowski/memory/` — the memory file serves as spec.

---

## Starting state

- Branch to create off `origin/main`: `lsm/pr-b2-cleanup`
- Post-PR-B1 base: commit `86ec118`
- 99 tests currently passing in `minkowski-lsm`

## File structure

**Modify (by task):**

- Task 1: `crates/minkowski-lsm/src/writer.rs` + `crates/minkowski-lsm/src/manifest_ops.rs` — propagate `SeqRange` through the flush call chain.
- Task 2: `crates/minkowski-lsm/src/types.rs` (add `PageCount`) + `crates/minkowski-lsm/src/manifest.rs` (field type change + constructor) + `crates/minkowski-lsm/src/manifest_log.rs` (encode/decode) + `crates/minkowski-lsm/src/manifest_ops.rs` (construction site).
- Task 3: `crates/minkowski-lsm/src/manifest.rs` — `archetype_coverage: Box<[u16]>`.
- Task 4: `crates/minkowski-lsm/Cargo.toml` (add dev-dep), `crates/minkowski-lsm/src/types.rs` (tombstones), `crates/minkowski-lsm/src/manifest_log.rs` (L3 test + tag const visibility), `crates/minkowski-lsm/tests/manifest_integration.rs` (tag import + reserved-bytes test + idempotency test).

**No new files.** All changes fit within existing module layout.

---

## Task 1: `writer::flush` signature — take `SeqRange`

**Goal:** Replace `sequence_range: (u64, u64)` in `writer::flush` with `SeqRange`. Propagate through `flush_and_record`. Closes the write-side slack where a caller could pass an invalid range and only hit validation at the next file open.

**Files:**
- Modify: `crates/minkowski-lsm/src/writer.rs`
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs`

- [ ] **Step 1: Examine current `writer::flush` signature**

Run: `grep -n "pub fn flush" crates/minkowski-lsm/src/writer.rs`
Expected: one match showing `pub fn flush(world: &World, sequence_range: (u64, u64), output_dir: &Path) -> Result<Option<PathBuf>, LsmError>`.

- [ ] **Step 2: Change `writer::flush` signature to take `SeqRange`**

In `crates/minkowski-lsm/src/writer.rs`, update the imports at the top:

```rust
use crate::types::{SeqNo, SeqRange};
```

Find the function signature:

```rust
pub fn flush(
    world: &World,
    sequence_range: (u64, u64),
    output_dir: &Path,
) -> Result<Option<PathBuf>, LsmError> {
```

Change to:

```rust
pub fn flush(
    world: &World,
    sequence_range: SeqRange,
    output_dir: &Path,
) -> Result<Option<PathBuf>, LsmError> {
```

Inside the function body, find the sites where `sequence_range.0` and `sequence_range.1` are used (they're written to the sorted-run header bytes). Replace with the `SeqRange` accessors:

```rust
// Before:
header.sequence_lo = sequence_range.0;
header.sequence_hi = sequence_range.1;

// After:
header.sequence_lo = sequence_range.lo().0;
header.sequence_hi = sequence_range.hi().0;
```

(The exact variable names may differ — find the lines that set `sequence_lo` / `sequence_hi` on the header struct and update accordingly. `SeqRange::lo()` and `::hi()` return `SeqNo`; `.0` extracts the inner u64.)

- [ ] **Step 3: Update `flush_and_record` in `manifest_ops.rs`**

In `crates/minkowski-lsm/src/manifest_ops.rs`, find `flush_and_record`:

```rust
pub fn flush_and_record(
    world: &World,
    sequence_range: (u64, u64),
    manifest: &mut LsmManifest,
    log: &mut ManifestLog,
    output_dir: &Path,
) -> Result<Option<PathBuf>, LsmError> {
    let Some(path) = flush(world, sequence_range, output_dir)? else {
        return Ok(None);
    };
    // ... rest
    log.append(&ManifestEntry::AddRunAndSequence {
        level: Level::L0,
        meta: meta.clone(),
        next_sequence: SeqNo(sequence_range.1),
    })?;
    manifest.add_run(Level::L0, meta);
    manifest.set_next_sequence(SeqNo(sequence_range.1));
```

Change the signature to take `SeqRange` and update the internal uses:

```rust
pub fn flush_and_record(
    world: &World,
    sequence_range: SeqRange,
    manifest: &mut LsmManifest,
    log: &mut ManifestLog,
    output_dir: &Path,
) -> Result<Option<PathBuf>, LsmError> {
    let Some(path) = flush(world, sequence_range, output_dir)? else {
        return Ok(None);
    };
    // ... rest
    log.append(&ManifestEntry::AddRunAndSequence {
        level: Level::L0,
        meta: meta.clone(),
        next_sequence: sequence_range.hi(),
    })?;
    manifest.add_run(Level::L0, meta);
    manifest.set_next_sequence(sequence_range.hi());
```

Note the `sequence_range.hi()` returns `SeqNo` directly — no wrapping needed.

- [ ] **Step 4: Update test call sites**

Run: `cargo check -p minkowski-lsm 2>&1 | head -30`
Expected: compilation errors at every test call site that passes `(u64, u64)` to `flush_and_record`. Enumerate and update each.

Call sites in `manifest_ops.rs`'s test module: change `flush_and_record(&world, (0, 10), ...)` to `flush_and_record(&world, SeqRange::new(SeqNo(0), SeqNo(10)).unwrap(), ...)`.

Call sites in `tests/manifest_integration.rs`: same pattern. Every `flush_and_record(&world, (lo, hi), ...)` becomes `flush_and_record(&world, SeqRange::new(SeqNo(lo), SeqNo(hi)).unwrap(), ...)`.

Add import to the test module or top of `manifest_ops.rs` if not present:

```rust
use crate::types::{SeqNo, SeqRange};
```

In `tests/manifest_integration.rs` (external file), the import should already be at the top from prior PRs. Verify.

Also update any `writer::flush` direct call sites in the crate. Search: `grep -rn "writer::flush\|::flush(" crates/minkowski-lsm/src/`. The only direct caller should be `flush_and_record`.

- [ ] **Step 5: Run cargo check**

Run: `cargo check -p minkowski-lsm`
Expected: clean compile.

- [ ] **Step 6: Run tests**

Run: `cargo test -p minkowski-lsm`
Expected: 99 tests pass, no regressions.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add crates/minkowski-lsm/src/writer.rs \
        crates/minkowski-lsm/src/manifest_ops.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "refactor(lsm): writer::flush and flush_and_record take SeqRange

Previously both functions accepted sequence_range as (u64, u64), a
raw tuple where (10, 5) was representable. PR A's SeqRange newtype
enforces lo <= hi at construction; propagating SeqRange through the
write side means an invalid range is rejected at the first SeqRange::new
call instead of silently writing a malformed header.

No wire format change — sequence_range.lo().0 / .hi().0 extract the
u64 bytes at the header write boundary. Every call site migrated to
SeqRange::new(SeqNo(lo), SeqNo(hi)).unwrap() at the point of construction."
```

If fmt hook modifies files, re-stage and commit new (never amend).

---

## Task 2: `PageCount(NonZeroU64)` newtype

**Goal:** Replace `SortedRunMeta.page_count: u64` with `PageCount(NonZeroU64)`. Collapses the runtime non-zero check into the type. Also provides type-level distinction from `size_bytes: u64` (currently trivially swappable as args).

**Files:**
- Modify: `crates/minkowski-lsm/src/types.rs` (new type)
- Modify: `crates/minkowski-lsm/src/manifest.rs` (field type + constructor + accessor + tests)
- Modify: `crates/minkowski-lsm/src/manifest_log.rs` (encode/decode)
- Modify: `crates/minkowski-lsm/src/manifest_ops.rs` (construction site in flush_and_record)
- Modify: `crates/minkowski-lsm/src/reader.rs` (if `SortedRunReader::page_count()` returns u64 — update to PageCount)
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs` (any `.page_count()` comparisons)

- [ ] **Step 1: Write failing unit tests for `PageCount`**

Add to `crates/minkowski-lsm/src/types.rs` test module:

```rust
    #[test]
    fn pagecount_rejects_zero() {
        assert!(PageCount::new(0).is_none());
    }

    #[test]
    fn pagecount_accepts_one() {
        let pc = PageCount::new(1).unwrap();
        assert_eq!(pc.get(), 1);
    }

    #[test]
    fn pagecount_accepts_large_values() {
        let pc = PageCount::new(u64::MAX).unwrap();
        assert_eq!(pc.get(), u64::MAX);
    }

    #[test]
    fn pagecount_roundtrip() {
        let pc = PageCount::new(42).unwrap();
        let raw: u64 = pc.get();
        let restored = PageCount::new(raw).unwrap();
        assert_eq!(pc, restored);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski-lsm --lib types::tests::pagecount`
Expected: FAIL — `PageCount` doesn't exist.

- [ ] **Step 3: Add `PageCount` to `types.rs`**

Add imports at the top of `crates/minkowski-lsm/src/types.rs`:

```rust
use std::num::NonZeroU64;
```

Add after the `Level` type:

```rust
/// A page count — guaranteed non-zero at construction.
///
/// Layout-compatible with `u64` via `std::num::NonZeroU64`, so
/// `Option<PageCount>` has the same size as `u64` (niche optimization).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PageCount(NonZeroU64);

impl PageCount {
    /// Construct a page count. Returns `None` if `value` is zero.
    pub fn new(value: u64) -> Option<Self> {
        NonZeroU64::new(value).map(Self)
    }

    /// Extract the underlying `u64`.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for PageCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.get())
    }
}
```

- [ ] **Step 4: Run the new tests**

Run: `cargo test -p minkowski-lsm --lib types::tests::pagecount`
Expected: 4 tests pass.

- [ ] **Step 5: Change `SortedRunMeta.page_count` field type**

In `crates/minkowski-lsm/src/manifest.rs`, update imports:

```rust
use crate::types::{PageCount, SeqRange};
```

Change the field type in the struct:

```rust
pub struct SortedRunMeta {
    path: PathBuf,
    sequence_range: SeqRange,
    archetype_coverage: Vec<u16>,
    page_count: PageCount,       // was: u64
    size_bytes: u64,
}
```

Update `SortedRunMeta::new` signature and body:

```rust
    pub fn new(
        path: PathBuf,
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,
        page_count: u64,             // constructor still takes u64 for convenience
        size_bytes: u64,
    ) -> Result<Self, LsmError> {
        if archetype_coverage.windows(2).any(|w| w[0] >= w[1]) {
            return Err(LsmError::Format(
                "archetype_coverage is not strictly sorted".to_owned(),
            ));
        }
        let page_count = PageCount::new(page_count).ok_or_else(|| {
            LsmError::Format("page_count must be non-zero".to_owned())
        })?;
        Ok(Self {
            path,
            sequence_range,
            archetype_coverage,
            page_count,
            size_bytes,
        })
    }
```

The old explicit `if page_count == 0 { return Err(...) }` is now subsumed by `PageCount::new` returning `None`.

Update the accessor:

```rust
    pub fn page_count(&self) -> PageCount {
        self.page_count
    }
```

- [ ] **Step 6: Update the encode path in `manifest_log.rs`**

Find the `AddRun` encode arm in `encode_entry` (`crates/minkowski-lsm/src/manifest_log.rs`):

```rust
            buf.extend_from_slice(&meta.page_count.to_le_bytes());
```

Change to (the accessor now returns `PageCount`, so extract the u64):

```rust
            buf.extend_from_slice(&meta.page_count.get().to_le_bytes());
```

Wait — this accesses the private field directly. Verify whether the encode path uses `meta.page_count` (private field access, same-crate OK) or `meta.page_count()` (accessor). After PR A, fields are fully private, so encode uses the accessor. Update both encode sites (`AddRun` and `AddRunAndSequence` arms) to use `.page_count().get()`:

```rust
            buf.extend_from_slice(&meta.page_count().get().to_le_bytes());
```

Apply to BOTH encode branches.

- [ ] **Step 7: Update the decode path in `manifest_log.rs`**

Find the `AddRun` decode arm. Currently:

```rust
            let page_count = read_u64_le(data, &mut offset)?;
            // ...
            let meta = SortedRunMeta::new(
                path,
                SeqRange::new(SeqNo(seq_lo), SeqNo(seq_hi))?,
                coverage,
                page_count,        // u64
                size_bytes,
            )?;
```

No change here — the constructor still takes `u64` and wraps internally. The validation path is correct (a corrupt frame with `page_count = 0` will fail `SortedRunMeta::new` → `LsmError::Format` → replay truncates).

Same for `AddRunAndSequence` decode arm: no change.

- [ ] **Step 8: Update `flush_and_record` construction site**

In `crates/minkowski-lsm/src/manifest_ops.rs`, find the `SortedRunMeta::new` call. `reader.page_count()` returns... let me check. Currently `SortedRunReader::page_count()` returns `u64`. After this task, it could stay `u64` (constructor handles wrapping) or become `PageCount` directly.

For now, keep `SortedRunReader::page_count() -> u64` unchanged (storage layer, not API-consumer-facing). `flush_and_record` passes `reader.page_count()` to `SortedRunMeta::new` which wraps it. No change needed in `flush_and_record`.

- [ ] **Step 9: Update test helpers and call sites**

Any test that constructs `SortedRunMeta::new(...)` passes a literal `u64` as `page_count`. No signature change for the constructor — still takes `u64`. No changes needed.

BUT: any code that reads `meta.page_count()` and compares it to a `u64` literal now needs `.get()`:

```rust
// Before:
assert_eq!(meta.page_count(), 1);

// After:
assert_eq!(meta.page_count().get(), 1);
```

Run `grep -rn "\.page_count()" crates/minkowski-lsm/ tests/` to enumerate; update each.

Also the existing test `sorted_run_meta_new_rejects_zero_page_count` in `manifest.rs` — its behavior is unchanged (still returns `Err(LsmError::Format)` on zero), but the error message text may have shifted from "page_count must be non-zero" to the same string (we kept the same text in Step 5). No test update needed.

- [ ] **Step 10: Run cargo check**

Run: `cargo check -p minkowski-lsm`
Expected: clean compile. If any `.page_count()` comparison against a bare u64 was missed, compile fails there — migrate.

- [ ] **Step 11: Run full tests**

Run: `cargo test -p minkowski-lsm`
Expected: 103 tests pass (99 existing + 4 new `pagecount_*`).

- [ ] **Step 12: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 13: Commit**

```bash
git add crates/minkowski-lsm/
git commit -m "feat(lsm): PageCount(NonZeroU64) newtype for SortedRunMeta.page_count

Replaces the runtime 'page_count > 0' check in SortedRunMeta::new with
a type-level guarantee. PageCount(NonZeroU64) gives niche optimization
(Option<PageCount> is still u64-sized) and type-level distinction from
size_bytes: u64 (previously trivially swappable as args).

Wire format unchanged: encode writes page_count.get().to_le_bytes();
decode reads u64 and SortedRunMeta::new wraps via PageCount::new,
surfacing zero as LsmError::Format (still tail-truncation on replay).

Accessor SortedRunMeta::page_count() now returns PageCount instead of
u64. Call sites that compared to a u64 literal need .get()."
```

---

## Task 3: `archetype_coverage: Box<[u16]>`

**Goal:** Change `SortedRunMeta.archetype_coverage` from `Vec<u16>` to `Box<[u16]>`. Saves 8 bytes per meta (drops capacity field) and signals fixed-shape after construction.

**Files:**
- Modify: `crates/minkowski-lsm/src/manifest.rs`

- [ ] **Step 1: Change the field type**

In `crates/minkowski-lsm/src/manifest.rs`:

```rust
pub struct SortedRunMeta {
    path: PathBuf,
    sequence_range: SeqRange,
    archetype_coverage: Box<[u16]>,     // was: Vec<u16>
    page_count: PageCount,
    size_bytes: u64,
}
```

- [ ] **Step 2: Update `SortedRunMeta::new` constructor**

The constructor still takes `Vec<u16>` (convenient for callers). Convert to `Box<[u16]>` at the boundary:

```rust
    pub fn new(
        path: PathBuf,
        sequence_range: SeqRange,
        archetype_coverage: Vec<u16>,         // still Vec for caller convenience
        page_count: u64,
        size_bytes: u64,
    ) -> Result<Self, LsmError> {
        if archetype_coverage.windows(2).any(|w| w[0] >= w[1]) {
            return Err(LsmError::Format(
                "archetype_coverage is not strictly sorted".to_owned(),
            ));
        }
        let page_count = PageCount::new(page_count).ok_or_else(|| {
            LsmError::Format("page_count must be non-zero".to_owned())
        })?;
        Ok(Self {
            path,
            sequence_range,
            archetype_coverage: archetype_coverage.into_boxed_slice(),
            page_count,
            size_bytes,
        })
    }
```

The `Vec::into_boxed_slice()` is O(1) if capacity == length, O(n) to reallocate otherwise. For coverage vecs constructed in-place and immediately stored, it's O(1) in practice.

- [ ] **Step 3: Verify the accessor is unchanged**

The accessor `archetype_coverage() -> &[u16]` already returns a slice. `Box<[u16]>` derefs to `&[u16]` the same way `Vec<u16>` does. No change needed to the accessor:

```rust
    pub fn archetype_coverage(&self) -> &[u16] {
        &self.archetype_coverage
    }
```

- [ ] **Step 4: Run cargo check**

Run: `cargo check -p minkowski-lsm`
Expected: clean compile. The `Box<[u16]>` field dereferences compatibly with `Vec<u16>` in the accessor; no other code should break.

- [ ] **Step 5: Run tests**

Run: `cargo test -p minkowski-lsm`
Expected: 103 tests pass, no regressions. (The 4 tests from Task 2 are already in.)

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/minkowski-lsm/src/manifest.rs
git commit -m "refactor(lsm): SortedRunMeta.archetype_coverage: Vec<u16> -> Box<[u16]>

Saves 8 bytes per meta (drops Vec's capacity field) and signals the
fixed-shape-after-construction invariant. Accessor unchanged (returns
&[u16] either way). Constructor still accepts Vec<u16> for caller
convenience; converts via into_boxed_slice() at the boundary.

No API change visible to consumers."
```

---

## Task 4: Test hardening — tombstones, L3 round-trip, tag constants, polish tests

**Goal:** Add `SeqNo` arithmetic tombstone tests, L3 encode/decode round-trip test, expose tag constants as `pub(crate)` and use them in regression tests, plus two optional polish tests (reserved-bytes forward-compat, zero-flush idempotency).

**Files:**
- Modify: `crates/minkowski-lsm/Cargo.toml` (add dev-dep)
- Modify: `crates/minkowski-lsm/src/types.rs` (tombstones)
- Modify: `crates/minkowski-lsm/src/manifest_log.rs` (expose tag constants, add L3 round-trip)
- Modify: `crates/minkowski-lsm/tests/manifest_integration.rs` (tag imports + 2 polish tests)

- [ ] **Step 1: Add `static_assertions` as a dev-dependency**

In `crates/minkowski-lsm/Cargo.toml`, find the `[dev-dependencies]` section (or add one if missing). Add:

```toml
[dev-dependencies]
static_assertions = "1"
# existing dev deps below this line
```

(If `static_assertions` is already in the workspace root's `[workspace.dependencies]`, use `static_assertions.workspace = true` instead. Check first.)

Run: `cargo build -p minkowski-lsm --tests 2>&1 | head -5`
Expected: clean build after dep resolution.

- [ ] **Step 2: Add `SeqNo` arithmetic tombstone tests**

In `crates/minkowski-lsm/src/types.rs`, inside the `#[cfg(test)] mod tests` block, add:

```rust
    // Tombstone tests: SeqNo must NOT implement Add/Sub/AddAssign/SubAssign.
    // "sequence numbers are identities, not sizes."
    use static_assertions::assert_not_impl_all;
    use std::ops::{Add, AddAssign, Sub, SubAssign};

    assert_not_impl_all!(SeqNo: Add<SeqNo>, Sub<SeqNo>, AddAssign<SeqNo>, SubAssign<SeqNo>);
    assert_not_impl_all!(SeqNo: Add<u64>, Sub<u64>, AddAssign<u64>, SubAssign<u64>);
```

`assert_not_impl_all!` is a compile-time assertion — if any of the listed traits become implemented for `SeqNo`, the crate fails to build. This is the right mechanism for "we never want this to compile."

Note: `assert_not_impl_all!` is a macro that expands at the module level, not inside a function. Place it outside any `#[test]` function but inside the test module.

- [ ] **Step 3: Run tests to verify tombstones compile and pass**

Run: `cargo test -p minkowski-lsm --lib types::tests`
Expected: all tests pass. If `SeqNo` accidentally has an arithmetic impl, compilation fails with a clear `assert_not_impl_all` error pointing at the offending trait.

- [ ] **Step 4: Expose tag constants as `pub(crate)`**

In `crates/minkowski-lsm/src/manifest_log.rs`, find the tag constants:

```rust
const TAG_ADD_RUN: u8 = 0x01;
const TAG_REMOVE_RUN: u8 = 0x02;
const TAG_PROMOTE_RUN: u8 = 0x03;
const TAG_SET_SEQUENCE: u8 = 0x04;
const TAG_ADD_RUN_AND_SEQUENCE: u8 = 0x05;
```

Change to `pub(crate)`:

```rust
pub(crate) const TAG_ADD_RUN: u8 = 0x01;
pub(crate) const TAG_REMOVE_RUN: u8 = 0x02;
pub(crate) const TAG_PROMOTE_RUN: u8 = 0x03;
pub(crate) const TAG_SET_SEQUENCE: u8 = 0x04;
pub(crate) const TAG_ADD_RUN_AND_SEQUENCE: u8 = 0x05;
```

- [ ] **Step 5: Add an `L3` encode/decode round-trip test**

In `crates/minkowski-lsm/src/manifest_log.rs` test module, find the existing `encode_decode_add_run_and_sequence` test. It uses `Level::L0` or similar. Add a new test exercising `Level::L3`:

```rust
    #[test]
    fn encode_decode_add_run_and_sequence_at_l3() {
        let meta = test_meta("l3.run");
        let entry = ManifestEntry::AddRunAndSequence {
            level: Level::L3,
            meta,
            next_sequence: SeqNo(42),
        };
        let payload = encode_entry(&entry).unwrap();
        let decoded = decode_entry(&payload).unwrap();
        assert_eq!(entry, decoded);
    }
```

This exercises the full encode→decode path at the maximum legal level, catching any off-by-one in `Level::new`'s bound check that the unit tests don't cover.

- [ ] **Step 6: Use tag constants in the regression tests**

In `crates/minkowski-lsm/tests/manifest_integration.rs`, update the imports at the top to include the tag constants:

```rust
use minkowski_lsm::manifest_log::{
    ManifestEntry, ManifestLog,
    TAG_ADD_RUN, TAG_REMOVE_RUN,
};
```

Wait — `pub(crate)` restricts visibility to the crate itself; `tests/` is a separate compilation unit. `pub(crate)` won't expose them externally. Two options:

**Option A**: Make them `pub` instead of `pub(crate)`. They're already cemented in the on-disk format, so external visibility is safe — but the API surface grows.

**Option B**: Define the constants in the test file directly, with a comment pointing at the canonical definition site. Less coupling.

Go with **Option B** to avoid growing the public API surface. At the top of `tests/manifest_integration.rs`:

```rust
// Tag bytes used by handcrafted-frame regression tests. Keep in sync with
// crates/minkowski-lsm/src/manifest_log.rs.
const TAG_ADD_RUN: u8 = 0x01;
const TAG_REMOVE_RUN: u8 = 0x02;
```

And revert the `pub(crate)` change from Step 4 (the constants stay module-private in `manifest_log.rs`).

Then in the regression tests (`replay_truncates_log_on_unsorted_coverage`, `replay_truncates_log_on_invalid_level_byte`, `replay_truncates_log_on_inverted_seq_range`), replace the hardcoded `0x01` / `0x02` with the named constants:

```rust
    // Before:
    // TAG_ADD_RUN = 0x01 (see manifest_log.rs::encode_entry AddRun branch)
    payload.push(0x01);

    // After:
    payload.push(TAG_ADD_RUN);
```

Drop the tag-comment since the constant name is self-documenting.

**Decision note**: this means future tag renumbering requires updating two places (the canonical definition and the test mirror). The sync comment flags this. Cross-check could be added in Task 4's future polish by round-tripping a real entry through `encode_entry` and asserting the first byte matches, but that's overkill for this pass.

Actually — on reflection, Option B (duplicating constants in the test file) is fragile because the test-file constants could drift from the production constants. Let me go with **Option A after all**: make the tag constants `pub` in `manifest_log.rs`. They're part of the on-disk format (documented in the module layout doc), so `pub` is appropriate — any future changes would be breaking anyway.

Revised instructions:

In `crates/minkowski-lsm/src/manifest_log.rs`:

```rust
pub const TAG_ADD_RUN: u8 = 0x01;
pub const TAG_REMOVE_RUN: u8 = 0x02;
pub const TAG_PROMOTE_RUN: u8 = 0x03;
pub const TAG_SET_SEQUENCE: u8 = 0x04;
pub const TAG_ADD_RUN_AND_SEQUENCE: u8 = 0x05;
```

In `crates/minkowski-lsm/tests/manifest_integration.rs` imports:

```rust
use minkowski_lsm::manifest_log::{
    ManifestEntry, ManifestLog, TAG_ADD_RUN, TAG_REMOVE_RUN,
};
```

And in the regression tests replace `0x01` / `0x02` with `TAG_ADD_RUN` / `TAG_REMOVE_RUN`.

- [ ] **Step 7: Add reserved-bytes forward-compat test (polish)**

Add to `tests/manifest_integration.rs`:

```rust
/// Reserved bytes in the header are documented as "ignored on read" for
/// forward-compat with future flags. Pin that behavior: a header with
/// non-zero reserved bytes followed by a valid frame must successfully
/// recover and apply the frame.
#[test]
fn recover_ignores_nonzero_reserved_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("reserved.log");

    // Write a v1 header with non-zero reserved bytes (bytes 5-7).
    fs::write(&log_path, b"MKMF\x01\xFF\xAA\x55").unwrap();

    // Recover should succeed on an otherwise-empty log.
    let (recovered, _) = ManifestLog::recover(&log_path).unwrap();
    assert_eq!(recovered.total_runs(), 0);
    assert_eq!(recovered.next_sequence(), SeqNo(0));
}
```

- [ ] **Step 8: Add zero-flush idempotency test (polish)**

Add to `tests/manifest_integration.rs`:

```rust
/// Calling recover() twice on the same path with no intervening writes
/// must produce identical state. Guards against a bug where re-opening
/// mutates the header or resets write_pos.
#[test]
fn recover_is_idempotent_with_no_flushes() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("idempotent.log");

    // First recover creates the file.
    let (manifest_a, log_a) = ManifestLog::recover(&log_path).unwrap();
    let bytes_after_first = fs::read(&log_path).unwrap();
    drop(log_a);

    // Second recover on the same path — no flushes between.
    let (manifest_b, log_b) = ManifestLog::recover(&log_path).unwrap();
    let bytes_after_second = fs::read(&log_path).unwrap();
    drop(log_b);

    assert_eq!(manifest_a.total_runs(), manifest_b.total_runs());
    assert_eq!(manifest_a.next_sequence(), manifest_b.next_sequence());
    assert_eq!(bytes_after_first, bytes_after_second);
}
```

- [ ] **Step 9: Run all tests**

Run: `cargo test -p minkowski-lsm`
Expected: 106 tests pass (103 + L3 round-trip + reserved-bytes + idempotency).

- [ ] **Step 10: Run clippy**

Run: `cargo clippy -p minkowski-lsm --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 11: Run workspace clippy (CI-equivalent)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add crates/minkowski-lsm/Cargo.toml \
        crates/minkowski-lsm/src/types.rs \
        crates/minkowski-lsm/src/manifest_log.rs \
        crates/minkowski-lsm/tests/manifest_integration.rs
git commit -m "test(lsm): tombstones, L3 round-trip, tag constants, polish tests

Five test-hardening items from the PR A + PR B1 review deferrals:

- SeqNo arithmetic tombstones: assert_not_impl_all!(SeqNo: Add, Sub,
  AddAssign, SubAssign) at the test-module level. Compile-time
  assertion that the 'identities not sizes' contract holds.
- L3 encode/decode round-trip: catches an off-by-one in Level::new's
  bound check at the full codec path (previously only covered at the
  constructor-level unit tests).
- Tag constants exposed as pub: TAG_ADD_RUN etc. are now imported in
  integration tests instead of hardcoded 0x01/0x02. Future tag
  renumbering is caught by the compiler, not by silent test-name drift.
- Reserved-bytes forward-compat test: pins the 'ignored on read'
  policy for non-zero reserved header bytes.
- Zero-flush idempotency test: recover -> drop -> recover on an
  untouched file must produce identical state.

Adds static_assertions as a dev-dependency for the tombstones."
```

---

## Task 5: Final verification + push + PR

**No code changes. Green-light gate.**

- [ ] **Step 1: Update local toolchain to match CI**

Run: `rustup update stable`
Expected: toolchain update or no-op.

- [ ] **Step 2: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace`
Expected: the 3 pre-existing `minkowski-observe` failures will still fail (they're pre-existing on main and CI doesn't run them — `cargo test -p minkowski` only). All other tests pass, including the 106 in `minkowski-lsm`.

- [ ] **Step 4: Run cargo fmt check**

Run: `cargo fmt --all -- --check`
Expected: clean.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin lsm/pr-b2-cleanup
gh pr create --title "chore(lsm): manifest cleanup polish (PR B2)" --body "$(cat <<'EOF'
## Summary

Six independently-queued cleanup items from PR A + PR B1 reviews. Pure quality-of-life improvements on the post-PR-B1 manifest subsystem.

Spec: memory file \`project_lsm_phase2_type_safety.md\`
Plan: \`docs/plans/2026-04-17-lsm-manifest-cleanup-polish-implementation-plan.md\`

## What landed

- **\`writer::flush\` + \`flush_and_record\` take \`SeqRange\`** instead of \`(u64, u64)\`. Closes the write-side slack where an invalid range succeeded at flush and only surfaced at next file open.
- **\`PageCount(NonZeroU64)\` newtype** replaces \`SortedRunMeta.page_count: u64\`. Collapses the runtime non-zero check into the type; niche optimization makes \`Option<PageCount>\` u64-sized.
- **\`archetype_coverage: Box<[u16]>\`** instead of \`Vec<u16>\`. Saves 8 bytes per meta, signals fixed-shape post-construction.
- **\`SeqNo\` arithmetic tombstones** — compile-time assertions that \`SeqNo\` doesn't implement \`Add\`/\`Sub\`/\`AddAssign\`/\`SubAssign\`.
- **L3 encode/decode round-trip test** — previously only covered at the constructor unit tests.
- **Tag constants exposed as \`pub\`** and imported in regression tests (replaces hardcoded \`0x01\` / \`0x02\`).
- **Two polish tests**: reserved-bytes forward-compat (pins 'ignored on read'), zero-flush idempotency (pins \`recover → drop → recover\` stability).

## Breaking changes

- \`writer::flush\` + \`flush_and_record\` signatures changed (internal-only; no external consumers of \`minkowski-lsm\` today).
- \`SortedRunMeta::page_count()\` now returns \`PageCount\` instead of \`u64\`. Callers that compared to a \`u64\` literal need \`.get()\`.

No wire format changes — encode/decode still reads/writes u64 bytes, PageCount wraps at the boundary.

## Tests

106 total in \`minkowski-lsm\` (up from 99):
- 4 new unit tests for \`PageCount\` (zero/valid/max/round-trip)
- 2 compile-time tombstone assertions for \`SeqNo\`
- 1 new encode/decode round-trip at \`Level::L3\`
- 2 polish integration tests (reserved-bytes, zero-flush idempotency)
- Regression tests migrated to named tag constants

## Test plan

- [x] \`cargo test -p minkowski-lsm\` — 106/106 pass
- [x] \`cargo clippy --workspace --all-targets -- -D warnings\` — clean
- [x] \`cargo fmt --all -- --check\` — clean
- [ ] CI pipeline (fmt, clippy, test, tsan, loom, claude-review)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Monitor CI; update memory after merge**

Once the PR merges, update:
- \`project_scaling_roadmap.md\`: note PR B2 landed.
- Delete or archive \`project_lsm_phase2_type_safety.md\` — scope fully consumed. Or repurpose for Phase 3 notes.

---

## Self-review (done inline before saving)

- **Spec coverage:**
  - Item 1 (writer::flush SeqRange) → Task 1
  - Item 2 (SeqNo arithmetic tombstones) → Task 4 Step 2
  - Item 3 (PageCount newtype) → Task 2
  - Item 4 (Box<[u16]> coverage) → Task 3
  - Item 5 (L3 round-trip test) → Task 4 Step 5
  - Item 6 (tag-constant coupling) → Task 4 Steps 4, 6 (revised to Option A mid-task)
  - Polish items (reserved bytes, idempotency) → Task 4 Steps 7, 8

- **Placeholder scan:** One instance of the plan pivoting mid-task (Step 6 of Task 4 debated Option A vs B, settled on A). The final instruction is Option A; the Option B prose is background reasoning the engineer can skim. Acceptable for transparency — the final step is concrete.

- **Type consistency:**
  - `PageCount::new(u64) -> Option<Self>` and `PageCount::get(self) -> u64` used consistently in Task 2.
  - `SortedRunMeta.page_count: PageCount` (field) vs `SortedRunMeta::new(..., page_count: u64, ...)` (parameter) — intentional: constructor accepts convenient u64, wraps internally.
  - `SeqRange` passed through `writer::flush` in Task 1 is the same `SeqRange` used by `SortedRunMeta` in Task 2's untouched context.

Self-review complete. One note-worthy area is the mid-task pivot in Task 4 Step 6 (Option A vs B) — the prose walks through the reasoning, which is fine for a subagent executing the plan since they see the decision rationale plus the final instruction.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-04-17-lsm-manifest-cleanup-polish-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Five tasks total; each is mechanical + testable in isolation.

**2. Inline Execution** — batch execution with checkpoints in this session.

Which approach?
