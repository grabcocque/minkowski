# CI Status Checks Design

## Goal

Add a GitHub Actions CI workflow that enforces fmt, clippy, tests, and Miri on every PR, providing the same checks that local pre-commit hooks run but as an enforceable gate.

## Approach

Single workflow (`.github/workflows/ci.yml`) with four sequential jobs chained via `needs:`. Sequential ordering gives fast-fail: a fmt failure short-circuits without burning minutes on Miri.

## Workflow Structure

**Triggers**: `pull_request` (all branches) + `push` to `main`.

**Jobs** (sequential chain):

1. **fmt** (~15s) — `cargo fmt --all -- --check`. Stable toolchain, no cache needed.
2. **clippy** (~45s, `needs: fmt`) — `cargo clippy --workspace --all-targets -- -D warnings`. Stable toolchain + clippy component, `Swatinem/rust-cache` with `shared-key: stable`.
3. **test** (~30s, `needs: clippy`) — `cargo test -p minkowski`. Stable toolchain, shared cache.
4. **miri** (~2-3min, `needs: test`) — Two-step run matching CLAUDE.md invocations:
   - `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p minkowski --lib -- --skip par_for_each`
   - `MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib par_for_each`
   - Nightly toolchain with miri + rust-src components, separate cache (`shared-key: nightly`).

**Toolchain management**: `dtolnay/rust-toolchain` action. Stable for jobs 1-3, nightly for miri.

**Caching**: `Swatinem/rust-cache` with per-toolchain shared keys so clippy and test reuse the same compiled artifacts.

## Branch Protection

After merge, configure GitHub branch protection on `main` to require the `miri` job (terminal job — implicitly requires all upstream jobs to pass).

## CLAUDE.md Update

Add a CI section documenting the workflow and its relationship to local pre-commit hooks.

## What Stays the Same

- `.pre-commit-config.yaml` — local hooks for fast developer feedback
- `.cargo/config.toml` — build flags
- Existing Claude Code workflows — orthogonal (code review, @claude mentions)
