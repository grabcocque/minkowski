# CI Status Checks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a GitHub Actions CI workflow that gates PRs on fmt, clippy, tests, and Miri.

**Architecture:** Single workflow with four sequential jobs chained via `needs:`. Uses `dtolnay/rust-toolchain` for toolchain management and `Swatinem/rust-cache` for caching. Miri runs two invocations matching the CLAUDE.md commands.

**Tech Stack:** GitHub Actions, Rust stable + nightly, Miri

---

### Task 1: Create CI workflow file

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write the workflow file**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  fmt:
    name: Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  clippy:
    name: Clippy
    needs: fmt
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: stable
      - run: cargo clippy --workspace --all-targets -- -D warnings

  test:
    name: Test
    needs: clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: stable
      - run: cargo test -p minkowski

  miri:
    name: Miri
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri, rust-src
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: nightly
      - name: Miri tests (skip par_for_each)
        run: cargo +nightly miri test -p minkowski --lib -- --skip par_for_each
        env:
          MIRIFLAGS: "-Zmiri-tree-borrows"
      - name: Miri par_for_each (ignore leaks)
        run: cargo +nightly miri test -p minkowski --lib par_for_each
        env:
          MIRIFLAGS: "-Zmiri-tree-borrows -Zmiri-ignore-leaks"
```

**Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: No output (valid YAML)

If `python3` not available, use: `cat .github/workflows/ci.yml | head -1` (manual visual check)

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add fmt/clippy/test/miri status checks"
```

### Task 2: Update CLAUDE.md with CI section

**Files:**
- Modify: `CLAUDE.md` (add CI section after Build & Test Commands)

**Step 1: Add CI section after the pre-commit hooks line (line 27)**

Insert after the `Pre-commit hooks run...` line:

```markdown

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every PR and push to main:

| Job | Toolchain | Command | `needs` |
|---|---|---|---|
| fmt | stable | `cargo fmt --all -- --check` | — |
| clippy | stable | `cargo clippy --workspace --all-targets -- -D warnings` | fmt |
| test | stable | `cargo test -p minkowski` | clippy |
| miri | nightly | Two-step Miri run (see Build & Test Commands) | test |

Sequential chain: fmt failure skips all downstream jobs. The `miri` job is the required status check for branch protection (implicitly requires all upstream jobs).
```

**Step 2: Verify CLAUDE.md is well-formed**

Run: `head -35 CLAUDE.md`
Expected: New CI section visible after Build & Test Commands

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CI section to CLAUDE.md"
```

### Task 3: Push and create PR

**Step 1: Create branch, push, and open PR**

```bash
git checkout -b ci/status-checks
git push -u origin ci/status-checks
gh pr create --title "ci: add fmt/clippy/test/miri status checks" --body "..."
```

PR body should summarize the four-job sequential pipeline and note that branch protection should be configured to require the `miri` job after merge.

**Step 2: After PR is merged, configure branch protection**

Manual step (GitHub UI): Settings → Branches → main → Require status checks → Add `miri`.
