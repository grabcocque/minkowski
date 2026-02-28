# Chunk-Based Iteration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align BlobVec columns to 64 bytes and expose chunk-based iteration that yields typed column slices, enabling LLVM auto-vectorization of inner loops.

**Architecture:** BlobVec allocation alignment is raised to 64 bytes (cache line). A new `Slice` associated type and `as_slice` method on `WorldQuery` construct typed slices from the existing `ThinSlicePtr` base pointers. `QueryIter::for_each_chunk` yields one slice tuple per archetype. The boids example is rewritten to use the chunk API for all vectorizable loops.

**Tech Stack:** Rust, `std::alloc::Layout`, `std::slice::from_raw_parts[_mut]`, `cargo-show-asm`

---

### Task 1: BlobVec 64-byte alignment

Raise BlobVec allocation alignment from `item_layout.align()` to `max(item_layout.align(), 64)`.

**Files:**
- Modify: `crates/minkowski/src/storage/blob_vec.rs`

**Step 1: Add the alignment helper**

In `crates/minkowski/src/storage/blob_vec.rs`, add a constant and helper function at the top of the `impl BlobVec` block (after line 19):

```rust
/// Minimum allocation alignment for all BlobVec columns.
/// 64 bytes = cache line on x86-64 and Apple Silicon.
/// Satisfies AVX-512 (64), AVX2 (32), SSE (16) alignment requirements.
const MIN_COLUMN_ALIGN: usize = 64;

/// Compute the allocation alignment for a BlobVec column.
/// Always at least MIN_COLUMN_ALIGN to ensure cache-line alignment.
fn alloc_align(item: &Layout) -> usize {
    item.align().max(MIN_COLUMN_ALIGN)
}
```

**Step 2: Apply in `new()`**

In `BlobVec::new()`, change line 28 from:
```rust
Layout::from_size_align(item_layout.size() * capacity, item_layout.align())
```
to:
```rust
Layout::from_size_align(item_layout.size() * capacity, alloc_align(&item_layout))
```

**Step 3: Apply in `grow()`**

In `BlobVec::grow()`, change the `new_layout` (around line 161-165) from:
```rust
let new_layout = Layout::from_size_align(
    size.checked_mul(new_capacity).expect("capacity overflow"),
    self.item_layout.align(),
)
```
to:
```rust
let new_layout = Layout::from_size_align(
    size.checked_mul(new_capacity).expect("capacity overflow"),
    alloc_align(&self.item_layout),
)
```

And the `old_layout` (around line 170-171) from:
```rust
let old_layout =
    Layout::from_size_align(size * self.capacity, self.item_layout.align()).unwrap();
```
to:
```rust
let old_layout =
    Layout::from_size_align(size * self.capacity, alloc_align(&self.item_layout)).unwrap();
```

**Step 4: Apply in `Drop`**

In `Drop for BlobVec`, change the dealloc layout (around line 191-192) from:
```rust
let layout =
    Layout::from_size_align(size * self.capacity, self.item_layout.align()).unwrap();
```
to:
```rust
let layout =
    Layout::from_size_align(size * self.capacity, alloc_align(&self.item_layout)).unwrap();
```

**Step 5: Add alignment test**

Add to the test module in `blob_vec.rs`:

```rust
    #[test]
    fn column_base_is_64_byte_aligned() {
        // Test with various component sizes/alignments
        for &(size, align) in &[(4, 4), (8, 8), (1, 1), (12, 4), (32, 16)] {
            let layout = Layout::from_size_align(size, align).unwrap();
            let mut bv = BlobVec::new(layout, None, 8);
            unsafe {
                let mut val = vec![0u8; size];
                bv.push(val.as_mut_ptr());
            }
            let base = unsafe { bv.get_ptr(0) } as usize;
            assert_eq!(
                base % 64, 0,
                "BlobVec base not 64-byte aligned for size={size}, align={align}, base={base:#x}"
            );
        }
    }
```

**Step 6: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 7: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

**Step 8: Commit**

```bash
git add crates/minkowski/src/storage/blob_vec.rs
git commit -m "feat: align BlobVec columns to 64 bytes (cache line)

All column allocations now land on 64-byte boundaries, satisfying
cache line alignment, AVX-512/AVX2/SSE requirements, and preventing
false sharing in parallel iteration.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `Slice` associated type to WorldQuery

Extend the `WorldQuery` trait with `Slice<'w>` and `as_slice`, then implement for `&T`, `&mut T`, `Entity`, and `Option<&T>`.

**Files:**
- Modify: `crates/minkowski/src/query/fetch.rs`

**Step 1: Extend the trait**

In `crates/minkowski/src/query/fetch.rs`, add to the `WorldQuery` trait (after line 33, after `type Fetch<'w>`):

```rust
    /// The type yielded when accessing a whole archetype as a slice.
    type Slice<'w>;
```

And add after the `fetch` method (after line 47):

```rust
    /// Construct a typed slice over the entire column for this archetype.
    ///
    /// # Safety
    /// `len` must equal the archetype length. Caller must ensure no aliasing violations.
    unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> Self::Slice<'w>;
```

**Step 2: Implement for `&T`**

Add to the `&T` impl (after the `fetch` method, around line 76):

```rust
    type Slice<'w> = &'w [T];

    unsafe fn as_slice<'w>(fetch: &ThinSlicePtr<T>, len: usize) -> &'w [T] {
        std::slice::from_raw_parts(fetch.ptr as *const T, len)
    }
```

Wait — `type Slice<'w>` needs to be with the other associated types. Let me specify the full updated impl blocks.

**Full updated `&T` impl:**

Replace the entire `unsafe impl<T: Component> WorldQuery for &T` block (lines 51-77) with:

```rust
unsafe impl<T: Component> WorldQuery for &T {
    type Item<'w> = &'w T;
    type Fetch<'w> = ThinSlicePtr<T>;
    type Slice<'w> = &'w [T];

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        let mut bits = FixedBitSet::new();
        let id = match registry.id::<T>() {
            Some(id) => id,
            None => registry.len(),
        };
        bits.grow(id + 1);
        bits.insert(id);
        bits
    }

    fn init_fetch(archetype: &Archetype, registry: &ComponentRegistry) -> ThinSlicePtr<T> {
        let id = registry.id::<T>().expect("component not registered");
        let col_idx = archetype.component_index[&id];
        unsafe { ThinSlicePtr::new(archetype.columns[col_idx].get_ptr(0) as *mut T) }
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w T {
        &*fetch.ptr.add(row)
    }

    unsafe fn as_slice<'w>(fetch: &ThinSlicePtr<T>, len: usize) -> &'w [T] {
        std::slice::from_raw_parts(fetch.ptr as *const T, len)
    }
}
```

**Full updated `&mut T` impl:**

```rust
unsafe impl<T: Component> WorldQuery for &mut T {
    type Item<'w> = &'w mut T;
    type Fetch<'w> = ThinSlicePtr<T>;
    type Slice<'w> = &'w mut [T];

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
        <&T>::required_ids(registry)
    }

    fn init_fetch(archetype: &Archetype, registry: &ComponentRegistry) -> ThinSlicePtr<T> {
        <&T>::init_fetch(archetype, registry)
    }

    unsafe fn fetch<'w>(fetch: &ThinSlicePtr<T>, row: usize) -> &'w mut T {
        &mut *fetch.ptr.add(row)
    }

    unsafe fn as_slice<'w>(fetch: &ThinSlicePtr<T>, len: usize) -> &'w mut [T] {
        std::slice::from_raw_parts_mut(fetch.ptr, len)
    }
}
```

**Full updated `Entity` impl:**

```rust
unsafe impl WorldQuery for Entity {
    type Item<'w> = Entity;
    type Fetch<'w> = ThinSlicePtr<Entity>;
    type Slice<'w> = &'w [Entity];

    fn required_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new()
    }

    fn init_fetch(archetype: &Archetype, _registry: &ComponentRegistry) -> ThinSlicePtr<Entity> {
        unsafe { ThinSlicePtr::new(archetype.entities.as_ptr() as *mut Entity) }
    }

    unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        *fetch.ptr.add(row)
    }

    unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> &'w [Entity] {
        std::slice::from_raw_parts(fetch.ptr as *const Entity, len)
    }
}
```

**Full updated `Option<&T>` impl:**

```rust
unsafe impl<T: Component> WorldQuery for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type Fetch<'w> = Option<ThinSlicePtr<T>>;
    type Slice<'w> = Option<&'w [T]>;

    fn required_ids(_registry: &ComponentRegistry) -> FixedBitSet {
        FixedBitSet::new()
    }

    fn init_fetch(archetype: &Archetype, registry: &ComponentRegistry) -> Option<ThinSlicePtr<T>> {
        let id = registry.id::<T>()?;
        let col_idx = archetype.component_index.get(&id)?;
        Some(unsafe { ThinSlicePtr::new(archetype.columns[*col_idx].get_ptr(0) as *mut T) })
    }

    unsafe fn fetch<'w>(fetch: &Option<ThinSlicePtr<T>>, row: usize) -> Option<&'w T> {
        fetch.as_ref().map(|f| &*f.ptr.add(row))
    }

    unsafe fn as_slice<'w>(fetch: &Option<ThinSlicePtr<T>>, len: usize) -> Option<&'w [T]> {
        fetch.as_ref().map(|f| std::slice::from_raw_parts(f.ptr as *const T, len))
    }
}
```

**Step 3: Update the tuple macro**

Replace the `impl_world_query_tuple` macro (lines 136-163) with:

```rust
macro_rules! impl_world_query_tuple {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        unsafe impl<$($name: WorldQuery),*> WorldQuery for ($($name,)*) {
            type Item<'w> = ($($name::Item<'w>,)*);
            type Fetch<'w> = ($($name::Fetch<'w>,)*);
            type Slice<'w> = ($($name::Slice<'w>,)*);

            fn required_ids(registry: &ComponentRegistry) -> FixedBitSet {
                let mut bits = FixedBitSet::new();
                $(
                    let sub = $name::required_ids(registry);
                    bits.grow(sub.len());
                    bits.union_with(&sub);
                )*
                bits
            }

            fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Self::Fetch<'w> {
                ($($name::init_fetch(archetype, registry),)*)
            }

            unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WorldQuery>::fetch($name, row),)*)
            }

            unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
                let ($($name,)*) = fetch;
                ($(<$name as WorldQuery>::as_slice($name, len),)*)
            }
        }
    };
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All existing tests pass. The new `Slice` and `as_slice` compile but aren't called yet.

**Step 5: Commit**

```bash
git add crates/minkowski/src/query/fetch.rs
git commit -m "feat: add WorldQuery::Slice and as_slice for typed column slices

Extends the trait with Slice associated type and as_slice method.
Implemented for &T, &mut T, Entity, Option<&T>, and tuples 1-12.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `for_each_chunk` to QueryIter

**Files:**
- Modify: `crates/minkowski/src/query/iter.rs`

**Step 1: Add `for_each_chunk`**

In `crates/minkowski/src/query/iter.rs`, add to `impl<'w, Q: WorldQuery> QueryIter<'w, Q>` (after `par_for_each`, around line 35):

```rust
    /// Iterate archetypes, yielding typed column slices per archetype.
    /// Each invocation of `f` receives all matched rows in one archetype
    /// as contiguous slices — enabling SIMD auto-vectorization.
    pub fn for_each_chunk<F>(self, mut f: F)
    where
        F: FnMut(Q::Slice<'w>),
    {
        for (fetch, len) in &self.fetches {
            if *len > 0 {
                let slices = unsafe { Q::as_slice(fetch, *len) };
                f(slices);
            }
        }
    }
```

**Step 2: Write tests**

Add to the test module in `iter.rs`:

```rust
    #[test]
    fn for_each_chunk_yields_correct_data() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 },));
        world.spawn((Pos { x: 3.0, y: 0.0 },));

        let mut total = 0.0f32;
        world.query::<&Pos>().for_each_chunk(|positions| {
            for p in positions {
                total += p.x;
            }
        });
        assert_eq!(total, 6.0);
    }

    #[test]
    fn for_each_chunk_mutation() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 10.0, dy: 0.0 }));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 20.0, dy: 0.0 }));

        world.query::<(&mut Pos, &Vel)>()
            .for_each_chunk(|(positions, velocities)| {
                for i in 0..positions.len() {
                    positions[i].x += velocities[i].dx;
                }
            });

        let xs: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        assert_eq!(xs, vec![11.0, 22.0]);
    }

    #[test]
    fn for_each_chunk_skips_empty() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.despawn(e);

        let mut chunk_count = 0;
        world.query::<&Pos>().for_each_chunk(|_| {
            chunk_count += 1;
        });
        assert_eq!(chunk_count, 0);
    }

    #[test]
    fn for_each_chunk_multiple_archetypes() {
        let mut world = World::new();
        // Archetype 1: Pos only
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        // Archetype 2: Pos + Vel
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));

        let mut chunk_count = 0;
        let mut total = 0.0f32;
        world.query::<&Pos>().for_each_chunk(|positions| {
            chunk_count += 1;
            for p in positions {
                total += p.x;
            }
        });
        assert_eq!(chunk_count, 2); // one per archetype
        assert_eq!(total, 3.0);
    }
```

**Step 3: Run tests**

Run: `cargo test -p minkowski --lib -- for_each_chunk`
Expected: 4 new tests PASS.

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/minkowski/src/query/iter.rs
git commit -m "feat: add QueryIter::for_each_chunk for SIMD-friendly iteration

Yields typed column slices per archetype. LLVM sees contiguous loops
over 64-byte-aligned memory, enabling auto-vectorization.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Rewrite boids to use chunk iteration

**Files:**
- Modify: `crates/minkowski/examples/boids.rs`

**Step 1: Rewrite vectorizable loops**

In `crates/minkowski/examples/boids.rs`, replace the three vectorizable loops in the frame loop.

Replace the "Step 1: Zero accelerations" loop (line 181-183):
```rust
        // Step 1: Zero accelerations
        for acc in world.query::<&mut Acceleration>() {
            acc.0 = Vec2::ZERO;
        }
```
with:
```rust
        // Step 1: Zero accelerations (chunk — enables vectorization)
        world.query::<&mut Acceleration>()
            .for_each_chunk(|accs| {
                for acc in accs.iter_mut() {
                    acc.0 = Vec2::ZERO;
                }
            });
```

Replace the "Step 5: Integration" loops (lines 254-262):
```rust
        for (vel, acc) in world.query::<(&mut Velocity, &Acceleration)>() {
            vel.0 += acc.0 * DT;
            vel.0 = vel.0.clamped(params.max_speed);
        }
        for (pos, vel) in world.query::<(&mut Position, &Velocity)>() {
            pos.0 += vel.0 * DT;
            pos.0.x = pos.0.x.rem_euclid(params.world_size);
            pos.0.y = pos.0.y.rem_euclid(params.world_size);
        }
```
with:
```rust
        world.query::<(&mut Velocity, &Acceleration)>()
            .for_each_chunk(|(vels, accs)| {
                for i in 0..vels.len() {
                    vels[i].0.x += accs[i].0.x * DT;
                    vels[i].0.y += accs[i].0.y * DT;
                    vels[i].0 = vels[i].0.clamped(params.max_speed);
                }
            });
        world.query::<(&mut Position, &Velocity)>()
            .for_each_chunk(|(poss, vels)| {
                for i in 0..poss.len() {
                    poss[i].0.x = (poss[i].0.x + vels[i].0.x * DT).rem_euclid(params.world_size);
                    poss[i].0.y = (poss[i].0.y + vels[i].0.y * DT).rem_euclid(params.world_size);
                }
            });
```

**Step 2: Run the example**

Run: `cargo run -p minkowski --example boids --release 2>&1 | tail -5`
Expected: Completes 1000 frames with reasonable stats.

**Step 3: Commit**

```bash
git add crates/minkowski/examples/boids.rs
git commit -m "perf: rewrite boids integration loops with for_each_chunk

Vectorizable loops now use chunk iteration, giving LLVM contiguous
aligned slices for auto-vectorization.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Assembly verification and documentation

**Files:**
- Create: `.cargo/config.toml`
- Modify: `CLAUDE.md`

**Step 1: Add cargo config for target-cpu=native**

Create `.cargo/config.toml`:

```toml
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=native"]

[target.'cfg(target_arch = "aarch64")']
rustflags = ["-C", "target-cpu=native"]
```

**Step 2: Verify vectorized assembly**

Run: `cargo build -p minkowski --example boids --release 2>&1`
Expected: Builds successfully.

Then inspect the binary for SIMD instructions:

Run: `objdump -d target/release/examples/boids | grep -c -E 'vaddps|vmulps|addps|mulps' 2>/dev/null || echo "check manually"`

This is a rough check — presence of packed float ops (`vaddps`, `vmulps` for AVX, `addps`, `mulps` for SSE) indicates auto-vectorization. If `cargo-show-asm` is installed, a more targeted check is possible.

**Step 3: Add alignment docs to CLAUDE.md**

Add after the "Query Caching" section in CLAUDE.md:

```markdown
### Column Alignment & Vectorization

BlobVec columns are allocated with 64-byte alignment (cache line). `QueryIter::for_each_chunk` yields typed `&[T]` / `&mut [T]` slices per archetype — LLVM can auto-vectorize loops over these slices.

Component types that are 16-byte-aligned (e.g., `#[repr(align(16))]` or naturally `[f32; 4]`) vectorize better than odd-sized ones. The engine guarantees 64-byte column alignment; component layout determines whether LLVM can pack operations.

Build with `-C target-cpu=native` (configured in `.cargo/config.toml`) to enable platform-specific SIMD instructions.
```

**Step 4: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean.

Run: `cargo run -p minkowski --example boids --release 2>&1 | tail -5`
Expected: Completes successfully.

**Step 5: Commit**

```bash
git add .cargo/config.toml CLAUDE.md
git commit -m "docs: add target-cpu=native config and alignment documentation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final verification (Miri)

**Step 1: Miri**

Run: `MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p minkowski --lib`
Expected: All tests pass.

**Step 2: Commit if any fixes needed**

If all clean, no commit needed.
