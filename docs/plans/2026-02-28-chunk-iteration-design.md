# Chunk-Based Iteration with 64-Byte Aligned Columns

## Problem

The current query system yields one item at a time via `WorldQuery::fetch(ptr, row)`. LLVM never sees a contiguous slice — it can't auto-vectorize. BlobVec allocates with the component's own alignment (e.g., 4 for f32), which misses cache line boundaries and SIMD alignment requirements.

## Scope

Three changes, one validation:
1. **BlobVec 64-byte alignment** — all column allocations land on cache line boundaries
2. **Chunk iteration API** — `WorldQuery::Slice` + `for_each_chunk` yields typed column slices per archetype
3. **Boids rewrite** — rewrite vectorizable loops to use `for_each_chunk`
4. **Assembly verification** — compile with `-C target-cpu=native`, inspect for packed float ops

Hot/cold metadata separation is deferred to a follow-up.

## Design

### BlobVec 64-byte Alignment

Add a helper function:
```rust
fn alloc_align(item: &Layout) -> usize {
    item.align().max(64)
}
```

Apply it in `BlobVec::new()`, `grow()`, and `Drop`. Every BlobVec column starts on a 64-byte boundary. This satisfies:
- Cache line alignment (64 bytes on x86-64 and Apple Silicon)
- AVX-512 (64-byte), AVX2 (32-byte), SSE (16-byte) alignment requirements
- Prevents false sharing in parallel iteration

Cost: up to 63 bytes wasted per column. With hundreds of archetypes × ~5 columns each, this is <50KB total — negligible.

### WorldQuery Trait Extension

Add `Slice` associated type and `as_slice` method:

```rust
pub unsafe trait WorldQuery {
    type Item<'w>;
    type Fetch<'w>: Send + Sync;
    type Slice<'w>;

    fn required_ids(registry: &ComponentRegistry) -> FixedBitSet;
    fn init_fetch<'w>(archetype: &'w Archetype, registry: &ComponentRegistry) -> Self::Fetch<'w>;
    unsafe fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w>;

    /// Construct a typed slice view over the entire column for this archetype.
    ///
    /// # Safety
    /// `len` must equal the archetype length. Caller must ensure no aliasing.
    unsafe fn as_slice<'w>(fetch: &Self::Fetch<'w>, len: usize) -> Self::Slice<'w>;
}
```

Slice type mappings:

| Query | `Slice<'w>` | `as_slice` implementation |
|---|---|---|
| `&T` | `&'w [T]` | `from_raw_parts(ptr, len)` |
| `&mut T` | `&'w mut [T]` | `from_raw_parts_mut(ptr, len)` |
| `Entity` | `&'w [Entity]` | `from_raw_parts(ptr, len)` |
| `Option<&T>` | `Option<&'w [T]>` | `Some(slice)` if present, `None` if absent |
| `(A, B, ...)` | `(A::Slice, B::Slice, ...)` | Tuple of each element's slice |

The base pointer comes from `ThinSlicePtr<T>` which is already stored in `Fetch`. `as_slice` just adds a length.

### for_each_chunk on QueryIter

```rust
impl<'w, Q: WorldQuery> QueryIter<'w, Q> {
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
}
```

Each archetype is one chunk. The closure receives typed slices covering all matched entities in that archetype. LLVM sees a contiguous loop over aligned memory — ideal for auto-vectorization.

A parallel variant is also possible:

```rust
pub fn par_for_each_chunk<F>(self, f: F)
where
    F: Fn(Q::Slice<'_>) + Send + Sync,
{
    // Each archetype chunk processed in parallel
    self.fetches.par_iter().for_each(|(fetch, len)| {
        if *len > 0 {
            let slices = unsafe { Q::as_slice(fetch, *len) };
            f(slices);
        }
    });
}
```

### Boids Rewrite

Loops that benefit from `for_each_chunk`:

**Zero accelerations:**
```rust
world.query::<&mut Acceleration>()
    .for_each_chunk(|accs| {
        for acc in accs.iter_mut() {
            acc.0 = Vec2::ZERO;
        }
    });
```

**Integration (vel += acc * dt):**
```rust
world.query::<(&mut Velocity, &Acceleration)>()
    .for_each_chunk(|(vels, accs)| {
        for i in 0..vels.len() {
            vels[i].0.x += accs[i].0.x * DT;
            vels[i].0.y += accs[i].0.y * DT;
            // clamp inline
        }
    });
```

**Integration (pos += vel * dt):**
```rust
world.query::<(&mut Position, &Velocity)>()
    .for_each_chunk(|(poss, vels)| {
        for i in 0..poss.len() {
            poss[i].0.x = (poss[i].0.x + vels[i].0.x * DT).rem_euclid(WORLD_SIZE);
            poss[i].0.y = (poss[i].0.y + vels[i].0.y * DT).rem_euclid(WORLD_SIZE);
        }
    });
```

The N² neighbor search stays as a snapshot + rayon — it's inherently random-access.

### Assembly Verification

Add `.cargo/config.toml`:
```toml
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=native"]
```

Build and inspect with `cargo-show-asm`:
```
cargo asm -p minkowski --example boids --release
```

What to look for in the integration loop:
- **Vectorized**: `vmovaps`, `vaddps`, `vmulps` (AVX2) or `addps`, `mulps` (SSE)
- **Not vectorized**: `addss`, `mulss` (scalar single)

### Alignment Documentation

Add to CLAUDE.md and README.md a note: component types that are 16-byte aligned (e.g., `#[repr(align(16))]` or naturally `[f32; 4]`) vectorize better than odd-sized ones. The engine guarantees 64-byte column alignment; component layout determines whether LLVM can pack operations.

### Testing

1. **BlobVec alignment**: assert `get_ptr(0)` is 64-byte aligned for various types
2. **Chunk iteration correctness**: `for_each_chunk` yields same data as per-element iteration
3. **Chunk mutation**: modify via chunk slice, verify via per-element read
4. **Empty archetype chunk**: `for_each_chunk` skips empty archetypes
5. **Multi-archetype chunk**: verify separate chunks per archetype
6. **Boids regression**: example still produces reasonable output

### Files

- Modify: `crates/minkowski/src/storage/blob_vec.rs` — 64-byte alignment
- Modify: `crates/minkowski/src/query/fetch.rs` — `Slice` type + `as_slice` for all impls + tuple macro
- Modify: `crates/minkowski/src/query/iter.rs` — `for_each_chunk` method
- Modify: `crates/minkowski/examples/boids.rs` — rewrite vectorizable loops
- Create: `.cargo/config.toml` — `target-cpu=native`
- Modify: `CLAUDE.md` — alignment documentation
