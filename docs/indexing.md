# Indexing

Indexes are user-owned data structures that compose with the query system — World has no awareness of them. The `SpatialIndex` trait has two methods: `rebuild` (full reconstruction) and `update` (incremental via `Changed<T>`). Stale entity references are caught automatically by generational ID validation.

For column-value lookups, `BTreeIndex<T>` provides O(log n) range queries and `HashIndex<T>` provides O(1) exact match. Both support validated queries (`get_valid`, `range_valid`) that filter despawned entities and removed components at query time.

When using `#[derive(Table)]`, field-level `#[index(btree)]` / `#[index(hash)]` attributes generate marker traits (`HasBTreeIndex<T>`, `HasHashIndex<T>`) that let `TablePlanner` enforce index presence at compile time. See [Schema & Mutation](../README.md#schema--mutation) for usage.

Indexes whose key types support [rkyv][rkyv] can be persisted to disk via `PersistentIndex`. On crash recovery, `load` + `update()` catches up from the saved tick — recovery time is proportional to the WAL tail, not world size. `AutoCheckpoint` can save registered indexes alongside snapshots automatically.

```rust
// Persistent index recovery
let post_restore_tick = world.change_tick();
wal.replay(&mut world, &codecs)?;

let mut idx = match load_btree_index::<Score>(&idx_path, post_restore_tick) {
    Ok(idx) => { idx.update(&mut world); idx }  // O(WAL tail)
    Err(_) => { let mut idx = BTreeIndex::new(); idx.rebuild(&mut world); idx }
};
```

Examples include a [uniform grid][uniform-grid] for neighbor search, a [Barnes-Hut][barnes-hut] quadtree for force approximation, and a persistent `BTreeIndex` in the [`persist`](../examples/examples/persist.rs) example.

<!-- Link definitions -->
[rkyv]: https://github.com/rkyv/rkyv
[uniform-grid]: https://en.wikipedia.org/wiki/Grid_(spatial_index)
[barnes-hut]: https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
