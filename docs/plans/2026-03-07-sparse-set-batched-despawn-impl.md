# Paged Sparse Set + Batched Despawn Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace HashMap-based sparse storage with a paged sparse set (BlobVec-backed), and add batched despawn with back-to-front sweep.

**Architecture:** Two independent optimizations sharing one PR. The sparse set is a new `PagedSparseSet` struct in `storage/sparse.rs` that replaces `HashMap<Entity, T>` with paged indices + dense BlobVec. Batched despawn adds `drop_in_place`/`copy_unchecked` to BlobVec and `World::despawn_batch` using group-sort-sweep. Both are `pub(crate)` except `despawn_batch` which is `pub`.

**Tech Stack:** Rust, no new dependencies.

---

### Task 1: PagedSparseSet — Core Data Structure

**Files:**
- Rewrite: `crates/minkowski/src/storage/sparse.rs`

**Step 1: Write the failing tests**

Add these tests at the bottom of sparse.rs inside a new `#[cfg(test)] mod tests` block. Remove the old tests entirely.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entity;
    use std::alloc::Layout;

    fn make_set<T: 'static>() -> PagedSparseSet {
        let drop_fn = if std::mem::needs_drop::<T>() {
            Some(unsafe { |ptr: *mut u8| std::ptr::drop_in_place(ptr as *mut T) } as unsafe fn(*mut u8))
        } else {
            None
        };
        PagedSparseSet::new(Layout::new::<T>(), drop_fn)
    }

    #[test]
    fn empty_set() {
        let set = make_set::<u32>();
        assert_eq!(set.len(), 0);
        let e = Entity::new(0, 0);
        assert!(!set.contains(e));
        assert!(set.get(e).is_none());
    }

    #[test]
    fn insert_and_get() {
        let mut set = make_set::<u32>();
        let e = Entity::new(0, 0);
        let val = 42u32;
        unsafe { set.insert(e, &val as *const u32 as *const u8); }
        assert_eq!(set.len(), 1);
        assert!(set.contains(e));
        let ptr = set.get(e).unwrap();
        assert_eq!(unsafe { *(ptr as *const u32) }, 42);
    }

    #[test]
    fn cross_page_lookup() {
        let mut set = make_set::<u32>();
        let e0 = Entity::new(0, 0);
        let e_far = Entity::new(5000, 0); // different page
        let v0 = 10u32;
        let v1 = 20u32;
        unsafe {
            set.insert(e0, &v0 as *const u32 as *const u8);
            set.insert(e_far, &v1 as *const u32 as *const u8);
        }
        assert_eq!(set.len(), 2);
        assert_eq!(unsafe { *(set.get(e0).unwrap() as *const u32) }, 10);
        assert_eq!(unsafe { *(set.get(e_far).unwrap() as *const u32) }, 20);
    }

    #[test]
    fn generation_rejection() {
        let mut set = make_set::<u32>();
        let e_gen0 = Entity::new(5, 0);
        let e_gen1 = Entity::new(5, 1); // same index, different generation
        let val = 99u32;
        unsafe { set.insert(e_gen0, &val as *const u32 as *const u8); }
        assert!(set.contains(e_gen0));
        assert!(!set.contains(e_gen1)); // must reject
        assert!(set.get(e_gen1).is_none());
    }

    #[test]
    fn remove_and_compact() {
        let mut set = make_set::<u32>();
        let e0 = Entity::new(0, 0);
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);
        let (v0, v1, v2) = (10u32, 20u32, 30u32);
        unsafe {
            set.insert(e0, &v0 as *const u32 as *const u8);
            set.insert(e1, &v1 as *const u32 as *const u8);
            set.insert(e2, &v2 as *const u32 as *const u8);
        }
        assert!(set.remove(e1)); // remove middle — e2 should swap into its slot
        assert_eq!(set.len(), 2);
        assert!(!set.contains(e1));
        // e0 and e2 still accessible
        assert_eq!(unsafe { *(set.get(e0).unwrap() as *const u32) }, 10);
        assert_eq!(unsafe { *(set.get(e2).unwrap() as *const u32) }, 30);
    }

    #[test]
    fn remove_absent_returns_false() {
        let mut set = make_set::<u32>();
        let e = Entity::new(0, 0);
        assert!(!set.remove(e));
    }

    #[test]
    fn iteration() {
        let mut set = make_set::<u32>();
        let entities: Vec<Entity> = (0..5).map(|i| Entity::new(i, 0)).collect();
        for (i, &e) in entities.iter().enumerate() {
            let val = (i as u32) * 10;
            unsafe { set.insert(e, &val as *const u32 as *const u8); }
        }
        let mut found: Vec<(Entity, u32)> = Vec::new();
        set.iter(|entity, ptr| {
            found.push((entity, unsafe { *(ptr as *const u32) }));
        });
        assert_eq!(found.len(), 5);
        // All entities present (order may differ from insertion)
        for &e in &entities {
            assert!(found.iter().any(|(fe, _)| *fe == e));
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib sparse`
Expected: Compilation errors — `PagedSparseSet` doesn't exist yet.

**Step 3: Implement PagedSparseSet**

Replace the entire contents of `crates/minkowski/src/storage/sparse.rs` (keep the `#[cfg(test)]` block from step 1):

```rust
use std::alloc::Layout;
use std::collections::HashMap;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::storage::blob_vec::BlobVec;

const PAGE_SIZE: usize = 4096;
const EMPTY: u32 = u32::MAX;

/// A paged sparse set backed by BlobVec for dense value storage.
///
/// Sparse pages map entity indices to dense-array positions. Dense arrays
/// store Entity handles (for generation checks) and component values
/// (type-erased in BlobVec). O(1) get/insert/remove, contiguous iteration.
pub(crate) struct PagedSparseSet {
    pages: Vec<Option<Box<[u32; PAGE_SIZE]>>>,
    dense_entities: Vec<Entity>,
    dense_values: BlobVec,
}

impl PagedSparseSet {
    pub fn new(item_layout: Layout, drop_fn: Option<unsafe fn(*mut u8)>) -> Self {
        Self {
            pages: Vec::new(),
            dense_entities: Vec::new(),
            dense_values: BlobVec::new(item_layout, drop_fn, 0),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dense_entities.len()
    }

    /// Look up the dense index for an entity, returning None if absent or stale.
    #[inline]
    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let idx = entity.index() as usize;
        let page_idx = idx / PAGE_SIZE;
        let slot = idx % PAGE_SIZE;
        let page = self.pages.get(page_idx)?.as_ref()?;
        let dense_idx = page[slot];
        if dense_idx == EMPTY {
            return None;
        }
        let dense_idx = dense_idx as usize;
        // Generation check — reject stale entries
        if self.dense_entities[dense_idx] != entity {
            return None;
        }
        Some(dense_idx)
    }

    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.dense_index(entity).is_some()
    }

    /// Returns a raw pointer to the component value, or None.
    #[inline]
    pub fn get(&self, entity: Entity) -> Option<*mut u8> {
        let dense_idx = self.dense_index(entity)?;
        Some(unsafe { self.dense_values.get_ptr(dense_idx) })
    }

    /// Returns a mutable raw pointer to the component value, or None.
    #[inline]
    pub fn get_mut(&mut self, entity: Entity) -> Option<*mut u8> {
        let dense_idx = self.dense_index(entity)?;
        Some(unsafe { self.dense_values.get_ptr(dense_idx) })
    }

    /// Insert a component value for an entity. Copies `layout.size()` bytes from `ptr`.
    ///
    /// If the entity already has a value, it is overwritten (old value is dropped).
    ///
    /// # Safety
    /// `ptr` must point to a valid, initialized value matching this set's layout.
    pub unsafe fn insert(&mut self, entity: Entity, ptr: *const u8) {
        if let Some(dense_idx) = self.dense_index(entity) {
            // Overwrite existing — drop old, copy new
            let dst = self.dense_values.get_ptr(dense_idx);
            if let Some(drop_fn) = self.dense_values.drop_fn {
                drop_fn(dst);
            }
            let size = self.dense_values.item_layout.size();
            if size > 0 {
                std::ptr::copy_nonoverlapping(ptr, dst, size);
            }
            return;
        }

        // New entry
        let idx = entity.index() as usize;
        let page_idx = idx / PAGE_SIZE;
        let slot = idx % PAGE_SIZE;

        // Ensure page exists
        if page_idx >= self.pages.len() {
            self.pages.resize_with(page_idx + 1, || None);
        }
        if self.pages[page_idx].is_none() {
            self.pages[page_idx] = Some(Box::new([EMPTY; PAGE_SIZE]));
        }

        let dense_idx = self.dense_entities.len() as u32;
        self.pages[page_idx].as_mut().unwrap()[slot] = dense_idx;
        self.dense_entities.push(entity);
        self.dense_values.push(ptr as *mut u8);
    }

    /// Remove the component for an entity. Returns true if it was present.
    pub fn remove(&mut self, entity: Entity) -> bool {
        let dense_idx = match self.dense_index(entity) {
            Some(idx) => idx,
            None => return false,
        };

        let last = self.dense_entities.len() - 1;

        // Clear the removed entity's page entry
        let idx = entity.index() as usize;
        self.pages[idx / PAGE_SIZE].as_mut().unwrap()[idx % PAGE_SIZE] = EMPTY;

        if dense_idx != last {
            // Swap-remove: move last into the gap
            let last_entity = self.dense_entities[last];
            // Update the swapped entity's page entry
            let last_idx = last_entity.index() as usize;
            self.pages[last_idx / PAGE_SIZE].as_mut().unwrap()[last_idx % PAGE_SIZE] =
                dense_idx as u32;
            self.dense_entities.swap_remove(dense_idx);
            unsafe {
                self.dense_values.swap_remove(dense_idx);
            }
        } else {
            // Removing the last element — just pop
            self.dense_entities.pop();
            unsafe {
                self.dense_values.swap_remove(dense_idx);
            }
        }

        true
    }

    /// Iterate over all entries, calling `f(entity, value_ptr)` for each.
    pub fn iter(&self, mut f: impl FnMut(Entity, *mut u8)) {
        for (i, &entity) in self.dense_entities.iter().enumerate() {
            let ptr = unsafe { self.dense_values.get_ptr(i) };
            f(entity, ptr);
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib sparse`
Expected: All 7 new tests pass.

**Step 5: Commit**

```bash
git add crates/minkowski/src/storage/sparse.rs
git commit -m "feat: PagedSparseSet with BlobVec dense storage"
```

---

### Task 2: SparseStorage — Typed Wrapper Without dyn Any

**Files:**
- Modify: `crates/minkowski/src/storage/sparse.rs` (add `SparseStorage` below `PagedSparseSet`)

**Step 1: Write the failing tests**

Append to the test module in sparse.rs:

```rust
    // ── SparseStorage tests ──

    #[test]
    fn sparse_storage_insert_and_get() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert::<Marker>(0, e, Marker(42));
        assert_eq!(storage.get::<Marker>(0, e), Some(&Marker(42)));
    }

    #[test]
    fn sparse_storage_get_missing() {
        let storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn sparse_storage_remove() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert::<Marker>(0, e, Marker(42));
        let removed = storage.remove::<Marker>(0, e);
        assert_eq!(removed, Some(Marker(42)));
        assert_eq!(storage.get::<Marker>(0, e), None);
    }

    #[test]
    fn sparse_storage_component_ids_and_iter() {
        let mut storage = SparseStorage::new();
        let e1 = Entity::new(0, 0);
        let e2 = Entity::new(1, 0);
        storage.insert::<u32>(0, e1, 42u32);
        storage.insert::<u32>(0, e2, 99u32);

        let ids = storage.component_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 0);

        let entries: Vec<_> = storage.iter::<u32>(0).unwrap().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn sparse_storage_remove_all() {
        let mut storage = SparseStorage::new();
        let e = Entity::new(0, 0);
        storage.insert::<u32>(0, e, 42u32);
        storage.insert::<Marker>(1, e, Marker(99));
        storage.remove_all(e);
        assert_eq!(storage.get::<u32>(0, e), None);
        assert_eq!(storage.get::<Marker>(1, e), None);
    }

    #[derive(Debug, PartialEq)]
    struct Marker(u32);
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib sparse`
Expected: Compilation errors — `SparseStorage` doesn't exist yet.

**Step 3: Implement SparseStorage**

Add below `PagedSparseSet` in sparse.rs:

```rust
/// Container for all sparse component sets. One `PagedSparseSet` per
/// registered sparse component type. No `dyn Any` — all sets are the same
/// concrete type (type erasure is in BlobVec).
pub(crate) struct SparseStorage {
    storages: HashMap<ComponentId, PagedSparseSet>,
}

#[allow(dead_code)]
impl SparseStorage {
    pub fn new() -> Self {
        Self {
            storages: HashMap::new(),
        }
    }

    pub fn insert<T: Component>(&mut self, comp_id: ComponentId, entity: Entity, value: T) {
        let set = self.storages.entry(comp_id).or_insert_with(|| {
            let drop_fn = if std::mem::needs_drop::<T>() {
                Some(drop_ptr::<T> as unsafe fn(*mut u8))
            } else {
                None
            };
            PagedSparseSet::new(Layout::new::<T>(), drop_fn)
        });
        let val = std::mem::ManuallyDrop::new(value);
        unsafe {
            set.insert(entity, &*val as *const T as *const u8);
        }
    }

    pub fn get<T: Component>(&self, comp_id: ComponentId, entity: Entity) -> Option<&T> {
        let set = self.storages.get(&comp_id)?;
        let ptr = set.get(entity)?;
        Some(unsafe { &*(ptr as *const T) })
    }

    pub fn get_mut<T: Component>(
        &mut self,
        comp_id: ComponentId,
        entity: Entity,
    ) -> Option<&mut T> {
        let set = self.storages.get_mut(&comp_id)?;
        let ptr = set.get_mut(entity)?;
        Some(unsafe { &mut *(ptr as *mut T) })
    }

    pub fn contains<T: Component>(&self, comp_id: ComponentId, entity: Entity) -> bool {
        self.storages
            .get(&comp_id)
            .is_some_and(|set| set.contains(entity))
    }

    pub fn remove<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<T> {
        let set = self.storages.get_mut(&comp_id)?;
        let ptr = set.get(entity)?;
        let value = unsafe { std::ptr::read(ptr as *const T) };
        // Remove without drop — we already read the value out
        // We need to do this carefully: get the dense_index, clear page,
        // swap_remove_no_drop equivalent
        // Actually, remove() calls swap_remove which drops. We need to
        // prevent the double drop. Read first, then remove.
        // BlobVec::swap_remove will call drop_fn on the slot, but we already
        // moved the value out. We need a different approach.
        //
        // Solution: use get() to read, then we need swap_remove_no_drop.
        // But PagedSparseSet::remove() uses BlobVec::swap_remove which drops.
        // We need a remove_no_drop variant.
        drop(value); // Actually, let's rethink this.

        // The correct approach: we can't safely extract the value through
        // the type-erased BlobVec without a remove_no_drop path.
        // Let's add a remove_no_drop to PagedSparseSet and use it here.
        todo!("need remove_no_drop on PagedSparseSet")
    }

    /// Remove entity from ALL sparse sets. O(1) per set.
    pub fn remove_all(&mut self, entity: Entity) {
        for set in self.storages.values_mut() {
            set.remove(entity);
        }
    }

    pub fn component_ids(&self) -> Vec<ComponentId> {
        self.storages.keys().copied().collect()
    }

    pub fn iter<T: Component>(
        &self,
        comp_id: ComponentId,
    ) -> Option<impl Iterator<Item = (Entity, &T)>> {
        let set = self.storages.get(&comp_id)?;
        let mut entries = Vec::new();
        set.iter(|entity, ptr| {
            entries.push((entity, unsafe { &*(ptr as *const T) }));
        });
        Some(entries.into_iter())
    }
}

unsafe fn drop_ptr<T>(ptr: *mut u8) {
    std::ptr::drop_in_place(ptr as *mut T);
}
```

Wait — the `remove<T>` method has a problem. We need to extract the value without BlobVec dropping it. Add a `remove_no_drop` method to `PagedSparseSet` that does the swap-remove bookkeeping (page updates, entity swap) but uses `swap_remove_no_drop` on the BlobVec instead of `swap_remove`.

Add to `PagedSparseSet`:

```rust
    /// Remove without dropping the value. Caller must have already
    /// read/moved the data. Returns true if the entity was present.
    pub fn remove_no_drop(&mut self, entity: Entity) -> bool {
        let dense_idx = match self.dense_index(entity) {
            Some(idx) => idx,
            None => return false,
        };

        let last = self.dense_entities.len() - 1;
        let idx = entity.index() as usize;
        self.pages[idx / PAGE_SIZE].as_mut().unwrap()[idx % PAGE_SIZE] = EMPTY;

        if dense_idx != last {
            let last_entity = self.dense_entities[last];
            let last_idx = last_entity.index() as usize;
            self.pages[last_idx / PAGE_SIZE].as_mut().unwrap()[last_idx % PAGE_SIZE] =
                dense_idx as u32;
            self.dense_entities.swap_remove(dense_idx);
            unsafe { self.dense_values.swap_remove_no_drop(dense_idx); }
        } else {
            self.dense_entities.pop();
            unsafe { self.dense_values.swap_remove_no_drop(dense_idx); }
        }

        true
    }
```

Then fix `SparseStorage::remove<T>`:

```rust
    pub fn remove<T: Component>(&mut self, comp_id: ComponentId, entity: Entity) -> Option<T> {
        let set = self.storages.get_mut(&comp_id)?;
        let ptr = set.get(entity)?;
        let value = unsafe { std::ptr::read(ptr as *const T) };
        set.remove_no_drop(entity);
        Some(value)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib sparse`
Expected: All 12 tests pass (7 PagedSparseSet + 5 SparseStorage).

**Step 5: Commit**

```bash
git add crates/minkowski/src/storage/sparse.rs
git commit -m "feat: SparseStorage wrapper — typed API without dyn Any"
```

---

### Task 3: Wire SparseStorage Into World

**Files:**
- Modify: `crates/minkowski/src/world.rs`

The World already uses `SparseStorage` — the struct name and method signatures (`get`, `get_mut`, `insert`, `remove`, `contains`, `iter`, `component_ids`) are preserved. The only changes:

**Step 1: Verify existing tests pass**

Run: `cargo test -p minkowski --lib`
Expected: If SparseStorage's API matches, all existing tests should compile and pass without changes to world.rs. If there are signature mismatches, fix them.

Known difference: the old `SparseStorage::contains` took a type parameter `T: Component`. The new one also does. Same for all methods. The `remove` return type is the same (`Option<T>`). The only potential issue is if world.rs uses any method not on the new `SparseStorage`.

**Step 2: Add sparse cleanup to World::despawn**

In `world.rs`, after the existing despawn logic (after `self.entities.dealloc(entity)` on line 329), add:

```rust
self.sparse.remove_all(entity);
```

**Step 3: Write test for sparse cleanup on despawn**

Add to the world.rs test module:

```rust
#[test]
fn despawn_cleans_sparse_components() {
    let mut world = World::new();
    let comp_id = world.components.register_sparse::<f32>();
    let entity = world.spawn((42u32,));
    world.sparse.insert::<f32>(comp_id, entity, 3.14);
    assert!(world.sparse.contains::<f32>(comp_id, entity));
    world.despawn(entity);
    assert!(!world.sparse.contains::<f32>(comp_id, entity));
}
```

**Step 4: Run tests**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass including the new one.

**Step 5: Commit**

```bash
git add crates/minkowski/src/world.rs crates/minkowski/src/storage/sparse.rs
git commit -m "feat: wire PagedSparseSet into World, eager sparse cleanup on despawn"
```

---

### Task 4: BlobVec — drop_in_place and copy_unchecked

**Files:**
- Modify: `crates/minkowski/src/storage/blob_vec.rs`

**Step 1: Write the failing tests**

Add to the blob_vec test module:

```rust
    #[test]
    fn drop_in_place_calls_drop_fn() {
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        #[allow(dead_code)]
        struct Tracked(u32);
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        let mut bv = bv_for::<Tracked>();
        unsafe {
            push_val(&mut bv, Tracked(1));
            push_val(&mut bv, Tracked(2));
            push_val(&mut bv, Tracked(3));
            bv.drop_in_place(1); // drop middle element
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
        // len unchanged — slot is logically uninitialized
        assert_eq!(bv.len(), 3);

        // Manually clean up: set len to skip the dropped slot
        // (in real usage, caller manages len)
        // For this test, just leak the rest to avoid double-drop
        bv.drop_fn = None; // prevent BlobVec::drop from re-dropping
        // remaining items would be dropped by BlobVec::drop normally
    }

    #[test]
    fn copy_unchecked_moves_data() {
        let mut bv = bv_for::<u32>();
        unsafe {
            push_val(&mut bv, 10u32);
            push_val(&mut bv, 20u32);
            push_val(&mut bv, 30u32);
            bv.copy_unchecked(2, 0); // copy row 2 into row 0
            assert_eq!(read_val::<u32>(&bv, 0), 30);
            assert_eq!(read_val::<u32>(&bv, 1), 20);
            assert_eq!(read_val::<u32>(&bv, 2), 30); // src still has data (bitwise copy)
        }
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib blob_vec`
Expected: Compilation errors — methods don't exist.

**Step 3: Implement the two methods**

Add to `impl BlobVec` in blob_vec.rs:

```rust
    /// Drop the element at `row` in place without moving anything.
    /// The slot becomes logically uninitialized — caller must not read it
    /// or must overwrite it before any future access.
    ///
    /// # Safety
    /// `row` must be in bounds (`row < len`). Caller must ensure the slot
    /// is not accessed again without being reinitialized.
    pub unsafe fn drop_in_place(&mut self, row: usize) {
        debug_assert!(row < self.len);
        if let Some(drop_fn) = self.drop_fn {
            drop_fn(self.ptr_at(row));
        }
    }

    /// Copy element from `src_row` to `dst_row` without dropping either.
    /// Bitwise copy — no drop on dst (must be uninitialized or already dropped),
    /// no drop on src (caller ensures it won't be accessed again).
    ///
    /// # Safety
    /// Both rows must be in bounds. `dst_row` must be uninitialized or already
    /// dropped. `src_row` must not be read again without reinitializing.
    pub unsafe fn copy_unchecked(&mut self, src_row: usize, dst_row: usize) {
        debug_assert!(src_row < self.len);
        debug_assert!(dst_row < self.len);
        let size = self.item_layout.size();
        if size > 0 {
            let src = self.ptr_at(src_row);
            let dst = self.ptr_at(dst_row);
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }
```

Also need to make `drop_fn` and `item_layout` accessible within the crate. They're currently private fields. Check if `PagedSparseSet::insert` needs `drop_fn` access — yes, it reads `self.dense_values.drop_fn`. Make them `pub(crate)`:

```rust
pub(crate) struct BlobVec {
    pub(crate) item_layout: Layout,
    pub(crate) drop_fn: Option<unsafe fn(*mut u8)>,
    // ... rest stays the same
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib blob_vec`
Expected: All tests pass including the 2 new ones.

**Step 5: Also add `set_len` for batch truncation**

Add to `impl BlobVec`:

```rust
    /// Set the length directly. Caller must ensure all elements in
    /// `new_len..old_len` have been dropped or moved out.
    ///
    /// # Safety
    /// `new_len` must be <= current len. Elements beyond new_len must be
    /// already dropped/moved.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.len);
        self.len = new_len;
    }
```

**Step 6: Commit**

```bash
git add crates/minkowski/src/storage/blob_vec.rs
git commit -m "feat: BlobVec::drop_in_place, copy_unchecked, set_len for batched despawn"
```

---

### Task 5: World::despawn_batch

**Files:**
- Modify: `crates/minkowski/src/world.rs`

**Step 1: Write the failing tests**

Add to the world.rs test module:

```rust
#[test]
fn despawn_batch_multiple_same_archetype() {
    let mut world = World::new();
    let a = world.spawn((1u32,));
    let b = world.spawn((2u32,));
    let c = world.spawn((3u32,));
    let d = world.spawn((4u32,));

    let count = world.despawn_batch(&[b, d]);
    assert_eq!(count, 2);
    assert!(!world.is_alive(b));
    assert!(!world.is_alive(d));
    assert!(world.is_alive(a));
    assert!(world.is_alive(c));
    assert_eq!(*world.get::<u32>(a).unwrap(), 1);
    assert_eq!(*world.get::<u32>(c).unwrap(), 3);
}

#[test]
fn despawn_batch_multiple_archetypes() {
    let mut world = World::new();
    let a = world.spawn((1u32,));
    let b = world.spawn((1u32, 1.0f32));
    let c = world.spawn((2u32,));
    let d = world.spawn((2u32, 2.0f32));

    let count = world.despawn_batch(&[a, d]);
    assert_eq!(count, 2);
    assert!(!world.is_alive(a));
    assert!(!world.is_alive(d));
    assert!(world.is_alive(b));
    assert!(world.is_alive(c));
}

#[test]
fn despawn_batch_skips_dead() {
    let mut world = World::new();
    let a = world.spawn((1u32,));
    let b = world.spawn((2u32,));
    world.despawn(a);

    let count = world.despawn_batch(&[a, b]);
    assert_eq!(count, 1); // a was already dead
    assert!(!world.is_alive(b));
}

#[test]
fn despawn_batch_single_entity() {
    let mut world = World::new();
    let a = world.spawn((1u32,));
    let count = world.despawn_batch(&[a]);
    assert_eq!(count, 1);
    assert!(!world.is_alive(a));
}

#[test]
fn despawn_batch_empty() {
    let mut world = World::new();
    let count = world.despawn_batch(&[]);
    assert_eq!(count, 0);
}

#[test]
fn despawn_batch_cleans_sparse() {
    let mut world = World::new();
    let comp_id = world.components.register_sparse::<f32>();
    let a = world.spawn((1u32,));
    let b = world.spawn((2u32,));
    world.sparse.insert::<f32>(comp_id, a, 3.14);
    // b has no sparse component

    let count = world.despawn_batch(&[a, b]);
    assert_eq!(count, 2);
    assert!(!world.sparse.contains::<f32>(comp_id, a));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p minkowski --lib despawn_batch`
Expected: Compilation error — `despawn_batch` doesn't exist.

**Step 3: Implement World::despawn_batch**

Add to `impl World` in world.rs:

```rust
    /// Despawn multiple entities efficiently. Groups by archetype, sorts rows
    /// descending, sweeps back-to-front. Returns the number of entities
    /// actually despawned (dead/unplaced entities are skipped).
    pub fn despawn_batch(&mut self, entities: &[Entity]) -> usize {
        self.drain_orphans();

        // Phase 1: Filter to alive+placed entities, group by archetype
        let mut by_archetype: HashMap<usize, Vec<(usize, Entity)>> = HashMap::new();
        for &entity in entities {
            if !self.entities.is_alive(entity) {
                continue;
            }
            let index = entity.index() as usize;
            let location = match self.entity_locations[index] {
                Some(loc) => loc,
                None => continue,
            };
            by_archetype
                .entry(location.archetype_id.0)
                .or_default()
                .push((location.row, entity));
        }

        let mut count = 0;

        // Phase 2: For each archetype, sort rows descending and sweep
        for (arch_idx, mut row_entities) in by_archetype {
            // Sort by row descending — process from back to front
            row_entities.sort_unstable_by(|a, b| b.0.cmp(&a.0));

            let archetype = &mut self.archetypes.archetypes[arch_idx];

            for &(row, entity) in &row_entities {
                let last = archetype.entities.len() - 1;

                // Drop component data at this row
                for col in &mut archetype.columns {
                    unsafe { col.drop_in_place(row); }
                }

                // If not the last row, copy last element into the gap
                if row < last {
                    for col in &mut archetype.columns {
                        unsafe { col.copy_unchecked(last, row); }
                    }
                    // Patch the moved entity's location
                    let moved_entity = archetype.entities[last];
                    archetype.entities[row] = moved_entity;
                    self.entity_locations[moved_entity.index() as usize] =
                        Some(EntityLocation {
                            archetype_id: ArchetypeId(arch_idx),
                            row,
                        });
                }

                // Truncate
                archetype.entities.truncate(last);
                for col in &mut archetype.columns {
                    unsafe { col.set_len(last); }
                }

                // Dealloc entity
                self.entity_locations[entity.index() as usize] = None;
                self.entities.dealloc(entity);
                count += 1;
            }
        }

        // Phase 3: Clean sparse storage
        for &entity in entities {
            // remove_all is a no-op for dead entities (generation mismatch)
            // but we already deallocated above, so we need to remove BEFORE dealloc.
            // Wait — we already deallocated the entity in the loop above.
            // remove_all uses PagedSparseSet::dense_index which checks generation.
            // After dealloc, the generation is bumped, so the old entity handle
            // won't match. We need to do sparse cleanup BEFORE dealloc.
        }

        count
    }
```

**Wait — ordering issue.** We dealloc entities (bumping generation) inside the archetype loop, but sparse `remove_all` checks generation. We need to do sparse cleanup before dealloc. Restructure:

```rust
    pub fn despawn_batch(&mut self, entities: &[Entity]) -> usize {
        self.drain_orphans();

        // Phase 1: Filter to alive+placed entities, group by archetype
        let mut by_archetype: HashMap<usize, Vec<(usize, Entity)>> = HashMap::new();
        let mut to_dealloc: Vec<Entity> = Vec::new();

        for &entity in entities {
            if !self.entities.is_alive(entity) {
                continue;
            }
            let index = entity.index() as usize;
            let location = match self.entity_locations[index] {
                Some(loc) => loc,
                None => continue,
            };
            by_archetype
                .entry(location.archetype_id.0)
                .or_default()
                .push((location.row, entity));
            to_dealloc.push(entity);
        }

        // Phase 2: Sparse cleanup (before dealloc bumps generations)
        for &entity in &to_dealloc {
            self.sparse.remove_all(entity);
        }

        // Phase 3: For each archetype, sort rows descending and sweep
        for (arch_idx, mut row_entities) in by_archetype {
            row_entities.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let archetype = &mut self.archetypes.archetypes[arch_idx];

            for &(row, _entity) in &row_entities {
                let last = archetype.entities.len() - 1;

                for col in &mut archetype.columns {
                    unsafe { col.drop_in_place(row); }
                }

                if row < last {
                    for col in &mut archetype.columns {
                        unsafe { col.copy_unchecked(last, row); }
                    }
                    let moved_entity = archetype.entities[last];
                    archetype.entities[row] = moved_entity;
                    self.entity_locations[moved_entity.index() as usize] =
                        Some(EntityLocation {
                            archetype_id: ArchetypeId(arch_idx),
                            row,
                        });
                }

                archetype.entities.truncate(last);
                for col in &mut archetype.columns {
                    unsafe { col.set_len(last); }
                }
            }
        }

        // Phase 4: Dealloc entities and clear locations
        for &entity in &to_dealloc {
            self.entity_locations[entity.index() as usize] = None;
            self.entities.dealloc(entity);
        }

        to_dealloc.len()
    }
```

You'll need to add `use std::collections::HashMap;` at the top of world.rs if not already present, and ensure `ArchetypeId` is in scope.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p minkowski --lib despawn_batch`
Expected: All 6 tests pass.

**Step 5: Run full test suite**

Run: `cargo test -p minkowski --lib`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add crates/minkowski/src/world.rs
git commit -m "feat: World::despawn_batch — group-sort-sweep with sparse cleanup"
```

---

### Task 6: EnumChangeSet::apply — Batch Despawns

**Files:**
- Modify: `crates/minkowski/src/changeset.rs`

**Step 1: Write the failing test**

Add to the changeset.rs test module:

```rust
#[test]
fn apply_batch_despawns_produces_correct_reverse() {
    let mut world = World::new();
    let a = world.spawn((10u32,));
    let b = world.spawn((20u32,));
    let c = world.spawn((30u32,));

    let mut cs = EnumChangeSet::new();
    cs.record_despawn(a);
    cs.record_despawn(c);

    let reverse = cs.apply(&mut world);

    // a and c are gone
    assert!(!world.is_alive(a));
    assert!(world.is_alive(b));
    assert!(!world.is_alive(c));
    assert_eq!(*world.get::<u32>(b).unwrap(), 20);

    // Applying reverse should re-spawn a and c
    let _re_reverse = reverse.apply(&mut world);
    // Note: re-spawned entities may have different generations
    // but the world should have 3 entities total
    let mut count = 0;
    world.query::<(&u32,)>().for_each(|(_val,)| count += 1);
    assert_eq!(count, 3);
}
```

**Step 2: Run test to verify it fails or passes**

Run: `cargo test -p minkowski --lib apply_batch_despawns`

This test should work with the current sequential apply too. The optimization is internal — the test validates behavioral equivalence.

**Step 3: Modify EnumChangeSet::apply to batch despawns**

In `changeset.rs`, modify the `apply` method. Partition despawn mutations out, capture their data upfront, then call `world.despawn_batch`:

```rust
    pub fn apply(mut self, world: &mut World) -> EnumChangeSet {
        self.drop_entries.clear();
        let mut reverse = EnumChangeSet::new();

        // Partition: separate despawns from other mutations
        let mut despawn_entities: Vec<Entity> = Vec::new();
        let mut other_mutations: Vec<&Mutation> = Vec::new();

        for mutation in &self.mutations {
            match mutation {
                Mutation::Despawn { entity } => {
                    despawn_entities.push(*entity);
                }
                other => {
                    other_mutations.push(other);
                }
            }
        }

        // Phase 1: Capture all despawn component data for reverse
        for &entity in &despawn_entities {
            let comp_data = world.read_all_components(entity).unwrap_or_default();
            reverse.record_spawn(entity, &comp_data);
        }

        // Phase 2: Batch despawn
        if !despawn_entities.is_empty() {
            world.despawn_batch(&despawn_entities);
        }

        // Phase 3: Apply remaining mutations in order
        for mutation in other_mutations {
            match mutation {
                Mutation::Spawn { entity, components } => {
                    // ... existing spawn logic ...
                }
                Mutation::Insert { entity, component_id, offset, layout } => {
                    let data_ptr = self.arena.get(*offset);
                    changeset_insert_raw(world, &mut reverse, *entity, *component_id, data_ptr, *layout);
                }
                Mutation::Remove { entity, component_id } => {
                    changeset_remove_raw(world, &mut reverse, *entity, *component_id);
                }
                Mutation::Despawn { .. } => unreachable!(),
            }
        }

        reverse
    }
```

Copy the existing `Mutation::Spawn` handling from the current code (lines 432-481) into the spawn arm. The key change is just extracting despawns into a batch.

**Step 4: Run full changeset tests**

Run: `cargo test -p minkowski --lib changeset`
Expected: All changeset tests pass, including existing reverse/undo tests.

**Step 5: Commit**

```bash
git add crates/minkowski/src/changeset.rs
git commit -m "feat: EnumChangeSet::apply batches despawns via World::despawn_batch"
```

---

### Task 7: Clippy + Format + Full Test Suite

**Files:** All modified files

**Step 1: Format**

Run: `cargo fmt --all`

**Step 2: Clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Fix any warnings.

**Step 3: Full test suite**

Run: `cargo test -p minkowski`
Expected: All tests pass (unit + doc tests).

**Step 4: Run examples**

Run: `cargo run -p minkowski-examples --example tactical --release`
This example uses sparse components — confirms the new storage works end-to-end.

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: clippy + fmt fixes"
```

---

### Task 8: PR

Use the `/pr` skill to create the pull request. Title: "Paged sparse set + batched despawn (#45)". The PR should reference the design doc.
