//! Memory pool allocator trait and implementations.
//!
//! The pool allocator is the single backing allocator for all internal
//! data structures (BlobVec, Arena, entity tables, sparse pages).
//! Two implementations: `SystemAllocator` (current behavior, default)
//! and `SlabPool` (TigerBeetle-style fixed budget with mmap).

use std::alloc::Layout;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Error returned when the memory pool is exhausted.
#[derive(Debug, Clone)]
pub struct PoolExhausted {
    pub requested: Layout,
}

impl fmt::Display for PoolExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "memory pool exhausted: requested {} bytes (align {})",
            self.requested.size(),
            self.requested.align()
        )
    }
}

impl std::error::Error for PoolExhausted {}

/// Hugepage configuration for the memory pool.
#[derive(Clone, Copy, Debug, Default)]
pub enum HugePages {
    /// Attempt 2 MiB hugepages, fall back to 4 KiB pages silently.
    #[default]
    Try,
    /// Require hugepages — fail if unavailable.
    Require,
    /// Use regular 4 KiB pages only.
    Off,
}

/// Backing allocator for all internal ECS data structures.
///
/// # Safety
///
/// Implementations must return properly aligned, non-overlapping memory
/// regions. `deallocate` must only be called with pointers and layouts
/// previously returned by `allocate`.
#[allow(dead_code)]
pub unsafe trait PoolAllocator: Send + Sync {
    /// Allocate a block satisfying `layout`.
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted>;

    /// Return a block to the pool.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to `allocate` with a
    /// compatible layout. The caller must not use `ptr` after this call.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Total capacity in bytes, if bounded. `None` for unbounded allocators.
    fn capacity(&self) -> Option<usize> {
        None
    }

    /// Bytes currently allocated. `None` if not tracked.
    fn used(&self) -> Option<usize> {
        None
    }
}

/// Default allocator -- delegates to `std::alloc`. Unbounded, panics on OOM.
/// This is the allocator used by `World::new()`.
pub(crate) struct SystemAllocator;

// SAFETY: SystemAllocator delegates to the global allocator which returns
// properly aligned, non-overlapping memory regions. Zero-size allocations
// return dangling aligned pointers (never passed to `std::alloc::dealloc`).
unsafe impl PoolAllocator for SystemAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            // Zero-size: return dangling aligned pointer.
            // `layout.align()` is guaranteed non-zero by Layout invariants.
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }
        // SAFETY: layout has non-zero size (checked above).
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).ok_or_else(|| {
            // Preserve current behavior: panic on OOM for system allocator.
            std::alloc::handle_alloc_error(layout)
        })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // SAFETY: caller guarantees `ptr` was returned by a prior `allocate`
        // with a compatible layout, and `layout.size() > 0`.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

/// Shared handle to a pool allocator. Cheaply cloneable.
#[allow(dead_code)]
pub(crate) type SharedPool = Arc<dyn PoolAllocator>;

/// Create the default (system) allocator as a shared pool.
#[allow(dead_code)]
pub(crate) fn default_pool() -> SharedPool {
    Arc::new(SystemAllocator)
}

// ── mlockall ────────────────────────────────────────────────────────

/// Call `mlockall(MCL_CURRENT | MCL_FUTURE)` to lock all current and future
/// memory mappings into physical RAM. This is a process-global operation.
///
/// Best-effort: silently ignored on platforms without `mlockall` or if the
/// call fails (insufficient privileges).
#[cfg(unix)]
pub(crate) fn mlockall_best_effort() {
    // SAFETY: mlockall is a process-global operation with no memory
    // unsafety. It may fail due to insufficient RLIMIT_MEMLOCK.
    unsafe {
        libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
    }
}

#[cfg(not(unix))]
pub(crate) fn mlockall_best_effort() {
    // No-op on non-Unix platforms.
}

// ── MmapRegion ──────────────────────────────────────────────────────

/// Memory region backed by an anonymous mmap with optional hugepage support.
///
/// Uses `memmap2` for cross-platform mmap. Pages are demand-paged by the OS;
/// the first access to each page triggers a soft fault.
struct MmapRegion {
    mmap: memmap2::MmapMut,
    #[allow(dead_code)]
    huge: bool,
}

impl MmapRegion {
    fn new(size: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        use memmap2::MmapOptions;
        let layout = Layout::from_size_align(size, 1).expect("valid layout");

        // Try hugepages first if requested.
        if matches!(hugepages, HugePages::Try | HugePages::Require) {
            let result = MmapOptions::new()
                .len(size)
                .populate() // MAP_POPULATE — pre-fault with hugepages
                .huge(Some(21)) // 2 MiB hugepages (2^21)
                .map_anon();

            match result {
                Ok(mmap) => return Ok(Self { mmap, huge: true }),
                Err(_) if matches!(hugepages, HugePages::Try) => {
                    // Fall through to regular pages.
                }
                Err(_) => return Err(PoolExhausted { requested: layout }),
            }
        }

        // Regular pages — pre-fault via fallback chain.
        // Pre-faulting is NOT optional: we must know at startup whether
        // the system can back the mapping with physical RAM. Three paths,
        // tried in order of preference:
        //
        // 1. MAP_POPULATE — kernel pre-faults pages at mmap time.
        //    Fastest on native Linux, but can hang/fail on WSL2 or
        //    older kernels.
        //
        // 2. mlock — forces pages into physical RAM after mmap.
        //    Works on most systems but requires sufficient RLIMIT_MEMLOCK.
        //
        // 3. Manual touch sweep — write one byte per page (4 KB stride).
        //    Always works, no privilege requirements. Slowest but guaranteed.
        let mut mmap = MmapOptions::new()
            .len(size)
            .populate() // Attempt 1: MAP_POPULATE
            .map_anon()
            .or_else(|_| {
                // populate() failed — try without it, then mlock/touch.
                MmapOptions::new().len(size).map_anon()
            })
            .map_err(|_| PoolExhausted { requested: layout })?;

        // If MAP_POPULATE succeeded, pages are already faulted. If it
        // didn't (we fell through to the plain mmap), we need to force
        // pages into RAM via mlock or manual touch.
        if mmap.lock().is_err() {
            // mlock failed (insufficient RLIMIT_MEMLOCK or platform
            // doesn't support it) — manual touch sweep as final fallback.
            Self::prefault_manual(&mut mmap);
        }

        Ok(Self { mmap, huge: false })
    }

    /// Write one byte per page to force the kernel to back the mapping
    /// with physical memory. This is the last-resort pre-fault path.
    fn prefault_manual(mmap: &mut memmap2::MmapMut) {
        const PAGE_SIZE: usize = 4096;
        let ptr = mmap.as_mut_ptr();
        let len = mmap.len();
        for offset in (0..len).step_by(PAGE_SIZE) {
            // Safety: offset < len, and len == mmap size.
            unsafe { ptr.add(offset).write_volatile(0) };
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.mmap.len()
    }
}

// ── SlabPool ────────────────────────────────────────────────────────

const NUM_SIZE_CLASSES: usize = 6;
const SIZE_CLASSES: [usize; NUM_SIZE_CLASSES] = [64, 256, 1024, 4096, 65_536, 1_048_576];

/// Proportional split tuned for ECS patterns (percentages of total budget):
/// - Class 0 (64 B):   1% — small components
/// - Class 1 (256 B):  2% — medium components
/// - Class 2 (1 KB):   4% — sparse pages, small columns
/// - Class 3 (4 KB):  20% — OS page, column growth
/// - Class 4 (64 KB): 40% — large column segments
/// - Class 5 (1 MB):  33% — bulk column pre-allocation
const PROPORTIONS: [usize; NUM_SIZE_CLASSES] = [1, 2, 4, 20, 40, 33];

/// Intrusive free-list node. Stored at the start of each free block.
/// Every size class is >= 64 bytes, so `FreeBlock` (pointer-sized) always fits.
#[repr(C)]
struct FreeBlock {
    next: *mut FreeBlock,
}

/// Return the size-class index for a given layout, or `None` if too large.
fn size_class_for(layout: Layout) -> Option<usize> {
    // The effective allocation size must satisfy both size and alignment.
    let size = layout.size().max(layout.align());
    SIZE_CLASSES.iter().position(|&s| s >= size)
}

/// TigerBeetle-style slab allocator backed by a single mmap'd region.
///
/// Memory is partitioned into fixed-size blocks across six size classes.
/// Allocation and deallocation are lock-free (CAS on per-class free lists).
pub(crate) struct SlabPool {
    region: MmapRegion,
    free_lists: [AtomicPtr<FreeBlock>; NUM_SIZE_CLASSES],
    used_bytes: AtomicUsize,
}

// SAFETY: All mutable state is behind atomics. FreeBlock pointers point
// into the mmap region which is not deallocated while SlabPool exists.
// The mmap region is owned by SlabPool and outlives all allocations.
unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    /// Create a new slab pool backed by an mmap region of `budget` bytes.
    pub(crate) fn new(budget: usize, hugepages: HugePages) -> Result<Self, PoolExhausted> {
        let mut region = MmapRegion::new(budget, hugepages)?;
        let base = region.as_mut_ptr();
        let total = region.len();

        // Initialize free lists as null.
        let free_lists = std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut()));

        // Partition the region into size-class blocks proportionally.
        let proportion_sum: usize = PROPORTIONS.iter().sum();
        let mut offset: usize = 0;

        for class in 0..NUM_SIZE_CLASSES {
            let block_size = SIZE_CLASSES[class];
            // Bytes allocated to this size class (rounded down to whole blocks).
            let class_bytes = total * PROPORTIONS[class] / proportion_sum;
            let block_count = class_bytes / block_size;

            // Build free list from last block to first so that the first
            // allocation pops the lowest address (better cache locality).
            let mut head: *mut FreeBlock = std::ptr::null_mut();
            for i in (0..block_count).rev() {
                let block_offset = offset + i * block_size;
                assert!(
                    block_offset + block_size <= total,
                    "block at offset {block_offset} (size {block_size}) exceeds region of {total} bytes"
                );
                // SAFETY: `block_offset + block_size <= total` (asserted above),
                // and `base` points to the start of a valid mmap region of
                // `total` bytes. Each block is at least 64 bytes, large enough
                // for a FreeBlock (pointer-sized).
                let block_ptr = unsafe { base.add(block_offset).cast::<FreeBlock>() };
                // SAFETY: `block_ptr` is properly aligned (64-byte blocks are
                // >= pointer alignment) and within the mmap region. We are the
                // sole writer during construction.
                unsafe { (*block_ptr).next = head };
                head = block_ptr;
            }

            // Single-threaded init — Relaxed is sufficient.
            free_lists[class].store(head, Ordering::Relaxed);
            offset += block_count * block_size;
        }

        Ok(Self {
            region,
            free_lists,
            used_bytes: AtomicUsize::new(0),
        })
    }
}

// SAFETY: SlabPool returns properly aligned, non-overlapping memory regions.
// Each block is a fixed-size slab from the mmap region. Blocks are at least
// 64 bytes (cache-line aligned), satisfying any layout whose alignment <= 64.
// Size classes are chosen such that `layout.size() <= block_size`. The free
// list is lock-free (CAS) and each block is returned to exactly one caller.
unsafe impl PoolAllocator for SlabPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, PoolExhausted> {
        if layout.size() == 0 {
            return Ok(NonNull::new(layout.align() as *mut u8).expect("alignment is non-zero"));
        }

        let class = size_class_for(layout).ok_or(PoolExhausted { requested: layout })?;

        // Lock-free CAS pop from free list.
        loop {
            let head = self.free_lists[class].load(Ordering::Acquire);
            if head.is_null() {
                return Err(PoolExhausted { requested: layout });
            }
            // SAFETY: `head` is non-null and points to a FreeBlock within
            // the mmap region. It was either written during init or pushed
            // back by a prior `deallocate`. No other thread can pop this
            // same block because we haven't CAS'd it out yet — if another
            // thread pops it first, our CAS will fail and we retry.
            let next = unsafe { (*head).next };
            if self.free_lists[class]
                .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.used_bytes
                    .fetch_add(SIZE_CLASSES[class], Ordering::Relaxed);
                return Ok(NonNull::new(head.cast()).expect("free list block is non-null"));
            }
            // CAS failed — another thread popped concurrently. Retry.
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        let Some(class) = size_class_for(layout) else {
            return;
        };

        let block = ptr.as_ptr().cast::<FreeBlock>();

        // Lock-free CAS push to free list.
        loop {
            let head = self.free_lists[class].load(Ordering::Acquire);
            // SAFETY: `block` was returned by a prior `allocate` from this
            // size class. It points within the mmap region and is at least
            // 64 bytes — large enough for FreeBlock. The caller guarantees
            // no other references exist.
            unsafe { (*block).next = head };
            if self.free_lists[class]
                .compare_exchange_weak(head, block, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.used_bytes
                    .fetch_sub(SIZE_CLASSES[class], Ordering::Relaxed);
                return;
            }
            // CAS failed — another thread pushed concurrently. Retry.
        }
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.region.len())
    }

    fn used(&self) -> Option<usize> {
        Some(self.used_bytes.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_allocator_allocate_and_deallocate() {
        let alloc = SystemAllocator;
        let layout = Layout::from_size_align(256, 64).unwrap();
        let ptr = alloc.allocate(layout).unwrap();
        // SAFETY: ptr points to 256 bytes of allocated memory.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0xAB, 256) };
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn system_allocator_zero_size() {
        let alloc = SystemAllocator;
        let layout = Layout::from_size_align(0, 1).unwrap();
        let ptr = alloc.allocate(layout).unwrap();
        // SAFETY: ptr was returned by `allocate` with this layout (zero-size).
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn system_allocator_capacity_is_none() {
        let alloc = SystemAllocator;
        assert!(alloc.capacity().is_none());
        assert!(alloc.used().is_none());
    }

    #[test]
    fn shared_pool_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedPool>();
    }

    #[test]
    fn pool_exhausted_display() {
        let err = PoolExhausted {
            requested: Layout::from_size_align(1024, 64).unwrap(),
        };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("64"));
    }

    // ── SlabPool tests ──────────────────────────────────────────────

    #[test]
    fn slab_pool_allocate_and_deallocate() {
        let pool = SlabPool::new(4 * 1024 * 1024, HugePages::Off).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!(pool.used().unwrap() > 0);
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_returns_error_on_exhaustion() {
        let pool = SlabPool::new(64 * 1024, HugePages::Off).unwrap();
        let layout = Layout::from_size_align(64, 64).unwrap();

        let mut ptrs = Vec::new();
        while let Ok(ptr) = pool.allocate(layout) {
            ptrs.push(ptr);
        }
        assert!(!ptrs.is_empty());

        // Deallocate one and re-allocate succeeds.
        // SAFETY: ptrs were returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptrs.pop().unwrap(), layout) };
        assert!(pool.allocate(layout).is_ok());
    }

    #[test]
    fn slab_pool_observability() {
        let pool = SlabPool::new(1024 * 1024, HugePages::Off).unwrap();
        assert_eq!(pool.capacity(), Some(1024 * 1024));
        assert_eq!(pool.used(), Some(0));

        let layout = Layout::from_size_align(64, 64).unwrap();
        let ptr = pool.allocate(layout).unwrap();
        assert!(pool.used().unwrap() > 0);
        // SAFETY: ptr was returned by `allocate` with this layout.
        unsafe { pool.deallocate(ptr, layout) };
    }

    #[test]
    fn slab_pool_concurrent_allocate() {
        let pool = Arc::new(SlabPool::new(16 * 1024 * 1024, HugePages::Off).unwrap());
        let layout = Layout::from_size_align(64, 64).unwrap();

        std::thread::scope(|s| {
            for _ in 0..4 {
                let pool = Arc::clone(&pool);
                s.spawn(move || {
                    let mut ptrs = Vec::new();
                    for _ in 0..100 {
                        if let Ok(ptr) = pool.allocate(layout) {
                            ptrs.push(ptr);
                        }
                    }
                    for ptr in ptrs {
                        // SAFETY: each ptr was returned by `allocate` and is
                        // only deallocated once by this thread.
                        unsafe { pool.deallocate(ptr, layout) };
                    }
                });
            }
        });
    }

    #[test]
    fn slab_pool_multiple_size_classes() {
        let pool = SlabPool::new(8 * 1024 * 1024, HugePages::Off).unwrap();

        let small = Layout::from_size_align(32, 8).unwrap();
        let medium = Layout::from_size_align(200, 8).unwrap();
        let large = Layout::from_size_align(4000, 8).unwrap();

        let p1 = pool.allocate(small).unwrap();
        let p2 = pool.allocate(medium).unwrap();
        let p3 = pool.allocate(large).unwrap();

        // All should be non-null and different.
        assert_ne!(p1.as_ptr(), p2.as_ptr());
        assert_ne!(p2.as_ptr(), p3.as_ptr());

        // SAFETY: each ptr was returned by `allocate` with the corresponding layout.
        unsafe {
            pool.deallocate(p1, small);
            pool.deallocate(p2, medium);
            pool.deallocate(p3, large);
        }
    }
}
