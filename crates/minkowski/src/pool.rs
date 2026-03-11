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
}
