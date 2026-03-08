//! Loom-based exhaustive concurrency tests for Minkowski's transactional primitives.
//!
//! Run: RUSTFLAGS="--cfg loom" cargo test -p minkowski --test loom_concurrency --features loom
//!
//! These tests use loom's deterministic scheduler to enumerate ALL possible
//! thread interleavings, verifying that concurrency invariants hold under
//! every schedule — not just the ones that happen to occur at runtime.
#![cfg(loom)]

use loom::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

// ── OrphanQueue tests ──────────────────────────────────────────────────

/// Simulates OrphanQueue: multiple Tx aborts push entity IDs concurrently
/// while World drains the queue. Verifies no entity ID is lost.
#[test]
fn orphan_queue_push_drain_no_lost_ids() {
    loom::model(|| {
        let queue: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

        let q1 = queue.clone();
        let t1 = thread::spawn(move || {
            q1.lock().unwrap().extend_from_slice(&[10, 11]);
        });

        let q2 = queue.clone();
        let t2 = thread::spawn(move || {
            q2.lock().unwrap().extend_from_slice(&[20, 21]);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let mut drained: Vec<u32> = queue.lock().unwrap().drain(..).collect();
        drained.sort();
        assert_eq!(drained, vec![10, 11, 20, 21]);
    });
}

/// Interleaved push and drain: one thread drains while another pushes.
/// Verifies all IDs appear exactly once across drain batches.
#[test]
fn orphan_queue_interleaved_push_drain() {
    loom::model(|| {
        let queue: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let results: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));

        let q1 = queue.clone();
        let pusher = thread::spawn(move || {
            q1.lock().unwrap().push(1);
            q1.lock().unwrap().push(2);
        });

        let q2 = queue.clone();
        let r = results.clone();
        let drainer = thread::spawn(move || {
            let batch: Vec<u32> = q2.lock().unwrap().drain(..).collect();
            r.lock().unwrap().push(batch);
        });

        pusher.join().unwrap();
        drainer.join().unwrap();

        let remainder: Vec<u32> = queue.lock().unwrap().drain(..).collect();
        results.lock().unwrap().push(remainder);

        let mut all: Vec<u32> = results
            .lock()
            .unwrap()
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        all.sort();
        assert_eq!(all, vec![1, 2]);
    });
}

// ── ColumnLockTable pattern tests ──────────────────────────────────────

/// Models ColumnLockTable's Mutex-protected read-write lock semantics.
/// The real ColumnLockTable uses a Mutex around the entire acquire operation,
/// so the check-then-act (read writer flag + increment readers) is atomic.
/// This test verifies that under that Mutex protection, an exclusive writer
/// and a shared reader never overlap.
#[test]
fn column_lock_exclusive_blocks_shared() {
    loom::model(|| {
        // The Mutex models ColumnLockTable's Mutex<HashMap<...>> —
        // all lock state checks and mutations happen under this lock.
        let lock_table = Arc::new(Mutex::new((false, 0u32))); // (writer, readers)

        // In-flight tracking for mutual exclusion assertion
        let writer_held = Arc::new(AtomicBool::new(false));
        let reader_held = Arc::new(AtomicBool::new(false));

        // Thread 1: try to acquire exclusive lock
        let lt1 = lock_table.clone();
        let wh = writer_held.clone();
        let rh1 = reader_held.clone();
        let t1 = thread::spawn(move || {
            let mut state = lt1.lock().unwrap();
            if !state.0 && state.1 == 0 {
                state.0 = true; // acquired exclusive
                drop(state); // release table mutex (lock is held)
                wh.store(true, Ordering::SeqCst);
                assert!(
                    !rh1.load(Ordering::SeqCst),
                    "exclusive lock held while shared lock also held"
                );
                wh.store(false, Ordering::SeqCst);
                lt1.lock().unwrap().0 = false; // release exclusive
            }
        });

        // Thread 2: try to acquire shared lock
        let lt2 = lock_table.clone();
        let rh = reader_held.clone();
        let wh2 = writer_held.clone();
        let t2 = thread::spawn(move || {
            let mut state = lt2.lock().unwrap();
            if !state.0 {
                state.1 += 1; // acquired shared
                drop(state);
                rh.store(true, Ordering::SeqCst);
                assert!(
                    !wh2.load(Ordering::SeqCst),
                    "shared lock held while exclusive lock also held"
                );
                rh.store(false, Ordering::SeqCst);
                lt2.lock().unwrap().1 -= 1; // release shared
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

/// Verifies the upgrade-not-downgrade consequence: when a transaction
/// requests both read and write on the same column, it must acquire
/// exclusive — meaning a concurrent shared reader is blocked.
/// This models the real scenario where dedup_by incorrectly kept Shared.
#[test]
fn column_lock_upgrade_blocks_concurrent_reader() {
    loom::model(|| {
        let lock_table = Arc::new(Mutex::new((false, 0u32))); // (writer, readers)
        let writer_held = Arc::new(AtomicBool::new(false));
        let reader_held = Arc::new(AtomicBool::new(false));

        // Thread 1: requests both read AND write on col_a → must acquire exclusive
        let lt1 = lock_table.clone();
        let wh = writer_held.clone();
        let rh1 = reader_held.clone();
        let t1 = thread::spawn(move || {
            let mut state = lt1.lock().unwrap();
            // Upgraded to exclusive (the correct behavior after dedup)
            if !state.0 && state.1 == 0 {
                state.0 = true;
                drop(state);
                wh.store(true, Ordering::SeqCst);
                // While we hold exclusive, no reader should be active
                assert!(
                    !rh1.load(Ordering::SeqCst),
                    "upgraded lock held while shared reader also active"
                );
                wh.store(false, Ordering::SeqCst);
                lt1.lock().unwrap().0 = false;
            }
        });

        // Thread 2: requests read-only on the same column
        let lt2 = lock_table.clone();
        let rh = reader_held.clone();
        let wh2 = writer_held.clone();
        let t2 = thread::spawn(move || {
            let mut state = lt2.lock().unwrap();
            if !state.0 {
                state.1 += 1;
                drop(state);
                rh.store(true, Ordering::SeqCst);
                assert!(
                    !wh2.load(Ordering::SeqCst),
                    "shared reader active while upgraded exclusive also held"
                );
                rh.store(false, Ordering::SeqCst);
                lt2.lock().unwrap().1 -= 1;
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

/// Two threads acquire locks in sorted order. Loom verifies no deadlock
/// under any interleaving — completing proves deadlock freedom.
#[test]
fn column_lock_sorted_acquisition_no_deadlock() {
    loom::model(|| {
        let lock_a = Arc::new(Mutex::new(()));
        let lock_b = Arc::new(Mutex::new(()));

        let a1 = lock_a.clone();
        let b1 = lock_b.clone();
        let t1 = thread::spawn(move || {
            let _ga = a1.lock().unwrap();
            let _gb = b1.lock().unwrap();
        });

        let a2 = lock_a.clone();
        let b2 = lock_b.clone();
        let t2 = thread::spawn(move || {
            let _ga = a2.lock().unwrap();
            let _gb = b2.lock().unwrap();
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

// ── EntityAllocator::reserve pattern tests ─────────────────────────────

/// Models EntityAllocator::reserve(): concurrent fetch_add on AtomicU32.
/// Verifies all returned indices are unique — no duplicate entity IDs.
#[test]
fn entity_reserve_no_duplicate_indices() {
    loom::model(|| {
        let counter = Arc::new(AtomicU32::new(0));

        let c1 = counter.clone();
        let t1 = thread::spawn(move || {
            let a = c1.fetch_add(1, Ordering::Relaxed);
            let b = c1.fetch_add(1, Ordering::Relaxed);
            vec![a, b]
        });

        let c2 = counter.clone();
        let t2 = thread::spawn(move || {
            let a = c2.fetch_add(1, Ordering::Relaxed);
            let b = c2.fetch_add(1, Ordering::Relaxed);
            vec![a, b]
        });

        let mut indices: Vec<u32> = Vec::new();
        indices.extend(t1.join().unwrap());
        indices.extend(t2.join().unwrap());

        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3]);
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    });
}
