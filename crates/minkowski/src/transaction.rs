use fixedbitset::FixedBitSet;

use crate::access::Access;
use crate::changeset::EnumChangeSet;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::lock_table::ColumnLockSet;
use crate::query::fetch::WorldQuery;
use crate::query::iter::QueryIter;
use crate::world::World;

/// Conflict information returned when a transaction commit fails.
pub struct Conflict {
    /// Which component columns had conflicting concurrent modifications.
    pub component_ids: FixedBitSet,
}

/// Strategy for transaction concurrency control.
///
/// The trait provides `begin()` which returns a strategy-specific transaction
/// object. The caller reads and writes through the transaction, then calls
/// `commit()` to apply changes atomically — or drops the transaction to abort.
///
/// Three built-in strategies:
/// - [`Sequential`] — zero-cost passthrough, no conflict detection
/// - [`Optimistic`] — live reads with tick-based validation at commit
/// - [`Pessimistic`] — cooperative column locks, guaranteed commit success
pub trait TransactionStrategy {
    /// The transaction object returned by `begin()`.
    type Tx<'w>;

    /// Begin a transaction. `access` declares which components will be
    /// read and written — used by Optimistic for tick snapshotting and
    /// by Pessimistic for lock acquisition.
    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> Self::Tx<'w>;
}

// ── Sequential ──────────────────────────────────────────────────────

/// Zero-cost transaction strategy. All operations delegate directly to World.
/// Commit always succeeds. No read-set, no changeset buffering, no validation.
///
/// Use when systems run sequentially and no conflict detection is needed.
pub struct Sequential;

impl TransactionStrategy for Sequential {
    type Tx<'w> = SequentialTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, _access: &Access) -> SequentialTx<'w> {
        SequentialTx { world }
    }
}

/// Transaction object for the [`Sequential`] strategy.
/// Transparent wrapper around `&mut World`.
pub struct SequentialTx<'w> {
    world: &'w mut World,
}

impl<'w> SequentialTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        self.world.spawn(bundle)
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.world.despawn(entity)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.world.insert(entity, component);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.world.remove::<T>(entity)
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.world.get_mut::<T>(entity)
    }

    /// Commit the transaction. Always succeeds for Sequential.
    /// Returns an empty reverse changeset (mutations went directly to World).
    pub fn commit(self) -> Result<EnumChangeSet, Conflict> {
        Ok(EnumChangeSet::new())
    }
}

// ── Optimistic ──────────────────────────────────────────────────────

/// Optimistic transaction strategy. Reads go directly to World (zero-copy).
/// Writes buffer into an EnumChangeSet. At commit, validates that no read
/// column was modified since begin — if conflict detected, returns Err.
///
/// Best for read-heavy workloads where conflicts are rare.
pub struct Optimistic;

impl TransactionStrategy for Optimistic {
    type Tx<'w> = OptimisticTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> OptimisticTx<'w> {
        let read_ticks = world.snapshot_column_ticks(access.reads());
        OptimisticTx {
            world,
            read_ticks,
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Optimistic`] strategy.
///
/// Reads are live (direct World access). Writes are buffered in an
/// `EnumChangeSet` and applied atomically at commit. If any column
/// that was read at begin has been modified by the time commit runs,
/// the transaction aborts with a [`Conflict`].
///
/// Drop without commit discards the buffered changeset (abort).
pub struct OptimisticTx<'w> {
    world: &'w mut World,
    read_ticks: Vec<(usize, ComponentId, crate::tick::Tick)>,
    changeset: EnumChangeSet,
}

impl<'w> OptimisticTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.changeset.insert::<T>(self.world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.changeset.remove::<T>(self.world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.world.alloc_entity();
        self.changeset.spawn_bundle(self.world, entity, bundle);
        entity
    }

    /// Validate and commit. Checks that no read column was modified since
    /// begin. Returns reverse changeset on success, Conflict on failure.
    pub fn commit(self) -> Result<EnumChangeSet, Conflict> {
        let conflicts = self.world.check_column_conflicts(&self.read_ticks);
        if !conflicts.is_empty() {
            return Err(Conflict {
                component_ids: conflicts,
            });
        }
        Ok(self.changeset.apply(self.world))
    }
}

// ── Pessimistic ─────────────────────────────────────────────────────

/// Pessimistic transaction strategy. Acquires cooperative column locks
/// at begin. Reads and writes are guaranteed conflict-free. Commit
/// always succeeds.
///
/// Best for write-heavy workloads or expensive computations where
/// optimistic retry would waste work.
pub struct Pessimistic;

impl TransactionStrategy for Pessimistic {
    type Tx<'w> = PessimisticTx<'w>;

    fn begin<'w>(&mut self, world: &'w mut World, access: &Access) -> PessimisticTx<'w> {
        let locks = world
            .lock_table
            .acquire(
                &world.archetypes.archetypes,
                access.reads(),
                access.writes(),
            )
            .expect("pessimistic begin: lock acquisition failed (held by another transaction)");
        PessimisticTx {
            world,
            locks: Some(locks),
            changeset: EnumChangeSet::new(),
        }
    }
}

/// Transaction object for the [`Pessimistic`] strategy.
///
/// Column locks are held for the transaction's lifetime. Reads go
/// through World (protected by shared locks). Writes buffer into
/// an `EnumChangeSet`. Commit applies the changeset and releases locks.
///
/// Drop without commit releases locks and discards the changeset (abort).
pub struct PessimisticTx<'w> {
    world: &'w mut World,
    locks: Option<ColumnLockSet>,
    changeset: EnumChangeSet,
}

impl<'w> PessimisticTx<'w> {
    pub fn query<Q: WorldQuery + 'static>(&mut self) -> QueryIter<'_, Q> {
        self.world.query::<Q>()
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.changeset.insert::<T>(self.world, entity, value);
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.changeset.remove::<T>(self.world, entity);
    }

    pub fn spawn<B: crate::bundle::Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.world.alloc_entity();
        self.changeset.spawn_bundle(self.world, entity, bundle);
        entity
    }

    /// Commit the transaction. Always succeeds (locks guarantee isolation).
    /// Applies the buffered changeset and releases locks.
    pub fn commit(mut self) -> Result<EnumChangeSet, Conflict> {
        let changeset = std::mem::take(&mut self.changeset);
        let reverse = changeset.apply(self.world);
        if let Some(locks) = self.locks.take() {
            self.world.lock_table.release(locks);
        }
        Ok(reverse)
    }
}

impl<'w> Drop for PessimisticTx<'w> {
    fn drop(&mut self) {
        if let Some(locks) = self.locks.take() {
            self.world.lock_table.release(locks);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::Access;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos(f32);
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel(f32);

    #[test]
    fn sequential_query_reads_spawned_entities() {
        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &Vel)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn sequential_mutation_is_immediate() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn((Pos(42.0),));
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        let _ = tx.commit();
    }

    #[test]
    fn sequential_commit_always_ok() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Sequential;
        let mut tx = strategy.begin(&mut world, &access);
        tx.spawn((Pos(1.0),));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn sequential_drop_is_noop() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);
        let mut strategy = Sequential;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.spawn((Pos(1.0),));
            // drop without commit — but mutations already applied
        }
        assert_eq!(world.query::<(&Pos,)>().count(), 1);
    }

    #[test]
    fn optimistic_commit_succeeds_without_conflict() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let count = tx.query::<(&Pos,)>().count();
        assert_eq!(count, 1);
        tx.insert::<Vel>(e, Vel(99.0));
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn optimistic_buffered_writes_not_visible_until_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(e, Pos(99.0));
        // Query still sees old value (write is buffered)
        let pos = tx.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit();
        // After commit, world has new value
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 99.0);
    }

    #[test]
    fn optimistic_drop_aborts_changeset() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        {
            let mut tx = strategy.begin(&mut world, &access);
            tx.insert::<Pos>(e, Pos(99.0));
            // drop without commit — changeset discarded
        }
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0); // unchanged
    }

    #[test]
    fn optimistic_conflict_detected_on_read_column_mutation() {
        use fixedbitset::FixedBitSet;

        let mut world = World::new();
        world.spawn((Pos(1.0), Vel(2.0)));

        // Test via the lower-level helpers since tx borrows world exclusively
        let pos_id = world.components.id::<Pos>().unwrap();
        let mut read_bits = FixedBitSet::with_capacity(pos_id + 1);
        read_bits.insert(pos_id);
        let snap = world.snapshot_column_ticks(&read_bits);

        // Mutate the read column
        for pos in world.query::<(&mut Pos,)>() {
            pos.0 .0 = 99.0;
        }

        let conflicts = world.check_column_conflicts(&snap);
        assert!(conflicts.contains(pos_id));
    }

    #[test]
    fn optimistic_spawn_allocates_entity() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Optimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn((Pos(42.0),));
        let result = tx.commit();
        assert!(result.is_ok());
        assert!(world.get::<Pos>(e).is_some());
    }

    #[test]
    fn pessimistic_commit_always_succeeds() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0), Vel(2.0)));
        let access = Access::of::<(&Pos, &mut Vel)>(&mut world);

        let mut strategy = Pessimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let _ = tx.query::<(&Pos,)>().count();
        tx.insert::<Vel>(e, Vel(99.0));
        assert!(tx.commit().is_ok());
    }

    #[test]
    fn pessimistic_buffered_writes_applied_at_commit() {
        let mut world = World::new();
        let e = world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic;
        let mut tx = strategy.begin(&mut world, &access);
        tx.insert::<Pos>(e, Pos(42.0));
        let pos = tx.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 1.0);
        let _ = tx.commit();
        let pos = world.query::<(&Pos,)>().next().unwrap();
        assert_eq!(pos.0 .0, 42.0);
    }

    #[test]
    fn pessimistic_drop_releases_locks() {
        let mut world = World::new();
        world.spawn((Pos(1.0),));
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic;
        {
            let _tx = strategy.begin(&mut world, &access);
        }
        // Locks released — can begin another transaction
        let tx2 = strategy.begin(&mut world, &access);
        let _ = tx2.commit();
    }

    #[test]
    fn pessimistic_spawn_visible_after_commit() {
        let mut world = World::new();
        let access = Access::of::<(&mut Pos,)>(&mut world);

        let mut strategy = Pessimistic;
        let mut tx = strategy.begin(&mut world, &access);
        let e = tx.spawn((Pos(77.0),));
        let _ = tx.commit();
        assert!(world.get::<Pos>(e).is_some());
        assert_eq!(world.get::<Pos>(e).unwrap().0, 77.0);
    }
}
