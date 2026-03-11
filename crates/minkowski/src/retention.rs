//! Expiry component and retention reducer for automatic entity cleanup.
//!
//! `Expiry` marks entities for despawn at a target tick. The retention
//! reducer scans for expired entities and batch-despawns them. The user
//! controls dispatch frequency — the engine never runs retention automatically.

use crate::tick::ChangeTick;

/// Marks an entity for despawn when the world tick reaches or exceeds this value.
///
/// Set at spawn time via [`World::change_tick`](crate::World::change_tick):
///
/// ```ignore
/// let ttl_ticks = 1000;
/// let deadline = ChangeTick::from_raw(world.change_tick().to_raw() + ttl_ticks);
/// world.spawn((data, Expiry(deadline)));
/// ```
///
/// The tick is a monotonic u64 from change detection — **not** wall-clock time.
/// For time-based TTL, convert duration to ticks based on your tick rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expiry(pub ChangeTick);

impl Expiry {
    /// Create an expiry at the given tick.
    pub fn at_tick(tick: ChangeTick) -> Self {
        Self(tick)
    }

    /// The deadline tick.
    pub fn deadline(&self) -> ChangeTick {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    #[test]
    fn expiry_is_a_component() {
        let mut world = World::new();
        let tick = world.change_tick();
        let e = world.spawn((Expiry::at_tick(tick),));
        assert_eq!(world.get::<Expiry>(e).unwrap().deadline(), tick);
    }

    #[test]
    fn expiry_round_trip() {
        let tick = ChangeTick::from_raw(42);
        let exp = Expiry::at_tick(tick);
        assert_eq!(exp.deadline().to_raw(), 42);
    }

    #[test]
    fn expiry_coexists_with_other_components() {
        let mut world = World::new();
        let tick = ChangeTick::from_raw(100);
        let e = world.spawn((42u32, Expiry::at_tick(tick)));
        assert_eq!(*world.get::<u32>(e).unwrap(), 42);
        assert_eq!(world.get::<Expiry>(e).unwrap().deadline().to_raw(), 100);
    }

    #[test]
    fn retention_despawns_expired_entities() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let tick = world.change_tick();
        let past = ChangeTick::from_raw(0); // already expired
        let future = ChangeTick::from_raw(tick.to_raw() + 1_000_000);

        let e_expired_1 = world.spawn((Expiry::at_tick(past), 1u32));
        let e_expired_2 = world.spawn((Expiry::at_tick(past), 2u32));
        let e_alive = world.spawn((Expiry::at_tick(future), 3u32));
        let e_no_expiry = world.spawn((4u32,));

        registry.run(&mut world, retention_id, ()).unwrap();

        assert!(!world.is_alive(e_expired_1));
        assert!(!world.is_alive(e_expired_2));
        assert!(world.is_alive(e_alive));
        assert!(world.is_alive(e_no_expiry));
        assert_eq!(*world.get::<u32>(e_alive).unwrap(), 3);
        assert_eq!(*world.get::<u32>(e_no_expiry).unwrap(), 4);
    }

    #[test]
    fn retention_is_idempotent() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let past = ChangeTick::from_raw(0);
        world.spawn((Expiry::at_tick(past),));

        registry.run(&mut world, retention_id, ()).unwrap();
        registry.run(&mut world, retention_id, ()).unwrap();

        let mut count = 0;
        world.query::<(&Expiry,)>().for_each(|_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn retention_noop_when_nothing_expired() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let future = ChangeTick::from_raw(u64::MAX);
        let e = world.spawn((Expiry::at_tick(future),));

        registry.run(&mut world, retention_id, ()).unwrap();

        assert!(world.is_alive(e));
    }

    #[test]
    fn retention_access_declares_despawns() {
        let mut world = World::new();
        let mut registry = crate::ReducerRegistry::new();
        let retention_id = registry.retention(&mut world);

        let access = registry.query_reducer_access(retention_id);
        assert!(access.despawns());
    }
}
