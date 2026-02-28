/// Monotonic tick counter with wrapping-aware comparison.
///
/// Used for change detection: each BlobVec column stores the tick at which
/// it was last mutably accessed. Queries compare column ticks against their
/// last-read tick to skip unchanged archetypes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Tick(u32);

impl Tick {
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Wrapping-aware comparison. Returns true if `self` is more recent than `other`.
    ///
    /// Handles overflow: treats any tick within `u32::MAX / 2` distance as "recent".
    pub fn is_newer_than(self, other: Tick) -> bool {
        let diff = self.0.wrapping_sub(other.0);
        diff > 0 && diff < u32::MAX / 2
    }

    /// Advance the tick by one (wrapping).
    pub fn advance(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero() {
        assert_eq!(Tick::default(), Tick::new(0));
    }

    #[test]
    fn advance_increments() {
        let mut t = Tick::new(0);
        t.advance();
        assert_eq!(t, Tick::new(1));
    }

    #[test]
    fn newer_than_basic() {
        let a = Tick::new(5);
        let b = Tick::new(3);
        assert!(a.is_newer_than(b));
        assert!(!b.is_newer_than(a));
    }

    #[test]
    fn newer_than_equal_is_false() {
        let a = Tick::new(5);
        assert!(!a.is_newer_than(a));
    }

    #[test]
    fn newer_than_wrapping() {
        let old = Tick::new(u32::MAX - 5);
        let new = Tick::new(3);
        assert!(new.is_newer_than(old));
        assert!(!old.is_newer_than(new));
    }

    #[test]
    fn advance_wraps() {
        let mut t = Tick::new(u32::MAX);
        t.advance();
        assert_eq!(t, Tick::new(0));
    }
}
