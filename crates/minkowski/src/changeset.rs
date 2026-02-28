use std::alloc::Layout;

use crate::component::ComponentId;
use crate::entity::Entity;

/// Contiguous byte arena for component data. Mutations store integer offsets
/// into this arena, avoiding per-mutation heap allocation.
pub(crate) struct Arena {
    data: Vec<u8>,
}

impl Arena {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Copy `layout.size()` bytes from `src` into the arena.
    /// Returns the byte offset where data was written.
    pub fn alloc(&mut self, src: *const u8, layout: Layout) -> usize {
        if layout.size() == 0 {
            return 0;
        }
        let align = layout.align();
        let offset = (self.data.len() + align - 1) & !(align - 1);
        self.data.resize(offset + layout.size(), 0);
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.data.as_mut_ptr().add(offset), layout.size());
        }
        offset
    }

    /// Get a raw pointer to data at the given offset.
    #[allow(dead_code)]
    pub fn get(&self, offset: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(offset) }
    }
}

/// A single structural mutation recorded in a ChangeSet.
#[allow(dead_code)]
pub(crate) enum Mutation {
    Spawn {
        entity: Entity,
        /// (ComponentId, arena offset, Layout) per component.
        components: Vec<(ComponentId, usize, Layout)>,
    },
    Despawn {
        entity: Entity,
    },
    Insert {
        entity: Entity,
        component_id: ComponentId,
        offset: usize,
        layout: Layout,
    },
    Remove {
        entity: Entity,
        component_id: ComponentId,
    },
}

/// Data-driven mutation buffer. Records structural mutations as an enum vec
/// with component bytes stored in a contiguous Arena.
pub struct EnumChangeSet {
    pub(crate) mutations: Vec<Mutation>,
    pub(crate) arena: Arena,
}

impl EnumChangeSet {
    pub fn new() -> Self {
        Self {
            mutations: Vec::new(),
            arena: Arena::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Record a spawn: entity + raw component data.
    pub fn record_spawn(
        &mut self,
        entity: Entity,
        components: &[(ComponentId, *const u8, Layout)],
    ) {
        let stored: Vec<(ComponentId, usize, Layout)> = components
            .iter()
            .map(|&(id, ptr, layout)| {
                let offset = self.arena.alloc(ptr, layout);
                (id, offset, layout)
            })
            .collect();
        self.mutations.push(Mutation::Spawn {
            entity,
            components: stored,
        });
    }

    /// Record a despawn.
    pub fn record_despawn(&mut self, entity: Entity) {
        self.mutations.push(Mutation::Despawn { entity });
    }

    /// Record inserting a component on an entity.
    pub fn record_insert(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        data: *const u8,
        layout: Layout,
    ) {
        let offset = self.arena.alloc(data, layout);
        self.mutations.push(Mutation::Insert {
            entity,
            component_id,
            offset,
            layout,
        });
    }

    /// Record removing a component from an entity.
    pub fn record_remove(&mut self, entity: Entity, component_id: ComponentId) {
        self.mutations.push(Mutation::Remove {
            entity,
            component_id,
        });
    }
}

impl Default for EnumChangeSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_read_back() {
        let mut arena = Arena::new();
        let value: u32 = 42;
        let layout = Layout::new::<u32>();
        let offset = arena.alloc(&value as *const u32 as *const u8, layout);
        let ptr = arena.get(offset) as *const u32;
        assert_eq!(unsafe { *ptr }, 42);
    }

    #[test]
    fn arena_alignment() {
        let mut arena = Arena::new();
        let byte: u8 = 0xFF;
        let _ = arena.alloc(&byte as *const u8, Layout::new::<u8>());

        let val: u64 = 123456789;
        let offset = arena.alloc(&val as *const u64 as *const u8, Layout::new::<u64>());
        assert_eq!(offset % 8, 0, "u64 offset must be 8-byte aligned");
    }

    #[test]
    fn arena_zst() {
        let mut arena = Arena::new();
        let layout = Layout::new::<()>();
        let offset = arena.alloc(std::ptr::null(), layout);
        assert_eq!(offset, 0);
    }

    #[test]
    fn record_and_count() {
        let mut cs = EnumChangeSet::new();
        let e = Entity::new(0, 0);
        cs.record_despawn(e);
        cs.record_remove(e, 1);
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn record_insert_stores_data() {
        let mut cs = EnumChangeSet::new();
        let e = Entity::new(0, 0);
        let value: u32 = 99;
        cs.record_insert(
            e,
            0,
            &value as *const u32 as *const u8,
            Layout::new::<u32>(),
        );
        assert_eq!(cs.len(), 1);
        if let Mutation::Insert { offset, .. } = &cs.mutations[0] {
            let ptr = cs.arena.get(*offset) as *const u32;
            assert_eq!(unsafe { *ptr }, 99);
        } else {
            panic!("expected Insert mutation");
        }
    }

    #[test]
    fn record_spawn_stores_components() {
        let mut cs = EnumChangeSet::new();
        let e = Entity::new(0, 0);
        let a: u32 = 1;
        let b: u64 = 2;
        let components = vec![
            (0, &a as *const u32 as *const u8, Layout::new::<u32>()),
            (1, &b as *const u64 as *const u8, Layout::new::<u64>()),
        ];
        cs.record_spawn(e, &components);
        assert_eq!(cs.len(), 1);
        if let Mutation::Spawn { components, .. } = &cs.mutations[0] {
            assert_eq!(components.len(), 2);
        } else {
            panic!("expected Spawn mutation");
        }
    }
}
