//! Python-facing spatial grid for radius queries over Position entities.

use crate::components::Position;
use crate::pyworld::PyWorld;
use minkowski::Entity;
use pyo3::prelude::*;
use std::collections::HashMap;

type CellMap = HashMap<(i32, i32), Vec<(Entity, f32, f32)>>;

#[pyclass(name = "SpatialGrid")]
pub struct PySpatialGrid {
    world_size: f32,
    inv_cell: f32,
    cells: CellMap,
    count: usize,
}

#[pymethods]
impl PySpatialGrid {
    #[new]
    fn new(world_size: f32, cell_size: f32) -> Self {
        assert!(world_size > 0.0, "world_size must be positive");
        assert!(cell_size > 0.0, "cell_size must be positive");
        PySpatialGrid {
            world_size,
            inv_cell: 1.0 / cell_size,
            cells: HashMap::new(),
            count: 0,
        }
    }

    /// Rebuild the grid from all entities with a Position component.
    fn rebuild(&mut self, world: &mut PyWorld) {
        self.cells.clear();
        self.count = 0;
        let ws = self.world_size;
        let inv = self.inv_cell;

        world
            .world
            .query::<(Entity, &Position)>()
            .for_each(|(entity, pos)| {
                let x = pos.x.rem_euclid(ws);
                let y = pos.y.rem_euclid(ws);
                let cx = (x * inv) as i32;
                let cy = (y * inv) as i32;
                self.cells.entry((cx, cy)).or_default().push((entity, x, y));
                self.count += 1;
            });
    }

    /// Return entity IDs within `radius` of `(x, y)`, using toroidal distance.
    fn query_radius(&self, x: f32, y: f32, radius: f32) -> Vec<u64> {
        let ws = self.world_size;
        let half = ws * 0.5;
        let r2 = radius * radius;
        let inv = self.inv_cell;

        let xw = x.rem_euclid(ws);
        let yw = y.rem_euclid(ws);

        let cells_r = (radius * inv).ceil() as i32;
        let cx = (xw * inv) as i32;
        let cy = (yw * inv) as i32;
        let grid_cells = (ws * inv).ceil() as i32;

        let mut result = Vec::new();
        for dx in -cells_r..=cells_r {
            for dy in -cells_r..=cells_r {
                let gx = (cx + dx).rem_euclid(grid_cells);
                let gy = (cy + dy).rem_euclid(grid_cells);
                if let Some(bucket) = self.cells.get(&(gx, gy)) {
                    for &(entity, ex, ey) in bucket {
                        let mut ddx = (ex - xw).abs();
                        if ddx > half {
                            ddx = ws - ddx;
                        }
                        let mut ddy = (ey - yw).abs();
                        if ddy > half {
                            ddy = ws - ddy;
                        }
                        if ddx * ddx + ddy * ddy <= r2 {
                            result.push(entity.to_bits());
                        }
                    }
                }
            }
        }
        result
    }

    /// Number of entities in the grid.
    #[getter]
    fn entity_count(&self) -> usize {
        self.count
    }
}
