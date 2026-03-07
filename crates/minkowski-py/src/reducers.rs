//! Pre-compiled Rust reducers callable from Python by name.
//!
//! Each reducer is registered via `ReducerRegistry::register_query` (read-write)
//! or `register_query_ref` (read-only). The `register_all` function returns a
//! name-to-ID map consumed by `PyReducerRegistry`.

use crate::components::{Acceleration, CellState, Heading, Mass, Position, Velocity};
use minkowski::{Entity, QueryMut, QueryReducerId, ReducerRegistry, World};
use std::collections::HashMap;

// ── Toroidal helpers ─────────────────────────────────────────────────

/// Wrap a coordinate into [0, world_size).
fn wrap(mut v: f32, world_size: f32) -> f32 {
    if v >= world_size {
        v -= world_size;
    } else if v < 0.0 {
        v += world_size;
    }
    v
}

/// Minimum-image signed difference on a toroidal domain.
fn wrapped_diff(a: f32, b: f32, world_size: f32) -> f32 {
    let d = a - b;
    if d > world_size * 0.5 {
        d - world_size
    } else if d < -world_size * 0.5 {
        d + world_size
    } else {
        d
    }
}

// ── Parameter tuples ─────────────────────────────────────────────────
// Each reducer receives a single `Args: Clone + 'static` value.
// We define typed tuples for each reducer's parameters.

/// Parameters for `boids_forces`.
#[derive(Clone)]
pub struct BoidsForceParams {
    pub world_size: f32,
    pub sep_r: f32,
    pub ali_r: f32,
    pub coh_r: f32,
    pub sep_w: f32,
    pub ali_w: f32,
    pub coh_w: f32,
    pub max_force: f32,
}

/// Parameters for `boids_integrate`.
#[derive(Clone)]
pub struct BoidsIntegrateParams {
    pub max_speed: f32,
    pub dt: f32,
    pub world_size: f32,
}

/// Parameters for `gravity`.
#[derive(Clone)]
pub struct GravityParams {
    pub g: f32,
    pub softening: f32,
    pub dt: f32,
    pub world_size: f32,
}

/// Parameters for `life_step`.
#[derive(Clone)]
pub struct LifeStepParams {
    pub width: usize,
    pub height: usize,
}

/// Parameters for `movement`.
#[derive(Clone)]
pub struct MovementParams {
    pub dt: f32,
    pub world_size: f32,
}

// ── Registration ─────────────────────────────────────────────────────

/// Register all built-in reducers. Returns a map of name -> QueryReducerId.
pub fn register_all(
    registry: &mut ReducerRegistry,
    world: &mut World,
) -> HashMap<String, QueryReducerId> {
    let mut map = HashMap::new();

    // ── boids_forces ──
    // Snapshot positions+velocities, build spatial grid, compute
    // separation/alignment/cohesion forces, write accelerations.
    let id = registry
        .register_query::<(&mut Acceleration, &Position, &Velocity, Entity), BoidsForceParams, _>(
            world,
            "boids_forces",
            |mut query: QueryMut<'_, (&mut Acceleration, &Position, &Velocity, Entity)>,
             params: BoidsForceParams| {
                // Pass 1: snapshot all boid data.
                let mut snapshot: Vec<(Entity, f32, f32, f32, f32)> = Vec::new();
                query.for_each(|(_acc, pos, vel, entity)| {
                    snapshot.push((entity, pos.x, pos.y, vel.x, vel.y));
                });

                // Build simple spatial grid for neighbor lookups.
                let cell_size = params.coh_r.max(params.ali_r).max(params.sep_r);
                let grid_w = (params.world_size / cell_size).ceil() as usize;
                let grid_w = grid_w.max(1);
                let mut cells: Vec<Vec<usize>> = vec![Vec::new(); grid_w * grid_w];
                for (i, &(_, px, py, _, _)) in snapshot.iter().enumerate() {
                    let cx = ((px / cell_size) as usize).min(grid_w - 1);
                    let cy = ((py / cell_size) as usize).min(grid_w - 1);
                    cells[cy * grid_w + cx].push(i);
                }

                // Compute forces into a local buffer.
                let mut forces: Vec<(Entity, f32, f32)> = Vec::with_capacity(snapshot.len());
                for &(entity, px, py, vx, vy) in &snapshot {
                    let cx = ((px / cell_size) as usize).min(grid_w - 1);
                    let cy = ((py / cell_size) as usize).min(grid_w - 1);

                    let mut sep_x = 0.0f32;
                    let mut sep_y = 0.0f32;
                    let mut ali_x = 0.0f32;
                    let mut ali_y = 0.0f32;
                    let mut coh_x = 0.0f32;
                    let mut coh_y = 0.0f32;
                    let mut sep_count = 0u32;
                    let mut ali_count = 0u32;
                    let mut coh_count = 0u32;

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                            let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                            for &j in &cells[ny * grid_w + nx] {
                                let (_, ox, oy, ovx, ovy) = snapshot[j];
                                let diff_x = wrapped_diff(ox, px, params.world_size);
                                let diff_y = wrapped_diff(oy, py, params.world_size);
                                let dist_sq = diff_x * diff_x + diff_y * diff_y;
                                if dist_sq < 1e-6 {
                                    continue;
                                }
                                let dist = dist_sq.sqrt();

                                if dist < params.sep_r {
                                    let inv = 1.0 / dist;
                                    let nx = diff_x * inv;
                                    let ny = diff_y * inv;
                                    sep_x -= nx * inv;
                                    sep_y -= ny * inv;
                                    sep_count += 1;
                                }
                                if dist < params.ali_r {
                                    ali_x += ovx;
                                    ali_y += ovy;
                                    ali_count += 1;
                                }
                                if dist < params.coh_r {
                                    coh_x += diff_x;
                                    coh_y += diff_y;
                                    coh_count += 1;
                                }
                            }
                        }
                    }

                    let mut fx = 0.0f32;
                    let mut fy = 0.0f32;
                    if sep_count > 0 {
                        let n = sep_count as f32;
                        fx += (sep_x / n) * params.sep_w;
                        fy += (sep_y / n) * params.sep_w;
                    }
                    if ali_count > 0 {
                        let n = ali_count as f32;
                        let desired_x = ali_x / n - vx;
                        let desired_y = ali_y / n - vy;
                        fx += desired_x * params.ali_w;
                        fy += desired_y * params.ali_w;
                    }
                    if coh_count > 0 {
                        let n = coh_count as f32;
                        fx += (coh_x / n) * params.coh_w;
                        fy += (coh_y / n) * params.coh_w;
                    }

                    // Clamp force magnitude
                    let f_sq = fx * fx + fy * fy;
                    if f_sq > params.max_force * params.max_force {
                        let f_len = f_sq.sqrt();
                        fx = fx / f_len * params.max_force;
                        fy = fy / f_len * params.max_force;
                    }

                    forces.push((entity, fx, fy));
                }

                // Pass 2: write forces into accelerations.
                // Build a lookup from Entity -> force index.
                let force_map: HashMap<u64, usize> = forces
                    .iter()
                    .enumerate()
                    .map(|(i, &(e, _, _))| (e.to_bits(), i))
                    .collect();

                query.for_each(|(acc, _pos, _vel, entity)| {
                    if let Some(&idx) = force_map.get(&entity.to_bits()) {
                        let (_, fx, fy) = forces[idx];
                        acc.x = fx;
                        acc.y = fy;
                    }
                });
            },
        );
    map.insert("boids_forces".to_string(), id);

    // ── boids_integrate ──
    let id = registry
        .register_query::<(&mut Position, &mut Velocity, &Acceleration), BoidsIntegrateParams, _>(
            world,
            "boids_integrate",
            |mut query: QueryMut<'_, (&mut Position, &mut Velocity, &Acceleration)>,
             params: BoidsIntegrateParams| {
                let max_speed = params.max_speed;
                let ws = params.world_size;
                let dt = params.dt;
                query.for_each_chunk(|(poss, vels, accs)| {
                    for i in 0..vels.len() {
                        vels[i].x += accs[i].x * dt;
                        vels[i].y += accs[i].y * dt;
                        // Clamp speed
                        let speed_sq = vels[i].x * vels[i].x + vels[i].y * vels[i].y;
                        if speed_sq > max_speed * max_speed {
                            let speed = speed_sq.sqrt();
                            vels[i].x = vels[i].x / speed * max_speed;
                            vels[i].y = vels[i].y / speed * max_speed;
                        }
                        // Integrate position + wrap
                        poss[i].x = wrap(poss[i].x + vels[i].x * dt, ws);
                        poss[i].y = wrap(poss[i].y + vels[i].y * dt, ws);
                    }
                });
            },
        );
    map.insert("boids_integrate".to_string(), id);

    // ── gravity ──
    // O(N^2) pairwise gravity. Snapshots positions+masses, computes forces,
    // applies to velocities, then integrates positions.
    let id = registry
        .register_query::<(&mut Velocity, &mut Position, &Mass, Entity), GravityParams, _>(
            world,
            "gravity",
            |mut query: QueryMut<'_, (&mut Velocity, &mut Position, &Mass, Entity)>,
             params: GravityParams| {
                // Snapshot all bodies.
                let mut bodies: Vec<(Entity, f32, f32, f32)> = Vec::new(); // entity, x, y, mass
                query.for_each(|(_vel, pos, mass, entity)| {
                    bodies.push((entity, pos.x, pos.y, mass.0));
                });

                let n = bodies.len();
                let mut accels: Vec<(f32, f32)> = vec![(0.0, 0.0); n];

                // Compute pairwise gravitational accelerations.
                for i in 0..n {
                    let (_, ix, iy, im) = bodies[i];
                    for j in (i + 1)..n {
                        let (_, jx, jy, jm) = bodies[j];
                        let dx = wrapped_diff(jx, ix, params.world_size);
                        let dy = wrapped_diff(jy, iy, params.world_size);
                        let dist_sq = dx * dx + dy * dy + params.softening * params.softening;
                        let dist = dist_sq.sqrt();
                        let force_mag = params.g / dist_sq;
                        let nx = dx / dist;
                        let ny = dy / dist;

                        // F = G * m_i * m_j / r^2, a = F/m
                        accels[i].0 += force_mag * jm * nx;
                        accels[i].1 += force_mag * jm * ny;
                        accels[j].0 -= force_mag * im * nx;
                        accels[j].1 -= force_mag * im * ny;
                    }
                }

                // Build entity -> accel lookup.
                let accel_map: HashMap<u64, usize> = bodies
                    .iter()
                    .enumerate()
                    .map(|(i, &(e, _, _, _))| (e.to_bits(), i))
                    .collect();

                // Apply: update velocities and positions.
                let dt = params.dt;
                let ws = params.world_size;
                query.for_each(|(vel, pos, _mass, entity)| {
                    if let Some(&idx) = accel_map.get(&entity.to_bits()) {
                        let (ax, ay) = accels[idx];
                        vel.x += ax * dt;
                        vel.y += ay * dt;
                        pos.x = wrap(pos.x + vel.x * dt, ws);
                        pos.y = wrap(pos.y + vel.y * dt, ws);
                    }
                });
            },
        );
    map.insert("gravity".to_string(), id);

    // ── life_step ──
    // One generation of Conway's Game of Life.
    // Assumes all CellState entities are spawned in row-major order.
    let id = registry.register_query::<(&mut CellState, Entity), LifeStepParams, _>(
        world,
        "life_step",
        |mut query: QueryMut<'_, (&mut CellState, Entity)>, params: LifeStepParams| {
            let w = params.width;
            let h = params.height;

            // Snapshot current cell states in entity order.
            let mut cells: Vec<(Entity, bool)> = Vec::new();
            query.for_each(|(cs, entity)| {
                cells.push((entity, cs.0));
            });

            let n = cells.len();
            if n != w * h {
                // Mismatch — skip silently. The Python layer should ensure
                // the entity count matches width * height.
                return;
            }

            // Compute neighbor counts.
            let mut new_states: Vec<bool> = Vec::with_capacity(n);
            for idx in 0..n {
                let x = idx % w;
                let y = idx / w;
                let left = if x == 0 { w - 1 } else { x - 1 };
                let right = if x == w - 1 { 0 } else { x + 1 };
                let up = if y == 0 { h - 1 } else { y - 1 };
                let down = if y == h - 1 { 0 } else { y + 1 };

                let neighbors = [
                    up * w + left,
                    up * w + x,
                    up * w + right,
                    y * w + left,
                    y * w + right,
                    down * w + left,
                    down * w + x,
                    down * w + right,
                ];

                let mut count = 0u8;
                for &ni in &neighbors {
                    if cells[ni].1 {
                        count += 1;
                    }
                }

                let alive = cells[idx].1;
                let new_alive = matches!((alive, count), (true, 2) | (true, 3) | (false, 3));
                new_states.push(new_alive);
            }

            // Write back. Build entity -> new_state lookup.
            let state_map: HashMap<u64, bool> = cells
                .iter()
                .enumerate()
                .map(|(i, &(e, _))| (e.to_bits(), new_states[i]))
                .collect();

            query.for_each(|(cs, entity)| {
                if let Some(&new) = state_map.get(&entity.to_bits()) {
                    cs.0 = new;
                }
            });
        },
    );
    map.insert("life_step".to_string(), id);

    // ── movement ──
    // Heading-based movement with toroidal wrapping.
    let id = registry.register_query::<(&mut Position, &Heading), MovementParams, _>(
        world,
        "movement",
        |mut query: QueryMut<'_, (&mut Position, &Heading)>, params: MovementParams| {
            let ws = params.world_size;
            let dt = params.dt;
            query.for_each_chunk(|(poss, headings)| {
                for i in 0..poss.len() {
                    let h = headings[i].0;
                    poss[i].x = wrap(poss[i].x + h.cos() * dt, ws);
                    poss[i].y = wrap(poss[i].y + h.sin() * dt, ws);
                }
            });
        },
    );
    map.insert("movement".to_string(), id);

    map
}

// ── Dispatch ─────────────────────────────────────────────────────────

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Extract an f32 kwarg with a default.
fn kwarg_f32(kwargs: Option<&Bound<'_, PyDict>>, key: &str, default: f32) -> PyResult<f32> {
    match kwargs.and_then(|kw| kw.get_item(key).ok().flatten()) {
        Some(v) => v.extract(),
        None => Ok(default),
    }
}

/// Extract a usize kwarg, required.
fn kwarg_usize(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<usize> {
    match kwargs.and_then(|kw| kw.get_item(key).ok().flatten()) {
        Some(v) => v.extract(),
        None => Err(PyValueError::new_err(format!(
            "missing required kwarg: {key}"
        ))),
    }
}

/// Dispatch a reducer call by name, extracting typed parameters from kwargs.
pub fn dispatch(
    registry: &ReducerRegistry,
    world: &mut World,
    name: &str,
    id: QueryReducerId,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    match name {
        "boids_forces" => {
            let params = BoidsForceParams {
                world_size: kwarg_f32(kwargs, "world_size", 500.0)?,
                sep_r: kwarg_f32(kwargs, "sep_r", 25.0)?,
                ali_r: kwarg_f32(kwargs, "ali_r", 50.0)?,
                coh_r: kwarg_f32(kwargs, "coh_r", 50.0)?,
                sep_w: kwarg_f32(kwargs, "sep_w", 1.5)?,
                ali_w: kwarg_f32(kwargs, "ali_w", 1.0)?,
                coh_w: kwarg_f32(kwargs, "coh_w", 1.0)?,
                max_force: kwarg_f32(kwargs, "max_force", 0.1)?,
            };
            registry.run(world, id, params);
        }
        "boids_integrate" => {
            let params = BoidsIntegrateParams {
                max_speed: kwarg_f32(kwargs, "max_speed", 4.0)?,
                dt: kwarg_f32(kwargs, "dt", 0.016)?,
                world_size: kwarg_f32(kwargs, "world_size", 500.0)?,
            };
            registry.run(world, id, params);
        }
        "gravity" => {
            let params = GravityParams {
                g: kwarg_f32(kwargs, "g", 6.674e-2)?,
                softening: kwarg_f32(kwargs, "softening", 1.0)?,
                dt: kwarg_f32(kwargs, "dt", 0.001)?,
                world_size: kwarg_f32(kwargs, "world_size", 500.0)?,
            };
            registry.run(world, id, params);
        }
        "life_step" => {
            let params = LifeStepParams {
                width: kwarg_usize(kwargs, "width")?,
                height: kwarg_usize(kwargs, "height")?,
            };
            registry.run(world, id, params);
        }
        "movement" => {
            let params = MovementParams {
                dt: kwarg_f32(kwargs, "dt", 0.016)?,
                world_size: kwarg_f32(kwargs, "world_size", 500.0)?,
            };
            registry.run(world, id, params);
        }
        _ => {
            return Err(PyValueError::new_err(format!("unknown reducer: {name}")));
        }
    }
    Ok(())
}
