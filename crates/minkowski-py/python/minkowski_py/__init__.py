"""Minkowski ECS — Python bindings for high-performance entity-component simulations.

Quick start::

    import minkowski_py as mk

    # Boids flocking simulation
    sim = mk.BoidsSim(n=2000, world_size=500.0)
    sim.step(100, record=True)
    df = sim.to_polars()          # Polars DataFrame
    history = sim.history_to_polars()

    # N-body gravity
    sim = mk.NBodySim(n=500)
    sim.step(200, record=True)

    # Game of Life
    sim = mk.LifeSim(width=64, height=64)
    sim.step(100, record=True)

All simulations export data via Apache Arrow for zero-copy handoff to
Polars DataFrames. Use ``to_polars()`` for current state and
``history_to_polars()`` for recorded trajectory data.
"""

from minkowski_py._minkowski import BoidsSim, NBodySim, LifeSim

__all__ = ["BoidsSim", "NBodySim", "LifeSim"]
__version__ = "0.1.0"
