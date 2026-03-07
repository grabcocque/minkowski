//! Python bindings for the Minkowski ECS engine.
//!
//! Exposes World, Entity, and ReducerRegistry directly. Queries return
//! Arrow RecordBatches (one-copy from BlobVec, zero-copy to Python via
//! pyo3-arrow C Data Interface).

use pyo3::prelude::*;

#[pymodule]
fn _minkowski(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
