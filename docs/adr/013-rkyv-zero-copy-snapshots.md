# ADR-013: rkyv Zero-Copy Snapshots

**Status:** Accepted
**Date:** 2026-03-06

## Context

Snapshot load deserializes every component value through bincode — allocating a `Vec<u8>` per component per entity. At scale (100K entities, 5 components), that's 500K allocations on load. The `WireFormat` trait was designed with rkyv as the intended second implementation, but maintaining two serialization backends adds complexity without payoff since the persist crate has no external users yet.

## Decision

Replace serde/bincode entirely with rkyv as the sole serialization backend in `minkowski-persist`. Remove the `WireFormat` trait and the `<W>` type parameter from `Wal`, `Snapshot`, and `Durable`. Add `Snapshot::load` for mmap-based snapshot loading that accesses archived data directly.

**Key insight: the persist crate is pre-1.0 with no external consumers — this is the time to make a clean switch rather than maintaining dual-format complexity.**

Component types registered with `CodecRegistry` must derive `rkyv::{Archive, Serialize, Deserialize}` instead of `serde::{Serialize, Deserialize}`. For maximum zero-copy benefit, components should also be `#[repr(C)]`. The core ECS crate (`minkowski`) has zero changes.

## Alternatives Considered

- Dual-format (bincode default, rkyv opt-in behind feature flag) — rejected because two serialization paths with a `CodecFormat` trait + default type parameter + feature flags is engineering debt with no payoff when there are no existing users to protect
- mmap-backed BlobVec (in-place operation) — rejected because the complexity of aligning archive data to BlobVec's 64-byte columns, handling growth, and managing copy-on-write far outweighs the marginal benefit over memcpy into BlobVec
- serde-rkyv bridge (keep serde derives, route through rkyv) — rejected because it defeats the purpose: the bridge serializes through serde, losing all zero-copy benefit

## Consequences

- Persistent components derive `rkyv::{Archive, Serialize, Deserialize}` instead of `serde::{Serialize, Deserialize}` — same effort, different syntax
- `Wal`, `Snapshot`, `Durable` are no longer generic over a wire format — simpler types, simpler construction
- `Snapshot::load` enables mmap-based snapshot loading — archived component bytes are copied directly into BlobVec columns without per-value typed deserialization
- `serde` and `bincode` dependencies removed from the persist crate; `rkyv` and `memmap2` added
- Existing snapshot files are not forward-compatible — a one-time migration (re-save with the new format)
- `#[repr(C)]` on components is recommended but not required — rkyv works without it, just without byte-for-byte zero-copy for those types
