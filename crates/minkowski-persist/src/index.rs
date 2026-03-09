use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufWriter, Write};
use std::path::Path;

use minkowski::component::Component;
use minkowski::{BTreeIndex, ChangeTick, Entity, HashIndex, SpatialIndex};
use rkyv::api::high::HighValidator;
use rkyv::bytecheck::CheckBytes;
use rkyv::de::Pool;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::{rancor, Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Index file magic identifying the persistent index format.
#[allow(dead_code)]
const INDEX_MAGIC: [u8; 8] = *b"MK2INDXK";

/// Header size: magic (8) + CRC32 (4) + reserved (4) + length (8) = 24.
#[allow(dead_code)]
pub(crate) const INDEX_HEADER_SIZE: usize = 24;

#[derive(Debug, thiserror::Error)]
pub enum IndexPersistError {
    #[error("index I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("index format error: {0}")]
    Format(String),
}

/// A secondary index that can be saved to disk and loaded on recovery.
///
/// `save` is object-safe — [`AutoCheckpoint`](crate::AutoCheckpoint) can hold
/// registered indexes and call `save` on each. `load` is on the concrete type
/// (returns `Self`, not object-safe).
///
/// Writes use atomic rename: data goes to a `.tmp` file, then is renamed
/// to the final path. A crash during write cannot corrupt the previous file.
///
/// After loading, call [`SpatialIndex::update`] to catch up with
/// mutations that occurred after the index was last saved.
pub trait PersistentIndex: SpatialIndex {
    /// Serialize the index state to a file.
    fn save(&self, path: &Path) -> Result<(), IndexPersistError>;
}

/// Write an index envelope: `[magic 8B][crc32 4B][reserved 4B][len u64][payload]`.
/// Uses atomic rename — writes to `{path}.tmp`, then renames to `path`.
#[allow(dead_code)]
pub(crate) fn write_index_file(path: &Path, payload: &[u8]) -> Result<(), IndexPersistError> {
    let tmp_path = path.with_extension("idx.tmp");
    let crc = crc32fast::hash(payload);
    let len = payload.len() as u64;

    let file = File::create(&tmp_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&INDEX_MAGIC)?;
    writer.write_all(&crc.to_le_bytes())?;
    writer.write_all(&[0u8; 4])?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    drop(writer);

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Read and validate an index file. Returns the verified payload bytes.
#[allow(dead_code)]
pub(crate) fn read_index_file(path: &Path) -> Result<Vec<u8>, IndexPersistError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < INDEX_HEADER_SIZE {
        return Err(IndexPersistError::Format("index file too small".into()));
    }
    if bytes[..8] != INDEX_MAGIC {
        return Err(IndexPersistError::Format("invalid index file magic".into()));
    }
    let stored_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let end = INDEX_HEADER_SIZE
        .checked_add(len)
        .ok_or_else(|| IndexPersistError::Format("invalid payload length".into()))?;
    if bytes.len() < end {
        return Err(IndexPersistError::Format(format!(
            "index truncated: expected {len} payload bytes, got {}",
            bytes.len() - INDEX_HEADER_SIZE
        )));
    }
    let payload = &bytes[INDEX_HEADER_SIZE..end];
    let actual_crc = crc32fast::hash(payload);
    if actual_crc != stored_crc {
        return Err(IndexPersistError::Format(format!(
            "index checksum mismatch: expected {stored_crc:#010x}, got {actual_crc:#010x}"
        )));
    }
    Ok(payload.to_vec())
}

// ---------------------------------------------------------------------------
// rkyv-based save/load for BTreeIndex and HashIndex
// ---------------------------------------------------------------------------

/// Type-erased payload for index persistence. Keys are serialized individually
/// as byte blobs to avoid propagating rkyv generic bounds through the trait.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct IndexPayload {
    /// Forward map: serialized key bytes → entity bits.
    entries: Vec<(Vec<u8>, Vec<u64>)>,
    /// Reverse map: entity bits → serialized key bytes.
    reverse: Vec<(u64, Vec<u8>)>,
    /// Last sync tick (raw u64).
    last_sync_tick: u64,
}

/// Serialize a single key value to bytes via rkyv.
fn serialize_key<T>(key: &T) -> Result<Vec<u8>, IndexPersistError>
where
    T: Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, rancor::Error>,
        >,
{
    rkyv::to_bytes::<rancor::Error>(key)
        .map(|v| v.to_vec())
        .map_err(|e| IndexPersistError::Format(format!("key serialization failed: {e}")))
}

/// Deserialize a single key value from bytes via rkyv.
fn deserialize_key<T>(bytes: &[u8]) -> Result<T, IndexPersistError>
where
    T: Archive,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    rkyv::from_bytes::<T, rancor::Error>(bytes)
        .map_err(|e| IndexPersistError::Format(format!("key deserialization failed: {e}")))
}

impl<T> PersistentIndex for BTreeIndex<T>
where
    T: Component
        + Ord
        + Clone
        + Hash
        + Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, rancor::Error>,
        >,
{
    fn save(&self, path: &Path) -> Result<(), IndexPersistError> {
        let (tree, reverse, last_sync) = self.as_raw_parts();

        let mut entries = Vec::with_capacity(tree.len());
        for (key, entities) in tree {
            let key_bytes = serialize_key(key)?;
            let entity_bits: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
            entries.push((key_bytes, entity_bits));
        }

        let mut rev = Vec::with_capacity(reverse.len());
        for (entity, key) in reverse {
            let key_bytes = serialize_key(key)?;
            rev.push((entity.to_bits(), key_bytes));
        }

        let payload = IndexPayload {
            entries,
            reverse: rev,
            last_sync_tick: last_sync.to_raw(),
        };

        let bytes = rkyv::to_bytes::<rancor::Error>(&payload)
            .map_err(|e| IndexPersistError::Format(format!("payload serialization failed: {e}")))?;

        write_index_file(path, &bytes)
    }
}

/// Load a `BTreeIndex<T>` from a file previously written by
/// [`PersistentIndex::save`].
pub fn load_btree_index<T>(path: &Path) -> Result<BTreeIndex<T>, IndexPersistError>
where
    T: Component + Ord + Clone + Hash + Archive,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    let bytes = read_index_file(path)?;
    let payload = rkyv::from_bytes::<IndexPayload, rancor::Error>(&bytes)
        .map_err(|e| IndexPersistError::Format(format!("payload deserialization failed: {e}")))?;

    let mut tree = std::collections::BTreeMap::new();
    for (key_bytes, entity_bits) in &payload.entries {
        let key: T = deserialize_key(key_bytes)?;
        let entities: Vec<Entity> = entity_bits.iter().map(|&b| Entity::from_bits(b)).collect();
        tree.insert(key, entities);
    }

    let mut reverse = HashMap::new();
    for &(entity_bits, ref key_bytes) in &payload.reverse {
        let key: T = deserialize_key(key_bytes)?;
        reverse.insert(Entity::from_bits(entity_bits), key);
    }

    let last_sync = ChangeTick::from_raw(payload.last_sync_tick);
    Ok(BTreeIndex::from_raw_parts(tree, reverse, last_sync))
}

impl<T> PersistentIndex for HashIndex<T>
where
    T: Component
        + Hash
        + Eq
        + Clone
        + Archive
        + for<'a> RkyvSerialize<
            rkyv::api::high::HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, rancor::Error>,
        >,
{
    fn save(&self, path: &Path) -> Result<(), IndexPersistError> {
        let (map, reverse, last_sync) = self.as_raw_parts();

        let mut entries = Vec::with_capacity(map.len());
        for (key, entities) in map {
            let key_bytes = serialize_key(key)?;
            let entity_bits: Vec<u64> = entities.iter().map(|e| e.to_bits()).collect();
            entries.push((key_bytes, entity_bits));
        }

        let mut rev = Vec::with_capacity(reverse.len());
        for (entity, key) in reverse {
            let key_bytes = serialize_key(key)?;
            rev.push((entity.to_bits(), key_bytes));
        }

        let payload = IndexPayload {
            entries,
            reverse: rev,
            last_sync_tick: last_sync.to_raw(),
        };

        let bytes = rkyv::to_bytes::<rancor::Error>(&payload)
            .map_err(|e| IndexPersistError::Format(format!("payload serialization failed: {e}")))?;

        write_index_file(path, &bytes)
    }
}

/// Load a `HashIndex<T>` from a file previously written by
/// [`PersistentIndex::save`].
pub fn load_hash_index<T>(path: &Path) -> Result<HashIndex<T>, IndexPersistError>
where
    T: Component + Hash + Eq + Clone + Archive,
    T::Archived: RkyvDeserialize<T, rancor::Strategy<Pool, rancor::Error>>
        + for<'a> CheckBytes<HighValidator<'a, rancor::Error>>,
{
    let bytes = read_index_file(path)?;
    let payload = rkyv::from_bytes::<IndexPayload, rancor::Error>(&bytes)
        .map_err(|e| IndexPersistError::Format(format!("payload deserialization failed: {e}")))?;

    let mut map = HashMap::new();
    for (key_bytes, entity_bits) in &payload.entries {
        let key: T = deserialize_key(key_bytes)?;
        let entities: Vec<Entity> = entity_bits.iter().map(|&b| Entity::from_bits(b)).collect();
        map.insert(key, entities);
    }

    let mut reverse = HashMap::new();
    for &(entity_bits, ref key_bytes) in &payload.reverse {
        let key: T = deserialize_key(key_bytes)?;
        reverse.insert(Entity::from_bits(entity_bits), key);
    }

    let last_sync = ChangeTick::from_raw(payload.last_sync_tick);
    Ok(HashIndex::from_raw_parts(map, reverse, last_sync))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minkowski::World;

    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[repr(C)]
    struct Score(u32);

    #[test]
    fn btree_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("score.idx");

        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));
        let e3 = world.spawn((Score(10),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_btree_index::<Score>(&path).unwrap();
        assert_eq!(loaded.get(&Score(10)).len(), 2);
        assert!(loaded.get(&Score(10)).contains(&e1));
        assert!(loaded.get(&Score(10)).contains(&e3));
        assert_eq!(loaded.get(&Score(20)).len(), 1);
        assert!(loaded.get(&Score(20)).contains(&e2));
    }

    #[test]
    fn btree_empty_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.idx");

        let mut world = World::new();
        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_btree_index::<Score>(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn hash_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("score_hash.idx");

        let mut world = World::new();
        let e1 = world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = HashIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        let loaded = load_hash_index::<Score>(&path).unwrap();
        assert_eq!(loaded.get(&Score(10)).len(), 1);
        assert!(loaded.get(&Score(10)).contains(&e1));
        assert_eq!(loaded.get(&Score(20)).len(), 1);
        assert!(loaded.get(&Score(20)).contains(&e2));
    }

    #[test]
    fn write_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.idx");
        let payload = b"hello world";
        write_index_file(&path, payload).unwrap();
        let loaded = read_index_file(&path).unwrap();
        assert_eq!(loaded, payload);
    }

    #[test]
    fn read_missing_file() {
        let result = read_index_file(Path::new("/nonexistent/test.idx"));
        assert!(matches!(result, Err(IndexPersistError::Io(_))));
    }

    #[test]
    fn read_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.idx");
        std::fs::write(&path, [0u8; 10]).unwrap();
        let result = read_index_file(&path);
        assert!(matches!(result, Err(IndexPersistError::Format(_))));
    }

    #[test]
    fn read_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("badmagic.idx");
        let mut data = vec![0u8; INDEX_HEADER_SIZE + 4];
        data[..8].copy_from_slice(b"NOTMAGIC");
        std::fs::write(&path, &data).unwrap();
        let result = read_index_file(&path);
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("magic"));
    }

    #[test]
    fn read_crc_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crc.idx");
        let payload = b"test data";
        write_index_file(&path, payload).unwrap();

        let mut data = std::fs::read(&path).unwrap();
        data[INDEX_HEADER_SIZE] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = read_index_file(&path);
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("checksum"));
    }

    #[test]
    fn btree_stale_catch_up_after_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stale.idx");

        let mut world = World::new();
        world.spawn((Score(10),));
        let e2 = world.spawn((Score(20),));

        let mut idx = BTreeIndex::<Score>::new();
        idx.rebuild(&mut world);
        idx.save(&path).unwrap();

        // Mutate after save
        *world.get_mut::<Score>(e2).unwrap() = Score(30);
        world.spawn((Score(40),));

        // Load stale index, catch up
        let mut loaded = load_btree_index::<Score>(&path).unwrap();
        loaded.update(&mut world);

        // Verify: Score(20) gone, Score(30) and Score(40) present
        assert!(loaded.get(&Score(20)).is_empty());
        assert_eq!(loaded.get(&Score(30)).len(), 1);
        assert_eq!(loaded.get(&Score(40)).len(), 1);
        assert_eq!(loaded.get(&Score(10)).len(), 1);
    }

    #[test]
    fn atomic_rename_leaves_no_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atomic.idx");
        write_index_file(&path, b"data").unwrap();
        assert!(path.exists());
        assert!(!path.with_extension("idx.tmp").exists());
    }
}
