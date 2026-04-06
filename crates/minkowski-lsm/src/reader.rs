use std::fs;
use std::path::Path;

use crate::error::LsmError;
use crate::format::*;
use crate::schema::SchemaSection;

/// A reference to a page within a sorted run.
pub struct PageRef<'a> {
    /// Header describing the page metadata.
    pub header: &'a PageHeader,
    /// Raw page data bytes (full `PAGE_SIZE * item_size`, zero-padded).
    pub data: &'a [u8],
}

// ── Internal mmap abstraction ───────────────────────────────────────────────

enum MappedData {
    #[cfg(not(miri))]
    Mmap(memmap2::Mmap),
    /// Used under Miri (which cannot mmap) and as a fallback.
    #[expect(dead_code, reason = "constructed only under cfg(miri)")]
    Vec(Vec<u8>),
}

impl MappedData {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(miri))]
            Self::Mmap(m) => m.as_ref(),
            Self::Vec(v) => v,
        }
    }
}

/// Reader for sorted run files. Memory-maps the file for zero-copy access.
pub struct SortedRunReader {
    data: MappedData,
    schema: SchemaSection,
    index: Vec<IndexEntry>,
    sequence_range: (u64, u64),
    page_count: u64,
}

/// Minimum file size: Header (64) + Footer (64).
const MIN_FILE_SIZE: usize = 128;

/// Parsed metadata extracted from a file buffer during validation.
struct ParsedMetadata {
    schema: SchemaSection,
    index: Vec<IndexEntry>,
    sequence_range: (u64, u64),
    page_count: u64,
}

/// Validate and parse a sorted-run file buffer.
///
/// Separated from `SortedRunReader::open` so that the borrow on `buf` does not
/// prevent moving `MappedData` into `Self`.
fn validate_and_parse(buf: &[u8]) -> Result<ParsedMetadata, LsmError> {
    // 2. Validate minimum size.
    if buf.len() < MIN_FILE_SIZE {
        return Err(LsmError::Format(format!(
            "file too small: {} bytes (minimum {MIN_FILE_SIZE})",
            buf.len(),
        )));
    }

    // 3. Read header, validate magic and version.
    let header = Header::from_bytes(buf[..64].try_into().expect("slice is 64 bytes"));

    if header.magic != MAGIC {
        return Err(LsmError::Format(format!(
            "bad magic: expected {MAGIC:?}, got {:?}",
            header.magic
        )));
    }
    if header.version != VERSION {
        return Err(LsmError::Format(format!(
            "unsupported version: expected {VERSION}, got {}",
            header.version
        )));
    }

    // 4. Validate header CRC (covers first 40 bytes).
    let computed_crc = crc32fast::hash(&buf[..40]);
    if header.header_crc32 != computed_crc {
        return Err(LsmError::Crc {
            offset: 0,
            expected: header.header_crc32,
            actual: computed_crc,
        });
    }

    let schema_count = header.schema_count;
    let sequence_range = (header.sequence_lo, header.sequence_hi);
    let page_count = header.page_count;

    // 5. Read footer (last 64 bytes).
    let footer_start = buf.len() - 64;
    let footer = Footer::from_bytes(
        buf[footer_start..footer_start + 64]
            .try_into()
            .expect("64 bytes"),
    );

    // 6. Read schema section.
    let schema_data = &buf[footer.schema_offset as usize..];
    let schema = SchemaSection::read_from(schema_data, schema_count)?;

    // 7. Parse sparse index.
    let idx_offset = footer.sparse_index_offset as usize;
    let idx_count = footer.sparse_index_count as usize;
    let idx_byte_len = idx_count * std::mem::size_of::<IndexEntry>();

    if idx_offset + idx_byte_len > buf.len() {
        return Err(LsmError::Format(
            "sparse index extends beyond file".to_owned(),
        ));
    }

    let mut index = Vec::with_capacity(idx_count);
    for i in 0..idx_count {
        let entry_start = idx_offset + i * 16;
        let entry_bytes: &[u8; 16] = buf[entry_start..entry_start + 16]
            .try_into()
            .expect("16 bytes");
        index.push(*IndexEntry::from_bytes(entry_bytes));
    }

    Ok(ParsedMetadata {
        schema,
        index,
        sequence_range,
        page_count,
    })
}

impl SortedRunReader {
    /// Open and validate a sorted run file.
    pub fn open(path: &Path) -> Result<Self, LsmError> {
        let data = Self::map_file(path)?;
        let parsed = validate_and_parse(data.as_slice())?;

        Ok(Self {
            data,
            schema: parsed.schema,
            index: parsed.index,
            sequence_range: parsed.sequence_range,
            page_count: parsed.page_count,
        })
    }

    /// Look up a page by `(arch_id, slot, page_index)`.
    ///
    /// Returns `None` if the page is not in this sorted run.
    pub fn get_page(&self, arch_id: u16, slot: u16, page_index: u16) -> Option<PageRef<'_>> {
        let key = (arch_id, slot, page_index);
        let pos = self
            .index
            .binary_search_by_key(&key, |e| (e.arch_id, e.slot, e.page_index))
            .ok()?;

        let entry = &self.index[pos];
        let buf = self.data.as_slice();
        let offset = entry.file_offset as usize;

        // Read PageHeader.
        let header_bytes: &[u8; 16] = buf[offset..offset + 16].try_into().expect("16 bytes");
        let header = PageHeader::from_bytes(header_bytes);

        // Compute data length.
        let item_size = self.item_size_for_slot(slot);
        let data_len = PAGE_SIZE * item_size;

        let data_start = offset + std::mem::size_of::<PageHeader>();
        let data = &buf[data_start..data_start + data_len];

        Some(PageRef { header, data })
    }

    /// Validate the CRC of a specific page.
    ///
    /// The CRC covers `row_count * item_size` bytes (the actual data, not
    /// zero-padding).
    pub fn validate_page_crc(&self, page: &PageRef<'_>) -> Result<(), LsmError> {
        let item_size = self.item_size_for_slot(page.header.slot);
        let actual_len = page.header.row_count as usize * item_size;
        let computed = crc32fast::hash(&page.data[..actual_len]);

        if computed != page.header.page_crc32 {
            return Err(LsmError::Crc {
                offset: 0,
                expected: page.header.page_crc32,
                actual: computed,
            });
        }
        Ok(())
    }

    /// Get the schema section.
    pub fn schema(&self) -> &SchemaSection {
        &self.schema
    }

    /// WAL sequence range covered by this sorted run.
    pub fn sequence_range(&self) -> (u64, u64) {
        self.sequence_range
    }

    /// Total number of pages in this sorted run.
    pub fn page_count(&self) -> u64 {
        self.page_count
    }

    /// Number of entries in the sparse index.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Item size in bytes for a given slot.
    fn item_size_for_slot(&self, slot: u16) -> usize {
        if slot == ENTITY_SLOT {
            std::mem::size_of::<u64>()
        } else {
            self.schema
                .entry_for_slot(slot)
                .expect("slot must exist in schema")
                .item_size as usize
        }
    }

    /// Memory-map the file (or read into Vec under Miri).
    fn map_file(path: &Path) -> Result<MappedData, LsmError> {
        #[cfg(miri)]
        {
            let bytes = fs::read(path)?;
            Ok(MappedData::Vec(bytes))
        }
        #[cfg(not(miri))]
        {
            let file = fs::File::open(path)?;
            // SAFETY: The file is opened read-only and we do not modify it.
            // The mapping is valid for the lifetime of `SortedRunReader`.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            Ok(MappedData::Mmap(mmap))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::flush;
    use minkowski::World;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    /// Helper: spawn entities and flush to a sorted run file.
    fn flush_world_with_pos(n: usize) -> (tempfile::TempDir, std::path::PathBuf, World) {
        let mut world = World::new();
        for i in 0..n {
            world.spawn((Pos {
                x: i as f32,
                y: i as f32 * 2.0,
            },));
        }
        let dir = tempfile::tempdir().unwrap();
        let path = flush(&world, (10, 20), dir.path()).unwrap().unwrap();
        (dir, path, world)
    }

    #[test]
    fn open_valid_file() {
        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        assert_eq!(reader.sequence_range(), (10, 20));
        assert_eq!(reader.schema().len(), 1); // Pos only
        assert!(reader.page_count() > 0);
        assert!(reader.index_len() > 0);
    }

    #[test]
    fn get_page_returns_data() {
        let (_dir, path, _world) = flush_world_with_pos(3);
        let reader = SortedRunReader::open(&path).unwrap();

        // Find the first index entry to know a valid key.
        assert!(reader.index_len() > 0);
        let entry = &reader.index[0];
        let page = reader
            .get_page(entry.arch_id, entry.slot, entry.page_index)
            .expect("page should exist");

        assert!(page.header.row_count > 0);
        assert!(!page.data.is_empty());
    }

    #[test]
    fn get_nonexistent_page_returns_none() {
        let (_dir, path, _world) = flush_world_with_pos(1);
        let reader = SortedRunReader::open(&path).unwrap();

        // Use an arch_id that cannot exist.
        let result = reader.get_page(255, 255, 255);
        assert!(result.is_none());
    }

    #[test]
    fn validate_page_crc_succeeds() {
        let (_dir, path, _world) = flush_world_with_pos(5);
        let reader = SortedRunReader::open(&path).unwrap();

        // Validate CRC of every page in the index.
        for entry in &reader.index {
            let page = reader
                .get_page(entry.arch_id, entry.slot, entry.page_index)
                .unwrap();
            reader.validate_page_crc(&page).unwrap();
        }
    }

    #[test]
    fn corrupted_magic_returns_error() {
        let (_dir, path, _world) = flush_world_with_pos(1);

        // Corrupt the first byte.
        let mut data = std::fs::read(&path).unwrap();
        data[0] = 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = SortedRunReader::open(&path);
        assert!(
            matches!(result, Err(LsmError::Format(_))),
            "expected Format error for corrupted magic"
        );
    }

    #[test]
    fn multi_component_index_lookup() {
        let mut world = World::new();
        for i in 0..10 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Vel {
                    dx: 1.0,
                    dy: i as f32,
                },
            ));
        }

        let dir = tempfile::tempdir().unwrap();
        let path = flush(&world, (0, 100), dir.path()).unwrap().unwrap();
        let reader = SortedRunReader::open(&path).unwrap();

        // Verify schema has 2 component entries.
        assert_eq!(reader.schema().len(), 2);

        // Verify every index entry is findable.
        for entry in &reader.index {
            let page = reader
                .get_page(entry.arch_id, entry.slot, entry.page_index)
                .expect("every indexed page must be findable");
            reader
                .validate_page_crc(&page)
                .expect("every page CRC must be valid");
        }

        // Verify entity pages are present (slot == ENTITY_SLOT).
        let has_entity_page = reader.index.iter().any(|e| e.slot == ENTITY_SLOT);
        assert!(has_entity_page, "must have at least one entity page");
    }
}
