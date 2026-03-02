pub mod codec;
pub mod format;
pub mod record;

pub use codec::{CodecError, CodecRegistry};
pub use format::{Bincode, WireFormat};
pub use record::*;
