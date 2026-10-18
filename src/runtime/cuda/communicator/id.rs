//! NCCL unique id: the bootstrap token every rank of one communicator shares.

use std::ffi::c_char;
use std::fmt;

use cudarc::nccl::{self, sys as nccl_sys};

use super::convert::nccl_err;
use crate::error::{Error, Result};

/// Size of an NCCL unique id in bytes.
pub const NCCL_ID_BYTES: usize = 128;

/// The unique id that joins ranks into one NCCL communicator.
///
/// One rank creates it with [`NcclId::new`]. The launcher sends [`NcclId::bytes`]
/// to every other rank over a file or socket. Each other rank rebuilds it with
/// [`NcclId::from_bytes`] or [`NcclId::from_slice`]. All ranks then call
/// [`NcclCommunicator::from_rank`](super::NcclCommunicator::from_rank) with it.
///
/// The bytes hold the bootstrap address of the creating rank. `Debug` omits them.
#[derive(Clone, Copy)]
pub struct NcclId {
    id: nccl::Id,
}

impl NcclId {
    /// Create a new unique id. Call this on one rank only.
    pub fn new() -> Result<Self> {
        let id = nccl::Id::new().map_err(|e| Error::Backend(nccl_err("ncclGetUniqueId", e)))?;
        Ok(Self { id })
    }

    /// Rebuild an id from the bytes another rank sent.
    pub fn from_bytes(bytes: [u8; NCCL_ID_BYTES]) -> Self {
        // c_char is i8 or u8 by target. Native-endian byte conversion keeps the bit pattern on both.
        let internal = bytes.map(|b| c_char::from_ne_bytes([b]));
        Self {
            id: nccl::Id::uninit(internal),
        }
    }

    /// Rebuild an id from a byte slice read from a file or socket.
    ///
    /// Returns `InvalidArgument` when the slice is not exactly [`NCCL_ID_BYTES`] long.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let array: [u8; NCCL_ID_BYTES] =
            bytes.try_into().map_err(|_| Error::InvalidArgument {
                arg: "bytes",
                reason: format!(
                    "NCCL unique id needs {NCCL_ID_BYTES} bytes, got {}; send the full output of NcclId::bytes",
                    bytes.len()
                ),
            })?;
        Ok(Self::from_bytes(array))
    }

    /// The id as bytes, to send to the other ranks.
    pub fn bytes(&self) -> [u8; NCCL_ID_BYTES] {
        self.id.internal().map(|c| c.to_ne_bytes()[0])
    }

    /// The id in the layout NCCL's init call takes.
    pub(super) fn raw(&self) -> nccl_sys::ncclUniqueId {
        nccl_sys::ncclUniqueId {
            internal: *self.id.internal(),
        }
    }
}

impl PartialEq for NcclId {
    fn eq(&self, other: &Self) -> bool {
        self.id.internal() == other.id.internal()
    }
}

impl Eq for NcclId {}

impl fmt::Debug for NcclId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NcclId").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_round_trip() {
        let mut bytes = [0u8; NCCL_ID_BYTES];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        // Includes bytes >= 0x80, which are negative where c_char is i8.
        bytes[0] = 0xFF;
        bytes[1] = 0x80;
        let id = NcclId::from_bytes(bytes);
        assert_eq!(id.bytes(), bytes);
        assert_eq!(id.raw().internal.map(|c| c.to_ne_bytes()[0]), bytes);
    }

    #[test]
    fn test_from_slice_round_trip() {
        let bytes = [0xA5u8; NCCL_ID_BYTES];
        let id = NcclId::from_slice(&bytes).unwrap();
        assert_eq!(id, NcclId::from_bytes(bytes));
    }

    #[test]
    fn test_from_slice_rejects_wrong_length() {
        assert!(NcclId::from_slice(&[0u8; NCCL_ID_BYTES - 1]).is_err());
        assert!(NcclId::from_slice(&[0u8; NCCL_ID_BYTES + 1]).is_err());
        assert!(NcclId::from_slice(&[]).is_err());
    }

    #[test]
    fn test_debug_hides_bytes() {
        let id = NcclId::from_bytes([0x41u8; NCCL_ID_BYTES]);
        assert_eq!(format!("{id:?}"), "NcclId { .. }");
    }

    #[test]
    fn test_new_id_round_trips_through_bytes() {
        // ncclGetUniqueId needs the NCCL library, not a GPU.
        let id = NcclId::new().unwrap();
        let rebuilt = NcclId::from_slice(&id.bytes()).unwrap();
        assert_eq!(rebuilt, id);
        assert_eq!(rebuilt.raw(), id.raw());
    }
}
