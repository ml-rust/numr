//! NCCL collective communication for multi-GPU and multi-process training.

mod collectives;
mod comm;
mod convert;
mod id;
mod stream_sync;

pub use comm::NcclCommunicator;
pub use id::{NCCL_ID_BYTES, NcclId};
