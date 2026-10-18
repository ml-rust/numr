//! In-process communicator: several threads in one process act as the ranks of a group.

mod comm;
mod host;
mod plan;
mod reduce;
mod shared;

pub use comm::{AbortOnPanic, LocalCommunicator};
