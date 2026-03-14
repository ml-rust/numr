//! Multi-device collective communication.

mod group;
#[cfg(feature = "distributed-gpu")]
mod hierarchical;
#[cfg(feature = "distributed")]
mod nexar;
#[cfg(feature = "distributed")]
mod nexar_compat;
mod noop;
mod traits;

#[cfg(feature = "distributed")]
pub use self::nexar::NexarNetCommunicator;
pub use group::{CommunicatorGroup, ParallelDim};
#[cfg(feature = "distributed-gpu")]
pub use hierarchical::HierarchicalCommunicator;
pub use noop::NoOpCommunicator;
pub use traits::{Communicator, ReduceOp, StreamSyncOps};
