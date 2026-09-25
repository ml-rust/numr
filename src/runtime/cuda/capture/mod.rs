//! CUDA graph capture exclusion on the shared per-device compute stream.

pub(crate) mod entry;
pub mod lock;
mod session;
mod stream;

pub use lock::{CapturePermit, DeviceCaptureLock, EnqueuePermit, thread_is_capturing};
pub use stream::{GuardedLaunchBuilder, GuardedStream};
