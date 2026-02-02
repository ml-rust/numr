//! Runtime traits for compute backend abstraction

pub mod client;
pub mod device;
pub mod runtime;

pub use client::RuntimeClient;
pub use device::Device;
pub use runtime::Runtime;
