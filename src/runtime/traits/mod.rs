//! Runtime traits for compute backend abstraction

pub mod arithmetic;
pub mod client;
pub mod device;
pub mod fused_mul_add;
pub mod profile;
pub mod runtime;

pub use arithmetic::FloatArithmeticContract;
pub use client::RuntimeClient;
pub use device::Device;
pub use fused_mul_add::FusedMulAddContract;
pub use profile::{DeviceArch, DeviceCaps, DeviceProfile};
pub use runtime::Runtime;
