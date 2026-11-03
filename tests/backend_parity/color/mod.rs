// Backend parity tests for ColorOps.

pub mod cpu;
pub mod cpu_packed;
pub mod cuda;
pub mod cuda_packed;
pub mod golden;
pub mod packed;
pub mod planes;
pub mod suite_errors;
pub mod suite_golden;
pub mod suite_packed;
pub mod suite_packed_errors;
pub mod suite_packed_rgb24;
pub mod suite_rgb24;
pub mod suite_roundtrip;
pub mod wgpu;
