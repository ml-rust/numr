// Backend parity tests for ImageOps.

pub mod cpu;
pub mod cuda;
pub mod reference;
pub mod resize_cpu;
pub mod resize_cuda;
pub mod resize_golden;
pub mod resize_suite;
pub mod resize_suite_views;
pub mod resize_wgpu;
pub mod suite;
pub mod suite_into;
pub mod wgpu;
