//! WebGPU `ColorOps`.
//!
//! This backend takes the trait's default methods, so every call returns
//! `Error::NotImplemented`. The backend has no WGSL colour shaders.

use crate::ops::ColorOps;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};

/// ColorOps for WebGPU: every method returns `Error::NotImplemented`.
impl ColorOps<WgpuRuntime> for WgpuClient {}
