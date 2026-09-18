//! WebGPU Walsh-Hadamard transform. Kernel lands in the next unit.

use crate::ops::FwhtOps;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl FwhtOps<WgpuRuntime> for WgpuClient {}
