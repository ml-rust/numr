//! CUDA Walsh-Hadamard transform. Kernel lands in the next unit.

use crate::ops::FwhtOps;
use crate::runtime::cuda::{CudaClient, CudaRuntime};

impl FwhtOps<CudaRuntime> for CudaClient {}
