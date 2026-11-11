//! CPU adjacent encodings through integer kernels.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::NextRepresentableOps;
use crate::runtime::cpu::kernels::simd::detect_simd;
use crate::runtime::cpu::{CpuClient, CpuRuntime, kernels};
use crate::runtime::{Device, ensure_contiguous};
use crate::tensor::Tensor;

impl NextRepresentableOps<CpuRuntime> for CpuClient {
    fn next_up(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        next_representable(self, a, true)
    }

    fn next_down(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        next_representable(self, a, false)
    }
}

fn next_representable(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    up: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let op = if up { "next_up" } else { "next_down" };
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    if !client.device.is_same(a.device()) {
        return Err(Error::DeviceMismatch);
    }
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;
    let len = a.numel();
    if len == 0 {
        return Ok(out);
    }
    // CPU layout copies use integer loads, preserving signaling NaNs and subnormals.
    let input = ensure_contiguous(a)?;
    let src = input.ptr();
    let dst = out.ptr();
    let level = detect_simd();
    let chunk = client.parallelism().chunk_size.unwrap_or(4096).max(1);
    client.par_for_each(len.div_ceil(chunk), len, |index| {
        let start = index * chunk;
        let count = chunk.min(len - start);
        // Each task writes a disjoint range of the fresh output allocation.
        unsafe {
            match dtype {
                DType::F32 => kernels::next_representable::next_f32(
                    (src as *const u32).add(start),
                    (dst as *mut u32).add(start),
                    count,
                    up,
                    level,
                ),
                DType::F64 => kernels::next_representable::next_f64(
                    (src as *const u64).add(start),
                    (dst as *mut u64).add(start),
                    count,
                    up,
                    level,
                ),
                _ => {}
            }
        }
    });
    Ok(out)
}
