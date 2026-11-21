//! CPU correctly rounded F64 functions through the host lane kernels.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CorrectlyRoundedOps, require_correctly_rounded_contracts};
use crate::runtime::cpu::kernels::correctly_rounded::{CrFunction, LaneCopy, map_f64};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::{Device, ensure_contiguous};
use crate::tensor::Tensor;

/// Floating operations one fast-path lane runs, as the parallel gate counts work.
const LANE_WORK: usize = 32;

/// Elements per task when the client sets no chunk size.
const DEFAULT_CHUNK: usize = 4096;

impl CorrectlyRoundedOps<CpuRuntime> for CpuClient {
    fn cr_exp(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        map(self, a, CrFunction::Exp, "cr_exp")
    }

    fn cr_log(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        map(self, a, CrFunction::Log, "cr_log")
    }

    fn cr_log1p(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        map(self, a, CrFunction::Log1p, "cr_log1p")
    }

    fn cr_acos(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        map(self, a, CrFunction::Acos, "cr_acos")
    }

    fn cr_pow_scalar(&self, a: &Tensor<CpuRuntime>, exponent: f64) -> Result<Tensor<CpuRuntime>> {
        map(self, a, CrFunction::PowScalar(exponent), "cr_pow_scalar")
    }
}

fn map(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    function: CrFunction,
    op: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    if dtype != DType::F64 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    if !client.device.is_same(a.device()) {
        return Err(Error::DeviceMismatch);
    }
    require_correctly_rounded_contracts::<CpuRuntime>(op)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;
    let len = a.numel();
    if len == 0 {
        return Ok(out);
    }
    // CPU layout copies use integer loads, so NaN payloads reach the kernel unchanged.
    let input = ensure_contiguous(a)?;
    let src = input.ptr();
    let dst = out.ptr();
    let chunk = client
        .parallelism()
        .chunk_size
        .unwrap_or(DEFAULT_CHUNK)
        .max(1);
    client.par_for_each(
        len.div_ceil(chunk),
        len.saturating_mul(LANE_WORK),
        |index| {
            let start = index * chunk;
            let count = chunk.min(len - start);
            // SAFETY: both buffers hold `len` F64 elements, and each task reads and writes
            // the disjoint range `start..start + count` of them.
            let (input, output) = unsafe {
                (
                    std::slice::from_raw_parts((src as *const f64).add(start), count),
                    std::slice::from_raw_parts_mut((dst as *mut f64).add(start), count),
                )
            };
            map_f64(function, input, output, LaneCopy::Best);
        },
    );
    Ok(out)
}
