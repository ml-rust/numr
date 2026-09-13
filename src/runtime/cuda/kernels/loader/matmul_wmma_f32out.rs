//! Tensor-core WMMA GEMM launchers that write the F32 accumulator.
//!
//! The `matmul_wmma_*_f32out_*` kernels in `matmul_wmma.cu` are the plain
//! `matmul_wmma_*` kernels with a float `C`: same staging, same tensor-core
//! K loop, same F32 accumulator, and an identity store in place of the
//! narrowing one. `matmul_wide` launches them where
//! `matmul_wmma_policy::use_wmma` admits the shape. Launch geometry and tile
//! choice come from `matmul_wmma_tile`, so a shape picks the same tile with
//! either output width.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::matmul_wmma_tile::{select_wmma_tile, wmma_kernel_name_f32out, wmma_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Launch 2-D (non-batched) WMMA GEMM for F16 or BF16 operands with an F32
/// output: `C[M,N] = A[M,K] @ B[K,N]`, `C` in F32.
///
/// # Safety
///
/// `a_ptr` and `b_ptr` must address `M×K` and `K×N` elements of `dtype`, and
/// `c_ptr` `M×N` f32 elements. Any M, N, K >= 1 is accepted.
pub unsafe fn launch_matmul_wmma_f32out_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name_f32out("matmul_wmma", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1, tile);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA f32out matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM for F16 or BF16 operands with an F32 output.
///
/// # Safety
///
/// `a_ptr` and `b_ptr` must address `a_batch×M×K` and `b_batch×K×N` elements
/// of `dtype`, and `c_ptr` `batch×M×N` f32 elements. Any M, N, K >= 1 is
/// accepted.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_matmul_wmma_f32out_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name_f32out("matmul_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA f32out batched matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::matmul_wmma_tile::{WmmaTile, wmma_kernel_name_f32out};
    use crate::dtype::DType;

    /// The name matches what `DEFINE_WMMA_F32OUT` pastes together.
    #[test]
    fn f32out_kernel_names_match_the_instantiations() {
        assert_eq!(
            wmma_kernel_name_f32out("matmul_wmma", DType::F16, WmmaTile::Tile128),
            "matmul_wmma_f16_f32out_128x128"
        );
        assert_eq!(
            wmma_kernel_name_f32out("matmul_wmma_batched", DType::BF16, WmmaTile::Tile64),
            "matmul_wmma_batched_bf16_f32out_64x64"
        );
        assert_eq!(
            wmma_kernel_name_f32out("matmul_wmma", DType::BF16, WmmaTile::Tile128x64),
            "matmul_wmma_bf16_f32out_128x64"
        );
    }
}
