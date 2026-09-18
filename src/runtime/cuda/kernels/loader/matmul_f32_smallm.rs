//! Small-M FP32 launcher against a transposed weight.
//!
//! `matmul_f32_smallm_bt` and its batched form run one thread per output
//! element, for `M <= MAX_SMALL_M` rows against a `[N, K]` weight. The tiled
//! family's 16-row tile pads 12 to 15 of its rows at these shapes. The kernel
//! forms each output as the same ascending FMA chain the tiled kernels do, so
//! the bits are identical and a row's result stays independent of M.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::{BLOCK_SIZE, LaunchConfig, MAX_GRID_DIM_YZ};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Widest `M` the one-thread-per-output kernel serves; wider rows take the
/// tiled family.
///
/// The cutoff moves no bits. Every kernel on the F32 `x @ Wᵀ` path forms an
/// output element as one `fma` per k, k ascending, from a +0 accumulator
/// (`matmul_f32_tiled.cuh` by contraction of `acc += a * b`,
/// `matmul_f32_smallm.cuh` by explicit `fmaf`), and the tiled kernel's
/// padding k's are `fma(0, 0, acc) == acc`. So a row's result is the same
/// bits at every M whichever side of this cutoff it lands on, which is what
/// `tests/cuda_matmul_batch_invariance.rs` checks. The cutoff is speed only.
///
/// 4 is the widest M where one thread per output beats the 16-row tile on
/// `[M, 5120] x [5120, 48]ᵀ`. That figure is unconfirmed: the timing run
/// that fixes it is still to come. Measure both kernels before moving it.
pub const MAX_SMALL_M: usize = 4;

/// Widest `N` the one-thread-per-output kernel serves; wider weights take
/// the tiled family.
///
/// Adjacent threads read B rows `K` floats apart, so a warp's loads at one k
/// touch 32 lines and nothing coalesces; the tiled kernel stages B through
/// shared memory and does not pay that. Per launch on an RTX 3060 (nsys),
/// small-M kernel vs tiled, M <= 4:
///
/// | N       | K      | small-M | tiled   |
/// |---------|--------|---------|---------|
/// | 48, 96  | 1000   | 27 us   | 51 us   |
/// | 48, 96  | 5120   | 140 us  | 247 us  |
/// | 48, 96  | 17408  | 887 us  | 836 us  |
/// | 5120    | 5120   | 1120 us | 346 us  |
/// | 5120    | 17408  | 3790 us | 1160 us |
///
/// The kernel wins while the row count is small enough that its per-row
/// stream stays in cache and loses once the uncoalesced traffic dominates,
/// which on these shapes is past a few hundred rows. 256 is one block of
/// threads; the boundary between 96 and 5120 is unmeasured, so re-measure
/// before moving it.
pub const MAX_SMALL_N: usize = 256;

/// Threads per block, over `n`.
const SMALLM_BLOCK: u32 = BLOCK_SIZE;

/// Grid `(ceil(N/256), M, batch)`, block `(256, 1, 1)`, no shared memory.
#[inline]
fn smallm_launch_config(m: usize, n: usize, batch: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(SMALLM_BLOCK), m as u32, batch as u32),
        block_dim: (SMALLM_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Whether the small-M kernel takes this launch: F32, `1..=MAX_SMALL_M` rows,
/// `1..=MAX_SMALL_N` columns, and a batch the grid's `z` axis can hold.
///
/// `n == 0` is left to the tiled path so an empty output keeps the behaviour
/// it has there. `k == 0` is served: the kernel writes the +0 accumulator.
#[inline]
fn smallm_applies(dtype: DType, m: usize, n: usize, batch: usize) -> bool {
    dtype == DType::F32
        && (1..=MAX_SMALL_M).contains(&m)
        && (1..=MAX_SMALL_N).contains(&n)
        && (1..=MAX_GRID_DIM_YZ as usize).contains(&batch)
}

/// Launch `C[M,N] = A[M,K] · Bᵀ` with `b_ptr` the contiguous `[N, K]` matrix,
/// one thread per output. Returns `Ok(false)` when the shape or dtype is not
/// the small-M, small-N F32 case, so the caller falls through to the tiled
/// kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: N * K elements
/// - C: M * N elements
pub unsafe fn launch_matmul_smallm_bt_kernel(
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
) -> Result<bool> {
    if !smallm_applies(dtype, m, n, 1) {
        return Ok(false);
    }
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func = get_kernel_function(&module, "matmul_f32_smallm_bt")?;
    let cfg = smallm_launch_config(m, n, 1);

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
                "CUDA matmul F32 small-M kernel 'matmul_f32_smallm_bt' launch failed: {:?}",
                e
            ))
        })?;
    }
    Ok(true)
}

/// Batched form of [`launch_matmul_smallm_bt_kernel`]: `b_ptr` is
/// `[batch, N, K]`, and `a_batch` / `b_batch` broadcast an operand held once
/// over every batch (`b % count`, as the tiled batched kernels do). Returns
/// `Ok(false)` when the launch is not the small-M F32 case or `batch` exceeds
/// the grid's `z` extent.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: a_batch * M * K elements
/// - B: b_batch * N * K elements
/// - C: batch * M * N elements
pub unsafe fn launch_matmul_batched_smallm_bt_kernel(
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
) -> Result<bool> {
    if !smallm_applies(dtype, m, n, batch) {
        return Ok(false);
    }
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func = get_kernel_function(&module, "matmul_batched_f32_smallm_bt")?;
    let cfg = smallm_launch_config(m, n, batch);

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
                "CUDA batched matmul F32 small-M kernel 'matmul_batched_f32_smallm_bt' \
                 launch failed: {:?}",
                e
            ))
        })?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_config_is_one_block_row_per_m_and_256_threads_over_n() {
        let cfg = smallm_launch_config(4, 5120, 3);
        assert_eq!(cfg.grid_dim, (20, 4, 3));
        assert_eq!(cfg.block_dim, (256, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn launch_config_rounds_a_partial_block_up() {
        assert_eq!(smallm_launch_config(1, 48, 1).grid_dim, (1, 1, 1));
        assert_eq!(smallm_launch_config(1, 257, 1).grid_dim, (2, 1, 1));
    }

    #[test]
    fn applies_only_to_f32_at_one_to_max_rows() {
        assert!(smallm_applies(DType::F32, 1, 48, 1));
        assert!(smallm_applies(DType::F32, MAX_SMALL_M, 48, 1));
        assert!(!smallm_applies(DType::F32, MAX_SMALL_M + 1, 48, 1));
        assert!(!smallm_applies(DType::F32, 0, 48, 1));
        assert!(!smallm_applies(DType::F32, 1, 0, 1));
        assert!(smallm_applies(DType::F32, 1, MAX_SMALL_N, 1));
        assert!(!smallm_applies(DType::F32, 1, MAX_SMALL_N + 1, 1));
        assert!(!smallm_applies(DType::F16, 1, 48, 1));
        assert!(!smallm_applies(DType::F64, 1, 48, 1));
    }

    #[test]
    fn applies_only_when_the_batch_fits_grid_z() {
        assert!(smallm_applies(DType::F32, 1, 48, MAX_GRID_DIM_YZ as usize));
        assert!(!smallm_applies(
            DType::F32,
            1,
            48,
            MAX_GRID_DIM_YZ as usize + 1
        ));
        assert!(!smallm_applies(DType::F32, 1, 48, 0));
    }
}
