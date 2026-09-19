//! Small-M FP32 launcher against a transposed weight.
//!
//! `matmul_f32_smallm_bt` and its batched form run one thread per output
//! element, for `M <= MAX_SMALL_M` rows against a `[N, K]` weight. A block
//! owns `SMALLM_ROWS_PER_BLOCK` rows of the weight and stages them through
//! static shared memory with all 256 of its threads, so the weight stream is
//! spread over `ceil(N / SMALLM_ROWS_PER_BLOCK)` SMs. The tiled family's
//! 16-row tile pads 12 to 15 of its rows at these shapes. The kernel forms
//! each output as the same ascending FMA chain the tiled kernels do, so the
//! bits are identical and a row's result stays independent of M.
//!
//! The gate ([`smallm_applies`]) bounds the grid to a wave count of the
//! device's SMs; the count comes from a per-device probe
//! (`matmul_f32_smallm_tune.rs`) with a measured fallback.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaClient;

use super::launch_dims::{BLOCK_SIZE, LaunchConfig, MAX_GRID_DIM_YZ};
use super::matmul_f32_smallm_tune::smallm_max_waves;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Widest `M` the one-thread-per-output kernel serves; wider rows take the
/// tiled family. This is the range the wave rule ([`SmallmLimits`]) was
/// checked over, not a tuned cutoff: the rule itself decides inside it.
///
/// The cutoff moves no bits. Every kernel on the F32 `x @ Wᵀ` path forms an
/// output element as one `fma` per k, k ascending, from a +0 accumulator
/// (`matmul_f32_tiled.cuh` by contraction of `acc += a * b`,
/// `matmul_f32_smallm.cuh` by explicit `fmaf`), and the tiled kernel's
/// padding k's are `fma(0, 0, acc) == acc`. So a row's result is the same
/// bits at every M whichever side of this cutoff it lands on, which is what
/// `tests/cuda_matmul_batch_invariance.rs` checks. The cutoff is speed only.
///
/// 64 is the widest M measured (`examples/cuda_matmul_smallm_profile.rs`,
/// nsys medians of 8, K = 5120). Past 64 the tiled family's shape rule
/// moves off the 16-row tile, so the comparison is unmeasured.
pub const MAX_SMALL_M: usize = 64;

/// Widest `N` the one-thread-per-output kernel serves; wider weights take
/// the tiled family. This is the range the wave rule ([`SmallmLimits`])
/// was checked over, not a tuned cutoff: the rule itself decides inside it.
///
/// The tiled kernel reuses each staged B element across its 16 A rows, so
/// past a few thousand outputs it moves less memory per output. Measured
/// with `examples/cuda_matmul_smallm_profile.rs` (nsys medians of 8, K =
/// 5120) across M in {1, 4, 16, 64} and N in {48, 256, 512, 1024, 5120}:
/// the tiled kernel is flat in N over this range while small-M grows with
/// `M x N`, so the wave bound in [`SmallmLimits`] is the cutoff that
/// matters.
pub const MAX_SMALL_N: usize = 1024;

/// The device-side bounds [`smallm_applies`] checks the grid against: the
/// launch applies when `ceil(N / SMALLM_ROWS_PER_BLOCK) x M` is at most
/// `max_waves x sm_count`.
///
/// The small-M kernel runs one block of 256 threads per
/// `SMALLM_ROWS_PER_BLOCK` rows of B per A row, each block streaming its
/// rows in full, so its time grows with the block count while the tiled
/// kernel stays flat over the range in the table on [`MAX_SMALL_N`]; the
/// crossover is a block count per SM, a wave count. `max_waves` is that
/// count for this device: probed once per device by
/// [`smallm_max_waves`], or `SMALLM_MAX_WAVES_FALLBACK` when tuning is off
/// or the probe fails. Both kernels give identical bits, so a wrong bound
/// costs speed only. The batch axis is not in the bound: it multiplies both
/// kernels' grids alike and is unmeasured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallmLimits {
    /// The device's `compute_units`; 0 (profile not read) admits nothing.
    pub sm_count: usize,
    /// Most waves of small-M blocks over `sm_count` the kernel serves.
    pub max_waves: usize,
}

impl SmallmLimits {
    /// The limits for `client`'s device: SM count from the cached profile,
    /// wave bound from the per-device tune cache. The first call on a
    /// device runs the probe (a few ms); later calls read the cache.
    pub fn of(client: &CudaClient) -> Self {
        Self {
            sm_count: client.device.profile().compute_units as usize,
            max_waves: smallm_max_waves(client),
        }
    }

    /// Most small-M blocks these limits admit.
    #[inline]
    fn max_blocks(self) -> usize {
        self.max_waves * self.sm_count
    }
}

/// Threads per block: the first `SMALLM_ROWS_PER_BLOCK` own an output, all
/// of them load the block's B rows.
const SMALLM_BLOCK: u32 = BLOCK_SIZE;

/// Rows of B, so output columns, per block; `SMALLM_ROWS` in
/// `matmul_f32_smallm.cuh`. Small so that a narrow weight still spreads
/// over several SMs: N = 48 runs six blocks, N = `MAX_SMALL_N` thirty-two.
pub const SMALLM_ROWS_PER_BLOCK: u32 = 8;

/// Grid `(ceil(N / SMALLM_ROWS_PER_BLOCK), M, batch)`, block `(256, 1, 1)`.
/// The kernel's shared tile is static (`SMALLM_SMEM_FLOATS` in
/// `matmul_f32_smallm.cuh`), so no dynamic shared memory is requested.
#[inline]
fn smallm_launch_config(m: usize, n: usize, batch: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(SMALLM_ROWS_PER_BLOCK),
            m as u32,
            batch as u32,
        ),
        block_dim: (SMALLM_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Small-M blocks of `[M, K] x [N, K]ᵀ`: one per `SMALLM_ROWS_PER_BLOCK`
/// columns per row.
#[inline]
pub fn smallm_block_count(m: usize, n: usize) -> usize {
    n.div_ceil(SMALLM_ROWS_PER_BLOCK as usize) * m
}

/// Whether the small-M kernel takes this launch: F32, `1..=MAX_SMALL_M` rows,
/// `1..=MAX_SMALL_N` columns, a grid within `limits`, and a batch the
/// grid's `z` axis can hold.
///
/// `n == 0` is left to the tiled path so an empty output keeps the
/// behaviour it has there. `k == 0` is served: the kernel writes the +0
/// accumulator.
#[inline]
pub fn smallm_applies(
    dtype: DType,
    m: usize,
    n: usize,
    batch: usize,
    limits: SmallmLimits,
) -> bool {
    dtype == DType::F32
        && (1..=MAX_SMALL_M).contains(&m)
        && (1..=MAX_SMALL_N).contains(&n)
        && smallm_block_count(m, n) <= limits.max_blocks()
        && (1..=MAX_GRID_DIM_YZ as usize).contains(&batch)
}

/// Launch `C[M,N] = A[M,K] · Bᵀ` with `b_ptr` the contiguous `[N, K]` matrix,
/// one thread per output. Returns `Ok(false)` when [`smallm_applies`] says
/// no for this client's device, so the caller falls through to the tiled
/// kernel. The first call on a device runs the wave probe (a few ms);
/// blazr's warmup absorbs that.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: N * K elements
/// - C: M * N elements
pub unsafe fn launch_matmul_smallm_bt_kernel(
    client: &CudaClient,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    if !smallm_applies(dtype, m, n, 1, SmallmLimits::of(client)) {
        return Ok(false);
    }
    unsafe {
        launch_matmul_smallm_bt_f32_ungated(
            client.context(),
            client.stream(),
            client.device.index,
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
        )?;
    }
    Ok(true)
}

/// Launch `matmul_f32_smallm_bt` on F32 operands with no gate: any `M`,
/// `N` and `K` the grid holds. The wave probe and the parity tests use it
/// to run the kernel on shapes the gate declines; the matmul dispatch never
/// does.
///
/// # Safety
///
/// As [`launch_matmul_smallm_bt_kernel`], and the operands are F32.
pub unsafe fn launch_matmul_smallm_bt_f32_ungated(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
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
    Ok(())
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
    client: &CudaClient,
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
    if !smallm_applies(dtype, m, n, batch, SmallmLimits::of(client)) {
        return Ok(false);
    }
    let context = client.context();
    let stream = client.stream();
    let module = get_or_load_module(context, client.device.index, kernel_names::MATMUL_MODULE)?;
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
    fn launch_config_is_one_block_per_row_group_and_m_with_256_threads() {
        let cfg = smallm_launch_config(4, 5120, 3);
        assert_eq!(cfg.grid_dim, (640, 4, 3));
        assert_eq!(cfg.block_dim, (256, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn launch_config_rounds_a_partial_row_group_up() {
        assert_eq!(smallm_launch_config(1, 48, 1).grid_dim, (6, 1, 1));
        assert_eq!(smallm_launch_config(1, 49, 1).grid_dim, (7, 1, 1));
        assert_eq!(smallm_launch_config(1, 1, 1).grid_dim, (1, 1, 1));
        assert_eq!(
            smallm_launch_config(1, MAX_SMALL_N, 1).grid_dim,
            (128, 1, 1)
        );
    }

    /// Fixture limits for this test module: 28 SMs, 12 waves.
    const LIMITS: SmallmLimits = SmallmLimits {
        sm_count: 28,
        max_waves: 12,
    };

    #[test]
    fn applies_only_to_f32_at_one_to_max_rows() {
        assert!(smallm_applies(DType::F32, 1, 48, 1, LIMITS));
        assert!(smallm_applies(DType::F32, MAX_SMALL_M, 40, 1, LIMITS));
        assert!(!smallm_applies(DType::F32, MAX_SMALL_M + 1, 1, 1, LIMITS));
        assert!(!smallm_applies(DType::F32, 0, 48, 1, LIMITS));
        assert!(!smallm_applies(DType::F32, 1, 0, 1, LIMITS));
        assert!(smallm_applies(DType::F32, 1, MAX_SMALL_N, 1, LIMITS));
        assert!(!smallm_applies(DType::F32, 1, MAX_SMALL_N + 1, 1, LIMITS));
        assert!(!smallm_applies(DType::F16, 1, 48, 1, LIMITS));
        assert!(!smallm_applies(DType::F64, 1, 48, 1, LIMITS));
    }

    #[test]
    fn block_count_is_row_groups_times_m() {
        assert_eq!(smallm_block_count(1, 48), 6);
        assert_eq!(smallm_block_count(64, 41), 6 * 64);
        assert_eq!(smallm_block_count(3, MAX_SMALL_N), 384);
    }

    #[test]
    fn applies_only_within_the_wave_bound() {
        // 28 SMs x 12 waves = 336 blocks. 2 x 1024: 256 blocks fit; 3 x 1024:
        // 384 do not. 64 x 40: 320 fit; 64 x 48: 384 do not.
        assert!(smallm_applies(DType::F32, 2, MAX_SMALL_N, 1, LIMITS));
        assert!(!smallm_applies(DType::F32, 3, MAX_SMALL_N, 1, LIMITS));
        assert!(smallm_applies(DType::F32, MAX_SMALL_M, 40, 1, LIMITS));
        assert!(!smallm_applies(DType::F32, MAX_SMALL_M, 48, 1, LIMITS));
        // A partial row group is a whole block: 41 columns are 6 blocks.
        assert!(!smallm_applies(DType::F32, MAX_SMALL_M, 41, 1, LIMITS));
        // More SMs or more waves admit more blocks; an unread profile
        // admits none.
        let wider = SmallmLimits {
            sm_count: 32,
            ..LIMITS
        };
        assert!(smallm_applies(DType::F32, MAX_SMALL_M, 48, 1, wider));
        let deeper = SmallmLimits {
            max_waves: 14,
            ..LIMITS
        };
        assert!(smallm_applies(DType::F32, MAX_SMALL_M, 48, 1, deeper));
        let unread = SmallmLimits {
            sm_count: 0,
            ..LIMITS
        };
        assert!(!smallm_applies(DType::F32, 1, 1, 1, unread));
    }

    #[test]
    fn applies_only_when_the_batch_fits_grid_z() {
        let z = MAX_GRID_DIM_YZ as usize;
        assert!(smallm_applies(DType::F32, 1, 48, z, LIMITS));
        assert!(!smallm_applies(DType::F32, 1, 48, z + 1, LIMITS));
        assert!(!smallm_applies(DType::F32, 1, 48, 0, LIMITS));
    }
}
