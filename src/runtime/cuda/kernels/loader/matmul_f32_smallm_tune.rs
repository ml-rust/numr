//! Per-device wave bound for the small-M transposed-weight kernel.
//!
//! `matmul_f32_smallm_bt` wins while its grid is a few waves of the
//! device's SMs and loses once the tiled kernel's B reuse pays off; where
//! the crossover sits depends on the part. [`smallm_max_waves`] probes it
//! once per device through `tune::tuned`: a ladder of shapes spanning the
//! wave range, each timed on both kernels, and the wave count of the last
//! rung that wins before the first that loses. Both kernels give identical
//! bits (`tests/cuda_matmul_smallm_bt_parity.rs`), so the pick is speed
//! only.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::tune::{time_launches, tuned};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

use super::matmul::launch_matmul_kernel_bt;
use super::matmul_f32_smallm::{launch_matmul_smallm_bt_f32_ungated, smallm_block_count};

/// Wave bound when tuning is off (`NUMR_CUDA_TUNE=0`) or the probe fails.
///
/// Measured on one Ampere-class part (`examples/cuda_matmul_smallm_profile.rs`,
/// nsys medians of 8, K = 5120): every product `M x N` at or under 3072
/// outputs wins, and 3072 outputs are 384 blocks, 13.7 waves of that
/// part's 28 SMs; 12 rounds down for safety. On other parts it is a
/// heuristic, which is why the probe exists.
pub const SMALLM_MAX_WAVES_FALLBACK: usize = 12;

/// Tune-cache key of the wave bound.
pub const SMALLM_MAX_WAVES_KEY: &str = "matmul_f32_smallm_bt.max_waves";

/// Largest wave bound the probe returns: past it every shape inside
/// `MAX_SMALL_M x MAX_SMALL_N` is admitted on any part with 128 SMs or more.
pub const SMALLM_MAX_WAVES_CEILING: usize = 64;

/// Depth of every probe rung: the FFN width the kernel was measured at.
const PROBE_K: usize = 5120;

/// Timed iterations per kernel per rung; `time_launches` takes the minimum.
const PROBE_ITERS: usize = 3;

/// `(N, M)` rungs spanning the wave range: 6, 32 and 128 blocks per A row
/// at M from 1 to 64, so 48 to 512 blocks. The largest rung sizes the
/// probe's buffers: `A [64, K]`, `W [1024, K]` and `C [64, 1024]` come to
/// 22 MB, allocated once and freed when the probe returns.
const PROBE_RUNGS: [(usize, usize); 10] = [
    (48, 8),
    (48, 16),
    (48, 32),
    (48, 64),
    (256, 4),
    (256, 8),
    (256, 16),
    (1024, 1),
    (1024, 2),
    (1024, 4),
];

/// The wave bound for `client`'s device.
///
/// The first call on a device runs the probe, about ten launches of each
/// kernel per rung, a few ms in all; blazr's warmup absorbs it. Later calls
/// read the per-device tune cache. With tuning off, or when the probe
/// returns `Err`, this is [`SMALLM_MAX_WAVES_FALLBACK`].
pub fn smallm_max_waves(client: &CudaClient) -> usize {
    tuned(
        client,
        SMALLM_MAX_WAVES_KEY,
        SMALLM_MAX_WAVES_FALLBACK,
        || probe_max_waves(client),
    )
}

/// One probed shape: its grid in waves of the device, and whether small-M
/// ran no slower than tiled.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rung {
    waves: f64,
    wins: bool,
}

/// Time both kernels on every rung and reduce to the wave bound.
///
/// The buffers are left uninitialized: FMA throughput does not depend on
/// the values, and nothing reads the product.
fn probe_max_waves(client: &CudaClient) -> Result<usize> {
    let sm_count = client.device.profile().compute_units as usize;
    if sm_count == 0 {
        return Err(Error::Internal(
            "small-M wave probe: device profile reports 0 compute units".into(),
        ));
    }
    let max_n = PROBE_RUNGS.iter().map(|r| r.0).max().unwrap_or(0);
    let max_m = PROBE_RUNGS.iter().map(|r| r.1).max().unwrap_or(0);
    let device = &client.device;
    let a = Tensor::<CudaRuntime>::empty(&[max_m, PROBE_K], DType::F32, device)?;
    let w = Tensor::<CudaRuntime>::empty(&[max_n, PROBE_K], DType::F32, device)?;
    let c = Tensor::<CudaRuntime>::empty(&[max_m, max_n], DType::F32, device)?;
    let index = device.index;

    let mut rungs = Vec::with_capacity(PROBE_RUNGS.len());
    for &(n, m) in &PROBE_RUNGS {
        // The first `m` rows of `A` and the first `n` rows of `W` are
        // contiguous prefixes, so the largest buffers serve every rung.
        let small = time_launches(client, PROBE_ITERS, || unsafe {
            launch_matmul_smallm_bt_f32_ungated(
                client.context(),
                client.stream(),
                index,
                a.ptr(),
                w.ptr(),
                c.ptr(),
                m,
                n,
                PROBE_K,
            )
        })?;
        let tiled = time_launches(client, PROBE_ITERS, || {
            let ran = unsafe {
                launch_matmul_kernel_bt(
                    client.context(),
                    client.stream(),
                    index,
                    DType::F32,
                    a.ptr(),
                    w.ptr(),
                    c.ptr(),
                    m,
                    n,
                    PROBE_K,
                )
            }?;
            if ran {
                Ok(())
            } else {
                Err(Error::Internal(format!(
                    "small-M wave probe: tiled bt launcher declined M={m} N={n} K={PROBE_K}"
                )))
            }
        })?;
        rungs.push(Rung {
            waves: smallm_block_count(m, n) as f64 / sm_count as f64,
            wins: small <= tiled,
        });
    }
    Ok(max_waves_from(rungs))
}

/// The wave count of the last winning rung strictly below the first losing
/// rung, floored and clamped to `1..=SMALLM_MAX_WAVES_CEILING`.
///
/// Every rung wins: the ceiling. The first rung loses: 1, not 0. The
/// sub-wave decode shapes (N = 48 at M <= 4 is at most 24 blocks) are where
/// the kernel's lead is largest, and a loss on the smallest rung is more
/// likely a timing outlier than a real crossover below one wave.
fn max_waves_from(mut rungs: Vec<Rung>) -> usize {
    rungs.sort_by(|x, y| x.waves.total_cmp(&y.waves));
    let Some(loss) = rungs.iter().position(|r| !r.wins) else {
        return SMALLM_MAX_WAVES_CEILING;
    };
    let loss_waves = rungs[loss].waves;
    let last_win = rungs[..loss]
        .iter()
        .map(|r| r.waves)
        .filter(|&w| w < loss_waves)
        .fold(None, |best: Option<f64>, w| {
            Some(best.map_or(w, |b| b.max(w)))
        });
    match last_win {
        None => 1,
        Some(w) => (w.floor() as usize).clamp(1, SMALLM_MAX_WAVES_CEILING),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rung(waves: f64, wins: bool) -> Rung {
        Rung { waves, wins }
    }

    #[test]
    fn every_rung_winning_returns_the_ceiling() {
        let rungs = vec![rung(1.7, true), rung(13.7, true), rung(82.3, true)];
        assert_eq!(max_waves_from(rungs), SMALLM_MAX_WAVES_CEILING);
    }

    #[test]
    fn first_rung_losing_returns_one() {
        let rungs = vec![rung(1.7, false), rung(4.6, true)];
        assert_eq!(max_waves_from(rungs), 1);
    }

    #[test]
    fn bound_is_the_floor_of_the_last_win_before_the_first_loss() {
        // Unsorted on purpose: the scan sorts by waves.
        let rungs = vec![
            rung(18.3, false),
            rung(4.6, true),
            rung(13.7, true),
            rung(9.1, true),
            rung(82.3, true),
        ];
        assert_eq!(max_waves_from(rungs), 13);
    }

    #[test]
    fn a_loss_at_the_same_wave_count_excludes_that_rung() {
        let rungs = vec![rung(4.6, true), rung(9.1, true), rung(9.1, false)];
        assert_eq!(max_waves_from(rungs), 4);
    }

    #[test]
    fn a_sub_wave_win_clamps_to_one() {
        let rungs = vec![rung(0.4, true), rung(1.7, false)];
        assert_eq!(max_waves_from(rungs), 1);
    }

    #[test]
    fn rungs_span_the_wave_range_and_fit_the_buffers() {
        let blocks: Vec<usize> = PROBE_RUNGS
            .iter()
            .map(|&(n, m)| smallm_block_count(m, n))
            .collect();
        assert_eq!(blocks.iter().min(), Some(&48));
        assert_eq!(blocks.iter().max(), Some(&512));
        let max_n = PROBE_RUNGS.iter().map(|r| r.0).max().unwrap_or(0);
        let max_m = PROBE_RUNGS.iter().map(|r| r.1).max().unwrap_or(0);
        let bytes = 4 * (max_m * PROBE_K + max_n * PROBE_K + max_m * max_n);
        assert!(bytes < 100 << 20, "probe buffers are {bytes} bytes");
    }
}
