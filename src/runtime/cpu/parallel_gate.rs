//! Work-size gate for rayon dispatch in CPU ops.
//!
//! A rayon fork-join injects a job into the pool, wakes workers and joins them.
//! That costs microseconds, which is tens of thousands of scalar operations.
//! An op smaller than [`PARALLEL_MIN_WORK`] therefore runs on the calling
//! thread, without `install_parallelism` and without a parallel iterator.
//!
//! The gate never changes a result. Every gated site walks the same units
//! (batches, rows, fixed column chunks) with the same per-unit code on both
//! branches. Only the scheduling of those units changes.

use super::client::CpuClient;

/// Scalar operations below which a CPU op skips rayon.
///
/// Work is counted in the unit the site does most: elements read for a scan
/// or reduction, multiply-adds (`m * n * k` per batch) for a matmul, and
/// `n * log2(n)` butterflies per row for an FFT. 32768 is eight times the
/// 4096-element bound `kernels::complex` already uses for one-op elementwise
/// kernels. It errs toward running parallel: an op at the bound does a few
/// microseconds of work, the same order as the fork-join it would pay.
pub(crate) const PARALLEL_MIN_WORK: usize = 32 * 1024;

/// Work estimate for `rows` FFTs of length `n`: `rows * n * ceil(log2(n))`.
///
/// A Bluestein size runs two transforms of length `M >= 2n - 1`, so its real
/// cost is higher. The estimate then undercounts, which only keeps a borderline
/// batch on the calling thread.
#[cfg_attr(not(feature = "rayon"), allow(dead_code))]
pub(crate) fn fft_work(rows: usize, n: usize) -> usize {
    let stages = n.next_power_of_two().trailing_zeros().max(1) as usize;
    rows.saturating_mul(n).saturating_mul(stages)
}

impl CpuClient {
    /// True when `work` scalar operations are enough to pay for a rayon fork-join.
    #[inline]
    #[cfg_attr(not(feature = "rayon"), allow(dead_code))]
    pub(crate) fn parallel_worthwhile(&self, work: usize) -> bool {
        work >= PARALLEL_MIN_WORK
    }

    /// Run `f` inside this client's pool when `parallel`, on the calling thread otherwise.
    ///
    /// For a site whose parallel iterator sits inside `f` behind the same flag:
    /// skipping the install avoids the hop into a dedicated pool when `f` will
    /// not fork.
    #[cfg(feature = "rayon")]
    pub(crate) fn install_parallelism_if<F, T>(&self, parallel: bool, f: F) -> T
    where
        F: FnOnce() -> T + Send,
        T: Send,
    {
        if parallel {
            self.install_parallelism(f)
        } else {
            f()
        }
    }

    /// Run `f` on the calling thread. Without rayon there is no pool to enter.
    #[cfg(not(feature = "rayon"))]
    pub(crate) fn install_parallelism_if<F, T>(&self, _parallel: bool, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        f()
    }

    /// Run `body(i)` for every `i` in `0..count`.
    ///
    /// The indices run on the pool when `count > 1` and `work` passes
    /// [`Self::parallel_worthwhile`], and in ascending order on the calling
    /// thread otherwise. `body` must not depend on which thread runs it.
    pub(crate) fn par_for_each<F>(&self, count: usize, work: usize, body: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        #[cfg(feature = "rayon")]
        if count > 1 && self.parallel_worthwhile(work) {
            use rayon::prelude::*;
            let min_len = self.rayon_min_len();
            self.install_parallelism(|| {
                (0..count)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .for_each(&body);
            });
            return;
        }
        #[cfg(not(feature = "rayon"))]
        let _ = work;

        (0..count).for_each(body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuDevice;
    use std::sync::Mutex;

    #[test]
    fn test_gate_threshold() {
        let client = CpuClient::new(CpuDevice::new());
        assert!(!client.parallel_worthwhile(PARALLEL_MIN_WORK - 1));
        assert!(client.parallel_worthwhile(PARALLEL_MIN_WORK));
    }

    #[test]
    fn test_par_for_each_visits_every_index_once() {
        let client = CpuClient::new(CpuDevice::new());
        for &work in &[1usize, PARALLEL_MIN_WORK] {
            let seen = Mutex::new(vec![0u32; 37]);
            client.par_for_each(37, work, |i| {
                if let Ok(mut s) = seen.lock() {
                    s[i] += 1;
                }
            });
            let seen = seen.into_inner().unwrap_or_default();
            assert_eq!(seen, vec![1u32; 37], "work={work}");
        }
    }

    #[test]
    fn test_small_work_runs_in_order_on_caller() {
        let client = CpuClient::new(CpuDevice::new());
        let caller = std::thread::current().id();
        let order = Mutex::new(Vec::new());
        client.par_for_each(10, 1, |i| {
            assert_eq!(std::thread::current().id(), caller);
            if let Ok(mut o) = order.lock() {
                o.push(i);
            }
        });
        let order = order.into_inner().unwrap_or_default();
        assert_eq!(order, (0..10).collect::<Vec<_>>());
    }
}
