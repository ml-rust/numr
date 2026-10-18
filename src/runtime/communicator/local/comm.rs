//! `LocalCommunicator`: one rank of an in-process group, plus the panic guard.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use super::plan::{Kind, Post};
use super::shared::Shared;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::communicator::{Communicator, ReduceOp};

/// One rank of an in-process group: each thread holds one rank and calls collectives on it.
///
/// Use it to test data-parallel code on CPU threads without GPUs or a network.
///
/// - Pointers are host addresses only, such as CPU-runtime tensor storage or a `Vec`.
///   A device pointer is undefined behavior, as the trait contract states.
/// - Every collective blocks until all ranks join, so `sync` is a no-op.
/// - Reductions fold in rank order `b0 op b1 op ... op b(W-1)`. Results are bit-identical
///   on every rank and on every run.
/// - Reductions accept F64, F32, I64, I32, I16, I8, U64, U32, U16 and U8.
/// - F16 and BF16 reductions need the `f16` feature. They widen to f32, fold in rank order,
///   and round once.
/// - Integer reductions wrap on overflow. Float `Max` and `Min` let a number win over a NaN.
/// - `broadcast` and `all_gather` move bytes, so they accept every dtype.
/// - Ranks that disagree on the call, count, dtype, op or root all receive the same error.
///   The group stays usable after that error.
/// - `send` and `recv` return an error.
/// - `split` returns `None`.
///
/// # Aborts
///
/// Every wait gives up after the group timeout and aborts the group.
/// [`abort`](Self::abort), dropping any rank, and a panic under
/// [`abort_on_panic`](Self::abort_on_panic) also abort it.
/// After an abort, every pending and later call on every rank returns an error.
/// A call whose round already finished still returns `Ok`.
///
/// # Example
///
/// ```
/// use numr::dtype::DType;
/// use numr::runtime::{Communicator, LocalCommunicator, ReduceOp};
///
/// let comms = LocalCommunicator::group(2);
/// std::thread::scope(|s| {
///     for comm in &comms {
///         s.spawn(move || {
///             let _guard = comm.abort_on_panic();
///             let mut grad = vec![comm.rank() as f32 + 1.0; 4];
///             // SAFETY: `grad` is a live host buffer of 4 f32 values.
///             let done = unsafe {
///                 comm.all_reduce(grad.as_mut_ptr() as u64, 4, DType::F32, ReduceOp::Sum)
///             };
///             assert!(done.is_ok());
///             assert_eq!(grad, vec![3.0; 4]);
///         });
///     }
/// });
/// ```
pub struct LocalCommunicator {
    rank: usize,
    shared: Arc<Shared>,
}

impl LocalCommunicator {
    /// Wait limit of [`group`](Self::group).
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    /// Create a group of `world` ranks with [`DEFAULT_TIMEOUT`](Self::DEFAULT_TIMEOUT).
    ///
    /// Rank `i` is at index `i`. `world == 0` returns an empty vector.
    pub fn group(world: usize) -> Vec<Self> {
        Self::with_timeout(world, Self::DEFAULT_TIMEOUT)
    }

    /// Create a group of `world` ranks whose waits give up after `timeout`.
    ///
    /// Rank `i` is at index `i`. `world == 0` returns an empty vector.
    pub fn with_timeout(world: usize, timeout: Duration) -> Vec<Self> {
        if world == 0 {
            return Vec::new();
        }
        let shared = Arc::new(Shared::new(world, timeout));
        (0..world)
            .map(|rank| Self {
                rank,
                shared: Arc::clone(&shared),
            })
            .collect()
    }

    /// Abort the group. Every pending and later call on every rank returns an error.
    pub fn abort(&self) {
        self.shared.abort();
    }

    /// True once the group is aborted.
    pub fn is_aborted(&self) -> bool {
        self.shared.is_aborted()
    }

    /// Return a guard that aborts the group if this thread panics while the guard lives.
    ///
    /// Create it first thing in each rank thread. Peers blocked on the panicking rank then
    /// return an error at once instead of waiting for the timeout.
    pub fn abort_on_panic(&self) -> AbortOnPanic<'_> {
        AbortOnPanic { comm: self }
    }

    /// # Safety
    ///
    /// `send` and `recv` must meet the `Communicator` contract for `kind`.
    unsafe fn post(
        &self,
        kind: Kind,
        send: u64,
        recv: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        let post = Post {
            kind,
            send,
            recv,
            count,
            dtype,
        };
        // SAFETY: forwarded from this function's contract.
        unsafe { self.shared.run(self.rank, post) }
    }
}

impl Drop for LocalCommunicator {
    /// Dropping a rank aborts the group, since its peers can never finish another round.
    fn drop(&mut self) {
        self.shared.abort();
    }
}

impl fmt::Debug for LocalCommunicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalCommunicator")
            .field("rank", &self.rank)
            .field("world", &self.shared.world())
            .field("aborted", &self.is_aborted())
            .finish()
    }
}

/// Aborts its group when dropped during a panic. Create it with
/// [`LocalCommunicator::abort_on_panic`].
#[derive(Debug)]
#[must_use = "the guard aborts the group only while it lives"]
pub struct AbortOnPanic<'a> {
    comm: &'a LocalCommunicator,
}

impl Drop for AbortOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.comm.abort();
        }
    }
}

fn point_to_point() -> Error {
    Error::Backend("LocalCommunicator: point-to-point is not supported".to_string())
}

impl Communicator for LocalCommunicator {
    fn world_size(&self) -> usize {
        self.shared.world()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    unsafe fn all_reduce(&self, ptr: u64, count: usize, dtype: DType, op: ReduceOp) -> Result<()> {
        // SAFETY: the caller meets the trait contract for `ptr`.
        unsafe { self.post(Kind::AllReduce(op), ptr, ptr, count, dtype) }
    }

    unsafe fn broadcast(&self, ptr: u64, count: usize, dtype: DType, root: usize) -> Result<()> {
        // SAFETY: the caller meets the trait contract for `ptr`.
        unsafe { self.post(Kind::Broadcast { root }, ptr, ptr, count, dtype) }
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        // SAFETY: the caller meets the trait contract for both pointers.
        unsafe { self.post(Kind::AllGather, send_ptr, recv_ptr, count, dtype) }
    }

    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()> {
        // SAFETY: the caller meets the trait contract for both pointers.
        unsafe { self.post(Kind::ReduceScatter(op), send_ptr, recv_ptr, count, dtype) }
    }

    unsafe fn send(&self, _: u64, _: usize, _: DType, _: usize, _: u32) -> Result<()> {
        Err(point_to_point())
    }

    unsafe fn recv(&self, _: u64, _: usize, _: DType, _: usize, _: u32) -> Result<()> {
        Err(point_to_point())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        // SAFETY: a barrier moves no bytes, so the null pointers are never read or written.
        unsafe { self.post(Kind::Barrier, 0, 0, 0, DType::U8) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Pod;
    use std::fmt::Debug;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    const OPS: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min];

    /// Run `f` on every rank, one thread per rank, and collect the results in rank order.
    fn on_ranks<T: Send>(
        comms: &[LocalCommunicator],
        f: impl Fn(&LocalCommunicator) -> T + Sync,
    ) -> Vec<T> {
        std::thread::scope(|s| {
            let f = &f;
            let handles: Vec<_> = comms.iter().map(|c| s.spawn(move || f(c))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn all_reduce<T: Pod>(
        comm: &LocalCommunicator,
        mut data: Vec<T>,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<Vec<T>> {
        unsafe { comm.all_reduce(data.as_mut_ptr() as u64, data.len(), dtype, op)? };
        Ok(data)
    }

    /// Check every rank against a fold of `data` in rank order, byte for byte.
    fn check_all_reduce<T: Pod + Send + Debug>(
        world: usize,
        dtype: DType,
        op: ReduceOp,
        data: impl Fn(usize) -> Vec<T> + Sync,
        combine: impl Fn(T, T) -> T,
    ) {
        let mut expected = data(0);
        for rank in 1..world {
            for (e, v) in expected.iter_mut().zip(data(rank)) {
                *e = combine(*e, v);
            }
        }
        let comms = LocalCommunicator::group(world);
        let results = on_ranks(&comms, |c| {
            all_reduce(c, data(c.rank()), dtype, op).unwrap()
        });
        for (rank, got) in results.iter().enumerate() {
            assert_eq!(
                bytemuck::cast_slice::<T, u8>(got),
                bytemuck::cast_slice::<T, u8>(&expected),
                "rank {rank}, {op:?}, {dtype:?}, world {world}"
            );
        }
    }

    fn float_combine<T: num_traits::Float>(op: ReduceOp) -> impl Fn(T, T) -> T {
        move |a, b| match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Prod => a * b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        }
    }

    fn i32_combine(op: ReduceOp) -> impl Fn(i32, i32) -> i32 {
        move |a, b| match op {
            ReduceOp::Sum => a.wrapping_add(b),
            ReduceOp::Prod => a.wrapping_mul(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        }
    }

    fn f32_data(rank: usize) -> Vec<f32> {
        let rows = [
            [1e8, 1.5, -3.0, 7.25, 0.1],
            [1.0, -2.0, 0.5, 3.0, 0.2],
            [-1e8, 4.0, 2.0, -1.0, 0.3],
        ];
        rows[rank].to_vec()
    }

    fn f64_data(rank: usize) -> Vec<f64> {
        let rows = [
            [1e17, 1.5, -3.0, 0.1],
            [1.0, -2.0, 0.5, 0.2],
            [-1e17, 4.0, 2.0, 0.3],
        ];
        rows[rank].to_vec()
    }

    fn i32_data(rank: usize) -> Vec<i32> {
        let rows = [[7, -3, i32::MAX, 0], [-2, 5, 1, 9], [4, -6, 1, -9]];
        rows[rank].to_vec()
    }

    #[test]
    fn all_reduce_matches_rank_order_fold() {
        for world in [2, 3] {
            for op in OPS {
                check_all_reduce(world, DType::F32, op, f32_data, float_combine(op));
                check_all_reduce(world, DType::F64, op, f64_data, float_combine(op));
                check_all_reduce(world, DType::I32, op, i32_data, i32_combine(op));
            }
        }
    }

    #[test]
    fn f32_sum_order_is_fixed_across_runs() {
        for _ in 0..20 {
            let comms = LocalCommunicator::group(3);
            let results = on_ranks(&comms, |c| {
                all_reduce(c, f32_data(c.rank()), DType::F32, ReduceOp::Sum).unwrap()
            });
            for got in &results {
                // (1e8 + 1) rounds to 1e8, so rank order gives 0. Any other order gives 1.
                assert_eq!(got[0].to_bits(), 0.0f32.to_bits());
                assert_eq!(
                    bytemuck::cast_slice::<f32, u8>(got),
                    bytemuck::cast_slice::<f32, u8>(&results[0])
                );
            }
        }
    }

    #[test]
    fn broadcast_copies_root_buffer() {
        for root in [0, 1] {
            let comms = LocalCommunicator::group(3);
            let results = on_ranks(&comms, |c| {
                let mut data: Vec<f32> = (0..4).map(|i| (c.rank() * 10 + i) as f32).collect();
                unsafe { c.broadcast(data.as_mut_ptr() as u64, 4, DType::F32, root) }.unwrap();
                data
            });
            let expected: Vec<f32> = (0..4).map(|i| (root * 10 + i) as f32).collect();
            assert!(results.iter().all(|got| *got == expected), "root {root}");
        }
    }

    #[test]
    fn broadcast_accepts_dtypes_without_reduction() {
        let comms = LocalCommunicator::group(2);
        let results = on_ranks(&comms, |c| {
            let mut data = vec![c.rank() as u8; 3];
            unsafe { c.broadcast(data.as_mut_ptr() as u64, 3, DType::Bool, 1) }.unwrap();
            data
        });
        assert_eq!(results, vec![vec![1u8; 3]; 2]);
    }

    #[test]
    fn all_gather_concatenates_in_rank_order() {
        let comms = LocalCommunicator::group(3);
        let results = on_ranks(&comms, |c| {
            let send = [(c.rank() * 10) as i32, (c.rank() * 10 + 1) as i32];
            let mut recv = vec![0i32; 6];
            unsafe {
                c.all_gather(
                    send.as_ptr() as u64,
                    recv.as_mut_ptr() as u64,
                    2,
                    DType::I32,
                )
            }
            .unwrap();
            recv
        });
        assert_eq!(results, vec![vec![0, 1, 10, 11, 20, 21]; 3]);
    }

    #[test]
    fn reduce_scatter_gives_each_rank_its_slice() {
        let comms = LocalCommunicator::group(3);
        let send = |rank: usize| -> Vec<i64> { (0..6).map(|j| (rank * 100 + j) as i64).collect() };
        let results = on_ranks(&comms, |c| {
            let send = send(c.rank());
            let mut recv = vec![0i64; 2];
            unsafe {
                c.reduce_scatter(
                    send.as_ptr() as u64,
                    recv.as_mut_ptr() as u64,
                    2,
                    DType::I64,
                    ReduceOp::Sum,
                )
            }
            .unwrap();
            recv
        });
        for (rank, got) in results.iter().enumerate() {
            let expected: Vec<i64> = (0..2)
                .map(|j| (0..3).map(|r| send(r)[rank * 2 + j]).sum())
                .collect();
            assert_eq!(*got, expected, "rank {rank}");
        }
    }

    #[test]
    fn barrier_holds_every_rank_until_all_arrive() {
        let comms = LocalCommunicator::group(3);
        let arrived = AtomicUsize::new(0);
        on_ranks(&comms, |c| {
            for round in 0..10 {
                arrived.fetch_add(1, Ordering::SeqCst);
                c.barrier().unwrap();
                assert!(arrived.load(Ordering::SeqCst) >= 3 * (round + 1));
                c.barrier().unwrap();
            }
        });
    }

    #[test]
    fn consecutive_rounds_never_mix() {
        let comms = LocalCommunicator::group(3);
        on_ranks(&comms, |c| {
            let r = c.rank();
            for k in 0..200usize {
                let n = k % 5 + 1;
                if k % 2 == 0 {
                    let data: Vec<i64> = (0..n).map(|i| ((r + 1) * k + i) as i64).collect();
                    let got = all_reduce(c, data, DType::I64, ReduceOp::Sum).unwrap();
                    let expected: Vec<i64> = (0..n).map(|i| (6 * k + 3 * i) as i64).collect();
                    assert_eq!(got, expected, "rank {r}, round {k}");
                } else {
                    let data = vec![(r * 1000 + k) as f64; n];
                    let got = all_reduce(c, data, DType::F64, ReduceOp::Max).unwrap();
                    assert_eq!(got, vec![(2000 + k) as f64; n], "rank {r}, round {k}");
                }
            }
        });
    }

    #[test]
    fn count_mismatch_errors_on_every_rank() {
        let comms = LocalCommunicator::group(2);
        let results = on_ranks(&comms, |c| {
            let n = 4 - c.rank();
            let first = all_reduce(c, vec![1.0f32; n], DType::F32, ReduceOp::Sum);
            let next = all_reduce(c, vec![1.0f32; 2], DType::F32, ReduceOp::Sum);
            (first.map_err(|e| e.to_string()), next.unwrap())
        });
        for (first, next) in results {
            let msg = first.unwrap_err();
            assert!(msg.contains("count 4") && msg.contains("count 3"), "{msg}");
            // A mismatch fails one round only. The group stays usable.
            assert_eq!(next, vec![2.0f32; 2]);
        }
    }

    #[test]
    fn dtype_and_op_mismatch_error() {
        let comms = LocalCommunicator::group(2);
        let results = on_ranks(&comms, |c| {
            let dtype = [DType::F32, DType::I32][c.rank()];
            let dtypes =
                all_reduce(c, vec![0u32; 4], dtype, ReduceOp::Sum).map_err(|e| e.to_string());
            let op = [ReduceOp::Sum, ReduceOp::Max][c.rank()];
            let ops = all_reduce(c, vec![0u32; 4], DType::U32, op).map_err(|e| e.to_string());
            (dtypes, ops)
        });
        for (dtypes, ops) in results {
            assert!(dtypes.unwrap_err().contains("I32"));
            assert!(ops.unwrap_err().contains("Max"));
        }
    }

    #[test]
    fn unsupported_reduction_dtype_errors_on_every_rank() {
        let comms = LocalCommunicator::group(2);
        let results = on_ranks(&comms, |c| {
            all_reduce(c, vec![1u8; 2], DType::Bool, ReduceOp::Sum).map_err(|e| e.to_string())
        });
        for result in results {
            assert!(result.unwrap_err().contains("Bool"));
        }
    }

    #[test]
    fn abort_unblocks_a_waiting_peer() {
        let comms = LocalCommunicator::group(2);
        let start = Instant::now();
        let result = std::thread::scope(|s| {
            let waiter =
                s.spawn(|| all_reduce(&comms[0], vec![1.0f32; 4], DType::F32, ReduceOp::Sum));
            std::thread::sleep(Duration::from_millis(100));
            comms[1].abort();
            waiter.join().unwrap()
        });
        assert!(result.unwrap_err().to_string().contains("aborted"));
        assert!(start.elapsed() < LocalCommunicator::DEFAULT_TIMEOUT / 2);
        assert!(comms[1].barrier().is_err());
    }

    #[test]
    fn timeout_errors_and_aborts_the_group() {
        let comms = LocalCommunicator::with_timeout(2, Duration::from_millis(50));
        let msg = comms[0].barrier().unwrap_err().to_string();
        assert!(msg.contains("timed out"), "{msg}");
        assert!(comms[1].is_aborted());
        assert!(comms[1].barrier().is_err());
    }

    #[test]
    fn panic_guard_aborts_peers() {
        let comms = LocalCommunicator::group(2);
        let (waiter, panicker) = std::thread::scope(|s| {
            let waiter = s.spawn(|| comms[0].barrier());
            let panicker = s.spawn(|| {
                let _guard = comms[1].abort_on_panic();
                panic!("rank 1 fails before the barrier");
            });
            (waiter.join().unwrap(), panicker.join())
        });
        assert!(panicker.is_err());
        assert!(waiter.unwrap_err().to_string().contains("aborted"));
    }

    #[test]
    fn dropping_a_rank_aborts_the_group() {
        let mut comms = LocalCommunicator::group(2);
        drop(comms.pop());
        assert!(comms[0].barrier().is_err());
    }

    #[test]
    fn world_one_behaves_like_noop() {
        let comms = LocalCommunicator::group(1);
        let c = &comms[0];
        assert_eq!((c.world_size(), c.rank()), (1, 0));
        let got = all_reduce(c, vec![1.0f32, 2.0], DType::F32, ReduceOp::Sum).unwrap();
        assert_eq!(got, vec![1.0, 2.0]);
        let mut data = [3.0f32, 4.0];
        unsafe { c.broadcast(data.as_mut_ptr() as u64, 2, DType::F32, 0) }.unwrap();
        assert_eq!(data, [3.0, 4.0]);
        let send = [5.0f32, 6.0];
        let mut recv = [0.0f32; 2];
        unsafe {
            c.all_gather(
                send.as_ptr() as u64,
                recv.as_mut_ptr() as u64,
                2,
                DType::F32,
            )
        }
        .unwrap();
        assert_eq!(recv, send);
        let mut recv = [0.0f32; 2];
        unsafe {
            c.reduce_scatter(
                send.as_ptr() as u64,
                recv.as_mut_ptr() as u64,
                2,
                DType::F32,
                ReduceOp::Sum,
            )
        }
        .unwrap();
        assert_eq!(recv, send);
        c.sync().unwrap();
        c.barrier().unwrap();
        assert!(c.split(0, 0).unwrap().is_none());
    }

    #[test]
    fn point_to_point_errors() {
        let comms = LocalCommunicator::group(2);
        let data = [1.0f32];
        let ptr = data.as_ptr() as u64;
        let send = unsafe { comms[0].send(ptr, 1, DType::F32, 1, 0) };
        let recv = unsafe { comms[1].recv(ptr, 1, DType::F32, 0, 0) };
        assert!(
            send.unwrap_err()
                .to_string()
                .contains("point-to-point is not supported")
        );
        assert!(
            recv.unwrap_err()
                .to_string()
                .contains("point-to-point is not supported")
        );
    }

    #[test]
    fn group_shape_and_thread_safety() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LocalCommunicator>();
        assert!(LocalCommunicator::group(0).is_empty());
        let comms = LocalCommunicator::group(4);
        for (i, c) in comms.iter().enumerate() {
            assert_eq!((c.rank(), c.world_size()), (i, 4));
        }
    }
}
