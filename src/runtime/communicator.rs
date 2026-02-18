//! Multi-device collective communication
//!
//! Provides the `Communicator` trait for collective and point-to-point
//! communication across devices. This is a runtime-level concept — not
//! ML-specific. Distributed FFT, parallel linear algebra, Monte Carlo
//! simulations, and ML gradient sync all need these primitives.
//!
//! Per-backend implementations:
//! - `NoOpCommunicator` — single device (world_size=1), always available
//! - `NcclCommunicator` — NCCL for NVIDIA GPUs (feature `cuda`)
//! - `MpiCommunicator` — MPI for multi-node CPU (feature `mpi`)

use crate::dtype::DType;
use crate::error::Result;

/// Reduction operation for collective communication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Element-wise sum across ranks
    Sum,
    /// Element-wise product across ranks
    Prod,
    /// Element-wise minimum across ranks
    Min,
    /// Element-wise maximum across ranks
    Max,
}

/// Multi-device collective communication
///
/// Operates on device pointers (`u64`) + element count + `DType`, matching
/// NCCL's and MPI's native calling conventions. The `u64` pointer is the
/// same abstraction as `Runtime::allocate()` / `Runtime::deallocate()`.
///
/// `DType` provides unambiguous type information so backends can dispatch
/// to the correct reduction unit (e.g., f16 vs bf16 vs i16 are all 2 bytes
/// but require different hardware reduction units).
///
/// # Safety
///
/// All pointer-based methods are `unsafe fn` because passing an invalid `u64`
/// (dangling, wrong device, wrong provenance) causes undefined behavior.
/// Callers MUST ensure:
/// - **NCCL**: pointers are GPU device pointers from the same CUDA context
/// - **MPI**: pointers are valid host pointers
/// - Pointer provenance matches the communicator backend
/// - Buffers remain allocated until `sync()` or `barrier()`
///
/// Higher-level wrappers (boostr's distributed patterns) accept `Tensor<R>`
/// and extract pointers internally, providing a safe public API.
///
/// # Drop contract
///
/// Dropping with pending non-blocking operations attempts best-effort sync
/// with a bounded timeout. On failure the destructor **logs** the error
/// (via `tracing::error!`) and proceeds — it **never panics**.
///
/// # Thread safety
///
/// `Send + Sync` so it can be stored in `Arc`. If multiple threads call
/// `send()`/`recv()` concurrently, submission order is implementation-defined.
/// For deterministic ordering, serialize submissions externally.
pub trait Communicator: Send + Sync {
    /// Number of participants
    fn world_size(&self) -> usize;

    /// This participant's rank (0-indexed)
    fn rank(&self) -> usize;

    /// AllReduce in-place: reduce across all ranks, result on all ranks.
    ///
    /// Completion semantics are implementation-defined. On NCCL the operation
    /// is non-blocking (stream-ordered). **Portable code must call `sync()`
    /// before reading the result buffer.**
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid device pointer with at least `count` elements of `dtype`.
    unsafe fn all_reduce(&self, ptr: u64, count: usize, dtype: DType, op: ReduceOp) -> Result<()>;

    /// Broadcast from root rank to all other ranks.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid device pointer with at least `count` elements of `dtype`.
    unsafe fn broadcast(&self, ptr: u64, count: usize, dtype: DType, root: usize) -> Result<()>;

    /// AllGather: each rank contributes `count` elements, result is
    /// `count * world_size` elements on all ranks.
    ///
    /// # Safety
    ///
    /// - `send_ptr` must point to at least `count` elements
    /// - `recv_ptr` must point to at least `count * world_size` elements
    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()>;

    /// ReduceScatter: reduce + scatter. Each rank gets a different slice
    /// of the reduced result.
    ///
    /// # Safety
    ///
    /// - `send_ptr` must point to at least `count * world_size` elements
    /// - `recv_ptr` must point to at least `count` elements
    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()>;

    /// Point-to-point send to a specific rank (non-blocking).
    ///
    /// The send buffer must NOT be modified or deallocated until `sync()`.
    ///
    /// `tag` is used for message matching on MPI. On NCCL, `tag` is accepted
    /// but ignored (stream-ordered submission determines matching).
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid device pointer with at least `count` elements of `dtype`.
    unsafe fn send(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        dest: usize,
        tag: u32,
    ) -> Result<()>;

    /// Point-to-point receive from a specific rank (non-blocking).
    ///
    /// The recv buffer contains valid data only after `sync()` or `barrier()`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid device pointer with at least `count` elements of `dtype`.
    unsafe fn recv(&self, ptr: u64, count: usize, dtype: DType, src: usize, tag: u32)
    -> Result<()>;

    /// Wait for all pending operations to complete.
    ///
    /// After sync returns, all output/recv buffers contain valid data and
    /// all send/input buffers are safe to reuse.
    fn sync(&self) -> Result<()>;

    /// Barrier: block until all ranks reach this point.
    ///
    /// Implies `sync()` — all pending operations complete before the barrier.
    fn barrier(&self) -> Result<()>;
}

/// No-op communicator for single-device operation (world_size=1).
///
/// - In-place collectives (`all_reduce`, `broadcast`): true no-ops
/// - Separate-buffer collectives (`all_gather`, `reduce_scatter`): memcpy send→recv
/// - Point-to-point (`send`, `recv`): no-ops (nothing to communicate)
/// - `sync`, `barrier`: no-ops
#[derive(Clone, Debug, Default)]
pub struct NoOpCommunicator;

impl Communicator for NoOpCommunicator {
    fn world_size(&self) -> usize {
        1
    }

    fn rank(&self) -> usize {
        0
    }

    unsafe fn all_reduce(
        &self,
        _ptr: u64,
        _count: usize,
        _dtype: DType,
        _op: ReduceOp,
    ) -> Result<()> {
        // Single device: buffer already contains the "reduced" result
        Ok(())
    }

    unsafe fn broadcast(
        &self,
        _ptr: u64,
        _count: usize,
        _dtype: DType,
        _root: usize,
    ) -> Result<()> {
        // Single device: buffer already has root's data (we are root)
        Ok(())
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        // Single device: copy send → recv (output = input for world_size=1)
        if send_ptr != recv_ptr {
            let bytes = count * dtype.size_in_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(send_ptr as *const u8, recv_ptr as *mut u8, bytes);
            }
        }
        Ok(())
    }

    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        _op: ReduceOp,
    ) -> Result<()> {
        // Single device: the "reduced" result is just the input,
        // and the single rank gets the full slice
        if send_ptr != recv_ptr {
            let bytes = count * dtype.size_in_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(send_ptr as *const u8, recv_ptr as *mut u8, bytes);
            }
        }
        Ok(())
    }

    unsafe fn send(
        &self,
        _ptr: u64,
        _count: usize,
        _dtype: DType,
        _dest: usize,
        _tag: u32,
    ) -> Result<()> {
        // Single device: no-op
        Ok(())
    }

    unsafe fn recv(
        &self,
        _ptr: u64,
        _count: usize,
        _dtype: DType,
        _src: usize,
        _tag: u32,
    ) -> Result<()> {
        // Single device: no-op
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_metadata() {
        let comm = NoOpCommunicator;
        assert_eq!(comm.world_size(), 1);
        assert_eq!(comm.rank(), 0);
    }

    #[test]
    fn test_noop_all_reduce() {
        let comm = NoOpCommunicator;
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            comm.all_reduce(data.as_mut_ptr() as u64, 4, DType::F32, ReduceOp::Sum)
                .unwrap();
        }
        // Data unchanged (single device)
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_noop_broadcast() {
        let comm = NoOpCommunicator;
        let mut data = [1.0f32, 2.0];
        unsafe {
            comm.broadcast(data.as_mut_ptr() as u64, 2, DType::F32, 0)
                .unwrap();
        }
        assert_eq!(data, [1.0, 2.0]);
    }

    #[test]
    fn test_noop_all_gather() {
        let comm = NoOpCommunicator;
        let send = [1.0f32, 2.0, 3.0];
        let mut recv = [0.0f32; 3];
        unsafe {
            comm.all_gather(
                send.as_ptr() as u64,
                recv.as_mut_ptr() as u64,
                3,
                DType::F32,
            )
            .unwrap();
        }
        assert_eq!(recv, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_noop_reduce_scatter() {
        let comm = NoOpCommunicator;
        let send = [10.0f32, 20.0];
        let mut recv = [0.0f32; 2];
        unsafe {
            comm.reduce_scatter(
                send.as_ptr() as u64,
                recv.as_mut_ptr() as u64,
                2,
                DType::F32,
                ReduceOp::Sum,
            )
            .unwrap();
        }
        assert_eq!(recv, [10.0, 20.0]);
    }

    #[test]
    fn test_noop_send_recv() {
        let comm = NoOpCommunicator;
        let data = [1.0f32];
        unsafe {
            comm.send(data.as_ptr() as u64, 1, DType::F32, 0, 0)
                .unwrap();
            comm.recv(data.as_ptr() as u64, 1, DType::F32, 0, 0)
                .unwrap();
        }
    }

    #[test]
    fn test_noop_sync_barrier() {
        let comm = NoOpCommunicator;
        comm.sync().unwrap();
        comm.barrier().unwrap();
    }

    #[test]
    fn test_noop_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoOpCommunicator>();
    }

    #[test]
    fn test_noop_all_gather_same_ptr() {
        // When send_ptr == recv_ptr, should be a no-op (no copy needed)
        let comm = NoOpCommunicator;
        let mut data = [1.0f32, 2.0];
        let ptr = data.as_mut_ptr() as u64;
        unsafe {
            comm.all_gather(ptr, ptr, 2, DType::F32).unwrap();
        }
        assert_eq!(data, [1.0, 2.0]);
    }

    #[test]
    fn test_reduce_op_variants() {
        // Ensure all ReduceOp variants exist and are distinct
        let ops = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min, ReduceOp::Max];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j]);
            }
        }
    }
}
