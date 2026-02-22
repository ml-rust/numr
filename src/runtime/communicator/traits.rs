//! Communicator trait and reduction operations.

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

    /// Split this communicator into sub-communicators by color and key.
    ///
    /// All ranks must call `split()` collectively. Ranks with the same `color`
    /// end up in the same sub-communicator, ordered by `key`.
    ///
    /// Returns `None` for backends that don't support splitting (e.g., NCCL
    /// without `ncclCommSplit`, or the no-op communicator).
    fn split(&self, _color: u32, _key: u32) -> Result<Option<Box<dyn Communicator>>> {
        Ok(None)
    }
}
