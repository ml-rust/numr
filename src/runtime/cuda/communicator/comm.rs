//! The NCCL communicator handle: creation, abort, error polling, stream waits, and drop.

use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudarc::driver::CudaStream;
use cudarc::driver::sys::{CUresult, cuStreamQuery};
use cudarc::nccl::result::{NcclError, NcclStatus};
use cudarc::nccl::{result as nccl_result, sys as nccl_sys};

use super::convert::{context_err, nccl_err, to_nccl_int};
use super::id::NcclId;
use crate::error::{Error, Result};
use crate::runtime::cuda::context::bind_context;

/// The raw NCCL handle. It is null after [`NcclCommunicator::abort`].
struct RawComm(nccl_sys::ncclComm_t);

// SAFETY: an `ncclComm_t` points to a heap object that any host thread can use.
// NCCL forbids concurrent calls on one communicator. The `Mutex` that holds
// `RawComm` serializes every call that reads the handle.
unsafe impl Send for RawComm {}

/// One rank of an NCCL communicator.
///
/// numr creates and owns the `ncclComm_t`. Every NCCL call goes through
/// `cudarc::nccl::result`. A mutex serializes calls, so one communicator is
/// safe to share across threads.
///
/// Drop aborts the handle when [`abort`](Self::abort) has not run.
/// `ncclCommDestroy` waits for queued operations, so it hangs when a peer is gone.
/// Call `sync` before drop to finish queued work.
pub struct NcclCommunicator {
    comm: Mutex<RawComm>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) rank: usize,
    pub(super) world_size: usize,
}

impl NcclCommunicator {
    /// Create one communicator per stream in this process, one GPU per stream.
    ///
    /// Rank `i` uses `streams[i]`. Every stream must be on a different device.
    pub fn from_streams(streams: Vec<Arc<CudaStream>>) -> Result<Vec<Self>> {
        if streams.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "streams",
                reason: "NCCL needs at least one stream; pass one stream per GPU".into(),
            });
        }
        let world_size = streams.len();
        let n_ranks = to_nccl_int("streams", world_size)?;
        let ordinals = streams
            .iter()
            .map(|s| to_nccl_int("device ordinal", s.context().ordinal()))
            .collect::<Result<Vec<i32>>>()?;
        let mut handles: Vec<nccl_sys::ncclComm_t> = vec![ptr::null_mut(); world_size];
        // SAFETY: `handles` and `ordinals` each hold `n_ranks` elements, as
        // ncclCommInitAll requires. NCCL writes one handle per slot on success.
        unsafe { nccl_result::comm_init_all(handles.as_mut_ptr(), n_ranks, ordinals.as_ptr()) }
            .map_err(|e| {
                Error::Backend(format!(
                    "NCCL init of {world_size} ranks on devices {ordinals:?}: {}",
                    nccl_err("ncclCommInitAll", e)
                ))
            })?;
        Ok(handles
            .into_iter()
            .zip(streams)
            .enumerate()
            .map(|(rank, (handle, stream))| Self {
                comm: Mutex::new(RawComm(handle)),
                stream,
                rank,
                world_size,
            })
            .collect())
    }

    /// Join rank `rank` of a `world_size`-rank communicator identified by `id`.
    ///
    /// Each rank calls this once, from its own process or thread, with the same `id`.
    /// The call blocks until all `world_size` ranks join. A peer that never joins
    /// leaves it blocked, and no handle exists yet to abort.
    pub fn from_rank(
        stream: Arc<CudaStream>,
        rank: usize,
        world_size: usize,
        id: NcclId,
    ) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(Error::InvalidArgument {
                arg: "rank",
                reason: format!(
                    "rank {rank} is outside world size {world_size}; pass 0 <= rank < world_size"
                ),
            });
        }
        let n_ranks = to_nccl_int("world_size", world_size)?;
        let nccl_rank = to_nccl_int("rank", rank)?;
        // ncclCommInitRank attaches to the current device, so bind the stream's context.
        bind_context(stream.context())
            .map_err(|e| Error::Backend(format!("NCCL rank {rank}: {}", context_err(e))))?;
        let mut handle: nccl_sys::ncclComm_t = ptr::null_mut();
        // SAFETY: `handle` is a valid out slot for one handle. `id.raw()` is a
        // full unique id. NCCL writes `handle` only on success.
        unsafe { nccl_result::comm_init_rank(&mut handle, n_ranks, id.raw(), nccl_rank) }.map_err(
            |e| {
                Error::Backend(format!(
                    "NCCL rank {rank} of {world_size}: {}",
                    nccl_err("ncclCommInitRank", e)
                ))
            },
        )?;
        Ok(Self {
            comm: Mutex::new(RawComm(handle)),
            stream,
            rank,
            world_size,
        })
    }

    /// The CUDA stream every NCCL call of this rank runs on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Abort the communicator. A second call returns `Ok` and does nothing.
    ///
    /// Abort stops this rank's queued NCCL kernels, so a `sync` blocked on them returns.
    /// It can run from another thread while `sync` waits. It waits for an NCCL call
    /// already running on another thread to return. Peers learn of the abort only
    /// when their transport reports the closed connection. Each rank needs its own
    /// watchdog that calls `abort`. Every later call on this communicator returns an error.
    pub fn abort(&self) -> Result<()> {
        let handle = std::mem::replace(&mut self.lock().0, ptr::null_mut());
        if handle.is_null() {
            return Ok(());
        }
        // SAFETY: `handle` came from a successful NCCL init. The slot now holds
        // null, so no other call can read `handle`, and Drop cannot abort it twice.
        unsafe { nccl_result::comm_abort(handle) }
            .map(|_| ())
            .map_err(|e| self.rank_err(nccl_err("ncclCommAbort", e)))
    }

    /// Return the asynchronous error NCCL recorded for this communicator, if any.
    ///
    /// A peer error surfaces here, not as a failed call. After an error the
    /// communicator is unusable: call [`abort`](Self::abort) and create a new one.
    pub fn async_error(&self) -> Result<()> {
        let mut state = nccl_sys::ncclResult_t::ncclSuccess;
        // SAFETY: `comm` is a live handle while the lock is held. `state` is a
        // valid out slot.
        self.with_comm("ncclCommGetAsyncError", |comm, _| unsafe {
            nccl_sys::ncclCommGetAsyncError(comm, &mut state).result()
        })?;
        state
            .result()
            .map(|_| ())
            .map_err(|e| self.rank_err(nccl_err("an asynchronous NCCL operation", e)))
    }

    /// Run one NCCL call with the live handle and the bound stream.
    ///
    /// Holds the lock for the call. Returns an error when the communicator is aborted.
    pub(super) fn with_comm<F>(&self, call: &'static str, f: F) -> Result<()>
    where
        F: FnOnce(
            nccl_sys::ncclComm_t,
            nccl_sys::cudaStream_t,
        ) -> std::result::Result<NcclStatus, NcclError>,
    {
        let guard = self.lock();
        if guard.0.is_null() {
            return Err(self.rank_err(format!(
                "{call} on an aborted communicator; create a new communicator"
            )));
        }
        let stream = self.bound_stream()?;
        f(guard.0, stream)
            .map(|_| ())
            .map_err(|e| self.rank_err(nccl_err(call, e)))
    }

    /// Wait until the communicator stream drains, polling NCCL for asynchronous errors.
    ///
    /// An error NCCL reports aborts the communicator and returns the error. A completed
    /// stream on an aborted communicator also returns an error, because the
    /// abort discarded queued results.
    pub(super) fn wait_stream(&self) -> Result<()> {
        self.bind()?;
        loop {
            // SAFETY: the handle belongs to `self.stream`, which outlives this call.
            let status = unsafe { cuStreamQuery(self.stream.cu_stream()) };
            match status {
                CUresult::CUDA_SUCCESS => return self.async_error(),
                CUresult::CUDA_ERROR_NOT_READY => {}
                other => {
                    return Err(self.rank_err(format!(
                        "cuStreamQuery on the communicator stream failed with {other:?}"
                    )));
                }
            }
            if let Err(e) = self.async_error() {
                return Err(match self.abort() {
                    Ok(()) => e,
                    Err(abort_err) => {
                        // Join the bare messages so "Backend error:" prints once.
                        let bare = |err: Error| match err {
                            Error::Backend(msg) => msg,
                            other => other.to_string(),
                        };
                        Error::Backend(format!("{}; then {}", bare(e), bare(abort_err)))
                    }
                });
            }
            std::thread::yield_now();
        }
    }

    /// The raw CUDA stream handle, with the communicator's context current on this thread.
    pub(super) fn bound_stream(&self) -> Result<nccl_sys::cudaStream_t> {
        self.bind()?;
        Ok(self.stream.cu_stream() as nccl_sys::cudaStream_t)
    }

    /// Make the communicator's CUDA context current on the calling thread.
    pub(super) fn bind(&self) -> Result<()> {
        bind_context(self.stream.context()).map_err(|e| self.rank_err(context_err(e)))
    }

    /// A backend error prefixed with this rank and the world size.
    fn rank_err(&self, msg: impl std::fmt::Display) -> Error {
        Error::Backend(format!(
            "NCCL rank {} of {}: {msg}",
            self.rank, self.world_size
        ))
    }

    fn lock(&self) -> MutexGuard<'_, RawComm> {
        // The guarded section only makes FFI calls, so a poisoned lock still holds a valid handle.
        self.comm.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        let raw = self.comm.get_mut().unwrap_or_else(PoisonError::into_inner);
        let handle = std::mem::replace(&mut raw.0, ptr::null_mut());
        if handle.is_null() {
            return;
        }
        // SAFETY: `handle` came from a successful NCCL init and was never aborted.
        // `&mut self` rules out any other call. `self.stream` keeps the device context alive.
        // Drop cannot return the error, and the handle is unusable either way.
        let _ = unsafe { nccl_result::comm_abort(handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::communicator::{Communicator, ReduceOp};
    use crate::runtime::cuda::communicator::id::NCCL_ID_BYTES;
    use cudarc::driver::{CudaContext, DevicePtr};

    fn device_count() -> usize {
        CudaContext::device_count().map_or(0, |n| n.max(0) as usize)
    }

    /// A one-rank communicator on device 0, or `None` with a skip message.
    fn single_rank(test: &str) -> Option<NcclCommunicator> {
        let n = device_count();
        if n < 1 {
            eprintln!("skipping {test}: needs 1 CUDA device, found {n}");
            return None;
        }
        let stream = CudaContext::new(0).unwrap().default_stream();
        Some(NcclCommunicator::from_rank(stream, 0, 1, NcclId::new().unwrap()).unwrap())
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NcclCommunicator>();
    }

    #[test]
    fn test_from_streams_rejects_empty() {
        assert!(NcclCommunicator::from_streams(Vec::new()).is_err());
    }

    #[test]
    fn test_from_rank_rejects_rank_outside_world() {
        let n = device_count();
        if n < 1 {
            eprintln!(
                "skipping test_from_rank_rejects_rank_outside_world: needs 1 CUDA device, found {n}"
            );
            return;
        }
        let stream = CudaContext::new(0).unwrap().default_stream();
        let id = NcclId::from_bytes([0u8; NCCL_ID_BYTES]);
        assert!(NcclCommunicator::from_rank(stream.clone(), 1, 1, id).is_err());
        assert!(NcclCommunicator::from_rank(stream, 0, 0, id).is_err());
    }

    #[test]
    fn test_world_size_one_metadata_and_async_error() {
        let Some(comm) = single_rank("test_world_size_one_metadata_and_async_error") else {
            return;
        };
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        comm.async_error().unwrap();
    }

    #[test]
    fn test_world_size_one_all_reduce_and_barrier() {
        let Some(comm) = single_rank("test_world_size_one_all_reduce_and_barrier") else {
            return;
        };
        let stream = comm.stream().clone();
        let slice = stream.clone_htod(&[1.5f32, -2.0, 3.0]).unwrap();
        let ptr = {
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr as u64
        };
        unsafe { comm.all_reduce(ptr, 3, DType::F32, ReduceOp::Sum).unwrap() };
        comm.sync().unwrap();
        assert_eq!(stream.clone_dtoh(&slice).unwrap(), vec![1.5f32, -2.0, 3.0]);
        comm.barrier().unwrap();
    }

    #[test]
    fn test_abort_is_idempotent() {
        let Some(comm) = single_rank("test_abort_is_idempotent") else {
            return;
        };
        comm.abort().unwrap();
        comm.abort().unwrap();
        assert!(comm.async_error().is_err());
        assert!(comm.sync().is_err());
        assert!(comm.barrier().is_err());
        // An aborted communicator returns before NCCL reads the pointer.
        assert!(unsafe { comm.all_reduce(0, 0, DType::F32, ReduceOp::Sum) }.is_err());
        // Drop runs here and must not abort a second time.
    }

    #[test]
    fn test_from_streams_world_size_one_abort() {
        let n = device_count();
        if n < 1 {
            eprintln!(
                "skipping test_from_streams_world_size_one_abort: needs 1 CUDA device, found {n}"
            );
            return;
        }
        let stream = CudaContext::new(0).unwrap().default_stream();
        let comms = NcclCommunicator::from_streams(vec![stream]).unwrap();
        assert_eq!(comms.len(), 1);
        assert_eq!(comms[0].world_size(), 1);
        comms[0].sync().unwrap();
        comms[0].abort().unwrap();
        comms[0].abort().unwrap();
    }
}
