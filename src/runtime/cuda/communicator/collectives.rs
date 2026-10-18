//! `Communicator` for NCCL: collectives and point-to-point calls.
//!
//! Every call goes through raw `nccl::result` FFI, because numr picks the
//! `DType` at run time and cudarc's safe API needs it at compile time.

use std::ffi::c_void;

use cudarc::nccl::{result as nccl_result, sys as nccl_sys};

use super::comm::NcclCommunicator;
use super::convert::{dtype_to_nccl, reduce_op_to_nccl, to_nccl_int};
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::communicator::{Communicator, ReduceOp, StreamSyncOps};

impl Communicator for NcclCommunicator {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    unsafe fn all_reduce(&self, ptr: u64, count: usize, dtype: DType, op: ReduceOp) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        let nccl_op = reduce_op_to_nccl(op);
        // In place: sendbuff == recvbuff.
        // SAFETY: the caller guarantees `ptr` holds `count` elements of `dtype`
        // on this rank's device. `comm` is live while `with_comm` holds the lock.
        self.with_comm("ncclAllReduce", |comm, stream| unsafe {
            nccl_result::all_reduce(
                ptr as *const c_void,
                ptr as *mut c_void,
                count,
                nccl_dtype,
                nccl_op,
                comm,
                stream,
            )
        })
    }

    unsafe fn broadcast(&self, ptr: u64, count: usize, dtype: DType, root: usize) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        let root = to_nccl_int("root", root)?;
        // In place: sendbuff == recvbuff.
        // SAFETY: the caller guarantees `ptr` holds `count` elements of `dtype`
        // on this rank's device. `comm` is live while `with_comm` holds the lock.
        self.with_comm("ncclBroadcast", |comm, stream| unsafe {
            nccl_result::broadcast(
                ptr as *const c_void,
                ptr as *mut c_void,
                count,
                nccl_dtype,
                root,
                comm,
                stream,
            )
        })
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        // SAFETY: the caller guarantees `send_ptr` holds `count` elements and
        // `recv_ptr` holds `count * world_size` elements of `dtype`. `comm` is live
        // while `with_comm` holds the lock.
        self.with_comm("ncclAllGather", |comm, stream| unsafe {
            nccl_result::all_gather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                count,
                nccl_dtype,
                comm,
                stream,
            )
        })
    }

    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        let nccl_op = reduce_op_to_nccl(op);
        // SAFETY: the caller guarantees `send_ptr` holds `count * world_size`
        // elements and `recv_ptr` holds `count` elements of `dtype`. `comm` is live
        // while `with_comm` holds the lock.
        self.with_comm("ncclReduceScatter", |comm, stream| unsafe {
            nccl_result::reduce_scatter(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                count,
                nccl_dtype,
                nccl_op,
                comm,
                stream,
            )
        })
    }

    unsafe fn send(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        dest: usize,
        _tag: u32,
    ) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        let dest = to_nccl_int("dest", dest)?;
        // SAFETY: the caller guarantees `ptr` holds `count` elements of `dtype`.
        // `comm` is live while `with_comm` holds the lock.
        self.with_comm("ncclSend", |comm, stream| unsafe {
            nccl_result::send(ptr as *const c_void, count, nccl_dtype, dest, comm, stream)
        })
    }

    unsafe fn recv(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        src: usize,
        _tag: u32,
    ) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        let src = to_nccl_int("src", src)?;
        // SAFETY: the caller guarantees `ptr` has room for `count` elements of
        // `dtype`. `comm` is live while `with_comm` holds the lock.
        self.with_comm("ncclRecv", |comm, stream| unsafe {
            nccl_result::recv(ptr as *mut c_void, count, nccl_dtype, src, comm, stream)
        })
    }

    /// Wait for queued work on the communicator stream.
    ///
    /// Polls NCCL for asynchronous errors while it waits. An error NCCL reports
    /// aborts the communicator and returns the error.
    /// [`NcclCommunicator::abort`] from another thread also ends the wait with an error.
    fn sync(&self) -> Result<()> {
        self.wait_stream()
    }

    fn as_stream_sync(&self) -> Option<&dyn StreamSyncOps> {
        Some(self)
    }

    fn barrier(&self) -> Result<()> {
        // NCCL has no barrier. A zero-element all-reduce between two syncs acts as one.
        self.sync()?;
        // SAFETY: a zero-element all-reduce reads and writes no memory.
        // `comm` is live while `with_comm` holds the lock.
        self.with_comm("ncclAllReduce (barrier)", |comm, stream| unsafe {
            nccl_result::all_reduce(
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                nccl_sys::ncclDataType_t::ncclFloat32,
                nccl_sys::ncclRedOp_t::ncclSum,
                comm,
                stream,
            )
        })?;
        self.sync()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
    use cudarc::nccl::result as nr;

    use super::*;
    use crate::runtime::cuda::communicator::NcclId;

    // Helper: get raw device pointer from a CudaSlice for test use
    fn slice_ptr<T>(slice: &cudarc::driver::CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
        let (ptr, _guard) = slice.device_ptr(stream);
        ptr as u64
    }

    /// Default streams on the first two devices, or `None` with a skip message.
    fn two_streams(test: &str) -> Option<Vec<Arc<CudaStream>>> {
        let n = CudaContext::device_count().map_or(0, |n| n.max(0) as usize);
        if n < 2 {
            eprintln!("skipping {test}: NCCL multi-rank tests need 2 CUDA devices, found {n}");
            return None;
        }
        Some(
            (0..2)
                .map(|i| CudaContext::new(i).unwrap().default_stream())
                .collect(),
        )
    }

    // ── Multi-GPU tests (require 2+ GPUs) ──────────────────────────────

    #[test]
    fn test_nccl_metadata() {
        let Some(streams) = two_streams("test_nccl_metadata") else {
            return;
        };
        let comms = NcclCommunicator::from_streams(streams).unwrap();
        assert_eq!(comms.len(), 2);
        assert_eq!(comms[0].world_size(), 2);
        assert_eq!(comms[1].world_size(), 2);
        assert_eq!(comms[0].rank(), 0);
        assert_eq!(comms[1].rank(), 1);
    }

    #[test]
    fn test_nccl_all_reduce_f32() {
        let Some(streams) = two_streams("test_nccl_all_reduce_f32") else {
            return;
        };
        let n = 4;
        let n_devices = streams.len();
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        // Each rank has [rank+1, rank+1, rank+1, rank+1]
        let mut slices = Vec::new();
        for (i, stream) in streams.iter().enumerate() {
            slices.push(stream.clone_htod(&vec![(i + 1) as f32; n]).unwrap());
        }

        nr::group_start().unwrap();
        for (i, comm) in comms.iter().enumerate() {
            unsafe {
                comm.all_reduce(
                    slice_ptr(&slices[i], &streams[i]),
                    n,
                    DType::F32,
                    ReduceOp::Sum,
                )
                .unwrap();
            }
        }
        nr::group_end().unwrap();

        for (i, comm) in comms.iter().enumerate() {
            comm.sync().unwrap();
            let out = streams[i].clone_dtoh(&slices[i]).unwrap();
            let expected = (n_devices * (n_devices + 1)) as f32 / 2.0;
            for v in &out {
                assert!(
                    (*v - expected).abs() < 1e-5,
                    "rank {i}: expected {expected}, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_nccl_from_rank_all_reduce_across_threads() {
        let Some(streams) = two_streams("test_nccl_from_rank_all_reduce_across_threads") else {
            return;
        };
        // One thread per rank stands in for one process per rank.
        let id = NcclId::from_bytes(NcclId::new().unwrap().bytes());
        let handles: Vec<_> = streams
            .into_iter()
            .enumerate()
            .map(|(rank, stream)| {
                std::thread::spawn(move || {
                    let comm = NcclCommunicator::from_rank(stream.clone(), rank, 2, id).unwrap();
                    let slice = stream.clone_htod(&[(rank + 1) as f32; 4]).unwrap();
                    unsafe {
                        comm.all_reduce(slice_ptr(&slice, &stream), 4, DType::F32, ReduceOp::Sum)
                            .unwrap();
                    }
                    comm.sync().unwrap();
                    comm.async_error().unwrap();
                    stream.clone_dtoh(&slice).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), vec![3.0f32; 4]);
        }
    }

    #[test]
    fn test_nccl_broadcast() {
        let Some(streams) = two_streams("test_nccl_broadcast") else {
            return;
        };
        let n = 4;
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        let mut slices = Vec::new();
        for (i, stream) in streams.iter().enumerate() {
            let value = if i == 0 { 42.0f32 } else { 0.0f32 };
            slices.push(stream.clone_htod(&vec![value; n]).unwrap());
        }

        nr::group_start().unwrap();
        for (i, comm) in comms.iter().enumerate() {
            unsafe {
                comm.broadcast(slice_ptr(&slices[i], &streams[i]), n, DType::F32, 0)
                    .unwrap();
            }
        }
        nr::group_end().unwrap();

        for (i, comm) in comms.iter().enumerate() {
            comm.sync().unwrap();
            let out = streams[i].clone_dtoh(&slices[i]).unwrap();
            assert_eq!(out, vec![42.0f32; n], "rank {i} broadcast mismatch");
        }
    }

    #[test]
    fn test_nccl_all_gather() {
        let Some(streams) = two_streams("test_nccl_all_gather") else {
            return;
        };
        let n = 2; // elements per rank
        let n_devices = streams.len();
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        let mut send_slices = Vec::new();
        let mut recv_slices = Vec::new();
        for (i, stream) in streams.iter().enumerate() {
            send_slices.push(stream.clone_htod(&vec![(i + 1) as f32; n]).unwrap());
            recv_slices.push(stream.alloc_zeros::<f32>(n * n_devices).unwrap());
        }

        nr::group_start().unwrap();
        for (i, comm) in comms.iter().enumerate() {
            unsafe {
                comm.all_gather(
                    slice_ptr(&send_slices[i], &streams[i]),
                    slice_ptr(&recv_slices[i], &streams[i]),
                    n,
                    DType::F32,
                )
                .unwrap();
            }
        }
        nr::group_end().unwrap();

        for (i, comm) in comms.iter().enumerate() {
            comm.sync().unwrap();
            let out = streams[i].clone_dtoh(&recv_slices[i]).unwrap();
            // Expected: [1.0, 1.0, 2.0, 2.0] for 2 devices
            let mut expected = Vec::new();
            for rank in 0..n_devices {
                expected.extend(std::iter::repeat_n((rank + 1) as f32, n));
            }
            assert_eq!(out, expected, "rank {i} all_gather mismatch");
        }
    }

    #[test]
    fn test_nccl_send_recv() {
        let Some(streams) = two_streams("test_nccl_send_recv") else {
            return;
        };
        let n = 4;
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        let send_slice = streams[0].clone_htod(&vec![99.0f32; n]).unwrap();
        let recv_slice = streams[1].alloc_zeros::<f32>(n).unwrap();

        nr::group_start().unwrap();
        unsafe {
            comms[0]
                .send(slice_ptr(&send_slice, &streams[0]), n, DType::F32, 1, 0)
                .unwrap();
            comms[1]
                .recv(slice_ptr(&recv_slice, &streams[1]), n, DType::F32, 0, 0)
                .unwrap();
        }
        nr::group_end().unwrap();

        comms[0].sync().unwrap();
        comms[1].sync().unwrap();
        let out = streams[1].clone_dtoh(&recv_slice).unwrap();
        assert_eq!(out, vec![99.0f32; n]);
    }

    #[test]
    fn test_nccl_sync_barrier() {
        let Some(streams) = two_streams("test_nccl_sync_barrier") else {
            return;
        };
        let comms = NcclCommunicator::from_streams(streams).unwrap();

        for comm in &comms {
            comm.sync().unwrap();
        }
        // barrier requires all ranks to participate
        nr::group_start().unwrap();
        for comm in &comms {
            comm.barrier().unwrap();
        }
        nr::group_end().unwrap();
    }
}
