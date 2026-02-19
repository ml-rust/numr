//! NCCL-backed collective communication for multi-GPU
//!
//! Wraps cudarc's `nccl::Comm` and implements numr's `Communicator` trait.
//! Uses raw `nccl::result` FFI to handle runtime `DType` dispatch (cudarc's
//! safe API requires compile-time `NcclType` generics).

use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::nccl::{self, result as nccl_result, sys as nccl_sys};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::communicator::{Communicator, ReduceOp};

/// NCCL communicator wrapping a single `cudarc::nccl::Comm` (one per rank).
pub struct NcclCommunicator {
    comm: nccl::Comm,
}

// SAFETY: NCCL comms are thread-safe for submission from the owning thread.
// The Comm internally holds an Arc<CudaStream> which is Send+Sync.
unsafe impl Send for NcclCommunicator {}
unsafe impl Sync for NcclCommunicator {}

impl NcclCommunicator {
    /// Wrap an existing cudarc NCCL communicator.
    pub fn new(comm: nccl::Comm) -> Self {
        Self { comm }
    }

    /// Create communicators for all given streams (single-process, multi-GPU).
    ///
    /// Returns one `NcclCommunicator` per stream, with ranks assigned in order.
    pub fn from_streams(streams: Vec<Arc<CudaStream>>) -> Result<Vec<Self>> {
        let comms = nccl::Comm::from_devices(streams)
            .map_err(|e| Error::Backend(format!("NCCL init failed: {e:?}")))?;
        Ok(comms.into_iter().map(|c| Self { comm: c }).collect())
    }

    /// Access the underlying cudarc `Comm`.
    pub fn inner(&self) -> &nccl::Comm {
        &self.comm
    }

    /// Get the raw NCCL comm handle for FFI calls.
    fn raw_comm(&self) -> nccl_sys::ncclComm_t {
        // Access the private field via the Comm's public API indirectly.
        // We need the raw pointer. Comm stores it as `comm: sys::ncclComm_t`.
        // Unfortunately cudarc doesn't expose this directly, so we use
        // a transmute-based approach to read the first field.
        //
        // SAFETY: Comm's first field is `comm: sys::ncclComm_t` (a raw pointer).
        // This is verified by cudarc 0.18's source code.
        unsafe { std::ptr::read((&self.comm as *const nccl::Comm).cast::<nccl_sys::ncclComm_t>()) }
    }

    /// Get the raw CUDA stream handle for FFI calls.
    fn raw_stream(&self) -> nccl_sys::cudaStream_t {
        self.comm.stream().cu_stream() as nccl_sys::cudaStream_t
    }
}

/// Map numr `DType` to NCCL data type.
fn dtype_to_nccl(dtype: DType) -> Result<nccl_sys::ncclDataType_t> {
    match dtype {
        DType::F32 => Ok(nccl_sys::ncclDataType_t::ncclFloat32),
        DType::F64 => Ok(nccl_sys::ncclDataType_t::ncclFloat64),
        DType::F16 => Ok(nccl_sys::ncclDataType_t::ncclFloat16),
        DType::BF16 => Ok(nccl_sys::ncclDataType_t::ncclBfloat16),
        DType::FP8E4M3 => Ok(nccl_sys::ncclDataType_t::ncclFloat8e4m3),
        DType::FP8E5M2 => Ok(nccl_sys::ncclDataType_t::ncclFloat8e5m2),
        DType::I32 => Ok(nccl_sys::ncclDataType_t::ncclInt32),
        DType::I64 => Ok(nccl_sys::ncclDataType_t::ncclInt64),
        DType::I8 => Ok(nccl_sys::ncclDataType_t::ncclInt8),
        DType::U32 => Ok(nccl_sys::ncclDataType_t::ncclUint32),
        DType::U8 => Ok(nccl_sys::ncclDataType_t::ncclUint8),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "nccl_communication",
        }),
    }
}

/// Map numr `ReduceOp` to NCCL reduction operation.
fn reduce_op_to_nccl(op: ReduceOp) -> nccl_sys::ncclRedOp_t {
    match op {
        ReduceOp::Sum => nccl_sys::ncclRedOp_t::ncclSum,
        ReduceOp::Prod => nccl_sys::ncclRedOp_t::ncclProd,
        ReduceOp::Min => nccl_sys::ncclRedOp_t::ncclMin,
        ReduceOp::Max => nccl_sys::ncclRedOp_t::ncclMax,
    }
}

/// Convert NCCL error to numr error.
fn nccl_err(e: nccl_result::NcclError) -> Error {
    Error::Backend(format!("NCCL error: {e:?}"))
}

impl Communicator for NcclCommunicator {
    fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    fn rank(&self) -> usize {
        self.comm.rank()
    }

    unsafe fn all_reduce(&self, ptr: u64, count: usize, dtype: DType, op: ReduceOp) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        let nccl_op = reduce_op_to_nccl(op);
        // In-place: sendbuff == recvbuff
        unsafe {
            nccl_result::all_reduce(
                ptr as *const std::ffi::c_void,
                ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                nccl_op,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        Ok(())
    }

    unsafe fn broadcast(&self, ptr: u64, count: usize, dtype: DType, root: usize) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        // In-place: sendbuff == recvbuff
        unsafe {
            nccl_result::broadcast(
                ptr as *const std::ffi::c_void,
                ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                root as i32,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        Ok(())
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        let nccl_dtype = dtype_to_nccl(dtype)?;
        unsafe {
            nccl_result::all_gather(
                send_ptr as *const std::ffi::c_void,
                recv_ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        Ok(())
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
        unsafe {
            nccl_result::reduce_scatter(
                send_ptr as *const std::ffi::c_void,
                recv_ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                nccl_op,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        Ok(())
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
        unsafe {
            nccl_result::send(
                ptr as *const std::ffi::c_void,
                count,
                nccl_dtype,
                dest as i32,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        Ok(())
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
        unsafe {
            nccl_result::recv(
                ptr as *mut std::ffi::c_void,
                count,
                nccl_dtype,
                src as i32,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        self.comm
            .stream()
            .synchronize()
            .map_err(|e| Error::Backend(format!("CUDA stream sync failed: {e}")))?;
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        // NCCL has no explicit barrier. Sync the stream first, then do a
        // zero-byte all_reduce as a collective synchronization point.
        self.sync()?;
        unsafe {
            nccl_result::all_reduce(
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                nccl_sys::ncclDataType_t::ncclFloat32,
                nccl_sys::ncclRedOp_t::ncclSum,
                self.raw_comm(),
                self.raw_stream(),
            )
            .map_err(nccl_err)?;
        }
        self.sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NcclCommunicator>();
    }

    #[test]
    fn test_dtype_to_nccl_mapping() {
        assert!(dtype_to_nccl(DType::F32).is_ok());
        assert!(dtype_to_nccl(DType::F64).is_ok());
        assert!(dtype_to_nccl(DType::F16).is_ok());
        assert!(dtype_to_nccl(DType::BF16).is_ok());
        assert!(dtype_to_nccl(DType::I32).is_ok());
        assert!(dtype_to_nccl(DType::I64).is_ok());
        assert!(dtype_to_nccl(DType::U32).is_ok());
        assert!(dtype_to_nccl(DType::U8).is_ok());
        assert!(dtype_to_nccl(DType::Bool).is_err());
    }

    #[test]
    fn test_reduce_op_mapping() {
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Sum),
            nccl_sys::ncclRedOp_t::ncclSum
        );
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Prod),
            nccl_sys::ncclRedOp_t::ncclProd
        );
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Min),
            nccl_sys::ncclRedOp_t::ncclMin
        );
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Max),
            nccl_sys::ncclRedOp_t::ncclMax
        );
    }

    // Helper: get raw device pointer from a CudaSlice for test use
    fn slice_ptr<T>(slice: &cudarc::driver::CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
        use cudarc::driver::DevicePtr;
        let (ptr, _guard) = slice.device_ptr(stream);
        ptr as u64
    }

    // ── Multi-GPU tests (require 2+ GPUs) ──────────────────────────────

    #[test]
    #[ignore]
    fn test_nccl_metadata() {
        let ctx0 = cudarc::driver::CudaContext::new(0).unwrap();
        let ctx1 = cudarc::driver::CudaContext::new(1).unwrap();
        let streams = vec![ctx0.default_stream(), ctx1.default_stream()];
        let comms = NcclCommunicator::from_streams(streams).unwrap();
        assert_eq!(comms.len(), 2);
        assert_eq!(comms[0].world_size(), 2);
        assert_eq!(comms[1].world_size(), 2);
        assert_eq!(comms[0].rank(), 0);
        assert_eq!(comms[1].rank(), 1);
    }

    #[test]
    #[ignore]
    fn test_nccl_all_reduce_f32() {
        use cudarc::driver::CudaContext;
        use cudarc::nccl::result as nr;

        let n = 4;
        let n_devices = CudaContext::device_count().unwrap().min(2) as usize;
        if n_devices < 2 {
            return;
        }

        let streams: Vec<_> = (0..n_devices)
            .map(|i| {
                let ctx = CudaContext::new(i).unwrap();
                ctx.default_stream()
            })
            .collect();
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        // Each rank has [rank+1, rank+1, rank+1, rank+1]
        let mut slices = Vec::new();
        for i in 0..n_devices {
            let data = vec![(i + 1) as f32; n];
            let slice = streams[i].clone_htod(&data).unwrap();
            slices.push(slice);
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
    #[ignore]
    fn test_nccl_broadcast() {
        use cudarc::driver::CudaContext;
        use cudarc::nccl::result as nr;

        let n = 4;
        let n_devices = CudaContext::device_count().unwrap().min(2) as usize;
        if n_devices < 2 {
            return;
        }

        let streams: Vec<_> = (0..n_devices)
            .map(|i| CudaContext::new(i).unwrap().default_stream())
            .collect();
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        let mut slices = Vec::new();
        for (i, stream) in streams.iter().enumerate() {
            let data = if i == 0 {
                vec![42.0f32; n]
            } else {
                vec![0.0f32; n]
            };
            slices.push(stream.clone_htod(&data).unwrap());
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
    #[ignore]
    fn test_nccl_all_gather() {
        use cudarc::driver::CudaContext;
        use cudarc::nccl::result as nr;

        let n = 2; // elements per rank
        let n_devices = CudaContext::device_count().unwrap().min(2) as usize;
        if n_devices < 2 {
            return;
        }

        let streams: Vec<_> = (0..n_devices)
            .map(|i| CudaContext::new(i).unwrap().default_stream())
            .collect();
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        let mut send_slices = Vec::new();
        let mut recv_slices = Vec::new();
        for (i, stream) in streams.iter().enumerate() {
            let data = vec![(i + 1) as f32; n];
            send_slices.push(stream.clone_htod(&data).unwrap());
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
    #[ignore]
    fn test_nccl_send_recv() {
        use cudarc::driver::CudaContext;
        use cudarc::nccl::result as nr;

        let n = 4;
        let n_devices = CudaContext::device_count().unwrap().min(2) as usize;
        if n_devices < 2 {
            return;
        }

        let streams: Vec<_> = (0..n_devices)
            .map(|i| CudaContext::new(i).unwrap().default_stream())
            .collect();
        let comms = NcclCommunicator::from_streams(streams.clone()).unwrap();

        let send_data = vec![99.0f32; n];
        let send_slice = streams[0].clone_htod(&send_data).unwrap();
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
    #[ignore]
    fn test_nccl_sync_barrier() {
        use cudarc::driver::CudaContext;

        let n_devices = CudaContext::device_count().unwrap().min(2) as usize;
        if n_devices < 2 {
            return;
        }

        let streams: Vec<_> = (0..n_devices)
            .map(|i| CudaContext::new(i).unwrap().default_stream())
            .collect();
        let comms = NcclCommunicator::from_streams(streams).unwrap();

        for comm in &comms {
            comm.sync().unwrap();
        }
        // barrier requires all ranks to participate
        cudarc::nccl::result::group_start().unwrap();
        for comm in &comms {
            comm.barrier().unwrap();
        }
        cudarc::nccl::result::group_end().unwrap();
    }
}
