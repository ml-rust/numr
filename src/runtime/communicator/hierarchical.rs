//! Hierarchical communicator: NCCL intra-node + nexar inter-node.
//!
//! Wraps [`nexar_nccl::HierarchicalComm`] and implements [`Communicator`] so
//! that numr's distributed patterns work transparently over hierarchical
//! GPU clusters. Uses NCCL for same-node GPU-GPU (NVLink/PCIe) and nexar
//! QUIC for cross-node communication.

use super::nexar_compat::{to_nexar_dtype, to_nexar_op};
use super::{Communicator, ReduceOp};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Maps a nexar-nccl error to a numr error.
fn map_err(e: nexar_nccl::NcclCommError) -> Error {
    Error::Backend(format!("hierarchical communicator: {e}"))
}

/// Maps a nexar error to a numr error.
fn map_nexar_err(e: nexar::NexarError) -> Error {
    Error::Backend(format!("hierarchical communicator (nexar): {e}"))
}

/// Hierarchical communicator backed by [`nexar_nccl::HierarchicalComm`].
///
/// Combines NCCL for intra-node GPU-GPU with nexar for inter-node
/// communication. This is the standard 2D decomposition used by
/// Megatron-LM and DeepSpeed.
///
/// # Construction
///
/// Use [`nexar_nccl::form_hierarchical_comm`] to create the underlying
/// `HierarchicalComm`, then wrap it:
///
/// ```ignore
/// let hcomm = unsafe { form_hierarchical_comm(nexar_client, stream).await? };
/// let rt = tokio::runtime::Runtime::new()?;
/// let comm = HierarchicalCommunicator::new(hcomm, rt);
/// ```
pub struct HierarchicalCommunicator {
    comm: nexar_nccl::HierarchicalComm,
    rt: tokio::runtime::Runtime,
}

impl HierarchicalCommunicator {
    /// Wrap an existing `HierarchicalComm` with a tokio runtime for async→sync bridging.
    pub fn new(comm: nexar_nccl::HierarchicalComm, rt: tokio::runtime::Runtime) -> Self {
        Self { comm, rt }
    }

    /// Reference to the underlying hierarchical communicator.
    pub fn inner(&self) -> &nexar_nccl::HierarchicalComm {
        &self.comm
    }
}

impl Communicator for HierarchicalCommunicator {
    fn world_size(&self) -> usize {
        self.comm.world_size() as usize
    }

    fn rank(&self) -> usize {
        self.comm.rank() as usize
    }

    unsafe fn all_reduce(&self, ptr: u64, count: usize, dtype: DType, op: ReduceOp) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        let no = to_nexar_op(op);
        self.rt
            .block_on(unsafe { self.comm.allreduce(ptr, count, nd, no) })
            .map_err(map_err)
    }

    unsafe fn broadcast(&self, ptr: u64, count: usize, dtype: DType, root: usize) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        self.rt
            .block_on(unsafe { self.comm.broadcast(ptr, count, nd, root as u32) })
            .map_err(map_err)
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        self.rt
            .block_on(unsafe { self.comm.allgather(send_ptr, recv_ptr, count, nd) })
            .map_err(map_err)
    }

    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()> {
        // HierarchicalComm doesn't expose reduce_scatter directly.
        // Compose: allreduce the full buffer, then each rank copies its chunk.
        //
        // allreduce is in-place on send_ptr, so we need send_ptr to hold the
        // full data (count * world_size elements). After allreduce, each rank
        // copies its slice (rank * count .. (rank+1) * count) into recv_ptr.
        let nd = to_nexar_dtype(dtype)?;
        let no = to_nexar_op(op);
        let ws = self.comm.world_size() as usize;
        let total = count * ws;

        // Step 1: allreduce the full buffer in-place
        self.rt
            .block_on(unsafe { self.comm.allreduce(send_ptr, total, nd, no) })
            .map_err(map_err)?;

        // Step 2: copy this rank's chunk to recv_ptr
        let elem_size = dtype.size_in_bytes();
        let offset = self.comm.rank() as usize * count * elem_size;
        let bytes = count * elem_size;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (send_ptr as *const u8).add(offset),
                recv_ptr as *mut u8,
                bytes,
            );
        }
        Ok(())
    }

    unsafe fn send(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        dest: usize,
        tag: u32,
    ) -> Result<()> {
        // Route through the nexar client for point-to-point.
        let nd = to_nexar_dtype(dtype)?;
        let size = count * nd.size_in_bytes();
        self.rt
            .block_on(unsafe { self.comm.nexar().send(ptr, size, dest as u32, tag) })
            .map_err(map_nexar_err)
    }

    unsafe fn recv(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        src: usize,
        tag: u32,
    ) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        let size = count * nd.size_in_bytes();
        self.rt
            .block_on(unsafe { self.comm.nexar().recv(ptr, size, src as u32, tag) })
            .map_err(map_nexar_err)
    }

    fn sync(&self) -> Result<()> {
        self.comm.synchronize().map_err(map_err)
    }

    fn barrier(&self) -> Result<()> {
        self.rt.block_on(self.comm.barrier()).map_err(map_err)
    }
}
