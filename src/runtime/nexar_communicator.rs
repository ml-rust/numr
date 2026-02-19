//! nexar-backed distributed communicator for inter-node collective operations.
//!
//! Wraps [`nexar::SyncClient`] and implements [`Communicator`] so that numr's
//! existing distributed patterns (gradient sync, tensor parallelism) work
//! transparently over QUIC transport.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::communicator::{Communicator, ReduceOp};

/// Maps a numr `DType` to a nexar `DataType`.
///
/// Returns `Err` for types nexar doesn't support (Complex, Bool, FP8, I16, U16).
fn to_nexar_dtype(dtype: DType) -> Result<nexar::DataType> {
    match dtype {
        DType::F32 => Ok(nexar::DataType::F32),
        DType::F64 => Ok(nexar::DataType::F64),
        DType::F16 => Ok(nexar::DataType::F16),
        DType::BF16 => Ok(nexar::DataType::BF16),
        DType::I8 => Ok(nexar::DataType::I8),
        DType::I32 => Ok(nexar::DataType::I32),
        DType::I64 => Ok(nexar::DataType::I64),
        DType::U8 => Ok(nexar::DataType::U8),
        DType::U32 => Ok(nexar::DataType::U32),
        DType::U64 => Ok(nexar::DataType::U64),
        _ => Err(Error::Backend(format!(
            "nexar: unsupported dtype {dtype:?} for collective operation"
        ))),
    }
}

/// Maps a numr `ReduceOp` to a nexar `ReduceOp`.
fn to_nexar_op(op: ReduceOp) -> nexar::ReduceOp {
    match op {
        ReduceOp::Sum => nexar::ReduceOp::Sum,
        ReduceOp::Prod => nexar::ReduceOp::Prod,
        ReduceOp::Min => nexar::ReduceOp::Min,
        ReduceOp::Max => nexar::ReduceOp::Max,
    }
}

/// Maps a nexar error to a numr error.
fn map_err(e: nexar::NexarError) -> Error {
    Error::Backend(format!("nexar: {e}"))
}

/// Distributed communicator backed by [`nexar::SyncClient`].
///
/// Provides inter-node collective operations (allreduce, broadcast, etc.)
/// over QUIC transport. For intra-node GPU-GPU communication, use
/// `NcclCommunicator` instead — NVLink/PCIe is orders of magnitude faster
/// than any network.
///
/// # Usage
///
/// ```ignore
/// use nexar::{CpuAdapter, SyncClient};
/// use numr::runtime::{NexarNetCommunicator, Communicator};
/// use std::sync::Arc;
///
/// let adapter = Arc::new(CpuAdapter::new());
/// let clients = SyncClient::bootstrap_local(4, adapter).unwrap();
/// let comms: Vec<NexarNetCommunicator> = clients
///     .into_iter()
///     .map(NexarNetCommunicator::new)
///     .collect();
/// ```
pub struct NexarNetCommunicator {
    client: nexar::SyncClient,
}

impl NexarNetCommunicator {
    /// Wrap an existing nexar `SyncClient`.
    pub fn new(client: nexar::SyncClient) -> Self {
        Self { client }
    }
}

impl Communicator for NexarNetCommunicator {
    fn world_size(&self) -> usize {
        self.client.world_size() as usize
    }

    fn rank(&self) -> usize {
        self.client.rank() as usize
    }

    unsafe fn all_reduce(&self, ptr: u64, count: usize, dtype: DType, op: ReduceOp) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        let no = to_nexar_op(op);
        unsafe { self.client.all_reduce(ptr, count, nd, no).map_err(map_err) }
    }

    unsafe fn broadcast(&self, ptr: u64, count: usize, dtype: DType, root: usize) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        unsafe {
            self.client
                .broadcast(ptr, count, nd, root as u32)
                .map_err(map_err)
        }
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        unsafe {
            self.client
                .all_gather(send_ptr, recv_ptr, count, nd)
                .map_err(map_err)
        }
    }

    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        let no = to_nexar_op(op);
        unsafe {
            self.client
                .reduce_scatter(send_ptr, recv_ptr, count, nd, no)
                .map_err(map_err)
        }
    }

    unsafe fn send(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        dest: usize,
        tag: u32,
    ) -> Result<()> {
        let nd = to_nexar_dtype(dtype)?;
        let size = count * nd.size_in_bytes();
        unsafe {
            self.client
                .send(ptr, size, dest as u32, tag)
                .map_err(map_err)
        }
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
        unsafe {
            self.client
                .recv(ptr, size, src as u32, tag)
                .map_err(map_err)
        }
    }

    fn sync(&self) -> Result<()> {
        // nexar operations are synchronous (block_on), so sync is a no-op.
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        self.client.barrier().map_err(map_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(to_nexar_dtype(DType::F32).unwrap(), nexar::DataType::F32);
        assert_eq!(to_nexar_dtype(DType::F64).unwrap(), nexar::DataType::F64);
        assert_eq!(to_nexar_dtype(DType::F16).unwrap(), nexar::DataType::F16);
        assert_eq!(to_nexar_dtype(DType::BF16).unwrap(), nexar::DataType::BF16);
        assert_eq!(to_nexar_dtype(DType::I8).unwrap(), nexar::DataType::I8);
        assert_eq!(to_nexar_dtype(DType::I32).unwrap(), nexar::DataType::I32);
        assert_eq!(to_nexar_dtype(DType::I64).unwrap(), nexar::DataType::I64);
        assert_eq!(to_nexar_dtype(DType::U8).unwrap(), nexar::DataType::U8);
        assert_eq!(to_nexar_dtype(DType::U32).unwrap(), nexar::DataType::U32);
        assert_eq!(to_nexar_dtype(DType::U64).unwrap(), nexar::DataType::U64);
    }

    #[test]
    fn test_dtype_mapping_unsupported() {
        assert!(to_nexar_dtype(DType::Bool).is_err());
        assert!(to_nexar_dtype(DType::Complex64).is_err());
        assert!(to_nexar_dtype(DType::Complex128).is_err());
    }

    #[test]
    fn test_reduce_op_mapping() {
        assert_eq!(to_nexar_op(ReduceOp::Sum), nexar::ReduceOp::Sum);
        assert_eq!(to_nexar_op(ReduceOp::Prod), nexar::ReduceOp::Prod);
        assert_eq!(to_nexar_op(ReduceOp::Min), nexar::ReduceOp::Min);
        assert_eq!(to_nexar_op(ReduceOp::Max), nexar::ReduceOp::Max);
    }

    #[test]
    fn test_nexar_communicator_metadata() {
        let adapter = std::sync::Arc::new(nexar::CpuAdapter::new());
        let clients = nexar::SyncClient::bootstrap_local(2, adapter).unwrap();
        let comms: Vec<NexarNetCommunicator> =
            clients.into_iter().map(NexarNetCommunicator::new).collect();

        assert_eq!(comms[0].world_size(), 2);
        assert_eq!(comms[0].rank(), 0);
        assert_eq!(comms[1].rank(), 1);
    }

    #[test]
    fn test_nexar_allreduce_f32() {
        let adapter = std::sync::Arc::new(nexar::CpuAdapter::new());
        let clients = nexar::SyncClient::bootstrap_local(2, adapter).unwrap();
        let comms: Vec<NexarNetCommunicator> =
            clients.into_iter().map(NexarNetCommunicator::new).collect();

        // Each rank has its own data; run allreduce concurrently.
        std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .iter()
                .enumerate()
                .map(|(i, comm)| {
                    s.spawn(move || {
                        let val = (i + 1) as f32;
                        let mut data = vec![val; 4];
                        let ptr = data.as_mut_ptr() as u64;
                        unsafe {
                            comm.all_reduce(ptr, 4, DType::F32, ReduceOp::Sum).unwrap();
                        }
                        data
                    })
                })
                .collect();

            for h in handles {
                let data = h.join().unwrap();
                // 1.0 + 2.0 = 3.0
                assert_eq!(data, vec![3.0f32; 4]);
            }
        });
    }
}
