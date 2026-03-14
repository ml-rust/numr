//! No-op communicator for single-device operation.

use crate::dtype::DType;
use crate::error::Result;

use super::{Communicator, ReduceOp};

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
        let ops = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min, ReduceOp::Max];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j]);
            }
        }
    }
}
