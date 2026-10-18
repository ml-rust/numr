//! `StreamSyncOps` for NCCL: events that order the compute and communicator streams.

use cudarc::driver::sys::{
    CUevent, CUevent_flags, CUresult, CUstream, cuEventCreate, cuEventDestroy_v2, cuEventRecord,
    cuStreamWaitEvent,
};

use super::comm::NcclCommunicator;
use crate::error::{Error, Result};
use crate::runtime::communicator::StreamSyncOps;

/// Turn a driver result into a numr error that names the failed call.
fn driver_check(result: CUresult, call: &str) -> Result<()> {
    if result == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Backend(format!("{call} failed: {result:?}")))
    }
}

impl StreamSyncOps for NcclCommunicator {
    fn create_event(&self) -> Result<u64> {
        self.bind()?;
        let mut event = std::ptr::null_mut();
        // SAFETY: `event` is a valid out slot, and the communicator context is current.
        let result =
            unsafe { cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) };
        driver_check(result, "cuEventCreate")?;
        Ok(event as u64)
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller passes an event from `create_event` that it has not destroyed.
        let result = unsafe { cuEventDestroy_v2(event as CUevent) };
        driver_check(result, "cuEventDestroy")
    }

    fn record_on_comm_stream(&self, event: u64) -> Result<()> {
        let stream = self.bound_stream()? as CUstream;
        // SAFETY: `event` comes from `create_event`. `stream` belongs to this communicator.
        let result = unsafe { cuEventRecord(event as CUevent, stream) };
        driver_check(result, "cuEventRecord on comm stream")
    }

    fn record_on_stream(&self, event: u64, stream_handle: u64) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller passes a live event and a live stream of this context.
        let result = unsafe { cuEventRecord(event as CUevent, stream_handle as CUstream) };
        driver_check(result, "cuEventRecord on stream")
    }

    fn comm_stream_wait_event(&self, event: u64) -> Result<()> {
        let stream = self.bound_stream()? as CUstream;
        // SAFETY: `event` comes from `create_event`. `stream` belongs to this communicator.
        let result = unsafe { cuStreamWaitEvent(stream, event as CUevent, 0) };
        driver_check(result, "cuStreamWaitEvent on comm stream")
    }

    fn stream_wait_event(&self, stream_handle: u64, event: u64) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller passes a live stream of this context and a live event.
        let result = unsafe { cuStreamWaitEvent(stream_handle as CUstream, event as CUevent, 0) };
        driver_check(result, "cuStreamWaitEvent on stream")
    }

    /// Same as `Communicator::sync`: polls NCCL for asynchronous errors while it waits.
    fn sync_comm_stream(&self) -> Result<()> {
        self.wait_stream()
    }
}
