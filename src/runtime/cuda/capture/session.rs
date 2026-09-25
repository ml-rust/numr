//! RAII bracket around one CUDA graph capture region.
//!
//! A capture holds three pieces of shared state at once: the device's capture
//! lock, the allocator's freeze flag with its optional arena, and the driver's
//! capture mode on the compute stream. Leaving any of them set turns every
//! later operation on the device into an error, so all three are released by
//! `Drop`, including when the capture closure panics and the stack unwinds.

use cudarc::driver::safe::CudaGraph as CudarcGraph;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use super::GuardedStream;
use super::lock::CapturePermit;
use crate::error::Result;
use crate::runtime::common::Allocator;
use crate::runtime::cuda::allocator::CudaAllocator;

/// Instantiation flags used for every capture in this crate.
///
/// Graph-managed memory allocated during capture is freed on each launch.
/// Tensors the caller allocated before capture are untouched by it.
pub(crate) const INSTANTIATE_FLAGS: CUgraphInstantiate_flags =
    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

/// An open capture region on one device's compute stream.
pub(crate) struct CaptureSession<'a> {
    _permit: CapturePermit<'a>,
    stream: &'a GuardedStream,
    allocator: &'a CudaAllocator,
    /// The driver stream is in capture mode and `end_capture` is still owed.
    capturing: bool,
    /// The allocator is frozen and `unfreeze` is still owed.
    frozen: bool,
}

impl<'a> CaptureSession<'a> {
    /// Open a capture region on `stream`.
    ///
    /// Order: take the device's capture lock, run `setup` (which installs an
    /// arena when the caller wants one), freeze the allocator, then put the
    /// stream in capture mode. A failure at any step undoes the earlier ones.
    ///
    /// # Errors
    ///
    /// Returns an error when this thread is already capturing, when `setup`
    /// fails, or when the driver refuses to begin capture.
    pub(crate) fn begin(
        stream: &'a GuardedStream,
        allocator: &'a CudaAllocator,
        setup: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let permit = CapturePermit::acquire(stream.capture_lock(), stream.device_index())?;
        setup()?;

        let mut session = Self {
            _permit: permit,
            stream,
            allocator,
            capturing: false,
            frozen: false,
        };

        // Frozen allocations go straight to the driver (or into the arena), so
        // the graph gets real allocation nodes instead of free-list reuse.
        session.allocator.freeze();
        session.frozen = true;

        session
            .stream
            .raw()
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
        session.capturing = true;

        Ok(session)
    }

    /// Close the capture region.
    ///
    /// Returns the captured graph, or `None` when the closure recorded
    /// nothing, together with the arena's peak footprint read before the
    /// allocator is restored.
    ///
    /// # Errors
    ///
    /// Returns an error when the driver rejects `end_capture`.
    pub(crate) fn end(mut self) -> (Result<Option<CudarcGraph>>, Option<usize>) {
        let graph = self.stream.arc().end_capture(INSTANTIATE_FLAGS);
        self.capturing = false;

        // Read before the restore: unfreezing clears the arena bookkeeping.
        let arena_bytes_used = self.allocator.arena_high_water();
        self.allocator.unfreeze();
        self.frozen = false;

        (graph.map_err(Into::into), arena_bytes_used)
    }
}

impl Drop for CaptureSession<'_> {
    fn drop(&mut self) {
        if self.capturing {
            // The closure left the stream capturing, so the region is
            // abandoned: close it and discard whatever was recorded. Leaving
            // it open would fail every later operation on this device.
            let _ = self.stream.arc().end_capture(INSTANTIATE_FLAGS);
        }
        if self.frozen {
            self.allocator.unfreeze();
        }
    }
}
