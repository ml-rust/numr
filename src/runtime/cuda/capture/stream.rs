//! Compute stream wrapper that pairs every kernel launch with the device's
//! capture lock.
//!
//! Kernel launchers receive a [`GuardedStream`], never a bare `CudaStream`.
//! The only way to build a launch from it is [`GuardedStream::launch_builder`],
//! whose [`GuardedLaunchBuilder::launch`] takes the read side of the capture
//! lock around the enqueue. A launcher that forgot the lock would have to name
//! `CudaStream`, which the type does not hand out except through
//! [`GuardedStream::raw`].

use std::sync::Arc;

use cudarc::driver::safe::{
    CudaEvent, CudaFunction, CudaStream, DriverError, LaunchArgs, LaunchConfig, PushKernelArg,
};

use super::super::context::bind_context;
use super::lock::{DeviceCaptureLock, EnqueuePermit, device_lock};

/// A device's compute stream together with the capture lock that guards it.
///
/// Clones share one stream and one lock, matching the single cached client per
/// device.
#[derive(Clone)]
pub struct GuardedStream {
    stream: Arc<CudaStream>,
    device_index: usize,
    lock: DeviceCaptureLock,
}

impl std::fmt::Debug for GuardedStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedStream")
            .field("device_index", &self.device_index)
            .finish_non_exhaustive()
    }
}

impl GuardedStream {
    /// Wrap `stream` as the compute stream of `device_index`.
    pub(crate) fn new(stream: Arc<CudaStream>, device_index: usize) -> Self {
        let lock = device_lock(device_index);
        Self {
            stream,
            device_index,
            lock,
        }
    }

    /// Start a kernel launch on this stream.
    ///
    /// Arguments are pushed host-side; the capture lock is taken inside
    /// [`GuardedLaunchBuilder::launch`], so a builder kept alive across other
    /// work holds nothing.
    #[inline]
    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> GuardedLaunchBuilder<'a> {
        GuardedLaunchBuilder {
            inner: self.stream.launch_builder(func),
            lock: &self.lock,
            device_index: self.device_index,
        }
    }

    /// Prepare the calling thread for one enqueue made through the driver API
    /// directly, such as a memcpy.
    ///
    /// Makes the stream's context current on the calling thread, then takes
    /// the read side of the capture lock. Any thread can therefore issue the
    /// driver call that follows, whether or not it created the client.
    ///
    /// The permit must cover the enqueue and nothing else. Holding it across
    /// a kernel launch on the same device would take the read side twice on
    /// one thread, which a waiting writer turns into a deadlock.
    ///
    /// # Errors
    ///
    /// Returns the driver's error when the context cannot be made current.
    #[inline]
    pub fn enqueue_permit(&self) -> Result<EnqueuePermit<'_>, DriverError> {
        bind_context(self.stream.context())?;
        Ok(EnqueuePermit::acquire(&self.lock, self.device_index))
    }

    /// Wait for every operation already submitted to this stream.
    ///
    /// Excluded against a concurrent capture: synchronizing a stream that
    /// another thread is capturing is rejected by the driver under the
    /// GLOBAL capture mode this crate uses, so the wait holds the read side
    /// of the device's capture lock for its duration.
    ///
    /// Do not call this while already holding a permit for the same device —
    /// a second read side on one thread deadlocks against a waiting writer.
    /// Under a permit, call `raw().synchronize()` instead.
    ///
    /// # Errors
    ///
    /// Returns the driver's error when the wait fails.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        let _permit = self.enqueue_permit()?;
        self.stream.synchronize()
    }

    /// The bare stream.
    ///
    /// Escape hatch for the few calls that cannot go through this type:
    /// capture begin and end, event record and wait, and driver calls made
    /// inside a scope that already holds a permit (the memcpy funnels, the
    /// allocator's driver paths).
    ///
    /// Anything that touches this stream's state — an enqueue, or a wait on
    /// it — must run under [`GuardedStream::enqueue_permit`], whether taken
    /// by the caller or by the wrapper it goes through. The permit also makes
    /// the stream's context current, which a direct driver call needs.
    #[inline]
    pub fn raw(&self) -> &CudaStream {
        &self.stream
    }

    /// The reference-counted stream, for handles that must own it.
    #[inline]
    pub fn arc(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Index of the device this stream belongs to.
    #[inline]
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// The device's capture lock, for the capture entry points.
    #[inline]
    pub(crate) fn capture_lock(&self) -> &DeviceCaptureLock {
        &self.lock
    }
}

/// A kernel launch in preparation on a [`GuardedStream`].
///
/// `arg` forwards to cudarc; `launch` holds the capture lock's read side for
/// the enqueue.
pub struct GuardedLaunchBuilder<'a> {
    inner: LaunchArgs<'a>,
    lock: &'a std::sync::RwLock<()>,
    device_index: usize,
}

// SAFETY: forwards to cudarc's own implementation for the same argument type;
// this adds no argument handling of its own.
unsafe impl<'a, T> PushKernelArg<T> for GuardedLaunchBuilder<'a>
where
    LaunchArgs<'a>: PushKernelArg<T>,
{
    #[inline(always)]
    fn arg(&mut self, arg: T) -> &mut Self {
        self.inner.arg(arg);
        self
    }
}

impl GuardedLaunchBuilder<'_> {
    /// Enqueue the kernel, holding the device's capture lock until the driver
    /// call returns.
    ///
    /// # Safety
    ///
    /// Same contract as cudarc's `LaunchArgs::launch`: the pushed arguments
    /// must match the kernel's signature and every device pointer must be
    /// valid for the launch.
    #[inline]
    pub unsafe fn launch(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        let _permit = EnqueuePermit::acquire(self.lock, self.device_index);
        unsafe { self.inner.launch(cfg) }
    }
}
