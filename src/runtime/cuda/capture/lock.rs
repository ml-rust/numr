//! Per-device exclusion between CUDA graph capture and ordinary enqueues.
//!
//! One compute stream per device is shared by every thread. A stream in
//! capture mode records everything enqueued on it into the graph under
//! construction, so an unrelated thread's launch or memcpy is swallowed by
//! that graph or faults. A per-device `RwLock` separates the two:
//!
//! - A capture holds the WRITE side for its whole region.
//! - Each enqueue on the compute stream holds the READ side for the duration
//!   of that one enqueue.
//!
//! Per-enqueue read scope is enough. A capture that begins between two
//! enqueues of one operation is safe: it only delays the second enqueue.
//!
//! # Lock order
//!
//! The capture lock is the outermost lock in this crate. A thread that holds
//! it may take the module cache, the allocator's free list, captured-pointer
//! set or arena, and the tuning cache or probe lock. None of those may be
//! held while acquiring the capture lock.
//!
//! # Re-entrancy
//!
//! The capturing thread must not block on its own read side. A thread-local
//! records the device it is capturing; while set, the read helper for that
//! device returns an empty permit. A capture closure therefore runs its
//! launches without acquiring anything.
//!
//! The thread-local is per thread, so a capture closure must not fan its work
//! onto other threads: those threads would block on the read side until the
//! capture ends.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

/// Capture lock for one device, shared by every clone of that device's client.
pub type DeviceCaptureLock = Arc<RwLock<()>>;

/// Device index -> capture lock. Consulted once, when a [`GuardedStream`] is
/// built; the enqueue path holds the `Arc` and performs no lookup.
///
/// [`GuardedStream`]: super::GuardedStream
static LOCKS: OnceLock<Mutex<HashMap<usize, DeviceCaptureLock>>> = OnceLock::new();

/// Lock the registry, recovering a poisoned mutex. The registry maps an index
/// to a lock and is never partially updated, so a panic elsewhere leaves it
/// usable.
#[inline]
fn lock_registry(
    registry: &Mutex<HashMap<usize, DeviceCaptureLock>>,
) -> MutexGuard<'_, HashMap<usize, DeviceCaptureLock>> {
    registry.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The capture lock for `device_index`, created on first request.
pub fn device_lock(device_index: usize) -> DeviceCaptureLock {
    let registry = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = lock_registry(registry);
    guard
        .entry(device_index)
        .or_insert_with(|| Arc::new(RwLock::new(())))
        .clone()
}

thread_local! {
    /// Device this thread is capturing on, and the capture nesting depth.
    /// Depth zero means this thread is not capturing.
    static CAPTURING: Cell<(usize, u32)> = const { Cell::new((usize::MAX, 0)) };
}

/// Whether THIS thread is inside a capture region on `device_index`.
///
/// This is not "the stream is in capture mode": a thread that is merely
/// waiting on a capture started elsewhere answers `false`. Work that is
/// illegal to record into a graph asks this, never the driver.
#[inline]
pub fn thread_is_capturing(device_index: usize) -> bool {
    CAPTURING.with(|c| {
        let (device, depth) = c.get();
        depth > 0 && device == device_index
    })
}

/// The device this thread is capturing on, or `None` when it is not capturing.
#[inline]
pub fn thread_capture_device() -> Option<usize> {
    CAPTURING.with(|c| {
        let (device, depth) = c.get();
        (depth > 0).then_some(device)
    })
}

/// Exclusive hold on a device's compute stream for the duration of a capture.
///
/// Acquiring marks the thread-local; dropping clears it and releases the
/// write side, including when the stack is unwinding.
pub struct CapturePermit<'a> {
    _write: RwLockWriteGuard<'a, ()>,
    device: usize,
}

impl<'a> CapturePermit<'a> {
    /// Take the write side of `lock` for `device_index`.
    ///
    /// A poisoned lock is recovered: the lock carries no state, and a thread
    /// that panicked mid-capture ran its own cleanup on unwind.
    ///
    /// # Errors
    ///
    /// Returns an error when this thread is already capturing. Re-entering
    /// capture would deadlock on the write side and is not a valid CUDA
    /// sequence in any case.
    pub fn acquire(lock: &'a RwLock<()>, device_index: usize) -> crate::error::Result<Self> {
        if let Some(active) = thread_capture_device() {
            return Err(crate::error::Error::Internal(format!(
                "CUDA graph capture is already in progress on this thread \
                 (device {active}); nested capture on device {device_index} \
                 is not supported"
            )));
        }
        let write = lock.write().unwrap_or_else(PoisonError::into_inner);
        CAPTURING.with(|c| c.set((device_index, 1)));
        Ok(Self {
            _write: write,
            device: device_index,
        })
    }

    /// Device this permit covers.
    #[inline]
    pub fn device_index(&self) -> usize {
        self.device
    }
}

impl Drop for CapturePermit<'_> {
    fn drop(&mut self) {
        CAPTURING.with(|c| c.set((usize::MAX, 0)));
    }
}

/// Hold on the read side for one enqueue, or nothing when the calling thread
/// is the one capturing this device.
pub struct EnqueuePermit<'a>(Option<RwLockReadGuard<'a, ()>>);

impl<'a> EnqueuePermit<'a> {
    /// Take the read side of `lock` unless this thread is capturing
    /// `device_index`.
    #[inline]
    pub fn acquire(lock: &'a RwLock<()>, device_index: usize) -> Self {
        if thread_is_capturing(device_index) {
            return Self(None);
        }
        Self(Some(lock.read().unwrap_or_else(PoisonError::into_inner)))
    }

    /// Whether this permit holds the read side.
    #[inline]
    pub fn holds_lock(&self) -> bool {
        self.0.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::thread;

    /// Device indices no other test in this crate touches.
    const DEV_A: usize = 8_201;
    const DEV_B: usize = 8_202;

    #[test]
    fn same_device_returns_same_lock() {
        let a = device_lock(DEV_A);
        let b = device_lock(DEV_A);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn different_devices_get_different_locks() {
        let a = device_lock(DEV_A);
        let b = device_lock(DEV_B);
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn capture_permit_marks_only_its_own_device() {
        let lock = device_lock(DEV_A);
        let permit = CapturePermit::acquire(&lock, DEV_A).expect("acquire");
        assert!(thread_is_capturing(DEV_A));
        assert!(!thread_is_capturing(DEV_B));
        assert_eq!(thread_capture_device(), Some(DEV_A));
        drop(permit);
        assert!(!thread_is_capturing(DEV_A));
        assert_eq!(thread_capture_device(), None);
    }

    #[test]
    fn nested_capture_on_one_thread_errors_instead_of_deadlocking() {
        let lock = device_lock(DEV_A);
        let outer = CapturePermit::acquire(&lock, DEV_A).expect("outer");
        let inner = CapturePermit::acquire(&lock, DEV_A);
        assert!(inner.is_err(), "nested capture must be refused");
        drop(outer);
    }

    /// The capturing thread launches through the read helper without blocking
    /// on the write side it already holds.
    #[test]
    fn capturing_thread_bypasses_its_own_read_side() {
        let lock = device_lock(DEV_A);
        let permit = CapturePermit::acquire(&lock, DEV_A).expect("acquire");
        for _ in 0..4 {
            let enqueue = EnqueuePermit::acquire(&lock, DEV_A);
            assert!(!enqueue.holds_lock(), "capturing thread took the read side");
        }
        drop(permit);
        let enqueue = EnqueuePermit::acquire(&lock, DEV_A);
        assert!(enqueue.holds_lock(), "read side not taken after capture");
    }

    /// A panic inside a capture region releases the permit on unwind, so the
    /// next capture proceeds.
    #[test]
    fn panic_releases_capture_permit() {
        let lock = device_lock(DEV_B);
        let panicked = std::panic::catch_unwind(|| {
            let lock = device_lock(DEV_B);
            let _permit = CapturePermit::acquire(&lock, DEV_B).expect("acquire");
            panic!("closure failed");
        });
        assert!(panicked.is_err());
        assert!(!thread_is_capturing(DEV_B), "thread-local survived unwind");
        let again = CapturePermit::acquire(&lock, DEV_B);
        assert!(again.is_ok(), "capture permit not released on unwind");
    }

    /// A capture on one thread and enqueues on another both complete: the
    /// enqueues wait, then run.
    #[test]
    fn concurrent_enqueue_waits_for_capture_then_runs() {
        let lock = device_lock(DEV_A);
        let done = Arc::new(AtomicU32::new(0));

        let permit = CapturePermit::acquire(&lock, DEV_A).expect("acquire");

        let worker_lock = Arc::clone(&lock);
        let worker_done = Arc::clone(&done);
        let worker = thread::spawn(move || {
            let _enqueue = EnqueuePermit::acquire(&worker_lock, DEV_A);
            worker_done.fetch_add(1, Ordering::SeqCst);
        });

        thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(
            done.load(Ordering::SeqCst),
            0,
            "an enqueue ran during capture"
        );

        drop(permit);
        worker.join().expect("worker");
        assert_eq!(done.load(Ordering::SeqCst), 1);
    }
}
