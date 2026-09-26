use std::collections::HashSet;

use super::core::CudaAllocator;
use super::free_list::FREE_LIST_CAP;
use crate::runtime::Allocator;

impl Allocator for CudaAllocator {
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // Graph-capture path: bypass the Rust free list.
        //
        // When an arena is installed: redirect allocations into the
        // bump-pointer arena so that the device addresses baked into graph
        // kernel-parameter blocks are stable across replays.
        //
        // When no arena is installed: fall through to driver_alloc, which
        // creates a proper graph alloc-node (existing behaviour).
        //
        // Either way, record the pointer in `captured_ptrs` so `unfreeze()`
        // can assert it never migrated into the free list.
        if self.thread_owns_capture() {
            let ptr = {
                let mut arena_guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(ref mut arena) = *arena_guard {
                    arena.allocate(size_bytes)?
                } else {
                    // No arena: fall through to driver.
                    drop(arena_guard);
                    unsafe { self.driver_alloc(size_bytes) }?
                }
            };
            self.captured_ptrs
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(ptr);
            return Ok(ptr);
        }

        // Fast path: pop from the free list if a cached buffer exists.
        {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(ptr) = fl.pop(size_bytes as u64) {
                return Ok(ptr);
            }
        }

        // Cold miss: ask the driver.
        unsafe { self.driver_alloc(size_bytes) }
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // Graph-capture path: handle differently depending on whether the
        // pointer came from the arena or from the driver.
        if self.thread_owns_capture() {
            let mut arena_guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(ref mut arena) = *arena_guard {
                // Pointer came from the arena: record the logical free
                // (bookkeeping only — the arena is strictly monotone and never
                // rewinds high_water). Do NOT call cuMemFreeAsync; the arena
                // buffer is not owned by the driver pool.
                arena.deallocate(ptr);
            } else {
                // No arena: pointer came from driver_alloc; issue a driver
                // free node so the graph records the deallocation.
                drop(arena_guard);
                unsafe { self.driver_free(ptr) };
            }
            return;
        }

        let mut evict: Vec<u64> = Vec::new();
        {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            let size = size_bytes as u64;
            match fl.map.get_mut(&size) {
                // Per-bucket cap exceeded: evict the oldest of this size and
                // keep the new one (byte total unchanged — same-size swap).
                Some(bucket) if bucket.len() >= FREE_LIST_CAP => {
                    if let Some(old) = bucket.pop_front() {
                        evict.push(old);
                    }
                    bucket.push_back(ptr);
                }
                _ => fl.push(size, ptr),
            }
            // Global byte cap: return oldest-largest cached buffers to the
            // driver until total cached bytes are back under the cap. Bounds
            // retention on many-shape workloads that spawn unbounded size
            // buckets.
            evict.extend(fl.evict_to_cap(self.free_list_cap_bytes));
        }

        // Free outside the lock so the lock is not held during driver calls.
        for old_ptr in evict {
            unsafe { self.driver_free(old_ptr) };
        }
    }

    fn is_frozen(&self) -> bool {
        self.frozen.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn freeze(&self) -> bool {
        self.frozen
            .store(true, std::sync::atomic::Ordering::Relaxed);
        true
    }

    fn unfreeze(&self) {
        self.frozen
            .store(false, std::sync::atomic::Ordering::Relaxed);

        // Clear the arena bookkeeping.  The device buffer itself is NOT freed
        // here — it is owned by the Tensor<CudaRuntime> held in
        // CapturedGraph::arena, and will be freed when CapturedGraph is dropped
        // (after the CudaGraph handle is destroyed).
        self.clear_arena();

        // ---- Graph-capture bookkeeping (defense-in-depth) ----
        //
        // Drain the captured_ptrs set and verify that none of those addresses
        // are present in the free list. If a match is found it means a pointer
        // that was allocated during the freeze window was later passed to the
        // *un-frozen* `deallocate()` path, which cached it in `free_list`. On
        // the next allocation the Rust cache would hand that address back to a
        // non-graph caller while the CUDA graph still owns it — silent
        // graph-state corruption. Panic early so the bug is caught immediately.
        let captured: HashSet<u64> = {
            let mut set = self.captured_ptrs.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::take(&mut *set)
        };

        if captured.is_empty() {
            return;
        }

        // Non-empty set means the closure did not explicitly free all of its
        // graph-internal scratch. With AUTO_FREE_ON_LAUNCH the driver will
        // handle the frees on next launch — this is expected for intermediate
        // buffers that live until graph replay. Log once so it is visible
        // without being fatal.
        #[cfg(debug_assertions)]
        eprintln!(
            "[numr::cuda] unfreeze: {} pointer(s) from the freeze window were \
             still live at unfreeze (will be freed by the driver on next graph \
             launch — expected for graph-internal scratch).",
            captured.len()
        );

        // CRITICAL assertion: none of the captured pointers must appear in
        // the free list. Finding one there means the Rust cache absorbed a
        // graph-internal address — a definite bug.
        #[cfg(debug_assertions)]
        {
            let fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            for bucket in fl.map.values() {
                for &cached_ptr in bucket {
                    debug_assert!(
                        !captured.contains(&cached_ptr),
                        "[numr::cuda] GRAPH CORRUPTION DETECTED: pointer 0x{:x} was \
                         allocated during a freeze window (graph-capture) but was \
                         subsequently absorbed into the Rust free list via the \
                         un-frozen deallocate() path. On next allocation this \
                         address would be handed to a non-graph caller while the \
                         CUDA graph still holds a reference to it.",
                        cached_ptr
                    );
                }
            }
        }
    }

    fn reset(&self) -> crate::error::Result<()> {
        // Drain the free list and return all cached buffers to the driver pool.
        // Callers must have dropped all live tensors (which call `deallocate`)
        // before calling `reset`, so every pointer here is idle on the stream.
        let drained: Vec<u64> = {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            fl.total_bytes = 0;
            fl.map
                .drain()
                .flat_map(|(_, bucket)| bucket.into_iter())
                .collect()
        };
        for ptr in drained {
            unsafe { self.driver_free(ptr) };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::Allocator;
    use crate::runtime::cuda::capture::CapturePermit;
    use crate::runtime::cuda::client::CudaClient;
    use crate::runtime::cuda::device::CudaDevice;

    /// Verify that `unfreeze()` fires the corruption assertion when a pointer
    /// that was allocated during a freeze window is found in the free list.
    ///
    /// A freeze-allocated pointer is injected directly into the free list,
    /// standing in for an op that routes a graph-internal address through the
    /// un-frozen deallocate path. `unfreeze()` must detect the overlap and
    /// panic.
    ///
    /// `#[ignore]` because `CudaAllocator::driver_alloc` needs a live CUDA
    /// context and GPU. Run with:
    ///
    /// ```text
    /// cargo test --features cuda captured_ptrs_unfreeze_detects_corruption -- --ignored
    /// ```
    ///
    /// Without a GPU, read the invariant off the code: the debug_assert in
    /// `unfreeze()` iterates `free_list` and panics on any pointer in
    /// `captured_ptrs`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn captured_ptrs_unfreeze_detects_corruption() {
        // --- Setup: build a real CudaAllocator via CudaClient::new_uncached ---
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        // Allocate two pointers normally (pre-freeze, goes into free list on drop).
        let _p1 = alloc.allocate(256).expect("alloc p1");
        let _p2 = alloc.allocate(512).expect("alloc p2");

        // Transition into freeze mode, standing in for the start of a CUDA
        // graph capture. The permit marks THIS thread as the capturing one,
        // half of what puts `allocate` on the graph path. See
        // [`CudaAllocator::thread_owns_capture`].
        let _permit = CapturePermit::acquire(alloc.stream.capture_lock(), 0).expect("permit");
        alloc.freeze();
        assert!(alloc.is_frozen(), "allocator should be frozen");

        // Allocate a third pointer — this goes directly to the driver and is
        // recorded in `captured_ptrs`.
        let p3 = alloc.allocate(128).expect("alloc p3 during freeze");
        assert_ne!(p3, 0, "frozen alloc must return a non-null pointer");

        // Verify p3 is tracked in captured_ptrs.
        {
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(
                set.contains(&p3),
                "p3 must be present in captured_ptrs after frozen allocate"
            );
        }

        // --- Inject the bug: manually push p3 into the free list ---
        //
        // This simulates a future op author calling `deallocate(p3, 128)` on
        // an un-frozen allocator (e.g., because they cached the raw pointer and
        // called deallocate after the freeze window closed).
        {
            let mut fl = alloc.free_list.lock().unwrap();
            fl.push(128, p3);
        }

        // `unfreeze()` must detect the overlap and panic (debug builds only).
        // In release builds the assert compiles away so we skip the
        // should_panic check to avoid a false-pass.
        #[cfg(debug_assertions)]
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                alloc.unfreeze();
            }));
            assert!(
                result.is_err(),
                "unfreeze() must panic when a captured pointer leaks into the free list"
            );
        }

        // In release builds: just call unfreeze and verify the set is cleared
        // (no assertion fires, but bookkeeping must still be coherent).
        #[cfg(not(debug_assertions))]
        {
            alloc.unfreeze();
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(
                set.is_empty(),
                "captured_ptrs must be empty after unfreeze (release build)"
            );
        }
    }

    /// Verify that `captured_ptrs` is properly cleared after a clean freeze
    /// window where the closure frees everything it allocated.
    ///
    /// Allocate during freeze, free during freeze (driver_free, which does NOT
    /// touch free_list), then unfreeze. `captured_ptrs` is non-empty at
    /// unfreeze time, which is typical for graph-internal scratch the CUDA
    /// runtime owns, but it never overlaps free_list, so nothing panics.
    ///
    /// `#[ignore]`: needs a live GPU.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn captured_ptrs_cleared_after_clean_unfreeze() {
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        let _permit = CapturePermit::acquire(alloc.stream.capture_lock(), 0).expect("permit");
        alloc.freeze();
        let p = alloc.allocate(64).expect("alloc during freeze");
        assert_ne!(p, 0);

        // captured_ptrs must contain p.
        {
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(set.contains(&p), "p must be in captured_ptrs");
        }

        // Unfreeze without injecting p into the free list — clean path.
        // Should NOT panic.
        alloc.unfreeze();

        // captured_ptrs must be cleared after unfreeze.
        {
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(set.is_empty(), "captured_ptrs must be empty after unfreeze");
        }

        // frozen flag must be cleared.
        assert!(!alloc.is_frozen(), "allocator must be unfrozen");
    }

    /// A thread that is NOT capturing allocates off the ordinary path even
    /// while another thread holds a freeze window open.
    ///
    /// One client per device means one allocator shared by every thread. A
    /// second render served from the CAPTURING thread's arena gives two
    /// threads one device address, and its later free lands in the ordinary
    /// free list, which `unfreeze` reports as corruption.
    ///
    /// The freeze flag stays on for the whole test. What decides the path is
    /// whether this thread owns the capture.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn a_non_capturing_thread_is_not_served_from_the_capture_arena() {
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        alloc.freeze();
        assert!(alloc.is_frozen(), "the window is open");

        // No permit was taken, so this thread does not own the capture.
        let p = alloc.allocate(256).expect("alloc beside a capture");
        assert_ne!(p, 0);
        assert!(
            !alloc
                .captured_ptrs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&p),
            "a pointer from a non-capturing thread is not graph-owned"
        );

        // It frees back to the ordinary free list, the matching path.
        alloc.deallocate(p, 256);
        alloc.unfreeze();
    }

    /// A panic while the free-list mutex is held must not disable the allocator.
    ///
    /// One poisoned lock must not turn every later allocation into a panic.
    /// Recovery is sound because the free list is a cache: the worst a torn
    /// mutation leaves behind is a `total_bytes` that disagrees with the
    /// bucket contents.
    ///
    /// Sabotage check: `.unwrap()` the `free_list` locks and this test fails
    /// with `called `Result::unwrap()` on an `Err` value: PoisonError { .. }`.
    ///
    /// `#[ignore]` for the same reason as the tests above — `driver_alloc`
    /// needs a live CUDA context.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn allocation_survives_a_poisoned_free_list() {
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        // Put a buffer in the free list so the poisoned state has real content.
        let p = alloc.allocate(1024).expect("alloc before poisoning");
        alloc.deallocate(p, 1024);

        // Poison the mutex: a thread panics while holding the guard.
        let poisoner = alloc.clone();
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.free_list.lock().expect("lock is not yet poisoned");
            panic!("deliberate panic while holding the free-list lock");
        });
        assert!(handle.join().is_err(), "the poisoning thread must panic");
        assert!(
            alloc.free_list.is_poisoned(),
            "the free-list mutex must be poisoned"
        );

        // Every free-list path must still work.
        let p2 = alloc.allocate(1024).expect("alloc after poisoning");
        assert_ne!(p2, 0, "allocation must return a real pointer");
        alloc.deallocate(p2, 1024);
        alloc.reset().expect("reset after poisoning");
    }
}
