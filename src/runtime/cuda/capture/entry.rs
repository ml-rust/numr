//! The two CUDA graph capture entry points.
//!
//! Both hold the device's capture lock for the whole region, so no other
//! thread can enqueue on the shared compute stream while the graph is being
//! recorded, and both release every piece of capture state through
//! [`CaptureSession`] even when the closure panics.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::CapturedGraph;
use crate::runtime::cuda::client::CudaClient;
use crate::runtime::cuda::graph::CudaGraph;
use crate::runtime::cuda::runtime::CudaRuntime;
use crate::tensor::Tensor;

use super::session::CaptureSession;

/// Capture the work `f` enqueues into a replayable graph.
///
/// Intermediate tensors allocated inside `f` become graph-managed memory and
/// are freed on each replay. Tensors in `inputs` and `outputs` are allocated
/// by the caller before capture, so their addresses stay valid; clones of them
/// are held by the returned graph.
///
/// # Threading
///
/// `f` runs on the calling thread and must enqueue all of its work there.
/// Work fanned onto another thread would block on the capture lock until the
/// region closes, and would not be recorded.
///
/// # Errors
///
/// Returns an error when this thread is already capturing, when the driver
/// rejects capture begin or end, when `f` fails, or when `f` recorded no work.
pub(crate) fn capture_graph_into<F>(
    client: &CudaClient,
    inputs: &[&Tensor<CudaRuntime>],
    outputs: &[&Tensor<CudaRuntime>],
    f: F,
) -> Result<CapturedGraph<CudaRuntime>>
where
    F: FnOnce(&CudaClient) -> Result<()>,
{
    // Cheap Arc bumps that keep the I/O buffers alive for the graph's life.
    let owned_inputs: Vec<Tensor<CudaRuntime>> = inputs.iter().map(|t| (*t).clone()).collect();
    let owned_outputs: Vec<Tensor<CudaRuntime>> = outputs.iter().map(|t| (*t).clone()).collect();

    let session = CaptureSession::begin(&client.stream, &client.allocator, || Ok(()))?;
    let closure_result = f(client);
    let (graph_result, _) = session.end();

    closure_result?;
    let cudarc_graph = graph_result?.ok_or_else(|| {
        Error::Backend(
            "CUDA graph capture produced no operations — closure recorded nothing".into(),
        )
    })?;

    Ok(CapturedGraph::new(
        CudaGraph::new(cudarc_graph),
        owned_inputs,
        owned_outputs,
    ))
}

/// Capture `f` with a bump-pointer arena serving every intermediate
/// allocation made inside the region.
///
/// # Why the arena exists
///
/// Memory allocated inside a capture region is graph-managed and freed on
/// each replay. The first replay frees those addresses; the second replay's
/// kernel nodes then dereference freed memory and the driver reports an
/// illegal address. The arena is allocated before capture begins, so it is
/// not graph-managed and its address is stable for the graph's whole life.
///
/// `arena_bytes` must cover every intermediate `f` creates. When it does not,
/// `f` fails with a `Backend` error naming the requested bytes, the bytes
/// used and `arena_bytes`, and no graph is produced. The returned graph
/// reports its peak arena footprint through
/// [`CapturedGraph::arena_bytes_used`], so a caller can recapture tighter.
///
/// # Drop ordering
///
/// `CapturedGraph` declares `arena` after `graph`, `inputs` and `outputs`, so
/// the arena buffer outlives the compiled graph handle.
///
/// # Errors
///
/// Returns an error when the arena allocation fails, when an arena is already
/// installed on this client, when this thread is already capturing, when the
/// driver rejects capture begin or end, or when `f` fails.
pub(crate) fn capture_graph_into_with_arena<F>(
    client: &CudaClient,
    inputs: &[&Tensor<CudaRuntime>],
    outputs: &[&Tensor<CudaRuntime>],
    arena_bytes: usize,
    f: F,
) -> Result<CapturedGraph<CudaRuntime>>
where
    F: FnOnce(&CudaClient) -> Result<()>,
{
    // F32 storage is a neutral element type; the arena is handed out as raw
    // bytes by the bump-pointer logic.
    let arena_elems = arena_bytes.div_ceil(std::mem::size_of::<f32>());
    let arena_tensor = Tensor::<CudaRuntime>::empty(&[arena_elems], DType::F32, &client.device)
        .map_err(|e| {
            Error::Backend(format!(
                "capture_graph_into_with_arena: arena allocation failed \
                 ({arena_bytes} bytes): {e}"
            ))
        })?;
    let arena_ptr = arena_tensor.ptr();

    let owned_inputs: Vec<Tensor<CudaRuntime>> = inputs.iter().map(|t| (*t).clone()).collect();
    let owned_outputs: Vec<Tensor<CudaRuntime>> = outputs.iter().map(|t| (*t).clone()).collect();

    let session = CaptureSession::begin(&client.stream, &client.allocator, || {
        client.allocator.install_arena(arena_ptr, arena_bytes)
    })?;
    let closure_result = f(client);
    let (graph_result, arena_bytes_used) = session.end();

    closure_result?;
    let cudarc_graph = graph_result?.ok_or_else(|| {
        Error::Backend(
            "CUDA graph capture (with_arena) produced no operations — \
             closure recorded nothing"
                .into(),
        )
    })?;

    Ok(CapturedGraph::new_with_arena(
        CudaGraph::new(cudarc_graph),
        owned_inputs,
        owned_outputs,
        arena_tensor,
        arena_bytes_used.unwrap_or(0),
    ))
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    use crate::dtype::DType;
    use crate::ops::BinaryOps;
    use crate::runtime::Runtime;
    use crate::runtime::common::Allocator;
    use crate::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, is_cuda_available};
    use crate::runtime::traits::client::RuntimeClient;
    use crate::tensor::Tensor;

    /// The device every test here uses, or `None` when the machine has no
    /// CUDA device. A test that gets `None` returns without asserting.
    fn device_and_client() -> Option<(CudaDevice, CudaClient)> {
        if !is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((device, client))
    }

    /// A capture on one thread and ordinary work on another both finish with
    /// correct results. Without the capture lock the second thread's launches
    /// are recorded into the first thread's graph or fault.
    #[test]
    fn capture_and_concurrent_work_both_produce_correct_results() {
        let Some((device, client)) = device_and_client() else {
            return;
        };

        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_device = device.clone();
        let worker_client = client.clone();
        let worker = thread::spawn(move || -> bool {
            // This thread did not create the client and binds nothing itself:
            // numr makes the context current for it.
            let x =
                Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &worker_device)
                    .expect("worker input x");
            let y = Tensor::<CudaRuntime>::from_slice(
                &[10.0f32, 20.0, 30.0, 40.0],
                &[4],
                &worker_device,
            )
            .expect("worker input y");
            let mut all_correct = true;
            while !worker_stop.load(Ordering::Relaxed) {
                let z = worker_client.add(&x, &y).expect("worker add");
                worker_client.synchronize();
                all_correct &= z.to_vec::<f32>() == vec![11.0, 22.0, 33.0, 44.0];
            }
            all_correct
        });

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)
            .expect("capture input a");
        let b = Tensor::<CudaRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device)
            .expect("capture input b");
        let c = Tensor::<CudaRuntime>::zeros(&[4], DType::F32, &device).expect("capture output c");

        for _ in 0..8 {
            let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
                // Inside the region this thread reports itself as capturing,
                // which is what lets its own launches skip the read side.
                assert!(cc.is_capturing(), "capturing thread reports otherwise");
                cc.add_into(&c, &a, &b)
            })
            .expect("capture_graph_into");
            captured.launch().expect("graph launch");
            client.synchronize();
            assert_eq!(c.to_vec::<f32>(), vec![11.0, 22.0, 33.0, 44.0]);
        }

        stop.store(true, Ordering::Relaxed);
        assert!(worker.join().expect("worker thread"), "worker got bad data");
    }

    /// A panic inside the capture closure unwinds past capture end and the
    /// allocator restore. Both must still happen, or the device is unusable.
    #[test]
    fn panic_in_capture_closure_leaves_device_usable() {
        let Some((device, client)) = device_and_client() else {
            return;
        };

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)
            .expect("input a");
        let b = Tensor::<CudaRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device)
            .expect("input b");
        let c = Tensor::<CudaRuntime>::zeros(&[4], DType::F32, &device).expect("output c");

        let unwound = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
                cc.add_into(&c, &a, &b)?;
                panic!("capture closure failed");
            });
        }));
        assert!(unwound.is_err(), "the closure's panic did not propagate");

        assert!(!client.allocator.is_frozen(), "allocator left frozen");
        assert!(!client.allocator.has_arena(), "arena left installed");
        assert!(
            !client.is_capturing(),
            "thread still marked as capturing after unwind"
        );

        // The stream is usable: a fresh capture opens, records and replays.
        let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
            cc.add_into(&c, &a, &b)
        })
        .expect("capture after panic");
        captured.launch().expect("graph launch after panic");
        client.synchronize();
        assert_eq!(c.to_vec::<f32>(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// A wait on the compute stream from another thread blocks for the whole
    /// capture and then succeeds. Synchronizing a stream that another thread
    /// is capturing is rejected by the driver, so the wait must queue behind
    /// the capture rather than run during it.
    #[test]
    fn concurrent_synchronize_waits_for_capture() {
        let Some((device, client)) = device_and_client() else {
            return;
        };

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)
            .expect("input a");
        let b = Tensor::<CudaRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device)
            .expect("input b");
        let c = Tensor::<CudaRuntime>::zeros(&[4], DType::F32, &device).expect("output c");

        let waiting = Arc::new(AtomicBool::new(false));
        let synced = Arc::new(AtomicBool::new(false));

        let worker_client = client.clone();
        let worker_waiting = Arc::clone(&waiting);
        let worker_synced = Arc::clone(&synced);
        let worker = thread::spawn(move || {
            worker_waiting.store(true, Ordering::SeqCst);
            worker_client.synchronize();
            worker_synced.store(true, Ordering::SeqCst);
        });

        let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
            cc.add_into(&c, &a, &b)?;
            while !waiting.load(Ordering::SeqCst) {
                thread::yield_now();
            }
            // Give the worker time to reach the lock, then confirm its wait
            // has not run inside this region.
            thread::sleep(std::time::Duration::from_millis(100));
            assert!(
                !synced.load(Ordering::SeqCst),
                "a wait on the compute stream ran during capture"
            );
            Ok(())
        })
        .expect("capture_graph_into");

        worker.join().expect("worker thread");
        assert!(synced.load(Ordering::SeqCst), "the wait never completed");

        captured.launch().expect("graph launch");
        client.synchronize();
        assert_eq!(c.to_vec::<f32>(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// A capture closure that enqueues many kernels completes: the thread-local
    /// bypass keeps it off the read side of the lock it already holds for
    /// writing. Without the bypass the first launch would deadlock.
    #[test]
    fn capture_closure_launches_without_deadlocking() {
        let Some((device, client)) = device_and_client() else {
            return;
        };

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)
            .expect("input a");
        let b = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device)
            .expect("input b");
        let c = Tensor::<CudaRuntime>::zeros(&[4], DType::F32, &device).expect("output c");

        let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
            cc.add_into(&c, &a, &b)?;
            for _ in 0..16 {
                cc.add_into(&c, &c, &b)?;
            }
            Ok(())
        })
        .expect("capture with many launches");

        captured.launch().expect("graph launch");
        client.synchronize();
        assert_eq!(c.to_vec::<f32>(), vec![18.0, 19.0, 20.0, 21.0]);
    }
}
