//! A `CudaClient` works from any thread, not only the one that created it.
//!
//! Every worker here is a freshly spawned thread that has made no CUDA call
//! before it touches the client, so no context is current on it. Each worker
//! uploads, computes, reads back and frees through numr alone.
//!
//! Run with:
//!   cd numr && cargo test --release --features cuda --test cuda_cross_thread

#![cfg(feature = "cuda")]

use std::thread;

use numr::ops::{BinaryOps, MatmulOps};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Device 0 and its client, created on the calling thread, or `None` when the
/// machine has no CUDA device.
fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

/// Upload, add, matmul, read back and free on the calling thread.
///
/// `seed` varies the inputs per worker so a result read from another
/// worker's buffer does not pass. Returns the stage that failed and its error.
fn round_trip(client: &CudaClient, device: &CudaDevice, seed: f32) -> Result<(), String> {
    let a_host = [seed, seed + 1.0, seed + 2.0, seed + 3.0];
    let b_host = [10.0f32, 20.0, 30.0, 40.0];

    let a = Tensor::<CudaRuntime>::from_slice(&a_host, &[2, 2], device)
        .map_err(|e| format!("from_slice a: {e}"))?;
    let b = Tensor::<CudaRuntime>::from_slice(&b_host, &[2, 2], device)
        .map_err(|e| format!("from_slice b: {e}"))?;

    let sum = client.add(&a, &b).map_err(|e| format!("add: {e}"))?;
    let got: Vec<f32> = sum.try_to_vec().map_err(|e| format!("to_vec add: {e}"))?;
    let want: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();
    if got != want {
        return Err(format!("add: got {got:?}, want {want:?}"));
    }

    let prod = client.matmul(&a, &b).map_err(|e| format!("matmul: {e}"))?;
    let got: Vec<f32> = prod
        .try_to_vec()
        .map_err(|e| format!("to_vec matmul: {e}"))?;
    let want = vec![
        a_host[0] * b_host[0] + a_host[1] * b_host[2],
        a_host[0] * b_host[1] + a_host[1] * b_host[3],
        a_host[2] * b_host[0] + a_host[3] * b_host[2],
        a_host[2] * b_host[1] + a_host[3] * b_host[3],
    ];
    if got != want {
        return Err(format!("matmul: got {got:?}, want {want:?}"));
    }

    drop(sum);
    drop(prod);
    drop(a);
    drop(b);

    // A fresh allocation after the frees still reads back what was written.
    let again = Tensor::<CudaRuntime>::from_slice(&a_host, &[4], device)
        .map_err(|e| format!("from_slice after free: {e}"))?;
    let got: Vec<f32> = again
        .try_to_vec()
        .map_err(|e| format!("to_vec after free: {e}"))?;
    if got != a_host {
        return Err(format!("after free: got {got:?}, want {a_host:?}"));
    }
    Ok(())
}

/// One fresh thread uses a client created on another thread.
#[test]
fn fresh_thread_uses_client_created_elsewhere() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let worker = thread::spawn(move || round_trip(&client, &device, 1.0));
    let outcome = worker.join().expect("worker thread panicked");
    if let Err(e) = outcome {
        panic!("fresh thread failed: {e}");
    }
}

/// Run `f` on a fresh thread and record its error under `what`.
fn on_fresh_thread<F>(errors: &mut Vec<String>, what: &str, f: F)
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    match thread::spawn(f).join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => errors.push(format!("{what}: {e}")),
        Err(_) => errors.push(format!("{what}: worker thread panicked")),
    }
}

/// Entry points whose first driver call takes no stream: the driver can
/// only resolve their context from the calling thread. Each runs as the
/// first CUDA work of its own fresh thread.
#[test]
fn stream_less_entry_points_work_on_fresh_threads() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let host = [1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::<CudaRuntime>::from_slice(&host, &[4], &device).expect("input");
    let mut errors = Vec::new();

    let (c, d, tt) = (client.clone(), device.clone(), t.clone());
    on_fresh_thread(&mut errors, "pipelined readback", move || {
        let event = tt
            .record_event()
            .map_err(|e| format!("record_event: {e}"))?;
        let got: Vec<f32> = tt
            .to_vec_pipelined(event)
            .map_err(|e| format!("to_vec_pipelined: {e}"))?;
        drop((c, d));
        if got != host {
            return Err(format!("got {got:?}, want {host:?}"));
        }
        Ok(())
    });

    let d = device.clone();
    on_fresh_thread(&mut errors, "memory_info", move || {
        let (free, total) = d.memory_info().map_err(|e| e.to_string())?;
        if total == 0 || free > total {
            return Err(format!("free {free}, total {total}"));
        }
        Ok(())
    });

    let d = device.clone();
    on_fresh_thread(&mut errors, "device sync", move || {
        d.sync().map_err(|e| e.to_string())
    });

    let (c, tt) = (client.clone(), t.clone());
    on_fresh_thread(
        &mut errors,
        "event record and copy-stream wait",
        move || {
            let event = c
                .record_event_on_compute()
                .map_err(|e| format!("record_event_on_compute: {e}"))?;
            let waited = c
                .copy_stream_wait_event(event)
                .map_err(|e| format!("copy_stream_wait_event: {e}"));
            c.destroy_event(event);
            drop(tt);
            waited
        },
    );

    let (c, tt) = (client.clone(), t.clone());
    on_fresh_thread(&mut errors, "histogram", move || {
        use numr::ops::StatisticalOps;
        let (hist, _edges) = c
            .histogram(&tt, 4, None)
            .map_err(|e| format!("histogram: {e}"))?;
        let got: Vec<i64> = hist.try_to_vec().map_err(|e| format!("to_vec: {e}"))?;
        if got != [1, 1, 1, 1] {
            return Err(format!("got {got:?}, want [1, 1, 1, 1]"));
        }
        Ok(())
    });

    assert!(
        errors.is_empty(),
        "fresh-thread entry points failed: {errors:#?}"
    );
}

/// Several fresh threads share one client concurrently.
#[test]
fn several_threads_share_one_client_concurrently() {
    const THREADS: usize = 6;
    const ROUNDS: usize = 8;

    let Some((client, device)) = cuda() else {
        return;
    };
    let workers: Vec<_> = (0..THREADS)
        .map(|t| {
            let client = client.clone();
            let device = device.clone();
            thread::spawn(move || -> Result<(), String> {
                for r in 0..ROUNDS {
                    let seed = (t * ROUNDS + r) as f32;
                    round_trip(&client, &device, seed)
                        .map_err(|e| format!("thread {t} round {r}: {e}"))?;
                }
                Ok(())
            })
        })
        .collect();

    let errors: Vec<String> = workers
        .into_iter()
        .filter_map(|w| match w.join() {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(e),
            Err(_) => Some("worker thread panicked".to_string()),
        })
        .collect();
    assert!(
        errors.is_empty(),
        "shared-client workers failed: {errors:?}"
    );
}

/// A thread whose current context belongs to device 1 still uses device 0's
/// client correctly. Skips on a machine with fewer than two devices.
#[test]
fn thread_bound_to_other_device_uses_device_zero_client() {
    if CudaDevice::count().unwrap_or(0) < 2 {
        return;
    }
    let Some((client, device)) = cuda() else {
        return;
    };
    let worker = thread::spawn(move || -> Result<(), String> {
        // Device 1 work first leaves device 1's context current here.
        let other = CudaDevice::new(1);
        let other_client =
            CudaClient::new(other.clone()).map_err(|e| format!("device 1 client: {e}"))?;
        round_trip(&other_client, &other, 3.0).map_err(|e| format!("device 1: {e}"))?;
        round_trip(&client, &device, 5.0).map_err(|e| format!("device 0 after 1: {e}"))?;
        round_trip(&other_client, &other, 7.0).map_err(|e| format!("device 1 after 0: {e}"))
    });
    let outcome = worker.join().expect("worker thread panicked");
    if let Err(e) = outcome {
        panic!("mixed-device thread failed: {e}");
    }
}
