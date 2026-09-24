//! `capture_graph_into_with_arena` reports the peak arena footprint of the
//! recorded closure through `CapturedGraph::arena_bytes_used`. Own binary: a
//! global-mode capture must not share a process with tests that record
//! events or synchronize on other streams.
//!
//! Run with:
//!   cd numr && cargo test --release --features cuda --test cuda_arena_capture

#![cfg(feature = "cuda")]

use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const ARENA_BYTES: usize = 1 << 20;

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

/// One intermediate `add` allocates inside the arena; `add_into` writes the
/// stable output. The reported footprint is positive and within the arena,
/// and the replayed graph computes `(a + b) + a`.
#[test]
fn arena_capture_reports_bytes_used() {
    use numr::ops::BinaryOps as _;

    let Some((client, device)) = cuda() else {
        return;
    };
    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; 64], &[64], &device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; 64], &[64], &device).unwrap();
    let c = Tensor::<CudaRuntime>::from_slice(&vec![0.0f32; 64], &[64], &device).unwrap();

    let captured =
        CudaRuntime::capture_graph_into_with_arena(&client, &[&a, &b], &[&c], ARENA_BYTES, |cc| {
            let sum = cc.add(&a, &b)?;
            cc.add_into(&c, &sum, &a)
        })
        .unwrap();
    assert!(!client.is_capturing());

    let used = captured
        .arena_bytes_used()
        .expect("arena capture reports usage");
    assert!(used > 0, "the intermediate add must land in the arena");
    assert!(
        used <= ARENA_BYTES,
        "used {used} exceeds arena {ARENA_BYTES}"
    );

    captured.launch().unwrap();
    captured.launch().unwrap();
    let out = c.to_vec::<f32>();
    assert!(out.iter().all(|&v| v == 4.0), "expected 4.0, got {out:?}");
    drop(captured);

    plain_capture_reports_no_arena(&client, &device);
    tiny_arena_reports_typed_exhaustion(&client, &device);
}

/// A capture driven with an arena too small for the closure's first
/// intermediate allocation fails with the typed `Error::ArenaExhausted`
/// variant, carrying the real requested/used/capacity numbers — not only a
/// formatted string a caller would have to parse. Lives in the same test as
/// the others: two global-mode captures in one process cannot overlap, and
/// the test harness runs test functions in parallel.
fn tiny_arena_reports_typed_exhaustion(client: &CudaClient, device: &CudaDevice) {
    use numr::error::Error;
    use numr::ops::BinaryOps as _;

    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; 64], &[64], device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; 64], &[64], device).unwrap();
    let c = Tensor::<CudaRuntime>::from_slice(&vec![0.0f32; 64], &[64], device).unwrap();

    // 64 f32 = 256 bytes, aligned up to 256; a one-byte arena cannot hold it.
    const TINY_ARENA_BYTES: usize = 1;
    let err = CudaRuntime::capture_graph_into_with_arena(
        client,
        &[&a, &b],
        &[&c],
        TINY_ARENA_BYTES,
        |cc| {
            let sum = cc.add(&a, &b)?;
            cc.add_into(&c, &sum, &a)
        },
    )
    .expect_err("a 1-byte arena cannot hold a 256-byte intermediate");
    assert!(!client.is_capturing());

    let (requested, used, capacity) = err
        .as_arena_exhausted()
        .unwrap_or_else(|| panic!("expected arena exhaustion, got {err}"));
    assert!(requested > 0, "requested must name the failing allocation");
    assert_eq!(used, 0, "nothing was allocated before the first request");
    assert_eq!(
        capacity, TINY_ARENA_BYTES,
        "capacity must be the arena size"
    );
    assert!(
        matches!(
            err,
            Error::ArenaExhausted { .. } | Error::AllocFailed { .. }
        ),
        "unexpected error shape: {err}"
    );
}

/// A plain capture has no arena, so it reports `None`. Lives in the same
/// test as the arena case: two global-mode captures in one process cannot
/// overlap, and the test harness runs test functions in parallel.
fn plain_capture_reports_no_arena(client: &CudaClient, device: &CudaDevice) {
    use numr::ops::BinaryOps as _;

    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; 64], &[64], device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; 64], &[64], device).unwrap();
    let c = Tensor::<CudaRuntime>::from_slice(&vec![0.0f32; 64], &[64], device).unwrap();

    let captured =
        CudaRuntime::capture_graph_into(client, &[&a, &b], &[&c], |cc| cc.add_into(&c, &a, &b))
            .unwrap();
    assert_eq!(captured.arena_bytes_used(), None);
}
