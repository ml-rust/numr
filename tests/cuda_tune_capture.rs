//! Inside a graph capture `tuned` returns the fallback without running the
//! probe, so a first-touch tune cannot invalidate the capture; outside, the
//! probe runs. Own binary: a global-mode capture must not share a process
//! with tests that record events or synchronize on other streams.
//!
//! Run with:
//!   cd numr && cargo test --release --features cuda --test cuda_tune_capture

#![cfg(feature = "cuda")]

use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

#[test]
fn tuned_does_not_probe_inside_graph_capture() {
    use numr::ops::BinaryOps as _;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::tuned;
    use std::cell::Cell;

    let Some((client, device)) = cuda() else {
        return;
    };
    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; 64], &[64], &device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; 64], &[64], &device).unwrap();
    let c = Tensor::<CudaRuntime>::from_slice(&vec![0.0f32; 64], &[64], &device).unwrap();

    assert!(!client.is_capturing());
    let probes = Cell::new(0u32);
    let inside = Cell::new(None);
    let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
        assert!(cc.is_capturing());
        inside.set(Some(tuned(cc, "test.capture_guard", 11u32, || {
            probes.set(probes.get() + 1);
            Ok(99u32)
        })));
        cc.add_into(&c, &a, &b)
    })
    .unwrap();
    captured.launch().unwrap();
    assert!(!client.is_capturing());

    assert_eq!(inside.get(), Some(11), "capture must see the fallback");
    assert_eq!(probes.get(), 0, "the probe ran inside capture");
    let after = tuned(&client, "test.capture_guard", 11u32, || {
        probes.set(probes.get() + 1);
        Ok(99u32)
    });
    assert_eq!(after, 99);
    assert_eq!(probes.get(), 1);
}
