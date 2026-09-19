//! `time_launches` returns a finite, positive, bounded time for a real
//! kernel, and less for an empty closure than for a tensor op.
//!
//! Run with:
//!   cd numr && cargo test --release --features cuda --test cuda_tune_timing

#![cfg(feature = "cuda")]

use numr::ops::BinaryOps;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime, time_launches};
use numr::tensor::Tensor;

const N: usize = 1 << 20;
const ITERS: usize = 5;
/// Upper bound on one iteration of a 1M-element add, in microseconds.
const MAX_MICROS: f32 = 100_000.0;

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

#[test]
fn tensor_op_time_is_finite_positive_and_bounded() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; N], &[N], &device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; N], &[N], &device).unwrap();

    let micros = time_launches(&client, ITERS, || client.add(&a, &b).map(|_| ())).unwrap();

    assert!(micros.is_finite(), "time is not finite: {micros}");
    assert!(micros > 0.0, "time is not positive: {micros}");
    assert!(micros < MAX_MICROS, "time exceeds bound: {micros} us");
}

#[test]
fn empty_closure_is_faster_than_tensor_op() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let a = Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; N], &[N], &device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&vec![2.0f32; N], &[N], &device).unwrap();

    let op = time_launches(&client, ITERS, || client.add(&a, &b).map(|_| ())).unwrap();
    let empty = time_launches(&client, ITERS, || Ok(())).unwrap();

    assert!(empty.is_finite() && empty >= 0.0, "empty time: {empty}");
    assert!(
        empty < op,
        "empty closure ({empty} us) is not faster than add ({op} us)"
    );
}

#[test]
fn zero_iters_is_an_error() {
    let Some((client, _device)) = cuda() else {
        return;
    };
    assert!(time_launches(&client, 0, || Ok(())).is_err());
}
