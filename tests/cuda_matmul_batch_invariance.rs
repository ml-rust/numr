//! A row's CUDA F32 matmul result does not depend on how many rows share the
//! launch.
//!
//! Every F32 shape runs the compile-time tiled family, whose tiles all
//! accumulate one FMA per k in k order per output element, and the
//! transposed-weight path at M <= 4 runs the one-thread-per-output kernel,
//! which forms the same chain. So row `r` of an M-row product must be the
//! same bits as the 1-row product of row `r` alone, across the kernel
//! boundary (4 rows) and the tile boundaries the shape rule crosses (16, 64,
//! 128 rows), for a contiguous `[K, N]` operand and for the transposed
//! `[N, K]` weight view `Linear` multiplies by, in the 2-D and the batched
//! forms.
//!
//! Run with:
//!   cd numr && cargo test --features cuda --test cuda_matmul_batch_invariance

#![cfg(feature = "cuda")]

use numr::ops::MatmulOps;
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Row counts on both sides of the small-M kernel cutoff (4 | 5) and of every
/// tile boundary the shape rule has, plus the decode batches and the DiT CFG
/// pair.
const ROW_COUNTS: [usize; 12] = [1, 2, 3, 4, 5, 8, 16, 17, 22, 64, 65, 200];

/// Widths and depths: a wide weight, a narrow one, and ragged sizes that
/// leave partial tiles on every axis.
const SHAPES: [(usize, usize); 4] = [(1024, 1024), (1000, 1000), (64, 4096), (1536, 100)];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 251) as f32) * 0.004 - 0.5)
        .collect()
}

fn tensor(device: &CudaDevice, data: &[f32], shape: &[usize]) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, device).expect("tensor")
}

/// `[m, n]` result of rows `0..m` of `a_rows` against the weight, with the
/// weight as the transposed `[n, k]` view or as a contiguous `[k, n]` matrix.
fn product(
    client: &CudaClient,
    device: &CudaDevice,
    a_rows: &[f32],
    m: usize,
    k: usize,
    w: &Tensor<CudaRuntime>,
    transposed: bool,
) -> Vec<f32> {
    let a = tensor(device, &a_rows[..m * k], &[m, k]);
    let b = if transposed {
        w.t().expect("view")
    } else {
        w.t().expect("view").contiguous().expect("copy")
    };
    let out = client.matmul(&a, &b).expect("matmul");
    client.synchronize();
    out.to_vec::<f32>()
}

/// The batched form: `[2, m, k] @ [2, k, n]` with two different weights, so a
/// row must also not depend on the batch slice it sits in.
fn batched_product(
    client: &CudaClient,
    device: &CudaDevice,
    a_rows: &[f32],
    m: usize,
    k: usize,
    w2: &Tensor<CudaRuntime>,
    transposed: bool,
) -> Vec<f32> {
    let mut data = a_rows[..m * k].to_vec();
    data.extend_from_slice(&a_rows[..m * k]);
    let a = tensor(device, &data, &[2, m, k]);
    let b = if transposed {
        w2.transpose(1, 2).expect("view")
    } else {
        w2.transpose(1, 2)
            .expect("view")
            .contiguous()
            .expect("copy")
    };
    let out = client.matmul(&a, &b).expect("batched matmul");
    client.synchronize();
    out.to_vec::<f32>()
}

fn check_rows(what: &str, n: usize, k: usize, m: usize, got: &[f32], alone: &[f32]) {
    for r in 0..m {
        for c in 0..n {
            let g = got[r * n + c];
            let w = alone[r * n + c];
            assert!(
                g.to_bits() == w.to_bits(),
                "{what} N={n} K={k}: row {r} col {c} at M={m} is {g:e} ({:#010x}), alone it is {w:e} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }
}

#[test]
fn f32_rows_do_not_depend_on_the_batch() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    let max_m = *ROW_COUNTS.iter().max().expect("row counts");
    for &(n, k) in &SHAPES {
        let w = tensor(&device, &values(n * k, 2), &[n, k]);
        let w2 = tensor(&device, &values(2 * n * k, 5), &[2, n, k]);
        let a_rows = values(max_m * k, 1);
        for transposed in [true, false] {
            let what = if transposed { "x @ W.T" } else { "x @ B" };
            let mut alone = Vec::with_capacity(max_m * n);
            for r in 0..max_m {
                alone.extend(product(
                    &client,
                    &device,
                    &a_rows[r * k..],
                    1,
                    k,
                    &w,
                    transposed,
                ));
            }
            let mut alone_batched = Vec::with_capacity(2 * max_m * n);
            let mut alone_slice1 = Vec::with_capacity(max_m * n);
            for r in 0..max_m {
                let one =
                    batched_product(&client, &device, &a_rows[r * k..], 1, k, &w2, transposed);
                alone_batched.extend_from_slice(&one[..n]);
                alone_slice1.extend_from_slice(&one[n..]);
            }
            for &m in &ROW_COUNTS {
                let out = product(&client, &device, &a_rows, m, k, &w, transposed);
                check_rows(what, n, k, m, &out, &alone);
                let out = batched_product(&client, &device, &a_rows, m, k, &w2, transposed);
                check_rows(
                    &format!("batched {what} slice 0"),
                    n,
                    k,
                    m,
                    &out[..m * n],
                    &alone_batched,
                );
                check_rows(
                    &format!("batched {what} slice 1"),
                    n,
                    k,
                    m,
                    &out[m * n..],
                    &alone_slice1,
                );
            }
        }
    }
}
