//! The small-M transposed-weight kernel is bit-identical to the tiled one.
//!
//! `matmul_f32_smallm_bt` runs one thread per output for `x @ Wᵀ` at M <= 4
//! and forms each element as `fmaf(a[k], b[k], acc)` for k ascending, the
//! chain the tiled `matmul_f32_tiled_bt_*` kernels form by contraction. Both
//! launchers run here on the same device buffers, and every element must
//! match to the bit: that is what keeps a row's result independent of M when
//! the dispatch crosses from one kernel to the other.
//!
//! Depths cover a multiple of 32 (whole tiles), the FFN width, a depth that
//! is not a multiple of 32 (ragged last tile) and an odd depth (scalar loads
//! in the small-M kernel; every multiple of four takes its float4 path).
//! Widths run up to `MAX_SMALL_N`; one past it, and the 5120-wide weight the
//! kernel loses on, must make the launcher decline so the tiled kernel runs.
//!
//! Run with:
//!   cd numr && cargo test --features cuda --test cuda_matmul_smallm_bt_parity

#![cfg(feature = "cuda")]

use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::kernels::{
    MAX_SMALL_N, launch_matmul_batched_kernel_bt, launch_matmul_batched_smallm_bt_kernel,
    launch_matmul_kernel_bt, launch_matmul_smallm_bt_kernel,
};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const ROWS: [usize; 3] = [1, 2, 4];
const COLS: [usize; 3] = [48, 96, MAX_SMALL_N];
/// Widths the launcher must decline: one past the cutoff, and the wide
/// weight where the uncoalesced row reads lose to the tiled kernel.
const WIDE_COLS: [usize; 2] = [MAX_SMALL_N + 1, 5120];
const DEPTHS: [usize; 4] = [5120, 17408, 1000, 1001];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

/// Deterministic pseudo-random floats in `[-1, 1)` with full mantissas, so
/// the FMA chain exercises rounding at every step.
fn random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn tensor(device: &CudaDevice, data: &[f32], shape: &[usize]) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, device).expect("tensor")
}

fn check_bits(what: &str, m: usize, n: usize, tiled: &[f32], smallm: &[f32]) {
    assert_eq!(tiled.len(), smallm.len(), "{what}: output lengths differ");
    for r in 0..m {
        for c in 0..n {
            let t = tiled[r * n + c];
            let s = smallm[r * n + c];
            assert!(
                t.to_bits() == s.to_bits(),
                "{what}: row {r} col {c} tiled {t:e} ({:#010x}) small-M {s:e} ({:#010x})",
                t.to_bits(),
                s.to_bits()
            );
        }
    }
}

/// Both 2-D launchers on the same `A [m, k]` and `W [n, k]` buffers.
fn compare_2d(client: &CudaClient, device: &CudaDevice, m: usize, n: usize, k: usize) {
    let a = tensor(device, &random(m * k, 1), &[m, k]);
    let w = tensor(device, &random(n * k, 2), &[n, k]);
    let tiled = Tensor::<CudaRuntime>::empty(&[m, n], DType::F32, device).expect("out");
    let smallm = Tensor::<CudaRuntime>::empty(&[m, n], DType::F32, device).expect("out");

    let (ran_tiled, ran_smallm) = unsafe {
        (
            launch_matmul_kernel_bt(
                client.context(),
                client.stream(),
                device.id(),
                DType::F32,
                a.ptr(),
                w.ptr(),
                tiled.ptr(),
                m,
                n,
                k,
            )
            .expect("tiled launch"),
            launch_matmul_smallm_bt_kernel(
                client.context(),
                client.stream(),
                device.id(),
                DType::F32,
                a.ptr(),
                w.ptr(),
                smallm.ptr(),
                m,
                n,
                k,
            )
            .expect("small-M launch"),
        )
    };
    client.synchronize();
    assert!(ran_tiled, "M={m} N={n} K={k}: tiled bt launcher declined");
    assert!(ran_smallm, "M={m} N={n} K={k}: small-M launcher declined");

    check_bits(
        &format!("M={m} N={n} K={k}"),
        m,
        n,
        &tiled.to_vec::<f32>(),
        &smallm.to_vec::<f32>(),
    );
}

/// Both batched launchers on `A [a_batch, m, k]` and `W [b_batch, n, k]`,
/// `batch` outputs, an operand with count 1 broadcast over the batch.
/// `counts` is `(batch, a_batch, b_batch)`.
fn compare_batched(
    client: &CudaClient,
    device: &CudaDevice,
    counts: (usize, usize, usize),
    m: usize,
    n: usize,
    k: usize,
) {
    let (batch, a_batch, b_batch) = counts;
    let a = tensor(device, &random(a_batch * m * k, 3), &[a_batch, m, k]);
    let w = tensor(device, &random(b_batch * n * k, 4), &[b_batch, n, k]);
    let tiled = Tensor::<CudaRuntime>::empty(&[batch, m, n], DType::F32, device).expect("out");
    let smallm = Tensor::<CudaRuntime>::empty(&[batch, m, n], DType::F32, device).expect("out");

    let (ran_tiled, ran_smallm) = unsafe {
        (
            launch_matmul_batched_kernel_bt(
                client.context(),
                client.stream(),
                device.id(),
                DType::F32,
                a.ptr(),
                w.ptr(),
                tiled.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )
            .expect("tiled batched launch"),
            launch_matmul_batched_smallm_bt_kernel(
                client.context(),
                client.stream(),
                device.id(),
                DType::F32,
                a.ptr(),
                w.ptr(),
                smallm.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )
            .expect("small-M batched launch"),
        )
    };
    client.synchronize();
    let what = format!("batch={batch} a_batch={a_batch} b_batch={b_batch} M={m} N={n} K={k}");
    assert!(ran_tiled, "{what}: tiled bt launcher declined");
    assert!(ran_smallm, "{what}: small-M launcher declined");

    check_bits(
        &what,
        batch * m,
        n,
        &tiled.to_vec::<f32>(),
        &smallm.to_vec::<f32>(),
    );
}

/// The small-M launcher declines `[m, k] x [n, k]ᵀ` without launching, so
/// the output buffer stays untouched.
fn declines_2d(client: &CudaClient, device: &CudaDevice, m: usize, n: usize, k: usize) {
    let a = tensor(device, &random(m * k, 1), &[m, k]);
    let w = tensor(device, &random(n * k, 2), &[n, k]);
    let sentinel = vec![f32::NAN; m * n];
    let out = tensor(device, &sentinel, &[m, n]);
    let ran = unsafe {
        launch_matmul_smallm_bt_kernel(
            client.context(),
            client.stream(),
            device.id(),
            DType::F32,
            a.ptr(),
            w.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )
        .expect("small-M launch")
    };
    client.synchronize();
    assert!(
        !ran,
        "M={m} N={n} K={k}: small-M launcher ran past MAX_SMALL_N"
    );
    assert!(
        out.to_vec::<f32>().iter().all(|v| v.is_nan()),
        "M={m} N={n} K={k}: declined launcher wrote the output"
    );
}

#[test]
fn smallm_bt_declines_past_max_small_n() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &m in &ROWS {
        for &n in &WIDE_COLS {
            declines_2d(&client, &device, m, n, 1000);
        }
    }
}

#[test]
fn smallm_bt_matches_tiled_bt_to_the_bit() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &m in &ROWS {
        for &n in &COLS {
            for &k in &DEPTHS {
                compare_2d(&client, &device, m, n, k);
            }
        }
    }
}

#[test]
fn batched_smallm_bt_matches_tiled_bt_to_the_bit() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    // `[3, 2, 1000] x [3, 48, 1000]ᵀ`, every slice its own operands.
    compare_batched(&client, &device, (3, 3, 3), 2, 48, 1000);
    // One weight broadcast over three activation slices, and the reverse.
    compare_batched(&client, &device, (3, 3, 1), 2, 48, 1000);
    compare_batched(&client, &device, (3, 1, 3), 2, 48, 1000);
}
