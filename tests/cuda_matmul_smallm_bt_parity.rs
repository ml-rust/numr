//! The small-M transposed-weight kernel is bit-identical to the tiled one.
//!
//! `matmul_f32_smallm_bt` runs one thread per output for `x @ Wᵀ` at small
//! M and forms each element as `fmaf(a[k], b[k], acc)` for k ascending, the
//! chain the tiled `matmul_f32_tiled_bt_*` kernels form by contraction. Both
//! launchers run here on the same device buffers, and every element must
//! match to the bit: that is what keeps a row's result independent of M when
//! the dispatch crosses from one kernel to the other.
//!
//! Which shapes the small-M launcher serves depends on the device's SM
//! count and its tuned wave bound (`smallm_applies` with
//! `SmallmLimits::of`), so every shape is checked against that oracle: a
//! served shape must launch and match, a declined shape must not touch the
//! output. The rows cover the decode batches and `MAX_SMALL_M`; the widest
//! column at the widest row is past the wave bound on any part, so both
//! branches run. The tune tests pin the bound's range and stability and
//! force the kernel on the shapes just inside and just outside it.
//!
//! Depths cover a multiple of 32 (whole tiles), the FFN width, a depth that
//! is not a multiple of 32 (ragged last tile), odd depths (scalar loads in
//! the small-M kernel; every multiple of four takes its float4 path), and
//! the small-M kernel's chunk edges: K equal to one chunk (512), whole
//! chunks (5120, 17408), one chunk plus a partial one (1000, 1001), K under
//! one chunk (128, 100, 129) and K under one float4 (3). Widths cover
//! N = 1, a width that crosses a warp (33), whole row groups of the
//! kernel's 8-row blocks (48, 64, 96, 128, `MAX_SMALL_N`) and partial last
//! groups (33, 65, 129, 40); one past `MAX_SMALL_N`, and the 5120-wide
//! weight the kernel loses on, must make the launcher decline so the tiled
//! kernel runs.
//!
//! Run with:
//!   cd numr && cargo test --features cuda --test cuda_matmul_smallm_bt_parity

#![cfg(feature = "cuda")]

use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::kernels::{
    MAX_SMALL_M, MAX_SMALL_N, SMALLM_MAX_WAVES_CEILING, SMALLM_ROWS_PER_BLOCK, SmallmLimits,
    launch_matmul_batched_kernel_bt, launch_matmul_batched_smallm_bt_kernel,
    launch_matmul_kernel_bt, launch_matmul_smallm_bt_f32_ungated, launch_matmul_smallm_bt_kernel,
    smallm_applies, smallm_block_count, smallm_max_waves,
};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const ROWS: [usize; 4] = [1, 2, 4, MAX_SMALL_M];
const COLS: [usize; 10] = [1, 33, 40, 48, 64, 65, 96, 128, 129, MAX_SMALL_N];
/// Widths the launcher must decline: one past the cutoff, and the wide
/// weight where the tiled kernel's B reuse wins.
const WIDE_COLS: [usize; 2] = [MAX_SMALL_N + 1, 5120];
const DEPTHS: [usize; 9] = [5120, 17408, 1000, 1001, 512, 128, 100, 129, 3];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

/// Whether the small-M launcher serves `[m, k] x [n, k]ᵀ` on this device.
fn served(client: &CudaClient, m: usize, n: usize, batch: usize) -> bool {
    smallm_applies(DType::F32, m, n, batch, SmallmLimits::of(client))
}

/// Compare when the launcher serves the shape, otherwise check it declines.
fn check_2d(client: &CudaClient, device: &CudaDevice, m: usize, n: usize, k: usize) {
    if served(client, m, n, 1) {
        compare_2d(client, device, m, n, k);
    } else {
        declines_2d(client, device, m, n, k);
    }
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

/// Both 2-D launchers on the same `A [m, k]` and `W [n, k]` buffers. With
/// `forced`, the small-M kernel runs through its ungated entry, so the
/// shape need not pass the gate.
fn compare_2d_with(
    client: &CudaClient,
    device: &CudaDevice,
    m: usize,
    n: usize,
    k: usize,
    forced: bool,
) {
    let a = tensor(device, &random(m * k, 1), &[m, k]);
    let w = tensor(device, &random(n * k, 2), &[n, k]);
    let tiled = Tensor::<CudaRuntime>::empty(&[m, n], DType::F32, device).expect("out");
    let smallm = Tensor::<CudaRuntime>::empty(&[m, n], DType::F32, device).expect("out");

    let ran_tiled = unsafe {
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
    }
    .expect("tiled launch");
    let ran_smallm = if forced {
        unsafe {
            launch_matmul_smallm_bt_f32_ungated(
                client.context(),
                client.stream(),
                device.id(),
                a.ptr(),
                w.ptr(),
                smallm.ptr(),
                m,
                n,
                k,
            )
        }
        .expect("forced small-M launch");
        true
    } else {
        unsafe {
            launch_matmul_smallm_bt_kernel(
                client,
                DType::F32,
                a.ptr(),
                w.ptr(),
                smallm.ptr(),
                m,
                n,
                k,
            )
        }
        .expect("small-M launch")
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

/// Both 2-D launchers through their gates.
fn compare_2d(client: &CudaClient, device: &CudaDevice, m: usize, n: usize, k: usize) {
    compare_2d_with(client, device, m, n, k, false);
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
                client,
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
    assert_eq!(
        ran_smallm,
        served(client, m, n, batch),
        "{what}: small-M launcher disagrees with smallm_applies"
    );
    if !ran_smallm {
        return;
    }

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
        launch_matmul_smallm_bt_kernel(client, DType::F32, a.ptr(), w.ptr(), out.ptr(), m, n, k)
            .expect("small-M launch")
    };
    client.synchronize();
    assert!(
        !ran,
        "M={m} N={n} K={k}: small-M launcher ran past its bound"
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
    declines_2d(&client, &device, MAX_SMALL_M + 1, 1, 1000);
}

#[test]
fn smallm_bt_declines_past_the_wave_bound() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    // `MAX_SMALL_M x MAX_SMALL_N` is 8192 blocks: past 12 waves on any
    // part with fewer than 683 SMs.
    assert!(
        !served(&client, MAX_SMALL_M, MAX_SMALL_N, 1),
        "the widest shape is inside the wave bound on this device"
    );
    declines_2d(&client, &device, MAX_SMALL_M, MAX_SMALL_N, 1000);
}

#[test]
fn smallm_bt_matches_tiled_bt_to_the_bit() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    let mut compared = 0usize;
    for &m in &ROWS {
        for &n in &COLS {
            let runs = served(&client, m, n, 1);
            compared += usize::from(runs);
            for &k in &DEPTHS {
                check_2d(&client, &device, m, n, k);
            }
        }
    }
    assert!(compared > 0, "the launcher served no shape on this device");
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

/// The shapes just inside and just outside this device's block bound, as
/// `(m, n)`: the same M with one more row group, or the same N with one
/// more row. `None` for the outer shape when the bound admits every shape
/// in range.
fn shapes_at_the_bound(limits: SmallmLimits) -> ((usize, usize), Option<(usize, usize)>) {
    let rows = SMALLM_ROWS_PER_BLOCK as usize;
    let groups = MAX_SMALL_N / rows;
    let limit = limits.max_waves * limits.sm_count;
    if limit < groups {
        // Fewer blocks than one full-width row: widen N by one row group.
        ((1, limit * rows), Some((1, (limit + 1) * rows)))
    } else {
        // At least one full-width row: add rows at full width.
        let m_in = (limit / groups).min(MAX_SMALL_M);
        let outer = (m_in < MAX_SMALL_M).then_some((m_in + 1, MAX_SMALL_N));
        ((m_in, MAX_SMALL_N), outer)
    }
}

#[test]
fn tuned_wave_bound_is_in_range_and_stable() {
    let Some((client, _device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    let first = smallm_max_waves(&client);
    assert!(
        (1..=SMALLM_MAX_WAVES_CEILING).contains(&first),
        "tuned wave bound {first} is outside 1..={SMALLM_MAX_WAVES_CEILING}"
    );
    assert_eq!(smallm_max_waves(&client), first, "second call differs");
    assert_eq!(SmallmLimits::of(&client).max_waves, first);
}

#[test]
fn smallm_bt_matches_tiled_bt_at_both_sides_of_the_tuned_bound() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    let limits = SmallmLimits::of(&client);
    let (inner, outer) = shapes_at_the_bound(limits);
    let (m, n) = inner;
    assert!(
        smallm_block_count(m, n) <= limits.max_waves * limits.sm_count,
        "M={m} N={n} is not inside the bound"
    );
    assert!(
        served(&client, m, n, 1),
        "M={m} N={n}: inner shape declined"
    );
    compare_2d(&client, &device, m, n, 1000);
    compare_2d_with(&client, &device, m, n, 1000, true);
    if let Some((m, n)) = outer {
        assert!(
            smallm_block_count(m, n) > limits.max_waves * limits.sm_count,
            "M={m} N={n} is not outside the bound"
        );
        declines_2d(&client, &device, m, n, 1000);
        compare_2d_with(&client, &device, m, n, 1000, true);
    }
}
