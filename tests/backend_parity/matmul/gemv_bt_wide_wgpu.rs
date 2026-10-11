// GEMV-BT on WebGPU with more output columns than one dispatch dimension
// holds (65535 workgroups). An LLM head is `[hidden] x [vocab, hidden]^T`
// with a vocab of 128256, which takes this path at decode.
//
// Every test below is `#[cfg(feature = "wgpu")]`, so these imports are too.
#[cfg(feature = "wgpu")]
use numr::ops::MatmulOps;
#[cfg(feature = "wgpu")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;

/// `a [m, k] x b [n, k]^T` on CPU and WebGPU, `b` passed as a transposed
/// view so WebGPU takes GEMV-BT. `batch` > 1 runs the batched kernel.
#[cfg(feature = "wgpu")]
fn assert_wide_gemv_bt(batch: usize, m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i as f32) * 0.37).sin())
        .collect();
    let b: Vec<f32> = (0..batch * n * k)
        .map(|i| ((i as f32) * 0.011).cos() * 0.5)
        .collect();
    let (a_shape, b_shape) = if batch == 1 {
        (vec![m, k], vec![n, k])
    } else {
        (vec![batch, m, k], vec![batch, n, k])
    };
    let rank = a_shape.len();

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&a, &a_shape, &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&b, &b_shape, &cpu_device)
        .expect("CPU B")
        .transpose(rank as isize - 2, rank as isize - 1)
        .expect("CPU B^T");
    let cpu = cpu_client
        .matmul(&a_cpu, &b_cpu)
        .expect("CPU matmul")
        .to_vec::<f32>();

    with_wgpu_backend_or_skip(|client, device| {
        let a_gpu = Tensor::<WgpuRuntime>::from_slice(&a, &a_shape, &device).expect("wgpu A");
        let b_gpu = Tensor::<WgpuRuntime>::from_slice(&b, &b_shape, &device)
            .expect("wgpu B")
            .transpose(rank as isize - 2, rank as isize - 1)
            .expect("wgpu B^T");
        let gpu = client
            .matmul(&a_gpu, &b_gpu)
            .expect("wgpu matmul")
            .to_vec::<f32>();
        assert_eq!(gpu.len(), cpu.len(), "output length");
        for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            let tol = 1e-4 * (1.0 + c.abs());
            assert!(
                (c - g).abs() <= tol,
                "batch {batch} m {m} n {n}: column {} differs, cpu {c} wgpu {g}",
                i % n
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn gemv_bt_wider_than_one_dispatch_dimension() {
    assert_wide_gemv_bt(1, 2, 70_001, 64);
}

#[cfg(feature = "wgpu")]
#[test]
fn batched_gemv_bt_wider_than_one_dispatch_dimension() {
    assert_wide_gemv_bt(2, 3, 65_537, 32);
}
