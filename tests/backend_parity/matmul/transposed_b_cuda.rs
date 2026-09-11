// CUDA parity for F32 matmul against a transposed-view B operand.
//
// A `Linear` weight is `[N, K]`, and `x @ W.T` hands matmul a `[K, N]` view
// with strides `[1, K]`. Above the GEMV cutoff the CUDA backend reads that
// buffer in place through the transposed-B tiled kernels
// (`matmul_f32_tiled_bt_*`, `matmul_batched_f32_tiled_bt_*`) instead of
// copying the whole weight per call. These cases pin that path against the
// CPU backend, whose own in-place read `tests/matmul_transposed_b.rs` pins
// against the contiguous product.
//
// Shapes are chosen so every kernel variant runs: both tiles (64x64x32 for
// m <= 64 or n <= 64, 128x128x8 above), K a multiple of four (float4 tile
// loads) and not (scalar loads), ragged edges on every dim, and the batched
// form with and without a broadcast operand.

#![cfg(feature = "cuda")]

use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{assert_tensor_allclose, create_cpu_client};

fn values(len: usize, seed: f64) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as f64) * seed + 0.37).sin() * 0.5) as f32)
        .collect()
}

/// `A[batch.., m, k] @ W[batch.., n, k].T` on one backend, W passed as the
/// transposed view.
fn matmul_view<R: Runtime<DType = DType>>(
    client: &R::Client,
    device: &R::Device,
    a_shape: &[usize],
    w_shape: &[usize],
) -> Tensor<R>
where
    R::Client: MatmulOps<R>,
{
    let a_len: usize = a_shape.iter().product();
    let w_len: usize = w_shape.iter().product();
    let a = Tensor::<R>::from_slice(&values(a_len, 0.0011), a_shape, device).unwrap();
    let w = Tensor::<R>::from_slice(&values(w_len, 0.0019), w_shape, device).unwrap();
    let last = w_shape.len() - 1;
    let b_view = w.transpose((last - 1) as isize, last as isize).unwrap();
    client.matmul(&a, &b_view).unwrap()
}

fn check(cases: &[(&[usize], &[usize])]) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let expected: Vec<_> = cases
        .iter()
        .map(|(a, w)| matmul_view::<numr::runtime::cpu::CpuRuntime>(&cpu_client, &cpu_device, a, w))
        .collect();
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, (a, w)) in cases.iter().enumerate() {
            let got =
                matmul_view::<numr::runtime::cuda::CudaRuntime>(&cuda_client, &cuda_device, a, w);
            assert_tensor_allclose(
                &got,
                &expected[idx],
                DType::F32,
                &format!("matmul transposed-B CUDA vs CPU case {idx} A{a:?} W{w:?}"),
            );
        }
    });
}

#[test]
fn test_matmul_transposed_b_2d_cuda_matches_cpu() {
    check(&[
        // 64x64x32 tile, float4 K, first M above the GEMV cutoff.
        (&[17, 64], &[96, 64]),
        // 64x64x32 tile, K not a multiple of four, every dim ragged.
        (&[48, 130], &[200, 130]),
        // 128x128x8 tile, float4 K, ragged M and N.
        (&[200, 100], &[150, 100]),
        // 128x128x8 tile, scalar K.
        (&[130, 141], &[77, 141]),
        // K shorter than one tile depth.
        (&[70, 5], &[90, 5]),
    ]);
}

#[test]
fn test_matmul_transposed_b_batched_cuda_matches_cpu() {
    check(&[
        // 64x64x32 tile, per-batch weights.
        (&[3, 70, 130], &[3, 50, 130]),
        // 128x128x8 tile, per-batch weights, float4 K.
        (&[2, 200, 100], &[2, 150, 100]),
        // One weight shared over the batch.
        (&[3, 70, 128], &[1, 50, 128]),
    ]);
}
