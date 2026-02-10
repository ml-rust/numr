// Backend parity tests for EinsumOps trait
//
// Tests verify that einsum operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::EinsumOps;
use numr::tensor::Tensor;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::backend_parity::helpers::assert_single_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

struct EinsumTest {
    notation: &'static str,
    inputs: Vec<(Vec<f32>, Vec<usize>)>,
}

impl EinsumTest {
    fn new(notation: &'static str, inputs: Vec<(Vec<f32>, Vec<usize>)>) -> Self {
        EinsumTest { notation, inputs }
    }
}

fn test_einsum_parity(test_cases: Vec<EinsumTest>) {
    for test_case in &test_cases {
        // CPU baseline
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensors: Vec<_> = test_case
            .inputs
            .iter()
            .map(|(data, shape)| Tensor::from_slice(data, shape, &cpu_device))
            .collect();
        let cpu_refs: Vec<_> = cpu_tensors.iter().collect();
        let cpu_result = cpu_client
            .einsum(test_case.notation, &cpu_refs)
            .expect("CPU einsum failed")
            .to_vec::<f32>();

        // CUDA parity
        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensors: Vec<_> = test_case
                .inputs
                .iter()
                .map(|(data, shape)| Tensor::from_slice(data, shape, &cuda_device))
                .collect();
            let cuda_refs: Vec<_> = cuda_tensors.iter().collect();
            let cuda_result = cuda_client
                .einsum(test_case.notation, &cuda_refs)
                .expect("CUDA einsum failed")
                .to_vec::<f32>();
            assert_single_parity_f32(&cpu_result, &cuda_result, test_case.notation, "cuda");
        });

        // WebGPU parity
        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensors: Vec<_> = test_case
                .inputs
                .iter()
                .map(|(data, shape)| Tensor::from_slice(data, shape, &wgpu_device))
                .collect();
            let wgpu_refs: Vec<_> = wgpu_tensors.iter().collect();
            let wgpu_result = wgpu_client
                .einsum(test_case.notation, &wgpu_refs)
                .expect("WebGPU einsum failed")
                .to_vec::<f32>();
            assert_single_parity_f32(&cpu_result, &wgpu_result, test_case.notation, "wgpu");
        });
    }
}

// ============================================================================
// Einsum Parity Tests
// ============================================================================

#[test]
fn test_einsum_matmul_parity() {
    // Matrix multiplication: ij,jk->ik
    // A: 2x3, B: 3x2 -> C: 2x2
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    test_einsum_parity(vec![EinsumTest::new(
        "ij,jk->ik",
        vec![(a, vec![2, 3]), (b, vec![3, 2])],
    )]);
}

#[test]
fn test_einsum_batched_matmul_parity() {
    // Batched matrix multiplication: bij,bjk->bik
    let a = vec![
        // Batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ];
    let b = vec![
        // Batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ];

    test_einsum_parity(vec![EinsumTest::new(
        "bij,bjk->bik",
        vec![(a, vec![2, 2, 3]), (b, vec![2, 3, 2])],
    )]);
}

#[test]
fn test_einsum_transpose_parity() {
    // Transpose: ij->ji
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    test_einsum_parity(vec![EinsumTest::new("ij->ji", vec![(a, vec![2, 3])])]);
}

#[test]
fn test_einsum_outer_product_parity() {
    // Outer product: i,j->ij
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0, 7.0];

    test_einsum_parity(vec![EinsumTest::new(
        "i,j->ij",
        vec![(a, vec![3]), (b, vec![4])],
    )]);
}

#[test]
fn test_einsum_trace_parity() {
    // Trace: ii->
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    test_einsum_parity(vec![EinsumTest::new("ii->", vec![(a, vec![3, 3])])]);
}

#[test]
fn test_einsum_elementwise_parity() {
    // Element-wise multiplication (Hadamard product): ij,ij->ij
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    test_einsum_parity(vec![EinsumTest::new(
        "ij,ij->ij",
        vec![(a, vec![2, 3]), (b, vec![2, 3])],
    )]);
}

#[test]
fn test_einsum_sum_parity() {
    // Sum all elements: ij->
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    test_einsum_parity(vec![EinsumTest::new("ij->", vec![(a, vec![2, 3])])]);
}

#[test]
fn test_einsum_reduction_parity() {
    // Row sum: ij->i
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    test_einsum_parity(vec![EinsumTest::new("ij->i", vec![(a, vec![2, 3])])]);
}
