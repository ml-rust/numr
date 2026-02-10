// Backend parity tests for ReduceOps trait
//
// Tests verify that all ReduceOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::backend_parity::helpers::assert_case_parity_f32;
use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

struct ReduceTest {
    data: Vec<f32>,
    shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
}

impl ReduceTest {
    fn new(data: Vec<f32>, shape: Vec<usize>, dims: Vec<usize>, keepdim: bool) -> Self {
        ReduceTest {
            data,
            shape,
            dims,
            keepdim,
        }
    }
}

fn apply_reduce_op<R: Runtime>(
    client: &impl ReduceOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    dims: &[usize],
    keepdim: bool,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "sum" => client.sum(tensor, dims, keepdim),
        "mean" => client.mean(tensor, dims, keepdim),
        "max" => client.max(tensor, dims, keepdim),
        "min" => client.min(tensor, dims, keepdim),
        "prod" => client.prod(tensor, dims, keepdim),
        "any" => client.any(tensor, dims, keepdim),
        "all" => client.all(tensor, dims, keepdim),
        _ => panic!("Unknown reduce op: {}", op),
    }
}

fn test_reduce_parity(op: &str, test_cases: Vec<ReduceTest>) {
    // CPU baseline
    let cpu_results: Vec<Vec<f32>> = test_cases
        .iter()
        .map(|tc| {
            let (client, device) = create_cpu_client();
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &device);
            apply_reduce_op(&client, op, &tensor, &tc.dims, tc.keepdim)
                .expect("CPU operation failed")
                .to_vec::<f32>()
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &cuda_device);
            let result = apply_reduce_op(&cuda_client, op, &tensor, &tc.dims, tc.keepdim)
                .expect("CUDA operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "cuda");
        }
    });

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &wgpu_device);
            let result = apply_reduce_op(&wgpu_client, op, &tensor, &tc.dims, tc.keepdim)
                .expect("WebGPU operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "wgpu");
        }
    });
}

// ============================================================================
// Reduce Operation Parity Tests
// ============================================================================

#[test]
fn test_sum_parity() {
    test_reduce_parity(
        "sum",
        vec![
            // 1D full reduction
            ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
            // 1D full reduction with keepdim
            ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], true),
            // 2D reduce rows
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            // 2D reduce columns
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            // 3D reduce
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 2, 2],
                vec![1],
                false,
            ),
            // 3D multi-dim reduce
            ReduceTest::new(
                (1..=24).map(|v| v as f32).collect(),
                vec![2, 3, 4],
                vec![1, 2],
                false,
            ),
        ],
    );
}

#[test]
fn test_mean_parity() {
    test_reduce_parity(
        "mean",
        vec![
            ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            ReduceTest::new(
                (1..=24).map(|v| v as f32).collect(),
                vec![2, 3, 4],
                vec![0, 2],
                true,
            ),
        ],
    );
}

#[test]
fn test_max_parity() {
    test_reduce_parity(
        "max",
        vec![
            ReduceTest::new(vec![1.0, 4.0, 2.0, 3.0], vec![4], vec![0], false),
            ReduceTest::new(
                vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            ReduceTest::new(
                vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            ReduceTest::new(
                (1..=24).map(|v| v as f32).collect(),
                vec![2, 3, 4],
                vec![0, 1],
                false,
            ),
        ],
    );
}

#[test]
fn test_min_parity() {
    test_reduce_parity(
        "min",
        vec![
            ReduceTest::new(vec![1.0, 4.0, 2.0, 3.0], vec![4], vec![0], false),
            ReduceTest::new(
                vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            ReduceTest::new(
                vec![5.0, 2.0, 3.0, 1.0, 6.0, 4.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            ReduceTest::new(
                (1..=24).map(|v| v as f32).collect(),
                vec![2, 3, 4],
                vec![0, 1],
                false,
            ),
        ],
    );
}

#[test]
fn test_prod_parity() {
    test_reduce_parity(
        "prod",
        vec![
            ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
            ReduceTest::new(
                vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            ReduceTest::new(
                vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![1, 2, 3],
                vec![0, 2],
                false,
            ),
        ],
    );
}

#[test]
fn test_any_parity() {
    test_reduce_parity(
        "any",
        vec![
            // All zeros
            ReduceTest::new(vec![0.0, 0.0, 0.0, 0.0], vec![4], vec![0], false),
            // Some non-zero
            ReduceTest::new(vec![0.0, 1.0, 0.0, 2.0], vec![4], vec![0], false),
            // 2D reduce
            ReduceTest::new(
                vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            // 2D reduce along axis 1
            ReduceTest::new(
                vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            ReduceTest::new(
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                vec![1, 2, 3],
                vec![0, 2],
                false,
            ),
        ],
    );
}

#[test]
fn test_all_parity() {
    test_reduce_parity(
        "all",
        vec![
            // All non-zero
            ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
            // Some zeros
            ReduceTest::new(vec![1.0, 0.0, 2.0, 3.0], vec![4], vec![0], false),
            // 2D reduce
            ReduceTest::new(
                vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0],
                vec![2, 3],
                vec![0],
                false,
            ),
            // 2D reduce along axis 1 with zero
            ReduceTest::new(
                vec![1.0, 2.0, 0.0, 1.0, 2.0, 3.0],
                vec![2, 3],
                vec![1],
                false,
            ),
            ReduceTest::new(
                vec![1.0, 2.0, 3.0, 1.0, 0.0, 3.0],
                vec![1, 2, 3],
                vec![0, 2],
                false,
            ),
        ],
    );
}

#[test]
fn test_cpu_reduce_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(64)));

    // Large enough to exercise non-last-dim reduction paths where parallel scheduling matters.
    let shape = [96, 64, 32];
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.013).sin() + (i as f32 * 0.007).cos())
        .collect();
    let boolish_data: Vec<f32> = (0..numel)
        .map(|i| if i % 13 == 0 { 0.0 } else { 1.0 })
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&data, &shape, &device);
    let b = Tensor::<CpuRuntime>::from_slice(&boolish_data, &shape, &device);

    let sum_base: Vec<f32> = default_client.sum(&a, &[1], false).unwrap().to_vec();
    let sum_cfg: Vec<f32> = configured_client.sum(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&sum_base, &sum_cfg, "cpu_reduce_parallelism_sum");

    let mean_base: Vec<f32> = default_client.mean(&a, &[1], false).unwrap().to_vec();
    let mean_cfg: Vec<f32> = configured_client.mean(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&mean_base, &mean_cfg, "cpu_reduce_parallelism_mean");

    let max_base: Vec<f32> = default_client.max(&a, &[1], false).unwrap().to_vec();
    let max_cfg: Vec<f32> = configured_client.max(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&max_base, &max_cfg, "cpu_reduce_parallelism_max");

    let prod_base: Vec<f32> = default_client.prod(&a, &[1], false).unwrap().to_vec();
    let prod_cfg: Vec<f32> = configured_client.prod(&a, &[1], false).unwrap().to_vec();
    assert_parity_f32(&prod_base, &prod_cfg, "cpu_reduce_parallelism_prod");

    let any_base: Vec<f32> = default_client.any(&b, &[1], false).unwrap().to_vec();
    let any_cfg: Vec<f32> = configured_client.any(&b, &[1], false).unwrap().to_vec();
    assert_parity_f32(&any_base, &any_cfg, "cpu_reduce_parallelism_any");
}
