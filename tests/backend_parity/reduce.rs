// Backend parity tests for ReduceOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

#[derive(Clone)]
struct ReduceTest {
    data: Vec<f64>,
    shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
}

impl ReduceTest {
    fn new(data: Vec<f64>, shape: Vec<usize>, dims: Vec<usize>, keepdim: bool) -> Self {
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

fn test_reduce_parity(op: &str, test_cases: &[ReduceTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let tensor = tensor_from_f64(&tc.data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_reduce_op(&cpu_client, op, &tensor, &tc.dims, tc.keepdim)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_reduce_op(&cuda_client, op, &tensor, &tc.dims, tc.keepdim)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_reduce_op(&wgpu_client, op, &tensor, &tc.dims, tc.keepdim)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

macro_rules! reduce_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_reduce_parity($op, $cases, dtype);
            }
        }
    };
}

// ============================================================================
// Reduce Operation Parity Tests
// ============================================================================

reduce_case!(
    test_sum_parity,
    "sum",
    &[
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], true),
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
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
            vec![1],
            false,
        ),
        ReduceTest::new(
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![1, 2],
            false,
        ),
    ]
);

reduce_case!(
    test_mean_parity,
    "mean",
    &[
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
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![0, 2],
            true,
        ),
    ]
);

reduce_case!(
    test_max_parity,
    "max",
    &[
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
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![0, 1],
            false,
        ),
    ]
);

reduce_case!(
    test_min_parity,
    "min",
    &[
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
            (1..=24).map(|v| v as f64).collect(),
            vec![2, 3, 4],
            vec![0, 1],
            false,
        ),
    ]
);

reduce_case!(
    test_prod_parity,
    "prod",
    &[
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
    ]
);

reduce_case!(
    test_any_parity,
    "any",
    &[
        ReduceTest::new(vec![0.0, 0.0, 0.0, 0.0], vec![4], vec![0], false),
        ReduceTest::new(vec![0.0, 1.0, 0.0, 2.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            vec![2, 3],
            vec![0],
            false,
        ),
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
    ]
);

reduce_case!(
    test_all_parity,
    "all",
    &[
        ReduceTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![0], false),
        ReduceTest::new(vec![1.0, 0.0, 2.0, 3.0], vec![4], vec![0], false),
        ReduceTest::new(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0],
            vec![2, 3],
            vec![0],
            false,
        ),
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
    ]
);

// ============================================================================
// CPU Parallelism Config Test (F32-specific, not dtype-parameterized)
// ============================================================================

#[test]
fn test_cpu_reduce_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(64)));

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
