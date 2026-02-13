// Backend parity tests for SortOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::SortingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

#[test]
fn test_sort_parity() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let shape = vec![8];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_sorted = cpu_client
            .sort(&cpu_tensor, 0, false)
            .unwrap_or_else(|e| panic!("CPU sort failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_sorted = cuda_client
                    .sort(&cuda_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("CUDA sort failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_sorted,
                    &cpu_sorted,
                    dtype,
                    &format!("sort CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_sorted = wgpu_client
                    .sort(&wgpu_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("WebGPU sort failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_sorted,
                    &cpu_sorted,
                    dtype,
                    &format!("sort WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_argsort_parity() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let shape = vec![5];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_indices = cpu_client
            .argsort(&cpu_tensor, 0, false)
            .unwrap_or_else(|e| panic!("CPU argsort failed for {dtype:?}: {e}"));
        let cpu_data: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_indices = cuda_client
                    .argsort(&cuda_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("CUDA argsort failed for {dtype:?}: {e}"));
                let cuda_data: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_data, cuda_data,
                    "argsort CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_indices = wgpu_client
                    .argsort(&wgpu_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("WebGPU argsort failed for {dtype:?}: {e}"));
                let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_data, wgpu_as_i64,
                    "argsort WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}

#[test]
fn test_topk_parity() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let shape = vec![8];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let (cpu_vals, cpu_indices) = cpu_client
            .topk(&cpu_tensor, 3, 0, true, true)
            .unwrap_or_else(|e| panic!("CPU topk failed for {dtype:?}: {e}"));
        let cpu_i: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let (cuda_vals, cuda_indices) = cuda_client
                    .topk(&cuda_tensor, 3, 0, true, true)
                    .unwrap_or_else(|e| panic!("CUDA topk failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_vals,
                    &cpu_vals,
                    dtype,
                    &format!("topk values CUDA vs CPU [{dtype:?}]"),
                );
                let cuda_i: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_i, cuda_i,
                    "topk indices CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let (wgpu_vals, wgpu_indices) = wgpu_client
                    .topk(&wgpu_tensor, 3, 0, true, true)
                    .unwrap_or_else(|e| panic!("WebGPU topk failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_vals,
                    &cpu_vals,
                    dtype,
                    &format!("topk values WebGPU vs CPU [{dtype:?}]"),
                );
                let wgpu_i: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_i.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_i, wgpu_as_i64,
                    "topk indices WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}

#[test]
fn test_unique_parity() {
    let data = vec![1.0, 2.0, 2.0, 3.0, 1.0, 4.0];
    let shape = vec![6];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_unique = cpu_client
            .unique(&cpu_tensor, true)
            .unwrap_or_else(|e| panic!("CPU unique failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_unique = cuda_client
                    .unique(&cuda_tensor, true)
                    .unwrap_or_else(|e| panic!("CUDA unique failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_unique,
                    &cpu_unique,
                    dtype,
                    &format!("unique CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_unique = wgpu_client
                    .unique(&wgpu_tensor, true)
                    .unwrap_or_else(|e| panic!("WebGPU unique failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_unique,
                    &cpu_unique,
                    dtype,
                    &format!("unique WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_nonzero_parity() {
    let data = vec![0.0, 1.0, 0.0, 2.0, 3.0];
    let shape = vec![5];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_indices = cpu_client
            .nonzero(&cpu_tensor)
            .unwrap_or_else(|e| panic!("CPU nonzero failed for {dtype:?}: {e}"));
        let cpu_data: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_indices = cuda_client
                    .nonzero(&cuda_tensor)
                    .unwrap_or_else(|e| panic!("CUDA nonzero failed for {dtype:?}: {e}"));
                let cuda_data: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_data, cuda_data,
                    "nonzero CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_indices = wgpu_client
                    .nonzero(&wgpu_tensor)
                    .unwrap_or_else(|e| panic!("WebGPU nonzero failed for {dtype:?}: {e}"));
                let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_data, wgpu_as_i64,
                    "nonzero WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}

#[test]
fn test_searchsorted_parity() {
    let sorted_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let values_data = vec![2.0, 4.0, 6.0, 8.0];
    let sorted_shape = vec![5];
    let values_shape = vec![4];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_sorted =
            tensor_from_f64(&sorted_data, &sorted_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| {
                    panic!("CPU tensor_from_f64 (sorted) failed for {dtype:?}: {e}")
                });
        let cpu_values =
            tensor_from_f64(&values_data, &values_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| {
                    panic!("CPU tensor_from_f64 (values) failed for {dtype:?}: {e}")
                });
        let cpu_indices = cpu_client
            .searchsorted(&cpu_sorted, &cpu_values, false)
            .unwrap_or_else(|e| panic!("CPU searchsorted failed for {dtype:?}: {e}"));
        let cpu_data: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_sorted = tensor_from_f64(
                    &sorted_data,
                    &sorted_shape,
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| {
                    panic!("CUDA tensor_from_f64 (sorted) failed for {dtype:?}: {e}")
                });
                let cuda_values = tensor_from_f64(
                    &values_data,
                    &values_shape,
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| {
                    panic!("CUDA tensor_from_f64 (values) failed for {dtype:?}: {e}")
                });
                let cuda_indices = cuda_client
                    .searchsorted(&cuda_sorted, &cuda_values, false)
                    .unwrap_or_else(|e| panic!("CUDA searchsorted failed for {dtype:?}: {e}"));
                let cuda_data: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_data, cuda_data,
                    "searchsorted CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_sorted = tensor_from_f64(
                    &sorted_data,
                    &sorted_shape,
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| {
                    panic!("WebGPU tensor_from_f64 (sorted) failed for {dtype:?}: {e}")
                });
                let wgpu_values = tensor_from_f64(
                    &values_data,
                    &values_shape,
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| {
                    panic!("WebGPU tensor_from_f64 (values) failed for {dtype:?}: {e}")
                });
                let wgpu_indices = wgpu_client
                    .searchsorted(&wgpu_sorted, &wgpu_values, false)
                    .unwrap_or_else(|e| panic!("WebGPU searchsorted failed for {dtype:?}: {e}"));
                let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_data, wgpu_as_i64,
                    "searchsorted WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}
