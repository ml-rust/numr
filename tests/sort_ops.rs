//! Integration tests for sorting and search operations
//!
//! Tests verify correctness across:
//! - Different dtypes (f32, f64, i32, i64)
//! - Various dimensions and shapes
//! - Ascending/descending order
//! - Edge cases (empty, single element, duplicates)

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ComplexOps, ConditionalOps, CumulativeOps, IndexingOps,
    LinalgOps, LogicalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    StatisticalOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Sort Tests
// ============================================================================

#[test]
fn test_sort_1d_ascending() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    let sorted = client.sort(&a, 0, false).unwrap();

    assert_eq!(sorted.shape(), &[5]);
    let data: Vec<f32> = sorted.to_vec();
    assert_eq!(data, [1.0, 1.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sort_1d_descending() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    let sorted = client.sort(&a, 0, true).unwrap();

    assert_eq!(sorted.shape(), &[5]);
    let data: Vec<f32> = sorted.to_vec();
    assert_eq!(data, [5.0, 4.0, 3.0, 1.0, 1.0]);
}

#[test]
fn test_sort_2d_along_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [[3, 1], [2, 4]]
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 2.0, 4.0], &[2, 2], &device);
    let sorted = client.sort(&a, 0, false).unwrap();

    // Sort along rows: [[2, 1], [3, 4]]
    assert_eq!(sorted.shape(), &[2, 2]);
    let data: Vec<f32> = sorted.to_vec();
    assert_eq!(data, [2.0, 1.0, 3.0, 4.0]);
}

#[test]
fn test_sort_2d_along_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [[3, 1], [4, 2]]
    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 2.0], &[2, 2], &device);
    let sorted = client.sort(&a, 1, false).unwrap();

    // Sort along columns: [[1, 3], [2, 4]]
    assert_eq!(sorted.shape(), &[2, 2]);
    let data: Vec<f32> = sorted.to_vec();
    assert_eq!(data, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_sort_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 2.0], &[2, 2], &device);
    let sorted = client.sort(&a, -1, false).unwrap(); // Same as dim=1

    let data: Vec<f32> = sorted.to_vec();
    assert_eq!(data, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_sort_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::empty(&[0], DType::F32, &device);
    let sorted = client.sort(&a, 0, false).unwrap();

    assert_eq!(sorted.shape(), &[0]);
}

#[test]
fn test_sort_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[42.0f32], &[1], &device);
    let sorted = client.sort(&a, 0, false).unwrap();

    let data: Vec<f32> = sorted.to_vec();
    assert_eq!(data, [42.0]);
}

#[test]
fn test_sort_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[5i32, -3, 2, -1, 0], &[5], &device);
    let sorted = client.sort(&a, 0, false).unwrap();

    let data: Vec<i32> = sorted.to_vec();
    assert_eq!(data, [-3, -1, 0, 2, 5]);
}

#[test]
fn test_sort_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[std::f64::consts::PI, std::f64::consts::E, 1.41, 1.73],
        &[4],
        &device,
    );
    let sorted = client.sort(&a, 0, false).unwrap();

    let data: Vec<f64> = sorted.to_vec();
    assert_eq!(
        data,
        [1.41, 1.73, std::f64::consts::E, std::f64::consts::PI]
    );
}

// ============================================================================
// Sort with Indices Tests
// ============================================================================

#[test]
fn test_sort_with_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0], &[3], &device);
    let (values, indices) = client.sort_with_indices(&a, 0, false).unwrap();

    let v: Vec<f32> = values.to_vec();
    let i: Vec<i64> = indices.to_vec();

    assert_eq!(v, [1.0, 3.0, 4.0]);
    assert_eq!(i, [1, 0, 2]);
}

#[test]
fn test_sort_with_indices_descending() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0], &[3], &device);
    let (values, indices) = client.sort_with_indices(&a, 0, true).unwrap();

    let v: Vec<f32> = values.to_vec();
    let i: Vec<i64> = indices.to_vec();

    assert_eq!(v, [4.0, 3.0, 1.0]);
    assert_eq!(i, [2, 0, 1]);
}

// ============================================================================
// Argsort Tests
// ============================================================================

#[test]
fn test_argsort_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    let indices = client.argsort(&a, 0, false).unwrap();

    let i: Vec<i64> = indices.to_vec();
    // Original: [3, 1, 4, 1, 5]
    // Sorted:   [1, 1, 3, 4, 5]
    // Indices:  [1, 3, 0, 2, 4] (stable sort: first 1 at idx 1, second 1 at idx 3)
    assert_eq!(i, [1, 3, 0, 2, 4]);
}

#[test]
fn test_argsort_descending() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0], &[3], &device);
    let indices = client.argsort(&a, 0, true).unwrap();

    let i: Vec<i64> = indices.to_vec();
    assert_eq!(i, [2, 0, 1]); // [4, 3, 1]
}

// ============================================================================
// Top-K Tests
// ============================================================================

#[test]
fn test_topk_largest() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    let (values, indices) = client.topk(&a, 2, 0, true, true).unwrap();

    assert_eq!(values.shape(), &[2]);
    let v: Vec<f32> = values.to_vec();
    let i: Vec<i64> = indices.to_vec();

    assert_eq!(v, [5.0, 4.0]);
    assert_eq!(i, [4, 2]);
}

#[test]
fn test_topk_smallest() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device);
    let (values, indices) = client.topk(&a, 2, 0, false, true).unwrap();

    assert_eq!(values.shape(), &[2]);
    let v: Vec<f32> = values.to_vec();
    let i: Vec<i64> = indices.to_vec();

    assert_eq!(v, [1.0, 1.0]);
    assert_eq!(i, [1, 3]);
}

#[test]
fn test_topk_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [[5, 2, 8], [1, 9, 3]]
    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0], &[2, 3], &device);
    let (values, indices) = client.topk(&a, 2, 1, true, true).unwrap();

    assert_eq!(values.shape(), &[2, 2]);
    let v: Vec<f32> = values.to_vec();
    let i: Vec<i64> = indices.to_vec();

    // Row 0: top-2 of [5, 2, 8] = [8, 5]
    // Row 1: top-2 of [1, 9, 3] = [9, 3]
    assert_eq!(v, [8.0, 5.0, 9.0, 3.0]);
    assert_eq!(i, [2, 0, 1, 2]);
}

#[test]
fn test_topk_k_equals_n() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 2.0], &[3], &device);
    let (values, _) = client.topk(&a, 3, 0, true, true).unwrap();

    let v: Vec<f32> = values.to_vec();
    assert_eq!(v, [3.0, 2.0, 1.0]);
}

#[test]
fn test_topk_k_is_1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 5.0, 2.0], &[4], &device);
    let (values, indices) = client.topk(&a, 1, 0, true, true).unwrap();

    let v: Vec<f32> = values.to_vec();
    let i: Vec<i64> = indices.to_vec();

    assert_eq!(v, [5.0]);
    assert_eq!(i, [2]);
}

// ============================================================================
// Unique Tests
// ============================================================================

#[test]
fn test_unique_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0, 1.0], &[5], &device);
    let unique = client.unique(&a, true).unwrap();

    let u: Vec<f32> = unique.to_vec();
    assert_eq!(u, [1.0, 2.0, 3.0]);
}

#[test]
fn test_unique_all_same() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0, 5.0], &[3], &device);
    let unique = client.unique(&a, true).unwrap();

    let u: Vec<f32> = unique.to_vec();
    assert_eq!(u, [5.0]);
}

#[test]
fn test_unique_all_different() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 2.0], &[3], &device);
    let unique = client.unique(&a, true).unwrap();

    let u: Vec<f32> = unique.to_vec();
    assert_eq!(u, [1.0, 2.0, 3.0]);
}

#[test]
fn test_unique_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::empty(&[0], DType::F32, &device);
    let unique = client.unique(&a, true).unwrap();

    assert_eq!(unique.shape(), &[0]);
}

#[test]
fn test_unique_2d_flattened() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0], &[2, 2], &device);
    let unique = client.unique(&a, true).unwrap();

    // Should flatten and return unique values
    let u: Vec<f32> = unique.to_vec();
    assert_eq!(u, [1.0, 2.0, 3.0]);
}

#[test]
fn test_unique_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[5i32, 1, 3, 1, 5, 2], &[6], &device);
    let unique = client.unique(&a, true).unwrap();

    let u: Vec<i32> = unique.to_vec();
    assert_eq!(u, [1, 2, 3, 5]);
}

// ============================================================================
// Unique with Counts Tests
// ============================================================================

#[test]
fn test_unique_with_counts_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0, 1.0], &[5], &device);
    let (unique, inverse, counts) = client.unique_with_counts(&a).unwrap();

    let u: Vec<f32> = unique.to_vec();
    let inv: Vec<i64> = inverse.to_vec();
    let c: Vec<i64> = counts.to_vec();

    assert_eq!(u, [1.0, 2.0, 3.0]);
    // inverse maps each input element to its index in unique
    assert_eq!(inv, [0, 1, 1, 2, 0]); // 1->0, 2->1, 2->1, 3->2, 1->0
    assert_eq!(c, [2, 2, 1]); // 1 appears 2x, 2 appears 2x, 3 appears 1x
}

// ============================================================================
// Nonzero Tests
// ============================================================================

#[test]
fn test_nonzero_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 0.0, 2.0, 3.0], &[5], &device);
    let indices = client.nonzero(&a).unwrap();

    assert_eq!(indices.shape(), &[3, 1]);
    let i: Vec<i64> = indices.to_vec();
    assert_eq!(i, [1, 3, 4]); // indices of 1.0, 2.0, 3.0
}

#[test]
fn test_nonzero_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [[0, 1], [2, 0]]
    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 0.0], &[2, 2], &device);
    let indices = client.nonzero(&a).unwrap();

    assert_eq!(indices.shape(), &[2, 2]);
    let i: Vec<i64> = indices.to_vec();
    // Nonzero at (0,1) and (1,0)
    assert_eq!(i, [0, 1, 1, 0]);
}

#[test]
fn test_nonzero_all_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device);
    let indices = client.nonzero(&a).unwrap();

    assert_eq!(indices.shape(), &[0, 1]);
}

#[test]
fn test_nonzero_all_nonzero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let indices = client.nonzero(&a).unwrap();

    assert_eq!(indices.shape(), &[3, 1]);
    let i: Vec<i64> = indices.to_vec();
    assert_eq!(i, [0, 1, 2]);
}

#[test]
fn test_nonzero_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::empty(&[0], DType::F32, &device);
    let indices = client.nonzero(&a).unwrap();

    assert_eq!(indices.shape(), &[0, 1]);
}

#[test]
fn test_nonzero_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0i32, -1, 0, 5, 0], &[5], &device);
    let indices = client.nonzero(&a).unwrap();

    let i: Vec<i64> = indices.to_vec();
    assert_eq!(i, [1, 3]); // -1 at idx 1, 5 at idx 3
}

// ============================================================================
// Searchsorted Tests
// ============================================================================

#[test]
fn test_searchsorted_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let sorted = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 5.0, 7.0], &[4], &device);
    let values = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0], &[3], &device);

    let indices = client.searchsorted(&sorted, &values, false).unwrap();

    let i: Vec<i64> = indices.to_vec();
    // 2 -> 1 (between 1 and 3)
    // 4 -> 2 (between 3 and 5)
    // 6 -> 3 (between 5 and 7)
    assert_eq!(i, [1, 2, 3]);
}

#[test]
fn test_searchsorted_left_vs_right() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let sorted = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0], &[4], &device);
    let values = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

    let left = client.searchsorted(&sorted, &values, false).unwrap();
    let right = client.searchsorted(&sorted, &values, true).unwrap();

    let l: Vec<i64> = left.to_vec();
    let r: Vec<i64> = right.to_vec();

    assert_eq!(l, [1]); // leftmost position for 2.0
    assert_eq!(r, [3]); // rightmost position for 2.0
}

#[test]
fn test_searchsorted_edge_values() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let sorted = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 5.0], &[3], &device);
    let values = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 5.0, 6.0], &[4], &device);

    let indices = client.searchsorted(&sorted, &values, false).unwrap();

    let i: Vec<i64> = indices.to_vec();
    // 0 -> 0 (before all)
    // 1 -> 0 (at first element, leftmost)
    // 5 -> 2 (at last element, leftmost)
    // 6 -> 3 (after all)
    assert_eq!(i, [0, 0, 2, 3]);
}

#[test]
fn test_searchsorted_empty_sequence() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let sorted = Tensor::<CpuRuntime>::empty(&[0], DType::F32, &device);
    let values = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);

    let indices = client.searchsorted(&sorted, &values, false).unwrap();

    let i: Vec<i64> = indices.to_vec();
    assert_eq!(i, [0, 0]); // All values would be inserted at position 0
}

#[test]
fn test_searchsorted_empty_values() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let sorted = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let values = Tensor::<CpuRuntime>::empty(&[0], DType::F32, &device);

    let indices = client.searchsorted(&sorted, &values, false).unwrap();

    assert_eq!(indices.shape(), &[0]);
}

#[test]
fn test_searchsorted_i64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let sorted = Tensor::<CpuRuntime>::from_slice(&[10i64, 20, 30, 40], &[4], &device);
    let values = Tensor::<CpuRuntime>::from_slice(&[15i64, 25, 35], &[3], &device);

    let indices = client.searchsorted(&sorted, &values, false).unwrap();

    let i: Vec<i64> = indices.to_vec();
    assert_eq!(i, [1, 2, 3]);
}

// ============================================================================
// Stability Tests (stable sort preserves relative order of equal elements)
// ============================================================================

#[test]
fn test_sort_stability() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test with duplicates - indices should preserve relative order
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 1.0, 2.0, 1.0], &[5], &device);
    let (_, indices) = client.sort_with_indices(&a, 0, false).unwrap();

    let i: Vec<i64> = indices.to_vec();
    // Sorted: [1, 1, 1, 2, 2]
    // Stable: first 1 at 0, second 1 at 2, third 1 at 4, first 2 at 1, second 2 at 3
    assert_eq!(i, [0, 2, 4, 1, 3]);
}

// ============================================================================
// Backend Parity Tests (CUDA)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_parity {
    use numr::ops::*;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    use numr::tensor::Tensor;

    fn assert_close(cpu: &[f32], cuda: &[f32], tol: f32) {
        assert_eq!(cpu.len(), cuda.len(), "Length mismatch");
        for (i, (c, g)) in cpu.iter().zip(cuda.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff <= tol,
                "Mismatch at index {}: CPU={}, CUDA={}, diff={}",
                i,
                c,
                g,
                diff
            );
        }
    }

    #[test]
    fn test_sort_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[8], &cuda_device);

        let cpu_sorted = cpu_client.sort(&cpu_tensor, 0, false).unwrap();
        let cuda_sorted = cuda_client.sort(&cuda_tensor, 0, false).unwrap();

        let cpu_data: Vec<f32> = cpu_sorted.to_vec();
        let cuda_data: Vec<f32> = cuda_sorted.to_vec();
        assert_close(&cpu_data, &cuda_data, 1e-6);
    }

    #[test]
    fn test_argsort_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[5], &cuda_device);

        let cpu_indices = cpu_client.argsort(&cpu_tensor, 0, false).unwrap();
        let cuda_indices = cuda_client.argsort(&cuda_tensor, 0, false).unwrap();

        let cpu_data: Vec<i64> = cpu_indices.to_vec();
        let cuda_data: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_data, cuda_data);
    }

    #[test]
    fn test_topk_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[8], &cuda_device);

        let (cpu_vals, cpu_indices) = cpu_client.topk(&cpu_tensor, 3, 0, true, true).unwrap();
        let (cuda_vals, cuda_indices) = cuda_client.topk(&cuda_tensor, 3, 0, true, true).unwrap();

        let cpu_v: Vec<f32> = cpu_vals.to_vec();
        let cuda_v: Vec<f32> = cuda_vals.to_vec();
        assert_close(&cpu_v, &cuda_v, 1e-6);

        let cpu_i: Vec<i64> = cpu_indices.to_vec();
        let cuda_i: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_i, cuda_i);
    }

    #[test]
    fn test_unique_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [1.0f32, 2.0, 2.0, 3.0, 1.0, 4.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[6], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[6], &cuda_device);

        let cpu_unique = cpu_client.unique(&cpu_tensor, true).unwrap();
        let cuda_unique = cuda_client.unique(&cuda_tensor, true).unwrap();

        let cpu_data: Vec<f32> = cpu_unique.to_vec();
        let cuda_data: Vec<f32> = cuda_unique.to_vec();
        assert_close(&cpu_data, &cuda_data, 1e-6);
    }

    #[test]
    fn test_nonzero_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [0.0f32, 1.0, 0.0, 2.0, 3.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[5], &cuda_device);

        let cpu_indices = cpu_client.nonzero(&cpu_tensor).unwrap();
        let cuda_indices = cuda_client.nonzero(&cuda_tensor).unwrap();

        assert_eq!(cpu_indices.shape(), cuda_indices.shape());
        let cpu_data: Vec<i64> = cpu_indices.to_vec();
        let cuda_data: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_data, cuda_data);
    }

    #[test]
    fn test_searchsorted_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let sorted_data = [1.0f32, 3.0, 5.0, 7.0, 9.0];
        let values_data = [2.0f32, 4.0, 6.0, 8.0];

        let cpu_sorted = Tensor::<CpuRuntime>::from_slice(&sorted_data, &[5], &cpu_device);
        let cpu_values = Tensor::<CpuRuntime>::from_slice(&values_data, &[4], &cpu_device);
        let cuda_sorted = Tensor::<CudaRuntime>::from_slice(&sorted_data, &[5], &cuda_device);
        let cuda_values = Tensor::<CudaRuntime>::from_slice(&values_data, &[4], &cuda_device);

        let cpu_indices = cpu_client
            .searchsorted(&cpu_sorted, &cpu_values, false)
            .unwrap();
        let cuda_indices = cuda_client
            .searchsorted(&cuda_sorted, &cuda_values, false)
            .unwrap();

        let cpu_data: Vec<i64> = cpu_indices.to_vec();
        let cuda_data: Vec<i64> = cuda_indices.to_vec();
        assert_eq!(cpu_data, cuda_data);
    }
}

// ============================================================================
// Backend Parity Tests (WebGPU)
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_parity {
    use numr::ops::*;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::runtime::wgpu::WgpuRuntime;
    use numr::tensor::Tensor;

    fn assert_close(cpu: &[f32], wgpu: &[f32], tol: f32) {
        assert_eq!(cpu.len(), wgpu.len(), "Length mismatch");
        for (i, (c, g)) in cpu.iter().zip(wgpu.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff <= tol,
                "Mismatch at index {}: CPU={}, WebGPU={}, diff={}",
                i,
                c,
                g,
                diff
            );
        }
    }

    #[test]
    fn test_sort_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuRuntime::default_device();
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        // Keep data small enough for WebGPU's sort limit (512 elements)
        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[8], &wgpu_device);

        let cpu_sorted = cpu_client.sort(&cpu_tensor, 0, false).unwrap();
        let wgpu_sorted = wgpu_client.sort(&wgpu_tensor, 0, false).unwrap();

        let cpu_data: Vec<f32> = cpu_sorted.to_vec();
        let wgpu_data: Vec<f32> = wgpu_sorted.to_vec();
        assert_close(&cpu_data, &wgpu_data, 1e-6);
    }

    #[test]
    fn test_argsort_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuRuntime::default_device();
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[5], &wgpu_device);

        let cpu_indices = cpu_client.argsort(&cpu_tensor, 0, false).unwrap();
        let wgpu_indices = wgpu_client.argsort(&wgpu_tensor, 0, false).unwrap();

        // Note: WebGPU uses I32 for indices, CPU uses I64
        let cpu_data: Vec<i64> = cpu_indices.to_vec();
        let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
        let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
        assert_eq!(cpu_data, wgpu_as_i64);
    }

    #[test]
    fn test_topk_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuRuntime::default_device();
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[8], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[8], &wgpu_device);

        let (cpu_vals, cpu_indices) = cpu_client.topk(&cpu_tensor, 3, 0, true, true).unwrap();
        let (wgpu_vals, wgpu_indices) = wgpu_client.topk(&wgpu_tensor, 3, 0, true, true).unwrap();

        let cpu_v: Vec<f32> = cpu_vals.to_vec();
        let wgpu_v: Vec<f32> = wgpu_vals.to_vec();
        assert_close(&cpu_v, &wgpu_v, 1e-6);

        // WebGPU uses I32, convert for comparison
        let cpu_i: Vec<i64> = cpu_indices.to_vec();
        let wgpu_i: Vec<i32> = wgpu_indices.to_vec();
        let wgpu_as_i64: Vec<i64> = wgpu_i.iter().map(|&x| x as i64).collect();
        assert_eq!(cpu_i, wgpu_as_i64);
    }

    #[test]
    fn test_searchsorted_parity() {
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuRuntime::default_device();
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        let sorted_data = [1.0f32, 3.0, 5.0, 7.0, 9.0];
        let values_data = [2.0f32, 4.0, 6.0, 8.0];

        let cpu_sorted = Tensor::<CpuRuntime>::from_slice(&sorted_data, &[5], &cpu_device);
        let cpu_values = Tensor::<CpuRuntime>::from_slice(&values_data, &[4], &cpu_device);
        let wgpu_sorted = Tensor::<WgpuRuntime>::from_slice(&sorted_data, &[5], &wgpu_device);
        let wgpu_values = Tensor::<WgpuRuntime>::from_slice(&values_data, &[4], &wgpu_device);

        let cpu_indices = cpu_client
            .searchsorted(&cpu_sorted, &cpu_values, false)
            .unwrap();
        let wgpu_indices = wgpu_client
            .searchsorted(&wgpu_sorted, &wgpu_values, false)
            .unwrap();

        // WebGPU uses I32, convert for comparison
        let cpu_data: Vec<i64> = cpu_indices.to_vec();
        let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
        let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
        assert_eq!(cpu_data, wgpu_as_i64);
    }
}
