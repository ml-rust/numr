//! Advanced indexing tests: scatter_reduce, gather_nd, bincount

use numr::ops::{IndexingOps, ScatterReduceOp};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Scatter Reduce Tests
// ============================================================================

#[test]
fn test_scatter_reduce_sum_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // dst tensor
    let dst = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

    // Scatter indices [0, 0, 2] - two values go to index 0
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2], &[3], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    let result = client
        .scatter_reduce(&dst, 0, &indices, &src, ScatterReduceOp::Sum, false)
        .unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    // index 0: 1.0 + 2.0 = 3.0, index 2: 3.0
    assert_eq!(data, [3.0, 0.0, 3.0, 0.0]);
}

#[test]
fn test_scatter_reduce_sum_include_self() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // dst tensor with initial values
    let dst = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2], &[3], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    let result = client
        .scatter_reduce(&dst, 0, &indices, &src, ScatterReduceOp::Sum, true)
        .unwrap();

    let data: Vec<f32> = result.to_vec();
    // index 0: 10.0 + 1.0 + 2.0 = 13.0
    // index 1: 20.0 (unchanged)
    // index 2: 30.0 + 3.0 = 33.0
    // index 3: 40.0 (unchanged)
    assert_eq!(data, [13.0, 20.0, 33.0, 40.0]);
}

#[test]
fn test_scatter_reduce_max() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let dst = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2, 2], &[4], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 3.0, 1.0, 7.0], &[4], &device);

    let result = client
        .scatter_reduce(&dst, 0, &indices, &src, ScatterReduceOp::Max, false)
        .unwrap();

    let data: Vec<f32> = result.to_vec();
    // index 0: max(5.0, 3.0) = 5.0
    // index 2: max(1.0, 7.0) = 7.0
    assert_eq!(data[0], 5.0);
    assert_eq!(data[2], 7.0);
}

#[test]
fn test_scatter_reduce_min() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let dst = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2, 2], &[4], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 3.0, 1.0, 7.0], &[4], &device);

    let result = client
        .scatter_reduce(&dst, 0, &indices, &src, ScatterReduceOp::Min, false)
        .unwrap();

    let data: Vec<f32> = result.to_vec();
    // index 0: min(5.0, 3.0) = 3.0
    // index 2: min(1.0, 7.0) = 1.0
    assert_eq!(data[0], 3.0);
    assert_eq!(data[2], 1.0);
}

#[test]
fn test_scatter_reduce_prod() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let dst = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2], &[3], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 5.0], &[3], &device);

    let result = client
        .scatter_reduce(&dst, 0, &indices, &src, ScatterReduceOp::Prod, false)
        .unwrap();

    let data: Vec<f32> = result.to_vec();
    // index 0: 2.0 * 3.0 = 6.0 (starts from 1.0 identity)
    // index 2: 5.0
    assert_eq!(data[0], 6.0);
    assert_eq!(data[2], 5.0);
}

#[test]
fn test_scatter_reduce_mean() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let dst = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 2], &[4], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 6.0, 9.0, 4.0], &[4], &device);

    let result = client
        .scatter_reduce(&dst, 0, &indices, &src, ScatterReduceOp::Mean, false)
        .unwrap();

    let data: Vec<f32> = result.to_vec();
    // index 0: mean(3.0, 6.0, 9.0) = 6.0
    // index 2: mean(4.0) = 4.0
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[2] - 4.0).abs() < 1e-6);
}

// ============================================================================
// Gather ND Tests
// ============================================================================

#[test]
fn test_gather_nd_2d_full_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D input [2, 2]
    let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[2, 2], &device);

    // Indices pointing to [0,0], [1,1] - full coordinate per index
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 1, 1], &[2, 2], &device);

    let result = client.gather_nd(&input, &indices).unwrap();

    // Output shape: [2] (indices.shape[:-1] = [2], no trailing dims since M=2=ndim)
    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // input[0,0] = 0.0, input[1,1] = 3.0
    assert_eq!(data, [0.0, 3.0]);
}

#[test]
fn test_gather_nd_2d_partial_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D input [3, 4]
    let input = Tensor::<CpuRuntime>::from_slice(
        &[
            0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ],
        &[3, 4],
        &device,
    );

    // Indices selecting rows: [[0], [2]] - partial coordinates
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2], &[2, 1], &device);

    let result = client.gather_nd(&input, &indices).unwrap();

    // Output shape: indices.shape[:-1] + input.shape[1:] = [2] + [4] = [2, 4]
    assert_eq!(result.shape(), &[2, 4]);
    let data: Vec<f32> = result.to_vec();
    // input[0,:] = [0,1,2,3], input[2,:] = [8,9,10,11]
    assert_eq!(data, [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn test_gather_nd_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0], &[5], &device);

    // Indices: [[1], [3], [0]]
    let indices = Tensor::<CpuRuntime>::from_slice(&[1i64, 3, 0], &[3, 1], &device);

    let result = client.gather_nd(&input, &indices).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [20.0, 40.0, 10.0]);
}

// ============================================================================
// Bincount Tests
// ============================================================================

#[test]
fn test_bincount_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 1, 3, 2, 1, 3], &[7], &device);

    let result = client.bincount(&input, None, 0).unwrap();

    // max value is 3, so output length is 4
    assert_eq!(result.shape(), &[4]);
    let data: Vec<i64> = result.to_vec();
    // 0 appears 1 time, 1 appears 3 times, 2 appears 1 time, 3 appears 2 times
    assert_eq!(data, [1, 3, 1, 2]);
}

#[test]
fn test_bincount_with_minlength() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 1], &[3], &device);

    let result = client.bincount(&input, None, 5).unwrap();

    // minlength is 5, so output has at least 5 elements
    assert_eq!(result.shape(), &[5]);
    let data: Vec<i64> = result.to_vec();
    assert_eq!(data, [1, 2, 0, 0, 0]);
}

#[test]
fn test_bincount_with_weights() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 1, 3, 2, 1, 3], &[7], &device);
    let weights =
        Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.0, 1.5, 2.0, 1.0, 0.5, 3.0], &[7], &device);

    let result = client.bincount(&input, Some(&weights), 0).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    // bin 0: 0.5
    // bin 1: 1.0 + 1.5 + 0.5 = 3.0
    // bin 2: 1.0
    // bin 3: 2.0 + 3.0 = 5.0
    assert!((data[0] - 0.5).abs() < 1e-6);
    assert!((data[1] - 3.0).abs() < 1e-6);
    assert!((data[2] - 1.0).abs() < 1e-6);
    assert!((data[3] - 5.0).abs() < 1e-6);
}

#[test]
fn test_bincount_i32_input() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[0i32, 2, 2, 1], &[4], &device);

    let result = client.bincount(&input, None, 0).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<i64> = result.to_vec();
    assert_eq!(data, [1, 1, 2]);
}

#[test]
fn test_bincount_empty_bins() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input has gaps (no 1 or 3)
    let input = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 4, 2], &[4], &device);

    let result = client.bincount(&input, None, 0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<i64> = result.to_vec();
    assert_eq!(data, [1, 0, 2, 0, 1]);
}
