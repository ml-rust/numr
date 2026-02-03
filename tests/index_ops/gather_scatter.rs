//! Index select, gather, and scatter tests

use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Index Select Tests
// ============================================================================

#[test]
fn test_index_select_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 0], &[2], &device);

    let result = client.index_select(&input, 0, &indices).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn test_index_select_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 0], &[2], &device);

    let result = client.index_select(&input, 1, &indices).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [3.0, 1.0, 6.0, 4.0]);
}

// ============================================================================
// Gather Tests
// ============================================================================

#[test]
fn test_gather_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    // Gather along dim 0
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1, 0], &[2, 2], &device);

    let result = client.gather(&input, 0, &indices).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    // [0,0]: input[0,0] = 1.0
    // [0,1]: input[2,1] = 6.0
    // [1,0]: input[1,0] = 3.0
    // [1,1]: input[0,1] = 2.0
    assert_eq!(data, [1.0, 6.0, 3.0, 2.0]);
}

#[test]
fn test_gather_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    // Gather along dim 1
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1, 0], &[2, 2], &device);

    let result = client.gather(&input, 1, &indices).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    // [0,0]: input[0,0] = 1.0
    // [0,1]: input[0,2] = 3.0
    // [1,0]: input[1,1] = 5.0
    // [1,1]: input[1,0] = 4.0
    assert_eq!(data, [1.0, 3.0, 5.0, 4.0]);
}

// ============================================================================
// Scatter Tests
// ============================================================================

#[test]
fn test_scatter_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[3, 2], &device);
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1, 0], &[2, 2], &device);
    let src = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    let result = client.scatter(&input, 0, &indices, &src).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f32> = result.to_vec();
    // Scatter places src values at positions specified by indices
    // [0,0] <- 1.0 (indices[0,0]=0)
    // [2,1] <- 2.0 (indices[0,1]=2)
    // [1,0] <- 3.0 (indices[1,0]=1)
    // [0,1] <- 4.0 (indices[1,1]=0)
    assert_eq!(data, [1.0, 4.0, 3.0, 0.0, 0.0, 2.0]);
}
