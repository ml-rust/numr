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

// ============================================================================
// Gather 2D Tests
// ============================================================================

#[test]
fn test_gather_2d_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 3x4 matrix (row-major):
    // [[0, 1, 2, 3],
    //  [4, 5, 6, 7],
    //  [8, 9, 10, 11]]
    let input = Tensor::<CpuRuntime>::from_slice(
        &[
            0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ],
        &[3, 4],
        &device,
    );

    // Select: (0,0), (1,2), (2,3)
    let rows = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 3], &[3], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<f32> = result.to_vec();
    // (0,0) -> 0.0, (1,2) -> 6.0, (2,3) -> 11.0
    assert_eq!(data, [0.0, 6.0, 11.0]);
}

#[test]
fn test_gather_2d_repeated_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2x3 matrix:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    // Repeated indices: (0,1), (0,1), (1,0), (1,0)
    let rows = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 1, 1], &[4], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[1i64, 1, 0, 0], &[4], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    // (0,1) -> 2.0, (0,1) -> 2.0, (1,0) -> 4.0, (1,0) -> 4.0
    assert_eq!(data, [2.0, 2.0, 4.0, 4.0]);
}

#[test]
fn test_gather_2d_single_element() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    let rows = Tensor::<CpuRuntime>::from_slice(&[1i64], &[1], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[2i64], &[1], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[1]);
    let data: Vec<f32> = result.to_vec();
    // (1,2) -> 6.0
    assert_eq!(data, [6.0]);
}

#[test]
fn test_gather_2d_different_positions() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    // Test different positions than other tests
    let rows = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[2i64, 0], &[2], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f32> = result.to_vec();
    // (0,2) -> 3.0, (1,0) -> 4.0
    assert_eq!(data, [3.0, 4.0]);
}

#[test]
fn test_gather_2d_diagonal() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 4x4 matrix:
    // [[ 1,  2,  3,  4],
    //  [ 5,  6,  7,  8],
    //  [ 9, 10, 11, 12],
    //  [13, 14, 15, 16]]
    let input = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[4, 4],
        &device,
    );

    // Extract diagonal: (0,0), (1,1), (2,2), (3,3)
    let rows = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[4], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[4], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    // Diagonal: 1, 6, 11, 16
    assert_eq!(data, [1.0, 6.0, 11.0, 16.0]);
}

#[test]
fn test_gather_2d_i32_dtype() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3], &device);

    let rows = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[1i64, 2], &[2], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<i32> = result.to_vec();
    // (0,1) -> 2, (1,2) -> 6
    assert_eq!(data, [2, 6]);
}

#[test]
fn test_gather_2d_f64_dtype() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input =
        Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    let rows = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);
    let cols = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

    let result = client.gather_2d(&input, &rows, &cols).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data: Vec<f64> = result.to_vec();
    // (0,0) -> 1.0, (1,1) -> 5.0
    assert_eq!(data, [1.0, 5.0]);
}
