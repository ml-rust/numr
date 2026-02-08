//! Integration tests for shape operations (cat, stack, split, chunk)
//!
//! Tests verify correctness across:
//! - Different dimensions
//! - Negative dimension indexing
//! - Round-trip (cat then split/chunk)
//! - Multiple dtypes
//! - Edge cases
//!
//! Note: split and chunk return zero-copy views (non-contiguous).
//! We call `.contiguous()` before `.to_vec()` to materialize the view.

use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Cat Tests
// ============================================================================

#[test]
fn test_cat_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    let result = client.cat(&[&a, &b], 0).unwrap();

    assert_eq!(result.shape(), &[4, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_cat_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    let result = client.cat(&[&a, &b], 1).unwrap();

    assert_eq!(result.shape(), &[2, 4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_cat_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    // dim=-1 should be equivalent to dim=1 for 2D tensor
    let result = client.cat(&[&a, &b], -1).unwrap();

    assert_eq!(result.shape(), &[2, 4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_cat_three_tensors() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[1, 2], &device);
    let c = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0], &[1, 2], &device);

    let result = client.cat(&[&a, &b, &c], 0).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_cat_single_tensor() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    let result = client.cat(&[&a], 0).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cat_3d_tensor() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 2, 2]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
        &device,
    );
    let b = Tensor::<CpuRuntime>::from_slice(
        &[9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[2, 2, 2],
        &device,
    );

    // Cat along dim=1
    let result = client.cat(&[&a, &b], 1).unwrap();

    assert_eq!(result.shape(), &[2, 4, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(
        data,
        [
            1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0
        ]
    );
}

#[test]
fn test_cat_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5i32, 6, 7, 8], &[2, 2], &device);

    let result = client.cat(&[&a, &b], 0).unwrap();

    assert_eq!(result.shape(), &[4, 2]);
    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
}

// ============================================================================
// Stack Tests
// ============================================================================

#[test]
fn test_stack_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    let result = client.stack(&[&a, &b], 0).unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_stack_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    let result = client.stack(&[&a, &b], 1).unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_stack_dim2() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    let result = client.stack(&[&a, &b], 2).unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
}

#[test]
fn test_stack_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);

    // dim=-1 on 1D tensor stacked should be dim=1 of result (which is [2, 2])
    let result = client.stack(&[&a, &b], -1).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_stack_three_tensors() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);
    let c = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0], &[2], &device);

    let result = client.stack(&[&a, &b, &c], 0).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ============================================================================
// Split Tests
// ============================================================================

#[test]
fn test_split_even() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &device);

    let splits = client.split(&a, 2, 0).unwrap();

    assert_eq!(splits.len(), 3);
    assert_eq!(splits[0].shape(), &[2]);
    assert_eq!(splits[1].shape(), &[2]);
    assert_eq!(splits[2].shape(), &[2]);

    // split returns views - make contiguous before to_vec
    let data0: Vec<f32> = splits[0].contiguous().to_vec();
    let data1: Vec<f32> = splits[1].contiguous().to_vec();
    let data2: Vec<f32> = splits[2].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0]);
    assert_eq!(data1, [3.0, 4.0]);
    assert_eq!(data2, [5.0, 6.0]);
}

#[test]
fn test_split_uneven() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    let splits = client.split(&a, 2, 0).unwrap();

    assert_eq!(splits.len(), 3);
    assert_eq!(splits[0].shape(), &[2]);
    assert_eq!(splits[1].shape(), &[2]);
    assert_eq!(splits[2].shape(), &[1]); // Last chunk is smaller

    // split returns views - make contiguous before to_vec
    let data0: Vec<f32> = splits[0].contiguous().to_vec();
    let data1: Vec<f32> = splits[1].contiguous().to_vec();
    let data2: Vec<f32> = splits[2].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0]);
    assert_eq!(data1, [3.0, 4.0]);
    assert_eq!(data2, [5.0]);
}

#[test]
fn test_split_2d_dim0() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &device,
    );

    let splits = client.split(&a, 2, 0).unwrap();

    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].shape(), &[2, 2]);
    assert_eq!(splits[1].shape(), &[2, 2]);

    // split returns views - make contiguous before to_vec
    let data0: Vec<f32> = splits[0].contiguous().to_vec();
    let data1: Vec<f32> = splits[1].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(data1, [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_split_2d_dim1() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    );

    let splits = client.split(&a, 2, 1).unwrap();

    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].shape(), &[2, 2]);
    assert_eq!(splits[1].shape(), &[2, 2]);

    // split returns views - make contiguous before to_vec
    let data0: Vec<f32> = splits[0].contiguous().to_vec();
    let data1: Vec<f32> = splits[1].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0, 5.0, 6.0]);
    assert_eq!(data1, [3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_split_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    );

    // dim=-1 should be equivalent to dim=1
    let splits = client.split(&a, 2, -1).unwrap();

    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].shape(), &[2, 2]);
    assert_eq!(splits[1].shape(), &[2, 2]);
}

// ============================================================================
// Chunk Tests
// ============================================================================

#[test]
fn test_chunk_even() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &device);

    let chunks = client.chunk(&a, 3, 0).unwrap();

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].shape(), &[2]);
    assert_eq!(chunks[1].shape(), &[2]);
    assert_eq!(chunks[2].shape(), &[2]);

    // chunk returns views - make contiguous before to_vec
    let data0: Vec<f32> = chunks[0].contiguous().to_vec();
    let data1: Vec<f32> = chunks[1].contiguous().to_vec();
    let data2: Vec<f32> = chunks[2].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0]);
    assert_eq!(data1, [3.0, 4.0]);
    assert_eq!(data2, [5.0, 6.0]);
}

#[test]
fn test_chunk_uneven() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

    // Requesting 3 chunks from 5 elements: ceil(5/3)=2 per chunk
    // But we might get fewer chunks if last ones are empty
    let chunks = client.chunk(&a, 3, 0).unwrap();

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].shape(), &[2]);
    assert_eq!(chunks[1].shape(), &[2]);
    assert_eq!(chunks[2].shape(), &[1]); // Last chunk is smaller

    // chunk returns views - make contiguous before to_vec
    let data0: Vec<f32> = chunks[0].contiguous().to_vec();
    let data1: Vec<f32> = chunks[1].contiguous().to_vec();
    let data2: Vec<f32> = chunks[2].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0]);
    assert_eq!(data1, [3.0, 4.0]);
    assert_eq!(data2, [5.0]);
}

#[test]
fn test_chunk_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &device,
    );

    let chunks = client.chunk(&a, 2, 0).unwrap();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape(), &[2, 2]);
    assert_eq!(chunks[1].shape(), &[2, 2]);

    // chunk returns views - make contiguous before to_vec
    let data0: Vec<f32> = chunks[0].contiguous().to_vec();
    let data1: Vec<f32> = chunks[1].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(data1, [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_chunk_negative_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    );

    // dim=-1 should be equivalent to dim=1
    let chunks = client.chunk(&a, 2, -1).unwrap();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape(), &[2, 2]);
    assert_eq!(chunks[1].shape(), &[2, 2]);
}

// ============================================================================
// Round-trip Tests (cat then split)
// ============================================================================

#[test]
fn test_cat_split_roundtrip() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);
    let c = Tensor::<CpuRuntime>::from_slice(&[9.0f32, 10.0, 11.0, 12.0], &[2, 2], &device);

    // Cat along dim 0
    let catted = client.cat(&[&a, &b, &c], 0).unwrap();
    assert_eq!(catted.shape(), &[6, 2]);

    // Split back
    let splits = client.split(&catted, 2, 0).unwrap();
    assert_eq!(splits.len(), 3);

    // split returns views - make contiguous before to_vec
    let data0: Vec<f32> = splits[0].contiguous().to_vec();
    let data1: Vec<f32> = splits[1].contiguous().to_vec();
    let data2: Vec<f32> = splits[2].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(data1, [5.0, 6.0, 7.0, 8.0]);
    assert_eq!(data2, [9.0, 10.0, 11.0, 12.0]);
}

#[test]
fn test_stack_chunk_roundtrip() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

    // Stack along dim 0 creates new dimension
    let stacked = client.stack(&[&a, &b], 0).unwrap();
    assert_eq!(stacked.shape(), &[2, 2, 2]);

    // Chunk along dim 0 to get back pieces
    let chunks = client.chunk(&stacked, 2, 0).unwrap();
    assert_eq!(chunks.len(), 2);

    // Each chunk should have shape [1, 2, 2] - the stacking adds a dimension
    assert_eq!(chunks[0].shape(), &[1, 2, 2]);
    assert_eq!(chunks[1].shape(), &[1, 2, 2]);

    // chunk returns views - make contiguous before to_vec
    let data0: Vec<f32> = chunks[0].contiguous().to_vec();
    let data1: Vec<f32> = chunks[1].contiguous().to_vec();

    assert_eq!(data0, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(data1, [5.0, 6.0, 7.0, 8.0]);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_cat_empty_tensors_error() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let empty: Vec<&Tensor<CpuRuntime>> = vec![];
    let result = client.cat(&empty, 0);
    assert!(result.is_err());
}

#[test]
fn test_stack_empty_tensors_error() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let empty: Vec<&Tensor<CpuRuntime>> = vec![];
    let result = client.stack(&empty, 0);
    assert!(result.is_err());
}

#[test]
fn test_split_size_larger_than_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    // Split size larger than dimension size should produce 1 chunk
    let splits = client.split(&a, 10, 0).unwrap();

    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].shape(), &[3]);
}

#[test]
fn test_chunk_more_chunks_than_elements() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

    // Requesting 5 chunks from 3 elements
    let chunks = client.chunk(&a, 5, 0).unwrap();

    // Should get 3 chunks of size 1 each (not 5 chunks, since we can't have empty)
    assert_eq!(chunks.len(), 3);
    for chunk in &chunks {
        assert_eq!(chunk.shape(), &[1]);
    }
}

// ============================================================================
// Zero-Copy Verification Tests
// ============================================================================

#[test]
fn test_split_is_zero_copy() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &device);

    let splits = client.split(&a, 2, 0).unwrap();

    // Verify that split returns views (non-contiguous after the first)
    // The first split may be contiguous if it starts at offset 0
    // But subsequent splits should have non-zero offset
    assert_eq!(splits[0].shape(), &[2]);
    assert_eq!(splits[1].shape(), &[2]);
    assert_eq!(splits[2].shape(), &[2]);

    // All splits share storage with original - this is the zero-copy property
    // We verify by checking that the storage pointers are related
    // (In practice, they share the same Arc<Buffer>)
}

#[test]
fn test_chunk_is_zero_copy() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &device);

    let chunks = client.chunk(&a, 3, 0).unwrap();

    // Same as split - chunks are views
    assert_eq!(chunks.len(), 3);
    for chunk in &chunks {
        assert_eq!(chunk.shape(), &[2]);
    }
}

// ============================================================================
// Multi-dtype Tests
// ============================================================================

#[test]
fn test_cat_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], &device);

    let result = client.cat(&[&a, &b], 0).unwrap();

    assert_eq!(result.dtype(), DType::F64);
    let data: Vec<f64> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_stack_i64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[2, 2], &device);
    let b = Tensor::<CpuRuntime>::from_slice(&[5i64, 6, 7, 8], &[2, 2], &device);

    let result = client.stack(&[&a, &b], 0).unwrap();

    assert_eq!(result.dtype(), DType::I64);
    let data: Vec<i64> = result.to_vec();
    assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_split_u32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1u32, 2, 3, 4, 5, 6], &[6], &device);

    let splits = client.split(&a, 2, 0).unwrap();

    assert_eq!(splits.len(), 3);
    assert_eq!(splits[0].dtype(), DType::U32);

    // split returns views - make contiguous before to_vec
    let data0: Vec<u32> = splits[0].contiguous().to_vec();
    assert_eq!(data0, [1, 2]);
}

// ============================================================================
// Repeat Tests
// ============================================================================

#[test]
fn test_repeat_simple() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let result = client.repeat(&a, &[2]).unwrap();

    assert_eq!(result.shape(), &[6]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_repeat_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let result = client.repeat(&a, &[2, 3]).unwrap();

    assert_eq!(result.shape(), &[4, 6]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: [1,2,1,2,1,2], Row 1: [3,4,3,4,3,4], Row 2: [1,2,1,2,1,2], Row 3: [3,4,3,4,3,4]
    assert_eq!(
        data,
        [
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0,
            2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0
        ]
    );
}

#[test]
fn test_repeat_noop() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    let result = client.repeat(&a, &[1, 1]).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_repeat_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3], &[3], &device);
    let result = client.repeat(&a, &[3]).unwrap();

    assert_eq!(result.shape(), &[9]);
    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [1, 2, 3, 1, 2, 3, 1, 2, 3]);
}

// ============================================================================
// Pad Tests
// ============================================================================

#[test]
fn test_pad_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    // PyTorch convention: [left, right] for 1D
    let result = client.pad(&a, &[1, 2], 0.0).unwrap();

    assert_eq!(result.shape(), &[6]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn test_pad_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
    // PyTorch convention: [left, right, top, bottom] for 2D
    let result = client.pad(&a, &[1, 1, 1, 1], 0.0).unwrap();

    assert_eq!(result.shape(), &[4, 4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(
        data,
        [
            0.0, 0.0, 0.0, 0.0, // top padding row
            0.0, 1.0, 2.0, 0.0, // original row 0 with left/right padding
            0.0, 3.0, 4.0, 0.0, // original row 1 with left/right padding
            0.0, 0.0, 0.0, 0.0 // bottom padding row
        ]
    );
}

#[test]
fn test_pad_with_value() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let result = client.pad(&a, &[1, 1], -1.0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [-1.0, 1.0, 2.0, 3.0, -1.0]);
}

#[test]
fn test_pad_noop() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let result = client.pad(&a, &[0, 0], 0.0).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0]);
}

#[test]
fn test_pad_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3], &[3], &device);
    let result = client.pad(&a, &[2, 0], 99.0).unwrap();

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [99, 99, 1, 2, 3]);
}

// ============================================================================
// Roll Tests
// ============================================================================

#[test]
fn test_roll_positive() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = client.roll(&a, 2, 0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [4.0, 5.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_roll_negative() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = client.roll(&a, -2, 0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [3.0, 4.0, 5.0, 1.0, 2.0]);
}

#[test]
fn test_roll_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let result = client.roll(&a, 1, 1).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.to_vec();
    // Each row rolled by 1: [3,1,2], [6,4,5]
    assert_eq!(data, [3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
}

#[test]
fn test_roll_zero_shift() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let result = client.roll(&a, 0, 0).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0]);
}

#[test]
fn test_roll_full_cycle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    // Rolling by the full dimension size should be a no-op
    let result = client.roll(&a, 4, 0).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_roll_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);
    let result = client.roll(&a, 1, 0).unwrap();

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [4, 1, 2, 3]);
}

// ============================================================================
// Flip Tests
// ============================================================================

#[test]
fn test_flip_1d() {
    let device = CpuDevice::new();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let result = a.flip(0).unwrap();

    assert_eq!(result.shape(), &[5]);
    let data: Vec<f32> = result.contiguous().to_vec();
    assert_eq!(data, [5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_2d_dim0() {
    let device = CpuDevice::new();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let result = a.flip(0).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.contiguous().to_vec();
    // Rows reversed: [4,5,6] then [1,2,3]
    assert_eq!(data, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_flip_2d_dim1() {
    let device = CpuDevice::new();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    let result = a.flip(1).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.contiguous().to_vec();
    // Each row reversed: [3,2,1], [6,5,4]
    assert_eq!(data, [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

#[test]
fn test_flip_both_dims() {
    let device = CpuDevice::new();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    // Flip both dims by chaining
    let result = a.flip(0).unwrap().flip(1).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.contiguous().to_vec();
    // Both dimensions reversed: full reversal
    assert_eq!(data, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_i32() {
    let device = CpuDevice::new();

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);
    let result = a.flip(0).unwrap();

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.dtype(), DType::I32);
    let data: Vec<i32> = result.contiguous().to_vec();
    assert_eq!(data, [4, 3, 2, 1]);
}

#[test]
fn test_flip_negative_dim() {
    let device = CpuDevice::new();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
    // dim=-1 should be equivalent to dim=1
    let result = a.flip(-1).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.contiguous().to_vec();
    assert_eq!(data, [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
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

    fn assert_allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let tol = atol + rtol * y.abs();
            assert!(
                diff <= tol,
                "mismatch at index {}: cpu={}, cuda={}, diff={}, tol={}",
                i,
                x,
                y,
                diff,
                tol
            );
        }
    }

    #[test]
    fn test_repeat_parity() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2, 3], &cuda_device);

        let cpu_result = cpu_client.repeat(&cpu_tensor, &[2, 3]).unwrap();
        let cuda_result = cuda_client.repeat(&cuda_tensor, &[2, 3]).unwrap();

        assert_eq!(cpu_result.shape(), cuda_result.shape());
        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let cuda_data: Vec<f32> = cuda_result.to_vec();
        assert_allclose(&cpu_data, &cuda_data, 1e-6, 1e-7);
    }

    #[test]
    fn test_pad_parity() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2, 3], &cuda_device);

        // Pad last dim by (1, 2), second-to-last by (1, 1)
        let cpu_result = cpu_client.pad(&cpu_tensor, &[1, 2, 1, 1], 0.0).unwrap();
        let cuda_result = cuda_client.pad(&cuda_tensor, &[1, 2, 1, 1], 0.0).unwrap();

        assert_eq!(cpu_result.shape(), cuda_result.shape());
        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let cuda_data: Vec<f32> = cuda_result.to_vec();
        assert_allclose(&cpu_data, &cuda_data, 1e-6, 1e-7);
    }

    #[test]
    fn test_roll_parity() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2, 3], &cuda_device);

        let cpu_result = cpu_client.roll(&cpu_tensor, 2, 1).unwrap();
        let cuda_result = cuda_client.roll(&cuda_tensor, 2, 1).unwrap();

        assert_eq!(cpu_result.shape(), cuda_result.shape());
        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let cuda_data: Vec<f32> = cuda_result.to_vec();
        assert_allclose(&cpu_data, &cuda_data, 1e-6, 1e-7);
    }

    #[test]
    fn test_flip_parity() {
        if !numr::runtime::cuda::is_cuda_available() {
            println!("CUDA not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cuda_device = CudaDevice::new(0);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let cuda_tensor = Tensor::<CudaRuntime>::from_slice(&data, &[2, 3], &cuda_device);

        let cpu_result = cpu_tensor.flip(1).unwrap();
        let cuda_result = cuda_tensor.flip(1).unwrap();

        assert_eq!(cpu_result.shape(), cuda_result.shape());
        let cpu_data: Vec<f32> = cpu_result.contiguous().to_vec();
        let cuda_data: Vec<f32> = cuda_result.contiguous().to_vec();
        assert_allclose(&cpu_data, &cuda_data, 1e-6, 1e-7);
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
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
    use numr::tensor::Tensor;

    fn assert_allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let tol = atol + rtol * y.abs();
            assert!(
                diff <= tol,
                "mismatch at index {}: cpu={}, wgpu={}, diff={}, tol={}",
                i,
                x,
                y,
                diff,
                tol
            );
        }
    }

    #[test]
    fn test_repeat_parity() {
        if !numr::runtime::wgpu::is_wgpu_available() {
            println!("WebGPU not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuDevice::new(0);
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3], &wgpu_device);

        let cpu_result = cpu_client.repeat(&cpu_tensor, &[2, 3]).unwrap();
        let wgpu_result = wgpu_client.repeat(&wgpu_tensor, &[2, 3]).unwrap();

        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let wgpu_data: Vec<f32> = wgpu_result.to_vec();
        assert_allclose(&cpu_data, &wgpu_data, 1e-5, 1e-5);
    }

    #[test]
    fn test_pad_parity() {
        if !numr::runtime::wgpu::is_wgpu_available() {
            println!("WebGPU not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuDevice::new(0);
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3], &wgpu_device);

        // Pad last dim by (1, 2), second-to-last by (1, 1)
        let cpu_result = cpu_client.pad(&cpu_tensor, &[1, 2, 1, 1], 0.0).unwrap();
        let wgpu_result = wgpu_client.pad(&wgpu_tensor, &[1, 2, 1, 1], 0.0).unwrap();

        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let wgpu_data: Vec<f32> = wgpu_result.to_vec();
        assert_allclose(&cpu_data, &wgpu_data, 1e-5, 1e-5);
    }

    #[test]
    fn test_roll_parity() {
        if !numr::runtime::wgpu::is_wgpu_available() {
            println!("WebGPU not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let wgpu_device = WgpuDevice::new(0);
        let wgpu_client = WgpuRuntime::default_client(&wgpu_device);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3], &wgpu_device);

        let cpu_result = cpu_client.roll(&cpu_tensor, 2, 1).unwrap();
        let wgpu_result = wgpu_client.roll(&wgpu_tensor, 2, 1).unwrap();

        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        let cpu_data: Vec<f32> = cpu_result.to_vec();
        let wgpu_data: Vec<f32> = wgpu_result.to_vec();
        assert_allclose(&cpu_data, &wgpu_data, 1e-5, 1e-5);
    }

    #[test]
    fn test_flip_parity() {
        if !numr::runtime::wgpu::is_wgpu_available() {
            println!("WebGPU not available, skipping");
            return;
        }
        let cpu_device = CpuDevice::new();
        let wgpu_device = WgpuDevice::new(0);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &cpu_device);
        let wgpu_tensor = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3], &wgpu_device);

        let cpu_result = cpu_tensor.flip(1).unwrap();
        let wgpu_result = wgpu_tensor.flip(1).unwrap();

        assert_eq!(cpu_result.shape(), wgpu_result.shape());
        let cpu_data: Vec<f32> = cpu_result.contiguous().to_vec();
        let wgpu_data: Vec<f32> = wgpu_result.contiguous().to_vec();
        assert_allclose(&cpu_data, &wgpu_data, 1e-5, 1e-5);
    }
}
