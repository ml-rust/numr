//! Integration tests for index operations (embedding_lookup, gather, scatter, index_select)
//!
//! Tests verify correctness across:
//! - Different dtypes (f32, f64, i32)
//! - Various embedding dimensions
//! - Boundary conditions
//! - Edge cases (single element, out of bounds handling)

use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Embedding Lookup Tests
// ============================================================================

#[test]
fn test_embedding_lookup_basic_f32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Embedding table: 4 words, embedding dim 3
    // vocab_size=4, embedding_dim=3
    let embeddings = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, 3.0, // word 0
            4.0, 5.0, 6.0, // word 1
            7.0, 8.0, 9.0, // word 2
            10.0, 11.0, 12.0, // word 3
        ],
        &[4, 3],
        &device,
    );

    // Look up words at indices [1, 3, 0]
    let indices = Tensor::<CpuRuntime>::from_slice(&[1i64, 3, 0], &[3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(
        data,
        [
            4.0, 5.0, 6.0, // word 1
            10.0, 11.0, 12.0, // word 3
            1.0, 2.0, 3.0, // word 0
        ]
    );
}

#[test]
fn test_embedding_lookup_basic_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Embedding table: 3 words, embedding dim 2
    let embeddings = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f64, 2.0, // word 0
            3.0, 4.0, // word 1
            5.0, 6.0, // word 2
        ],
        &[3, 2],
        &device,
    );

    let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 0, 1], &[3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f64> = result.to_vec();
    assert_eq!(
        data,
        [
            5.0, 6.0, // word 2
            1.0, 2.0, // word 0
            3.0, 4.0, // word 1
        ]
    );
}

#[test]
fn test_embedding_lookup_basic_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Integer embedding table
    let embeddings = Tensor::<CpuRuntime>::from_slice(
        &[
            10i32, 20, // word 0
            30, 40, // word 1
            50, 60, // word 2
        ],
        &[3, 2],
        &device,
    );

    let indices = Tensor::<CpuRuntime>::from_slice(&[1i64, 2], &[2], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, [30, 40, 50, 60]);
}

#[test]
fn test_embedding_lookup_single_index() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let embeddings =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    // Single index lookup
    let indices = Tensor::<CpuRuntime>::from_slice(&[1i64], &[1], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[1, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [4.0, 5.0, 6.0]);
}

#[test]
fn test_embedding_lookup_repeated_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let embeddings = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    // Repeated indices
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 1, 1, 0], &[5], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[5, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(
        data,
        [
            1.0, 2.0, // word 0
            1.0, 2.0, // word 0
            3.0, 4.0, // word 1
            3.0, 4.0, // word 1
            1.0, 2.0, // word 0
        ]
    );
}

#[test]
fn test_embedding_lookup_2d_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Embedding table: 4 words, embedding dim 2
    let embeddings = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, // word 0
            3.0, 4.0, // word 1
            5.0, 6.0, // word 2
            7.0, 8.0, // word 3
        ],
        &[4, 2],
        &device,
    );

    // 2D indices: shape [2, 3]
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 0, 1], &[2, 3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    // Output shape should be [2, 3, 2]
    assert_eq!(result.shape(), &[2, 3, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(
        data,
        [
            1.0, 2.0, // [0,0]: word 0
            3.0, 4.0, // [0,1]: word 1
            5.0, 6.0, // [0,2]: word 2
            7.0, 8.0, // [1,0]: word 3
            1.0, 2.0, // [1,1]: word 0
            3.0, 4.0, // [1,2]: word 1
        ]
    );
}

#[test]
fn test_embedding_lookup_large_embedding_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Large embedding dim (common in real models: 256, 512, 768, etc.)
    let embedding_dim = 64;
    let vocab_size = 10;

    // Create embeddings where each row i is filled with value (i+1)*0.1
    let mut emb_data = Vec::with_capacity(vocab_size * embedding_dim);
    for i in 0..vocab_size {
        for _ in 0..embedding_dim {
            emb_data.push((i + 1) as f32 * 0.1);
        }
    }

    let embeddings =
        Tensor::<CpuRuntime>::from_slice(&emb_data, &[vocab_size, embedding_dim], &device);

    let indices = Tensor::<CpuRuntime>::from_slice(&[5i64, 2, 9], &[3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[3, embedding_dim]);
    let data: Vec<f32> = result.to_vec();

    // Verify first element of each row
    assert!((data[0] - 0.6).abs() < 1e-6); // word 5: 0.6
    assert!((data[embedding_dim] - 0.3).abs() < 1e-6); // word 2: 0.3
    assert!((data[2 * embedding_dim] - 1.0).abs() < 1e-6); // word 9: 1.0
}

#[test]
fn test_embedding_lookup_all_same_index() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let embeddings =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    // All indices are the same
    let indices = Tensor::<CpuRuntime>::from_slice(&[1i64, 1, 1, 1], &[4], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[4, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
}

#[test]
fn test_embedding_lookup_sequential_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let embeddings = Tensor::<CpuRuntime>::from_slice(
        &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[4, 2],
        &device,
    );

    // Sequential indices (effectively full table lookup)
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[4], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[4, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_embedding_lookup_reverse_indices() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let embeddings =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

    // Reverse order indices
    let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 1, 0], &[3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
}

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
// Error Cases
// ============================================================================

#[test]
fn test_embedding_lookup_non_2d_embeddings() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 1D embeddings should fail
    let embeddings = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

    let result = client.embedding_lookup(&embeddings, &indices);
    assert!(result.is_err());
}

#[test]
fn test_embedding_lookup_wrong_index_dtype() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let embeddings = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

    // Float indices should fail
    let indices = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device);

    let result = client.embedding_lookup(&embeddings, &indices);
    assert!(result.is_err());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_embedding_lookup_min_vocab() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Minimum vocab size: 1 word
    let embeddings = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0], &[3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_embedding_lookup_min_embedding_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Minimum embedding dim: 1
    let embeddings = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device);
    let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 0, 1], &[3], &device);

    let result = client.embedding_lookup(&embeddings, &indices).unwrap();

    assert_eq!(result.shape(), &[3, 1]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [3.0, 1.0, 2.0]);
}
