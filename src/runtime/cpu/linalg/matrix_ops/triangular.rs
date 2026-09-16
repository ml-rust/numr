//! Upper/lower triangular extraction

use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::validate_matrix_2d;
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Upper triangular part of a matrix
///
/// Supports all numeric dtypes (not just F32/F64).
pub fn triu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    use crate::runtime::cpu::helpers::dispatch_dtype;
    dispatch_dtype!(dtype, T => {
        triu_typed::<T>(client, a, m, n, diagonal)
    }, "triu")
}

fn triu_typed<T: Element>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    // Single allocation: clone input, then zero out the lower triangle in-place
    let mut data: Vec<T> = a.to_vec();

    for row in 0..m {
        // Zero columns below the diagonal: col < row + diagonal
        let threshold = (row as i64 + diagonal).max(0) as usize;
        let end = threshold.min(n);
        for col in 0..end {
            data[row * n + col] = T::zero();
        }
    }

    Tensor::<CpuRuntime>::from_slice(&data, &[m, n], device)
}

/// Lower triangular part of a matrix
///
/// Supports all numeric dtypes (not just F32/F64).
pub fn tril_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    use crate::runtime::cpu::helpers::dispatch_dtype;
    dispatch_dtype!(dtype, T => {
        tril_typed::<T>(client, a, m, n, diagonal)
    }, "tril")
}

fn tril_typed<T: Element>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    // Single allocation: clone input, then zero out the upper triangle in-place
    let mut data: Vec<T> = a.to_vec();

    for row in 0..m {
        // Zero columns above the diagonal: col > row + diagonal
        let threshold = (row as i64 + diagonal + 1).max(0) as usize;
        let start = threshold.min(n);
        for col in start..n {
            data[row * n + col] = T::zero();
        }
    }

    Tensor::<CpuRuntime>::from_slice(&data, &[m, n], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_triu_3x3_default_diagonal() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.triu(&a, 0).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Expected: [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
        assert_eq!(data, [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn test_triu_3x3_diagonal_1() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.triu(&a, 1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Expected: [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
        assert_eq!(data, [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_triu_3x3_negative_diagonal() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.triu(&a, -1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Expected: [[1, 2, 3], [4, 5, 6], [0, 8, 9]]
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]);
    }

    #[test]
    fn test_tril_3x3_default_diagonal() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.tril(&a, 0).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Expected: [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
        assert_eq!(data, [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_tril_3x3_diagonal_1() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.tril(&a, 1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Expected: [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
        assert_eq!(data, [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_triu_rectangular_2x4() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2, 3, 4], [5, 6, 7, 8]]
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            device,
        )
        .unwrap();

        let result = client.triu(&a, 0).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Expected: [[1, 2, 3, 4], [0, 6, 7, 8]]
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_triu_i32() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3], device)
            .unwrap();

        let result = client.triu(&a, 0).unwrap();
        let data: Vec<i32> = result.to_vec();

        assert_eq!(data, [1, 2, 3, 0, 5, 6, 0, 0, 9]);
    }

    #[test]
    fn test_triu_f64() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            device,
        )
        .unwrap();

        let result = client.triu(&a, 0).unwrap();
        let data: Vec<f64> = result.to_vec();

        assert_eq!(data, [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }
}
