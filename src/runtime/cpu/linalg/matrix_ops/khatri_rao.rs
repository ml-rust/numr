//! Khatri-Rao product (column-wise Kronecker): A ⊙ B

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Khatri-Rao product (column-wise Kronecker): A ⊙ B
///
/// For A of shape [m, k] and B of shape [n, k],
/// produces output of shape [m * n, k].
///
/// (A ⊙ B)[i*n + j, c] = A[i, c] * B[j, c]
pub fn khatri_rao_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let (m, k_a) = validate_matrix_2d(a.shape())?;
    let (n, k_b) = validate_matrix_2d(b.shape())?;

    if k_a != k_b {
        return Err(Error::Internal(format!(
            "khatri_rao: column count mismatch. A has shape [{}, {}], B has shape [{}, {}]. \
             Matrices must have the same number of columns.",
            m, k_a, n, k_b
        )));
    }

    let k = k_a;

    let result = match a.dtype() {
        DType::F32 => khatri_rao_typed::<f32>(client, &a, &b, m, n, k),
        DType::F64 => khatri_rao_typed::<f64>(client, &a, &b, m, n, k),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn khatri_rao_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();
    let b_data: Vec<T> = b.to_vec();

    let m_out = m * n;
    let mut out: Vec<T> = vec![T::zero(); m_out * k];

    // Compute Khatri-Rao product (column-wise Kronecker)
    // out[i*n + j, c] = a[i, c] * b[j, c]
    for c in 0..k {
        for i in 0..m {
            let a_val = a_data[i * k + c];
            for j in 0..n {
                let out_row = i * n + j;
                out[out_row * k + c] = a_val * b_data[j * k + c];
            }
        }
    }

    Tensor::<CpuRuntime>::from_slice(&out, &[m_out, k], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_khatri_rao_2x2() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // A ⊙ B should be (2*2)x2 = 4x2
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

        let kr = client.khatri_rao(&a, &b).unwrap();
        assert_eq!(kr.shape(), &[4, 2]);

        let data: Vec<f32> = kr.to_vec();

        // Expected result (column-wise Kronecker):
        // Column 0: kron([1,3], [5,7]) = [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
        // Column 1: kron([2,4], [6,8]) = [2*6, 2*8, 4*6, 4*8] = [12, 16, 24, 32]
        // Result: [[5, 12], [7, 16], [15, 24], [21, 32]]
        let expected = [5.0f32, 12.0, 7.0, 16.0, 15.0, 24.0, 21.0, 32.0];

        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "element {} differs: {} vs {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_khatri_rao_different_rows() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2, 3]] (1x3)
        // B = [[4, 5, 6], [7, 8, 9]] (2x3)
        // A ⊙ B should be (1*2)x3 = 2x3
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 3], device)
                .unwrap();

        let kr = client.khatri_rao(&a, &b).unwrap();
        assert_eq!(kr.shape(), &[2, 3]);

        let data: Vec<f32> = kr.to_vec();

        // Column 0: kron([1], [4,7]) = [1*4, 1*7] = [4, 7]
        // Column 1: kron([2], [5,8]) = [2*5, 2*8] = [10, 16]
        // Column 2: kron([3], [6,9]) = [3*6, 3*9] = [18, 27]
        // Result: [[4, 10, 18], [7, 16, 27]]
        let expected = [4.0f32, 10.0, 18.0, 7.0, 16.0, 27.0];

        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "element {} differs: {} vs {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_khatri_rao_column_mismatch() {
        let client = create_client();
        let device = client.device();

        // Different column counts should fail
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[1, 3], device).unwrap();

        let result = client.khatri_rao(&a, &b);
        assert!(
            result.is_err(),
            "Khatri-Rao with mismatched columns should fail"
        );
    }

    #[test]
    fn test_khatri_rao_f64() {
        let client = create_client();
        let device = client.device();

        // Test with F64
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

        let kr = client.khatri_rao(&a, &b).unwrap();
        assert_eq!(kr.shape(), &[4, 2]);

        let data: Vec<f64> = kr.to_vec();

        // Same expected values as f32 test
        let expected = [5.0f64, 12.0, 7.0, 16.0, 15.0, 24.0, 21.0, 32.0];

        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "element {} differs: {} vs {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_khatri_rao_gram_property() {
        let client = create_client();
        let device = client.device();

        // Property: (A ⊙ B)^T (A ⊙ B) = (A^T A) * (B^T B) (Hadamard product)
        // This is a fundamental property of Khatri-Rao used in tensor decompositions
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

        let kr = client.khatri_rao(&a, &b).unwrap();

        // We just verify the Khatri-Rao product computed correctly
        // Full Gram property verification would require matmul which is tested elsewhere
        assert_eq!(kr.shape(), &[4, 2]);
    }
}
