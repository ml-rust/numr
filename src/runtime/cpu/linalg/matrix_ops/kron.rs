//! Kronecker product: A ⊗ B

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Kronecker product: A ⊗ B
///
/// For A of shape [m_a, n_a] and B of shape [m_b, n_b],
/// produces output of shape [m_a * m_b, n_a * n_b].
///
/// (A ⊗ B)[i*m_b + k, j*n_b + l] = A[i, j] * B[k, l]
pub fn kron_impl(
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
    let (m_a, n_a) = validate_matrix_2d(a.shape())?;
    let (m_b, n_b) = validate_matrix_2d(b.shape())?;

    let result = match a.dtype() {
        DType::F32 => kron_typed::<f32>(client, &a, &b, m_a, n_a, m_b, n_b),
        DType::F64 => kron_typed::<f64>(client, &a, &b, m_a, n_a, m_b, n_b),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn kron_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    m_a: usize,
    n_a: usize,
    m_b: usize,
    n_b: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();
    let b_data: Vec<T> = b.to_vec();

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let mut out: Vec<T> = vec![T::zero(); m_out * n_out];

    // Compute Kronecker product
    // out[i_a * m_b + i_b, j_a * n_b + j_b] = a[i_a, j_a] * b[i_b, j_b]
    for i_a in 0..m_a {
        for j_a in 0..n_a {
            let a_val = a_data[i_a * n_a + j_a];
            for i_b in 0..m_b {
                for j_b in 0..n_b {
                    let i_out = i_a * m_b + i_b;
                    let j_out = j_a * n_b + j_b;
                    out[i_out * n_out + j_out] = a_val * b_data[i_b * n_b + j_b];
                }
            }
        }
    }

    Tensor::<CpuRuntime>::from_slice(&out, &[m_out, n_out], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_kron_2x2_identity() {
        let client = create_client();
        let device = client.device();

        // I₂ ⊗ I₂ = I₄
        let i2 =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();
        let kron = client.kron(&i2, &i2).unwrap();

        assert_eq!(kron.shape(), &[4, 4]);

        let data: Vec<f32> = kron.to_vec();
        // Should be 4x4 identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (data[i * 4 + j] - expected).abs() < 1e-5,
                    "kron[{},{}] = {} expected {}",
                    i,
                    j,
                    data[i * 4 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_kron_2x2_simple() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
        // A ⊗ B should be 4x4
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 5.0, 6.0, 7.0], &[2, 2], device).unwrap();

        let kron = client.kron(&a, &b).unwrap();
        assert_eq!(kron.shape(), &[4, 4]);

        let data: Vec<f32> = kron.to_vec();

        // Expected result:
        // [[1*0, 1*5, 2*0, 2*5],     [[0,  5,  0, 10],
        //  [1*6, 1*7, 2*6, 2*7],  =   [6,  7, 12, 14],
        //  [3*0, 3*5, 4*0, 4*5],      [0, 15,  0, 20],
        //  [3*6, 3*7, 4*6, 4*7]]      [18, 21, 24, 28]]
        #[rustfmt::skip]
        let expected = [
            0.0, 5.0, 0.0, 10.0,
            6.0, 7.0, 12.0, 14.0,
            0.0, 15.0, 0.0, 20.0,
            18.0, 21.0, 24.0, 28.0,
        ];

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
    fn test_kron_scalar_property() {
        let client = create_client();
        let device = client.device();

        // 1x1 ⊗ A = scalar * A
        let scalar = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1, 1], device).unwrap();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let kron = client.kron(&scalar, &a).unwrap();
        assert_eq!(kron.shape(), &[2, 2]);

        let data: Vec<f32> = kron.to_vec();
        let expected = [3.0f32, 6.0, 9.0, 12.0]; // 3 * A

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_kron_rectangular() {
        let client = create_client();
        let device = client.device();

        // 2x3 ⊗ 3x2 = 6x6
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                .unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], device)
                .unwrap();

        let kron = client.kron(&a, &b).unwrap();
        assert_eq!(kron.shape(), &[6, 6]);

        // Verify a few elements manually
        // kron[0,0] = a[0,0] * b[0,0] = 1 * 1 = 1
        // kron[0,1] = a[0,0] * b[0,1] = 1 * 2 = 2
        // kron[3,0] = a[1,0] * b[0,0] = 4 * 1 = 4
        let data: Vec<f32> = kron.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-5, "kron[0,0]");
        assert!((data[1] - 2.0).abs() < 1e-5, "kron[0,1]");
        assert!((data[3 * 6 + 0] - 4.0).abs() < 1e-5, "kron[3,0]");
    }

    #[test]
    fn test_kron_f64() {
        let client = create_client();
        let device = client.device();

        // Test with F64
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

        let kron = client.kron(&a, &b).unwrap();
        assert_eq!(kron.shape(), &[4, 4]);

        let data: Vec<f64> = kron.to_vec();

        // kron[0,0] = 1*5 = 5, kron[0,1] = 1*6 = 6
        // kron[1,0] = 1*7 = 7, kron[1,1] = 1*8 = 8
        assert!((data[0] - 5.0).abs() < 1e-10);
        assert!((data[1] - 6.0).abs() < 1e-10);
        assert!((data[4] - 7.0).abs() < 1e-10);
        assert!((data[5] - 8.0).abs() < 1e-10);
    }
}
