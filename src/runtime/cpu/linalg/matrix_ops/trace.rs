//! Trace: sum of diagonal elements

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Trace: sum of diagonal elements
pub fn trace_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => trace_typed::<f32>(client, &a, m, n),
        DType::F64 => trace_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn trace_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut trace = T::zero();
    for i in 0..k {
        trace = trace + a_data[i * n + i];
    }

    Tensor::<CpuRuntime>::from_slice(&[trace], &[], device)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::algorithm::LinearAlgebraAlgorithms;

    #[test]
    fn test_trace() {
        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]]
        // trace = 1 + 4 = 5
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let tr = client.trace(&a).unwrap();
        let tr_val: Vec<f32> = tr.to_vec();

        assert!((tr_val[0] - 5.0).abs() < 1e-5);
    }
}
