//! Tensor operation adapters for special functions
//!
//! Provides `apply_unary`, `apply_binary`, and `apply_ternary` helpers that
//! apply scalar functions element-wise over tensors with proper error handling.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

use crate::runtime::cpu::{CpuDevice, CpuRuntime};

/// Apply a unary scalar function element-wise over a tensor.
pub fn apply_unary<F>(
    x: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64) -> f64,
{
    match x.dtype() {
        DType::F32 => {
            let data: Vec<f32> = x.to_vec();
            let result: Vec<f32> = data.iter().map(|&v| f(v as f64) as f32).collect();
            Ok(Tensor::from_slice(&result, x.shape(), device))
        }
        DType::F64 => {
            let data: Vec<f64> = x.to_vec();
            let result: Vec<f64> = data.iter().map(|&v| f(v)).collect();
            Ok(Tensor::from_slice(&result, x.shape(), device))
        }
        _ => unreachable!("dtype validated by caller"),
    }
}

/// Apply a binary scalar function element-wise over two tensors.
///
/// Both tensors must have matching shapes (broadcasting not supported).
pub fn apply_binary<F>(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64) -> f64,
{
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    match a.dtype() {
        DType::F32 => {
            let a_data: Vec<f32> = a.to_vec();
            let b_data: Vec<f32> = b.to_vec();
            let result: Vec<f32> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&av, &bv)| f(av as f64, bv as f64) as f32)
                .collect();
            Ok(Tensor::from_slice(&result, a.shape(), device))
        }
        DType::F64 => {
            let a_data: Vec<f64> = a.to_vec();
            let b_data: Vec<f64> = b.to_vec();
            let result: Vec<f64> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&av, &bv)| f(av, bv))
                .collect();
            Ok(Tensor::from_slice(&result, a.shape(), device))
        }
        _ => unreachable!("dtype validated by caller"),
    }
}

/// Apply a ternary scalar function element-wise over three tensors.
///
/// All tensors must have matching shapes (broadcasting not supported).
pub fn apply_ternary<F>(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    c: &Tensor<CpuRuntime>,
    device: &CpuDevice,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64, f64, f64) -> f64,
{
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    if a.shape() != c.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: c.shape().to_vec(),
        });
    }

    match a.dtype() {
        DType::F32 => {
            let a_data: Vec<f32> = a.to_vec();
            let b_data: Vec<f32> = b.to_vec();
            let c_data: Vec<f32> = c.to_vec();
            let result: Vec<f32> = a_data
                .iter()
                .zip(b_data.iter())
                .zip(c_data.iter())
                .map(|((&av, &bv), &cv)| f(av as f64, bv as f64, cv as f64) as f32)
                .collect();
            Ok(Tensor::from_slice(&result, a.shape(), device))
        }
        DType::F64 => {
            let a_data: Vec<f64> = a.to_vec();
            let b_data: Vec<f64> = b.to_vec();
            let c_data: Vec<f64> = c.to_vec();
            let result: Vec<f64> = a_data
                .iter()
                .zip(b_data.iter())
                .zip(c_data.iter())
                .map(|((&av, &bv), &cv)| f(av, bv, cv))
                .collect();
            Ok(Tensor::from_slice(&result, a.shape(), device))
        }
        _ => unreachable!("dtype validated by caller"),
    }
}
