//! Shape operations on dual tensors

use crate::autograd::DualTensor;
use crate::error::Result;
use crate::runtime::Runtime;

/// Dual reshape: reshape(a, ȧ) = (reshape(a), reshape(ȧ))
pub fn dual_reshape<R>(a: &DualTensor<R>, shape: &[usize]) -> Result<DualTensor<R>>
where
    R: Runtime,
{
    let primal = a.primal().reshape(shape)?;

    let tangent = match a.tangent() {
        Some(at) => Some(at.reshape(shape)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual transpose: transpose(a, ȧ) = (transpose(a), transpose(ȧ))
pub fn dual_transpose<R>(a: &DualTensor<R>, dim0: isize, dim1: isize) -> Result<DualTensor<R>>
where
    R: Runtime,
{
    let primal = a.primal().transpose(dim0, dim1)?;

    let tangent = match a.tangent() {
        Some(at) => Some(at.transpose(dim0, dim1)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual narrow (slice): narrow(a, ȧ) = (narrow(a), narrow(ȧ))
pub fn dual_narrow<R>(
    a: &DualTensor<R>,
    dim: isize,
    start: usize,
    length: usize,
) -> Result<DualTensor<R>>
where
    R: Runtime,
{
    let primal = a.primal().narrow(dim, start, length)?;

    let tangent = match a.tangent() {
        Some(at) => Some(at.narrow(dim, start, length)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual unsqueeze: unsqueeze(a, ȧ) = (unsqueeze(a), unsqueeze(ȧ))
pub fn dual_unsqueeze<R>(a: &DualTensor<R>, dim: isize) -> Result<DualTensor<R>>
where
    R: Runtime,
{
    let primal = a.primal().unsqueeze(dim)?;

    let tangent = match a.tangent() {
        Some(at) => Some(at.unsqueeze(dim)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual squeeze: squeeze(a, ȧ) = (squeeze(a), squeeze(ȧ))
pub fn dual_squeeze<R>(a: &DualTensor<R>, dim: Option<isize>) -> DualTensor<R>
where
    R: Runtime,
{
    let primal = a.primal().squeeze(dim);
    let tangent = a.tangent().map(|at| at.squeeze(dim));
    DualTensor::new(primal, tangent)
}

/// Dual permute: permute(a, ȧ) = (permute(a), permute(ȧ))
pub fn dual_permute<R>(a: &DualTensor<R>, dims: &[usize]) -> Result<DualTensor<R>>
where
    R: Runtime,
{
    let primal = a.primal().permute(dims)?;

    let tangent = match a.tangent() {
        Some(at) => Some(at.permute(dims)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual contiguous: contiguous(a, ȧ) = (contiguous(a), contiguous(ȧ))
pub fn dual_contiguous<R>(a: &DualTensor<R>) -> DualTensor<R>
where
    R: Runtime,
{
    let primal = a.primal().contiguous();
    let tangent = a.tangent().map(|at| at.contiguous());
    DualTensor::new(primal, tangent)
}
