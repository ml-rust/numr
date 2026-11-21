//! CUDA `CorrectlyRoundedOps` on every CORE-MATH hard case, against the host functions.
//!
//! The host functions match MPFR in the sibling modules, so CUDA bits equal to host bits
//! are correctly rounded too.

use super::hard_cases::{load, load_fixture};
use super::oracle::Function;
use super::pow_hard_cases::load_pow;
use numr::algorithm::correctly_rounded as cr;
use numr::prelude::*;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

fn cuda() -> (CudaClient, CudaDevice) {
    assert!(
        numr::runtime::cuda::is_cuda_available(),
        "the cuda feature is enabled but no CUDA device is available"
    );
    let device = CudaDevice::new(0);
    (CudaRuntime::default_client(&device), device)
}

/// Panic with the first differing elements when `actual` differs from `host(input)`.
fn check_bits(label: &str, input: &[f64], actual: &[f64], host: impl Fn(f64) -> f64) {
    assert_eq!(input.len(), actual.len(), "{label}: output length");
    let mismatches: Vec<String> = input
        .iter()
        .zip(actual)
        .filter(|&(&x, y)| host(x).to_bits() != y.to_bits())
        .map(|(x, y)| {
            format!(
                "x={:#018x} cuda={:#018x} host={:#018x}",
                x.to_bits(),
                y.to_bits(),
                host(*x).to_bits()
            )
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{label}: {} of {} elements differ, first: {:?}",
        mismatches.len(),
        input.len(),
        &mismatches[..mismatches.len().min(8)]
    );
}

#[test]
fn test_cuda_equals_host_on_every_hard_case() -> Result<()> {
    let (client, device) = cuda();
    for function in Function::ALL {
        let data = load(function);
        let input = Tensor::<CudaRuntime>::from_slice(&data, &[data.len()], &device)?;
        let output = match function {
            Function::Exp => client.cr_exp(&input)?,
            Function::Log => client.cr_log(&input)?,
            Function::Log1p => client.cr_log1p(&input)?,
        };
        check_bits(
            &format!("{function:?} hard cases"),
            &data,
            &output.to_vec::<f64>(),
            |x| function.numr(x),
        );
    }
    Ok(())
}

#[test]
fn test_cuda_acos_equals_host_on_every_hard_case() -> Result<()> {
    let (client, device) = cuda();
    let data = load_fixture("acos");
    let input = Tensor::<CudaRuntime>::from_slice(&data, &[data.len()], &device)?;
    let output = client.cr_acos(&input)?;
    check_bits("acos hard cases", &data, &output.to_vec::<f64>(), cr::acos);
    Ok(())
}

#[test]
fn test_cuda_pow_equals_host_on_every_hard_case() -> Result<()> {
    let (client, device) = cuda();
    let pairs = load_pow();
    let xs: Vec<f64> = pairs.iter().map(|&(x, _)| x).collect();
    let all = Tensor::<CudaRuntime>::from_slice(&xs, &[xs.len()], &device)?;
    // The fixture lists pairs grouped by exponent; each group is one call on a slice.
    let mut start = 0usize;
    let mut groups = 0usize;
    for group in pairs.chunk_by(|a, b| a.1.to_bits() == b.1.to_bits()) {
        let y = group[0].1;
        let input = all.narrow(0, start, group.len())?;
        let output = client.cr_pow_scalar(&input, y)?.to_vec::<f64>();
        check_bits(
            &format!("pow hard cases, y={:#018x}", y.to_bits()),
            &xs[start..start + group.len()],
            &output,
            |x| cr::pow(x, y),
        );
        start += group.len();
        groups += 1;
    }
    assert_eq!(start, pairs.len());
    assert!(groups > 8, "pow fixture holds {groups} exponents");
    Ok(())
}
