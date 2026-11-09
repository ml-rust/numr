//! Exact backend parity inside the normal arithmetic envelope.

use super::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use super::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use super::helpers::with_wgpu_backend;
use crate::common::{assert_tensor_allclose_tol, create_cpu_client};
use numr::prelude::*;

fn outputs<R, C>(client: &C, device: &R::Device, dtype: DType) -> Result<Vec<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    C: TwoSumOps<R> + TypeConversionOps<R>,
{
    let small = if dtype == DType::F64 {
        2f64.powi(-54)
    } else {
        2f64.powi(-25)
    };
    let cases = [
        (0.1, 0.9),
        (1.0, small),
        (-1.0, -small),
        (1.0, -1.0),
        (1e20, 1.0),
    ];
    let a: Vec<_> = (0..65).map(|i| cases[i % cases.len()].0).collect();
    let b: Vec<_> = (0..65).map(|i| cases[i % cases.len()].1).collect();
    let make = |data: &[f64], shape: &[usize]| tensor_from_f64(data, shape, dtype, device, client);
    let (sum, residual) = client.two_sum(&make(&a, &[65])?, &make(&b, &[65])?)?;
    let storage = make(&[99.0, 0.1, 1.0, 99.0, -1.0, 0.1], &[2, 3])?;
    let view = storage.narrow(1, 1, 2)?.transpose(0, 1)?;
    let column = make(&[0.9, small], &[2, 1])?;
    let (view_sum, view_residual) = client.two_sum(&view, &column)?;
    let (scalar_sum, scalar_residual) = client.two_sum(&make(&[0.1], &[])?, &make(&[0.9], &[])?)?;
    let empty = Tensor::empty(&[0, 3], dtype, device)?;
    let (empty_sum, empty_residual) = client.two_sum(&empty, &make(&[1.0, 2.0, 3.0], &[1, 3])?)?;
    assert_eq!(empty_sum.shape(), &[0, 3]);
    assert_eq!(empty_residual.shape(), &[0, 3]);
    let invalid_shape = make(&[1.0, 2.0], &[2])?;
    assert!(client.two_sum(&empty, &invalid_shape).is_err());
    let invalid_dtype = Tensor::empty(&[0], DType::I32, device)?;
    assert!(matches!(
        client.two_sum(&invalid_dtype, &invalid_dtype),
        Err(Error::UnsupportedDType { .. })
    ));
    Ok(vec![
        sum,
        residual,
        view_sum,
        view_residual,
        scalar_sum,
        scalar_residual,
        empty_sum,
        empty_residual,
    ])
}

fn compare<R: Runtime<DType = DType>>(actual: &[Tensor<R>], expected: &[Tensor<CpuRuntime>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_tensor_allclose_tol(actual, expected, 0.0, 0.0, "two_sum exact normal parity");
    }
}

#[test]
fn two_sum_normal_arithmetic_matches_across_supported_backends() -> Result<()> {
    let (client, device) = create_cpu_client();
    for dtype in [DType::F32, DType::F64] {
        let expected = outputs(&client, &device, dtype)?;
        compare(&expected, &expected);
        #[cfg(feature = "cuda")]
        with_cuda_backend(|client, device| {
            let result = outputs(&client, &device, dtype).map(|actual| compare(&actual, &expected));
            assert!(result.is_ok(), "two_sum CUDA: {result:?}");
        });
        #[cfg(feature = "wgpu")]
        if dtype == DType::F32 {
            with_wgpu_backend(|client, device| {
                let result =
                    outputs(&client, &device, dtype).map(|actual| compare(&actual, &expected));
                assert!(result.is_ok(), "two_sum WebGPU: {result:?}");
                let input = Tensor::<WgpuRuntime>::empty(&[0], DType::F64, &device);
                let result = input.and_then(|input| client.two_sum(&input, &input));
                assert!(matches!(result, Err(Error::UnsupportedDType { .. })));
            });
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn two_sum_cuda_retains_f64_minimum_subnormal_residual() {
    with_cuda_backend(|client, device| {
        let run = || -> Result<()> {
            let a =
                Tensor::<CudaRuntime>::from_slice(&[f64::MAX, f64::MIN_POSITIVE], &[2], &device)?;
            let b = Tensor::<CudaRuntime>::from_slice(
                &[
                    f64::from_bits(1),
                    -f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1),
                ],
                &[2],
                &device,
            )?;
            let (sum, residual) = client.two_sum(&a, &b)?;
            assert_eq!(sum.try_to_vec::<f64>()?, vec![f64::MAX, f64::from_bits(1)]);
            assert_eq!(residual.try_to_vec::<f64>()?, vec![f64::from_bits(1), 0.0]);
            Ok(())
        };
        let result = run();
        assert!(result.is_ok(), "two_sum CUDA subnormal: {result:?}");
    });
}
