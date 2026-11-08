//! Magnitude parity across floating dtypes and tensor layouts.

use numr::prelude::*;

use super::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use super::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use super::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, assert_tensor_allclose_tol, create_cpu_client,
    parity_dtypes,
};

fn outputs<R, C>(client: &C, device: &R::Device, dtype: DType) -> Vec<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: HypotOps<R> + TypeConversionOps<R>,
{
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, device, client).unwrap()
    };
    let a = make(
        &[
            3.0,
            -5.0,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0,
        ],
        &[8],
    );
    let b = make(&[4.0, 12.0, -0.0, -0.0, 3.0, f64::NAN, 0.0, f64::NAN], &[8]);
    let values = client.hypot(&a, &b).unwrap();
    let reverse = client.hypot(&b, &a).unwrap();
    let zero_pairs = make(&[-0.0, 0.0], &[2]);
    let zeros = client.hypot(&zero_pairs, &zero_pairs).unwrap();
    let zeros = client.cast(&zeros, DType::F32).unwrap().to_vec::<f32>();
    assert!(zeros.into_iter().all(|value| value.to_bits() == 0));
    let a = make(&[99.0, 3.0, 5.0, 99.0, 8.0, 7.0], &[2, 3]);
    let view = a.narrow(1, 1, 2).unwrap().transpose(0, 1).unwrap();
    let column = make(&[4.0, 12.0], &[2, 1]);
    let broadcast = client.hypot(&view, &column).unwrap();
    assert_eq!(broadcast.shape(), &[2, 2]);
    let scalar = make(&[4.0], &[]);
    let scalar_output = client.hypot(&scalar, &scalar).unwrap();
    assert_eq!(scalar_output.shape(), &[] as &[usize]);
    let empty = Tensor::empty(&[0, 3], dtype, device).unwrap();
    let row = make(&[3.0, 4.0, 5.0], &[1, 3]);
    let empty_output = client.hypot(&empty, &row).unwrap();
    assert_eq!(empty_output.shape(), &[0, 3]);
    let mismatch = make(&[1.0, 2.0], &[2]);
    assert!(client.hypot(&empty, &mismatch).is_err());
    let integer = Tensor::empty(&[0], DType::I32, device).unwrap();
    assert!(matches!(
        client.hypot(&integer, &integer),
        Err(Error::UnsupportedDType { .. })
    ));
    let mismatch_dtype = Tensor::empty(&[0, 3], DType::I32, device).unwrap();
    assert!(client.hypot(&empty, &mismatch_dtype).is_err());
    vec![values, reverse, broadcast, scalar_output, empty_output]
}

fn compare<R: Runtime<DType = DType>>(
    actual: &[Tensor<R>],
    expected: &[Tensor<CpuRuntime>],
    dtype: DType,
) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert_tensor_allclose(actual, expected, dtype, "hypot parity");
    }
}

#[test]
fn hypot_all_float_dtypes() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let reference = outputs(&cpu_client, &cpu_device, dtype);
        compare(&reference, &reference, dtype);
        #[cfg(feature = "cuda")]
        if crate::common::is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                compare(&outputs(&client, &device, dtype), &reference, dtype);
            });
        }
        #[cfg(feature = "wgpu")]
        if crate::common::is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                compare(&outputs(&client, &device, dtype), &reference, dtype);
            });
        }
    }
}

fn extreme_output<R, C>(client: &C, device: &R::Device, dtype: DType) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: HypotOps<R> + TypeConversionOps<R>,
{
    let (large, small, tiny) = if dtype == DType::F64 {
        (f64::MAX / 2.0, 1e-300, f64::from_bits(1))
    } else {
        ((f32::MAX / 2.0) as f64, 1e-30, f32::from_bits(1) as f64)
    };
    let a = tensor_from_f64(
        &[large, small, tiny, tiny, -0.0],
        &[5],
        dtype,
        device,
        client,
    )
    .unwrap();
    let b = tensor_from_f64(
        &[large, small, tiny, 0.0, -0.0],
        &[5],
        dtype,
        device,
        client,
    )
    .unwrap();
    client.hypot(&a, &b).unwrap()
}

#[test]
fn hypot_extreme_scales() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in [DType::F32, DType::F64] {
        let reference = extreme_output(&cpu_client, &cpu_device, dtype);
        // Zero absolute tolerance detects flushed subnormals and underflowed squares.
        let rtol = if dtype == DType::F64 { 1e-14 } else { 1e-6 };
        #[cfg(feature = "cuda")]
        with_cuda_backend(|client, device| {
            let result = extreme_output(&client, &device, dtype);
            assert_tensor_allclose_tol(&result, &reference, rtol, 0.0, "hypot CUDA scales");
        });
        #[cfg(feature = "wgpu")]
        if crate::common::is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let result = extreme_output(&client, &device, dtype);
                assert_tensor_allclose_tol(&result, &reference, rtol, 0.0, "hypot WebGPU scales");
            });
        }
        assert_tensor_allclose_tol(&reference, &reference, rtol, 0.0, "hypot CPU scales");
    }
}

#[cfg(feature = "wgpu")]
fn subnormal_magnitude_bits(a: u32, b: u32) -> u32 {
    let sum = u64::from(a).pow(2) + u64::from(b).pow(2);
    let root = sum.isqrt();
    (root + u64::from(sum > root * root + root)) as u32
}

#[cfg(feature = "wgpu")]
#[test]
fn hypot_wgpu_subnormal_bits_and_normal_boundary() {
    with_wgpu_backend(|client, device| {
        let pairs = [
            (0u32, 0u32),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (5, 12),
            (0x3fffff, 0x400000),
            (0x7ffffe, 0x7fffff),
            (0x7fffff, 0x7fffff),
            (0x7fffff, 1),
        ];
        let a = pairs.map(|(a, _)| f32::from_bits(a));
        let b = pairs.map(|(_, b)| f32::from_bits(b));
        let a = Tensor::<WgpuRuntime>::from_slice(&a, &[pairs.len()], &device).unwrap();
        let b = Tensor::<WgpuRuntime>::from_slice(&b, &[pairs.len()], &device).unwrap();
        let output = client.hypot(&a, &b).unwrap().to_vec::<f32>();
        for (value, (a, b)) in output.into_iter().zip(pairs) {
            assert_eq!(
                value.to_bits(),
                subnormal_magnitude_bits(a, b),
                "hypot bits ({a:#x}, {b:#x})"
            );
        }
        let a = Tensor::<WgpuRuntime>::from_slice(&[f32::MIN_POSITIVE; 3], &[3], &device).unwrap();
        let b = Tensor::<WgpuRuntime>::from_slice(
            &[0.0, f32::from_bits(1), f32::from_bits(2)],
            &[3],
            &device,
        )
        .unwrap();
        let output = client.hypot(&a, &b).unwrap().to_vec::<f32>();
        assert!(
            output
                .into_iter()
                .all(|value| value.to_bits() == f32::MIN_POSITIVE.to_bits())
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn hypot_wgpu_subnormal_offset_views_broadcast_without_flushing() {
    with_wgpu_backend(|client, device| {
        let data = [
            99.0,
            f32::from_bits(1),
            f32::from_bits(2),
            99.0,
            f32::from_bits(3),
            f32::from_bits(0x7fffff),
        ];
        let storage = Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();
        let view = storage.narrow(1, 1, 2).unwrap().transpose(0, 1).unwrap();
        let column = Tensor::<WgpuRuntime>::from_slice(
            &[f32::from_bits(2), f32::from_bits(3)],
            &[2, 1],
            &device,
        )
        .unwrap();
        let output = client.hypot(&view, &column).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        for (value, (a, b)) in
            output
                .to_vec::<f32>()
                .into_iter()
                .zip([(1, 2), (3, 2), (2, 3), (0x7fffff, 3)])
        {
            assert_eq!(value.to_bits(), subnormal_magnitude_bits(a, b));
        }
    });
}
