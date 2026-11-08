//! CPU magnitude semantics and input checks.

use numr::prelude::*;

#[test]
fn hypot_f64_preserves_extreme_scales_and_special_values() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let tiny = f64::from_bits(1);
    let pairs = [
        (3.0, 4.0),
        (f64::MAX / 2.0, f64::MAX / 2.0),
        (1e-300, -1e-300),
        (tiny, tiny),
        (tiny, 0.0),
        (f64::MIN_POSITIVE, tiny),
        (f64::MAX, tiny),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (f64::NAN, 2.0),
        (2.0, f64::NAN),
        (f64::NAN, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NAN),
        (f64::MAX, f64::MAX),
    ];
    let a = Tensor::from_slice(&pairs.map(|(a, _)| a), &[pairs.len()], &device)?;
    let b = Tensor::from_slice(&pairs.map(|(_, b)| b), &[pairs.len()], &device)?;
    let output = client.hypot(&a, &b)?;
    assert_eq!(output.dtype(), DType::F64);
    for (actual, (a, b)) in output.to_vec::<f64>().into_iter().zip(pairs) {
        let expected = a.hypot(b);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected == 0.0 || expected.is_infinite() {
            assert_eq!(actual.to_bits(), expected.to_bits());
        } else {
            assert!(actual.is_finite() && actual > 0.0);
            assert!((actual - expected).abs() <= expected * 4e-16 + tiny);
        }
    }
    Ok(())
}

#[test]
fn hypot_f32_preserves_extreme_scales_and_subnormals() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let tiny = f32::from_bits(1);
    let pairs = [
        (f32::MAX / 2.0, f32::MAX / 2.0),
        (1e-30, -1e-30),
        (tiny, tiny),
        (tiny, 0.0),
        (f32::MIN_POSITIVE, tiny),
        (f32::MAX, tiny),
        (-0.0, -0.0),
        (f32::NAN, f32::NEG_INFINITY),
        (f32::INFINITY, f32::NAN),
        (f32::NAN, 3.0),
    ];
    let a = Tensor::from_slice(&pairs.map(|(a, _)| a), &[pairs.len()], &device)?;
    let b = Tensor::from_slice(&pairs.map(|(_, b)| b), &[pairs.len()], &device)?;
    let output = client.hypot(&a, &b)?;
    assert_eq!(output.dtype(), DType::F32);
    for (actual, (a, b)) in output.to_vec::<f32>().into_iter().zip(pairs) {
        let expected = a.hypot(b);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected == 0.0 || expected.is_infinite() {
            assert_eq!(actual.to_bits(), expected.to_bits());
        } else {
            assert!(actual.is_finite() && actual > 0.0);
            assert!(
                (actual as f64 - expected as f64).abs() <= expected as f64 * 2e-7 + tiny as f64
            );
        }
    }
    Ok(())
}

#[test]
fn hypot_broadcasts_noncontiguous_offset_views_and_scalars() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let storage = Tensor::from_slice(&[99.0f64, 3.0, 5.0, 99.0, 8.0, 7.0], &[2, 3], &device)?;
    let a = storage.narrow(1, 1, 2)?.transpose(0, 1)?;
    let b = Tensor::from_slice(&[4.0f64, 12.0], &[2, 1], &device)?;
    let output = client.hypot(&a, &b)?;
    assert_eq!(output.shape(), &[2, 2]);
    for (actual, expected) in
        output
            .to_vec::<f64>()
            .into_iter()
            .zip([5.0, 8.0f64.hypot(4.0), 13.0, 7.0f64.hypot(12.0)])
    {
        assert!((actual - expected).abs() <= expected * 4e-16);
    }
    let scalar = Tensor::from_slice(&[4.0f64], &[], &device)?;
    let output = client.hypot(&scalar, &scalar)?;
    assert_eq!(output.shape(), &[] as &[usize]);
    assert!((output.to_vec::<f64>()[0] - 4.0f64.hypot(4.0)).abs() < 1e-14);
    Ok(())
}

#[test]
fn hypot_handles_empty_broadcasts_and_rejects_invalid_inputs() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = Tensor::<CpuRuntime>::empty(&[0, 3], DType::F64, &device)?;
    let b = Tensor::<CpuRuntime>::ones(&[1, 3], DType::F64, &device)?;
    assert_eq!(client.hypot(&a, &b)?.shape(), &[0, 3]);
    let other = Tensor::<CpuRuntime>::ones(&[2], DType::F64, &device)?;
    assert!(matches!(
        client.hypot(&a, &other),
        Err(Error::BroadcastError { .. })
    ));
    let other = Tensor::<CpuRuntime>::ones(&[1, 3], DType::F32, &device)?;
    assert!(matches!(
        client.hypot(&a, &other),
        Err(Error::DTypeMismatch { .. })
    ));
    for dtype in [DType::I32, DType::U32, DType::Bool, DType::Complex64] {
        let a = Tensor::<CpuRuntime>::empty(&[0], dtype, &device)?;
        assert!(matches!(
            client.hypot(&a, &a),
            Err(Error::UnsupportedDType { .. })
        ));
    }
    Ok(())
}

#[test]
fn hypot_narrow_floats_compute_in_f32() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let input = Tensor::from_slice(&[3.0f32, 12.0], &[2], &device)?;
    let other = Tensor::from_slice(&[4.0f32, 5.0], &[2], &device)?;
    let expected = Tensor::from_slice(&[5.0f32, 13.0], &[2], &device)?;
    let dtypes: &[DType] = &[
        #[cfg(feature = "f16")]
        DType::F16,
        #[cfg(feature = "f16")]
        DType::BF16,
        #[cfg(feature = "fp8")]
        DType::FP8E4M3,
        #[cfg(feature = "fp8")]
        DType::FP8E5M2,
    ];
    for &dtype in dtypes {
        let a = client.cast(&input, dtype)?;
        let b = client.cast(&other, dtype)?;
        let output = client.hypot(&a, &b)?;
        assert_eq!(output.dtype(), dtype);
        let expected = client.cast(&client.cast(&expected, dtype)?, DType::F32)?;
        assert_eq!(
            client.cast(&output, DType::F32)?.to_vec::<f32>(),
            expected.to_vec::<f32>()
        );
    }
    Ok(())
}

#[test]
fn hypot_rejects_disabled_float_formats() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let dtypes: &[DType] = &[
        #[cfg(not(feature = "f16"))]
        DType::F16,
        #[cfg(not(feature = "f16"))]
        DType::BF16,
        #[cfg(not(feature = "fp8"))]
        DType::FP8E4M3,
        #[cfg(not(feature = "fp8"))]
        DType::FP8E5M2,
    ];
    for &dtype in dtypes {
        let a = Tensor::<CpuRuntime>::empty(&[0], dtype, &device)?;
        assert!(matches!(
            client.hypot(&a, &a),
            Err(Error::UnsupportedDType { .. })
        ));
    }
    Ok(())
}
