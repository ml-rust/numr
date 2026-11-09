//! CPU rounded-addition residuals and input admission.

use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;

fn configured() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (client, device)
}

#[test]
fn two_sum_f64_retains_exact_rounding_residuals_at_simd_and_tail_lengths() -> Result<()> {
    let (client, device) = configured();
    let tiny = f64::from_bits(1);
    let cases = [
        (0.1, 0.9, 1.0, 2f64.powi(-55)),
        (1.0, 2f64.powi(-54), 1.0, 2f64.powi(-54)),
        (-1.0, -2f64.powi(-54), -1.0, -2f64.powi(-54)),
        (1.0, -1.0, 0.0, 0.0),
        (tiny, tiny, 2.0 * tiny, 0.0),
        (
            f64::MIN_POSITIVE,
            -f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1),
            tiny,
            0.0,
        ),
        (f64::MAX, tiny, f64::MAX, tiny),
        (f64::MAX, -f64::MAX, 0.0, 0.0),
        (f64::MAX / 2.0, f64::MAX / 2.0, f64::MAX, 0.0),
    ];
    for count in [1, 8, 31, 32, 33, 64, 65] {
        let a: Vec<_> = (0..count).map(|i| cases[i % cases.len()].0).collect();
        let b: Vec<_> = (0..count).map(|i| cases[i % cases.len()].1).collect();
        let (sum, residual) = client.two_sum(
            &Tensor::from_slice(&a, &[count], &device)?,
            &Tensor::from_slice(&b, &[count], &device)?,
        )?;
        assert_eq!(sum.dtype(), DType::F64);
        for (i, (sum, residual)) in sum
            .try_to_vec::<f64>()?
            .into_iter()
            .zip(residual.try_to_vec::<f64>()?)
            .enumerate()
        {
            let expected = cases[i % cases.len()];
            assert_eq!(sum.to_bits(), expected.2.to_bits(), "sum lane {i}");
            assert_eq!(
                residual.to_bits(),
                expected.3.to_bits(),
                "residual lane {i}"
            );
        }
    }
    Ok(())
}

#[test]
fn two_sum_f32_retains_subnormals_and_nonzero_normal_residuals() -> Result<()> {
    let (client, device) = configured();
    let tiny = f32::from_bits(1);
    let a = [0.1f32, 1.0, -1.0, tiny, f32::MAX, f32::MIN_POSITIVE];
    let b = [
        0.9f32,
        2f32.powi(-25),
        -2f32.powi(-25),
        tiny,
        tiny,
        -f32::from_bits(f32::MIN_POSITIVE.to_bits() - 1),
    ];
    let (sum, residual) = client.two_sum(
        &Tensor::from_slice(&a, &[a.len()], &device)?,
        &Tensor::from_slice(&b, &[b.len()], &device)?,
    )?;
    let residual_values = residual.try_to_vec::<f32>()?;
    assert_eq!(residual_values[4].to_bits(), tiny.to_bits());
    for (((sum, residual), a), b) in sum
        .try_to_vec::<f32>()?
        .into_iter()
        .zip(residual_values)
        .zip(a)
        .zip(b)
    {
        assert_eq!(sum, a + b);
        // F64 checks reconstruction for nearby magnitudes. The bit check above covers the widely separated pair.
        assert_eq!(
            f64::from(sum) + f64::from(residual),
            f64::from(a) + f64::from(b)
        );
    }
    Ok(())
}

#[test]
fn two_sum_broadcasts_offset_noncontiguous_views_and_scalars() -> Result<()> {
    let (client, device) = configured();
    let storage = Tensor::from_slice(&[99.0f64, 0.1, 1.0, 99.0, -1.0, 0.1], &[2, 3], &device)?;
    let a = storage.narrow(1, 1, 2)?.transpose(0, 1)?;
    let b = Tensor::from_slice(&[0.9f64, 2f64.powi(-54)], &[2, 1], &device)?;
    let (sum, residual) = client.two_sum(&a, &b)?;
    assert_eq!(sum.shape(), &[2, 2]);
    assert_eq!(residual.shape(), &[2, 2]);
    assert_eq!(
        sum.try_to_vec::<f64>()?,
        vec![
            0.1 + 0.9,
            -1.0 + 0.9,
            1.0 + 2f64.powi(-54),
            0.1 + 2f64.powi(-54)
        ]
    );
    let residuals = residual.try_to_vec::<f64>()?;
    assert_eq!(residuals[0], 2f64.powi(-55));
    assert_eq!(residuals[2], 2f64.powi(-54));
    let scalar = Tensor::from_slice(&[0.1f64], &[], &device)?;
    let other = Tensor::from_slice(&[0.9f64], &[], &device)?;
    let (sum, residual) = client.two_sum(&scalar, &other)?;
    assert_eq!(sum.shape(), &[] as &[usize]);
    assert_eq!(sum.item::<f64>()?, 1.0);
    assert_eq!(residual.item::<f64>()?, 2f64.powi(-55));
    Ok(())
}

#[test]
fn two_sum_preserves_ieee_nonfinite_sum_and_nan_residual() -> Result<()> {
    let (client, device) = configured();
    let a = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    let b = [1.0, 1.0, f64::NEG_INFINITY, 1.0, f64::MAX];
    let (sum, residual) = client.two_sum(
        &Tensor::from_slice(&a, &[a.len()], &device)?,
        &Tensor::from_slice(&b, &[b.len()], &device)?,
    )?;
    for ((sum, a), b) in sum.try_to_vec::<f64>()?.into_iter().zip(a).zip(b) {
        if (a + b).is_nan() {
            assert!(sum.is_nan());
        } else {
            assert_eq!(sum.to_bits(), (a + b).to_bits());
        }
    }
    assert!(residual.try_to_vec::<f64>()?.into_iter().all(f64::is_nan));
    Ok(())
}

#[test]
fn two_sum_handles_empty_shapes_and_rejects_unsupported_inputs() -> Result<()> {
    let (client, device) = configured();
    let a = Tensor::<CpuRuntime>::empty(&[0, 3], DType::F64, &device)?;
    let b = Tensor::<CpuRuntime>::ones(&[1, 3], DType::F64, &device)?;
    let (sum, residual) = client.two_sum(&a, &b)?;
    assert_eq!(sum.shape(), &[0, 3]);
    assert_eq!(residual.shape(), &[0, 3]);
    let wrong_shape = Tensor::<CpuRuntime>::ones(&[2], DType::F64, &device)?;
    assert!(matches!(
        client.two_sum(&a, &wrong_shape),
        Err(Error::BroadcastError { .. })
    ));
    let wrong_dtype = Tensor::<CpuRuntime>::ones(&[1, 3], DType::F32, &device)?;
    assert!(matches!(
        client.two_sum(&a, &wrong_dtype),
        Err(Error::DTypeMismatch { .. })
    ));
    for dtype in [
        DType::I32,
        DType::U32,
        DType::Bool,
        DType::Complex64,
        DType::F16,
        DType::BF16,
        DType::FP8E4M3,
        DType::FP8E5M2,
    ] {
        let value = Tensor::<CpuRuntime>::empty(&[0], dtype, &device)?;
        assert!(matches!(
            client.two_sum(&value, &value),
            Err(Error::UnsupportedDType { .. })
        ));
    }
    Ok(())
}
