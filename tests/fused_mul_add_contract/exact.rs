use numr::ops::{BinaryOps, ScalarOps};
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;

fn configured() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (client, device)
}

#[test]
fn fused_mul_add_f64_rounds_once_at_scalar_vector_and_tail_lengths() -> Result<()> {
    let (client, device) = configured();
    // Expected bits round exact dyadic products plus addends to nearest, ties to even.
    let cases: [(u64, u64, u64, u64); 9] = [
        (
            0xbf947ae147ae147b,
            0x4049000000000000,
            0x3ff0000000000000,
            0xbc78000000000000,
        ),
        (
            0xbf947ae147ae147b,
            0x4048ffffffffffff,
            0x3ff0000000000000,
            0x3ca17ae147ae147b,
        ),
        (
            0xbfd3333333333333,
            0x3fe0000000000000,
            0x3fc3333333333333,
            0,
        ),
        (
            0x7fefffffffffffff,
            0x4000000000000000,
            0xffefffffffffffff,
            0x7fefffffffffffff,
        ),
        (1, 0x3fe0000000000000, 0, 0),
        (1, 0x3ff0000000000000, 1, 2),
        (
            0x8000000000000000,
            0x3ff0000000000000,
            0x8000000000000000,
            0x8000000000000000,
        ),
        (
            0x3ff0000000000000,
            0x3ff0000000000000,
            0xbff0000000000000,
            0,
        ),
        (
            0x8000000000000001,
            0x3fe0000000000000,
            0x8000000000000000,
            0x8000000000000000,
        ),
    ];
    for count in [1, 7, 8, 15, 16, 17, 31, 32, 33, 65, 257] {
        let a: Vec<_> = (0..count)
            .map(|i| f64::from_bits(cases[i % cases.len()].0))
            .collect();
        let b: Vec<_> = (0..count)
            .map(|i| f64::from_bits(cases[i % cases.len()].1))
            .collect();
        let c: Vec<_> = (0..count)
            .map(|i| f64::from_bits(cases[i % cases.len()].2))
            .collect();
        let result = client.fused_mul_add(
            &Tensor::from_slice(&a, &[count], &device)?,
            &Tensor::from_slice(&b, &[count], &device)?,
            &Tensor::from_slice(&c, &[count], &device)?,
        )?;
        assert_eq!(result.shape(), &[count]);
        for (i, actual) in result.try_to_vec::<f64>()?.into_iter().enumerate() {
            assert_eq!(
                actual.to_bits(),
                cases[i % cases.len()].3,
                "count={count},lane={i}"
            );
        }
    }
    Ok(())
}

#[test]
fn fused_mul_add_f32_rounds_once_at_scalar_vector_and_tail_lengths() -> Result<()> {
    let (client, device) = configured();
    let cases: [(u32, u32, u32, u32); 9] = [
        (0xbca3d70a, 0x42480000, 0x3f800000, 0x32c00000),
        (0xbca3d70a, 0x4247ffff, 0x3f800000, 0x33d3d70a),
        (0xbe99999a, 0x3f000000, 0x3e19999a, 0),
        (0x7f7fffff, 0x40000000, 0xff7fffff, 0x7f7fffff),
        (1, 0x3f000000, 0, 0),
        (1, 0x3f800000, 1, 2),
        (0x80000000, 0x3f800000, 0x80000000, 0x80000000),
        (0x3f800000, 0x3f800000, 0xbf800000, 0),
        (0x80000001, 0x3f000000, 0x80000000, 0x80000000),
    ];
    for count in [1, 7, 8, 15, 16, 17, 31, 32, 33, 65, 257] {
        let a: Vec<_> = (0..count)
            .map(|i| f32::from_bits(cases[i % cases.len()].0))
            .collect();
        let b: Vec<_> = (0..count)
            .map(|i| f32::from_bits(cases[i % cases.len()].1))
            .collect();
        let c: Vec<_> = (0..count)
            .map(|i| f32::from_bits(cases[i % cases.len()].2))
            .collect();
        let result = client.fused_mul_add(
            &Tensor::from_slice(&a, &[count], &device)?,
            &Tensor::from_slice(&b, &[count], &device)?,
            &Tensor::from_slice(&c, &[count], &device)?,
        )?;
        assert_eq!(result.shape(), &[count]);
        for (i, actual) in result.try_to_vec::<f32>()?.into_iter().enumerate() {
            assert_eq!(
                actual.to_bits(),
                cases[i % cases.len()].3,
                "count={count},lane={i}"
            );
        }
    }
    Ok(())
}

#[test]
fn fused_mul_add_scalar_f32_converts_each_parameter_before_fusion() -> Result<()> {
    let (client, device) = configured();
    for count in [1, 17, 33, 65] {
        let input = Tensor::from_slice(&vec![1.0f32; count], &[count], &device)?;
        // Each halfway scalar rounds to the even F32 endpoint before arithmetic.
        for (scale, bias) in [(1.0 + 2f64.powi(-24), -1.0), (1.0, -1.0 - 2f64.powi(-24))] {
            let result = client.fused_mul_add_scalar(&input, scale, bias)?;
            for actual in result.try_to_vec::<f32>()? {
                assert_eq!(
                    actual.to_bits(),
                    0,
                    "count={count},scale={scale},bias={bias}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fused_mul_add_scalar_f64_preserves_cancellation_and_product_range() -> Result<()> {
    let (client, device) = configured();
    for count in [1, 17, 33, 65] {
        for (input, scale, bias, expected) in [
            (50.0, -0.02, 1.0, 0xbc78000000000000u64),
            (f64::MAX, 2.0, -f64::MAX, f64::MAX.to_bits()),
            (f64::from_bits(1), 1.0, f64::from_bits(1), 2),
        ] {
            let a = Tensor::from_slice(&vec![input; count], &[count], &device)?;
            for actual in client
                .fused_mul_add_scalar(&a, scale, bias)?
                .try_to_vec::<f64>()?
            {
                assert_eq!(actual.to_bits(), expected, "count={count}");
            }
        }
    }
    Ok(())
}

#[test]
fn fused_mul_add_special_values_follow_ieee_categories() -> Result<()> {
    let (client, device) = configured();
    let a = Tensor::from_slice(
        &[f64::NAN, f64::INFINITY, 0.0, f64::INFINITY],
        &[4],
        &device,
    )?;
    let b = Tensor::from_slice(&[1.0f64, 2.0, f64::INFINITY, 1.0], &[4], &device)?;
    let c = Tensor::from_slice(&[0.0f64, 1.0, 1.0, f64::NEG_INFINITY], &[4], &device)?;
    let result = client.fused_mul_add(&a, &b, &c)?.try_to_vec::<f64>()?;
    assert!(result[0].is_nan());
    assert_eq!(result[1], f64::INFINITY);
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
    Ok(())
}

#[test]
fn fused_mul_add_f32_special_values_follow_ieee_categories() -> Result<()> {
    let (client, device) = configured();
    let a = Tensor::from_slice(
        &[f32::NAN, f32::INFINITY, 0.0, f32::INFINITY],
        &[4],
        &device,
    )?;
    let b = Tensor::from_slice(&[1.0f32, 2.0, f32::INFINITY, 1.0], &[4], &device)?;
    let c = Tensor::from_slice(&[0.0f32, 1.0, 1.0, f32::NEG_INFINITY], &[4], &device)?;
    let result = client.fused_mul_add(&a, &b, &c)?.try_to_vec::<f32>()?;
    assert!(result[0].is_nan());
    assert_eq!(result[1], f32::INFINITY);
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
    Ok(())
}
