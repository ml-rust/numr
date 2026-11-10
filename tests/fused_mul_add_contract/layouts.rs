use numr::ops::{BinaryOps, ScalarOps};
use numr::prelude::*;

macro_rules! layout_case {
    ($name:ident, $float:ty, $cases:expr) => {
        #[test]
        fn $name() -> Result<()> {
            let device = CpuDevice::new();
            let client = CpuClient::new(device.clone());
            let cases = $cases;
            let view = |lane: usize| -> Result<Tensor<CpuRuntime>> {
                let values = [
                    0.0,
                    <$float>::from_bits(cases[0][lane]),
                    <$float>::from_bits(cases[1][lane]),
                    0.0,
                    <$float>::from_bits(cases[2][lane]),
                    <$float>::from_bits(cases[3][lane]),
                ];
                Tensor::from_slice(&values, &[2, 3], &device)?
                    .narrow(1, 1, 2)?
                    .transpose(0, 1)
            };
            let result = client.fused_mul_add(&view(0)?, &view(1)?, &view(2)?)?;
            assert_eq!(result.shape(), &[2, 2]);
            for (actual, source) in result.try_to_vec::<$float>()?.into_iter().zip([0, 2, 1, 3]) {
                assert_eq!(actual.to_bits(), cases[source][3], "source={source}");
            }

            let expanded = |lane: usize| -> Result<Tensor<CpuRuntime>> {
                Tensor::from_slice(&[<$float>::from_bits(cases[0][lane])], &[1, 1], &device)?
                    .broadcast_to(&[3, 11])
            };
            let result = client.fused_mul_add(&expanded(0)?, &expanded(1)?, &expanded(2)?)?;
            assert_eq!(result.shape(), &[3, 11]);
            for actual in result.try_to_vec::<$float>()? {
                assert_eq!(actual.to_bits(), cases[0][3]);
            }

            let result = client.fused_mul_add_scalar(
                &expanded(1)?,
                f64::from(<$float>::from_bits(cases[0][0])),
                f64::from(<$float>::from_bits(cases[0][2])),
            )?;
            for actual in result.try_to_vec::<$float>()? {
                assert_eq!(actual.to_bits(), cases[0][3]);
            }
            Ok(())
        }
    };
}

layout_case!(
    fused_mul_add_f64_preserves_offset_transposed_and_expanded_views,
    f64,
    [
        [
            0xbf947ae147ae147bu64,
            0x4049000000000000,
            0x3ff0000000000000,
            0xbc78000000000000
        ],
        [
            0xbf947ae147ae147b,
            0x4048ffffffffffff,
            0x3ff0000000000000,
            0x3ca17ae147ae147b
        ],
        [
            0xbfd3333333333333,
            0x3fe0000000000000,
            0x3fc3333333333333,
            0
        ],
        [
            0x7fefffffffffffff,
            0x4000000000000000,
            0xffefffffffffffff,
            0x7fefffffffffffff
        ],
    ]
);

layout_case!(
    fused_mul_add_f32_preserves_offset_transposed_and_expanded_views,
    f32,
    [
        [0xbca3d70au32, 0x42480000, 0x3f800000, 0x32c00000],
        [0xbca3d70a, 0x4247ffff, 0x3f800000, 0x33d3d70a],
        [0xbe99999a, 0x3f000000, 0x3e19999a, 0],
        [0x7f7fffff, 0x40000000, 0xff7fffff, 0x7f7fffff],
    ]
);

#[test]
fn fused_mul_add_accepts_empty_shapes_and_rank_zero_inputs() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    for dtype in [DType::F32, DType::F64] {
        for shape in [&[0usize][..], &[2, 0, 3][..]] {
            let empty = Tensor::<CpuRuntime>::empty(shape, dtype, &device)?;
            let result = client.fused_mul_add(&empty, &empty, &empty)?;
            assert_eq!(result.shape(), shape);
            assert_eq!(result.dtype(), dtype);
            assert_eq!(result.numel(), 0);
            let scalar_result = client.fused_mul_add_scalar(&empty, 2.0, 1.0)?;
            assert_eq!(scalar_result.shape(), shape);
            assert_eq!(scalar_result.numel(), 0);
        }
    }
    let a = Tensor::from_slice(&[50.0f64], &[], &device)?;
    let b = Tensor::from_slice(&[-0.02f64], &[], &device)?;
    let c = Tensor::from_slice(&[1.0f64], &[], &device)?;
    let result = client.fused_mul_add(&a, &b, &c)?;
    assert!(result.shape().is_empty());
    assert_eq!(result.try_to_vec::<f64>()?[0].to_bits(), 0xbc78000000000000);
    Ok(())
}

#[test]
fn fused_mul_add_rejects_implicit_broadcast_and_dtype_mismatch() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let a = Tensor::from_slice(&[1.0f64, 2.0], &[2], &device)?;
    let singleton = Tensor::from_slice(&[1.0f64], &[1], &device)?;
    assert!(matches!(
        client.fused_mul_add(&a, &singleton, &a),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        client.fused_mul_add(&a, &a, &singleton),
        Err(Error::ShapeMismatch { .. })
    ));
    let other_dtype = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    assert!(matches!(
        client.fused_mul_add(&a, &other_dtype, &a),
        Err(Error::DTypeMismatch { .. })
    ));
    assert!(matches!(
        client.fused_mul_add(&a, &a, &other_dtype),
        Err(Error::DTypeMismatch { .. })
    ));
    Ok(())
}
