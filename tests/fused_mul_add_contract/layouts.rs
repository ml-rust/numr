use numr::ops::{BinaryOps, ScalarOps};
use numr::prelude::*;

macro_rules! layout_case {
    ($name:ident, $float:ty, $cases:expr) => {
        fn $name<R>(client: &R::Client, device: &R::Device) -> Result<()>
        where
            R: Runtime<DType = DType>,
            R::Client: BinaryOps<R> + ScalarOps<R>,
        {
            let cases = $cases;
            let view = |lane: usize| -> Result<Tensor<R>> {
                let values = [
                    0.0,
                    <$float>::from_bits(cases[0][lane]),
                    <$float>::from_bits(cases[1][lane]),
                    0.0,
                    <$float>::from_bits(cases[2][lane]),
                    <$float>::from_bits(cases[3][lane]),
                ];
                Tensor::from_slice(&values, &[2, 3], device)?
                    .narrow(1, 1, 2)?
                    .transpose(0, 1)
            };
            let result = client.fused_mul_add(&view(0)?, &view(1)?, &view(2)?)?;
            assert_eq!(result.shape(), &[2, 2]);
            for (actual, source) in result.try_to_vec::<$float>()?.into_iter().zip([0, 2, 1, 3]) {
                assert_eq!(actual.to_bits(), cases[source][3], "source={source}");
            }

            let scalar_view = view(1)?;
            let scalar_result = client.fused_mul_add_scalar(&scalar_view, 1.0, 0.0)?;
            assert_eq!(scalar_result.shape(), &[2, 2]);
            for (actual, source) in scalar_result
                .try_to_vec::<$float>()?
                .into_iter()
                .zip([0, 2, 1, 3])
            {
                assert_eq!(actual.to_bits(), cases[source][1]);
            }

            let expanded = |lane: usize| -> Result<Tensor<R>> {
                Tensor::from_slice(&[<$float>::from_bits(cases[0][lane])], &[1, 1], device)?
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

fn fused_mul_add_accepts_empty_shapes_and_rank_zero_inputs<R>(
    client: &R::Client,
    device: &R::Device,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R> + ScalarOps<R>,
{
    for dtype in [DType::F32, DType::F64] {
        for shape in [&[0usize][..], &[2, 0, 3][..]] {
            let empty = Tensor::<R>::empty(shape, dtype, device)?;
            let result = client.fused_mul_add(&empty, &empty, &empty)?;
            assert_eq!(result.shape(), shape);
            assert_eq!(result.dtype(), dtype);
            assert_eq!(result.numel(), 0);
            let scalar_result = client.fused_mul_add_scalar(&empty, 2.0, 1.0)?;
            assert_eq!(scalar_result.shape(), shape);
            assert_eq!(scalar_result.numel(), 0);
        }
    }
    let a = Tensor::from_slice(&[50.0f64], &[], device)?;
    let b = Tensor::from_slice(&[-0.02f64], &[], device)?;
    let c = Tensor::from_slice(&[1.0f64], &[], device)?;
    let result = client.fused_mul_add(&a, &b, &c)?;
    assert!(result.shape().is_empty());
    assert_eq!(result.try_to_vec::<f64>()?[0].to_bits(), 0xbc78000000000000);
    let scalar_result = client.fused_mul_add_scalar(&a, -0.02, 1.0)?;
    assert!(scalar_result.shape().is_empty());
    assert_eq!(
        scalar_result.try_to_vec::<f64>()?[0].to_bits(),
        0xbc78000000000000
    );
    let a = Tensor::from_slice(&[f32::from_bits(0xbca3d70a)], &[], device)?;
    let b = Tensor::from_slice(&[50.0f32], &[], device)?;
    let c = Tensor::from_slice(&[1.0f32], &[], device)?;
    let result = client.fused_mul_add(&a, &b, &c)?;
    assert!(result.shape().is_empty());
    assert_eq!(result.try_to_vec::<f32>()?[0].to_bits(), 0x32c00000);
    let result = client.fused_mul_add_scalar(&a, 50.0, 1.0)?;
    assert!(result.shape().is_empty());
    assert_eq!(result.try_to_vec::<f32>()?[0].to_bits(), 0x32c00000);
    Ok(())
}

fn fused_mul_add_rejects_implicit_broadcast_and_dtype_mismatch<R>(
    client: &R::Client,
    device: &R::Device,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R> + ScalarOps<R>,
{
    let a = Tensor::from_slice(&[1.0f64, 2.0], &[2], device)?;
    let singleton = Tensor::from_slice(&[1.0f64], &[1], device)?;
    assert!(matches!(
        client.fused_mul_add(&a, &singleton, &a),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        client.fused_mul_add(&a, &a, &singleton),
        Err(Error::ShapeMismatch { .. })
    ));
    let reshaped = Tensor::from_slice(&[1.0f64, 2.0], &[1, 2], device)?;
    assert!(matches!(
        client.fused_mul_add(&a, &reshaped, &a),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        client.fused_mul_add(&a, &a, &reshaped),
        Err(Error::ShapeMismatch { .. })
    ));
    let empty = Tensor::<R>::empty(&[0], DType::F64, device)?;
    let other_empty = Tensor::<R>::empty(&[2, 0], DType::F64, device)?;
    assert!(matches!(
        client.fused_mul_add(&empty, &other_empty, &empty),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        client.fused_mul_add(&empty, &empty, &other_empty),
        Err(Error::ShapeMismatch { .. })
    ));
    let other_dtype = Tensor::from_slice(&[1.0f32, 2.0], &[2], device)?;
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

macro_rules! layout_tests {
    ($runtime:ty, $setup:expr) => {
        #[test]
        fn fused_mul_add_f64_preserves_offset_transposed_and_expanded_views() -> Result<()> {
            let (client, device) = $setup?;
            super::fused_mul_add_f64_preserves_offset_transposed_and_expanded_views::<$runtime>(
                &client, &device,
            )
        }
        #[test]
        fn fused_mul_add_f32_preserves_offset_transposed_and_expanded_views() -> Result<()> {
            let (client, device) = $setup?;
            super::fused_mul_add_f32_preserves_offset_transposed_and_expanded_views::<$runtime>(
                &client, &device,
            )
        }
        #[test]
        fn fused_mul_add_accepts_empty_shapes_and_rank_zero_inputs() -> Result<()> {
            let (client, device) = $setup?;
            super::fused_mul_add_accepts_empty_shapes_and_rank_zero_inputs::<$runtime>(
                &client, &device,
            )
        }
        #[test]
        fn fused_mul_add_rejects_implicit_broadcast_and_dtype_mismatch() -> Result<()> {
            let (client, device) = $setup?;
            super::fused_mul_add_rejects_implicit_broadcast_and_dtype_mismatch::<$runtime>(
                &client, &device,
            )
        }
    };
}

mod cpu {
    use super::*;
    fn configured_cpu() -> Result<(CpuClient, CpuDevice)> {
        let device = CpuDevice::new();
        Ok((CpuClient::new(device.clone()), device))
    }
    layout_tests!(CpuRuntime, configured_cpu());
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    fn configured_cuda() -> Result<(CudaClient, CudaDevice)> {
        let device = CudaDevice::new(0);
        let client =
            CudaClient::new(device.clone()).map_err(|error| Error::Backend(error.to_string()))?;
        Ok((client, device))
    }
    layout_tests!(CudaRuntime, configured_cuda());

    #[test]
    fn fused_mul_add_rejects_foreign_device_empty_inputs() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let local = Tensor::<CudaRuntime>::empty(&[0], DType::F64, &device)?;
        // Empty storage needs no physical device or allocation.
        let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F64, &CudaDevice::new(usize::MAX))?;
        for (a, b, c) in [
            (&foreign, &local, &local),
            (&local, &foreign, &local),
            (&local, &local, &foreign),
        ] {
            assert!(matches!(
                client.fused_mul_add(a, b, c),
                Err(Error::DeviceMismatch)
            ));
        }
        assert!(matches!(
            client.fused_mul_add_scalar(&foreign, 2.0, 1.0),
            Err(Error::DeviceMismatch)
        ));
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn fused_mul_add_rejects_counts_beyond_kernel_index_range() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let input =
            Tensor::from_slice(&[1.0f64], &[1], &device)?.broadcast_to(&[u32::MAX as usize + 1])?;
        assert!(matches!(
            client.fused_mul_add(&input, &input, &input),
            Err(Error::BackendLimitation {
                backend: "cuda",
                operation: "fused_mul_add",
                ..
            })
        ));
        assert!(matches!(
            client.fused_mul_add_scalar(&input, 2.0, 1.0),
            Err(Error::BackendLimitation {
                backend: "cuda",
                operation: "fused_mul_add_scalar",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn fused_add_mul_preserves_offset_transposed_and_expanded_views() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let a = Tensor::from_slice(&[0.0f64, 1.0, 2.0, 0.0, 3.0, 4.0], &[2, 3], &device)?
            .narrow(1, 1, 2)?
            .transpose(0, 1)?;
        let b = Tensor::from_slice(&[0.0f64, 5.0, 6.0, 0.0, 7.0, 8.0], &[2, 3], &device)?
            .narrow(1, 1, 2)?
            .transpose(0, 1)?;
        let c = Tensor::from_slice(&[2.0f64], &[1, 1], &device)?.broadcast_to(&[2, 2])?;
        let result = client.fused_add_mul(&a, &b, &c)?;
        assert_eq!(result.shape(), &[2, 2]);
        // Logical rows are (1 + 5, 3 + 7) * 2 and (2 + 6, 4 + 8) * 2.
        assert_eq!(result.try_to_vec::<f64>()?, [12.0, 20.0, 16.0, 24.0]);
        Ok(())
    }

    #[test]
    fn fused_add_mul_rejects_short_operands_and_same_count_different_shapes() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let a = Tensor::from_slice(&[1.0f64, 2.0], &[2], &device)?;
        let short = Tensor::from_slice(&[1.0f64], &[1], &device)?;
        let ranked = Tensor::from_slice(&[1.0f64, 2.0], &[1, 2], &device)?;
        for operand in [&short, &ranked] {
            assert!(matches!(
                client.fused_add_mul(&a, operand, &a),
                Err(Error::ShapeMismatch { .. })
            ));
            assert!(matches!(
                client.fused_add_mul(&a, &a, operand),
                Err(Error::ShapeMismatch { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn fused_add_mul_rejects_foreign_device_empty_inputs() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let local = Tensor::<CudaRuntime>::empty(&[0], DType::F64, &device)?;
        // Zero-byte storage carries device identity without creating a foreign CUDA context.
        let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F64, &CudaDevice::new(usize::MAX))?;
        for (a, b, c) in [
            (&foreign, &local, &local),
            (&local, &foreign, &local),
            (&local, &local, &foreign),
        ] {
            assert!(matches!(
                client.fused_add_mul(a, b, c),
                Err(Error::DeviceMismatch)
            ));
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn fused_add_mul_rejects_counts_beyond_kernel_index_range() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let input =
            Tensor::from_slice(&[1.0f64], &[1], &device)?.broadcast_to(&[u32::MAX as usize + 1])?;
        assert!(matches!(
            client.fused_add_mul(&input, &input, &input),
            Err(Error::BackendLimitation {
                backend: "cuda",
                operation: "fused_add_mul",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn fused_operations_preserve_negative_stride_views() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let a = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device)?.flip(1)?;
        let b = Tensor::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], &device)?.flip(0)?;
        let c = Tensor::from_slice(&[2.0f64], &[1, 1], &device)?.broadcast_to(&[2, 2])?;
        assert!(a.strides().iter().any(|&stride| stride < 0));
        assert!(b.strides().iter().any(|&stride| stride < 0));
        let result = client.fused_mul_add(&a, &b, &c)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.try_to_vec::<f64>()?, [16.0, 10.0, 22.0, 20.0]);
        let result = client.fused_add_mul(&a, &b, &c)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.try_to_vec::<f64>()?, [18.0, 18.0, 18.0, 18.0]);
        let result = client.fused_mul_add_scalar(&a, 2.0, 1.0)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.try_to_vec::<f64>()?, [5.0, 3.0, 9.0, 7.0]);
        Ok(())
    }

    #[test]
    fn fused_operations_reject_strided_rank_beyond_copy_kernel_limit() -> Result<()> {
        let (client, device) = configured_cuda()?;
        let shape = vec![2; numr::runtime::cuda::kernels::MAX_DIMS + 1];
        let input = Tensor::from_slice(&[1.0f64], &[1], &device)?.broadcast_to(&shape)?;
        assert!(!input.is_contiguous());
        for (operation, result) in [
            (
                "fused_mul_add",
                client.fused_mul_add(&input, &input, &input),
            ),
            (
                "fused_add_mul",
                client.fused_add_mul(&input, &input, &input),
            ),
            (
                "fused_mul_add_scalar",
                client.fused_mul_add_scalar(&input, 2.0, 1.0),
            ),
        ] {
            assert!(matches!(result, Err(Error::BackendLimitation {
                backend: "cuda", operation: actual, ..
            }) if actual == operation));
        }
        Ok(())
    }
}
