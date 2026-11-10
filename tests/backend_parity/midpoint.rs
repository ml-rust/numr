//! Midpoint range and layout parity against Rust scalar arithmetic.

use numr::prelude::*;

use crate::common::create_cpu_client;

fn tensor<R: Runtime<DType = DType>>(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = data.iter().map(|&value| value as f32).collect();
            Tensor::from_slice(&data, shape, device)
        }
        DType::F64 => Tensor::from_slice(data, shape, device),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "midpoint test input",
        }),
    }
}

fn values<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> Result<Vec<f64>> {
    match tensor.dtype() {
        DType::F32 => Ok(tensor
            .try_to_vec::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect()),
        DType::F64 => tensor.try_to_vec::<f64>(),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "midpoint test output",
        }),
    }
}

fn scalar(a: f64, b: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 {
        f64::from((a as f32).midpoint(b as f32))
    } else {
        a.midpoint(b)
    }
}

fn assert_values(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: element count");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "{context}[{index}]: expected NaN, received {actual}"
            );
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{context}[{index}]");
        }
    }
}

fn pairs(dtype: DType) -> Vec<(f64, f64)> {
    let (maximum, adjacent_maximum, normal, tiny, half_maximum, above_half_maximum) =
        if dtype == DType::F32 {
            (
                f64::from(f32::MAX),
                f64::from(f32::MAX.next_down()),
                f64::from(f32::MIN_POSITIVE),
                f64::from(f32::from_bits(1)),
                f64::from(f32::MAX * 0.5),
                f64::from((f32::MAX * 0.5).next_up()),
            )
        } else {
            (
                f64::MAX,
                f64::MAX.next_down(),
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                f64::MAX * 0.5,
                (f64::MAX * 0.5).next_up(),
            )
        };
    vec![
        (0.0, 0.0),
        (-0.0, -0.0),
        (0.0, -0.0),
        (1.0, 4.0),
        (-5.5, 8.0),
        (maximum, maximum),
        (maximum, adjacent_maximum),
        (-maximum, maximum),
        (normal, normal),
        (tiny, tiny),
        (tiny, tiny * 2.0),
        (tiny, 0.0),
        (-tiny, 0.0),
        (-tiny, -tiny),
        (maximum, tiny),
        (-maximum, tiny),
        (half_maximum, half_maximum),
        (above_half_maximum, above_half_maximum),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        (f64::INFINITY, f64::NEG_INFINITY),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 1.0),
    ]
}

fn outputs<R, C>(client: &C, device: &R::Device, dtype: DType) -> Result<Vec<Vec<f64>>>
where
    R: Runtime<DType = DType>,
    C: MidpointOps<R>,
{
    let pairs = pairs(dtype);
    let a: Vec<_> = pairs.iter().map(|&(a, _)| a).collect();
    let b: Vec<_> = pairs.iter().map(|&(_, b)| b).collect();
    let a = tensor(&a, &[pairs.len()], dtype, device)?;
    let b = tensor(&b, &[pairs.len()], dtype, device)?;
    let expected: Vec<_> = pairs.iter().map(|&(a, b)| scalar(a, b, dtype)).collect();
    let result = client.midpoint(&a, &b)?;
    assert_eq!(result.dtype(), dtype);
    assert!(result.device().is_same(device));
    let forward = values(&result)?;
    assert_values(&forward, &expected, "scalar midpoint");
    let reverse = values(&client.midpoint(&b, &a)?)?;
    assert_values(&reverse, &expected, "reversed midpoint");

    let storage = tensor(&[99.0, 1.0, 4.0, 99.0, -5.5, 8.0], &[2, 3], dtype, device)?;
    let view = storage.narrow(1, 1, 2)?.transpose(0, 1)?;
    let column = tensor(&[4.0, 12.0], &[2, 1], dtype, device)?;
    let broadcast = client.midpoint(&view, &column)?;
    assert_eq!(broadcast.shape(), &[2, 2]);
    let broadcast = values(&broadcast)?;
    let expected: Vec<_> = [(1.0, 4.0), (-5.5, 4.0), (4.0, 12.0), (8.0, 12.0)]
        .into_iter()
        .map(|(a, b)| scalar(a, b, dtype))
        .collect();
    assert_values(&broadcast, &expected, "offset transposed broadcast");

    let scalar_a = tensor(&[1.0], &[], dtype, device)?;
    let scalar_b = tensor(&[4.0], &[], dtype, device)?;
    let scalar_output = client.midpoint(&scalar_a, &scalar_b)?;
    assert_eq!(scalar_output.shape(), &[] as &[usize]);
    let scalar_output = values(&scalar_output)?;
    assert_values(&scalar_output, &[scalar(1.0, 4.0, dtype)], "scalar tensor");

    let empty = Tensor::empty(&[0, 3], dtype, device)?;
    let row = tensor(&[1.0, 2.0, 3.0], &[1, 3], dtype, device)?;
    let empty_output = client.midpoint(&empty, &row)?;
    assert_eq!(empty_output.shape(), &[0, 3]);
    assert_eq!(empty_output.dtype(), dtype);
    assert!(empty_output.device().is_same(device));
    assert!(values(&empty_output)?.is_empty());
    let incompatible = tensor(&[1.0, 2.0], &[2], dtype, device)?;
    assert!(client.midpoint(&empty, &incompatible).is_err());
    let different_dtype = if dtype == DType::F32 {
        DType::F64
    } else {
        DType::F32
    };
    let different = Tensor::empty(&[0, 3], different_dtype, device)?;
    assert!(matches!(
        client.midpoint(&empty, &different),
        Err(Error::DTypeMismatch { .. })
    ));
    for unsupported in [DType::I32, DType::F16, DType::BF16] {
        let input = Tensor::empty(&[0], unsupported, device)?;
        assert!(matches!(
            client.midpoint(&input, &input),
            Err(Error::UnsupportedDType { dtype, op: "midpoint" }) if dtype == unsupported
        ));
    }
    Ok(vec![forward, reverse, broadcast, scalar_output])
}

#[test]
fn midpoint_cpu_matches_scalar_range_and_layouts() -> Result<()> {
    let (client, device) = create_cpu_client();
    for dtype in [DType::F32, DType::F64] {
        outputs(&client, &device, dtype)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn midpoint_cuda_matches_scalar_and_cpu() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (cpu, cpu_device) = create_cpu_client();
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone())?;
    let other_device = CudaDevice::new(1);
    let empty = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &device)?;
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &other_device)?;
    assert!(matches!(
        client.midpoint(&empty, &foreign),
        Err(Error::DeviceMismatch)
    ));
    assert!(matches!(
        client.midpoint(&foreign, &foreign),
        Err(Error::DeviceMismatch)
    ));
    for dtype in [DType::F32, DType::F64] {
        let actual = outputs(&client, &device, dtype)?;
        let expected = outputs(&cpu, &cpu_device, dtype)?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_values(actual, expected, "CUDA CPU parity");
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn midpoint_wgpu_rejects_unqualified_provider_and_dtype()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    let input = Tensor::<WgpuRuntime>::from_slice(&[1.0f32], &[1], &device)?;
    assert!(matches!(
        client.midpoint(&input, &input),
        Err(Error::BackendLimitation {
            operation: "midpoint",
            ..
        })
    ));
    let empty_f32 = Tensor::<WgpuRuntime>::empty(&[0], DType::F32, &device)?;
    assert!(matches!(
        client.midpoint(&empty_f32, &empty_f32),
        Err(Error::BackendLimitation {
            operation: "midpoint",
            ..
        })
    ));
    let empty_f64 = Tensor::<WgpuRuntime>::empty(&[0], DType::F64, &device)?;
    assert!(matches!(
        client.midpoint(&empty_f64, &empty_f64),
        Err(Error::UnsupportedDType {
            dtype: DType::F64,
            op: "midpoint"
        })
    ));
    Ok(())
}
