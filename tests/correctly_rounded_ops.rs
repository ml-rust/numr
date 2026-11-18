//! CPU `CorrectlyRoundedOps` admission, shapes and special values.

use numr::algorithm::correctly_rounded::{self as cr, CANONICAL_NAN};
use numr::prelude::*;

fn client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    (CpuClient::new(device.clone()), device)
}

type CpuOp = fn(&CpuClient, &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>;

const OPS: [(&str, CpuOp); 4] = [
    ("cr_exp", |c, a| c.cr_exp(a)),
    ("cr_log", |c, a| c.cr_log(a)),
    ("cr_log1p", |c, a| c.cr_log1p(a)),
    ("cr_pow_scalar", |c, a| c.cr_pow_scalar(a, 4.0 / 3.0)),
];

#[test]
fn correctly_rounded_rejects_every_dtype_but_f64() -> Result<()> {
    let (client, device) = client();
    for dtype in [DType::F32, DType::I64, DType::I32, DType::U32, DType::Bool] {
        let input = Tensor::<CpuRuntime>::zeros(&[3], dtype, &device)?;
        for (name, op) in OPS {
            match op(&client, &input) {
                Err(Error::UnsupportedDType { dtype: got, op }) => {
                    assert_eq!(got, dtype);
                    assert_eq!(op, name);
                }
                other => panic!("{name} {dtype:?}: {other:?}"),
            }
        }
    }
    Ok(())
}

#[test]
fn correctly_rounded_keeps_empty_shapes() -> Result<()> {
    let (client, device) = client();
    for shape in [&[0][..], &[3, 0], &[0, 2, 4]] {
        let input = Tensor::<CpuRuntime>::empty(shape, DType::F64, &device)?;
        for (name, op) in OPS {
            let out = op(&client, &input)?;
            assert_eq!(out.shape(), shape, "{name}");
            assert_eq!(out.dtype(), DType::F64, "{name}");
            assert_eq!(out.numel(), 0, "{name}");
        }
    }
    Ok(())
}

#[test]
fn correctly_rounded_maps_zero_dimensional_tensors() -> Result<()> {
    let (client, device) = client();
    let x = 0.156_043_060_815_255_54;
    let input = Tensor::<CpuRuntime>::from_slice(&[x], &[], &device)?;
    let expected = [cr::exp(x), cr::log(x), cr::log1p(x), cr::pow(x, 4.0 / 3.0)];
    for ((name, op), want) in OPS.into_iter().zip(expected) {
        let out = op(&client, &input)?;
        assert!(out.shape().is_empty(), "{name}");
        let got: Vec<f64> = out.try_to_vec()?;
        assert_eq!(got.len(), 1, "{name}");
        assert_eq!(got[0].to_bits(), want.to_bits(), "{name}");
    }
    Ok(())
}

#[test]
fn correctly_rounded_canonicalizes_nan_payloads() -> Result<()> {
    let (client, device) = client();
    let nans = [
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff4_0000_0000_dead),
        f64::from_bits(0x7ff8_0000_0000_beef),
        f64::from_bits(0xffff_ffff_ffff_ffff),
    ];
    let data: Vec<f64> = nans.iter().copied().cycle().take(37).collect();
    let input = Tensor::<CpuRuntime>::from_slice(&data, &[data.len()], &device)?;
    for (name, op) in OPS {
        for y in op(&client, &input)?.try_to_vec::<f64>()? {
            assert_eq!(y.to_bits(), CANONICAL_NAN.to_bits(), "{name}");
        }
    }
    Ok(())
}

#[test]
fn correctly_rounded_pow_scalar_follows_annex_f_for_the_exponent() -> Result<()> {
    let (client, device) = client();
    let data = [f64::NAN, -2.0, 0.0, -0.0, 1.0, f64::INFINITY, 4.0];
    let input = Tensor::<CpuRuntime>::from_slice(&data, &[data.len()], &device)?;
    for y in [0.0, -0.0, f64::NAN, 3.0, -3.0, 0.5, f64::INFINITY] {
        let got: Vec<f64> = client.cr_pow_scalar(&input, y)?.try_to_vec()?;
        for (&x, g) in data.iter().zip(got) {
            assert_eq!(g.to_bits(), cr::pow(x, y).to_bits(), "pow({x}, {y})");
        }
    }
    Ok(())
}
