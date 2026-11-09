//! Analytic all-masked limits, NaN propagation, and unchanged finite/empty reductions.

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::common::is_dtype_supported;
use crate::common::{assert_tensor_allclose, create_cpu_client};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{CumulativeOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

struct Case {
    name: String,
    data: Vec<f64>,
    shape: [usize; 2],
    dim: usize,
    expected: Vec<f64>,
}

fn matrix(name: &str, rows: &[Vec<f64>], width: usize, strided: bool, expected: Vec<f64>) -> Case {
    let data = if strided {
        (0..width)
            .flat_map(|i| rows.iter().map(move |row| row[i]))
            .collect()
    } else {
        rows.iter().flatten().copied().collect()
    };
    Case {
        name: format!("{name}_width{width}_rows{}_strided{strided}", rows.len()),
        data,
        shape: if strided {
            [width, rows.len()]
        } else {
            [rows.len(), width]
        },
        dim: if strided { 0 } else { 1 },
        expected,
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for width in [1, 2, 31, 32, 33, 64, 65, 128] {
        for strided in [false, true] {
            for value in [2.0, 1024.0] {
                let rows = vec![vec![value; width]; 2];
                cases.push(matrix(
                    "equal_finite",
                    &rows,
                    width,
                    strided,
                    vec![value + (width as f64).ln(); 2],
                ));
            }
            let masked = vec![f64::NEG_INFINITY; width];
            cases.push(matrix(
                "all_masked",
                &[masked.clone(), masked.clone()],
                width,
                strided,
                vec![f64::NEG_INFINITY; 2],
            ));
            let mut finite = masked.clone();
            finite[width / 2] = 2.0;
            cases.push(matrix(
                "finite_and_masked",
                &[finite.clone(), masked.clone()],
                width,
                strided,
                vec![2.0, f64::NEG_INFINITY],
            ));
            for position in [0, width / 2, width - 1] {
                let mut nan = masked.clone();
                nan[position] = f64::NAN;
                cases.push(matrix(
                    &format!("nan_at{position}"),
                    &[nan, finite.clone()],
                    width,
                    strided,
                    vec![f64::NAN, 2.0],
                ));
            }
            // Also exercise [N,2] contiguous and [2,N] strided layouts.
            let mut rows = vec![vec![f64::NEG_INFINITY; 2]; width];
            let mut expected = vec![f64::NEG_INFINITY; width];
            for (i, row) in rows.iter_mut().enumerate() {
                match i % 3 {
                    0 => {}
                    1 => {
                        row[0] = f64::NAN;
                        expected[i] = f64::NAN;
                    }
                    _ => {
                        row[1] = 2.0;
                        expected[i] = 2.0;
                    }
                }
            }
            cases.push(matrix("two_element_rows", &rows, 2, strided, expected));
            cases.push(matrix("empty_batch", &[], width, strided, vec![]));
            cases.push(matrix(
                "empty_reduction",
                &[vec![], vec![]],
                0,
                strided,
                vec![f64::NEG_INFINITY; 2],
            ));
        }
    }
    cases
}

fn dtypes() -> Vec<DType> {
    let base = [DType::F32, DType::F64];
    #[cfg(feature = "f16")]
    {
        base.into_iter().chain([DType::F16, DType::BF16]).collect()
    }
    #[cfg(not(feature = "f16"))]
    {
        base.to_vec()
    }
}

fn run<R, C>(client: &C, device: &R::Device, dtype: DType, cases: &[Case]) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: CumulativeOps<R> + TypeConversionOps<R>,
{
    let (expected_client, expected_device) = create_cpu_client();
    for case in cases {
        let input = tensor_from_f64(&case.data, &case.shape, dtype, device, client)?;
        for keepdim in [false, true] {
            let result = client.logsumexp(&input, &[case.dim], keepdim)?;
            let mut expected_shape = case.shape.to_vec();
            if keepdim {
                expected_shape[case.dim] = 1;
            } else {
                expected_shape.remove(case.dim);
            }
            assert_eq!(
                result.shape(),
                expected_shape,
                "{} keepdim={keepdim}",
                case.name
            );
            assert_eq!(result.dtype(), dtype, "{} dtype", case.name);
            let expected = tensor_from_f64(
                &case.expected,
                &expected_shape,
                dtype,
                &expected_device,
                &expected_client,
            )?;
            assert_tensor_allclose(
                &result,
                &expected,
                dtype,
                &format!("{} keepdim={keepdim}", case.name),
            );
        }
    }
    Ok(())
}

#[test]
fn logsumexp_masked_limits_cpu() -> Result<()> {
    let (client, device) = create_cpu_client();
    let cases = cases();
    for dtype in dtypes() {
        run(&client, &device, dtype, &cases)?;
    }
    let integer = Tensor::from_slice(&[1i32, 2], &[2], &device)?;
    assert!(
        client.logsumexp(&integer, &[0], false).is_err(),
        "integer dtype admission remains unchanged"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn logsumexp_masked_limits_cuda() -> Result<()> {
    let cases = cases();
    let mut result = Ok(());
    with_cuda_backend(|client, device| {
        for dtype in dtypes() {
            if is_dtype_supported("cuda", dtype) {
                result = run(&client, &device, dtype, &cases);
                if result.is_err() {
                    return;
                }
            }
        }
    });
    result
}

#[cfg(feature = "wgpu")]
#[test]
fn logsumexp_masked_limits_wgpu() -> Result<()> {
    let cases = cases();
    let mut result = Ok(());
    with_wgpu_backend(|client, device| {
        for dtype in dtypes() {
            if is_dtype_supported("wgpu", dtype) {
                result = run(&client, &device, dtype, &cases);
                if result.is_err() {
                    return;
                }
            }
        }
    });
    result
}
