use numr::ops::UnaryOps;
use numr::prelude::*;

// Exact rational oracle: bracket sqrt(x) by adjacent binary64 values, then compare
// x with their squared midpoint. Equality selects the even significand.
// No hardware sqrt or floating-point approximation generates these expected bits.
// Cases cover signed zeros, subnormals, the normal boundary, neighboring inputs,
// irrational roots, high finite values, and positive infinity.
const CASES: [[u64; 2]; 20] = [
    [0x0000000000000000, 0x0000000000000000],
    [0x8000000000000000, 0x8000000000000000],
    [0x0000000000000001, 0x1e60000000000000],
    [0x0000000000000002, 0x1e66a09e667f3bcd],
    [0x0000000000000003, 0x1e6bb67ae8584caa],
    [0x000fffffffffffff, 0x1fffffffffffffff],
    [0x0010000000000000, 0x2000000000000000],
    [0x0010000000000001, 0x2000000000000000],
    [0x3fefffffffffffff, 0x3fefffffffffffff],
    [0x3ff0000000000000, 0x3ff0000000000000],
    [0x3ff0000000000001, 0x3ff0000000000000],
    [0x3ff0000000000002, 0x3ff0000000000001],
    [0x3fffffffffffffff, 0x3ff6a09e667f3bcc],
    [0x4000000000000000, 0x3ff6a09e667f3bcd],
    [0x4000000000000001, 0x3ff6a09e667f3bcd],
    [0x400fffffffffffff, 0x3fffffffffffffff],
    [0x4010000000000001, 0x4000000000000000],
    [0x7fdfffffffffffff, 0x5fe6a09e667f3bcc],
    [0x7fefffffffffffff, 0x5fefffffffffffff],
    [0x7ff0000000000000, 0x7ff0000000000000],
];

fn check_bits<R: Runtime<DType = DType>>(
    output: &Tensor<R>,
    expected: impl IntoIterator<Item = u64>,
) -> Result<()> {
    let actual = output.try_to_vec::<f64>()?;
    let expected: Vec<_> = expected.into_iter().collect();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected,
            "runtime={},shape={:?},index={index}",
            R::name(),
            output.shape()
        );
    }
    Ok(())
}

fn square_root_dispatch<R: Runtime<DType = DType>>() -> Result<()>
where
    R::Client: UnaryOps<R>,
{
    let device = R::default_device();
    let client = R::default_client(&device);
    // Rotation puts every fixture in scalar inputs, vector lanes, and scalar tails.
    for len in [1usize, 2, 7, 31, 32, 33, 63, 64, 65, 257] {
        for rotation in 0..CASES.len() {
            let cases: Vec<_> = CASES.iter().cycle().skip(rotation).take(len).collect();
            let input: Vec<_> = cases.iter().map(|case| f64::from_bits(case[0])).collect();
            let tensor = Tensor::<R>::from_slice(&input, &[len], &device)?;
            let output = client.sqrt(&tensor)?;
            assert_eq!(output.shape(), &[len]);
            assert_eq!(output.dtype(), DType::F64);
            check_bits(&output, cases.iter().map(|case| case[1]))?;
        }
    }

    // Narrowing introduces an offset and row gaps before transposition.
    let mut storage = vec![0.0f64; 2 * (CASES.len() + 1)];
    for row in 0..2 {
        for (column, case) in CASES.iter().enumerate() {
            storage[row * (CASES.len() + 1) + column + 1] = f64::from_bits(case[0]);
        }
    }
    let view = Tensor::<R>::from_slice(&storage, &[2, CASES.len() + 1], &device)?
        .narrow(1, 1, CASES.len())?
        .transpose(0, 1)?;
    let output = client.sqrt(&view)?;
    assert_eq!(output.shape(), &[CASES.len(), 2]);
    check_bits(&output, CASES.iter().flat_map(|case| [case[1]; 2]))?;

    let expanded = Tensor::<R>::from_slice(&[f64::from_bits(CASES[2][0])], &[1, 1], &device)?
        .broadcast_to(&[3, 33])?;
    let output = client.sqrt(&expanded)?;
    assert_eq!(output.shape(), &[3, 33]);
    check_bits(&output, std::iter::repeat_n(CASES[2][1], 99))?;

    let special = [f64::NEG_INFINITY, -1.0, -f64::from_bits(1), f64::NAN];
    let input: Vec<_> = special.into_iter().cycle().take(65).collect();
    let output = client.sqrt(&Tensor::<R>::from_slice(&input, &[65], &device)?)?;
    for value in output.try_to_vec::<f64>()? {
        assert!(value.is_nan(), "runtime={}", R::name());
    }
    Ok(())
}

#[test]
fn cpu_f64_square_root_preserves_rounding_across_dispatch_and_views() -> Result<()> {
    square_root_dispatch::<CpuRuntime>()
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_f64_square_root_preserves_rounding_across_dispatch_and_views() -> Result<()> {
    square_root_dispatch::<numr::runtime::cuda::CudaRuntime>()
}
