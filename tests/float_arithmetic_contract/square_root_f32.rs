//! Exact binary32 references across native dispatch, views, and independent output storage.

#[path = "../common/sqrt_f32_oracle.rs"]
mod oracle;

use numr::ops::{BinaryOps, UnaryOps};
use numr::prelude::*;
use numr::runtime::RuntimeClient;

type CheckResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn output_bits<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> Result<Vec<u32>> {
    Ok(tensor
        .try_to_vec::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn input_bits<R, C>(client: &C, input: &Tensor<R>) -> Result<Vec<u32>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + RuntimeClient<R>,
{
    let copy = Tensor::empty(input.shape(), input.dtype(), client.device())?;
    client.copy_into(&copy, input)?;
    output_bits(&copy)
}

fn check<R: Runtime<DType = DType>>(
    output: &Tensor<R>,
    expected: impl IntoIterator<Item = u32>,
    context: &str,
) -> Result<()> {
    let actual = output_bits(output)?;
    let expected: Vec<_> = expected.into_iter().collect();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_eq!(
            actual,
            expected,
            "{context},runtime={},index={index}",
            R::name()
        );
    }
    Ok(())
}

fn dispatch<R, C>(client: &C) -> CheckResult
where
    R: Runtime<DType = DType>,
    C: UnaryOps<R> + BinaryOps<R> + RuntimeClient<R>,
{
    let cases = oracle::cases();
    assert_eq!(cases.len(), 200_308);
    let values: Vec<_> = cases
        .iter()
        .map(|&(input, _)| f32::from_bits(input))
        .collect();
    let input = Tensor::<R>::from_slice(&values, &[values.len()], client.device())?;
    let output = client.sqrt(&input)?;
    assert_eq!(output.shape(), input.shape());
    assert_eq!(output.dtype(), DType::F32);
    assert!(output.device().is_same(client.device()));
    assert!(!output.shares_storage_with(&input));
    check(
        &output,
        cases.iter().map(|&(_, expected)| expected),
        "full exponent corpus",
    )?;
    assert_eq!(
        input_bits(client, &input)?,
        cases.iter().map(|&(bits, _)| bits).collect::<Vec<_>>()
    );
    let zero = Tensor::zeros(input.shape(), DType::F32, client.device())?;
    client.copy_into(&output, &zero)?;
    assert_eq!(
        input_bits(client, &input)?,
        cases.iter().map(|&(bits, _)| bits).collect::<Vec<_>>()
    );
    let seven = Tensor::full_scalar(input.shape(), DType::F32, 7.0, client.device())?;
    client.copy_into(&input, &seven)?;
    assert!(output_bits(&output)?.into_iter().all(|bits| bits == 0));

    for len in [
        1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257,
    ] {
        for rotation in 0..oracle::WITNESSES.len() {
            let bits: Vec<_> = oracle::WITNESSES
                .iter()
                .copied()
                .cycle()
                .skip(rotation)
                .take(len)
                .collect();
            let values: Vec<_> = bits.iter().copied().map(f32::from_bits).collect();
            let input = Tensor::<R>::from_slice(&values, &[len], client.device())?;
            let output = client.sqrt(&input)?;
            assert_eq!(output.shape(), &[len]);
            assert_eq!(output.dtype(), DType::F32);
            assert!(!output.shares_storage_with(&input));
            check(
                &output,
                bits.into_iter().map(oracle::rounded_sqrt_bits),
                "scalar/vector tails",
            )?;
        }
    }

    let n = oracle::WITNESSES.len();
    let mut storage = vec![19.0f32; 2 * (n + 1)];
    for row in 0..2 {
        for (column, &bits) in oracle::WITNESSES.iter().enumerate() {
            storage[row * (n + 1) + column + 1] = f32::from_bits(bits);
        }
    }
    let view = Tensor::<R>::from_slice(&storage, &[2, n + 1], client.device())?
        .narrow(1, 1, n)?
        .transpose(0, 1)?;
    let original = input_bits(client, &view)?;
    let output = client.sqrt(&view)?;
    assert_eq!(output.shape(), &[n, 2]);
    assert!(!output.shares_storage_with(&view));
    check(
        &output,
        oracle::WITNESSES
            .iter()
            .flat_map(|&bits| [oracle::rounded_sqrt_bits(bits); 2]),
        "offset/transposed view",
    )?;
    assert_eq!(input_bits(client, &view)?, original);

    let broadcast = Tensor::<R>::from_slice(&[f32::from_bits(1)], &[1, 1], client.device())?
        .broadcast_to(&[3, 33])?;
    let output = client.sqrt(&broadcast)?;
    assert_eq!(output.shape(), &[3, 33]);
    check(
        &output,
        std::iter::repeat_n(oracle::rounded_sqrt_bits(1), 99),
        "broadcast view",
    )?;

    let scalar = Tensor::<R>::from_slice(&[2.0f32], &[], client.device())?;
    let output = client.sqrt(&scalar)?;
    assert_eq!(output.shape(), &[]);
    check(&output, [0x3fb5_04f3], "rank-zero input")?;
    let empty = Tensor::<R>::empty(&[2, 0, 3], DType::F32, client.device())?;
    let output = client.sqrt(&empty)?;
    assert_eq!(output.shape(), &[2, 0, 3]);
    assert_eq!(output.dtype(), DType::F32);
    assert!(output.device().is_same(client.device()));
    assert_eq!(output.numel(), 0);

    let exceptional = [
        0xff80_0000,
        0xbf80_0000,
        0x8000_0001,
        0x7fc0_1234,
        0x7f80_0001,
    ];
    let values: Vec<_> = exceptional
        .into_iter()
        .cycle()
        .take(65)
        .map(f32::from_bits)
        .collect();
    let output = client.sqrt(&Tensor::<R>::from_slice(&values, &[65], client.device())?)?;
    assert!(output.try_to_vec::<f32>()?.into_iter().all(f32::is_nan));
    Ok(())
}

#[test]
fn cpu_f32_square_root_preserves_exact_bits_across_dispatch_and_views() -> CheckResult {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    dispatch::<CpuRuntime, _>(&client)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_f32_square_root_preserves_exact_bits_across_dispatch_and_views() -> CheckResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device)?;
    dispatch::<CudaRuntime, _>(&client)
}
