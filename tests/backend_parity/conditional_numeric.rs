//! Check numeric predicate truth and exact selection across real dtypes.

#[cfg(feature = "cuda")]
use crate::common::backend_lock::with_cuda_backend;
use crate::common::{DTypeDomain, create_cpu_client, parity_dtypes};
use numr::{
    dtype::DType,
    error::Result,
    ops::{CompareOps, ConditionalOps, TensorOps, TypeConversionOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

fn tensor<R>(client: &R::Client, values: &[f64], shape: &[usize], dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R> + RuntimeClient<R>,
{
    let input = Tensor::from_slice(values, shape, client.device())?;
    if dtype == DType::F64 {
        Ok(input)
    } else {
        client.cast(&input, dtype)
    }
}

fn view<R>(client: &R::Client, values: &[f64], dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R> + RuntimeClient<R>,
{
    let storage = values
        .iter()
        .flat_map(|&value| [0.0, value])
        .collect::<Vec<_>>();
    Ok(tensor::<R>(client, &storage, &[values.len(), 2], dtype)?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)))
}

fn check<R>(
    client: &R::Client,
    output: &Tensor<R>,
    dtype: DType,
    shape: &[usize],
    expected: &[f64],
) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    assert_eq!(output.dtype(), dtype);
    assert_eq!(output.shape(), shape);
    // These matrix values are exact in every tested dtype. Wide integers use native I64 assertions below.
    assert_eq!(
        client.cast(output, DType::F64)?.try_to_vec::<f64>()?,
        expected
    );
    Ok(())
}

fn matrix<R>(client: &R::Client, backend: &str) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client:
        TensorOps<R> + ConditionalOps<R> + CompareOps<R> + TypeConversionOps<R> + RuntimeClient<R>,
{
    let dtypes = parity_dtypes(DTypeDomain::AllNumeric, backend);
    for &condition_dtype in &dtypes {
        for &value_dtype in &dtypes {
            let condition = tensor::<R>(client, &[0.0, 1.0, 0.0, 2.0], &[4], condition_dtype)?;
            let x = tensor::<R>(client, &[1.0, 3.0, 5.0, 7.0], &[4], value_dtype)?;
            let y = tensor::<R>(client, &[2.0, 4.0, 6.0, 8.0], &[4], value_dtype)?;
            let output = client.where_cond(&condition, &x, &y)?;
            check::<R>(client, &output, value_dtype, &[4], &[2.0, 3.0, 6.0, 7.0])?;
            let condition = view::<R>(client, &[0.0, 1.0, 0.0, 2.0], condition_dtype)?;
            let x = view::<R>(client, &[1.0, 3.0, 5.0, 7.0], value_dtype)?;
            let y = view::<R>(client, &[2.0, 4.0, 6.0, 8.0], value_dtype)?;
            assert!(condition.offset() > 0 && !condition.is_contiguous());
            let output = client.where_cond(&condition, &x, &y)?;
            check::<R>(client, &output, value_dtype, &[4], &[2.0, 3.0, 6.0, 7.0])?;

            let condition = view::<R>(client, &[0.0, 2.0], condition_dtype)?.unsqueeze(1)?;
            let x = view::<R>(client, &[1.0, 3.0], value_dtype)?.unsqueeze(0)?;
            let y = view::<R>(client, &[2.0, 4.0], value_dtype)?.unsqueeze(1)?;
            let output = client.where_cond(&condition, &x, &y)?;
            check::<R>(client, &output, value_dtype, &[2, 2], &[2.0, 2.0, 1.0, 3.0])?;
        }
    }
    for dtype in [DType::F32, DType::F64] {
        float_predicates::<R>(client, dtype)?;
    }
    integer_predicates::<R>(client)?;
    scalar_and_empty::<R>(client)
}

fn float_predicates<R>(client: &R::Client, dtype: DType) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client:
        TensorOps<R> + ConditionalOps<R> + CompareOps<R> + TypeConversionOps<R> + RuntimeClient<R>,
{
    let condition = view::<R>(
        client,
        &[
            0.0,
            -0.0,
            0.25,
            -0.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -256.0,
        ],
        dtype,
    )?;
    let base = 1_i64 << 60;
    let x = [
        base + 1,
        base + 3,
        base + 5,
        base + 7,
        base + 9,
        base + 11,
        base + 13,
        base + 15,
    ];
    let y = x.map(|value| -value);
    let x = Tensor::from_slice(&x, &[8], client.device())?;
    let y = Tensor::from_slice(&y, &[8], client.device())?;
    let expected = [
        -base - 1,
        -base - 3,
        base + 5,
        base + 7,
        base + 9,
        base + 11,
        base + 13,
        base + 15,
    ];
    let output = client.where_cond(&condition, &x, &y)?;
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.try_to_vec::<i64>()?, expected);
    let zero = Tensor::zeros(&[8], dtype, client.device())?;
    let comparison = client.ne(&condition, &zero)?;
    let output = client.where_cond(&comparison, &x, &y)?;
    assert_eq!(output.try_to_vec::<i64>()?, expected);

    let condition = tensor::<R>(client, &[-0.0, -0.25], &[2, 1], dtype)?;
    let x = Tensor::from_slice(&[base + 1, base + 3], &[1, 2], client.device())?;
    let y = Tensor::from_slice(&[-base - 1, -base - 3], &[2, 1], client.device())?;
    let output = client.where_cond(&condition, &x, &y)?;
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        output.try_to_vec::<i64>()?,
        [-base - 1, -base - 1, base + 1, base + 3]
    );
    Ok(())
}

fn integer_predicates<R>(client: &R::Client) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: ConditionalOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let base = 1_i64 << 60;
    let condition = Tensor::from_slice(&[0_i64, -256, base + 1, 0], &[4], client.device())?;
    let x = Tensor::from_slice(
        &[base + 1, base + 3, base + 5, base + 7],
        &[4],
        client.device(),
    )?;
    let y = Tensor::from_slice(
        &[-base - 1, -base - 3, -base - 5, -base - 7],
        &[4],
        client.device(),
    )?;
    let output = client.where_cond(&condition, &x, &y)?;
    assert_eq!(
        output.try_to_vec::<i64>()?,
        [-base - 1, base + 3, base + 5, -base - 7]
    );
    let zero = Tensor::zeros(&[4], DType::I64, client.device())?;
    let comparison = client.ne(&condition, &zero)?;
    let output = client.where_cond(&comparison, &x, &y)?;
    assert_eq!(
        output.try_to_vec::<i64>()?,
        [-base - 1, base + 3, base + 5, -base - 7]
    );
    Ok(())
}

fn scalar_and_empty<R>(client: &R::Client) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: ConditionalOps<R> + RuntimeClient<R>,
{
    let base = 1_i64 << 60;
    let condition = Tensor::from_slice(&[0.25_f64], &[], client.device())?;
    let x = Tensor::from_slice(&[base + 1], &[], client.device())?;
    let y = Tensor::from_slice(&[-base - 1], &[], client.device())?;
    let output = client.where_cond(&condition, &x, &y)?;
    assert!(output.shape().is_empty());
    assert_eq!(output.try_to_vec::<i64>()?, [base + 1]);
    let condition = Tensor::from_slice(&[] as &[f64], &[0], client.device())?;
    let x = Tensor::from_slice(&[] as &[i64], &[0], client.device())?;
    let y = Tensor::from_slice(&[] as &[i64], &[0], client.device())?;
    let output = client.where_cond(&condition, &x, &y)?;
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.shape(), &[0]);
    assert!(output.try_to_vec::<i64>()?.is_empty());
    Ok(())
}

#[test]
fn cpu_numeric_conditions_select_all_real_value_dtypes() -> Result<()> {
    use numr::runtime::cpu::CpuRuntime;
    let (client, _) = create_cpu_client();
    matrix::<CpuRuntime>(&client, "cpu")
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_numeric_conditions_select_all_real_value_dtypes() {
    use numr::runtime::cuda::CudaRuntime;
    with_cuda_backend(|client, _| {
        let result = matrix::<CudaRuntime>(&client, "cuda");
        assert!(result.is_ok(), "CUDA numeric condition matrix: {result:?}");
    });
}
