use super::common::{backend_lock::with_cuda_backend, create_cpu_client};
use super::division_reference::CASES;
use numr::dtype::DType;
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, cpu::CpuRuntime, cuda::CudaRuntime};
use numr::tensor::Tensor;

fn evaluate<R, C>(client: &C, device: &R::Device) -> numr::error::Result<()>
where
    R: Runtime<DType = DType>,
    C: ScalarOps<R>,
{
    for case in CASES {
        let numerator = f32::from_bits(case.numerator);
        let denominator = f64::from(f32::from_bits(case.denominator));
        let contiguous = Tensor::<R>::from_slice(&[numerator; 3], &[3], device)?;
        let sliced = Tensor::<R>::from_slice(
            &[99.0, numerator, 99.0, numerator, 99.0, numerator],
            &[3, 2],
            device,
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1));
        assert!(!sliced.is_contiguous());
        for (input, route) in [(contiguous, "scalar contiguous"), (sliced, "scalar sliced")] {
            let output = client
                .div_scalar(&input, denominator)?
                .try_to_vec::<f32>()?;
            assert_eq!(output.len(), 3);
            for value in output {
                case.check(value, route);
            }
        }
    }
    Ok(())
}

#[test]
fn f32_division_scalar_preserves_nearest_even_bits_and_classification() -> numr::error::Result<()> {
    let (client, device) = create_cpu_client();
    evaluate::<CpuRuntime, _>(&client, &device)?;
    let mut result = Ok(());
    with_cuda_backend(|client, device| {
        result = evaluate::<CudaRuntime, _>(&client, &device);
    });
    result
}
