use super::common::{backend_lock::with_cuda_backend, create_cpu_client};
use super::division_reference::{CASES, DivisionCase};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, cpu::CpuRuntime, cuda::CudaRuntime};
use numr::tensor::Tensor;

#[derive(Clone, Copy)]
enum Route {
    Contiguous,
    Sliced,
    Transposed,
    TrailingBroadcast,
    GeneralBroadcast,
}

fn evaluate<R, C>(client: &C, device: &R::Device, route: Route) -> numr::error::Result<()>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R>,
{
    let numerators = CASES
        .iter()
        .map(|case| f32::from_bits(case.numerator))
        .collect::<Vec<_>>();
    let denominators = CASES
        .iter()
        .map(|case| f32::from_bits(case.denominator))
        .collect::<Vec<_>>();
    let make = |values: &[f32], shape: &[usize]| Tensor::<R>::from_slice(values, shape, device);
    let (a, b, expected, label) = match route {
        Route::Contiguous => (
            make(&numerators, &[32])?,
            make(&denominators, &[32])?,
            CASES.to_vec(),
            "contiguous",
        ),
        Route::Sliced => {
            let a = numerators
                .iter()
                .flat_map(|value| [99.0, *value])
                .collect::<Vec<_>>();
            let b = denominators
                .iter()
                .flat_map(|value| [99.0, *value])
                .collect::<Vec<_>>();
            let a = make(&a, &[32, 2])?.narrow(1, 1, 1)?.squeeze(Some(1));
            let b = make(&b, &[32, 2])?.narrow(1, 1, 1)?.squeeze(Some(1));
            assert!(!a.is_contiguous() && !b.is_contiguous());
            (a, b, CASES.to_vec(), "sliced")
        }
        Route::Transposed => {
            let a = make(&numerators, &[4, 8])?.transpose(0, 1)?;
            let b = make(&denominators, &[4, 8])?.transpose(0, 1)?;
            assert!(!a.is_contiguous() && !b.is_contiguous());
            let expected = (0..8)
                .flat_map(|column| (0..4).map(move |row| CASES[row * 8 + column]))
                .collect();
            (a, b, expected, "transposed")
        }
        Route::TrailingBroadcast => (
            make(&numerators.repeat(2), &[2, 32])?,
            make(&denominators, &[1, 32])?,
            CASES.repeat(2),
            "trailing broadcast",
        ),
        Route::GeneralBroadcast => {
            let a = numerators
                .iter()
                .flat_map(|value| [*value; 3])
                .collect::<Vec<_>>();
            let expected = CASES.iter().flat_map(|case| [*case; 3]).collect();
            (
                make(&a, &[32, 3])?,
                make(&denominators, &[32, 1])?,
                expected,
                "general broadcast",
            )
        }
    };
    let actual = client.div(&a, &b)?.try_to_vec::<f32>()?;
    check(&actual, &expected, label);
    Ok(())
}

fn check(actual: &[f32], expected: &[DivisionCase], route: &str) {
    assert_eq!(actual.len(), expected.len(), "{route} output length");
    for (&actual, &case) in actual.iter().zip(expected) {
        case.check(actual, route);
    }
}

fn backends(route: Route) -> numr::error::Result<()> {
    let (client, device) = create_cpu_client();
    evaluate::<CpuRuntime, _>(&client, &device, route)?;
    let mut result = Ok(());
    with_cuda_backend(|client, device| {
        result = evaluate::<CudaRuntime, _>(&client, &device, route);
    });
    result
}

#[test]
fn f32_division_contiguous_preserves_nearest_even_bits() -> numr::error::Result<()> {
    backends(Route::Contiguous)
}
#[test]
fn f32_division_sliced_preserves_nearest_even_bits() -> numr::error::Result<()> {
    backends(Route::Sliced)
}
#[test]
fn f32_division_transposed_preserves_nearest_even_bits() -> numr::error::Result<()> {
    backends(Route::Transposed)
}
#[test]
fn f32_division_trailing_broadcast_preserves_nearest_even_bits() -> numr::error::Result<()> {
    backends(Route::TrailingBroadcast)
}
#[test]
fn f32_division_general_broadcast_preserves_nearest_even_bits() -> numr::error::Result<()> {
    backends(Route::GeneralBroadcast)
}
