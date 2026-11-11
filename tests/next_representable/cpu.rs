use numr::prelude::*;

#[test]
fn f32_matches_standard_neighbors_and_preserves_nan_bits() -> Result<()> {
    let device = CpuDevice::new();
    super::suite::f32_suite(&CpuRuntime::default_client(&device), &device)?;
    Ok(())
}

#[test]
fn f64_matches_standard_neighbors_and_preserves_nan_bits() -> Result<()> {
    let device = CpuDevice::new();
    super::suite::f64_suite(&CpuRuntime::default_client(&device), &device)?;
    Ok(())
}

#[test]
fn unsupported_dtypes_reject_empty_and_nonempty_inputs() -> Result<()> {
    let device = CpuDevice::new();
    super::suite::unsupported(&CpuRuntime::default_client(&device), &device, false)
}

#[test]
fn configured_parallel_client_preserves_neighbor_bits_and_storage_independence() -> Result<()> {
    use super::fixtures::{f32_bits, f64_bits, reference_f32, reference_f64};
    use numr::runtime::cpu::ParallelismConfig;

    let device = CpuDevice::new();
    let base = CpuRuntime::default_client(&device);
    let base_config = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    // Exceed the parallel work gate and leave a partial odd-sized chunk.
    let len = 32 * 1024 + 37;

    macro_rules! check {
        ($float:ty, $fixture:ident, $reference:ident) => {{
            let bits: Vec<_> = $fixture().into_iter().cycle().take(len).collect();
            let values: Vec<_> = bits.iter().copied().map(<$float>::from_bits).collect();
            let input = Tensor::<CpuRuntime>::from_slice(&values, &[len], &device)?;
            let read = |tensor: &Tensor<CpuRuntime>| -> Result<Vec<_>> {
                Ok(tensor
                    .try_to_vec::<$float>()?
                    .into_iter()
                    .map(<$float>::to_bits)
                    .collect())
            };
            let up = client.next_up(&input)?;
            let down = client.next_down(&input)?;
            let expected_up: Vec<_> = bits.iter().map(|&bits| $reference(bits, true)).collect();
            let expected_down: Vec<_> = bits.iter().map(|&bits| $reference(bits, false)).collect();
            assert_eq!(read(&up)?, expected_up);
            assert_eq!(read(&down)?, expected_down);
            assert_eq!(read(&input)?, bits);
            for output in [&up, &down] {
                assert_eq!(output.shape(), input.shape());
                assert_eq!(output.dtype(), input.dtype());
                assert!(output.device().is_same(&device));
                assert!(!output.shares_storage_with(&input));
            }
            assert!(!up.shares_storage_with(&down));
            let zeros = (0 as $float).to_bits().to_ne_bytes().repeat(len);
            CpuRuntime::copy_to_device(&zeros, up.ptr(), &device)?;
            assert_eq!(read(&input)?, bits);
            assert_eq!(read(&down)?, expected_down);
            let replacement = (7 as $float).to_bits().to_ne_bytes().repeat(len);
            CpuRuntime::copy_to_device(&replacement, input.ptr(), &device)?;
            assert_eq!(read(&up)?, vec![(0 as $float).to_bits(); len]);
            assert_eq!(read(&down)?, expected_down);
            assert_eq!(client.parallelism(), config);
            assert_eq!(base.parallelism(), base_config);
        }};
    }

    check!(f32, f32_bits, reference_f32);
    check!(f64, f64_bits, reference_f64);
    Ok(())
}
