//! `CorrectlyRoundedOps` across backends.
//!
//! CUDA must equal CPU bit for bit. The CPU tensor ops equal the host functions, which
//! the `correctly_rounded` test target checks against MPFR.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod inputs;

#[cfg(feature = "wgpu")]
use super::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use numr::prelude::*;

#[cfg(feature = "wgpu")]
#[test]
fn correctly_rounded_wgpu_rejects_every_dtype() {
    with_wgpu_backend_or_skip(|client, device| {
        for dtype in [DType::F64, DType::F32, DType::I32, DType::U32] {
            let input = Tensor::<WgpuRuntime>::empty(&[4], dtype, &device);
            let Ok(input) = input else {
                panic!("allocate {dtype:?}: {input:?}");
            };
            let results = [
                ("cr_exp", client.cr_exp(&input)),
                ("cr_log", client.cr_log(&input)),
                ("cr_log1p", client.cr_log1p(&input)),
                ("cr_acos", client.cr_acos(&input)),
                ("cr_pow_scalar", client.cr_pow_scalar(&input, 2.0)),
            ];
            for (name, result) in results {
                match result {
                    Err(Error::UnsupportedDType { dtype: got, op }) if dtype != DType::F64 => {
                        assert_eq!((got, op), (dtype, name));
                    }
                    Err(Error::BackendLimitation {
                        backend,
                        operation,
                        reason,
                    }) if dtype == DType::F64 => {
                        assert_eq!((backend, operation), ("wgpu", name));
                        assert!(reason.contains("CorrectlyRoundedOps"), "{reason}");
                    }
                    other => panic!("{name} {dtype:?}: {other:?}"),
                }
            }
        }
    });
}
