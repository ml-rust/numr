use super::arithmetic_types::DTYPES;
use numr::dtype::DType;
use numr::runtime::{FloatArithmeticContract, Runtime, cpu::CpuRuntime};

fn ieee_basic_types<R: Runtime<DType = DType>>() {
    for dtype in DTYPES {
        let expected = match dtype {
            DType::F32 | DType::F64 => FloatArithmeticContract::Ieee754Basic,
            _ => FloatArithmeticContract::BackendDefined,
        };
        assert_eq!(
            R::float_arithmetic_contract(dtype),
            expected,
            "runtime={},dtype={dtype:?}",
            R::name()
        );
    }
}

#[test]
fn cpu_advertises_only_f32_and_f64_basic_arithmetic() {
    ieee_basic_types::<CpuRuntime>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_advertises_only_f32_and_f64_basic_arithmetic() {
    ieee_basic_types::<numr::runtime::cuda::CudaRuntime>();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_keeps_every_dtype_backend_defined() {
    for dtype in DTYPES {
        assert_eq!(
            numr::runtime::wgpu::WgpuRuntime::float_arithmetic_contract(dtype),
            FloatArithmeticContract::BackendDefined,
            "dtype={dtype:?}"
        );
    }
}

#[test]
fn arithmetic_contract_is_shared_through_public_reexports() {
    let trait_export: numr::runtime::traits::FloatArithmeticContract =
        FloatArithmeticContract::BackendDefined;
    let prelude_export: numr::prelude::FloatArithmeticContract = trait_export;
    assert_eq!(prelude_export, FloatArithmeticContract::BackendDefined);
}
