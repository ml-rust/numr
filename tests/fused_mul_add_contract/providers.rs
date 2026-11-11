use super::arithmetic_types::DTYPES;
use numr::dtype::DType;
use numr::runtime::{FusedMulAddContract, Runtime, cpu::CpuRuntime};

#[test]
fn cpu_advertises_only_f32_and_f64_fused_mul_add() {
    for dtype in DTYPES {
        let expected = match dtype {
            DType::F32 | DType::F64 => FusedMulAddContract::Ieee754Fused,
            _ => FusedMulAddContract::BackendDefined,
        };
        assert_eq!(
            CpuRuntime::fused_mul_add_contract(dtype),
            expected,
            "dtype={dtype:?}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_advertises_only_f32_and_f64_fused_mul_add() {
    for dtype in DTYPES {
        assert_eq!(
            numr::runtime::cuda::CudaRuntime::fused_mul_add_contract(dtype),
            match dtype {
                DType::F32 | DType::F64 => FusedMulAddContract::Ieee754Fused,
                _ => FusedMulAddContract::BackendDefined,
            },
            "dtype={dtype:?}"
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_keeps_fused_mul_add_backend_defined() {
    for dtype in DTYPES {
        assert_eq!(
            numr::runtime::wgpu::WgpuRuntime::fused_mul_add_contract(dtype),
            FusedMulAddContract::BackendDefined,
            "dtype={dtype:?}"
        );
    }
}

#[test]
fn fused_mul_add_contract_is_shared_through_public_reexports() {
    let trait_export: numr::runtime::traits::FusedMulAddContract =
        FusedMulAddContract::BackendDefined;
    let prelude_export: numr::prelude::FusedMulAddContract = trait_export;
    assert_eq!(prelude_export, FusedMulAddContract::BackendDefined);
    assert_eq!(
        FusedMulAddContract::default(),
        FusedMulAddContract::BackendDefined
    );
}
