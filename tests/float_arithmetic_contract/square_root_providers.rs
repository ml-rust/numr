use super::arithmetic_types::DTYPES;
use numr::dtype::DType;
use numr::runtime::{Runtime, SquareRootContract, cpu::CpuRuntime};

fn f64_square_root_types<R: Runtime<DType = DType>>() {
    for dtype in DTYPES {
        let expected = match dtype {
            DType::F64 => SquareRootContract::Ieee754RoundToNearest,
            _ => SquareRootContract::BackendDefined,
        };
        assert_eq!(
            R::square_root_contract(dtype),
            expected,
            "runtime={},dtype={dtype:?}",
            R::name()
        );
    }
}

#[test]
fn cpu_advertises_only_f64_square_root() {
    f64_square_root_types::<CpuRuntime>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_advertises_only_f64_square_root() {
    f64_square_root_types::<numr::runtime::cuda::CudaRuntime>();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_keeps_square_root_backend_defined() {
    for dtype in DTYPES {
        assert_eq!(
            numr::runtime::wgpu::WgpuRuntime::square_root_contract(dtype),
            SquareRootContract::BackendDefined,
            "dtype={dtype:?}"
        );
    }
}

#[test]
fn square_root_contract_is_shared_through_public_reexports() {
    let trait_export: numr::runtime::traits::SquareRootContract =
        SquareRootContract::BackendDefined;
    let prelude_export: numr::prelude::SquareRootContract = trait_export;
    assert_eq!(prelude_export, SquareRootContract::BackendDefined);
    assert_eq!(
        SquareRootContract::default(),
        SquareRootContract::BackendDefined
    );
}
