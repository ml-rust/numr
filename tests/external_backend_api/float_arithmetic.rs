use super::{MockRuntime, arithmetic_types::DTYPES};
use numr::prelude::{FloatArithmeticContract, FusedMulAddContract, Runtime};

#[test]
fn external_runtime_defaults_to_backend_defined_arithmetic() {
    for dtype in DTYPES {
        assert_eq!(
            MockRuntime::float_arithmetic_contract(dtype),
            FloatArithmeticContract::BackendDefined,
            "dtype={dtype:?}"
        );
    }
}

#[test]
fn external_runtime_defaults_to_backend_defined_fused_mul_add() {
    for dtype in DTYPES {
        assert_eq!(
            MockRuntime::fused_mul_add_contract(dtype),
            FusedMulAddContract::BackendDefined,
            "dtype={dtype:?}"
        );
    }
}
