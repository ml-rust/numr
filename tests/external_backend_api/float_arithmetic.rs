use super::{MockRuntime, arithmetic_types::DTYPES};
use numr::prelude::{FloatArithmeticContract, Runtime};

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
