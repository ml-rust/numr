//! Compile-and-run test for `dispatch_dtype!` from OUTSIDE the numr crate.
//!
//! An integration test under `tests/` compiles as its own crate, which is
//! exactly the condition under which `DType` is `#[non_exhaustive]`. This
//! reproduces the failure downstream crates hit (`E0004: non-exhaustive
//! patterns: `_` not covered`) and proves the macro's wildcard arm fixes it.

use numr::dispatch_dtype;
use numr::dtype::DType;
use numr::error::Error;

fn size_of_dtype(dtype: DType) -> Result<usize, Error> {
    dispatch_dtype!(dtype, T => {
        Ok(std::mem::size_of::<T>())
    }, "size_of_dtype")
}

#[test]
fn dispatch_dtype_macro_expands_outside_numr() {
    assert_eq!(size_of_dtype(DType::F64).unwrap(), 8);
    assert_eq!(size_of_dtype(DType::F32).unwrap(), 4);
    assert_eq!(size_of_dtype(DType::I64).unwrap(), 8);
    assert_eq!(size_of_dtype(DType::I32).unwrap(), 4);
    assert_eq!(size_of_dtype(DType::I16).unwrap(), 2);
    assert_eq!(size_of_dtype(DType::I8).unwrap(), 1);
    assert_eq!(size_of_dtype(DType::U64).unwrap(), 8);
    assert_eq!(size_of_dtype(DType::U32).unwrap(), 4);
    assert_eq!(size_of_dtype(DType::U16).unwrap(), 2);
    assert_eq!(size_of_dtype(DType::U8).unwrap(), 1);
}

#[test]
fn dispatch_dtype_macro_computes_correct_values_outside_numr() {
    // The macro's arms must expand for every `DType` variant it lists
    // (including the always-compiled Complex64/Complex128 arms), so this
    // body only uses `size_of::<T>()`, which is valid for every bound type.
    fn byte_len(dtype: DType, count: usize) -> Result<usize, Error> {
        dispatch_dtype!(dtype, T => {
            Ok(std::mem::size_of::<T>() * count)
        }, "byte_len")
    }

    assert_eq!(byte_len(DType::I32, 4).unwrap(), 16);
    assert_eq!(byte_len(DType::U8, 4).unwrap(), 4);
    assert_eq!(byte_len(DType::I64, 4).unwrap(), 32);
    assert_eq!(byte_len(DType::F32, 4).unwrap(), 16);
}

#[test]
fn dispatch_dtype_macro_wildcard_arm_reports_unsupported_dtype_outside_numr() {
    // Bool has no concrete Rust type in dispatch_dtype! and must hit the
    // wildcard arm rather than fail to compile with a non-exhaustive match.
    let err = size_of_dtype(DType::Bool).unwrap_err();
    match err {
        Error::UnsupportedDType { dtype, op } => {
            assert_eq!(dtype, DType::Bool);
            assert_eq!(op, "size_of_dtype");
        }
        other => panic!("expected Error::UnsupportedDType, got {other:?}"),
    }
}
