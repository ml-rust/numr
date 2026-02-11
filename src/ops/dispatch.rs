//! DType dispatch utilities for GPU backends
//!
//! This module provides the `dispatch_dtype!` macro for runtime type dispatch
//! in GPU backend operations. It is used by CUDA and WGPU backends to convert
//! from `DType` enum to concrete generic types.
//!
//! # Usage
//!
//! ```ignore
//! fn my_operation(dtype: DType) -> Result<()> {
//!     dispatch_dtype!(dtype, T => {
//!         // T is now a concrete type (f32, f64, i32, etc.)
//!         let size = std::mem::size_of::<T>();
//!         Ok(())
//!     }, "my_operation");
//!
//!     unreachable!()
//! }
//! ```
//!
//! # Macro Details
//!
//! The `dispatch_dtype!` macro takes a `DType` value and executes a code block
//! with `T` bound to the corresponding Rust type. It handles all supported dtypes
//! including conditional compilation for F16/BF16.
//!
//! ## Arguments
//!
//! * `$dtype` - Expression evaluating to a `DType` value
//! * `$T` - Identifier to bind to the concrete type in the body
//! * `$body` - Code block to execute with `T` bound
//! * `$error_op` - Operation name for error messages (used when dtype is unsupported)
//!
//! ## Supported Types
//!
//! - `F64` -> `f64`
//! - `F32` -> `f32`
//! - `F16` -> `half::f16` (requires "f16" feature)
//! - `BF16` -> `half::bf16` (requires "f16" feature)
//! - `FP8E4M3` -> `crate::dtype::FP8E4M3` (always available)
//! - `FP8E5M2` -> `crate::dtype::FP8E5M2` (always available)
//! - `I64` -> `i64`
//! - `I32` -> `i32`
//! - `I16` -> `i16`
//! - `I8` -> `i8`
//! - `U64` -> `u64`
//! - `U32` -> `u32`
//! - `U16` -> `u16`
//! - `U8` -> `u8`
//! - `Bool` -> Returns `UnsupportedDType` error

// Feature-Gated Type Dispatch Helpers
//
// These two parameterized macros replace what would otherwise be 4 separate
// macros (one per type). The type is passed as a parameter, reducing duplication.
//
// - dispatch_f16_type!: For F16/BF16 types (requires "f16" feature)
// - dispatch_fp8_type!: For FP8E4M3/FP8E5M2 types (always available)

/// Internal helper macro to dispatch types requiring the "f16" feature.
/// Parameterized by type to avoid duplicating macro for F16 vs BF16.
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_f16_type {
    ($T:ident, $body:block, $dtype:expr, $error_op:expr, $type:ty) => {{
        #[cfg(feature = "f16")]
        {
            type $T = $type;
            $body
        }
        #[cfg(not(feature = "f16"))]
        {
            return Err($crate::error::Error::FeatureRequired {
                dtype: $dtype,
                feature: "f16",
            });
        }
    }};
}

/// Internal helper macro to dispatch types requiring the "fp8" feature.
/// Parameterized by type to avoid duplicating macro for FP8E4M3 vs FP8E5M2.
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_fp8_type {
    ($T:ident, $body:block, $dtype:expr, $error_op:expr, $type:ty) => {{
        #[cfg(feature = "fp8")]
        {
            type $T = $type;
            $body
        }
        #[cfg(not(feature = "fp8"))]
        {
            return Err($crate::error::Error::FeatureRequired {
                dtype: $dtype,
                feature: "fp8",
            });
        }
    }};
}

/// Macro for runtime dtype dispatch to typed operations.
///
/// This macro takes a `DType` value and executes a code block with `T` bound
/// to the corresponding Rust type. Feature-gated types (F16, BF16, FP8) use
/// parameterized helper macros to avoid code duplication.
#[macro_export]
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:block, $error_op:expr) => {
        match $dtype {
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F16 => {
                $crate::dispatch_f16_type!($T, $body, $dtype, $error_op, half::f16)
            }
            $crate::dtype::DType::BF16 => {
                $crate::dispatch_f16_type!($T, $body, $dtype, $error_op, half::bf16)
            }
            $crate::dtype::DType::FP8E4M3 => {
                $crate::dispatch_fp8_type!($T, $body, $dtype, $error_op, $crate::dtype::FP8E4M3)
            }
            $crate::dtype::DType::FP8E5M2 => {
                $crate::dispatch_fp8_type!($T, $body, $dtype, $error_op, $crate::dtype::FP8E5M2)
            }
            $crate::dtype::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::dtype::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::dtype::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::dtype::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::dtype::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::dtype::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::dtype::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::dtype::DType::Bool => {
                return Err($crate::error::Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $error_op,
                })
            }
            $crate::dtype::DType::Complex64 => {
                type $T = $crate::dtype::Complex64;
                $body
            }
            $crate::dtype::DType::Complex128 => {
                type $T = $crate::dtype::Complex128;
                $body
            }
        }
    };
}
