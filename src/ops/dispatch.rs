//! DType dispatch utilities for GPU backends
//!
//! This module provides the `dispatch_dtype!` macro for runtime type dispatch
//! in GPU backend operations. It is used by CUDA and WGPU backends to convert
//! from `DType` enum to concrete generic types.
//!
//! # Usage
//!
//! ```ignore
//! use numr::ops::dispatch_dtype;
//!
//! fn my_operation(dtype: DType) -> Result<()> {
//!     dispatch_dtype!(dtype, T => {
//!         // T is now a concrete type (f32, f64, i32, etc.)
//!         let value: T = T::zero();
//!         Ok(())
//!     }, "my_operation");
//!
//!     unreachable!()
//! }
//! ```

/// Macro for runtime dtype dispatch to typed operations.
///
/// This macro takes a `DType` value and executes a code block with `T` bound
/// to the corresponding Rust type. It handles all supported dtypes including
/// conditional compilation for F16/BF16.
///
/// # Arguments
///
/// * `$dtype` - Expression evaluating to a `DType` value
/// * `$T` - Identifier to bind to the concrete type in the body
/// * `$body` - Code block to execute with `T` bound
/// * `$error_op` - Operation name for error messages (used when dtype is unsupported)
///
/// # Example
///
/// ```ignore
/// dispatch_dtype!(tensor.dtype(), T => {
///     let data: Vec<T> = tensor.to_vec();
///     // Process typed data...
///     return Ok(result);
/// }, "my_operation");
/// ```
///
/// # Supported Types
///
/// - `F64` -> `f64`
/// - `F32` -> `f32`
/// - `F16` -> `half::f16` (requires "f16" feature)
/// - `BF16` -> `half::bf16` (requires "f16" feature)
/// - `FP8E4M3` -> `crate::dtype::FP8E4M3` (requires "fp8" feature)
/// - `FP8E5M2` -> `crate::dtype::FP8E5M2` (requires "fp8" feature)
/// - `I64` -> `i64`
/// - `I32` -> `i32`
/// - `I16` -> `i16`
/// - `I8` -> `i8`
/// - `U64` -> `u64`
/// - `U32` -> `u32`
/// - `U16` -> `u16`
/// - `U8` -> `u8`
/// - `Bool` -> Returns `UnsupportedDType` error
/// Internal helper macro to dispatch F16 type (with or without feature)
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_f16 {
    ($T:ident, $body:block, $dtype:expr, $error_op:expr) => {{
        #[cfg(feature = "f16")]
        {
            type $T = half::f16;
            $body
        }
        #[cfg(not(feature = "f16"))]
        {
            return Err($crate::error::Error::UnsupportedDType {
                dtype: $dtype,
                op: $error_op,
            });
        }
    }};
}

/// Internal helper macro to dispatch BF16 type (with or without feature)
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_bf16 {
    ($T:ident, $body:block, $dtype:expr, $error_op:expr) => {{
        #[cfg(feature = "f16")]
        {
            type $T = half::bf16;
            $body
        }
        #[cfg(not(feature = "f16"))]
        {
            return Err($crate::error::Error::UnsupportedDType {
                dtype: $dtype,
                op: $error_op,
            });
        }
    }};
}

/// Internal helper macro to dispatch FP8E4M3 type (with or without feature)
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_fp8e4m3 {
    ($T:ident, $body:block, $dtype:expr, $error_op:expr) => {{
        #[cfg(feature = "fp8")]
        {
            type $T = $crate::dtype::FP8E4M3;
            $body
        }
        #[cfg(not(feature = "fp8"))]
        {
            return Err($crate::error::Error::UnsupportedDType {
                dtype: $dtype,
                op: $error_op,
            });
        }
    }};
}

/// Internal helper macro to dispatch FP8E5M2 type (with or without feature)
#[macro_export]
#[doc(hidden)]
macro_rules! dispatch_fp8e5m2 {
    ($T:ident, $body:block, $dtype:expr, $error_op:expr) => {{
        #[cfg(feature = "fp8")]
        {
            type $T = $crate::dtype::FP8E5M2;
            $body
        }
        #[cfg(not(feature = "fp8"))]
        {
            return Err($crate::error::Error::UnsupportedDType {
                dtype: $dtype,
                op: $error_op,
            });
        }
    }};
}

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
                $crate::dispatch_f16!($T, $body, $dtype, $error_op)
            }
            $crate::dtype::DType::BF16 => {
                $crate::dispatch_bf16!($T, $body, $dtype, $error_op)
            }
            $crate::dtype::DType::FP8E4M3 => {
                $crate::dispatch_fp8e4m3!($T, $body, $dtype, $error_op)
            }
            $crate::dtype::DType::FP8E5M2 => {
                $crate::dispatch_fp8e5m2!($T, $body, $dtype, $error_op)
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
        }
    };
}

// Re-export at module level for qualified access
pub use dispatch_dtype;
