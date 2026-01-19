//! Common helper functions for WGSL shader generation

use crate::dtype::DType;
use crate::error::{Error, Result};

/// WGSL type name for a given DType
pub fn wgsl_type(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::I32 => Ok("i32"),
        DType::U32 => Ok("u32"),
        DType::F16 => Ok("f16"), // Requires extension
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "wgpu_shader",
        }),
    }
}

/// Short suffix for entry point names (e.g., "add_f32", "add_i32")
pub fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::I32 => Ok("i32"),
        DType::U32 => Ok("u32"),
        DType::F16 => Ok("f16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "wgpu_shader",
        }),
    }
}

/// Check if dtype is supported by WebGPU
pub fn is_wgpu_supported(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::I32 | DType::U32 | DType::F16)
}

/// Check if dtype is a float type in WGSL
pub fn is_wgsl_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16)
}

/// Check if dtype is an integer type in WGSL
pub fn is_wgsl_int(dtype: DType) -> bool {
    matches!(dtype, DType::I32 | DType::U32)
}
