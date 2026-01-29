//! WGSL shader generation for complex number operations
//!
//! Complex64 is represented as vec2<f32> where:
//! - .x = real part
//! - .y = imaginary part

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for complex conjugate operation.
///
/// Input: Complex64 (vec2<f32>)
/// Output: Complex64 (vec2<f32>)
/// Operation: conj(a + bi) = a - bi
pub fn generate_conj_shader() -> Result<String> {
    Ok(r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn conj_complex64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        let val = input[idx];
        output[idx] = vec2<f32>(val.x, -val.y);  // Real stays same, imaginary flips sign
    }
}
"#
    .to_string())
}

/// Generate WGSL shader for extracting real part.
///
/// Input: Complex64 (vec2<f32>)
/// Output: F32 (f32)
/// Operation: real(a + bi) = a
pub fn generate_real_shader() -> Result<String> {
    Ok(r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn real_complex64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        output[idx] = input[idx].x;  // Extract real component
    }
}
"#
    .to_string())
}

/// Generate WGSL shader for extracting imaginary part.
///
/// Input: Complex64 (vec2<f32>)
/// Output: F32 (f32)
/// Operation: imag(a + bi) = b
pub fn generate_imag_shader() -> Result<String> {
    Ok(r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn imag_complex64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        output[idx] = input[idx].y;  // Extract imaginary component
    }
}
"#
    .to_string())
}

/// Generate WGSL shader for computing phase angle.
///
/// Input: Complex64 (vec2<f32>)
/// Output: F32 (f32)
/// Operation: angle(a + bi) = atan2(b, a)
pub fn generate_angle_shader() -> Result<String> {
    Ok(r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn angle_complex64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        let val = input[idx];
        output[idx] = atan2(val.y, val.x);  // Phase angle in radians [-π, π]
    }
}
"#
    .to_string())
}

/// Generate WGSL shader for computing phase angle of real numbers.
///
/// Input: F32 (real numbers)
/// Output: F32
/// Operation: angle(x) = 0 if x >= 0, π if x < 0
///
/// Note: WGSL does not have a standard library with mathematical constants,
/// so PI must be defined as a literal constant in the shader source.
/// This matches Rust's std::f32::consts::PI value.
pub fn generate_angle_real_shader() -> Result<String> {
    Ok(r#"
struct Params {
    numel: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// PI constant (WGSL has no standard math library, so this is defined literally)
// Value matches std::f32::consts::PI exactly (f32 precision: ~7 significant digits)
const PI: f32 = 3.14159265f;

@compute @workgroup_size(256)
fn angle_real_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.numel) {
        let val = input[idx];
        output[idx] = select(0.0, PI, val < 0.0);  // 0 if x >= 0, π if x < 0
    }
}
"#
    .to_string())
}

/// Get the shader generator for a complex operation.
pub fn get_complex_shader_generator(op: &str) -> Result<fn() -> Result<String>> {
    match op {
        "conj" => Ok(generate_conj_shader),
        "real" => Ok(generate_real_shader),
        "imag" => Ok(generate_imag_shader),
        "angle" => Ok(generate_angle_shader),
        _ => Err(Error::Internal(format!(
            "Unknown complex operation: {}",
            op
        ))),
    }
}

/// Validate dtype for complex operations.
pub fn validate_complex_dtype(dtype: DType, op: &str) -> Result<()> {
    // WebGPU only supports Complex64 (no F64 support)
    if dtype != DType::Complex64 {
        let op_static: &'static str = match op {
            "conj" => "conj",
            "real" => "real",
            "imag" => "imag",
            "angle" => "angle",
            _ => "complex_op",
        };
        return Err(Error::UnsupportedDType {
            dtype,
            op: op_static,
        });
    }
    Ok(())
}

/// Get output dtype for complex operation.
pub fn complex_output_dtype(input_dtype: DType, op: &str) -> Result<DType> {
    validate_complex_dtype(input_dtype, op)?;

    match op {
        "conj" => Ok(DType::Complex64),              // Same as input
        "real" | "imag" | "angle" => Ok(DType::F32), // Extract float component
        _ => Err(Error::Internal(format!(
            "Unknown complex operation: {}",
            op
        ))),
    }
}
