//! WGSL shader generation for where_cond (ternary conditional select)
//!
//! Generates shaders for: where_cond(condition, x, y) → output
//! where `output[i] = condition[i] != 0 ? x[i] : y[i]`
//!
//! Supports multiple condition dtypes (F32, I32, U32) and multiple output dtypes.

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for where_cond operation.
///
/// Creates kernels for both element-wise and broadcast where operations.
/// The condition is tested for non-zero: any non-zero value is treated as true.
///
/// # Arguments
///
/// * `cond_dtype` - Data type of condition tensor (F32, I32, U32)
/// * `out_dtype` - Data type of x, y, and output tensors
///
/// # Entry Points
///
/// * `where_cond_{cond_suffix}_{out_suffix}` - Element-wise where
/// * `where_broadcast_cond_{cond_suffix}_{out_suffix}` - Broadcast where
pub fn generate_where_cond_shader(cond_dtype: DType, out_dtype: DType) -> Result<String> {
    let cond_t = wgsl_type(cond_dtype)?;
    let out_t = wgsl_type(out_dtype)?;
    let cond_suffix = dtype_suffix(cond_dtype)?;
    let out_suffix = dtype_suffix(out_dtype)?;

    // Generate zero literal for comparison
    let zero_cmp = match cond_dtype {
        DType::F32 | DType::F16 => "0.0",
        DType::I32 | DType::U32 => "0",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: cond_dtype,
                op: "where_cond (condition dtype)",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated where_cond shader for condition={cond_t}, output={out_t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

// ============================================================================
// Element-wise where_cond
// ============================================================================

struct WhereParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> where_cond_arr: array<{cond_t}>;
@group(0) @binding(1) var<storage, read_write> where_x: array<{out_t}>;
@group(0) @binding(2) var<storage, read_write> where_y: array<{out_t}>;
@group(0) @binding(3) var<storage, read_write> where_out: array<{out_t}>;
@group(0) @binding(4) var<uniform> where_params: WhereParams;

@compute @workgroup_size(256)
fn where_cond_{cond_suffix}_{out_suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < where_params.numel) {{
        // Condition is true if non-zero
        let cond_val = where_cond_arr[idx] != {zero_cmp};
        where_out[idx] = select(where_y[idx], where_x[idx], cond_val);
    }}
}}

// ============================================================================
// Broadcast where_cond
// ============================================================================

struct WhereBroadcastParams {{
    numel: u32,
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
}}

@group(0) @binding(0) var<storage, read_write> bc_cond: array<{cond_t}>;
@group(0) @binding(1) var<storage, read_write> bc_x: array<{out_t}>;
@group(0) @binding(2) var<storage, read_write> bc_y: array<{out_t}>;
@group(0) @binding(3) var<storage, read_write> bc_out: array<{out_t}>;
@group(0) @binding(4) var<storage, read_write> cond_strides: array<u32>;
@group(0) @binding(5) var<storage, read_write> x_strides: array<u32>;
@group(0) @binding(6) var<storage, read_write> y_strides: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_shape: array<u32>;
@group(0) @binding(8) var<uniform> bc_params: WhereBroadcastParams;

@compute @workgroup_size(256)
fn where_broadcast_cond_{cond_suffix}_{out_suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= bc_params.numel) {{
        return;
    }}

    // Convert linear index to multi-dimensional coords and compute offsets
    var remaining = idx;
    var cond_offset: u32 = 0u;
    var x_offset: u32 = 0u;
    var y_offset: u32 = 0u;

    for (var d: u32 = 0u; d < bc_params.ndim; d = d + 1u) {{
        let dim_size = out_shape[d];
        let coord = remaining / compute_out_stride(d, bc_params.ndim);
        remaining = remaining % compute_out_stride(d, bc_params.ndim);

        cond_offset = cond_offset + coord * cond_strides[d];
        x_offset = x_offset + coord * x_strides[d];
        y_offset = y_offset + coord * y_strides[d];
    }}

    // Apply condition
    let cond_val = bc_cond[cond_offset] != {zero_cmp};
    bc_out[idx] = select(bc_y[y_offset], bc_x[x_offset], cond_val);
}}

// Helper function to compute output stride at dimension d
fn compute_out_stride(d: u32, ndim: u32) -> u32 {{
    var stride: u32 = 1u;
    for (var i: u32 = d + 1u; i < ndim; i = i + 1u) {{
        stride = stride * out_shape[i];
    }}
    return stride;
}}
"#,
        cond_t = cond_t,
        out_t = out_t,
        cond_suffix = cond_suffix,
        out_suffix = out_suffix,
        zero_cmp = zero_cmp,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to validate WGSL shader syntax using naga parser
    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    #[test]
    fn test_where_cond_shader_f32_f32() {
        let shader = generate_where_cond_shader(DType::F32, DType::F32).unwrap();
        assert!(shader.contains("fn where_cond_f32_f32"));
        assert!(shader.contains("fn where_broadcast_cond_f32_f32"));
        assert!(shader.contains("array<f32>"));
        validate_wgsl_syntax(&shader).unwrap();
    }

    #[test]
    fn test_where_cond_shader_i32_f32() {
        let shader = generate_where_cond_shader(DType::I32, DType::F32).unwrap();
        assert!(shader.contains("fn where_cond_i32_f32"));
        assert!(shader.contains("fn where_broadcast_cond_i32_f32"));
        validate_wgsl_syntax(&shader).unwrap();
    }

    #[test]
    fn test_where_cond_shader_u32_f32() {
        let shader = generate_where_cond_shader(DType::U32, DType::F32).unwrap();
        assert!(shader.contains("fn where_cond_u32_f32"));
        validate_wgsl_syntax(&shader).unwrap();
    }

    #[test]
    fn test_where_cond_shader_f32_i32() {
        let shader = generate_where_cond_shader(DType::F32, DType::I32).unwrap();
        assert!(shader.contains("fn where_cond_f32_i32"));
        validate_wgsl_syntax(&shader).unwrap();
    }

    #[test]
    fn test_where_cond_shader_all_combinations() {
        let dtypes = [DType::F32, DType::I32, DType::U32];
        for cond_dtype in &dtypes {
            for out_dtype in &dtypes {
                let shader =
                    generate_where_cond_shader(*cond_dtype, *out_dtype).unwrap_or_else(|e| {
                        panic!(
                            "Failed to generate where_cond shader for {:?}/{:?}: {}",
                            cond_dtype, out_dtype, e
                        )
                    });
                validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                    panic!(
                        "Invalid WGSL for where_cond {:?}/{:?}:\n{}\n\nShader:\n{}",
                        cond_dtype, out_dtype, e, shader
                    )
                });
            }
        }
    }
}
