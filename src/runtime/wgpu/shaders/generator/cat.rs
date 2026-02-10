//! WGSL shader generation for shape operations (cat, repeat, pad, roll)

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// WGSL helper function to access packed `` `array<vec4<u32>, 2>` `` by index.
///
/// WGSL uniform buffers require 16-byte alignment for array elements. We pack 8 u32 values
/// into `` `2 vec4<u32>` `` to meet this requirement. This helper extracts individual values.
const WGSL_GET_PACKED_VALUE_HELPER: &str = r#"// Helper to access packed array<vec4<u32>, 2> by index
fn get_packed_value(arr: array<vec4<u32>, 2>, d: i32) -> u32 {
    let vec_idx = u32(d) / 4u;
    let comp_idx = u32(d) % 4u;
    if (vec_idx == 0u) {
        if (comp_idx == 0u) { return arr[0].x; }
        else if (comp_idx == 1u) { return arr[0].y; }
        else if (comp_idx == 2u) { return arr[0].z; }
        else { return arr[0].w; }
    } else {
        if (comp_idx == 0u) { return arr[1].x; }
        else if (comp_idx == 1u) { return arr[1].y; }
        else if (comp_idx == 2u) { return arr[1].z; }
        else { return arr[1].w; }
    }
}
"#;

/// Generate WGSL shader for cat_copy operation (one tensor at a time)
///
/// This kernel copies data from a source tensor to the appropriate position
/// in the concatenated output tensor. It's called once per input tensor.
pub fn generate_cat_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated cat operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct CatParams {{
    outer_size: u32,
    src_cat_size: u32,
    dst_cat_size: u32,
    cat_offset: u32,
    inner_size: u32,
    total_elements: u32,
}}

@group(0) @binding(0) var<storage, read_write> cat_src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> cat_dst: array<{t}>;
@group(0) @binding(2) var<uniform> cat_params: CatParams;

@compute @workgroup_size(256)
fn cat_copy_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= cat_params.total_elements) {{
        return;
    }}

    // Decompose idx into (outer, cat_i, inner) for source tensor
    let inner = idx % cat_params.inner_size;
    let remaining = idx / cat_params.inner_size;
    let cat_i = remaining % cat_params.src_cat_size;
    let outer = remaining / cat_params.src_cat_size;

    // Compute destination index
    let dst_idx = outer * cat_params.dst_cat_size * cat_params.inner_size
                + (cat_params.cat_offset + cat_i) * cat_params.inner_size
                + inner;

    cat_dst[dst_idx] = cat_src[idx];
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate WGSL shader for repeat operation (tile tensor along all dimensions)
///
/// This kernel tiles the source tensor by the given repeat factors.
pub fn generate_repeat_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated repeat operation for {t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

// Use vec4<u32> for 16-byte alignment in uniform buffer
struct RepeatParams {{
    ndim: u32,
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    src_shape: array<vec4<u32>, 2>,  // 8 u32 values packed into 2 vec4
    out_shape: array<vec4<u32>, 2>,
}}

{helper}

@group(0) @binding(0) var<storage, read_write> repeat_src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> repeat_dst: array<{t}>;
@group(0) @binding(2) var<uniform> repeat_params: RepeatParams;

@compute @workgroup_size(256)
fn repeat_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= repeat_params.total_elements) {{
        return;
    }}

    // Decompose idx into multi-dimensional output coordinates
    var remaining = idx;
    var src_idx = 0u;

    // Compute source strides first (row-major)
    var src_strides: array<u32, 8>;
    var stride = 1u;
    for (var d = i32(repeat_params.ndim) - 1; d >= 0; d = d - 1) {{
        src_strides[d] = stride;
        stride = stride * get_packed_value(repeat_params.src_shape, d);
    }}

    // Process dimensions from last to first
    for (var d = i32(repeat_params.ndim) - 1; d >= 0; d = d - 1) {{
        let out_dim = get_packed_value(repeat_params.out_shape, d);
        let coord = remaining % out_dim;
        remaining = remaining / out_dim;

        // Map to source coordinate using modulo
        let src_shape_d = get_packed_value(repeat_params.src_shape, d);
        let src_coord = coord % src_shape_d;
        src_idx = src_idx + src_coord * src_strides[d];
    }}

    repeat_dst[idx] = repeat_src[src_idx];
}}
"#,
        t = t,
        suffix = suffix,
        helper = WGSL_GET_PACKED_VALUE_HELPER
    ))
}

/// Generate WGSL shader for pad operation (add padding around tensor)
///
/// This kernel adds padding to a tensor with a fill value.
pub fn generate_pad_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated pad operation for {t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

// Use vec4<u32> for 16-byte alignment in uniform buffer
struct PadParams {{
    ndim: u32,
    total_elements: u32,
    fill_value: {t},
    _pad0: u32,
    src_shape: array<vec4<u32>, 2>,    // 8 u32 values packed into 2 vec4
    out_shape: array<vec4<u32>, 2>,
    pad_before: array<vec4<u32>, 2>,
}}

{helper}

@group(0) @binding(0) var<storage, read_write> pad_src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> pad_dst: array<{t}>;
@group(0) @binding(2) var<uniform> pad_params: PadParams;

@compute @workgroup_size(256)
fn pad_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= pad_params.total_elements) {{
        return;
    }}

    // Decompose idx into multi-dimensional output coordinates
    var remaining = idx;
    var coords: array<u32, 8>;
    var in_bounds = true;

    // Process dimensions from last to first
    for (var d = i32(pad_params.ndim) - 1; d >= 0; d = d - 1) {{
        let out_dim = get_packed_value(pad_params.out_shape, d);
        coords[d] = remaining % out_dim;
        remaining = remaining / out_dim;

        // Check if coordinate is in original tensor region
        let pb = get_packed_value(pad_params.pad_before, d);
        let ss = get_packed_value(pad_params.src_shape, d);
        if (coords[d] < pb || coords[d] >= pb + ss) {{
            in_bounds = false;
        }}
    }}

    if (in_bounds) {{
        // Compute source index
        var src_idx = 0u;
        var src_stride = 1u;
        for (var d = i32(pad_params.ndim) - 1; d >= 0; d = d - 1) {{
            let src_coord = coords[d] - get_packed_value(pad_params.pad_before, d);
            src_idx = src_idx + src_coord * src_stride;
            src_stride = src_stride * get_packed_value(pad_params.src_shape, d);
        }}
        pad_dst[idx] = pad_src[src_idx];
    }} else {{
        pad_dst[idx] = pad_params.fill_value;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        helper = WGSL_GET_PACKED_VALUE_HELPER
    ))
}

/// Generate WGSL shader for roll operation (circular shift along dimension)
///
/// This kernel shifts elements along a dimension with wrapping.
pub fn generate_roll_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated roll operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct RollParams {{
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    shift: u32,
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> roll_src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> roll_dst: array<{t}>;
@group(0) @binding(2) var<uniform> roll_params: RollParams;

@compute @workgroup_size(256)
fn roll_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= roll_params.total_elements) {{
        return;
    }}

    // Decompose idx into (outer, dim_coord, inner)
    let inner = idx % roll_params.inner_size;
    let remaining = idx / roll_params.inner_size;
    let dim_coord = remaining % roll_params.dim_size;
    let outer = remaining / roll_params.dim_size;

    // Compute source coordinate with roll (shift goes right, so source is shift positions left)
    let src_dim_coord = (dim_coord + roll_params.dim_size - roll_params.shift) % roll_params.dim_size;

    // Compute source linear index
    let src_idx = outer * roll_params.dim_size * roll_params.inner_size
                + src_dim_coord * roll_params.inner_size
                + inner;

    roll_dst[idx] = roll_src[src_idx];
}}
"#,
        t = t,
        suffix = suffix
    ))
}
