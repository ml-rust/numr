//! WGSL shaders for copy operations
//!
//! These shaders handle strided/non-contiguous to contiguous copies,
//! which is essential for tensor.contiguous() on GPU.

/// WGSL shader for strided to contiguous copy
///
/// This shader reads elements from a strided source buffer and writes
/// them contiguously to a destination buffer. It handles arbitrary
/// strides and dimensions.
///
/// Note: Uses storage buffer for params instead of uniform to avoid
/// WGSL alignment issues (uniform arrays require 16-byte alignment).
pub const STRIDED_COPY_SHADER: &str = r#"
// Maximum supported dimensions (can be extended if needed)
const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 256u;

// Parameters for strided copy (stored in storage buffer for flexible alignment)
// Layout: [numel, ndim, elem_size_units, src_offset_units, shape[8], strides[8]]
// Total: 4 u32 scalars + 8 u32 shape + 8 i32 strides = 20 elements
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// Read u32 param at index
fn get_param_u32(idx: u32) -> u32 {
    return params[idx];
}

// Read i32 param at index (reinterpret bits)
fn get_param_i32(idx: u32) -> i32 {
    return bitcast<i32>(params[idx]);
}

// Get shape value for dimension d
fn get_shape(d: u32) -> u32 {
    return params[4u + d];  // shape starts at index 4
}

// Get stride value for dimension d
fn get_stride(d: u32) -> i32 {
    return bitcast<i32>(params[12u + d]);  // strides start at index 12 (4 + 8)
}

// Convert linear index to multi-dimensional indices and calculate strided offset
fn get_strided_offset(linear_idx: u32, ndim: u32) -> i32 {
    var remaining = linear_idx;
    var offset: i32 = 0;

    // Iterate through dimensions in reverse order (row-major)
    for (var d: i32 = i32(ndim) - 1; d >= 0; d = d - 1) {
        let dim_size = get_shape(u32(d));
        let idx = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + i32(idx) * get_stride(u32(d));
    }

    return offset;
}

@compute @workgroup_size(256)
fn strided_copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    // Read params
    let numel = get_param_u32(0u);
    let ndim = get_param_u32(1u);
    let elem_size_units = get_param_u32(2u);
    let src_offset_units = get_param_u32(3u);

    if (gid >= numel) {
        return;
    }

    // Calculate source offset (in elements, then convert to units)
    let src_elem_offset = get_strided_offset(gid, ndim);
    let src_unit_offset = src_offset_units + u32(src_elem_offset) * elem_size_units;

    // Destination is contiguous
    let dst_unit_offset = gid * elem_size_units;

    // Copy element (handle different sizes)
    for (var i: u32 = 0u; i < elem_size_units; i = i + 1u) {
        dst[dst_unit_offset + i] = src[src_unit_offset + i];
    }
}
"#;
