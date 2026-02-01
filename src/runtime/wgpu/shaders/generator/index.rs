//! WGSL shader generation for index, gather, and scatter operations

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for index_select operation
pub fn generate_index_select_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated index_select operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct IndexSelectParams {{
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    index_len: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<{t}>;
@group(0) @binding(3) var<uniform> params: IndexSelectParams;

@compute @workgroup_size(256)
fn index_select_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.outer_size * params.index_len * params.inner_size;
    if (idx >= total) {{
        return;
    }}

    let inner = idx % params.inner_size;
    let sel_idx = (idx / params.inner_size) % params.index_len;
    let outer = idx / (params.index_len * params.inner_size);

    let index_val = indices[sel_idx];
    if (index_val < 0 || u32(index_val) >= params.dim_size) {{
        output[idx] = {zero};
        return;
    }}

    let src_offset = outer * params.dim_size * params.inner_size + u32(index_val) * params.inner_size + inner;
    output[idx] = input[src_offset];
}}
"#,
        t = t,
        suffix = suffix,
        zero = match dtype {
            DType::F32 | DType::F16 => "0.0",
            _ => "0",
        },
    ))
}

/// Generate WGSL shader for gather operation
pub fn generate_gather_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // For simplicity, we implement gather with max 4 dimensions
    // This is sufficient for most use cases
    Ok(format!(
        r#"// Auto-generated gather operations for {t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 4u;

struct GatherParams {{
    ndim: u32,
    dim: u32,
    total_elements: u32,
    _padding: u32,
    // Shape and strides packed: [input_shape[0..4], input_strides[0..4], output_shape[0..4], output_strides[0..4]]
    input_shape: vec4<u32>,
    input_strides: vec4<u32>,
    output_shape: vec4<u32>,
    output_strides: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> input: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<{t}>;
@group(0) @binding(3) var<uniform> params: GatherParams;

fn get_shape(arr: vec4<u32>, d: u32) -> u32 {{
    if (d == 0u) {{ return arr.x; }}
    else if (d == 1u) {{ return arr.y; }}
    else if (d == 2u) {{ return arr.z; }}
    else {{ return arr.w; }}
}}

@compute @workgroup_size(256)
fn gather_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.total_elements) {{
        return;
    }}

    var remaining = idx;
    var src_offset: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {{
        let out_stride = get_shape(params.output_strides, d);
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;

        if (d == params.dim) {{
            let index_val = indices[idx];
            let dim_size = get_shape(params.input_shape, d);
            if (index_val < 0 || u32(index_val) >= dim_size) {{
                output[idx] = {zero};
                return;
            }}
            src_offset = src_offset + u32(index_val) * get_shape(params.input_strides, d);
        }} else {{
            src_offset = src_offset + coord * get_shape(params.input_strides, d);
        }}
    }}

    output[idx] = input[src_offset];
}}
"#,
        t = t,
        suffix = suffix,
        zero = match dtype {
            DType::F32 | DType::F16 => "0.0",
            _ => "0",
        },
    ))
}

/// Generate WGSL shader for scatter operation
pub fn generate_scatter_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated scatter operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterParams {{
    ndim: u32,
    dim: u32,
    src_total: u32,
    _padding: u32,
    output_shape: vec4<u32>,
    output_strides: vec4<u32>,
    src_shape: vec4<u32>,
    src_strides: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> src: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<{t}>;
@group(0) @binding(3) var<uniform> params: ScatterParams;

fn get_shape(arr: vec4<u32>, d: u32) -> u32 {{
    if (d == 0u) {{ return arr.x; }}
    else if (d == 1u) {{ return arr.y; }}
    else if (d == 2u) {{ return arr.z; }}
    else {{ return arr.w; }}
}}

@compute @workgroup_size(256)
fn scatter_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.src_total) {{
        return;
    }}

    var remaining = idx;
    var dst_offset: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {{
        let src_stride = get_shape(params.src_strides, d);
        let coord = remaining / src_stride;
        remaining = remaining % src_stride;

        if (d == params.dim) {{
            let index_val = indices[idx];
            let dim_size = get_shape(params.output_shape, d);
            if (index_val < 0 || u32(index_val) >= dim_size) {{
                return;
            }}
            dst_offset = dst_offset + u32(index_val) * get_shape(params.output_strides, d);
        }} else {{
            dst_offset = dst_offset + coord * get_shape(params.output_strides, d);
        }}
    }}

    output[dst_offset] = src[idx];
}}

// Copy kernel for initializing output from input
@group(0) @binding(0) var<storage, read> copy_src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> copy_dst: array<{t}>;

struct CopyParams {{
    numel: u32,
}}

@group(0) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(256)
fn copy_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < copy_params.numel) {{
        copy_dst[idx] = copy_src[idx];
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}

/// Generate WGSL shader for index_put operation
///
/// This is the inverse of index_select: puts values from src at positions
/// specified by indices along a dimension. Output should be pre-initialized
/// with a copy of the input tensor.
pub fn generate_index_put_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated index_put operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct IndexPutParams {{
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    index_len: u32,
}}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read> src: array<{t}>;
@group(0) @binding(2) var<storage, read_write> output: array<{t}>;
@group(0) @binding(3) var<uniform> params: IndexPutParams;

@compute @workgroup_size(256)
fn index_put_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = params.outer_size * params.index_len * params.inner_size;
    if (idx >= total) {{
        return;
    }}

    let inner = idx % params.inner_size;
    let sel_idx = (idx / params.inner_size) % params.index_len;
    let outer = idx / (params.index_len * params.inner_size);

    let index_val = indices[sel_idx];
    if (index_val < 0 || u32(index_val) >= params.dim_size) {{
        return; // Out of bounds - skip
    }}

    let dst_offset = outer * params.dim_size * params.inner_size + u32(index_val) * params.inner_size + inner;
    output[dst_offset] = src[idx];
}}
"#,
        t = t,
        suffix = suffix,
    ))
}

/// Generate WGSL shader for embedding_lookup operation
///
/// This is the industry-standard embedding lookup operation used in neural networks
/// for word embeddings, entity embeddings, etc.
///
/// Input: embeddings [vocab_size, embedding_dim], indices [num_indices]
/// Output: output [num_indices, embedding_dim]
pub fn generate_embedding_lookup_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated embedding_lookup operation for {t}
// Industry-standard embedding table lookup used in neural networks.
// Each thread handles one index lookup and copies the full embedding row.

const WORKGROUP_SIZE: u32 = 256u;

struct EmbeddingLookupParams {{
    num_indices: u32,
    vocab_size: u32,
    embedding_dim: u32,
    _pad0: u32,
}}

@group(0) @binding(0) var<storage, read> embeddings: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<{t}>;
@group(0) @binding(3) var<uniform> params: EmbeddingLookupParams;

@compute @workgroup_size(256)
fn embedding_lookup_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.num_indices) {{
        return;
    }}

    let index_val = indices[idx];

    // Check bounds
    if (index_val < 0 || u32(index_val) >= params.vocab_size) {{
        // Out of bounds - fill with zeros
        let out_start = idx * params.embedding_dim;
        for (var i: u32 = 0u; i < params.embedding_dim; i = i + 1u) {{
            output[out_start + i] = {zero};
        }}
        return;
    }}

    // Copy the entire embedding row to output
    let emb_start = u32(index_val) * params.embedding_dim;
    let out_start = idx * params.embedding_dim;
    for (var i: u32 = 0u; i < params.embedding_dim; i = i + 1u) {{
        output[out_start + i] = embeddings[emb_start + i];
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero = match dtype {
            DType::F32 | DType::F16 => "0.0",
            _ => "0",
        },
    ))
}

/// Generate WGSL shader for index bounds validation.
///
/// Validates that all indices are within bounds [0, dim_size).
/// Atomically counts the number of out-of-bounds indices.
/// Returns count in error_count[0]. If count > 0, some indices are invalid.
pub fn generate_validate_indices_shader() -> String {
    r#"// Auto-generated index bounds validation kernel

const WORKGROUP_SIZE: u32 = 256u;

struct ValidateIndicesParams {
    index_len: u32,
    dim_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> error_count: atomic<u32>;
@group(0) @binding(2) var<uniform> params: ValidateIndicesParams;

@compute @workgroup_size(256)
fn validate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.index_len) {
        return;
    }

    let index_val = indices[idx];
    if (index_val < 0 || u32(index_val) >= params.dim_size) {
        atomicAdd(&error_count, 1u);
    }
}
"#
    .to_string()
}
