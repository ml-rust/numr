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

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
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

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
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

@group(0) @binding(0) var<storage, read_write> src: array<{t}>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
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
@group(0) @binding(0) var<storage, read_write> copy_src: array<{t}>;
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

@group(0) @binding(0) var<storage, read_write> indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> src: array<{t}>;
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

@group(0) @binding(0) var<storage, read_write> embeddings: array<{t}>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
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

/// Generate WGSL shader for gather_nd operation.
///
/// Gathers slices from input using N-dimensional indices.
pub fn generate_gather_nd_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated gather_nd operations for {t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

struct GatherNdParams {{
    num_slices: u32,
    slice_size: u32,
    index_depth: u32,
    ndim: u32,
    input_shape: array<u32, 8>,
    input_strides: array<u32, 8>,
}}

@group(0) @binding(0) var<storage, read> gather_nd_input: array<{t}>;
@group(0) @binding(1) var<storage, read> gather_nd_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> gather_nd_output: array<{t}>;
@group(0) @binding(3) var<uniform> gather_nd_params: GatherNdParams;

@compute @workgroup_size(256)
fn gather_nd_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = gather_nd_params.num_slices * gather_nd_params.slice_size;
    if (idx >= total) {{
        return;
    }}

    let slice_idx = idx / gather_nd_params.slice_size;
    let element_in_slice = idx % gather_nd_params.slice_size;

    // Compute input offset from indices
    var input_offset: u32 = 0u;
    let indices_offset = slice_idx * gather_nd_params.index_depth;

    for (var d: u32 = 0u; d < gather_nd_params.index_depth; d = d + 1u) {{
        let coord = gather_nd_indices[indices_offset + d];
        if (coord < 0 || u32(coord) >= gather_nd_params.input_shape[d]) {{
            gather_nd_output[idx] = {zero};
            return;
        }}
        input_offset = input_offset + u32(coord) * gather_nd_params.input_strides[d];
    }}

    // Add offset for element within slice
    if (gather_nd_params.slice_size > 1u) {{
        var remaining = element_in_slice;
        for (var d: u32 = gather_nd_params.index_depth; d < gather_nd_params.ndim; d = d + 1u) {{
            let dim_size = gather_nd_params.input_shape[d];
            let coord = remaining / gather_nd_params.input_strides[d];
            remaining = remaining % gather_nd_params.input_strides[d];
            input_offset = input_offset + coord * gather_nd_params.input_strides[d];
        }}
    }}

    gather_nd_output[idx] = gather_nd_input[input_offset + element_in_slice];
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

/// Generate WGSL shader for bincount operation.
///
/// Counts occurrences of each value in an integer tensor, optionally with weights.
/// Note: Uses atomic operations for accumulation.
pub fn generate_bincount_shader(weights_dtype: Option<DType>) -> Result<String> {
    if let Some(dtype) = weights_dtype {
        // Weighted bincount
        let t = wgsl_type(dtype)?;
        let suffix = dtype_suffix(dtype)?;

        Ok(format!(
            r#"// Auto-generated weighted bincount for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct BincountParams {{
    n: u32,
    minlength: u32,
    _pad0: u32,
    _pad1: u32,
}}

@group(0) @binding(0) var<storage, read> bincount_input: array<i32>;
@group(0) @binding(1) var<storage, read> bincount_weights: array<{t}>;
@group(0) @binding(2) var<storage, read_write> bincount_output: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> bincount_params: BincountParams;

@compute @workgroup_size(256)
fn bincount_weighted_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= bincount_params.n) {{
        return;
    }}

    let value = bincount_input[idx];
    if (value < 0 || u32(value) >= bincount_params.minlength) {{
        return;
    }}

    let weight = bincount_weights[idx];
    // For float weights, we need to use atomic operations
    // WebGPU only supports atomic ops on u32/i32, so we use bitcast
    let weight_bits = bitcast<u32>(weight);
    atomicAdd(&bincount_output[u32(value)], weight_bits);
}}
"#,
            t = t,
            suffix = suffix,
        ))
    } else {
        // Unweighted bincount
        Ok(r#"// Auto-generated unweighted bincount

const WORKGROUP_SIZE: u32 = 256u;

struct BincountParams {
    n: u32,
    minlength: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> bincount_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> bincount_output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> bincount_params: BincountParams;

@compute @workgroup_size(256)
fn bincount_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bincount_params.n) {
        return;
    }

    let value = bincount_input[idx];
    if (value < 0 || u32(value) >= bincount_params.minlength) {
        return;
    }

    atomicAdd(&bincount_output[u32(value)], 1u);
}
"#
        .to_string())
    }
}

/// Generate WGSL shader for scatter_reduce operation.
///
/// Scatters values with reduction (sum, max, min).
/// Note: Uses atomic operations.
pub fn generate_scatter_reduce_shader(dtype: DType, op: &str) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let atomic_op = match op {
        "sum" => "atomicAdd",
        "max" => "atomicMax",
        "min" => "atomicMin",
        _ => {
            return Err(crate::error::Error::InvalidArgument {
                arg: "op",
                reason: format!("scatter_reduce op must be sum, max, or min, got {}", op),
            });
        }
    };

    // For f32, we need CAS loops since atomicMax/Min only work on integers
    let is_float = matches!(dtype, DType::F32 | DType::F16);

    if is_float && op != "sum" {
        // Float max/min requires CAS loop
        Ok(format!(
            r#"// Auto-generated scatter_reduce_{op} for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterReduceParams {{
    dim: u32,
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    src_dim_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> scatter_src: array<{t}>;
@group(0) @binding(1) var<storage, read> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_{op}_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = scatter_params.outer_size * scatter_params.src_dim_size * scatter_params.inner_size;
    if (idx >= total) {{
        return;
    }}

    let inner = idx % scatter_params.inner_size;
    let src_dim_idx = (idx / scatter_params.inner_size) % scatter_params.src_dim_size;
    let outer = idx / (scatter_params.src_dim_size * scatter_params.inner_size);

    let index_val = scatter_indices[src_dim_idx];
    if (index_val < 0 || u32(index_val) >= scatter_params.dim_size) {{
        return;
    }}

    let src_val = scatter_src[idx];
    let dst_idx = outer * scatter_params.dim_size * scatter_params.inner_size + u32(index_val) * scatter_params.inner_size + inner;

    // CAS loop for {op}
    var old_bits: u32;
    var new_bits: u32;
    loop {{
        old_bits = atomicLoad(&scatter_dst[dst_idx]);
        let old_val = bitcast<f32>(old_bits);
        let new_val = {cmp_expr};
        new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&scatter_dst[dst_idx], old_bits, new_bits);
        if (result.exchanged) {{
            break;
        }}
    }}
}}
"#,
            t = t,
            suffix = suffix,
            op = op,
            cmp_expr = if op == "max" {
                "max(old_val, src_val)"
            } else {
                "min(old_val, src_val)"
            },
        ))
    } else if is_float {
        // Float sum uses atomicAdd with bitcast
        Ok(format!(
            r#"// Auto-generated scatter_reduce_sum for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterReduceParams {{
    dim: u32,
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    src_dim_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> scatter_src: array<{t}>;
@group(0) @binding(1) var<storage, read> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_sum_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = scatter_params.outer_size * scatter_params.src_dim_size * scatter_params.inner_size;
    if (idx >= total) {{
        return;
    }}

    let inner = idx % scatter_params.inner_size;
    let src_dim_idx = (idx / scatter_params.inner_size) % scatter_params.src_dim_size;
    let outer = idx / (scatter_params.src_dim_size * scatter_params.inner_size);

    let index_val = scatter_indices[src_dim_idx];
    if (index_val < 0 || u32(index_val) >= scatter_params.dim_size) {{
        return;
    }}

    let src_val = scatter_src[idx];
    let dst_idx = outer * scatter_params.dim_size * scatter_params.inner_size + u32(index_val) * scatter_params.inner_size + inner;

    // CAS loop for atomic float add
    var old_bits: u32;
    var new_bits: u32;
    loop {{
        old_bits = atomicLoad(&scatter_dst[dst_idx]);
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + src_val;
        new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&scatter_dst[dst_idx], old_bits, new_bits);
        if (result.exchanged) {{
            break;
        }}
    }}
}}
"#,
            t = t,
            suffix = suffix,
        ))
    } else {
        // Integer types can use native atomic ops
        Ok(format!(
            r#"// Auto-generated scatter_reduce_{op} for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterReduceParams {{
    dim: u32,
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    src_dim_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> scatter_src: array<{t}>;
@group(0) @binding(1) var<storage, read> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<atomic<{atomic_t}>>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_{op}_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = scatter_params.outer_size * scatter_params.src_dim_size * scatter_params.inner_size;
    if (idx >= total) {{
        return;
    }}

    let inner = idx % scatter_params.inner_size;
    let src_dim_idx = (idx / scatter_params.inner_size) % scatter_params.src_dim_size;
    let outer = idx / (scatter_params.src_dim_size * scatter_params.inner_size);

    let index_val = scatter_indices[src_dim_idx];
    if (index_val < 0 || u32(index_val) >= scatter_params.dim_size) {{
        return;
    }}

    let src_val = scatter_src[idx];
    let dst_idx = outer * scatter_params.dim_size * scatter_params.inner_size + u32(index_val) * scatter_params.inner_size + inner;

    {atomic_op}(&scatter_dst[dst_idx], src_val);
}}
"#,
            t = t,
            suffix = suffix,
            op = op,
            atomic_t = if dtype == DType::I32 { "i32" } else { "u32" },
            atomic_op = atomic_op,
        ))
    }
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

@group(0) @binding(0) var<storage, read_write> indices: array<i32>;
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
