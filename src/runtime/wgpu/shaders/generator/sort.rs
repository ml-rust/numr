//! WGSL shader generation for sorting operations
//!
//! Provides bitonic sort implementation for GPU-accelerated sorting.
//! Supports sort, argsort, topk, unique, nonzero, and searchsorted operations.

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Maximum sort size for shared memory (power of 2)
pub const MAX_SHARED_SORT_SIZE: usize = 512;

/// Generate WGSL shader for sort operations
pub fn generate_sort_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let (min_val, max_val) = match dtype {
        DType::F32 => ("-3.402823e+38", "3.402823e+38"),
        DType::I32 => ("-2147483648", "2147483647"),
        DType::U32 => ("0u", "4294967295u"),
        _ => return Err(Error::UnsupportedDType { dtype, op: "sort" }),
    };

    // Comparison function depends on type
    let cmp_less = match dtype {
        DType::F32 => "a < b",
        DType::I32 => "a < b",
        DType::U32 => "a < b",
        _ => "a < b",
    };

    Ok(format!(
        r#"// Auto-generated sort operations for {t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SORT_SIZE: u32 = 512u;

var<workgroup> shared_vals: array<{t}, 512>;
var<workgroup> shared_idxs: array<i32, 512>;

struct SortParams {{
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    descending: u32,
}}

struct TopkParams {{
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    k: u32,
    largest: u32,
    sorted: u32,
}}

struct SearchsortedParams {{
    seq_len: u32,
    num_values: u32,
    right: u32,
    _pad: u32,
}}

struct CountParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> sort_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> sort_output: array<{t}>;
@group(0) @binding(2) var<storage, read_write> sort_indices: array<i32>;
@group(0) @binding(3) var<uniform> sort_params: SortParams;

// Comparison helper
fn compare_less_{suffix}(a: {t}, b: {t}) -> bool {{
    return {cmp_less};
}}

// Bitonic compare and swap for sort with indices
fn bitonic_cas_{suffix}(i: u32, j: u32, dir: bool) {{
    let vi = shared_vals[i];
    let vj = shared_vals[j];
    let swap = select(compare_less_{suffix}(vi, vj), compare_less_{suffix}(vj, vi), dir);
    if (swap) {{
        shared_vals[i] = vj;
        shared_vals[j] = vi;
        let ti = shared_idxs[i];
        shared_idxs[i] = shared_idxs[j];
        shared_idxs[j] = ti;
    }}
}}

// Bitonic compare and swap for sort values only
fn bitonic_cas_values_{suffix}(i: u32, j: u32, dir: bool) {{
    let vi = shared_vals[i];
    let vj = shared_vals[j];
    let swap = select(compare_less_{suffix}(vi, vj), compare_less_{suffix}(vj, vi), dir);
    if (swap) {{
        shared_vals[i] = vj;
        shared_vals[j] = vi;
    }}
}}

// Sort with indices - returns both sorted values and original indices
@compute @workgroup_size(256)
fn sort_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = sort_params.outer_size;
    let sort_size = sort_params.sort_size;
    let inner_size = sort_params.inner_size;
    let descending = sort_params.descending != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {{
        return;
    }}

    // Pad to next power of 2
    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {{
        p = p << 1u;
    }}
    n = min(p, MAX_SORT_SIZE);

    // Load data into shared memory
    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {{
        if (i < sort_size) {{
            let idx = base_offset + i * inner_size;
            shared_vals[i] = sort_input[idx];
            shared_idxs[i] = i32(i);
        }} else {{
            // Pad with max/min based on sort direction
            shared_vals[i] = select({t}({max_val}), {t}({min_val}), descending);
            shared_idxs[i] = i32(i);
        }}
    }}
    workgroupBarrier();

    // Bitonic sort
    for (var k: u32 = 2u; k <= n; k = k << 1u) {{
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {{
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {{
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                let ascending_local = ((ij / k) % 2u == 0u) != descending;

                if (ij_pair < n) {{
                    bitonic_cas_{suffix}(ij, ij_pair, ascending_local);
                }}
            }}
            workgroupBarrier();
        }}
    }}

    // Write sorted values and indices
    for (var i = tid; i < sort_size; i = i + WORKGROUP_SIZE) {{
        let out_idx = base_offset + i * inner_size;
        sort_output[out_idx] = shared_vals[i];
        sort_indices[out_idx] = shared_idxs[i];
    }}
}}

// Sort values only (no indices)
@compute @workgroup_size(256)
fn sort_values_only_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = sort_params.outer_size;
    let sort_size = sort_params.sort_size;
    let inner_size = sort_params.inner_size;
    let descending = sort_params.descending != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {{
        return;
    }}

    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {{
        p = p << 1u;
    }}
    n = min(p, MAX_SORT_SIZE);

    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {{
        if (i < sort_size) {{
            let idx = base_offset + i * inner_size;
            shared_vals[i] = sort_input[idx];
        }} else {{
            shared_vals[i] = select({t}({max_val}), {t}({min_val}), descending);
        }}
    }}
    workgroupBarrier();

    // Bitonic sort
    for (var k: u32 = 2u; k <= n; k = k << 1u) {{
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {{
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {{
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                let ascending_local = ((ij / k) % 2u == 0u) != descending;

                if (ij_pair < n) {{
                    bitonic_cas_values_{suffix}(ij, ij_pair, ascending_local);
                }}
            }}
            workgroupBarrier();
        }}
    }}

    for (var i = tid; i < sort_size; i = i + WORKGROUP_SIZE) {{
        let out_idx = base_offset + i * inner_size;
        sort_output[out_idx] = shared_vals[i];
    }}
}}

// Argsort - returns indices only
@compute @workgroup_size(256)
fn argsort_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = sort_params.outer_size;
    let sort_size = sort_params.sort_size;
    let inner_size = sort_params.inner_size;
    let descending = sort_params.descending != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {{
        return;
    }}

    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {{
        p = p << 1u;
    }}
    n = min(p, MAX_SORT_SIZE);

    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {{
        if (i < sort_size) {{
            let idx = base_offset + i * inner_size;
            shared_vals[i] = sort_input[idx];
            shared_idxs[i] = i32(i);
        }} else {{
            shared_vals[i] = select({t}({max_val}), {t}({min_val}), descending);
            shared_idxs[i] = i32(i);
        }}
    }}
    workgroupBarrier();

    // Bitonic sort
    for (var k: u32 = 2u; k <= n; k = k << 1u) {{
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {{
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {{
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                let ascending_local = ((ij / k) % 2u == 0u) != descending;

                if (ij_pair < n) {{
                    bitonic_cas_{suffix}(ij, ij_pair, ascending_local);
                }}
            }}
            workgroupBarrier();
        }}
    }}

    // Write indices only
    for (var i = tid; i < sort_size; i = i + WORKGROUP_SIZE) {{
        let out_idx = base_offset + i * inner_size;
        sort_indices[out_idx] = shared_idxs[i];
    }}
}}
"#,
        t = t,
        suffix = suffix,
        min_val = min_val,
        max_val = max_val,
        cmp_less = cmp_less,
    ))
}

/// Generate WGSL shader for topk operation
pub fn generate_topk_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let (min_val, max_val) = match dtype {
        DType::F32 => ("-3.402823e+38", "3.402823e+38"),
        DType::I32 => ("-2147483648", "2147483647"),
        DType::U32 => ("0u", "4294967295u"),
        _ => return Err(Error::UnsupportedDType { dtype, op: "topk" }),
    };

    let cmp_less = match dtype {
        DType::F32 => "a < b",
        DType::I32 => "a < b",
        DType::U32 => "a < b",
        _ => "a < b",
    };

    Ok(format!(
        r#"// Auto-generated topk operations for {t}

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SORT_SIZE: u32 = 512u;

var<workgroup> shared_vals: array<{t}, 512>;
var<workgroup> shared_idxs: array<i32, 512>;

struct TopkParams {{
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    k: u32,
    largest: u32,
    sorted: u32,
}}

@group(0) @binding(0) var<storage, read_write> topk_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> topk_values: array<{t}>;
@group(0) @binding(2) var<storage, read_write> topk_indices: array<i32>;
@group(0) @binding(3) var<uniform> topk_params: TopkParams;

fn compare_less_{suffix}(a: {t}, b: {t}) -> bool {{
    return {cmp_less};
}}

fn bitonic_cas_{suffix}(i: u32, j: u32, dir: bool) {{
    let vi = shared_vals[i];
    let vj = shared_vals[j];
    let swap = select(compare_less_{suffix}(vi, vj), compare_less_{suffix}(vj, vi), dir);
    if (swap) {{
        shared_vals[i] = vj;
        shared_vals[j] = vi;
        let ti = shared_idxs[i];
        shared_idxs[i] = shared_idxs[j];
        shared_idxs[j] = ti;
    }}
}}

@compute @workgroup_size(256)
fn topk_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let outer_idx = group_id.x;
    let inner_idx = group_id.y;
    let tid = local_id.x;

    let outer_size = topk_params.outer_size;
    let sort_size = topk_params.sort_size;
    let inner_size = topk_params.inner_size;
    let k = topk_params.k;
    let largest = topk_params.largest != 0u;

    if (outer_idx >= outer_size || inner_idx >= inner_size) {{
        return;
    }}

    var n = sort_size;
    var p: u32 = 1u;
    while (p < n) {{
        p = p << 1u;
    }}
    n = min(p, MAX_SORT_SIZE);

    let base_offset = outer_idx * sort_size * inner_size + inner_idx;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {{
        if (i < sort_size) {{
            let idx = base_offset + i * inner_size;
            shared_vals[i] = topk_input[idx];
            shared_idxs[i] = i32(i);
        }} else {{
            shared_vals[i] = select({t}({max_val}), {t}({min_val}), largest);
            shared_idxs[i] = i32(i);
        }}
    }}
    workgroupBarrier();

    // Bitonic sort (descending if largest, ascending if smallest)
    for (var k_: u32 = 2u; k_ <= n; k_ = k_ << 1u) {{
        for (var j: u32 = k_ >> 1u; j > 0u; j = j >> 1u) {{
            for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {{
                // Calculate bitonic network indices
                let ij = (i / j) * 2u * j + (i % j);
                let ij_pair = ij + j;

                // Direction depends on which half of the network we're in
                // For largest: descending (true), for smallest: ascending (false)
                let ascending_local = ((ij / k_) % 2u == 0u) != largest;

                if (ij_pair < n) {{
                    bitonic_cas_{suffix}(ij, ij_pair, ascending_local);
                }}
            }}
            workgroupBarrier();
        }}
    }}

    // Write top-k values and indices
    let out_base = outer_idx * k * inner_size + inner_idx;
    for (var i = tid; i < k; i = i + WORKGROUP_SIZE) {{
        let out_idx = out_base + i * inner_size;
        topk_values[out_idx] = shared_vals[i];
        topk_indices[out_idx] = shared_idxs[i];
    }}
}}
"#,
        t = t,
        suffix = suffix,
        min_val = min_val,
        max_val = max_val,
        cmp_less = cmp_less,
    ))
}

/// Generate WGSL shader for searchsorted operation
pub fn generate_searchsorted_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated searchsorted operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct SearchsortedParams {{
    seq_len: u32,
    num_values: u32,
    right: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> ss_seq: array<{t}>;
@group(0) @binding(1) var<storage, read_write> ss_values: array<{t}>;
@group(0) @binding(2) var<storage, read_write> ss_output: array<i32>;
@group(0) @binding(3) var<uniform> ss_params: SearchsortedParams;

@compute @workgroup_size(256)
fn searchsorted_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;

    if (idx >= ss_params.num_values) {{
        return;
    }}

    let value = ss_values[idx];
    let seq_len = ss_params.seq_len;
    let right = ss_params.right != 0u;

    // Binary search
    var lo: u32 = 0u;
    var hi: u32 = seq_len;

    while (lo < hi) {{
        let mid = lo + (hi - lo) / 2u;
        let seq_val = ss_seq[mid];

        var go_right: bool;
        if (right) {{
            go_right = seq_val <= value;
        }} else {{
            go_right = seq_val < value;
        }}

        if (go_right) {{
            lo = mid + 1u;
        }} else {{
            hi = mid;
        }}
    }}

    ss_output[idx] = i32(lo);
}}
"#,
        t = t,
        suffix = suffix,
    ))
}

/// Generate WGSL shader for nonzero counting (phase 1)
pub fn generate_count_nonzero_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let zero_check = match dtype {
        DType::F32 => "input[idx] != 0.0",
        DType::I32 => "input[idx] != 0",
        DType::U32 => "input[idx] != 0u",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "count_nonzero",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated count_nonzero operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_count: array<u32, 256>;

struct CountParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> count_output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> count_params: CountParams;

@compute @workgroup_size(256)
fn count_nonzero_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let tid = local_id.x;
    let numel = count_params.numel;

    // Each thread counts its elements
    var local_count: u32 = 0u;
    var idx = global_id.x;
    while (idx < numel) {{
        if ({zero_check}) {{
            local_count = local_count + 1u;
        }}
        idx = idx + WORKGROUP_SIZE * 256u; // stride by total threads
    }}

    shared_count[tid] = local_count;
    workgroupBarrier();

    // Tree reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            shared_count[tid] = shared_count[tid] + shared_count[tid + s];
        }}
        workgroupBarrier();
    }}

    // Thread 0 adds to global counter
    if (tid == 0u) {{
        atomicAdd(&count_output[0], shared_count[0]);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero_check = zero_check,
    ))
}

/// Generate WGSL shader for gathering nonzero indices (phase 2)
pub fn generate_gather_nonzero_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let zero_check = match dtype {
        DType::F32 => "input[idx] != 0.0",
        DType::I32 => "input[idx] != 0",
        DType::U32 => "input[idx] != 0u",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "gather_nonzero",
            });
        }
    };

    Ok(format!(
        r#"// Auto-generated gather_nonzero operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct CountParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> indices_output: array<i32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> count_params: CountParams;

@compute @workgroup_size(256)
fn gather_nonzero_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let numel = count_params.numel;
    var idx = global_id.x;

    while (idx < numel) {{
        if ({zero_check}) {{
            let out_idx = atomicAdd(&counter[0], 1u);
            indices_output[out_idx] = i32(idx);
        }}
        idx = idx + WORKGROUP_SIZE * 256u;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero_check = zero_check,
    ))
}

/// Generate WGSL shader for flat_to_multi_index
pub fn generate_flat_to_multi_index_shader() -> Result<String> {
    Ok(r#"// Convert flat indices to multi-dimensional indices

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

struct FlatToMultiParams {
    nnz: u32,
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
    shape: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> flat_indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> multi_indices: array<i32>;
@group(0) @binding(2) var<uniform> params: FlatToMultiParams;

fn get_shape_dim(d: u32) -> u32 {
    return params.shape[d / 4u][d % 4u];
}

@compute @workgroup_size(256)
fn flat_to_multi_index(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.nnz) {
        return;
    }

    var flat_idx = u32(flat_indices[idx]);
    let ndim = params.ndim;

    // Compute strides on the fly (row-major)
    // and convert flat index to multi-index
    for (var d: u32 = ndim; d > 0u; d = d - 1u) {
        let dim = d - 1u;
        let dim_size = get_shape_dim(dim);
        let coord = flat_idx % dim_size;
        flat_idx = flat_idx / dim_size;

        // Store: multi_indices[idx * ndim + dim] = coord
        multi_indices[idx * ndim + dim] = i32(coord);
    }
}
"#
    .to_string())
}

/// Generate WGSL shader for unique operations
pub fn generate_unique_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated unique operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_count: array<u32, 256>;

struct UniqueParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> sorted_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> unique_output: array<{t}>;
@group(0) @binding(2) var<storage, read_write> unique_counter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> unique_params: UniqueParams;

// Count unique elements (on sorted input)
@compute @workgroup_size(256)
fn count_unique_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let tid = local_id.x;
    let numel = unique_params.numel;

    var local_count: u32 = 0u;
    var idx = global_id.x;

    while (idx < numel) {{
        // Count if first element or different from previous
        if (idx == 0u || sorted_input[idx] != sorted_input[idx - 1u]) {{
            local_count = local_count + 1u;
        }}
        idx = idx + WORKGROUP_SIZE * 256u;
    }}

    shared_count[tid] = local_count;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            shared_count[tid] = shared_count[tid] + shared_count[tid + s];
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        atomicAdd(&unique_counter[0], shared_count[0]);
    }}
}}

// Extract unique elements
@compute @workgroup_size(256)
fn extract_unique_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let numel = unique_params.numel;
    var idx = global_id.x;

    while (idx < numel) {{
        // Write if first element or different from previous
        if (idx == 0u || sorted_input[idx] != sorted_input[idx - 1u]) {{
            let out_idx = atomicAdd(&unique_counter[0], 1u);
            unique_output[out_idx] = sorted_input[idx];
        }}
        idx = idx + WORKGROUP_SIZE * 256u;
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}

/// Generate WGSL shader for unique_with_counts operations
pub fn generate_unique_with_counts_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated unique_with_counts operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct UniqueCountsParams {{
    numel: u32,
    num_unique: u32,
    _pad0: u32,
    _pad1: u32,
}}

// Mark boundaries in sorted array (where value changes)
// Output: flags[i] = 1 if sorted[i] != sorted[i-1] (or i=0), else 0
@group(0) @binding(0) var<storage, read> sorted_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> boundary_flags: array<u32>;
@group(0) @binding(2) var<uniform> params: UniqueCountsParams;

@compute @workgroup_size(256)
fn mark_boundaries_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let numel = params.numel;

    if (idx >= numel) {{
        return;
    }}

    // Mark boundary: first element or different from previous
    if (idx == 0u || sorted_input[idx] != sorted_input[idx - 1u]) {{
        boundary_flags[idx] = 1u;
    }} else {{
        boundary_flags[idx] = 0u;
    }}
}}

// Scatter unique values and compute counts using prefix sum indices
// prefix_sum[i] contains the output index for element at position i (if it's a boundary)
// We write: unique_values[prefix_sum[i]-1] = sorted[i] when flags[i] == 1
// counts[prefix_sum[i]-1] = (next boundary position - i) computed from adjacent prefix sums
@group(0) @binding(0) var<storage, read> scatter_sorted: array<{t}>;
@group(0) @binding(1) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(2) var<storage, read_write> unique_values: array<{t}>;
@group(0) @binding(3) var<storage, read_write> inverse_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> counts: array<i32>;
@group(0) @binding(5) var<uniform> scatter_params: UniqueCountsParams;

@compute @workgroup_size(256)
fn scatter_unique_with_counts_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let numel = scatter_params.numel;
    let num_unique = scatter_params.num_unique;

    if (idx >= numel) {{
        return;
    }}

    // The prefix sum gives us 1-based output indices
    let out_idx_plus1 = prefix_sum[idx];

    // Check if this is a boundary by comparing with previous prefix sum
    let is_boundary = (idx == 0u) || (prefix_sum[idx] != prefix_sum[idx - 1u]);

    // Write inverse index: which unique element does this sorted element map to
    inverse_indices[idx] = i32(out_idx_plus1 - 1u);

    if (is_boundary) {{
        let out_idx = out_idx_plus1 - 1u;
        unique_values[out_idx] = scatter_sorted[idx];

        // Compute count: find next boundary position
        // The count is (next_boundary_position - idx)
        // If we're the last unique, count to numel
        if (out_idx + 1u >= num_unique) {{
            // Last unique element
            counts[out_idx] = i32(numel - idx);
        }} else {{
            // Find next boundary: it's where prefix_sum increases next
            // We need to find the smallest j > idx where prefix_sum[j] > out_idx_plus1
            // Actually, we can compute this differently:
            // The run length is the distance to the next boundary
            // For efficiency, we'll use a second pass or a different approach

            // For now, scan forward (not ideal but correct)
            var run_len: u32 = 1u;
            var j = idx + 1u;
            while (j < numel && prefix_sum[j] == out_idx_plus1) {{
                run_len = run_len + 1u;
                j = j + 1u;
            }}
            counts[out_idx] = i32(run_len);
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}
