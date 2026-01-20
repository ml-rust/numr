//! WGSL shader generation for matrix multiplication operations

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate WGSL shader for matrix multiplication
pub fn generate_matmul_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated matmul operations for {t}

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<{t}, 16>, 16>;
var<workgroup> tile_b: array<array<{t}, 16>, 16>;

struct MatmulParams {{
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> matmul_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> matmul_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> matmul_c: array<{t}>;
@group(0) @binding(3) var<uniform> matmul_params: MatmulParams;

@compute @workgroup_size(16, 16, 1)
fn matmul_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {{
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    var sum: {t} = {zero};

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t_idx: u32 = 0u; t_idx < num_tiles; t_idx = t_idx + 1u) {{
        let a_col = t_idx * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {{
            tile_a[local_id.y][local_id.x] = matmul_a[row * K + a_col];
        }} else {{
            tile_a[local_id.y][local_id.x] = {zero};
        }}

        let b_row = t_idx * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {{
            tile_b[local_id.y][local_id.x] = matmul_b[b_row * N + col];
        }} else {{
            tile_b[local_id.y][local_id.x] = {zero};
        }}

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {{
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }}

        workgroupBarrier();
    }}

    if (row < M && col < N) {{
        matmul_c[row * N + col] = sum;
    }}
}}

@compute @workgroup_size(16, 16, 1)
fn batched_matmul_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                            @builtin(local_invocation_id) local_id: vec3<u32>,
                            @builtin(workgroup_id) group_id: vec3<u32>) {{
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;
    let batch_size = matmul_params.batch_size;

    let batch = group_id.z;
    if (batch >= batch_size) {{
        return;
    }}

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    let a_batch_offset = batch * M * K;
    let b_batch_offset = batch * K * N;
    let c_batch_offset = batch * M * N;

    var sum: {t} = {zero};

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t_idx: u32 = 0u; t_idx < num_tiles; t_idx = t_idx + 1u) {{
        let a_col = t_idx * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {{
            tile_a[local_id.y][local_id.x] = matmul_a[a_batch_offset + row * K + a_col];
        }} else {{
            tile_a[local_id.y][local_id.x] = {zero};
        }}

        let b_row = t_idx * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {{
            tile_b[local_id.y][local_id.x] = matmul_b[b_batch_offset + b_row * N + col];
        }} else {{
            tile_b[local_id.y][local_id.x] = {zero};
        }}

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {{
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }}

        workgroupBarrier();
    }}

    if (row < M && col < N) {{
        matmul_c[c_batch_offset + row * N + col] = sum;
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

/// Generate WGSL shader for fused matrix multiplication with bias addition
///
/// This implements C = A @ B + bias where:
/// - A has shape [M, K] or [batch, M, K]
/// - B has shape [K, N] or [batch, K, N]
/// - bias has shape [N] (1D, broadcast across all rows and batches)
/// - C has shape [M, N] or [batch, M, N]
///
/// The bias addition is fused into the GEMM epilogue for efficiency,
/// avoiding an extra memory round-trip.
pub fn generate_matmul_bias_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated matmul_bias operations for {t}
// C = A @ B + bias (fused epilogue)

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<{t}, 16>, 16>;
var<workgroup> tile_b: array<array<{t}, 16>, 16>;

struct MatmulBiasParams {{
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> matmul_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> matmul_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> matmul_bias: array<{t}>;
@group(0) @binding(3) var<storage, read_write> matmul_c: array<{t}>;
@group(0) @binding(4) var<uniform> matmul_params: MatmulBiasParams;

@compute @workgroup_size(16, 16, 1)
fn matmul_bias_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    var sum: {t} = {zero};

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t_idx: u32 = 0u; t_idx < num_tiles; t_idx = t_idx + 1u) {{
        let a_col = t_idx * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {{
            tile_a[local_id.y][local_id.x] = matmul_a[row * K + a_col];
        }} else {{
            tile_a[local_id.y][local_id.x] = {zero};
        }}

        let b_row = t_idx * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {{
            tile_b[local_id.y][local_id.x] = matmul_b[b_row * N + col];
        }} else {{
            tile_b[local_id.y][local_id.x] = {zero};
        }}

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {{
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }}

        workgroupBarrier();
    }}

    // Fused epilogue: add bias and write result
    if (row < M && col < N) {{
        matmul_c[row * N + col] = sum + matmul_bias[col];
    }}
}}

@compute @workgroup_size(16, 16, 1)
fn batched_matmul_bias_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                                 @builtin(local_invocation_id) local_id: vec3<u32>,
                                 @builtin(workgroup_id) group_id: vec3<u32>) {{
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;
    let batch_size = matmul_params.batch_size;

    let batch = group_id.z;
    if (batch >= batch_size) {{
        return;
    }}

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    let a_batch_offset = batch * M * K;
    let b_batch_offset = batch * K * N;
    let c_batch_offset = batch * M * N;

    var sum: {t} = {zero};

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t_idx: u32 = 0u; t_idx < num_tiles; t_idx = t_idx + 1u) {{
        let a_col = t_idx * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {{
            tile_a[local_id.y][local_id.x] = matmul_a[a_batch_offset + row * K + a_col];
        }} else {{
            tile_a[local_id.y][local_id.x] = {zero};
        }}

        let b_row = t_idx * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {{
            tile_b[local_id.y][local_id.x] = matmul_b[b_batch_offset + b_row * N + col];
        }} else {{
            tile_b[local_id.y][local_id.x] = {zero};
        }}

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {{
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }}

        workgroupBarrier();
    }}

    // Fused epilogue: add bias (same bias for all batches) and write result
    if (row < M && col < N) {{
        matmul_c[c_batch_offset + row * N + col] = sum + matmul_bias[col];
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
