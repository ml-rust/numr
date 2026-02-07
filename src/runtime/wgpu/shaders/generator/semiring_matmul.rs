//! WGSL shader generation for semiring matrix multiplication

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::semiring::SemiringOp;

/// Generate WGSL shader for semiring matrix multiplication.
///
/// Unlike standard matmul which uses (+, ×), semiring matmul uses
/// a custom (reduce, combine) pair. The shader is generated per (dtype, op)
/// combination with the operations baked in as WGSL functions.
///
/// Uses a simple one-thread-per-output-element approach (no shared-memory
/// tiling) because semiring operations don't distribute like (+, ×).
pub fn generate_semiring_matmul_shader(dtype: DType, op: SemiringOp) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let op_name = semiring_op_name(op);

    let is_float = matches!(dtype, DType::F32 | DType::F16);

    let (identity, combine_expr, reduce_expr) = semiring_wgsl_ops(op, is_float);

    Ok(format!(
        r#"// Auto-generated semiring matmul: {op_name} for {t}
// C[i,j] = reduce_k( combine(A[i,k], B[k,j]) )

struct SemiringMatmulParams {{
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> sr_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> sr_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> sr_c: array<{t}>;
@group(0) @binding(3) var<uniform> sr_params: SemiringMatmulParams;

fn sr_combine(a: {t}, b: {t}) -> {t} {{
    {combine_expr}
}}

fn sr_reduce(acc: {t}, val: {t}) -> {t} {{
    {reduce_expr}
}}

@compute @workgroup_size(16, 16, 1)
fn semiring_matmul_{op_name}_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {{
    let M = sr_params.M;
    let K = sr_params.K;
    let N = sr_params.N;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {{
        return;
    }}

    var acc: {t} = {identity};

    for (var kk: u32 = 0u; kk < K; kk = kk + 1u) {{
        let a_val = sr_a[row * K + kk];
        let b_val = sr_b[kk * N + col];
        acc = sr_reduce(acc, sr_combine(a_val, b_val));
    }}

    sr_c[row * N + col] = acc;
}}

@compute @workgroup_size(16, 16, 1)
fn batched_semiring_matmul_{op_name}_{suffix}(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {{
    let M = sr_params.M;
    let K = sr_params.K;
    let N = sr_params.N;
    let batch_size = sr_params.batch_size;

    let batch = global_id.z;
    if (batch >= batch_size) {{
        return;
    }}

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {{
        return;
    }}

    let a_offset = batch * M * K;
    let b_offset = batch * K * N;
    let c_offset = batch * M * N;

    var acc: {t} = {identity};

    for (var kk: u32 = 0u; kk < K; kk = kk + 1u) {{
        let a_val = sr_a[a_offset + row * K + kk];
        let b_val = sr_b[b_offset + kk * N + col];
        acc = sr_reduce(acc, sr_combine(a_val, b_val));
    }}

    sr_c[c_offset + row * N + col] = acc;
}}
"#,
        t = t,
        suffix = suffix,
        op_name = op_name,
        identity = identity,
        combine_expr = combine_expr,
        reduce_expr = reduce_expr,
    ))
}

fn semiring_op_name(op: SemiringOp) -> &'static str {
    match op {
        SemiringOp::MinPlus => "min_plus",
        SemiringOp::MaxPlus => "max_plus",
        SemiringOp::MaxMin => "max_min",
        SemiringOp::MinMax => "min_max",
        SemiringOp::OrAnd => "or_and",
        SemiringOp::PlusMax => "plus_max",
    }
}

/// Returns (identity, combine_expr, reduce_expr) as WGSL code strings.
fn semiring_wgsl_ops(op: SemiringOp, is_float: bool) -> (&'static str, &'static str, &'static str) {
    match op {
        // KEEP IN SYNC: ops/semiring.rs reduce_identity_f64(), cuda/kernels/semiring_matmul.cu
        SemiringOp::MinPlus => {
            // reduce=min, identity=+inf
            let identity = if is_float {
                "bitcast<f32>(0x7f800000u)"
            } else {
                "2147483647"
            };
            (identity, "return a + b;", "return min(acc, val);")
        }
        SemiringOp::MaxPlus => {
            // reduce=max, identity=-inf
            let identity = if is_float {
                "bitcast<f32>(0xff800000u)"
            } else {
                "-2147483647"
            };
            (identity, "return a + b;", "return max(acc, val);")
        }
        SemiringOp::MaxMin => {
            // reduce=max, identity=-inf
            let identity = if is_float {
                "bitcast<f32>(0xff800000u)"
            } else {
                "-2147483647"
            };
            (identity, "return min(a, b);", "return max(acc, val);")
        }
        SemiringOp::MinMax => {
            // reduce=min, identity=+inf
            let identity = if is_float {
                "bitcast<f32>(0x7f800000u)"
            } else {
                "2147483647"
            };
            (identity, "return max(a, b);", "return min(acc, val);")
        }
        SemiringOp::OrAnd => {
            let zero = if is_float { "0.0" } else { "0" };
            // OrAnd: combine=AND, reduce=OR
            // We inline the logic since we need conditional expressions
            // combine: (a != 0 && b != 0) ? 1 : 0
            // reduce: (acc != 0 || val != 0) ? 1 : 0
            // But WGSL doesn't have ternary, so we use select()
            (
                zero,
                if is_float {
                    "return select(0.0, 1.0, a != 0.0 && b != 0.0);"
                } else {
                    "return select(0, 1, a != 0 && b != 0);"
                },
                if is_float {
                    "return select(0.0, 1.0, acc != 0.0 || val != 0.0);"
                } else {
                    "return select(0, 1, acc != 0 || val != 0);"
                },
            )
        }
        SemiringOp::PlusMax => {
            let zero = if is_float { "0.0" } else { "0" };
            (zero, "return max(a, b);", "return acc + val;")
        }
    }
}
