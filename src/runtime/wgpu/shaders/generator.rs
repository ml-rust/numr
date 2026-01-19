//! WGSL shader generation for multi-dtype support
//!
//! WebGPU's WGSL does not support templates like CUDA/C++.
//! This module generates WGSL shader source code for each dtype.
//!
//! # Supported DTypes
//!
//! | DType | WGSL Type | Notes |
//! |-------|-----------|-------|
//! | F32   | f32       | Always available |
//! | I32   | i32       | Always available |
//! | U32   | u32       | Always available |
//! | F16   | f16       | Requires WebGPU f16 extension |
//!
//! # Architecture
//!
//! ```text
//! generate_binary_shader(DType::F32, "add") → WGSL source with f32 types
//! generate_binary_shader(DType::I32, "add") → WGSL source with i32 types
//! generate_binary_shader(DType::U32, "add") → WGSL source with u32 types
//! ```
//!
//! Shaders are cached by `(dtype, operation)` key in the pipeline cache.

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

// ============================================================================
// Binary Operation Shader Generation
// ============================================================================

/// Generate WGSL shader for binary element-wise operations
pub fn generate_binary_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = pow(binary_a[idx], binary_b[idx]);
    }}
}}
"#,
            suffix = suffix
        )
    } else {
        // Integer pow requires loop implementation
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        var base = binary_a[idx];
        var exp = binary_b[idx];
        var result: {t} = 1;
        // Simple integer power loop
        for (var i: {t} = 0; i < exp; i = i + 1) {{
            result = result * base;
        }}
        binary_out[idx] = result;
    }}
}}
"#,
            suffix = suffix,
            t = t
        )
    };

    Ok(format!(
        r#"// Auto-generated binary operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct BinaryParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> binary_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> binary_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> binary_out: array<{t}>;
@group(0) @binding(3) var<uniform> binary_params: BinaryParams;

@compute @workgroup_size(256)
fn add_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] + binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn sub_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] - binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn mul_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] * binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn div_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = binary_a[idx] / binary_b[idx];
    }}
}}

@compute @workgroup_size(256)
fn max_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = max(binary_a[idx], binary_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn min_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < binary_params.numel) {{
        binary_out[idx] = min(binary_a[idx], binary_b[idx]);
    }}
}}

{float_ops}
"#,
        t = t,
        suffix = suffix,
        float_ops = float_ops
    ))
}

// ============================================================================
// Unary Operation Shader Generation
// ============================================================================

/// Generate WGSL shader for unary element-wise operations
pub fn generate_unary_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Float-only operations
    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn sqrt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sqrt(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn exp_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = exp(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn log_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = log(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn sin_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sin(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn cos_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = cos(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn tan_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = tan(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn tanh_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = tanh(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn recip_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = 1.0 / unary_a[idx];
    }}
}}

@compute @workgroup_size(256)
fn floor_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = floor(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn ceil_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = ceil(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn round_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = round(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn relu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = max(unary_a[idx], 0.0);
    }}
}}

@compute @workgroup_size(256)
fn sigmoid_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = 1.0 / (1.0 + exp(-unary_a[idx]));
    }}
}}

@compute @workgroup_size(256)
fn silu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        unary_out[idx] = x / (1.0 + exp(-x));
    }}
}}

@compute @workgroup_size(256)
fn gelu_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        let c = 0.7978845608028654; // sqrt(2/pi)
        unary_out[idx] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
    }}
}}

@compute @workgroup_size(256)
fn isnan_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        // NaN != NaN in IEEE 754
        unary_out[idx] = select(0.0, 1.0, x != x);
    }}
}}

@compute @workgroup_size(256)
fn isinf_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        // Inf detection: x == x (not NaN) && x + 1 == x (overflow) && x != 0
        let is_inf = (x == x) && (x + 1.0 == x) && (x != 0.0);
        unary_out[idx] = select(0.0, 1.0, is_inf);
    }}
}}
"#,
            suffix = suffix
        )
    } else {
        // Integer types don't have these operations
        String::new()
    };

    Ok(format!(
        r#"// Auto-generated unary operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct UnaryParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> unary_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<{t}>;
@group(0) @binding(2) var<uniform> unary_params: UnaryParams;

@compute @workgroup_size(256)
fn neg_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = -{neg_prefix}unary_a[idx];
    }}
}}

@compute @workgroup_size(256)
fn abs_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = abs(unary_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn square_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        let x = unary_a[idx];
        unary_out[idx] = x * x;
    }}
}}

@compute @workgroup_size(256)
fn sign_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < unary_params.numel) {{
        unary_out[idx] = sign(unary_a[idx]);
    }}
}}

{float_ops}
"#,
        t = t,
        suffix = suffix,
        neg_prefix = if dtype == DType::U32 {
            "/*u32 neg*/"
        } else {
            ""
        },
        float_ops = float_ops
    ))
}

// ============================================================================
// Scalar Operation Shader Generation
// ============================================================================

/// Generate WGSL shader for scalar element-wise operations
pub fn generate_scalar_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    let float_ops = if is_wgsl_float(dtype) {
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = pow(scalar_a[idx], {t}(scalar_params.scalar));
    }}
}}
"#,
            suffix = suffix,
            t = t
        )
    } else {
        // Integer pow_scalar
        format!(
            r#"
@compute @workgroup_size(256)
fn pow_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        var base = scalar_a[idx];
        var exp = {t}(scalar_params.scalar);
        var result: {t} = 1;
        for (var i: {t} = 0; i < exp; i = i + 1) {{
            result = result * base;
        }}
        scalar_out[idx] = result;
    }}
}}
"#,
            suffix = suffix,
            t = t
        )
    };

    Ok(format!(
        r#"// Auto-generated scalar operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct ScalarParams {{
    numel: u32,
    scalar: f32,  // Always f32 for uniform, cast in shader
}}

@group(0) @binding(0) var<storage, read_write> scalar_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> scalar_out: array<{t}>;
@group(0) @binding(2) var<uniform> scalar_params: ScalarParams;

@compute @workgroup_size(256)
fn add_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] + {t}(scalar_params.scalar);
    }}
}}

@compute @workgroup_size(256)
fn sub_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] - {t}(scalar_params.scalar);
    }}
}}

@compute @workgroup_size(256)
fn mul_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] * {t}(scalar_params.scalar);
    }}
}}

@compute @workgroup_size(256)
fn div_scalar_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < scalar_params.numel) {{
        scalar_out[idx] = scalar_a[idx] / {t}(scalar_params.scalar);
    }}
}}

{float_ops}
"#,
        t = t,
        suffix = suffix,
        float_ops = float_ops
    ))
}

// ============================================================================
// Fill Operation Shader Generation
// ============================================================================

/// Generate WGSL shader for fill operation (set all elements to a constant value)
pub fn generate_fill_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated fill operation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct FillParams {{
    numel: u32,
    value: f32,  // Always f32 for uniform, cast in shader
}}

@group(0) @binding(0) var<storage, read_write> fill_out: array<{t}>;
@group(0) @binding(1) var<uniform> fill_params: FillParams;

@compute @workgroup_size(256)
fn fill_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < fill_params.numel) {{
        fill_out[idx] = {t}(fill_params.value);
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

// ============================================================================
// Compare Operation Shader Generation
// ============================================================================

/// Generate WGSL shader for comparison operations
pub fn generate_compare_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Output is always f32 for consistency (1.0 = true, 0.0 = false)
    Ok(format!(
        r#"// Auto-generated compare operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct CompareParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> compare_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> compare_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> compare_out: array<f32>;
@group(0) @binding(3) var<uniform> compare_params: CompareParams;

@compute @workgroup_size(256)
fn eq_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] == compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn ne_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] != compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn lt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] < compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn le_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] <= compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn gt_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] > compare_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn ge_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < compare_params.numel) {{
        compare_out[idx] = select(0.0, 1.0, compare_a[idx] >= compare_b[idx]);
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

// ============================================================================
// Reduce Operation Shader Generation
// ============================================================================

/// Generate WGSL shader for reduction operations
pub fn generate_reduce_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Workgroup shared memory for reductions
    Ok(format!(
        r#"// Auto-generated reduce operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> reduce_shared: array<{t}, 256>;

struct ReduceParams {{
    reduce_size: u32,
    outer_size: u32,
}}

@group(0) @binding(0) var<storage, read_write> reduce_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<{t}>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

@compute @workgroup_size(256)
fn reduce_sum_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let outer_idx = group_id.x;

    if (outer_idx >= reduce_params.outer_size) {{
        return;
    }}

    let reduce_size = reduce_params.reduce_size;
    let base_offset = outer_idx * reduce_size;

    // Each thread accumulates multiple elements
    var sum: {t} = {zero};
    var i: u32 = tid;
    while (i < reduce_size) {{
        sum = sum + reduce_input[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }}

    reduce_shared[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        reduce_output[outer_idx] = reduce_shared[0];
    }}
}}

@compute @workgroup_size(256)
fn reduce_max_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let outer_idx = group_id.x;

    if (outer_idx >= reduce_params.outer_size) {{
        return;
    }}

    let reduce_size = reduce_params.reduce_size;
    let base_offset = outer_idx * reduce_size;

    var max_val: {t} = {min_val};
    var i: u32 = tid;
    while (i < reduce_size) {{
        max_val = max(max_val, reduce_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }}

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        reduce_output[outer_idx] = reduce_shared[0];
    }}
}}

@compute @workgroup_size(256)
fn reduce_min_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let outer_idx = group_id.x;

    if (outer_idx >= reduce_params.outer_size) {{
        return;
    }}

    let reduce_size = reduce_params.reduce_size;
    let base_offset = outer_idx * reduce_size;

    var min_val: {t} = {max_val};
    var i: u32 = tid;
    while (i < reduce_size) {{
        min_val = min(min_val, reduce_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }}

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        reduce_output[outer_idx] = reduce_shared[0];
    }}
}}
"#,
        t = t,
        suffix = suffix,
        zero = match dtype {
            DType::F32 | DType::F16 => "0.0",
            _ => "0",
        },
        min_val = match dtype {
            DType::F32 => "-3.402823e+38", // -FLT_MAX
            DType::F16 => "-65504.0",
            DType::I32 => "-2147483648",
            DType::U32 => "0u",
            _ => "0",
        },
        max_val = match dtype {
            DType::F32 => "3.402823e+38", // FLT_MAX
            DType::F16 => "65504.0",
            DType::I32 => "2147483647",
            DType::U32 => "4294967295u",
            _ => "0",
        },
    ))
}

// ============================================================================
// Matmul Shader Generation
// ============================================================================

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

// ============================================================================
// Normalization Shader Generation
// ============================================================================

/// Generate WGSL shader for normalization operations (float types only)
pub fn generate_norm_shader(dtype: DType) -> Result<String> {
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "normalization (requires float type)",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated normalization operations for {t}

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> norm_shared: array<{t}, 256>;
var<workgroup> ln_shared_mean: array<{t}, 256>;
var<workgroup> ln_shared_var: array<{t}, 256>;

struct RmsNormParams {{
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}}

@group(0) @binding(0) var<storage, read_write> rms_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> rms_weight: array<{t}>;
@group(0) @binding(2) var<storage, read_write> rms_output: array<{t}>;
@group(0) @binding(3) var<uniform> rms_params: RmsNormParams;

@compute @workgroup_size(256)
fn rms_norm_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= rms_params.batch_size) {{
        return;
    }}

    let hidden_size = rms_params.hidden_size;
    let eps = {t}(rms_params.eps);
    let base_offset = batch_idx * hidden_size;

    // Compute sum of squares
    var sum_sq: {t} = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {{
        let val = rms_input[base_offset + i];
        sum_sq = sum_sq + val * val;
        i = i + WORKGROUP_SIZE;
    }}

    norm_shared[tid] = sum_sq;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            norm_shared[tid] = norm_shared[tid] + norm_shared[tid + s];
        }}
        workgroupBarrier();
    }}

    let rms = sqrt(norm_shared[0] / {t}(hidden_size) + eps);
    workgroupBarrier();

    // Normalize and apply weight
    i = tid;
    while (i < hidden_size) {{
        rms_output[base_offset + i] = rms_input[base_offset + i] / rms * rms_weight[i];
        i = i + WORKGROUP_SIZE;
    }}
}}

struct LayerNormParams {{
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}}

@group(0) @binding(0) var<storage, read_write> ln_input: array<{t}>;
@group(0) @binding(1) var<storage, read_write> ln_weight: array<{t}>;
@group(0) @binding(2) var<storage, read_write> ln_bias: array<{t}>;
@group(0) @binding(3) var<storage, read_write> ln_output: array<{t}>;
@group(0) @binding(4) var<uniform> ln_params: LayerNormParams;

@compute @workgroup_size(256)
fn layer_norm_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>) {{
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= ln_params.batch_size) {{
        return;
    }}

    let hidden_size = ln_params.hidden_size;
    let eps = {t}(ln_params.eps);
    let base_offset = batch_idx * hidden_size;

    // Compute mean
    var sum: {t} = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {{
        sum = sum + ln_input[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }}

    ln_shared_mean[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            ln_shared_mean[tid] = ln_shared_mean[tid] + ln_shared_mean[tid + s];
        }}
        workgroupBarrier();
    }}

    let mean_val = ln_shared_mean[0] / {t}(hidden_size);
    workgroupBarrier();

    // Compute variance
    var var_sum: {t} = 0.0;
    i = tid;
    while (i < hidden_size) {{
        let diff = ln_input[base_offset + i] - mean_val;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }}

    ln_shared_var[tid] = var_sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{
            ln_shared_var[tid] = ln_shared_var[tid] + ln_shared_var[tid + s];
        }}
        workgroupBarrier();
    }}

    let variance = ln_shared_var[0] / {t}(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Normalize and apply affine
    i = tid;
    while (i < hidden_size) {{
        let normalized = (ln_input[base_offset + i] - mean_val) * inv_std;
        ln_output[base_offset + i] = normalized * ln_weight[i] + ln_bias[i];
        i = i + WORKGROUP_SIZE;
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgsl_type() {
        assert_eq!(wgsl_type(DType::F32).unwrap(), "f32");
        assert_eq!(wgsl_type(DType::I32).unwrap(), "i32");
        assert_eq!(wgsl_type(DType::U32).unwrap(), "u32");
        assert!(wgsl_type(DType::F64).is_err()); // Not supported
    }

    #[test]
    fn test_generate_binary_shader() {
        let shader = generate_binary_shader(DType::F32).unwrap();
        assert!(shader.contains("fn add_f32"));
        assert!(shader.contains("fn sub_f32"));
        assert!(shader.contains("fn mul_f32"));
        assert!(shader.contains("array<f32>"));
    }

    #[test]
    fn test_generate_binary_shader_i32() {
        let shader = generate_binary_shader(DType::I32).unwrap();
        assert!(shader.contains("fn add_i32"));
        assert!(shader.contains("array<i32>"));
    }

    #[test]
    fn test_generate_unary_shader_float() {
        let shader = generate_unary_shader(DType::F32).unwrap();
        assert!(shader.contains("fn sqrt_f32"));
        assert!(shader.contains("fn exp_f32"));
        assert!(shader.contains("fn relu_f32"));
    }

    #[test]
    fn test_generate_unary_shader_int() {
        let shader = generate_unary_shader(DType::I32).unwrap();
        assert!(shader.contains("fn neg_i32"));
        assert!(shader.contains("fn abs_i32"));
        // Float ops should not be present
        assert!(!shader.contains("fn sqrt_i32"));
        assert!(!shader.contains("fn exp_i32"));
    }

    #[test]
    fn test_generate_reduce_shader() {
        let shader = generate_reduce_shader(DType::F32).unwrap();
        assert!(shader.contains("fn reduce_sum_f32"));
        assert!(shader.contains("fn reduce_max_f32"));
        assert!(shader.contains("fn reduce_min_f32"));
    }

    #[test]
    fn test_generate_matmul_shader() {
        let shader = generate_matmul_shader(DType::F32).unwrap();
        assert!(shader.contains("fn matmul_f32"));
        assert!(shader.contains("fn batched_matmul_f32"));
        assert!(shader.contains("tile_a"));
        assert!(shader.contains("tile_b"));
    }

    #[test]
    fn test_generate_norm_shader() {
        let shader = generate_norm_shader(DType::F32).unwrap();
        assert!(shader.contains("fn rms_norm_f32"));
        assert!(shader.contains("fn layer_norm_f32"));
    }

    #[test]
    fn test_generate_norm_shader_int_fails() {
        // Normalization is only for float types
        assert!(generate_norm_shader(DType::I32).is_err());
    }

    #[test]
    fn test_generate_compare_shader() {
        let shader = generate_compare_shader(DType::F32).unwrap();
        assert!(shader.contains("fn eq_f32"));
        assert!(shader.contains("fn lt_f32"));
        assert!(shader.contains("array<f32>")); // Output is f32
    }

    // ========================================================================
    // Multi-DType WGSL Syntax Validation Tests
    //
    // These tests validate that generated shaders are syntactically correct
    // WGSL by parsing them with naga. This catches issues like:
    // - Float literals in integer contexts (0.0 vs 0)
    // - Invalid type casts
    // - Missing/incorrect array types
    // ========================================================================

    /// Helper to validate WGSL shader syntax using naga parser (re-exported by wgpu)
    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    /// All dtypes that WebGPU supports
    const WGPU_DTYPES: &[DType] = &[DType::F32, DType::I32, DType::U32];

    #[test]
    fn test_binary_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_binary_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate binary shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for binary shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_unary_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_unary_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate unary shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for unary shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_scalar_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_scalar_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate scalar shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for scalar shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_reduce_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_reduce_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate reduce shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for reduce shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_compare_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_compare_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate compare shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for compare shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_matmul_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_matmul_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate matmul shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for matmul shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_norm_shader_syntax_float_only() {
        // Norm operations only support float types
        let shader = generate_norm_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for norm shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_fill_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_fill_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate fill shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for fill shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_integer_shaders_no_float_literals() {
        // Verify integer shaders don't contain float literals that would cause type errors
        for dtype in [DType::I32, DType::U32] {
            let unary = generate_unary_shader(dtype).unwrap();
            // Integer shaders should not contain standalone float operations
            // The float ops (sqrt, exp, etc.) should be excluded for integers
            assert!(
                !unary.contains("fn sqrt_"),
                "Integer unary shader should not contain sqrt for {:?}",
                dtype
            );
            assert!(
                !unary.contains("fn exp_"),
                "Integer unary shader should not contain exp for {:?}",
                dtype
            );
        }
    }
}
