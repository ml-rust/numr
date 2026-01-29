//! WGSL shader generation for special binary functions
//!
//! Generates shaders for: beta, gammainc, gammaincc

use super::super::common::{dtype_suffix, wgsl_type};
use super::{common_constants, lgamma_helpers};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for special binary functions (beta, gammainc, gammaincc)
pub fn generate_special_binary_shader(dtype: DType) -> Result<String> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special functions (WebGPU requires F32)",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated special binary functions for {t}

{constants}

struct SpecialBinaryParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> special_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> special_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> special_out: array<{t}>;
@group(0) @binding(3) var<uniform> special_params: SpecialBinaryParams;

// ============================================================================
// Helper Functions (shared lgamma)
// ============================================================================
{lgamma_helpers}

// Lower incomplete gamma series
fn gammainc_series(a: f32, x: f32) -> f32 {{
    if (x == 0.0) {{
        return 0.0;
    }}

    var term = 1.0 / a;
    var sum = term;

    for (var n = 1; n < MAX_ITER; n = n + 1) {{
        term = term * x / (a + f32(n));
        sum = sum + term;
        if (abs(term) < abs(sum) * EPSILON) {{
            break;
        }}
    }}

    return exp(-x + a * log(x) - lgamma_impl(a)) * sum;
}}

// Upper incomplete gamma continued fraction
fn gammaincc_cf(a: f32, x: f32) -> f32 {{
    var f = 1e30;
    var c = 1e30;
    var d = 0.0;

    for (var n = 1; n < MAX_ITER; n = n + 1) {{
        var an: f32;
        if (n % 2 == 1) {{
            an = f32((n + 1) / 2);
        }} else {{
            an = a - f32(n / 2);
        }}
        let bn = x + f32(n) - a;

        d = bn + an * d;
        if (abs(d) < TINY) {{
            d = TINY;
        }}
        c = bn + an / c;
        if (abs(c) < TINY) {{
            c = TINY;
        }}

        d = 1.0 / d;
        let delta = c * d;
        f = f * delta;

        if (abs(delta - 1.0) < EPSILON) {{
            break;
        }}
    }}

    return exp(-x + a * log(x) - lgamma_impl(a)) / f;
}}

fn gammainc_impl(a: f32, x: f32) -> f32 {{
    if (x < 0.0 || a <= 0.0) {{
        return bitcast<f32>(0x7FC00000u); // NaN
    }}
    if (x == 0.0) {{
        return 0.0;
    }}
    if (x < a + 1.0) {{
        return gammainc_series(a, x);
    }}
    return 1.0 - gammaincc_cf(a, x);
}}

fn gammaincc_impl(a: f32, x: f32) -> f32 {{
    if (x < 0.0 || a <= 0.0) {{
        return bitcast<f32>(0x7FC00000u); // NaN
    }}
    if (x == 0.0) {{
        return 1.0;
    }}
    if (x < a + 1.0) {{
        return 1.0 - gammainc_series(a, x);
    }}
    return gammaincc_cf(a, x);
}}

// ============================================================================
// Compute Kernels
// ============================================================================

@compute @workgroup_size(256)
fn beta_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        let a = special_a[idx];
        let b = special_b[idx];
        special_out[idx] = exp(lgamma_impl(a) + lgamma_impl(b) - lgamma_impl(a + b));
    }}
}}

@compute @workgroup_size(256)
fn gammainc_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = gammainc_impl(special_a[idx], special_b[idx]);
    }}
}}

@compute @workgroup_size(256)
fn gammaincc_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = gammaincc_impl(special_a[idx], special_b[idx]);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        constants = common_constants(),
        lgamma_helpers = lgamma_helpers()
    ))
}
