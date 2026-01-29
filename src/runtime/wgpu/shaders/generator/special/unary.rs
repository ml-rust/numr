//! WGSL shader generation for special unary functions
//!
//! Generates shaders for: erf, erfc, erfinv, gamma, lgamma, digamma

use super::super::common::{dtype_suffix, wgsl_type};
use super::{common_constants, lgamma_helpers};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for special unary functions (erf, erfc, erfinv, gamma, lgamma, digamma)
pub fn generate_special_unary_shader(dtype: DType) -> Result<String> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special functions (WebGPU requires F32)",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated special functions for {t}
// Algorithms: A&S for erf, Lanczos for gamma, asymptotic for digamma

{constants}

struct SpecialParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> special_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> special_out: array<{t}>;
@group(0) @binding(2) var<uniform> special_params: SpecialParams;

// ============================================================================
// Helper Functions
// ============================================================================

// Error function using Abramowitz & Stegun approximation 7.1.26
fn erf_impl(x: f32) -> f32 {{
    if (x == 0.0) {{
        return 0.0;
    }}

    let sgn = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);

    // Constants for A&S 7.1.26
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * ax);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-ax * ax);

    return sgn * y;
}}

// Complementary error function
fn erfc_impl(x: f32) -> f32 {{
    return 1.0 - erf_impl(x);
}}

// Inverse error function using rational approximation
fn erfinv_impl(x: f32) -> f32 {{
    if (x <= -1.0) {{
        return -1e30; // -inf approximation
    }}
    if (x >= 1.0) {{
        return 1e30; // +inf approximation
    }}
    if (x == 0.0) {{
        return 0.0;
    }}

    let sgn = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);

    // Rational approximation for central region
    if (ax <= 0.7) {{
        let x2 = ax * ax;
        let r = ax * ((((-0.140543331 * x2 + 0.914624893) * x2 - 1.645349621) * x2 + 0.886226899) /
                     ((((0.012229801 * x2 - 0.329097515) * x2 + 1.442710462) * x2 - 2.118377725) * x2 + 1.0));
        return sgn * r;
    }}

    // Tail approximation
    let z = sqrt(-log((1.0 - ax) / 2.0));
    let r = (((1.641345311 * z + 3.429567803) * z - 1.624906493) * z - 1.970840454) /
            ((1.637067800 * z + 3.543889200) * z + 1.0);
    return sgn * r;
}}
{lgamma_helpers}

// Gamma function
fn gamma_impl(x: f32) -> f32 {{
    if (x <= 0.0 && x == floor(x)) {{
        return 1e30; // Pole
    }}
    return exp(lgamma_impl(x));
}}

// Digamma for positive x using asymptotic expansion (no recursion)
fn digamma_positive(x: f32) -> f32 {{
    var result = 0.0;
    var xx = x;

    // Recurrence to shift to large x where asymptotic works
    while (xx < 6.0) {{
        result = result - 1.0 / xx;
        xx = xx + 1.0;
    }}

    // Asymptotic expansion
    let x2 = 1.0 / (xx * xx);
    result = result + log(xx) - 0.5 / xx;
    result = result - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0)));

    return result;
}}

// Digamma function (non-recursive)
fn digamma_impl(x: f32) -> f32 {{
    if (x <= 0.0 && x == floor(x)) {{
        return 1e30; // Pole at non-positive integers
    }}

    // Reflection formula for negative x (non-recursive)
    if (x < 0.0) {{
        // For negative x, 1-x > 0, so we can call digamma_positive directly
        return digamma_positive(1.0 - x) - PI / tan(PI * x);
    }}

    return digamma_positive(x);
}}

// ============================================================================
// Compute Kernels
// ============================================================================

@compute @workgroup_size(256)
fn erf_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = erf_impl(special_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn erfc_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = erfc_impl(special_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn erfinv_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = erfinv_impl(special_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn gamma_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = gamma_impl(special_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn lgamma_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = lgamma_impl(special_a[idx]);
    }}
}}

@compute @workgroup_size(256)
fn digamma_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = digamma_impl(special_a[idx]);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        constants = common_constants(),
        lgamma_helpers = lgamma_helpers()
    ))
}
