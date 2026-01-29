//! WGSL shader generation for special ternary functions
//!
//! Generates shaders for: betainc

use super::super::common::{dtype_suffix, wgsl_type};
use super::{common_constants, lgamma_helpers};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Generate WGSL shader for betainc (ternary: a, b, x)
pub fn generate_special_ternary_shader(dtype: DType) -> Result<String> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special functions (WebGPU requires F32)",
        });
    }

    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Auto-generated special ternary functions for {t}

{constants}

struct SpecialTernaryParams {{
    numel: u32,
}}

@group(0) @binding(0) var<storage, read_write> special_a: array<{t}>;
@group(0) @binding(1) var<storage, read_write> special_b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> special_x: array<{t}>;
@group(0) @binding(3) var<storage, read_write> special_out: array<{t}>;
@group(0) @binding(4) var<uniform> special_params: SpecialTernaryParams;

// ============================================================================
// Helper Functions (shared lgamma)
// ============================================================================
{lgamma_helpers}

// Regularized incomplete beta using continued fraction
fn betainc_cf(a: f32, b: f32, x: f32) -> f32 {{
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    var c = 1.0;
    var d = 1.0 - qab * x / qap;
    if (abs(d) < TINY) {{
        d = TINY;
    }}
    d = 1.0 / d;
    var h = d;

    for (var m = 1; m < MAX_ITER; m = m + 1) {{
        let m2 = 2 * m;

        var aa = f32(m) * (b - f32(m)) * x / ((qam + f32(m2)) * (a + f32(m2)));
        d = 1.0 + aa * d;
        if (abs(d) < TINY) {{
            d = TINY;
        }}
        c = 1.0 + aa / c;
        if (abs(c) < TINY) {{
            c = TINY;
        }}
        d = 1.0 / d;
        h = h * d * c;

        aa = -(a + f32(m)) * (qab + f32(m)) * x / ((a + f32(m2)) * (qap + f32(m2)));
        d = 1.0 + aa * d;
        if (abs(d) < TINY) {{
            d = TINY;
        }}
        c = 1.0 + aa / c;
        if (abs(c) < TINY) {{
            c = TINY;
        }}
        d = 1.0 / d;
        let delta = d * c;
        h = h * delta;

        if (abs(delta - 1.0) < EPSILON) {{
            break;
        }}
    }}

    let lnbeta = lgamma_impl(a) + lgamma_impl(b) - lgamma_impl(a + b);
    return exp(a * log(x) + b * log(1.0 - x) - lnbeta) * h / a;
}}

fn betainc_impl(a: f32, b: f32, x: f32) -> f32 {{
    if (x <= 0.0) {{
        return 0.0;
    }}
    if (x >= 1.0) {{
        return 1.0;
    }}

    // Use symmetry for better convergence (non-recursive version)
    if (x > (a + 1.0) / (a + b + 2.0)) {{
        // Compute directly without recursion using symmetry
        return 1.0 - betainc_cf(b, a, 1.0 - x);
    }}

    return betainc_cf(a, b, x);
}}

// ============================================================================
// Compute Kernels
// ============================================================================

@compute @workgroup_size(256)
fn betainc_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx < special_params.numel) {{
        special_out[idx] = betainc_impl(special_a[idx], special_b[idx], special_x[idx]);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        constants = common_constants(),
        lgamma_helpers = lgamma_helpers()
    ))
}
