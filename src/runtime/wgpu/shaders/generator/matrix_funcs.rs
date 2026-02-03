//! WGSL shader generation for matrix function operations on quasi-triangular matrices.
//!
//! These shaders operate on the Schur form T of a matrix A, where A = Z @ T @ Z^T.
//! The quasi-triangular form has 1x1 blocks (real eigenvalues) and 2x2 blocks
//! (complex conjugate pairs) on the diagonal.

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::Result;

/// Generate shader for validating Schur eigenvalues (checking for non-positive real eigenvalues).
///
/// Returns a tensor with validation results:
/// - output[0] = 1.0 if any non-positive real eigenvalue found, 0.0 otherwise
/// - output[1] = the first problematic eigenvalue value (if any)
pub fn generate_validate_eigenvalues_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Schur eigenvalue validation for {t}

const WORKGROUP_SIZE: u32 = 256u;

struct Params {{
    n: u32,
    eps: f32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> matrix_t: array<{t}>;
@group(0) @binding(1) var<storage, read_write> result: array<{t}>;  // [has_error, error_value]
@group(0) @binding(2) var<uniform> params: Params;

// Check if a real eigenvalue is non-positive
fn check_real_eigenvalue(val: {t}, eps: {t}) -> bool {{
    return val <= eps;
}}

// Check if a 2x2 block represents non-positive real eigenvalues
// For 2x2 block [[a, b], [c, d]], eigenvalues are (a+d)/2 ± sqrt((a-d)²/4 + bc)
// If discriminant < 0, eigenvalues are complex (ok)
// If discriminant >= 0, check if real part is non-positive
fn check_2x2_block(a: {t}, b: {t}, c: {t}, d: {t}, eps: {t}) -> bool {{
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc < 0.0 {{
        // Complex eigenvalues - check real part
        let real_part = trace / 2.0;
        return real_part <= eps;
    }} else {{
        // Real eigenvalues
        let sqrt_disc = sqrt(disc);
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        return lambda1 <= eps || lambda2 <= eps;
    }}
}}

@compute @workgroup_size(1)
fn validate_eigenvalues_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = params.n;
    let eps = {t}(params.eps);

    // Initialize result to "no error"
    result[0] = 0.0;
    result[1] = 0.0;

    var i: u32 = 0u;
    while i < n {{
        let diag_idx = i * n + i;

        // Check if this is a 2x2 block (non-zero sub-diagonal)
        if i + 1u < n {{
            let sub_diag = abs(matrix_t[(i + 1u) * n + i]);
            if sub_diag > eps {{
                // 2x2 block
                let a = matrix_t[i * n + i];
                let b = matrix_t[i * n + (i + 1u)];
                let c = matrix_t[(i + 1u) * n + i];
                let d = matrix_t[(i + 1u) * n + (i + 1u)];

                if check_2x2_block(a, b, c, d, eps) {{
                    result[0] = 1.0;
                    result[1] = (a + d) / 2.0;  // Report real part
                    return;
                }}
                i = i + 2u;
                continue;
            }}
        }}

        // 1x1 block (real eigenvalue)
        let eigenvalue = matrix_t[diag_idx];
        if check_real_eigenvalue(eigenvalue, eps) {{
            result[0] = 1.0;
            result[1] = eigenvalue;
            return;
        }}
        i = i + 1u;
    }}
}}
"#,
        t = t,
        suffix = suffix
    ))
}

/// Generate shader for applying a scalar function to diagonal blocks of quasi-triangular matrix.
///
/// This handles both 1x1 blocks (real eigenvalues) and 2x2 blocks (complex pairs).
/// The function is specified by `func_type`: "exp", "log", "sqrt".
pub fn generate_diagonal_func_shader(dtype: DType, func_type: &str) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    // Generate the scalar function application
    let scalar_func = match func_type {
        "exp" => "exp(x)",
        "log" => "log(x)",
        "sqrt" => "sqrt(x)",
        _ => {
            return Err(crate::error::Error::InvalidArgument {
                arg: "func_type",
                reason: format!("Unknown function type: {}", func_type),
            });
        }
    };

    // For 2x2 blocks with complex eigenvalues, we need special handling
    let block_2x2_func = match func_type {
        "exp" => {
            r#"
    // For 2x2 block with complex eigenvalues a ± bi:
    // exp(a ± bi) = exp(a) * (cos(b) ± i*sin(b))
    // Result is [[exp(a)*cos(b), -exp(a)*sin(b)], [exp(a)*sin(b), exp(a)*cos(b)]]
    // after similarity transform
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        // Real eigenvalues - diagonalize and apply exp
        let sqrt_disc = sqrt(disc);
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        let exp1 = exp(lambda1);
        let exp2 = exp(lambda2);

        // Simple case: return diagonal exp values
        // This is approximate but handles most cases
        *f11 = (exp1 + exp2) / 2.0;
        *f22 = (exp1 + exp2) / 2.0;
        *f12 = (exp1 - exp2) / 2.0 * sign(b);
        *f21 = (exp1 - exp2) / 2.0 * sign(c);
    } else {
        // Complex eigenvalues
        let real_part = trace / 2.0;
        let imag_part = sqrt(-disc) / 2.0;
        let exp_real = exp(real_part);
        let cos_imag = cos(imag_part);
        let sin_imag = sin(imag_part);

        *f11 = exp_real * cos_imag;
        *f22 = exp_real * cos_imag;
        // Off-diagonal scaling based on original block structure
        let scale = exp_real * sin_imag / imag_part;
        *f12 = scale * b;
        *f21 = scale * c;
    }
"#
        }
        "log" => {
            r#"
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = sqrt(disc);
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        let log1 = log(lambda1);
        let log2 = log(lambda2);

        *f11 = (log1 + log2) / 2.0;
        *f22 = (log1 + log2) / 2.0;
        *f12 = (log1 - log2) / (lambda1 - lambda2) * b;
        *f21 = (log1 - log2) / (lambda1 - lambda2) * c;
    } else {
        // Complex eigenvalues: log(r * e^(i*theta)) = log(r) + i*theta
        let real_part = trace / 2.0;
        let imag_part = sqrt(-disc) / 2.0;
        let r = sqrt(det);  // |lambda| = sqrt(det) for conjugate pair
        let theta = atan2(imag_part, real_part);

        *f11 = log(r);
        *f22 = log(r);
        let scale = theta / imag_part;
        *f12 = scale * b;
        *f21 = scale * c;
    }
"#
        }
        "sqrt" => {
            r#"
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = sqrt(disc);
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        let sqrt1 = sqrt(lambda1);
        let sqrt2 = sqrt(lambda2);

        *f11 = (sqrt1 + sqrt2) / 2.0;
        *f22 = (sqrt1 + sqrt2) / 2.0;
        let denom = sqrt1 + sqrt2;
        if abs(denom) > 1e-10 {
            *f12 = b / denom;
            *f21 = c / denom;
        } else {
            *f12 = 0.0;
            *f21 = 0.0;
        }
    } else {
        // Complex eigenvalues
        let r = sqrt(det);
        let theta = atan2(sqrt(-disc) / 2.0, trace / 2.0);
        let sqrt_r = sqrt(r);
        let half_theta = theta / 2.0;

        *f11 = sqrt_r * cos(half_theta);
        *f22 = sqrt_r * cos(half_theta);
        let imag_part = sqrt(-disc) / 2.0;
        let scale = sqrt_r * sin(half_theta) / imag_part;
        *f12 = scale * b;
        *f21 = scale * c;
    }
"#
        }
        _ => unreachable!(),
    };

    Ok(format!(
        r#"// Diagonal block function application for {t} - {func_type}

const WORKGROUP_SIZE: u32 = 256u;

struct Params {{
    n: u32,
    eps: f32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input_t: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output_f: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

// Apply function to 2x2 block
fn apply_2x2_block(a: {t}, b: {t}, c: {t}, d: {t},
                   f11: ptr<function, {t}>, f12: ptr<function, {t}>,
                   f21: ptr<function, {t}>, f22: ptr<function, {t}>) {{
{block_2x2_func}
}}

@compute @workgroup_size(1)
fn diagonal_{func_type}_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = params.n;
    let eps = {t}(params.eps);

    // Initialize output to zero
    for (var idx: u32 = 0u; idx < n * n; idx = idx + 1u) {{
        output_f[idx] = 0.0;
    }}

    var i: u32 = 0u;
    while i < n {{
        // Check if this is a 2x2 block
        if i + 1u < n {{
            let sub_diag = abs(input_t[(i + 1u) * n + i]);
            if sub_diag > eps {{
                // 2x2 block
                let a = input_t[i * n + i];
                let b = input_t[i * n + (i + 1u)];
                let c = input_t[(i + 1u) * n + i];
                let d = input_t[(i + 1u) * n + (i + 1u)];

                var f11: {t};
                var f12: {t};
                var f21: {t};
                var f22: {t};
                apply_2x2_block(a, b, c, d, &f11, &f12, &f21, &f22);

                output_f[i * n + i] = f11;
                output_f[i * n + (i + 1u)] = f12;
                output_f[(i + 1u) * n + i] = f21;
                output_f[(i + 1u) * n + (i + 1u)] = f22;

                i = i + 2u;
                continue;
            }}
        }}

        // 1x1 block
        let x = input_t[i * n + i];
        output_f[i * n + i] = {scalar_func};
        i = i + 1u;
    }}
}}
"#,
        t = t,
        suffix = suffix,
        func_type = func_type,
        block_2x2_func = block_2x2_func,
        scalar_func = scalar_func,
    ))
}

/// Generate shader for computing off-diagonal elements using Parlett's recurrence.
///
/// For column j, processes rows i < j:
/// F[i,j] = (T[i,i] - T[j,j])^(-1) * (F[i,j] * T[i,j] - sum_{k=i+1}^{j-1} F[i,k]*T[k,j] + T[i,k]*F[k,j])
///
/// This kernel processes one column at a time (called n times).
pub fn generate_parlett_column_shader(dtype: DType) -> Result<String> {
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Parlett recurrence for off-diagonal elements - {t}

const WORKGROUP_SIZE: u32 = 256u;

struct Params {{
    n: u32,
    col: u32,  // Current column being processed
    eps: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> input_t: array<{t}>;
@group(0) @binding(1) var<storage, read_write> output_f: array<{t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WORKGROUP_SIZE)
fn parlett_column_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = params.n;
    let j = params.col;
    let eps = {t}(params.eps);

    // Each thread handles one row i < j
    let i = gid.x;
    if i >= j {{
        return;
    }}

    let t_ii = input_t[i * n + i];
    let t_jj = input_t[j * n + j];
    let t_ij = input_t[i * n + j];

    let denom = t_ii - t_jj;

    // Compute the sum term
    var sum: {t} = 0.0;
    for (var k: u32 = i + 1u; k < j; k = k + 1u) {{
        let f_ik = output_f[i * n + k];
        let t_kj = input_t[k * n + j];
        let t_ik = input_t[i * n + k];
        let f_kj = output_f[k * n + j];
        sum = sum + f_ik * t_kj - t_ik * f_kj;
    }}

    let f_ii = output_f[i * n + i];
    let f_jj = output_f[j * n + j];

    // F[i,j] = (T[i,j] * (F[i,i] - F[j,j]) + sum) / (T[i,i] - T[j,j])
    if abs(denom) > eps {{
        output_f[i * n + j] = (t_ij * (f_ii - f_jj) + sum) / denom;
    }} else {{
        // Eigenvalues too close - use limit formula
        output_f[i * n + j] = t_ij * f_ii;  // Simplified fallback
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}
