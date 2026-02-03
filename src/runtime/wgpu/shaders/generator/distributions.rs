//! WGSL shader generation for probability distribution sampling operations
//!
//! Provides shaders for:
//! - Bernoulli: Binary outcomes with probability p
//! - Beta: Continuous on [0, 1] with shape parameters
//! - Gamma: Continuous on [0, inf) with shape/scale
//! - Exponential: Continuous on [0, inf) with rate
//! - Poisson: Discrete counts with rate lambda
//! - Binomial: Discrete successes in n trials
//! - Laplace: Double exponential distribution
//! - Chi-squared: Sum of squared normals
//! - Student's t: Heavy-tailed distribution
//! - F: Ratio of chi-squared variates

use super::common::{dtype_suffix, wgsl_type};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// PCG random number generator for WGSL with distribution helpers
const DISTRIBUTION_RNG_WGSL: &str = r#"
// PCG hash function for random number generation
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_init(seed: u32, idx: u32) -> u32 {
    return pcg_hash(seed ^ pcg_hash(idx));
}

fn pcg_uniform(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / 4294967296.0;
}

// Box-Muller for normal distribution
fn sample_normal(state: ptr<function, u32>) -> f32 {
    let u1 = max(pcg_uniform(state), 0.0000001);
    let u2 = pcg_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// Gamma via Marsaglia-Tsang method
fn sample_gamma_mt(state: ptr<function, u32>, shape: f32, scale: f32) -> f32 {
    var alpha = shape;
    var boost = 1.0;

    // Handle shape < 1 by boosting
    if alpha < 1.0 {
        boost = pow(pcg_uniform(state), 1.0 / alpha);
        alpha = alpha + 1.0;
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / sqrt(9.0 * d);

    // Rejection sampling
    for (var i = 0u; i < 100u; i = i + 1u) {
        var x: f32;
        var v: f32;

        // Generate valid v
        for (var j = 0u; j < 100u; j = j + 1u) {
            x = sample_normal(state);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }

        v = v * v * v;
        let u = pcg_uniform(state);
        let x2 = x * x;

        // Accept/reject
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v * boost * scale;
        }
        if log(u) < 0.5 * x2 + d * (1.0 - v + log(v)) {
            return d * v * boost * scale;
        }
    }

    // Fallback (should rarely reach)
    return d * boost * scale;
}
"#;

fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Generate WGSL shader for Bernoulli distribution sampling
pub fn generate_bernoulli_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "bernoulli")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Bernoulli distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct BernoulliParams {{
    numel: u32,
    seed: u32,
    p: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: BernoulliParams;

@compute @workgroup_size(256)
fn bernoulli_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        let u = pcg_uniform(&state);
        out[idx] = select({t}(0.0), {t}(1.0), u < params.p);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Beta distribution sampling
pub fn generate_beta_dist_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "beta")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Beta distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct BetaParams {{
    numel: u32,
    seed: u32,
    alpha: f32,
    beta: f32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: BetaParams;

@compute @workgroup_size(256)
fn beta_dist_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        let x = sample_gamma_mt(&state, params.alpha, 1.0);
        let y = sample_gamma_mt(&state, params.beta, 1.0);
        out[idx] = {t}(x / (x + y));
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Gamma distribution sampling
pub fn generate_gamma_dist_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "gamma")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Gamma distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct GammaParams {{
    numel: u32,
    seed: u32,
    shape: f32,
    scale: f32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: GammaParams;

@compute @workgroup_size(256)
fn gamma_dist_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        out[idx] = {t}(sample_gamma_mt(&state, params.shape, params.scale));
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Exponential distribution sampling
pub fn generate_exponential_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "exponential")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Exponential distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct ExponentialParams {{
    numel: u32,
    seed: u32,
    rate: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: ExponentialParams;

@compute @workgroup_size(256)
fn exponential_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        let u = max(pcg_uniform(&state), 0.0000001);
        out[idx] = {t}(-log(u) / params.rate);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Poisson distribution sampling
pub fn generate_poisson_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "poisson")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Poisson distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct PoissonParams {{
    numel: u32,
    seed: u32,
    lambda: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: PoissonParams;

@compute @workgroup_size(256)
fn poisson_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);

        // Knuth's algorithm for small lambda
        if params.lambda < 30.0 {{
            let L = exp(-params.lambda);
            var k = 0u;
            var p = 1.0;

            for (var i = 0u; i < 1000u; i = i + 1u) {{
                p = p * pcg_uniform(&state);
                if p <= L {{
                    break;
                }}
                k = k + 1u;
            }}
            out[idx] = {t}(f32(k));
        }} else {{
            // Normal approximation for large lambda
            let z = sample_normal(&state);
            let result = max(0.0, round(params.lambda + sqrt(params.lambda) * z));
            out[idx] = {t}(result);
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Binomial distribution sampling
pub fn generate_binomial_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "binomial")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Binomial distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct BinomialParams {{
    numel: u32,
    seed: u32,
    n_trials: u32,
    p: f32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: BinomialParams;

@compute @workgroup_size(256)
fn binomial_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);

        let n = params.n_trials;
        let p = params.p;

        // Direct simulation for small n
        if n <= 64u {{
            var successes = 0u;
            for (var i = 0u; i < n; i = i + 1u) {{
                if pcg_uniform(&state) < p {{
                    successes = successes + 1u;
                }}
            }}
            out[idx] = {t}(f32(successes));
        }} else {{
            // Normal approximation for large n
            let mean = f32(n) * p;
            let std_dev = sqrt(mean * (1.0 - p));
            let z = sample_normal(&state);
            let result = clamp(round(mean + std_dev * z), 0.0, f32(n));
            out[idx] = {t}(result);
        }}
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Laplace distribution sampling
pub fn generate_laplace_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "laplace")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Laplace distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct LaplaceParams {{
    numel: u32,
    seed: u32,
    loc: f32,
    scale: f32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: LaplaceParams;

@compute @workgroup_size(256)
fn laplace_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        let u = pcg_uniform(&state) - 0.5;
        let result = params.loc - params.scale * sign(u) * log(1.0 - 2.0 * abs(u));
        out[idx] = {t}(result);
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Chi-squared distribution sampling
pub fn generate_chi_squared_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "chi_squared")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Chi-squared distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct ChiSquaredParams {{
    numel: u32,
    seed: u32,
    df: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: ChiSquaredParams;

@compute @workgroup_size(256)
fn chi_squared_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        // Chi-squared(df) = Gamma(df/2, 2)
        out[idx] = {t}(sample_gamma_mt(&state, params.df / 2.0, 2.0));
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for Student's t distribution sampling
pub fn generate_student_t_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "student_t")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Student's t distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct StudentTParams {{
    numel: u32,
    seed: u32,
    df: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: StudentTParams;

@compute @workgroup_size(256)
fn student_t_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        let z = sample_normal(&state);
        let chi2 = sample_gamma_mt(&state, params.df / 2.0, 2.0);
        out[idx] = {t}(z / sqrt(chi2 / params.df));
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for F distribution sampling
pub fn generate_f_distribution_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "f_distribution")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// F distribution sampling for {t}
{rng}
const WORKGROUP_SIZE: u32 = 256u;

struct FDistributionParams {{
    numel: u32,
    seed: u32,
    df1: f32,
    df2: f32,
}}

@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@group(0) @binding(1) var<uniform> params: FDistributionParams;

@compute @workgroup_size(256)
fn f_distribution_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx < params.numel {{
        var state = pcg_init(params.seed, idx);
        let chi2_1 = sample_gamma_mt(&state, params.df1 / 2.0, 2.0);
        let chi2_2 = sample_gamma_mt(&state, params.df2 / 2.0, 2.0);
        out[idx] = {t}((chi2_1 / params.df1) / (chi2_2 / params.df2));
    }}
}}
"#,
        t = t,
        suffix = suffix,
        rng = DISTRIBUTION_RNG_WGSL
    ))
}

/// Generate WGSL shader for multinomial count operation
///
/// Performs CDF lookup for uniform samples and counts occurrences per category.
/// Used for multinomial sampling: given uniform samples and a CDF, counts how
/// many samples fall into each category.
pub fn generate_multinomial_count_shader(dtype: DType) -> Result<String> {
    check_float_dtype(dtype, "multinomial_count")?;
    let t = wgsl_type(dtype)?;
    let suffix = dtype_suffix(dtype)?;

    Ok(format!(
        r#"// Multinomial count shader for {t}
// Performs CDF lookup for uniform samples and counts occurrences per category

const WORKGROUP_SIZE: u32 = 256u;

struct MultinomialCountParams {{
    k: u32,           // Number of categories
    n_trials: u32,    // Number of trials per sample
    n_samples: u32,   // Number of samples
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> cdf: array<{t}>;
@group(0) @binding(1) var<storage, read_write> uniforms: array<{t}>;
@group(0) @binding(2) var<storage, read_write> counts: array<{t}>;
@group(0) @binding(3) var<uniform> params: MultinomialCountParams;

// Binary search to find category for uniform sample
fn find_category(u: {t}, k: u32) -> u32 {{
    var lo: u32 = 0u;
    var hi: u32 = k;
    while (lo < hi) {{
        let mid = lo + (hi - lo) / 2u;
        if (cdf[mid] <= u) {{
            lo = mid + 1u;
        }} else {{
            hi = mid;
        }}
    }}
    return min(lo, k - 1u);
}}

@compute @workgroup_size(256)
fn multinomial_count_{suffix}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let sample_idx = global_id.x;
    let k = params.k;
    let n_trials = params.n_trials;
    let n_samples = params.n_samples;

    if (sample_idx >= n_samples) {{
        return;
    }}

    // Initialize counts for this sample to zero
    for (var c: u32 = 0u; c < k; c++) {{
        counts[sample_idx * k + c] = {t}(0.0);
    }}

    // Process each trial
    for (var t_idx: u32 = 0u; t_idx < n_trials; t_idx++) {{
        let u = uniforms[sample_idx * n_trials + t_idx];
        let category = find_category(u, k);
        counts[sample_idx * k + category] += {t}(1.0);
    }}
}}
"#,
        t = t,
        suffix = suffix,
    ))
}
