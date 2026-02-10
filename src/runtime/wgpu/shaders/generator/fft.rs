//! WGSL shader generation for FFT operations
//!
//! Generates Stockham FFT shaders using `vec2<f32>` for complex numbers.
//! WGSL doesn't have native complex type, so we use vec2 (re, im).

use crate::error::Result;

/// Maximum FFT size for shared memory implementation
pub const MAX_WORKGROUP_FFT_SIZE: usize = 256;

/// Generate complex arithmetic helper functions
fn complex_helpers() -> &'static str {
    r#"
// Complex number helpers (vec2: x=real, y=imag)
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

fn csub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

fn cscale(a: vec2<f32>, s: f32) -> vec2<f32> {
    return vec2<f32>(a.x * s, a.y * s);
}

fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

// Compute e^(i*theta) = cos(theta) + i*sin(theta)
fn cexp_i(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}
"#
}

/// Generate batched Stockham FFT shader for small transforms
///
/// Each workgroup processes one FFT. Uses workgroup shared memory for ping-pong.
pub fn generate_stockham_fft_shader() -> Result<String> {
    Ok(format!(
        r#"// Stockham FFT shader for WebGPU
// Complex numbers as vec2<f32> (re, im)

const PI: f32 = 3.14159265358979323846;
const WORKGROUP_SIZE: u32 = 256u;

struct FftParams {{
    n: u32,
    log_n: u32,
    inverse: i32,
    scale: f32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}}

@group(0) @binding(0) var<storage, read_write> fft_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fft_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> fft_params: FftParams;

// Workgroup shared memory for ping-pong
var<workgroup> smem_a: array<vec2<f32>, {max_size}>;
var<workgroup> smem_b: array<vec2<f32>, {max_size}>;
{complex_helpers}

@compute @workgroup_size(WORKGROUP_SIZE)
fn stockham_fft_small(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let batch_idx = wg_id.x;
    let tid = local_id.x;
    let n = fft_params.n;
    let log_n = fft_params.log_n;
    let inverse = fft_params.inverse;
    let scale_factor = fft_params.scale;

    // Sign for twiddle factor
    let sign = select(-1.0, 1.0, inverse != 0);

    // Load input to shared memory
    let base_offset = batch_idx * n;
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {{
        smem_a[i] = fft_input[base_offset + i];
    }}
    workgroupBarrier();

    // Perform Stockham FFT stages
    var use_a = true;
    for (var stage: u32 = 0u; stage < log_n; stage = stage + 1u) {{
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {{
            let group = i / half_m;
            let pair = i % half_m;

            let even_idx = group * half_m + pair;
            let odd_idx = even_idx + n / 2u;

            let out_even_idx = group * m + pair;
            let out_odd_idx = out_even_idx + half_m;

            // Twiddle factor
            let theta = sign * 2.0 * PI * f32(pair) / f32(m);
            let twiddle = cexp_i(theta);

            var even_val: vec2<f32>;
            var odd_val: vec2<f32>;

            if (use_a) {{
                even_val = smem_a[even_idx];
                odd_val = cmul(smem_a[odd_idx], twiddle);
            }} else {{
                even_val = smem_b[even_idx];
                odd_val = cmul(smem_b[odd_idx], twiddle);
            }}

            let sum = cadd(even_val, odd_val);
            let diff = csub(even_val, odd_val);

            if (use_a) {{
                smem_b[out_even_idx] = sum;
                smem_b[out_odd_idx] = diff;
            }} else {{
                smem_a[out_even_idx] = sum;
                smem_a[out_odd_idx] = diff;
            }}
        }}

        workgroupBarrier();
        use_a = !use_a;
    }}

    // Write output with scaling
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {{
        var result: vec2<f32>;
        if (use_a) {{
            result = smem_a[i];
        }} else {{
            result = smem_b[i];
        }}
        fft_output[base_offset + i] = cscale(result, scale_factor);
    }}
}}

// Single stage kernel for large FFTs (N > workgroup FFT size)
@compute @workgroup_size(WORKGROUP_SIZE)
fn stockham_fft_stage(
    @builtin(global_invocation_id) gid: vec3<u32>
) {{
    let n = fft_params.n;
    let stage = fft_params.log_n;  // Reuse log_n as current stage
    let inverse = fft_params.inverse;
    let batch_idx = gid.y;

    let sign = select(-1.0, 1.0, inverse != 0);

    let m = 1u << (stage + 1u);
    let half_m = 1u << stage;

    let i = gid.x;
    if (i >= n / 2u) {{
        return;
    }}

    let group = i / half_m;
    let pair = i % half_m;

    let base_offset = batch_idx * n;
    let even_idx = base_offset + group * half_m + pair;
    let odd_idx = even_idx + n / 2u;

    let out_even_idx = base_offset + group * m + pair;
    let out_odd_idx = out_even_idx + half_m;

    // Twiddle factor
    let theta = sign * 2.0 * PI * f32(pair) / f32(m);
    let twiddle = cexp_i(theta);

    let even_val = fft_input[even_idx];
    let odd_val = cmul(fft_input[odd_idx], twiddle);

    fft_output[out_even_idx] = cadd(even_val, odd_val);
    fft_output[out_odd_idx] = csub(even_val, odd_val);
}}

// Scale complex array
@compute @workgroup_size(WORKGROUP_SIZE)
fn scale_complex(
    @builtin(global_invocation_id) gid: vec3<u32>
) {{
    let idx = gid.x;
    let n = fft_params.n;
    let scale_factor = fft_params.scale;

    if (idx < n) {{
        fft_output[idx] = cscale(fft_input[idx], scale_factor);
    }}
}}
"#,
        max_size = MAX_WORKGROUP_FFT_SIZE,
        complex_helpers = complex_helpers()
    ))
}

/// Generate FFT shift shader
pub fn generate_fftshift_shader() -> Result<String> {
    Ok(format!(
        r#"// FFT shift shader - shifts zero-frequency to center

const WORKGROUP_SIZE: u32 = 256u;

struct ShiftParams {{
    n: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read_write> shift_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> shift_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> shift_params: ShiftParams;
{complex_helpers}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fftshift(
    @builtin(global_invocation_id) gid: vec3<u32>
) {{
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = shift_params.n;

    if (idx >= n) {{
        return;
    }}

    let base_offset = batch_idx * n;
    let half_n = n / 2u;

    // Swap first half with second half
    var src_idx: u32;
    if (idx < half_n) {{
        src_idx = idx + half_n;
    }} else {{
        src_idx = idx - half_n;
    }}

    shift_output[base_offset + idx] = shift_input[base_offset + src_idx];
}}

@compute @workgroup_size(WORKGROUP_SIZE)
fn ifftshift(
    @builtin(global_invocation_id) gid: vec3<u32>
) {{
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = shift_params.n;

    if (idx >= n) {{
        return;
    }}

    let base_offset = batch_idx * n;
    let half_n = (n + 1u) / 2u;  // Ceiling division for odd n

    // Inverse shift
    var src_idx: u32;
    if (idx < n - half_n) {{
        src_idx = idx + half_n;
    }} else {{
        src_idx = idx - (n - half_n);
    }}

    shift_output[base_offset + idx] = shift_input[base_offset + src_idx];
}}
"#,
        complex_helpers = complex_helpers()
    ))
}

/// Generate rfft pack shader (real to complex)
pub fn generate_rfft_pack_shader() -> Result<String> {
    Ok(r#"// rfft pack shader - converts real input to complex

const WORKGROUP_SIZE: u32 = 256u;

struct PackParams {
    n: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> pack_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> pack_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> pack_params: PackParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn rfft_pack(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = pack_params.n;

    if (idx >= n) {
        return;
    }

    let in_offset = batch_idx * n;
    let out_offset = batch_idx * n;

    pack_output[out_offset + idx] = vec2<f32>(pack_input[in_offset + idx], 0.0);
}
"#
    .to_string())
}

/// Generate irfft unpack shader (complex to real)
pub fn generate_irfft_unpack_shader() -> Result<String> {
    Ok(r#"// irfft unpack shader - extracts real part from complex

const WORKGROUP_SIZE: u32 = 256u;

struct UnpackParams {
    n: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> unpack_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> unpack_output: array<f32>;
@group(0) @binding(2) var<uniform> unpack_params: UnpackParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn irfft_unpack(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = unpack_params.n;

    if (idx >= n) {
        return;
    }

    let in_offset = batch_idx * n;
    let out_offset = batch_idx * n;

    unpack_output[out_offset + idx] = unpack_input[in_offset + idx].x;
}
"#
    .to_string())
}

/// Generate Hermitian extend shader for rfft
pub fn generate_hermitian_extend_shader() -> Result<String> {
    Ok(
        r#"// Hermitian extend shader - extends N/2+1 complex to N complex using symmetry

const WORKGROUP_SIZE: u32 = 256u;

struct ExtendParams {
    n: u32,         // Full FFT size
    half_n: u32,    // N/2 + 1 (input size)
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> extend_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> extend_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> extend_params: ExtendParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn hermitian_extend(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = extend_params.n;
    let half_n = extend_params.half_n;

    if (idx >= n) {
        return;
    }

    let in_offset = batch_idx * half_n;
    let out_offset = batch_idx * n;

    if (idx < half_n) {
        // Direct copy for first half
        extend_output[out_offset + idx] = extend_input[in_offset + idx];
    } else {
        // Conjugate symmetry for second half: X[N-k] = conj(X[k])
        let k = n - idx;
        let val = extend_input[in_offset + k];
        extend_output[out_offset + idx] = vec2<f32>(val.x, -val.y);
    }
}
"#
        .to_string(),
    )
}

/// Generate rfft truncate shader
pub fn generate_rfft_truncate_shader() -> Result<String> {
    Ok(
        r#"// rfft truncate shader - keeps only N/2+1 complex values from full FFT

const WORKGROUP_SIZE: u32 = 256u;

struct TruncateParams {
    n: u32,         // Full FFT size (input)
    half_n: u32,    // N/2 + 1 (output size)
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> truncate_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> truncate_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> truncate_params: TruncateParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn rfft_truncate(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = truncate_params.n;
    let half_n = truncate_params.half_n;

    if (idx >= half_n) {
        return;
    }

    let in_offset = batch_idx * n;
    let out_offset = batch_idx * half_n;

    truncate_output[out_offset + idx] = truncate_input[in_offset + idx];
}
"#
        .to_string(),
    )
}

/// Generate copy complex shader
pub fn generate_copy_complex_shader() -> Result<String> {
    Ok(r#"// Copy complex array

const WORKGROUP_SIZE: u32 = 256u;

struct CopyParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> copy_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> copy_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_complex(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let n = copy_params.n;

    if (idx < n) {
        copy_output[idx] = copy_input[idx];
    }
}
"#
    .to_string())
}
