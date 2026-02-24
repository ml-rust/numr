// Stockham FFT shader for WebGPU
// Complex numbers as vec2<f32> (re, im)

const PI: f32 = 3.14159265358979323846;
const WORKGROUP_SIZE: u32 = 256u;

struct FftParams {
    n: u32,
    log_n: u32,
    inverse: i32,
    scale: f32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> fft_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fft_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> fft_params: FftParams;

// Workgroup shared memory for ping-pong
var<workgroup> smem_a: array<vec2<f32>, 256>;
var<workgroup> smem_b: array<vec2<f32>, 256>;

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

@compute @workgroup_size(WORKGROUP_SIZE)
fn stockham_fft_small(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
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
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {
        smem_a[i] = fft_input[base_offset + i];
    }
    workgroupBarrier();

    // Perform Stockham FFT stages
    var use_a = true;
    for (var stage: u32 = 0u; stage < log_n; stage = stage + 1u) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        for (var i = tid; i < n / 2u; i = i + WORKGROUP_SIZE) {
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

            if (use_a) {
                even_val = smem_a[even_idx];
                odd_val = cmul(smem_a[odd_idx], twiddle);
            } else {
                even_val = smem_b[even_idx];
                odd_val = cmul(smem_b[odd_idx], twiddle);
            }

            let sum = cadd(even_val, odd_val);
            let diff = csub(even_val, odd_val);

            if (use_a) {
                smem_b[out_even_idx] = sum;
                smem_b[out_odd_idx] = diff;
            } else {
                smem_a[out_even_idx] = sum;
                smem_a[out_odd_idx] = diff;
            }
        }

        workgroupBarrier();
        use_a = !use_a;
    }

    // Write output with scaling
    for (var i = tid; i < n; i = i + WORKGROUP_SIZE) {
        var result: vec2<f32>;
        if (use_a) {
            result = smem_a[i];
        } else {
            result = smem_b[i];
        }
        fft_output[base_offset + i] = cscale(result, scale_factor);
    }
}

// Single stage kernel for large FFTs (N > workgroup FFT size)
@compute @workgroup_size(WORKGROUP_SIZE)
fn stockham_fft_stage(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let n = fft_params.n;
    let stage = fft_params.log_n;  // Reuse log_n as current stage
    let inverse = fft_params.inverse;
    let batch_idx = gid.y;

    let sign = select(-1.0, 1.0, inverse != 0);

    let m = 1u << (stage + 1u);
    let half_m = 1u << stage;

    let i = gid.x;
    if (i >= n / 2u) {
        return;
    }

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
}

// Scale complex array
@compute @workgroup_size(WORKGROUP_SIZE)
fn scale_complex(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let n = fft_params.n;
    let scale_factor = fft_params.scale;

    if (idx < n) {
        fft_output[idx] = cscale(fft_input[idx], scale_factor);
    }
}
