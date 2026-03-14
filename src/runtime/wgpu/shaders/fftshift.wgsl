// FFT shift shader - shifts zero-frequency to center

const WORKGROUP_SIZE: u32 = 256u;

struct ShiftParams {
    n: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> shift_input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> shift_output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> shift_params: ShiftParams;

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
fn fftshift(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = shift_params.n;

    if (idx >= n) {
        return;
    }

    let base_offset = batch_idx * n;
    let half_n = n / 2u;

    // Swap first half with second half
    var src_idx: u32;
    if (idx < half_n) {
        src_idx = idx + half_n;
    } else {
        src_idx = idx - half_n;
    }

    shift_output[base_offset + idx] = shift_input[base_offset + src_idx];
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn ifftshift(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let batch_idx = gid.y;
    let n = shift_params.n;

    if (idx >= n) {
        return;
    }

    let base_offset = batch_idx * n;
    let half_n = (n + 1u) / 2u;  // Ceiling division for odd n

    // Inverse shift
    var src_idx: u32;
    if (idx < n - half_n) {
        src_idx = idx + half_n;
    } else {
        src_idx = idx - (n - half_n);
    }

    shift_output[base_offset + idx] = shift_input[base_offset + src_idx];
}
