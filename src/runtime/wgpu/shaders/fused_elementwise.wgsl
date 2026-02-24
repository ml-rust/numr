// Fused elementwise WGSL shaders (F32 only)
// fused_mul_add: out = a * b + c
// fused_add_mul: out = (a + b) * c
// fused_mul_add_scalar: out = a * scale + bias

struct TernaryParams {
    numel: u32,
}

struct ScalarFmaParams {
    numel: u32,
    scale: f32,
    bias: f32,
    _pad: u32,
}

// ============================================================================
// Ternary ops: 3 inputs (a, b, c), 1 output
// ============================================================================

@group(0) @binding(0) var<storage, read_write> tern_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> tern_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> tern_c: array<f32>;
@group(0) @binding(3) var<storage, read_write> tern_out: array<f32>;
@group(0) @binding(4) var<uniform> tern_params: TernaryParams;

@compute @workgroup_size(256)
fn fused_mul_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < tern_params.numel) {
        tern_out[idx] = fma(tern_a[idx], tern_b[idx], tern_c[idx]);
    }
}

@compute @workgroup_size(256)
fn fused_add_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < tern_params.numel) {
        tern_out[idx] = (tern_a[idx] + tern_b[idx]) * tern_c[idx];
    }
}
