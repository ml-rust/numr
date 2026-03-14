// Fused elementwise scalar WGSL shader (F32 only)
// fused_mul_add_scalar: out = a * scale + bias

struct ScalarFmaParams {
    numel: u32,
    scale: f32,
    bias: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> sfma_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> sfma_out: array<f32>;
@group(0) @binding(2) var<uniform> sfma_params: ScalarFmaParams;

@compute @workgroup_size(256)
fn fused_mul_add_scalar_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < sfma_params.numel) {
        sfma_out[idx] = fma(sfma_a[idx], sfma_params.scale, sfma_params.bias);
    }
}
