// Extract unique elements from a sorted i32 array using atomic counter

struct CountParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> sorted_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> unique_output: array<i32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn extract_unique_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numel) {
        return;
    }
    if (idx == 0u || sorted_input[idx] != sorted_input[idx - 1u]) {
        let out_idx = atomicAdd(&counter[0], 1u);
        unique_output[out_idx] = sorted_input[idx];
    }
}
