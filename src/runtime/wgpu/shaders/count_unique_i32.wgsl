// Count unique elements in a sorted i32 array

var<workgroup> shared_count: array<u32, 256>;

struct CountParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> sorted_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> count_output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: CountParams;

@compute @workgroup_size(256)
fn count_unique_i32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let numel = params.numel;

    var local_count: u32 = 0u;
    let idx = global_id.x;
    if (idx < numel) {
        if (idx == 0u || sorted_input[idx] != sorted_input[idx - 1u]) {
            local_count = 1u;
        }
    }

    shared_count[tid] = local_count;
    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_count[tid] = shared_count[tid] + shared_count[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        atomicAdd(&count_output[0], shared_count[0]);
    }
}
