// Full reductions (all elements to one value). F32 only.
// Entry points: full_reduce_sum_f32, full_reduce_max_f32, full_reduce_min_f32, full_reduce_prod_f32
// Uses WORKGROUP_SIZE and reduce_shared from reduce_f32.wgsl.

// ============================================================================
// Full Reduction (reduce all elements to single value)
// ============================================================================

struct FullReduceParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> full_reduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> full_reduce_output: array<f32>;
@group(0) @binding(2) var<uniform> full_reduce_params: FullReduceParams;

@compute @workgroup_size(256)
fn full_reduce_sum_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var sum: f32 = 0.0;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        sum = sum + full_reduce_input[i];
        i = i + stride;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

@compute @workgroup_size(256)
fn full_reduce_max_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        max_val = max(max_val, full_reduce_input[i]);
        i = i + stride;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

@compute @workgroup_size(256)
fn full_reduce_min_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var min_val: f32 = bitcast<f32>(0x7f800000u); // +inf: min's identity
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        min_val = min(min_val, full_reduce_input[i]);
        i = i + stride;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

@compute @workgroup_size(256)
fn full_reduce_prod_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>,
                        @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var prod: f32 = 1.0;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        prod = prod * full_reduce_input[i];
        i = i + stride;
    }

    reduce_shared[tid] = prod;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] * reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

