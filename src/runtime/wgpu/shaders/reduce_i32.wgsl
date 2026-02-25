// Reduction operations for I32.
// Entry points: reduce_sum_i32, reduce_max_i32, reduce_min_i32,
//               reduce_prod_i32, reduce_any_i32, reduce_all_i32,
//               full_reduce_sum_i32, full_reduce_max_i32, full_reduce_min_i32, full_reduce_prod_i32,
//               argmax_i32, argmin_i32

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> reduce_shared: array<i32, 256>;

struct ReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> reduce_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<i32>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

@compute @workgroup_size(256)
fn reduce_sum_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var sum: i32 = 0;
    var i: u32 = tid;
    while (i < reduce_size) {
        sum = sum + reduce_input[base_offset + i * inner_size];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s]; }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_max_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: i32 = (-2147483647i - 1i);
    var i: u32 = tid;
    while (i < reduce_size) {
        max_val = max(max_val, reduce_input[base_offset + i * inner_size]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_min_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: i32 = 2147483647i;
    var i: u32 = tid;
    while (i < reduce_size) {
        min_val = min(min_val, reduce_input[base_offset + i * inner_size]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_prod_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var prod: i32 = 1;
    var i: u32 = tid;
    while (i < reduce_size) {
        prod = prod * reduce_input[base_offset + i * inner_size];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = prod;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = reduce_shared[tid] * reduce_shared[tid + s]; }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_any_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var found_nonzero: i32 = 0;
    var i: u32 = tid;
    while (i < reduce_size) {
        if (reduce_input[base_offset + i * inner_size] != 0) { found_nonzero = 1; }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = found_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn reduce_all_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= reduce_params.numel_out) { return; }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var all_nonzero: i32 = 1;
    var i: u32 = tid;
    while (i < reduce_size) {
        if (reduce_input[base_offset + i * inner_size] == 0) { all_nonzero = 0; }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = all_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }

    if (tid == 0u) { reduce_output[output_idx] = reduce_shared[0]; }
}

// ============================================================================
// Full Reduction
// ============================================================================

struct FullReduceParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> full_reduce_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> full_reduce_output: array<i32>;
@group(0) @binding(2) var<uniform> full_reduce_params: FullReduceParams;

@compute @workgroup_size(256)
fn full_reduce_sum_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var sum: i32 = 0;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;
    while (i < numel) { sum = sum + full_reduce_input[i]; i = i + stride; }

    reduce_shared[tid] = sum;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { full_reduce_output[wid] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn full_reduce_max_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var max_val: i32 = (-2147483647i - 1i);
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;
    while (i < numel) { max_val = max(max_val, full_reduce_input[i]); i = i + stride; }

    reduce_shared[tid] = max_val;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { full_reduce_output[wid] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn full_reduce_min_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var min_val: i32 = 2147483647i;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;
    while (i < numel) { min_val = min(min_val, full_reduce_input[i]); i = i + stride; }

    reduce_shared[tid] = min_val;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { full_reduce_output[wid] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn full_reduce_prod_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>,
                        @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var prod: i32 = 1;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;
    while (i < numel) { prod = prod * full_reduce_input[i]; i = i + stride; }

    reduce_shared[tid] = prod;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = reduce_shared[tid] * reduce_shared[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { full_reduce_output[wid] = reduce_shared[0]; }
}

// ============================================================================
// Argmax / Argmin
// ============================================================================

var<workgroup> argmax_shared_val: array<i32, 256>;
var<workgroup> argmax_shared_idx: array<u32, 256>;

struct ArgReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> argreduce_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> argreduce_output: array<u32>;
@group(0) @binding(2) var<uniform> argreduce_params: ArgReduceParams;

@compute @workgroup_size(256)
fn argmax_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= argreduce_params.numel_out) { return; }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: i32 = (-2147483647i - 1i);
    var max_idx: u32 = 0u;
    var i: u32 = tid;
    while (i < reduce_size) {
        let val = argreduce_input[base_offset + i * inner_size];
        if (val > max_val) { max_val = val; max_idx = i; }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = max_val;
    argmax_shared_idx[tid] = max_idx;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (argmax_shared_val[tid + s] > argmax_shared_val[tid]) {
                argmax_shared_val[tid] = argmax_shared_val[tid + s];
                argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) { argreduce_output[output_idx] = argmax_shared_idx[0]; }
}

@compute @workgroup_size(256)
fn argmin_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= argreduce_params.numel_out) { return; }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: i32 = 2147483647i;
    var min_idx: u32 = 0u;
    var i: u32 = tid;
    while (i < reduce_size) {
        let val = argreduce_input[base_offset + i * inner_size];
        if (val < min_val) { min_val = val; min_idx = i; }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = min_val;
    argmax_shared_idx[tid] = min_idx;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (argmax_shared_val[tid + s] < argmax_shared_val[tid]) {
                argmax_shared_val[tid] = argmax_shared_val[tid + s];
                argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) { argreduce_output[output_idx] = argmax_shared_idx[0]; }
}
