// Dimension-wise reductions. F32 only.
// Entry points: reduce_sum_f32, reduce_mean_f32, reduce_max_f32, reduce_min_f32,
//               reduce_prod_f32, reduce_any_f32, reduce_all_f32
// Assembled with reduce_full_f32, reduce_arg_f32 and reduce_softmax_f32 in reduce.rs.

// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Shared memory for parallel reduction
var<workgroup> reduce_shared: array<f32, 256>;

// ============================================================================
// Reduction Parameters
// ============================================================================

struct ReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> reduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<f32>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

// ============================================================================
// Sum Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_sum_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        sum = sum + reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
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
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Mean Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_mean_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        sum = sum + reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
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
        reduce_output[output_idx] = reduce_shared[0] / f32(reduce_size);
    }
}

// ============================================================================
// Max Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_max_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        max_val = max(max_val, reduce_input[input_idx]);
        i = i + WORKGROUP_SIZE;
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
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Min Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_min_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: f32 = bitcast<f32>(0x7f800000u); // +inf: min's identity
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        min_val = min(min_val, reduce_input[input_idx]);
        i = i + WORKGROUP_SIZE;
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
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Product Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_prod_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var prod: f32 = 1.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        prod = prod * reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
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
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Any Reduction (returns 1.0 if any element is non-zero, 0.0 otherwise)
// ============================================================================

@compute @workgroup_size(256)
fn reduce_any_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var found_nonzero: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        if (reduce_input[input_idx] != 0.0) {
            found_nonzero = 1.0;
        }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = found_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// All Reduction (returns 1.0 if all elements are non-zero, 0.0 otherwise)
// ============================================================================

@compute @workgroup_size(256)
fn reduce_all_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var all_nonzero: f32 = 1.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        if (reduce_input[input_idx] == 0.0) {
            all_nonzero = 0.0;
        }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = all_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

