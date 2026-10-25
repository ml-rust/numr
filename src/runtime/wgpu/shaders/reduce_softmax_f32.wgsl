// Softmax forward and backward. F32 only.
// Entry points: softmax_f32, softmax_bwd_f32

// ============================================================================
// Softmax (numerically stable)
// ============================================================================

struct SoftmaxParams {
    batch_size: u32,
    dim_size: u32,
}

@group(0) @binding(0) var<storage, read_write> softmax_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> softmax_output: array<f32>;
@group(0) @binding(2) var<uniform> softmax_params: SoftmaxParams;

var<workgroup> softmax_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) group_id: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x + group_id.y * nwg.x;

    if (batch_idx >= softmax_params.batch_size) {
        return;
    }

    let dim_size = softmax_params.dim_size;
    let base_offset = batch_idx * dim_size;

    // Step 1: Find max for numerical stability
    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var i: u32 = tid;
    while (i < dim_size) {
        max_val = max(max_val, softmax_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }

    softmax_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            softmax_shared[tid] = max(softmax_shared[tid], softmax_shared[tid + s]);
        }
        workgroupBarrier();
    }

    let global_max = softmax_shared[0];
    workgroupBarrier();

    // Step 2: Compute sum of exp(x - max)
    var sum: f32 = 0.0;
    i = tid;
    while (i < dim_size) {
        sum = sum + exp(softmax_input[base_offset + i] - global_max);
        i = i + WORKGROUP_SIZE;
    }

    softmax_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            softmax_shared[tid] = softmax_shared[tid] + softmax_shared[tid + s];
        }
        workgroupBarrier();
    }

    let global_sum = softmax_shared[0];
    workgroupBarrier();

    // Step 3: Compute output = exp(x - max) / sum
    i = tid;
    while (i < dim_size) {
        softmax_output[base_offset + i] = exp(softmax_input[base_offset + i] - global_max) / global_sum;
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Softmax Backward
// d_input = output * (grad - dot), where dot = sum(grad * output)
// Uses same SoftmaxParams (batch_size, dim_size)
// Bindings: 0=grad(read), 1=output(read), 2=d_input(write), 3=params
// ============================================================================

@group(0) @binding(0) var<storage, read> sbwd_grad: array<f32>;
@group(0) @binding(1) var<storage, read> sbwd_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> sbwd_d_input: array<f32>;
@group(0) @binding(3) var<uniform> sbwd_params: SoftmaxParams;

var<workgroup> sbwd_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_bwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x + group_id.y * nwg.x;

    if (batch_idx >= sbwd_params.batch_size) {
        return;
    }

    let dim_size = sbwd_params.dim_size;
    let base_offset = batch_idx * dim_size;

    // Pass 1: dot = sum(grad * output)
    var dot: f32 = 0.0;
    var i: u32 = tid;
    while (i < dim_size) {
        dot = dot + sbwd_grad[base_offset + i] * sbwd_output[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }

    sbwd_shared[tid] = dot;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sbwd_shared[tid] = sbwd_shared[tid] + sbwd_shared[tid + s];
        }
        workgroupBarrier();
    }

    let global_dot = sbwd_shared[0];
    workgroupBarrier();

    // Pass 2: d_input = output * (grad - dot)
    i = tid;
    while (i < dim_size) {
        let idx = base_offset + i;
        sbwd_d_input[idx] = sbwd_output[idx] * (sbwd_grad[idx] - global_dot);
        i = i + WORKGROUP_SIZE;
    }
}
