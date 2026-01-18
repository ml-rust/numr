//! WGSL shader source code for reduction operations
//!
//! Includes sum, mean, max, min reductions along specified dimensions.
//! Uses workgroup-level parallel reduction for efficiency.

/// Reduction shader module source (F32 only)
pub const REDUCE_SHADER: &str = r#"
// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Shared memory for parallel reduction
var<workgroup> reduce_shared: array<f32, 256>;

// ============================================================================
// Reduction Parameters
// ============================================================================

// For single-dim reduction:
// - reduce_size: number of elements to reduce (along the dimension)
// - outer_size: product of dims before the reduce dim
// - inner_size: product of dims after the reduce dim
// Total output elements = outer_size * inner_size
// Input index = outer * reduce_size * inner_size + reduce * inner_size + inner

struct ReduceParams {
    reduce_size: u32,   // Size of dimension being reduced
    outer_size: u32,    // Product of dims before reduce dim
    inner_size: u32,    // Product of dims after reduce dim
    numel_out: u32,     // Total output elements
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

    // Compute outer and inner indices from output index
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;

    // Base offset in input array
    let base_offset = outer * reduce_size * inner_size + inner;

    // Each thread loads and sums its portion of the reduction
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        sum = sum + reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    // Parallel tree reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    // First thread writes the final result
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

    // Compute outer and inner indices from output index
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;

    // Base offset in input array
    let base_offset = outer * reduce_size * inner_size + inner;

    // Each thread loads and sums its portion of the reduction
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        sum = sum + reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    // Parallel tree reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    // First thread writes the final result (divide by reduce_size for mean)
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

    // Compute outer and inner indices from output index
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;

    // Base offset in input array
    let base_offset = outer * reduce_size * inner_size + inner;

    // Each thread loads and finds max of its portion
    var max_val: f32 = -3.40282346638528859812e+38;  // -FLT_MAX
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        max_val = max(max_val, reduce_input[input_idx]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    // Parallel tree reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    // First thread writes the final result
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

    // Compute outer and inner indices from output index
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;

    // Base offset in input array
    let base_offset = outer * reduce_size * inner_size + inner;

    // Each thread loads and finds min of its portion
    var min_val: f32 = 3.40282346638528859812e+38;  // FLT_MAX
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        min_val = min(min_val, reduce_input[input_idx]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    // Parallel tree reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    // First thread writes the final result
    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Full Reduction (reduce all elements to single value)
// ============================================================================

struct FullReduceParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> full_reduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> full_reduce_output: array<f32>;
@group(0) @binding(2) var<uniform> full_reduce_params: FullReduceParams;

// Two-pass reduction for large arrays:
// Pass 1: Each workgroup reduces to partial result
// Pass 2: Reduce partial results to final value

@compute @workgroup_size(256)
fn full_reduce_sum_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    // Each thread loads and sums elements with stride
    var sum: f32 = 0.0;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        sum = sum + full_reduce_input[i];
        i = i + stride;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    // Parallel tree reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    // First thread writes partial result
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

    // Each thread loads and finds max with stride
    var max_val: f32 = -3.40282346638528859812e+38;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        max_val = max(max_val, full_reduce_input[i]);
        i = i + stride;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    // Parallel tree reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    // First thread writes partial result
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

    // Each thread loads and finds min with stride
    var min_val: f32 = 3.40282346638528859812e+38;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        min_val = min(min_val, full_reduce_input[i]);
        i = i + stride;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    // Parallel tree reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    // First thread writes partial result
    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

// ============================================================================
// Argmax / Argmin (returns index of max/min value)
// ============================================================================

// Shared memory for index tracking
var<workgroup> argmax_shared_val: array<f32, 256>;
var<workgroup> argmax_shared_idx: array<u32, 256>;

struct ArgReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> argreduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> argreduce_output: array<u32>;
@group(0) @binding(2) var<uniform> argreduce_params: ArgReduceParams;

@compute @workgroup_size(256)
fn argmax_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= argreduce_params.numel_out) {
        return;
    }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    // Each thread finds max in its portion
    var max_val: f32 = -3.40282346638528859812e+38;
    var max_idx: u32 = 0u;
    var i: u32 = tid;

    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        let val = argreduce_input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = max_val;
    argmax_shared_idx[tid] = max_idx;
    workgroupBarrier();

    // Parallel tree reduction tracking both value and index
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (argmax_shared_val[tid + s] > argmax_shared_val[tid]) {
                argmax_shared_val[tid] = argmax_shared_val[tid + s];
                argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        argreduce_output[output_idx] = argmax_shared_idx[0];
    }
}

@compute @workgroup_size(256)
fn argmin_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= argreduce_params.numel_out) {
        return;
    }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    // Each thread finds min in its portion
    var min_val: f32 = 3.40282346638528859812e+38;
    var min_idx: u32 = 0u;
    var i: u32 = tid;

    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        let val = argreduce_input[input_idx];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = min_val;
    argmax_shared_idx[tid] = min_idx;
    workgroupBarrier();

    // Parallel tree reduction tracking both value and index
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (argmax_shared_val[tid + s] < argmax_shared_val[tid]) {
                argmax_shared_val[tid] = argmax_shared_val[tid + s];
                argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        argreduce_output[output_idx] = argmax_shared_idx[0];
    }
}

// ============================================================================
// Softmax (numerically stable)
// ============================================================================

// Softmax over the last dimension (most common case)
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

struct SoftmaxParams {
    batch_size: u32,    // Product of all dims except the last
    dim_size: u32,      // Size of the dimension to softmax over
}

@group(0) @binding(0) var<storage, read_write> softmax_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> softmax_output: array<f32>;
@group(0) @binding(2) var<uniform> softmax_params: SoftmaxParams;

var<workgroup> softmax_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= softmax_params.batch_size) {
        return;
    }

    let dim_size = softmax_params.dim_size;
    let base_offset = batch_idx * dim_size;

    // Step 1: Find max for numerical stability
    var max_val: f32 = -3.40282346638528859812e+38;
    var i: u32 = tid;
    while (i < dim_size) {
        max_val = max(max_val, softmax_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }

    softmax_shared[tid] = max_val;
    workgroupBarrier();

    // Reduce to find global max
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

    // Reduce to find global sum
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
"#;
