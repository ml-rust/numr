//! WGSL shader source code for normalization operations
//!
//! Includes RMS normalization and Layer normalization.
//! Both use workgroup-level parallel reductions for efficiency.

/// Normalization shader module source (F32 only)
pub const NORM_SHADER: &str = r#"
// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> norm_shared: array<f32, 256>;

// ============================================================================
// RMS Normalization
// ============================================================================
// rms_norm(x, weight, eps) = x / sqrt(mean(x^2) + eps) * weight
// Applied to last dimension

struct RmsNormParams {
    batch_size: u32,    // Product of all dims except the last
    hidden_size: u32,   // Size of the last dimension
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> rms_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> rms_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> rms_output: array<f32>;
@group(0) @binding(3) var<uniform> rms_params: RmsNormParams;

@compute @workgroup_size(256)
fn rms_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= rms_params.batch_size) {
        return;
    }

    let hidden_size = rms_params.hidden_size;
    let eps = rms_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Step 1: Compute sum of squares
    var sum_sq: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let val = rms_input[base_offset + i];
        sum_sq = sum_sq + val * val;
        i = i + WORKGROUP_SIZE;
    }

    norm_shared[tid] = sum_sq;
    workgroupBarrier();

    // Reduce to get total sum of squares
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            norm_shared[tid] = norm_shared[tid] + norm_shared[tid + s];
        }
        workgroupBarrier();
    }

    // Compute RMS: sqrt(mean(x^2) + eps)
    let rms = sqrt(norm_shared[0] / f32(hidden_size) + eps);
    workgroupBarrier();

    // Step 2: Normalize and apply weight
    i = tid;
    while (i < hidden_size) {
        rms_output[base_offset + i] = rms_input[base_offset + i] / rms * rms_weight[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Layer Normalization
// ============================================================================
// layer_norm(x, weight, bias, eps) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
// Applied to last dimension

struct LayerNormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> ln_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> ln_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> ln_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> ln_output: array<f32>;
@group(0) @binding(4) var<uniform> ln_params: LayerNormParams;

var<workgroup> ln_shared_mean: array<f32, 256>;
var<workgroup> ln_shared_var: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= ln_params.batch_size) {
        return;
    }

    let hidden_size = ln_params.hidden_size;
    let eps = ln_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Step 1: Compute mean
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        sum = sum + ln_input[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }

    ln_shared_mean[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ln_shared_mean[tid] = ln_shared_mean[tid] + ln_shared_mean[tid + s];
        }
        workgroupBarrier();
    }

    let mean = ln_shared_mean[0] / f32(hidden_size);
    workgroupBarrier();

    // Step 2: Compute variance
    var var_sum: f32 = 0.0;
    i = tid;
    while (i < hidden_size) {
        let diff = ln_input[base_offset + i] - mean;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }

    ln_shared_var[tid] = var_sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ln_shared_var[tid] = ln_shared_var[tid] + ln_shared_var[tid + s];
        }
        workgroupBarrier();
    }

    let variance = ln_shared_var[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply affine transformation
    i = tid;
    while (i < hidden_size) {
        let normalized = (ln_input[base_offset + i] - mean) * inv_std;
        ln_output[base_offset + i] = normalized * ln_weight[i] + ln_bias[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Layer Normalization without bias
// ============================================================================

@group(0) @binding(0) var<storage, read_write> ln_nb_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> ln_nb_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> ln_nb_output: array<f32>;
@group(0) @binding(3) var<uniform> ln_nb_params: LayerNormParams;

@compute @workgroup_size(256)
fn layer_norm_no_bias_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= ln_nb_params.batch_size) {
        return;
    }

    let hidden_size = ln_nb_params.hidden_size;
    let eps = ln_nb_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Step 1: Compute mean
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        sum = sum + ln_nb_input[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }

    ln_shared_mean[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ln_shared_mean[tid] = ln_shared_mean[tid] + ln_shared_mean[tid + s];
        }
        workgroupBarrier();
    }

    let mean = ln_shared_mean[0] / f32(hidden_size);
    workgroupBarrier();

    // Step 2: Compute variance
    var var_sum: f32 = 0.0;
    i = tid;
    while (i < hidden_size) {
        let diff = ln_nb_input[base_offset + i] - mean;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }

    ln_shared_var[tid] = var_sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ln_shared_var[tid] = ln_shared_var[tid] + ln_shared_var[tid + s];
        }
        workgroupBarrier();
    }

    let variance = ln_shared_var[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply weight only
    i = tid;
    while (i < hidden_size) {
        let normalized = (ln_nb_input[base_offset + i] - mean) * inv_std;
        ln_nb_output[base_offset + i] = normalized * ln_nb_weight[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Group Normalization
// ============================================================================
// group_norm(x, weight, bias, num_groups) normalizes over groups of channels
// Each group is normalized independently over the spatial and channel dimensions

struct GroupNormParams {
    batch_size: u32,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    channels_per_group: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> gn_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> gn_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> gn_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> gn_output: array<f32>;
@group(0) @binding(4) var<uniform> gn_params: GroupNormParams;

var<workgroup> gn_shared_mean: array<f32, 256>;
var<workgroup> gn_shared_var: array<f32, 256>;

@compute @workgroup_size(256)
fn group_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let bg_id = group_id.x;  // batch_id * num_groups + group_id

    let batch_size = gn_params.batch_size;
    let channels = gn_params.channels;
    let spatial = gn_params.spatial;
    let num_groups = gn_params.num_groups;
    let channels_per_group = gn_params.channels_per_group;
    let eps = gn_params.eps;

    if (bg_id >= batch_size * num_groups) {
        return;
    }

    let batch_id = bg_id / num_groups;
    let group_id_val = bg_id % num_groups;
    let c_start = group_id_val * channels_per_group;
    let group_size = channels_per_group * spatial;

    // Compute base offset in flattened NCHW layout
    // offset = batch_id * channels * spatial + group_id * channels_per_group * spatial
    let batch_offset = batch_id * channels * spatial;
    let group_offset = batch_offset + c_start * spatial;

    // Step 1: Compute sum for mean
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < group_size) {
        let c_offset = i / spatial;
        let s_offset = i % spatial;
        let idx = group_offset + c_offset * spatial + s_offset;
        sum = sum + gn_input[idx];
        i = i + WORKGROUP_SIZE;
    }

    gn_shared_mean[tid] = sum;
    workgroupBarrier();

    // Reduce sum to compute mean
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gn_shared_mean[tid] = gn_shared_mean[tid] + gn_shared_mean[tid + s];
        }
        workgroupBarrier();
    }

    let mean = gn_shared_mean[0] / f32(group_size);
    workgroupBarrier();

    // Step 2: Compute sum of squared differences for variance
    var var_sum: f32 = 0.0;
    i = tid;
    while (i < group_size) {
        let c_offset = i / spatial;
        let s_offset = i % spatial;
        let idx = group_offset + c_offset * spatial + s_offset;
        let diff = gn_input[idx] - mean;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }

    gn_shared_var[tid] = var_sum;
    workgroupBarrier();

    // Reduce variance
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gn_shared_var[tid] = gn_shared_var[tid] + gn_shared_var[tid + s];
        }
        workgroupBarrier();
    }

    let variance = gn_shared_var[0] / f32(group_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply per-channel weight and bias
    i = tid;
    while (i < group_size) {
        let c_offset = i / spatial;
        let s_offset = i % spatial;
        let idx = group_offset + c_offset * spatial + s_offset;
        let channel = c_start + c_offset;
        let normalized = (gn_input[idx] - mean) * inv_std;
        gn_output[idx] = normalized * gn_weight[channel] + gn_bias[channel];
        i = i + WORKGROUP_SIZE;
    }
}
"#;
