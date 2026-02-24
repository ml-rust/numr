// Log-sum-exp shader for f32
//
// Computes log(sum(exp(x))) in a numerically stable way:
// logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

struct LogsumexpParams {
    reduce_size: u32,
    outer_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: LogsumexpParams;

@compute @workgroup_size(256)
fn logsumexp_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {
        return;
    }

    let base = outer_idx * params.reduce_size;

    // Step 1: Find max value
    var max_val: f32 = -3.402823e+38;
    for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {
        let val = input[base + i];
        max_val = max(max_val, val);
    }

    // Step 2: Compute sum(exp(x - max))
    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {
        sum_exp = sum_exp + exp(input[base + i] - max_val);
    }

    // Step 3: Result = max + log(sum)
    output[outer_idx] = max_val + log(sum_exp);
}
