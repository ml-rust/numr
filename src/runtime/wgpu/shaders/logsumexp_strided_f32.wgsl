fn f32_from_bits(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}

// Strided log-sum-exp shader for f32

struct LogsumexpStridedParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: LogsumexpStridedParams;

@compute @workgroup_size(256)
fn logsumexp_strided_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = global_id.x + global_id.y * nwg.x * 256u;
    let total_inner = params.outer_size * params.inner_size;
    if (idx >= total_inner) {
        return;
    }

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;

    // Step 1: Find max value along reduce dimension
    var max_val: f32 = -3.402823e+38;
    var all_negative_infinity = true;
    var any_nan = false;
    for (var r: u32 = 0u; r < params.reduce_size; r = r + 1u) {
        let offset = outer_idx * params.reduce_size * params.inner_size + r * params.inner_size + inner_idx;
        let val = input[offset];
        let bits = bitcast<u32>(val);
        all_negative_infinity = all_negative_infinity && bits == 0xff800000u;
        any_nan = any_nan || (bits & 0x7fffffffu) > 0x7f800000u;
        max_val = max(max_val, val);
    }

    if (any_nan) {
        output[outer_idx * params.inner_size + inner_idx] = f32_from_bits(0x7fc00000u);
        return;
    }
    if (all_negative_infinity) {
        output[outer_idx * params.inner_size + inner_idx] = f32_from_bits(0xff800000u);
        return;
    }

    // Step 2: Compute sum(exp(x - max))
    var sum_exp: f32 = 0.0;
    for (var r: u32 = 0u; r < params.reduce_size; r = r + 1u) {
        let offset = outer_idx * params.reduce_size * params.inner_size + r * params.inner_size + inner_idx;
        sum_exp = sum_exp + exp(input[offset] - max_val);
    }

    // Step 3: Write result
    output[outer_idx * params.inner_size + inner_idx] = max_val + log(sum_exp);
}
