fn f32_from_bits(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}

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
fn logsumexp_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let outer_idx = global_id.x + global_id.y * nwg.x * 256u;
    if (outer_idx >= params.outer_size) {
        return;
    }

    let base = outer_idx * params.reduce_size;

    // Step 1: Find max value
    var max_val: f32 = -3.402823e+38;
    var all_negative_infinity = true;
    var any_nan = false;
    for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {
        let val = input[base + i];
        let bits = bitcast<u32>(val);
        all_negative_infinity = all_negative_infinity && bits == 0xff800000u;
        any_nan = any_nan || (bits & 0x7fffffffu) > 0x7f800000u;
        max_val = max(max_val, val);
    }

    if (any_nan) {
        output[outer_idx] = f32_from_bits(0x7fc00000u);
        return;
    }
    if (all_negative_infinity) {
        output[outer_idx] = f32_from_bits(0xff800000u);
        return;
    }

    // Step 2: Compute sum(exp(x - max))
    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {
        sum_exp = sum_exp + exp(input[base + i] - max_val);
    }

    // Step 3: Result = max + log(sum)
    output[outer_idx] = max_val + log(sum_exp);
}
