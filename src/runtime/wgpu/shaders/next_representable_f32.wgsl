// Integer bindings preserve NaN payloads, signaling bits, and subnormal encodings.
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn adjacent_bits(bits: u32, upward: bool) -> u32 {
    let magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return bits;
    }
    if (magnitude == 0u) {
        return select(0x80000001u, 1u, upward);
    }
    let negative = (bits & 0x80000000u) != 0u;
    if (magnitude == 0x7f800000u && upward != negative) {
        return bits;
    }
    if (upward != negative) {
        return bits + 1u;
    }
    return bits - 1u;
}

@compute @workgroup_size(256)
fn next_representable_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = gid.x + gid.y * groups.x * 256u;
    if (index >= params[0]) {
        return;
    }
    let rank = params[1];
    var offset = params[2];
    var remaining = index;
    var dimension = rank;
    loop {
        if (dimension == 0u) {
            break;
        }
        dimension = dimension - 1u;
        let extent = params[4u + dimension];
        let coordinate = remaining % extent;
        remaining = remaining / extent;
        // Signed strides use modular u32 arithmetic after host address-range checks.
        offset = offset + coordinate * params[4u + rank + dimension];
    }
    output[index] = adjacent_bits(input[offset], params[3] != 0u);
}
