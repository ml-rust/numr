// Cumulative product shader for i32

struct CumprodParams {
    scan_size: u32,
    outer_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> params: CumprodParams;

@compute @workgroup_size(256)
fn cumprod_i32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {
        return;
    }

    let base = outer_idx * params.scan_size;
    var acc: i32 = 1;
    for (var i: u32 = 0u; i < params.scan_size; i = i + 1u) {
        acc = acc * input[base + i];
        output[base + i] = acc;
    }
}
