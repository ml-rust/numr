// Strided cumulative sum shader for i32.
//
// Same `NumrI64` wide accumulator as cumsum_i32.wgsl - see its header comment.
// int_saturate.wgsl is prepended to this module too.

struct CumsumStridedParams {
    scan_size: u32,
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> params: CumsumStridedParams;

@compute @workgroup_size(256)
fn cumsum_strided_i32(
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

    var acc = NumrI64(0u, 0u);
    for (var s: u32 = 0u; s < params.scan_size; s = s + 1u) {
        let offset = outer_idx * params.scan_size * params.inner_size + s * params.inner_size + inner_idx;
        acc = numr_i64_add(acc, numr_i64_from_i32(input[offset]));
        output[offset] = numr_i64_to_i32_sat(acc);
    }
}
