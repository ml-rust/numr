// Reflect-mode pad for f32

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

// Use vec4<u32> for 16-byte alignment in uniform buffer
struct PadReflectParams {
    ndim: u32,
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    src_shape: array<vec4<u32>, 2>,    // 8 u32 values packed into 2 vec4
    out_shape: array<vec4<u32>, 2>,
    pad_before: array<vec4<u32>, 2>,
}

// Helper to access packed array<vec4<u32>, 2> by index
fn get_packed_value(arr: array<vec4<u32>, 2>, d: i32) -> u32 {
    let vec_idx = u32(d) / 4u;
    let comp_idx = u32(d) % 4u;
    if (vec_idx == 0u) {
        if (comp_idx == 0u) { return arr[0].x; }
        else if (comp_idx == 1u) { return arr[0].y; }
        else if (comp_idx == 2u) { return arr[0].z; }
        else { return arr[0].w; }
    } else {
        if (comp_idx == 0u) { return arr[1].x; }
        else if (comp_idx == 1u) { return arr[1].y; }
        else if (comp_idx == 2u) { return arr[1].z; }
        else { return arr[1].w; }
    }
}

// Mirror an output coordinate back into the source tensor, excluding the
// edge element. The caller (`validate_reflect_pad`) guarantees `before` and
// `after` are each strictly less than `size`, so a single mirror bounce
// always lands back inside `0..size`.
fn reflect_coord(out_coord: i32, before: i32, size: i32) -> i32 {
    let rel = out_coord - before;
    if (rel < 0) {
        return -rel;
    }
    if (rel < size) {
        return rel;
    }
    let over = rel - size;
    return size - 2 - over;
}

@group(0) @binding(0) var<storage, read_write> pad_reflect_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> pad_reflect_dst: array<f32>;
@group(0) @binding(2) var<uniform> pad_reflect_params: PadReflectParams;

@compute @workgroup_size(256)
fn pad_reflect_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= pad_reflect_params.total_elements) {
        return;
    }

    var remaining = idx;
    var coords: array<u32, 8>;

    for (var d = i32(pad_reflect_params.ndim) - 1; d >= 0; d = d - 1) {
        let out_dim = get_packed_value(pad_reflect_params.out_shape, d);
        coords[d] = remaining % out_dim;
        remaining = remaining / out_dim;
    }

    var src_idx = 0u;
    var src_stride = 1u;
    for (var d = i32(pad_reflect_params.ndim) - 1; d >= 0; d = d - 1) {
        let size = i32(get_packed_value(pad_reflect_params.src_shape, d));
        let before = i32(get_packed_value(pad_reflect_params.pad_before, d));
        let src_coord = u32(reflect_coord(i32(coords[d]), before, size));
        src_idx = src_idx + src_coord * src_stride;
        src_stride = src_stride * get_packed_value(pad_reflect_params.src_shape, d);
    }
    pad_reflect_dst[idx] = pad_reflect_src[src_idx];
}
