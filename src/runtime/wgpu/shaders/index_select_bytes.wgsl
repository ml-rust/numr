// index_select over raw bytes, for any element size.
//
// One thread writes one output word. Each of its 4 bytes is read from the
// source slice its index selects. The output buffer is allocated in 4-byte
// units, so bytes past `total_bytes` are padding and receive zeros.
//
// A dispatch dimension holds at most 65535 workgroups, so threads spread
// over a 2-D grid.

const GRID_X: u32 = 65535u;
const WORKGROUP_SIZE: u32 = 256u;

struct IndexSelectBytesParams {
    outer_size: u32,
    dim_size: u32,
    inner_bytes: u32,
    index_len: u32,
    total_bytes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: IndexSelectBytesParams;

@compute @workgroup_size(256)
fn index_select_bytes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x + gid.y * GRID_X * WORKGROUP_SIZE;
    let first = word * 4u;
    if (first >= params.total_bytes) {
        return;
    }
    let slice_bytes = params.index_len * params.inner_bytes;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let b = first + k;
        if (b >= params.total_bytes) {
            break;
        }
        let inner = b % params.inner_bytes;
        let sel = (b / params.inner_bytes) % params.index_len;
        let outer = b / slice_bytes;
        let index_val = indices[sel];
        if (index_val < 0 || u32(index_val) >= params.dim_size) {
            continue;
        }
        let src = (outer * params.dim_size + u32(index_val)) * params.inner_bytes + inner;
        let v = (input[src >> 2u] >> ((src & 3u) * 8u)) & 0xffu;
        packed = packed | (v << (k * 8u));
    }
    output[word] = packed;
}
