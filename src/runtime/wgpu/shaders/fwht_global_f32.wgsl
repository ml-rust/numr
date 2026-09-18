// Walsh-Hadamard transform, global pass, f32.
//
// One dispatch applies a single stride-`h` butterfly in place over the whole
// output buffer, for a stride too wide for the workgroup tile of
// fwht_local_f32.wgsl. The launcher runs one dispatch per stride
// `h = chunk, 2*chunk, ..., block_size / 2`, after the local pass.
//
// The Sylvester transform is a product of commuting stride-h butterflies, so
// applying the wide strides here after the narrow strides of the local pass
// is a valid decomposition, and the scale already folded in there carries
// through because every butterfly is linear.
//
// Invocation `p` owns the pair `(i, i + h)` with `i = (p / h) * 2h + p % h`,
// and no other invocation touches either element, so the in-place update on
// a storage buffer is race-free. `block_size` is a multiple of `2h` and
// `last_dim` a multiple of `block_size`, so a pair never crosses a block or
// a row.

struct FwhtParams {
    rows: u32,
    last_dim: u32,
    block_size: u32,
    chunk: u32,
    h: u32,
    has_signs: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: FwhtParams;

const WG: u32 = 256u;
// Workgroups per dispatch row; the launcher splits the pair index over x and y.
const GRID_X: u32 = 65535u;

@compute @workgroup_size(256)
fn fwht_global_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let pairs = (params.rows * params.last_dim) / 2u;
    let p = gid.x + wid.y * (GRID_X * WG);
    if (p >= pairs) {
        return;
    }
    let h = params.h;
    let i = (p / h) * (2u * h) + (p % h);
    let j = i + h;
    let u = output[i];
    let v = output[j];
    output[i] = u + v;
    output[j] = u - v;
}
