// 2:4 Structured Sparsity: Prune to 2:4 format (F32 only)
//
// For each group of 4 consecutive elements along K, keeps the 2 with largest magnitude.
// One workgroup thread per group.

struct Params {
    total_groups: u32,
    num_groups_per_row: u32,
    meta_cols: u32,
    half_k: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> dense: array<f32>;
@group(0) @binding(1) var<storage, read_write> compressed: array<f32>;
@group(0) @binding(2) var<storage, read_write> metadata: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sparse_24_prune_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.total_groups) {
        return;
    }

    let row = tid / params.num_groups_per_row;
    let g = tid % params.num_groups_per_row;
    let base = row * params.k + g * 4u;

    // Load 4 values
    let v0 = dense[base];
    let v1 = dense[base + 1u];
    let v2 = dense[base + 2u];
    let v3 = dense[base + 3u];

    // Compute magnitudes
    let m0 = abs(v0);
    let m1 = abs(v1);
    let m2 = abs(v2);
    let m3 = abs(v3);

    // Find top-2 by magnitude using selection network
    var idx0: u32 = 0u;
    var idx1: u32 = 1u;
    var mag0 = m0;
    var mag1 = m1;

    if (mag1 > mag0) {
        let ti = idx0; idx0 = idx1; idx1 = ti;
        let tf = mag0; mag0 = mag1; mag1 = tf;
    }

    if (m2 > mag1) {
        idx1 = 2u; mag1 = m2;
        if (mag1 > mag0) {
            let ti = idx0; idx0 = idx1; idx1 = ti;
            let tf = mag0; mag0 = mag1; mag1 = tf;
        }
    }

    if (m3 > mag1) {
        idx1 = 3u; mag1 = m3;
        if (mag1 > mag0) {
            let ti = idx0; idx0 = idx1; idx1 = ti;
        }
    }

    let first = min(idx0, idx1);
    let second = max(idx0, idx1);

    // Write compressed values
    let out_base = row * params.half_k + g * 2u;
    let vals = array<f32, 4>(v0, v1, v2, v3);
    compressed[out_base] = vals[first];
    compressed[out_base + 1u] = vals[second];

    // Build 4-bit bitmask and atomically OR into metadata
    let mask = (1u << first) | (1u << second);
    let word_idx = g / 8u;
    let nibble_idx = g % 8u;
    let meta_offset = row * params.meta_cols + word_idx;
    atomicOr(&metadata[meta_offset], mask << (nibble_idx * 4u));
}
