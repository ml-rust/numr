// 2:4 Structured Sparsity: Decompress to dense format (F32 only)
//
// Reconstructs dense matrix from compressed 2:4 format.
// One workgroup thread per group of 4 output elements.

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

@group(0) @binding(0) var<storage, read> compressed: array<f32>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> dense: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sparse_24_decompress_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.total_groups) {
        return;
    }

    let row = tid / params.num_groups_per_row;
    let g = tid % params.num_groups_per_row;

    // Read metadata
    let word_idx = g / 8u;
    let nibble_idx = g % 8u;
    let word = metadata[row * params.meta_cols + word_idx];
    let mask = (word >> (nibble_idx * 4u)) & 0xFu;

    // Read 2 compressed values
    let in_base = row * params.half_k + g * 2u;
    let v0 = compressed[in_base];
    let v1 = compressed[in_base + 1u];

    // Write to dense
    let out_base = row * params.k + g * 4u;
    dense[out_base] = 0.0;
    dense[out_base + 1u] = 0.0;
    dense[out_base + 2u] = 0.0;
    dense[out_base + 3u] = 0.0;

    var val_idx: u32 = 0u;
    for (var bit: u32 = 0u; bit < 4u; bit = bit + 1u) {
        if ((mask & (1u << bit)) != 0u) {
            if (val_idx == 0u) {
                dense[out_base + bit] = v0;
            } else {
                dense[out_base + bit] = v1;
            }
            val_idx = val_idx + 1u;
        }
    }
}
