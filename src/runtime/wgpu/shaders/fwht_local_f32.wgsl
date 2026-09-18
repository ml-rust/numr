// Walsh-Hadamard transform, local pass, f32.
//
// One workgroup owns one CHUNK-wide tile of the flattened buffer. It loads
// the tile into workgroup memory (multiplying by `signs` and the
// `1/sqrt(block_size)` scale on the way in), runs every butterfly stride
// `h < min(block_size, CHUNK)`, and stores the tile back.
//
// The Sylvester transform is a product of commuting stride-h butterflies, so
// running the strides below CHUNK here and the strides at or above CHUNK in
// fwht_global_f32.wgsl afterwards is a valid decomposition, and the scale
// folds into the load because the butterflies are linear.
//
// CHUNK is a power of two, at least 256, and divides `last_dim` whenever
// `block_size >= CHUNK`. The launcher rewrites the CHUNK line below before
// compiling, so one module exists per tile width.

struct FwhtParams {
    rows: u32,
    last_dim: u32,
    block_size: u32,
    chunk: u32,
    h: u32,
    has_signs: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> signs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: FwhtParams;

const WG: u32 = 256u;
const CHUNK: u32 = 4096u;
// Workgroups per dispatch row; the launcher splits the tile index over x and y.
const GRID_X: u32 = 65535u;

var<workgroup> tile: array<f32, CHUNK>;

@compute @workgroup_size(256)
fn fwht_local_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let total = params.rows * params.last_dim;
    let tile_idx = wid.x + wid.y * GRID_X;
    let base = tile_idx * CHUNK;
    if (base >= total) {
        return;
    }
    let t = lid.x;

    // Load, applying signs and the normalization scale.
    for (var i: u32 = t; i < CHUNK; i = i + WG) {
        let idx = base + i;
        if (idx < total) {
            var v = input[idx];
            if (params.has_signs != 0u) {
                v = v * signs[idx % params.last_dim];
            }
            tile[i] = v * params.scale;
        }
    }
    workgroupBarrier();

    // Butterfly strides that fit inside the tile. `total` is a multiple of
    // `block_size`, so a pair whose low index is in range has its high index
    // in range too.
    let local_limit = min(params.block_size, CHUNK);
    let half = CHUNK / 2u;
    for (var h: u32 = 1u; h < local_limit; h = h * 2u) {
        for (var p: u32 = t; p < half; p = p + WG) {
            let i = (p / h) * (2u * h) + (p % h);
            if (base + i < total) {
                let j = i + h;
                let u = tile[i];
                let v = tile[j];
                tile[i] = u + v;
                tile[j] = u - v;
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = t; i < CHUNK; i = i + WG) {
        let idx = base + i;
        if (idx < total) {
            output[idx] = tile[i];
        }
    }
}
