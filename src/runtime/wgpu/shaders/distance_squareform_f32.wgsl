const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> condensed: array<f32>;
@group(0) @binding(1) var<storage, read_write> square: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n * params.n;
    if (idx >= total) {
        return;
    }

    let i = idx / params.n;
    let j = idx % params.n;

    if (i == j) {
        // Diagonal is zero
        square[idx] = 0.0;
    } else if (i < j) {
        // Upper triangle: k = n*i - i*(i+1)/2 + j - i - 1
        let k = params.n * i - i * (i + 1u) / 2u + j - i - 1u;
        square[idx] = condensed[k];
    } else {
        // Lower triangle: mirror from upper
        let k = params.n * j - j * (j + 1u) / 2u + i - j - 1u;
        square[idx] = condensed[k];
    }
}
