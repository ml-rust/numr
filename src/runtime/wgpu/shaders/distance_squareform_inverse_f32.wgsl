const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> square: array<f32>;
@group(0) @binding(1) var<storage, read_write> condensed: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Convert condensed index k to (i, j) where i < j
fn condensed_to_ij(k: u32, n: u32) -> vec2<u32> {
    var i: u32 = 0u;
    var count: u32 = 0u;
    loop {
        let row_count = n - 1u - i;
        if (count + row_count > k) {
            let j = k - count + i + 1u;
            return vec2<u32>(i, j);
        }
        count += row_count;
        i++;
    }
    return vec2<u32>(0u, 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let total = params.n * (params.n - 1u) / 2u;
    if (k >= total) {
        return;
    }

    let ij = condensed_to_ij(k, params.n);
    let i = ij.x;
    let j = ij.y;

    condensed[k] = square[i * params.n + j];
}
