// Convert flat indices to multi-dimensional indices

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 8u;

struct FlatToMultiParams {
    nnz: u32,
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
    shape: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> flat_indices: array<i32>;
@group(0) @binding(1) var<storage, read_write> multi_indices: array<i32>;
@group(0) @binding(2) var<uniform> params: FlatToMultiParams;

fn get_shape_dim(d: u32) -> u32 {
    return params.shape[d / 4u][d % 4u];
}

@compute @workgroup_size(256)
fn flat_to_multi_index(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.nnz) {
        return;
    }

    var flat_idx = u32(flat_indices[idx]);
    let ndim = params.ndim;

    // Compute strides on the fly (row-major)
    // and convert flat index to multi-index
    for (var d: u32 = ndim; d > 0u; d = d - 1u) {
        let dim = d - 1u;
        let dim_size = get_shape_dim(dim);
        let coord = flat_idx % dim_size;
        flat_idx = flat_idx / dim_size;

        // Store: multi_indices[idx * ndim + dim] = coord
        multi_indices[idx * ndim + dim] = i32(coord);
    }
}
