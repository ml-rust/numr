// where_broadcast_cond: condition=f32, output=f32
// out[i] = cond[cond_offset] != 0.0 ? x[x_offset] : y[y_offset] (with broadcasting)

struct WhereBroadcastParams {
    numel: u32,
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> bc_cond: array<f32>;
@group(0) @binding(1) var<storage, read_write> bc_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> bc_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> bc_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> cond_strides: array<u32>;
@group(0) @binding(5) var<storage, read_write> x_strides: array<u32>;
@group(0) @binding(6) var<storage, read_write> y_strides: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_shape: array<u32>;
@group(0) @binding(8) var<uniform> bc_params: WhereBroadcastParams;

fn compute_out_stride(d: u32, ndim: u32) -> u32 {
    var stride: u32 = 1u;
    for (var i: u32 = d + 1u; i < ndim; i = i + 1u) {
        stride = stride * out_shape[i];
    }
    return stride;
}

@compute @workgroup_size(256)
fn where_broadcast_cond_f32_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bc_params.numel) {
        return;
    }

    var remaining = idx;
    var cond_offset: u32 = 0u;
    var x_offset: u32 = 0u;
    var y_offset: u32 = 0u;

    for (var d: u32 = 0u; d < bc_params.ndim; d = d + 1u) {
        let s = compute_out_stride(d, bc_params.ndim);
        let coord = remaining / s;
        remaining = remaining % s;
        cond_offset = cond_offset + coord * cond_strides[d];
        x_offset = x_offset + coord * x_strides[d];
        y_offset = y_offset + coord * y_strides[d];
    }

    let cond_val = bc_cond[cond_offset] != 0.0;
    bc_out[idx] = select(bc_y[y_offset], bc_x[x_offset], cond_val);
}
