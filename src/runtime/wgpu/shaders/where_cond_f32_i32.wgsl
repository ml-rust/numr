// where_cond: condition=f32, output=i32
// out[i] = cond[i] != 0.0 ? x[i] : y[i]

struct WhereParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> where_cond_arr: array<f32>;
@group(0) @binding(1) var<storage, read_write> where_x: array<i32>;
@group(0) @binding(2) var<storage, read_write> where_y: array<i32>;
@group(0) @binding(3) var<storage, read_write> where_out: array<i32>;
@group(0) @binding(4) var<uniform> where_params: WhereParams;

@compute @workgroup_size(256)
fn where_cond_f32_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < where_params.numel) {
        let cond_val = where_cond_arr[idx] != 0.0;
        where_out[idx] = select(where_y[idx], where_x[idx], cond_val);
    }
}
