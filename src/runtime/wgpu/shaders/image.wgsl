// Gather kernels for pixel_shuffle, pixel_unshuffle and upsample_nearest2d,
// 32-bit words (F32, I32, U32).
//
// The launcher in image.rs prepends the bindings, `load_src`, `store_dst`
// and `thread_index`. One variant binds the source and the destination as
// two buffers. The other binds one read-write buffer for both, for disjoint
// views of one storage.
//
// One invocation owns one output element. Its linear index decodes into
// canonical (n, c, h, w) output coordinates: W fastest for NCHW order, C
// fastest for NHWC order (`channels_last`). The op maps them to canonical
// source coordinates. Both sides are addressed through canonical
// [N, C, H, W] strides in elements. Strides are two's complement, and u32
// arithmetic wraps, so a negative stride adds correctly.
//
// Nearest source index per axis, matching PyTorch's CPU `nearest_idx`:
//   mode 0: src = dst
//   mode 1: src = dst >> 1
//   mode 2: src = min(floor(f32(dst) * scale), in - 1)

struct ImageParams {
    total: u32,
    dim_c: u32,
    dim_h: u32,
    dim_w: u32,
    channels_last: u32,
    factor: u32,
    in_h: u32,
    in_w: u32,
    mode_h: u32,
    mode_w: u32,
    scale_h: f32,
    scale_w: f32,
    src_base: u32,
    dst_base: u32,
    ss0: u32,
    ss1: u32,
    ss2: u32,
    ss3: u32,
    ds0: u32,
    ds1: u32,
    ds2: u32,
    ds3: u32,
    pad0: u32,
    pad1: u32,
}

// Canonical (n, c, h, w) output coordinates of linear index `i`.
fn output_coords(i: u32) -> vec4<u32> {
    if (params.channels_last != 0u) {
        let c = i % params.dim_c;
        var rest = i / params.dim_c;
        let w = rest % params.dim_w;
        rest = rest / params.dim_w;
        let h = rest % params.dim_h;
        return vec4<u32>(rest / params.dim_h, c, h, w);
    }
    let w = i % params.dim_w;
    var rest = i / params.dim_w;
    let h = rest % params.dim_h;
    rest = rest / params.dim_h;
    let c = rest % params.dim_c;
    return vec4<u32>(rest / params.dim_c, c, h, w);
}

fn src_index(p: vec4<u32>) -> u32 {
    return params.src_base + p.x * params.ss0 + p.y * params.ss1 + p.z * params.ss2
        + p.w * params.ss3;
}

fn dst_index(p: vec4<u32>) -> u32 {
    return params.dst_base + p.x * params.ds0 + p.y * params.ds1 + p.z * params.ds2
        + p.w * params.ds3;
}

fn nearest_src(dst: u32, in_size: u32, mode: u32, scale: f32) -> u32 {
    if (mode == 0u) {
        return dst;
    }
    if (mode == 1u) {
        return dst >> 1u;
    }
    let src = u32(floor(f32(dst) * scale));
    return min(src, in_size - 1u);
}

@compute @workgroup_size(256)
fn pixel_shuffle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let o = output_coords(t);
    let r = params.factor;
    let s = vec4<u32>(o.x, o.y * r * r + (o.z % r) * r + o.w % r, o.z / r, o.w / r);
    store_dst(dst_index(o), load_src(src_index(s)));
}

@compute @workgroup_size(256)
fn pixel_unshuffle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let o = output_coords(t);
    let r = params.factor;
    let rem = o.y % (r * r);
    let s = vec4<u32>(o.x, o.y / (r * r), o.z * r + rem / r, o.w * r + rem % r);
    store_dst(dst_index(o), load_src(src_index(s)));
}

@compute @workgroup_size(256)
fn upsample_nearest2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let o = output_coords(t);
    let sh = nearest_src(o.z, params.in_h, params.mode_h, params.scale_h);
    let sw = nearest_src(o.w, params.in_w, params.mode_w, params.scale_w);
    store_dst(dst_index(o), load_src(src_index(vec4<u32>(o.x, o.y, sh, sw))));
}
