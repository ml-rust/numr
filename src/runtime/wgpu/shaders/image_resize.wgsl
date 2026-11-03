// Separable pass kernel of resize (ImageOps::resize), F32.
//
// The launcher in image_resize.rs prepends the bindings, `load_src`,
// `store_dst` and `thread_index`. One variant binds the source and the
// destination as two buffers. The other binds one read-write buffer for both,
// for disjoint views of one storage.
//
// One invocation owns one destination element. Its linear index decodes into
// canonical (n, c, h, w): W fastest for NCHW order, C fastest for NHWC order
// (`channels_last`). The pass interpolates along `axis` (2 = H, 3 = W) and
// keeps the other coordinates. Strides are two's complement, and u32
// arithmetic wraps, so a negative stride adds correctly.
//
// Each output computes its own window and weights from its index. The rules
// are those of runtime/cpu/kernels/image_resize_taps.rs. `scale` and
// `invscale` come correctly rounded from the host, because WGSL division is
// not. WGSL lets an implementation fuse or split multiply-adds, so results
// match the CPU within rounding, not bit for bit.
//
// Modes (ResizeMode::kernel_mode):
//   0 copy, 1 linear, 2 cubic (a = -0.75), 3 AA linear, 4 AA cubic (a = -0.5)

struct ResizeParams {
    total: u32,
    dim_c: u32,
    dim_h: u32,
    dim_w: u32,
    channels_last: u32,
    axis: u32,
    mode: u32,
    in_size: u32,
    out_size: u32,
    src_base: u32,
    dst_base: u32,
    scale: f32,
    invscale: f32,
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
    pad2: u32,
}

// Canonical (n, c, h, w) destination coordinates of linear index `i`.
fn dst_coords(i: u32) -> vec4<u32> {
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

fn load_tap(base: u32, tap_stride: u32, k: i32) -> f32 {
    return load_src(base + u32(k) * tap_stride);
}

// std::max(a, b) and std::min(a, b).
fn cmax(a: f32, b: f32) -> f32 {
    return select(a, b, a < b);
}

fn cmin(a: f32, b: f32) -> f32 {
    return select(a, b, b < a);
}

// cubic_convolution1(x, -0.75) and cubic_convolution2(x, -0.75).
fn cubic1(x: f32) -> f32 {
    let t = 1.25 * x - 2.25;
    return t * x * x + 1.0;
}

fn cubic2(x: f32) -> f32 {
    let u = -0.75 * x + 3.75;
    let v = u * x - 6.0;
    return v * x + 3.0;
}

// Antialias filters: triangle (mode 3) or Keys cubic with a = -0.5 (mode 4).
fn aa_filter(arg: f32, mode: u32) -> f32 {
    let x = abs(arg);
    if (mode == 3u) {
        return select(0.0, 1.0 - x, x < 1.0);
    }
    if (x < 1.0) {
        let t = fma(1.5, x, -2.5);
        return fma(t * x, x, 1.0);
    }
    if (x < 2.0) {
        let u = fma(-0.5, x, 2.5);
        let v = fma(u, x, -4.0);
        return fma(v, x, 2.0);
    }
    return 0.0;
}

// trunc(f64(v) + 0.5) for v >= 0, exact in f32: floor(v) plus one when the
// fraction, which v - floor(v) gives exactly, is at least 0.5. A negative v
// gives at most 0, which the caller clamps to 0 as the C++ does.
fn aa_bound(v: f32) -> i32 {
    let fl = floor(v);
    return i32(fl) + select(0, 1, v - fl >= 0.5);
}

// f32((f64(d) + 0.5) * f64(invscale)): one rounding of the exact value when
// the implementation fuses `fma`.
fn aa_arg(d: f32, invscale: f32) -> f32 {
    return fma(d, invscale, 0.5 * invscale);
}

fn resize_value(base: u32, tap_stride: u32, o: i32) -> f32 {
    let mode = params.mode;
    let in_size = i32(params.in_size);
    let last = in_size - 1;
    if (mode == 0u) {
        return load_tap(base, tap_stride, o);
    }
    let scale = params.scale;
    if (mode == 1u) {
        var i0 = o;
        var i1 = o;
        var w0 = 1.0;
        var w1 = 0.0;
        if (params.in_size != params.out_size) {
            var s = scale * (f32(o) + 0.5) - 0.5;
            s = select(s, 0.0, s < 0.0);
            i0 = min(i32(floor(s)), last);
            w1 = cmin(cmax(s - f32(i0), 0.0), 1.0);
            i1 = i0 + select(0, 1, i0 < last);
            w0 = 1.0 - w1;
        }
        return fma(load_tap(base, tap_stride, i0), w0, load_tap(base, tap_stride, i1) * w1);
    }
    if (mode == 2u) {
        let s = fma(scale, f32(o) + 0.5, -0.5);
        let i = min(i32(floor(s)), last);
        let t = cmin(cmax(s - f32(i), 0.0), 1.0);
        let t2 = 1.0 - t;
        let k0 = clamp(i - 1, 0, last);
        let k1 = clamp(i, 0, last);
        let k2 = clamp(i + 1, 0, last);
        let k3 = clamp(i + 2, 0, last);
        var acc = fma(load_tap(base, tap_stride, k0), cubic2(t + 1.0),
            load_tap(base, tap_stride, k1) * cubic1(t));
        acc = fma(load_tap(base, tap_stride, k2), cubic1(t2), acc);
        return fma(load_tap(base, tap_stride, k3), cubic2(t2 + 1.0), acc);
    }

    let wide = scale >= 1.0;
    let half_taps = select(2.0, 1.0, mode == 3u);
    let support = select(half_taps, half_taps * scale, wide);
    let max_taps = i32(ceil(support)) * 2 + 1;
    let invscale = params.invscale;
    let center = scale * (f32(o) + 0.5);
    let xmin = max(aa_bound(center - support), 0);
    let xmax = min(aa_bound(center + support), in_size);
    let xsize = clamp(xmax - xmin, 0, max_taps);

    var total = 0.0;
    for (var j = 0; j < xsize; j = j + 1) {
        total = total + aa_filter(aa_arg(f32(xmin + j) - center, invscale), mode);
    }
    let normalise = total != 0.0;
    let blocked = select(0, 4 * ((xsize - 1) / 4), xsize > 0);
    var acc = 0.0;
    for (var j = 0; j < xsize; j = j + 1) {
        var w = aa_filter(aa_arg(f32(xmin + j) - center, invscale), mode);
        if (normalise) {
            w = w / total;
        }
        let x = load_tap(base, tap_stride, xmin + j);
        if (j == 0) {
            acc = x * w;
        } else if (j <= blocked) {
            acc = acc + x * w;
        } else {
            acc = fma(x, w, acc);
        }
    }
    return acc;
}

@compute @workgroup_size(256)
fn resize_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let p = dst_coords(t);
    var o: u32;
    var base: u32;
    var tap_stride: u32;
    if (params.axis == 2u) {
        o = p.z;
        base = params.src_base + p.x * params.ss0 + p.y * params.ss1 + p.w * params.ss3;
        tap_stride = params.ss2;
    } else {
        o = p.w;
        base = params.src_base + p.x * params.ss0 + p.y * params.ss1 + p.z * params.ss2;
        tap_stride = params.ss3;
    }
    let v = resize_value(base, tap_stride, i32(o));
    let d = params.dst_base + p.x * params.ds0 + p.y * params.ds1 + p.z * params.ds2
        + p.w * params.ds3;
    store_dst(d, v);
}
