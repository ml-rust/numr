// 4:2:0 YCbCr kernels of ColorOps on packed U32 plane words and F32 RGB.
//
// The launcher in color.rs prepends the bindings, `ld_<name>` and
// `st_<name>` for each operand (rgb, y, u, v), `thread_index` and
// color_float.wgsl. Every operand binds as `array<u32>`, and each distinct
// buffer binds once. For NV12 and P010, u and v are both the interleaved
// chroma plane.
//
// A plane row starts at word `base + row * rs`, and holds its samples back
// to back from byte 0 of that word, low byte first: 4 U8 samples or 2 U16
// samples per word. Strides are two's complement, and u32 arithmetic wraps,
// so a negative stride adds correctly. A P010 sample holds its value in the
// high 10 bits.
//
// Every value follows the f32 operation order of the CPU kernels in
// runtime/cpu/kernels/color.rs through the integer f32 routines, so the
// results are the CPU bits.
//
// yuv_to_rgb: one invocation per pixel. It reads plane words and writes
// three F32 words, so no two invocations write one word.
//
// rgb_to_yuv: one invocation per chroma row i and column group g. A group is
// the luma columns whose samples fill one chroma word: 4 for NV12 (one word
// U, V, U, V), 8 for I420 (one U word and one V word of 4 samples each) and
// 2 for P010 (one word U, V). The invocation owns chroma word g of row i of
// each chroma plane, and the luma words of rows 2i and 2i + 1 over its
// columns. Group boundaries fall on word boundaries in every plane, so every
// written word has one owner and no atomics are needed. A row's last group
// can cover fewer columns; it reads its partial last word and keeps the
// bytes past the row.

struct YuvParams {
    total: u32,
    height: u32,
    width: u32,
    groups: u32,
    format: u32,
    center: u32,
    y_base: u32,
    y_rs: u32,
    u_base: u32,
    u_rs: u32,
    v_base: u32,
    v_rs: u32,
    rgb_base: u32,
    rgb_c: u32,
    rgb_h: u32,
    rgb_w: u32,
    y_off: i32,
    c_off: i32,
    c_div16: u32,
    cr_r: u32,
    cb_b: u32,
    g_cb: u32,
    g_cr: u32,
    kr: u32,
    kg: u32,
    kb: u32,
    y_mul: u32,
    c_mul: u32,
    y_lo: u32,
    y_hi: u32,
    c_lo: u32,
    c_hi: u32,
    y_off_f: u32,
    c_off_f: u32,
    pad0: u32,
    pad1: u32,
}

const FORMAT_I420: u32 = 1u;
const FORMAT_P010: u32 = 2u;

fn wide() -> bool {
    return params.format == FORMAT_P010;
}

// Bits of one stored sample.
fn sample_bits() -> u32 {
    return select(8u, 16u, wide());
}

// Samples per word.
fn samples_per_word() -> u32 {
    return select(4u, 2u, wide());
}

// Word of sample s in row `row` of the plane at base with row stride rs.
fn word_index(base: u32, rs: u32, row: u32, s: u32) -> u32 {
    return base + row * rs + s / samples_per_word();
}

// Value of sample s in word w.
fn sample_value(w: u32, s: u32) -> i32 {
    if (wide()) {
        return i32(((w >> ((s & 1u) * 16u)) & 0xffffu) >> 6u);
    }
    return i32((w >> ((s & 3u) * 8u)) & 0xffu);
}

fn read_y(row: u32, col: u32) -> i32 {
    return sample_value(ld_y(word_index(params.y_base, params.y_rs, row, col)), col);
}

// Sample index of chroma column j: interleaved planes hold U at 2j, V at 2j + 1.
fn chroma_sample(j: u32, odd: u32) -> u32 {
    return select(2u * j + odd, j, params.format == FORMAT_I420);
}

fn read_cb(i: u32, j: u32) -> i32 {
    let s = chroma_sample(j, 0u);
    return sample_value(ld_u(word_index(params.u_base, params.u_rs, i, s)), s);
}

fn read_cr(i: u32, j: u32) -> i32 {
    let s = chroma_sample(j, 1u);
    return sample_value(ld_v(word_index(params.v_base, params.v_rs, i, s)), s);
}

// Chroma indices (x, y) and weights in quarters (z, w) for luma index `pos`
// along an axis of `len` chroma samples.
fn up_taps(pos: u32, len: u32, center: bool) -> vec4<u32> {
    let k = pos >> 1u;
    let next = min(k + 1u, len - 1u);
    let even = (pos & 1u) == 0u;
    if (center) {
        if (even) {
            return vec4<u32>(select(k - 1u, 0u, k == 0u), k, 1u, 3u);
        }
        return vec4<u32>(k, next, 3u, 1u);
    }
    if (even) {
        return vec4<u32>(k, k, 4u, 0u);
    }
    return vec4<u32>(k, next, 2u, 2u);
}

fn rgb_index(row: u32, col: u32) -> u32 {
    return params.rgb_base + row * params.rgb_h + col * params.rgb_w;
}

@compute @workgroup_size(256)
fn yuv_to_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let row = t / params.width;
    let col = t % params.width;
    let tv = up_taps(row, params.height / 2u, true);
    let th = up_taps(col, params.width / 2u, params.center != 0u);
    let rows = vec2<u32>(tv.x, tv.y);
    let cols = vec2<u32>(th.x, th.y);
    let wv = vec2<i32>(i32(tv.z), i32(tv.w));
    let wh = vec2<i32>(i32(th.z), i32(th.w));
    var cb16 = 0;
    var cr16 = 0;
    for (var p = 0u; p < 2u; p = p + 1u) {
        for (var q = 0u; q < 2u; q = q + 1u) {
            let wt = wv[p] * wh[q];
            cb16 = cb16 + wt * read_cb(rows[p], cols[q]);
            cr16 = cr16 + wt * read_cr(rows[p], cols[q]);
        }
    }
    let yn = f_div(f_from_i32(read_y(row, col) - params.y_off), params.y_mul);
    let cb = f_div(f_from_i32(cb16 - 16 * params.c_off), params.c_div16);
    let cr = f_div(f_from_i32(cr16 - 16 * params.c_off), params.c_div16);
    let r = f_add(yn, f_mul(params.cr_r, cr));
    let b = f_add(yn, f_mul(params.cb_b, cb));
    let g = f_add(f_add(yn, f_mul(params.g_cb, cb)), f_mul(params.g_cr, cr));
    let o = rgb_index(row, col);
    st_rgb(o, f_clamp01(r));
    st_rgb(o + params.rgb_c, f_clamp01(g));
    st_rgb(o + 2u * params.rgb_c, f_clamp01(b));
}

// (Y', Cb', Cr') of pixel (row, col), after clamping R, G and B to [0, 1].
fn encode_pixel(row: u32, col: u32) -> vec3<u32> {
    let o = rgb_index(row, col);
    let r = f_clamp01(ld_rgb(o));
    let g = f_clamp01(ld_rgb(o + params.rgb_c));
    let b = f_clamp01(ld_rgb(o + 2u * params.rgb_c));
    let y = f_add(f_add(f_mul(params.kr, r), f_mul(params.kg, g)), f_mul(params.kb, b));
    return vec3<u32>(y, f_div(f_sub(b, y), params.cb_b), f_div(f_sub(r, y), params.cr_r));
}

// Downsampled chroma of columns 2j - 1, 2j, 2j + 1 (t*: row 2i, b*: 2i + 1).
fn downsample(t0: u32, t1: u32, t2: u32, b0: u32, b1: u32, b2: u32) -> u32 {
    if (params.center != 0u) {
        return f_mul(f_add(f_add(t1, t2), f_add(b1, b2)), F_QUARTER);
    }
    let ht = f_add(f_add(t0, f_mul(F_TWO, t1)), t2);
    let hb = f_add(f_add(b0, f_mul(F_TWO, b1)), b2);
    return f_mul(f_add(ht, hb), F_EIGHTH);
}

fn chroma_code(v: u32) -> u32 {
    return f_quantize(v, params.c_mul, params.c_off_f, params.c_lo, params.c_hi);
}

// Stored sample of an integral code.
fn stored(code: u32) -> u32 {
    return select(code, code << 6u, wide());
}

fn write_y(idx: u32, val: u32, mask: u32) {
    var w = val;
    if (mask != 0xffffffffu) {
        w = (ld_y(idx) & ~mask) | val;
    }
    st_y(idx, w);
}

fn write_u(idx: u32, val: u32, mask: u32) {
    var w = val;
    if (mask != 0xffffffffu) {
        w = (ld_u(idx) & ~mask) | val;
    }
    st_u(idx, w);
}

fn write_v(idx: u32, val: u32, mask: u32) {
    var w = val;
    if (mask != 0xffffffffu) {
        w = (ld_v(idx) & ~mask) | val;
    }
    st_v(idx, w);
}

@compute @workgroup_size(256)
fn rgb_to_yuv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let i = t / params.groups;
    let g = t % params.groups;
    let i420 = params.format == FORMAT_I420;
    let span = select(select(4u, 2u, wide()), 8u, i420);
    let c0 = g * span;
    let n = min(span, params.width - c0);

    // Per row: luma codes of the n columns, and Cb', Cr' at local index
    // k + 1 for column c0 + k, with column c0 - 1 at index 0.
    var luma: array<u32, 16>;
    var cbv: array<u32, 18>;
    var crv: array<u32, 18>;
    for (var rr = 0u; rr < 2u; rr = rr + 1u) {
        let row = 2u * i + rr;
        for (var k = 0u; k < n; k = k + 1u) {
            let e = encode_pixel(row, c0 + k);
            luma[rr * 8u + k] =
                f_quantize(e.x, params.y_mul, params.y_off_f, params.y_lo, params.y_hi);
            cbv[rr * 9u + k + 1u] = e.y;
            crv[rr * 9u + k + 1u] = e.z;
        }
        if (params.center == 0u) {
            if (c0 == 0u) {
                cbv[rr * 9u] = cbv[rr * 9u + 1u];
                crv[rr * 9u] = crv[rr * 9u + 1u];
            } else {
                let e = encode_pixel(row, c0 - 1u);
                cbv[rr * 9u] = e.y;
                crv[rr * 9u] = e.z;
            }
        }
    }

    let spw = samples_per_word();
    let bits = sample_bits();
    let smask = select(0xffu, 0xffffu, wide());
    for (var rr = 0u; rr < 2u; rr = rr + 1u) {
        let row = 2u * i + rr;
        for (var first = 0u; first < n; first = first + spw) {
            var val = 0u;
            var mask = 0u;
            for (var s = 0u; s < min(spw, n - first); s = s + 1u) {
                val = val | (stored(luma[rr * 8u + first + s]) << (s * bits));
                mask = mask | (smask << (s * bits));
            }
            write_y(word_index(params.y_base, params.y_rs, row, c0 + first), val, mask);
        }
    }

    let nc = n / 2u;
    var cb_word = 0u;
    var cr_word = 0u;
    var cmask = 0u;
    for (var m = 0u; m < nc; m = m + 1u) {
        let a = 2u * m;
        let cb_avg = downsample(
            cbv[a], cbv[a + 1u], cbv[a + 2u], cbv[9u + a], cbv[10u + a], cbv[11u + a],
        );
        let cr_avg = downsample(
            crv[a], crv[a + 1u], crv[a + 2u], crv[9u + a], crv[10u + a], crv[11u + a],
        );
        let cb = stored(chroma_code(cb_avg));
        let cr = stored(chroma_code(cr_avg));
        if (i420) {
            cb_word = cb_word | (cb << (8u * m));
            cr_word = cr_word | (cr << (8u * m));
            cmask = cmask | (0xffu << (8u * m));
        } else {
            cb_word = cb_word | (cb << (a * bits)) | (cr << ((a + 1u) * bits));
            cmask = cmask | (smask << (a * bits)) | (smask << ((a + 1u) * bits));
        }
    }
    write_u(params.u_base + i * params.u_rs + g, cb_word, cmask);
    if (i420) {
        write_v(params.v_base + i * params.v_rs + g, cr_word, cmask);
    }
}
