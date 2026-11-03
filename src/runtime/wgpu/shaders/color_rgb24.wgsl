// RGB24 pack and unpack kernels of ColorOps between packed U32 words and
// F32 RGB.
//
// The launcher in color.rs prepends the bindings, `ld_<name>` and
// `st_<name>` for the operands src and dst, `thread_index` and
// color_float.wgsl. Every operand binds as `array<u32>`, and each distinct
// buffer binds once.
//
// Row `row` of the packed frame starts at word `p_base + row * p_rs` and
// holds bytes R, G, B of each pixel back to back from byte 0 of that word,
// low byte first. The float frame is addressed through [C, H, W] strides in
// words. Strides are two's complement, and u32 arithmetic wraps, so a
// negative stride adds correctly.
//
// rgb24_to_float: one invocation per float element. It reads words and
// writes one F32 word.
//
// float_to_rgb24: one invocation per packed word, so every written word has
// one owner and no atomics are needed. The last word of a row can hold bytes
// past `3 * width`. With `fresh` set they become 0; otherwise the invocation
// reads the word and keeps them.

struct Rgb24Params {
    total: u32,
    width: u32,
    row_words: u32,
    fresh: u32,
    p_base: u32,
    p_rs: u32,
    f_base: u32,
    f_c: u32,
    f_h: u32,
    f_w: u32,
    pad0: u32,
    pad1: u32,
}

fn float_index(row: u32, col: u32, c: u32) -> u32 {
    return params.f_base + c * params.f_c + row * params.f_h + col * params.f_w;
}

@compute @workgroup_size(256)
fn rgb24_to_float(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let c = t % 3u;
    let px = t / 3u;
    let col = px % params.width;
    let row = px / params.width;
    let byte = 3u * col + c;
    let w = ld_src(params.p_base + row * params.p_rs + byte / 4u);
    let v = (w >> ((byte & 3u) * 8u)) & 0xffu;
    st_dst(float_index(row, col, c), f_div(f_from_i32(i32(v)), F_255));
}

@compute @workgroup_size(256)
fn float_to_rgb24(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let row = t / params.row_words;
    let k = t % params.row_words;
    let row_bytes = 3u * params.width;
    var val = 0u;
    var mask = 0u;
    for (var s = 0u; s < 4u; s = s + 1u) {
        let byte = 4u * k + s;
        if (byte < row_bytes) {
            let x = f_clamp01(ld_src(float_index(row, byte / 3u, byte % 3u)));
            val = val | (f_round_u(f_mul(x, F_255)) << (8u * s));
            mask = mask | (0xffu << (8u * s));
        }
    }
    let idx = params.p_base + row * params.p_rs + k;
    if (mask != 0xffffffffu && params.fresh == 0u) {
        val = val | (ld_dst(idx) & ~mask);
    }
    st_dst(idx, val);
}
