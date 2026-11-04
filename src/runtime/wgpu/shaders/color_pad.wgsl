// Replicate padding of a packed luma plane on the bottom and right, between
// packed U32 words.
//
// The launcher in color.rs prepends the bindings, `ld_<name>` and `st_<name>`
// for the operands src and dst, and `thread_index`. Every operand binds as
// `array<u32>`.
//
// Source row `row` starts at word `src_base + row * src_rs` and holds samples
// back to back from byte 0 of that word, low byte first. The stride is two's
// complement, and u32 arithmetic wraps, so a negative stride adds correctly.
// Output words are tight: output word `t` is word `t % row_words` of row
// `t / row_words`.
//
// One invocation per output word, so every written word has one owner and no
// atomics are needed. Output sample [r, c] is source sample
// [min(r, src_rows - 1), min(c, src_width - 1)]. Bytes past `columns` in the
// last word of a row are 0.

struct PadLumaParams {
    total: u32,
    row_words: u32,
    src_base: u32,
    src_rs: u32,
    src_rows: u32,
    src_width: u32,
    columns: u32,
    sample_bytes: u32,
}

@compute @workgroup_size(256)
fn pad_packed_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params.total) {
        return;
    }
    let row = t / params.row_words;
    let k = t % params.row_words;
    let src_row = min(row, params.src_rows - 1u);
    let row_base = params.src_base + src_row * params.src_rs;
    let per_word = 4u / params.sample_bytes;
    let sample_bits = 8u * params.sample_bytes;
    let value_mask = (1u << sample_bits) - 1u;
    var val = 0u;
    for (var s = 0u; s < per_word; s = s + 1u) {
        let col = k * per_word + s;
        if (col < params.columns) {
            let byte = min(col, params.src_width - 1u) * params.sample_bytes;
            let w = ld_src(row_base + byte / 4u);
            let v = (w >> ((byte & 3u) * 8u)) & value_mask;
            val = val | (v << (s * sample_bits));
        }
    }
    st_dst(t, val);
}
