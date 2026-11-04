// Tile kernel of conv2d_fused for NHWC input and OHWI weight, with C_in a
// multiple of 4.
//
// The channels of one pixel and the input-channel weights of one tap touch in
// memory, so both bind as vec4 arrays. Each vec4 of input channels loads four
// input vectors and four weight vectors for sixty-four multiply-adds.

@group(0) @binding(0) var<storage, read> conv_input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> conv_weight: array<vec4<f32>>;

// One input vector against the same four input channels of four output channels.
fn dot_block(p: vec4<f32>, w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>, w3: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(dot(p, w0), dot(p, w1), dot(p, w2), dot(p, w3));
}

@compute @workgroup_size(4, 8, 1)
fn conv2d_tile_vec4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = block_at(wid, lid);
    if (b.x >= params.output_w || b.y >= params.output_h) {
        return;
    }

    let zero = vec4<f32>(0.0);
    let last_column = vec4<i32>(i32(params.width) - 1);
    let vectors = params.c_in / BLOCK;
    let taps = params.kernel_h * params.kernel_w;
    let offsets = channel_offsets(b.channels, taps * vectors);
    var sum0 = zero;
    var sum1 = zero;
    var sum2 = zero;
    var sum3 = zero;

    for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
        let row = tap_row(b.y, ky);
        if (row < 0 || row >= i32(params.height)) {
            continue;
        }
        let line = (b.batch * params.height + u32(row)) * params.width;

        for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let columns = tap_columns(b.x, kx);
            let inside = (columns >= vec4<i32>(0)) & (columns <= last_column);
            if (!any(inside)) {
                continue;
            }
            let all_inside = all(inside);
            let at = (vec4<u32>(line) + vec4<u32>(clamp(columns, vec4<i32>(0), last_column)))
                * vectors;
            let tap = ky * params.kernel_w + kx;
            let w_at = (b.channel * taps + tap) * vectors;

            for (var i: u32 = 0u; i < vectors; i = i + 1u) {
                let w0 = conv_weight[w_at + i];
                let w1 = conv_weight[w_at + offsets.x + i];
                let w2 = conv_weight[w_at + offsets.y + i];
                let w3 = conv_weight[w_at + offsets.z + i];
                var p0 = conv_input[at.x + i];
                var p1 = conv_input[at.y + i];
                var p2 = conv_input[at.z + i];
                var p3 = conv_input[at.w + i];
                if (!all_inside) {
                    p0 = select(zero, p0, inside.x);
                    p1 = select(zero, p1, inside.y);
                    p2 = select(zero, p2, inside.z);
                    p3 = select(zero, p3, inside.w);
                }
                sum0 = sum0 + dot_block(p0, w0, w1, w2, w3);
                sum1 = sum1 + dot_block(p1, w0, w1, w2, w3);
                sum2 = sum2 + dot_block(p2, w0, w1, w2, w3);
                sum3 = sum3 + dot_block(p3, w0, w1, w2, w3);
            }
        }
    }

    store_pixel(b, 0u, sum0);
    store_pixel(b, 1u, sum1);
    store_pixel(b, 2u, sum2);
    store_pixel(b, 3u, sum3);
}
