// Tile kernel of conv2d and conv2d_fused for any layout and channel count.
//
// The input and the weight are addressed through element strides, so the
// channels of one pixel need not touch in memory. Each input channel of a tap
// loads four input samples and four weights for sixteen multiply-adds.

@group(0) @binding(0) var<storage, read> conv_input: array<f32>;
@group(0) @binding(1) var<storage, read> conv_weight: array<f32>;

@compute @workgroup_size(4, 8, 1)
fn conv2d_tile_scalar(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = block_at(wid, lid);
    if (b.x >= params.output_w || b.y >= params.output_h) {
        return;
    }

    let zero = vec4<f32>(0.0);
    let last_column = vec4<i32>(i32(params.width) - 1);
    let offsets = channel_offsets(b.channels, params.wgt_o);
    var sum0 = zero;
    var sum1 = zero;
    var sum2 = zero;
    var sum3 = zero;

    for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
        let row = tap_row(b.y, ky);
        if (row < 0 || row >= i32(params.height)) {
            continue;
        }
        let line = b.batch * params.src_n + u32(row) * params.src_h;

        for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let columns = tap_columns(b.x, kx);
            let inside = (columns >= vec4<i32>(0)) & (columns <= last_column);
            if (!any(inside)) {
                continue;
            }
            let all_inside = all(inside);
            let at = vec4<u32>(line)
                + vec4<u32>(clamp(columns, vec4<i32>(0), last_column)) * params.src_w;
            let tap = ky * params.kernel_w + kx;
            let tap_weight = b.channel * params.wgt_o + tap * params.wgt_tap;

            for (var ic: u32 = 0u; ic < params.c_in; ic = ic + 1u) {
                let w_at = tap_weight + ic * params.wgt_i;
                let w = vec4<f32>(
                    conv_weight[w_at],
                    conv_weight[w_at + offsets.x],
                    conv_weight[w_at + offsets.y],
                    conv_weight[w_at + offsets.z],
                );
                let channel_at = ic * params.src_c;
                var p = vec4<f32>(
                    conv_input[at.x + channel_at],
                    conv_input[at.y + channel_at],
                    conv_input[at.z + channel_at],
                    conv_input[at.w + channel_at],
                );
                if (!all_inside) {
                    p = select(zero, p, inside);
                }
                sum0 = sum0 + p.x * w;
                sum1 = sum1 + p.y * w;
                sum2 = sum2 + p.z * w;
                sum3 = sum3 + p.w * w;
            }
        }
    }

    store_pixel(b, 0u, sum0);
    store_pixel(b, 1u, sum1);
    store_pixel(b, 2u, sum2);
    store_pixel(b, 3u, sum3);
}
