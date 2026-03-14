// Depthwise conv2d shader for f32
// Input layout: (N, C, H, W)
// Weight layout: (C, 1, K_h, K_w)
// Output layout: (N, C, H_out, W_out)

const WORKGROUP_SIZE: u32 = 256u;

struct DepthwiseConv2dParams {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> depthwise_input: array<f32>;
@group(0) @binding(1) var<storage, read> depthwise_weight: array<f32>;
@group(0) @binding(2) var<storage, read> depthwise_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> depthwise_output: array<f32>;
@group(0) @binding(4) var<uniform> depthwise_params: DepthwiseConv2dParams;

@compute @workgroup_size(256)
fn depthwise_conv2d_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = depthwise_params.batch * depthwise_params.channels * depthwise_params.output_h * depthwise_params.output_w;
    if (idx >= total) { return; }

    let ox = idx % depthwise_params.output_w;
    let oy = (idx / depthwise_params.output_w) % depthwise_params.output_h;
    let c = (idx / (depthwise_params.output_w * depthwise_params.output_h)) % depthwise_params.channels;
    let b = idx / (depthwise_params.channels * depthwise_params.output_h * depthwise_params.output_w);

    var sum: f32 = 0.0;

    for (var ky: u32 = 0u; ky < depthwise_params.kernel_h; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < depthwise_params.kernel_w; kx = kx + 1u) {
            let iy_signed = i32(oy * depthwise_params.stride_h + ky * depthwise_params.dilation_h) - i32(depthwise_params.pad_h);
            let ix_signed = i32(ox * depthwise_params.stride_w + kx * depthwise_params.dilation_w) - i32(depthwise_params.pad_w);

            if (iy_signed >= 0 && u32(iy_signed) < depthwise_params.height && ix_signed >= 0 && u32(ix_signed) < depthwise_params.width) {
                let iy = u32(iy_signed);
                let ix = u32(ix_signed);
                let input_idx = b * depthwise_params.channels * depthwise_params.height * depthwise_params.width
                    + c * depthwise_params.height * depthwise_params.width
                    + iy * depthwise_params.width
                    + ix;
                let weight_idx = c * depthwise_params.kernel_h * depthwise_params.kernel_w + ky * depthwise_params.kernel_w + kx;
                sum = sum + depthwise_input[input_idx] * depthwise_weight[weight_idx];
            }
        }
    }

    if (depthwise_params.has_bias != 0u) {
        sum = sum + depthwise_bias[c];
    }

    depthwise_output[idx] = sum;
}
