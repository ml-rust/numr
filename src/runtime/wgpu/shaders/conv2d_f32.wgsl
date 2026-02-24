// Conv2d shader for f32
// Input layout: (N, C_in, H, W)
// Weight layout: (C_out, C_in/groups, K_h, K_w)
// Output layout: (N, C_out, H_out, W_out)

const WORKGROUP_SIZE: u32 = 256u;

struct Conv2dParams {
    batch: u32,
    c_in: u32,
    height: u32,
    width: u32,
    c_out: u32,
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
    groups: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> conv2d_input: array<f32>;
@group(0) @binding(1) var<storage, read> conv2d_weight: array<f32>;
@group(0) @binding(2) var<storage, read> conv2d_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv2d_output: array<f32>;
@group(0) @binding(4) var<uniform> conv2d_params: Conv2dParams;

@compute @workgroup_size(256)
fn conv2d_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = conv2d_params.batch * conv2d_params.c_out * conv2d_params.output_h * conv2d_params.output_w;
    if (idx >= total) { return; }

    let ox = idx % conv2d_params.output_w;
    let oy = (idx / conv2d_params.output_w) % conv2d_params.output_h;
    let oc = (idx / (conv2d_params.output_w * conv2d_params.output_h)) % conv2d_params.c_out;
    let b = idx / (conv2d_params.c_out * conv2d_params.output_h * conv2d_params.output_w);

    let c_in_per_group = conv2d_params.c_in / conv2d_params.groups;
    let c_out_per_group = conv2d_params.c_out / conv2d_params.groups;
    let g = oc / c_out_per_group;
    let c_in_start = g * c_in_per_group;

    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < c_in_per_group; ic = ic + 1u) {
        let c_in_idx = c_in_start + ic;

        for (var ky: u32 = 0u; ky < conv2d_params.kernel_h; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < conv2d_params.kernel_w; kx = kx + 1u) {
                let iy_signed = i32(oy * conv2d_params.stride_h + ky * conv2d_params.dilation_h) - i32(conv2d_params.pad_h);
                let ix_signed = i32(ox * conv2d_params.stride_w + kx * conv2d_params.dilation_w) - i32(conv2d_params.pad_w);

                if (iy_signed >= 0 && u32(iy_signed) < conv2d_params.height && ix_signed >= 0 && u32(ix_signed) < conv2d_params.width) {
                    let iy = u32(iy_signed);
                    let ix = u32(ix_signed);
                    let input_idx = b * conv2d_params.c_in * conv2d_params.height * conv2d_params.width
                        + c_in_idx * conv2d_params.height * conv2d_params.width
                        + iy * conv2d_params.width
                        + ix;
                    let weight_idx = oc * c_in_per_group * conv2d_params.kernel_h * conv2d_params.kernel_w
                        + ic * conv2d_params.kernel_h * conv2d_params.kernel_w
                        + ky * conv2d_params.kernel_w
                        + kx;
                    sum = sum + conv2d_input[input_idx] * conv2d_weight[weight_idx];
                }
            }
        }
    }

    if (conv2d_params.has_bias != 0u) {
        sum = sum + conv2d_bias[oc];
    }

    conv2d_output[idx] = sum;
}
