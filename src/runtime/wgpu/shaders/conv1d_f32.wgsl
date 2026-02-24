// Conv1d shader for f32
// Input layout: (N, C_in, L)
// Weight layout: (C_out, C_in/groups, K)
// Output layout: (N, C_out, L_out)

const WORKGROUP_SIZE: u32 = 256u;

struct Conv1dParams {
    batch: u32,
    c_in: u32,
    length: u32,
    c_out: u32,
    kernel_size: u32,
    output_length: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> conv1d_input: array<f32>;
@group(0) @binding(1) var<storage, read> conv1d_weight: array<f32>;
@group(0) @binding(2) var<storage, read> conv1d_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv1d_output: array<f32>;
@group(0) @binding(4) var<uniform> conv1d_params: Conv1dParams;

@compute @workgroup_size(256)
fn conv1d_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = conv1d_params.batch * conv1d_params.c_out * conv1d_params.output_length;
    if (idx >= total) { return; }

    let ox = idx % conv1d_params.output_length;
    let oc = (idx / conv1d_params.output_length) % conv1d_params.c_out;
    let b = idx / (conv1d_params.c_out * conv1d_params.output_length);

    let c_in_per_group = conv1d_params.c_in / conv1d_params.groups;
    let c_out_per_group = conv1d_params.c_out / conv1d_params.groups;
    let g = oc / c_out_per_group;
    let c_in_start = g * c_in_per_group;

    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < c_in_per_group; ic = ic + 1u) {
        let c_in_idx = c_in_start + ic;

        for (var kx: u32 = 0u; kx < conv1d_params.kernel_size; kx = kx + 1u) {
            let ix_signed = i32(ox * conv1d_params.stride + kx * conv1d_params.dilation) - i32(conv1d_params.padding);

            if (ix_signed >= 0 && u32(ix_signed) < conv1d_params.length) {
                let ix = u32(ix_signed);
                let input_idx = b * conv1d_params.c_in * conv1d_params.length + c_in_idx * conv1d_params.length + ix;
                let weight_idx = oc * c_in_per_group * conv1d_params.kernel_size + ic * conv1d_params.kernel_size + kx;
                sum = sum + conv1d_input[input_idx] * conv1d_weight[weight_idx];
            }
        }
    }

    if (conv1d_params.has_bias != 0u) {
        sum = sum + conv1d_bias[oc];
    }

    conv1d_output[idx] = sum;
}
