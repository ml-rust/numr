// Shared part of the tile kernels of conv2d and conv2d_fused, F32.
//
// The launcher in conv2d_tile.rs prepends this text to one kernel file. The
// kernel file binds the input and the weight in its own element type.
//
// One workgroup owns one spatial tile and one block of four output channels.
// One invocation owns four adjacent output pixels of that tile, so it computes
// sixteen output elements in one pass over the taps. The dispatch grid is
// (channel blocks, tile columns, batch * tile rows).
//
// Everything after the K sum runs here: bias, activation, scaled residual.

struct Conv2dTileParams {
    c_in: u32,
    c_out: u32,
    height: u32,
    width: u32,
    output_h: u32,
    output_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    tile_rows: u32,
    // Input element strides per batch, channel, row and column.
    src_n: u32,
    src_c: u32,
    src_h: u32,
    src_w: u32,
    // Weight element strides per output channel, input channel and tap.
    wgt_o: u32,
    wgt_i: u32,
    wgt_tap: u32,
    // Output and residual element strides per batch, channel, row and column.
    dst_n: u32,
    dst_c: u32,
    dst_h: u32,
    dst_w: u32,
    has_bias: u32,
    // 0 none, 1 relu, 2 leaky relu, 3 prelu.
    activation: u32,
    prelu_len: u32,
    has_residual: u32,
    leaky_slope: f32,
    residual_scale: f32,
}

@group(0) @binding(2) var<storage, read> conv_bias: array<f32>;
@group(0) @binding(3) var<storage, read> conv_prelu: array<f32>;
@group(0) @binding(4) var<storage, read> conv_residual: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_output: array<f32>;
@group(0) @binding(6) var<uniform> params: Conv2dTileParams;

const TILE_COLUMNS: u32 = 16u;
const TILE_ROWS: u32 = 8u;
const BLOCK: u32 = 4u;

struct TileBlock {
    batch: u32,
    channel: u32,
    channels: u32,
    x: u32,
    y: u32,
}

// The output block of one invocation. Its x or y lies past the output when the
// tile crosses the edge, and the caller then returns.
fn block_at(wid: vec3<u32>, lid: vec3<u32>) -> TileBlock {
    let channel = wid.x * BLOCK;
    return TileBlock(
        wid.z / params.tile_rows,
        channel,
        min(BLOCK, params.c_out - channel),
        wid.y * TILE_COLUMNS + lid.x * BLOCK,
        (wid.z % params.tile_rows) * TILE_ROWS + lid.y,
    );
}

// Weight offsets of the block's second, third and fourth output channels. A
// slot past the block's channel count reads the first channel, so every index
// stays in range and no stored element uses that value.
fn channel_offsets(channels: u32, stride: u32) -> vec3<u32> {
    return vec3<u32>(
        select(0u, stride, channels > 1u),
        select(0u, 2u * stride, channels > 2u),
        select(0u, 3u * stride, channels > 3u),
    );
}

// Input row of tap row `ky` for output row `y`.
fn tap_row(y: u32, ky: u32) -> i32 {
    return i32(y * params.stride_h + ky * params.dilation_h) - i32(params.pad_h);
}

// Input columns of tap column `kx` for the four output columns from `x`.
fn tap_columns(x: u32, kx: u32) -> vec4<i32> {
    let first = i32(x * params.stride_w + kx * params.dilation_w) - i32(params.pad_w);
    return vec4<i32>(first) + vec4<i32>(0, 1, 2, 3) * i32(params.stride_w);
}

// The block's bias, zero past the block's channel count.
fn bias_block(channel: u32, channels: u32) -> vec4<f32> {
    var values = vec4<f32>(0.0);
    if (params.has_bias != 0u) {
        for (var k: u32 = 0u; k < channels; k = k + 1u) {
            values[k] = conv_bias[channel + k];
        }
    }
    return values;
}

// The block's PReLU slopes. A one-element weight is shared by every channel.
fn prelu_block(channel: u32, channels: u32) -> vec4<f32> {
    var values = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < channels; k = k + 1u) {
        let at = select(channel + k, 0u, params.prelu_len == 1u);
        values[k] = conv_prelu[at];
    }
    return values;
}

// Bias, activation and residual for one output pixel, then the store.
fn store_pixel(b: TileBlock, column: u32, sum: vec4<f32>) {
    let x = b.x + column;
    if (x >= params.output_w) {
        return;
    }

    var y = sum + bias_block(b.channel, b.channels);
    if (params.activation == 1u) {
        y = max(y, vec4<f32>(0.0));
    } else if (params.activation == 2u) {
        y = select(params.leaky_slope * y, y, y > vec4<f32>(0.0));
    } else if (params.activation == 3u) {
        y = select(prelu_block(b.channel, b.channels) * y, y, y > vec4<f32>(0.0));
    }

    let base = b.batch * params.dst_n + b.channel * params.dst_c + b.y * params.dst_h
        + x * params.dst_w;
    for (var k: u32 = 0u; k < b.channels; k = k + 1u) {
        let at = base + k * params.dst_c;
        var value = y[k];
        if (params.has_residual != 0u) {
            value = conv_residual[at] + params.residual_scale * value;
        }
        conv_output[at] = value;
    }
}
