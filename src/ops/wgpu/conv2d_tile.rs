//! Routing and arguments of the WebGPU tile kernels of `conv2d` and
//! `conv2d_fused`.
//!
//! A call takes the kernels when it runs in F32 with one group and has a
//! nonempty output that fits the dispatch grid. Every other call takes the
//! op's generic path.

use std::sync::Arc;

use wgpu::Buffer;

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_common::Conv2dParams;
use crate::ops::{ConvActivation, ConvLayout, ConvResidual};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::ops::helpers::{
    create_params_buffer, create_storage_buffer, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::conv2d_tile::{
    ACTIVATION_LEAKY, ACTIVATION_NONE, ACTIVATION_PRELU, ACTIVATION_RELU, Conv2dTileBindings,
    Conv2dTileParams, Conv2dTileShader, TileGrid, launch_conv2d_tile, tile_grid,
};
use crate::runtime::wgpu::shaders::linear_grid::element_count_u32;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

const OP: &str = "conv2d";

/// The grid of a call the tile kernels take, else `None`.
pub(super) fn tile_grid_for(conv: &Conv2dParams, dtype: DType) -> Option<TileGrid> {
    if dtype != DType::F32 || conv.groups != 1 {
        return None;
    }
    if conv.c_in == 0 || conv.height == 0 || conv.width == 0 {
        return None;
    }
    tile_grid(conv.batch, conv.c_out, conv.output_h, conv.output_w)
}

/// Element strides `(n, c, h, w)` of a contiguous image in `layout`.
fn image_strides(layout: ConvLayout, channels: usize, height: usize, width: usize) -> [u32; 4] {
    let (n, c, h, w) = match layout {
        ConvLayout::Nchw => (channels * height * width, height * width, width, 1),
        ConvLayout::Nhwc => (height * width * channels, 1, width * channels, channels),
    };
    [n as u32, c as u32, h as u32, w as u32]
}

/// Element strides `(output channel, input channel, tap)` of a contiguous
/// weight: OIHW for `Nchw`, OHWI for `Nhwc`.
fn weight_strides(layout: ConvLayout, conv: &Conv2dParams) -> [u32; 3] {
    let taps = conv.kernel_h * conv.kernel_w;
    let (o, i, tap) = match layout {
        ConvLayout::Nchw => (conv.c_in * taps, taps, 1),
        ConvLayout::Nhwc => (taps * conv.c_in, 1, conv.c_in),
    };
    [o as u32, i as u32, tap as u32]
}

/// The operands of one call after validation.
pub(super) struct TileCall<'a> {
    pub(super) input: &'a Tensor<WgpuRuntime>,
    pub(super) weight: &'a Tensor<WgpuRuntime>,
    pub(super) bias: Option<&'a Tensor<WgpuRuntime>>,
    pub(super) activation: ConvActivation<'a, WgpuRuntime>,
    pub(super) residual: Option<ConvResidual<'a, WgpuRuntime>>,
    pub(super) layout: ConvLayout,
    pub(super) conv: Conv2dParams,
    pub(super) grid: TileGrid,
}

/// The read operands of one call as whole-buffer bindings from element 0.
pub(super) struct TileBuffers {
    input: Arc<Buffer>,
    weight: Arc<Buffer>,
    bias: Option<Arc<Buffer>>,
    prelu: Option<Arc<Buffer>>,
    residual: Option<Arc<Buffer>>,
}

impl TileBuffers {
    /// True when `out` shares a buffer with a read operand. A compute pass
    /// cannot read and write one buffer.
    fn shares_with(&self, out: &Arc<Buffer>) -> bool {
        Arc::ptr_eq(out, &self.input)
            || Arc::ptr_eq(out, &self.weight)
            || [&self.bias, &self.prelu, &self.residual]
                .into_iter()
                .flatten()
                .any(|read| Arc::ptr_eq(out, read))
    }
}

/// A contiguous copy of `tensor` when it needs one, as a binding.
fn bind(tensor: &Tensor<WgpuRuntime>) -> Result<Arc<Buffer>> {
    get_tensor_buffer(&ensure_contiguous(tensor)?)
}

impl TileCall<'_> {
    /// Bind every read operand. A strided or offset operand copies first.
    pub(super) fn buffers(&self) -> Result<TileBuffers> {
        element_count_u32(self.input.numel(), OP)?;
        element_count_u32(self.weight.numel(), OP)?;
        let prelu = match self.activation {
            ConvActivation::Prelu(slope) => Some(bind(slope)?),
            _ => None,
        };
        Ok(TileBuffers {
            input: bind(self.input)?,
            weight: bind(self.weight)?,
            bias: self.bias.map(bind).transpose()?,
            prelu,
            residual: self.residual.map(|r| bind(r.tensor)).transpose()?,
        })
    }

    /// The uniform params of this call.
    fn params(&self) -> Conv2dTileParams {
        let conv = &self.conv;
        let src = image_strides(self.layout, conv.c_in, conv.height, conv.width);
        let dst = image_strides(self.layout, conv.c_out, conv.output_h, conv.output_w);
        let wgt = weight_strides(self.layout, conv);
        let (activation, leaky_slope, prelu_len) = match self.activation {
            ConvActivation::None => (ACTIVATION_NONE, 0.0, 0),
            ConvActivation::Relu => (ACTIVATION_RELU, 0.0, 0),
            ConvActivation::LeakyRelu(slope) => (ACTIVATION_LEAKY, slope as f32, 0),
            ConvActivation::Prelu(slope) => (ACTIVATION_PRELU, 0.0, slope.numel() as u32),
        };
        Conv2dTileParams {
            c_in: conv.c_in as u32,
            c_out: conv.c_out as u32,
            height: conv.height as u32,
            width: conv.width as u32,
            output_h: conv.output_h as u32,
            output_w: conv.output_w as u32,
            kernel_h: conv.kernel_h as u32,
            kernel_w: conv.kernel_w as u32,
            stride_h: conv.stride_h as u32,
            stride_w: conv.stride_w as u32,
            pad_h: conv.pad_top as u32,
            pad_w: conv.pad_left as u32,
            dilation_h: conv.dilation_h as u32,
            dilation_w: conv.dilation_w as u32,
            tile_rows: self.grid.rows,
            src_n: src[0],
            src_c: src[1],
            src_h: src[2],
            src_w: src[3],
            wgt_o: wgt[0],
            wgt_i: wgt[1],
            wgt_tap: wgt[2],
            dst_n: dst[0],
            dst_c: dst[1],
            dst_h: dst[2],
            dst_w: dst[3],
            has_bias: u32::from(self.bias.is_some()),
            activation,
            prelu_len,
            has_residual: u32::from(self.residual.is_some()),
            leaky_slope,
            residual_scale: self.residual.map_or(0.0, |r| r.scale as f32),
        }
    }

    /// Launch into `out`, which must be contiguous at offset 0 in a buffer no
    /// read operand uses.
    pub(super) fn launch(
        &self,
        client: &WgpuClient,
        out: &Tensor<WgpuRuntime>,
        buffers: &TileBuffers,
    ) -> Result<()> {
        element_count_u32(out.numel(), OP)?;
        let output = get_tensor_buffer(out)?;
        let params = create_params_buffer(client, &self.params());
        // Absent operands bind this buffer. The kernel never reads it.
        let unused = create_storage_buffer(client, &[0f32]);
        launch_conv2d_tile(
            client.pipeline_cache(),
            client.wgpu_queue(),
            Conv2dTileShader::select(self.layout, self.conv.c_in),
            &Conv2dTileBindings {
                input: &buffers.input,
                weight: &buffers.weight,
                bias: buffers.bias.as_deref().unwrap_or(&unused),
                prelu: buffers.prelu.as_deref().unwrap_or(&unused),
                residual: buffers.residual.as_deref().unwrap_or(&unused),
                output: &output,
                params: &params,
            },
            self.grid,
        );
        Ok(())
    }

    /// True when `out` can bind as this call's output, with no temporary.
    pub(super) fn binds_output(
        &self,
        out: &Tensor<WgpuRuntime>,
        buffers: &TileBuffers,
    ) -> Result<bool> {
        if !out.is_contiguous() || out.offset() != 0 {
            return Ok(false);
        }
        Ok(!buffers.shares_with(&get_tensor_buffer(out)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_follow_the_layout() {
        assert_eq!(
            image_strides(ConvLayout::Nchw, 3, 5, 7),
            [3 * 5 * 7, 5 * 7, 7, 1]
        );
        assert_eq!(
            image_strides(ConvLayout::Nhwc, 3, 5, 7),
            [5 * 7 * 3, 1, 7 * 3, 3]
        );
    }
}
