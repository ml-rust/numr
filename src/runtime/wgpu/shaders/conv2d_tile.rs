//! Launcher, params and dispatch grid of the conv2d tile kernels (F32).
//!
//! One workgroup covers one spatial tile and one block of four output channels.
//! The grid keeps output channels on x, so consecutive workgroups read the same
//! input tile. WebGPU does not fix the dispatch order, so this locality is a
//! performance property, never a correctness one.
//!
//! The WGSL backend adds a counter to every loop. Each pass of the kernels'
//! innermost loop holds at least sixteen multiply-adds, so the counter stays a
//! small share of the work.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::ops::ConvLayout;

/// Output pixels per tile along W. Matches `TILE_COLUMNS` in the WGSL.
const TILE_COLUMNS: usize = 16;
/// Output pixels per tile along H. Matches `TILE_ROWS` in the WGSL.
const TILE_ROWS: usize = 8;
/// Output channels per workgroup. Matches `BLOCK` in the WGSL.
const CHANNEL_BLOCK: usize = 4;
/// The WebGPU minimum for `maxComputeWorkgroupsPerDimension`.
const MAX_GROUPS_PER_DIM: usize = 65535;

const SCALAR_SHADER: &str = concat!(
    include_str!("conv2d_tile_common.wgsl"),
    include_str!("conv2d_tile_scalar.wgsl")
);
const VEC4_SHADER: &str = concat!(
    include_str!("conv2d_tile_common.wgsl"),
    include_str!("conv2d_tile_vec4.wgsl")
);

/// `activation` value of [`Conv2dTileParams`] with no activation.
pub(crate) const ACTIVATION_NONE: u32 = 0;
/// `activation` value of [`Conv2dTileParams`] for ReLU.
pub(crate) const ACTIVATION_RELU: u32 = 1;
/// `activation` value of [`Conv2dTileParams`] for leaky ReLU.
pub(crate) const ACTIVATION_LEAKY: u32 = 2;
/// `activation` value of [`Conv2dTileParams`] for PReLU.
pub(crate) const ACTIVATION_PRELU: u32 = 3;

/// Uniform params. The layout matches `Conv2dTileParams` in the WGSL.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Conv2dTileParams {
    pub(crate) c_in: u32,
    pub(crate) c_out: u32,
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) output_h: u32,
    pub(crate) output_w: u32,
    pub(crate) kernel_h: u32,
    pub(crate) kernel_w: u32,
    pub(crate) stride_h: u32,
    pub(crate) stride_w: u32,
    pub(crate) pad_h: u32,
    pub(crate) pad_w: u32,
    pub(crate) dilation_h: u32,
    pub(crate) dilation_w: u32,
    pub(crate) tile_rows: u32,
    pub(crate) src_n: u32,
    pub(crate) src_c: u32,
    pub(crate) src_h: u32,
    pub(crate) src_w: u32,
    pub(crate) wgt_o: u32,
    pub(crate) wgt_i: u32,
    pub(crate) wgt_tap: u32,
    pub(crate) dst_n: u32,
    pub(crate) dst_c: u32,
    pub(crate) dst_h: u32,
    pub(crate) dst_w: u32,
    pub(crate) has_bias: u32,
    pub(crate) activation: u32,
    pub(crate) prelu_len: u32,
    pub(crate) has_residual: u32,
    pub(crate) leaky_slope: f32,
    pub(crate) residual_scale: f32,
}

/// The dispatch grid of one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileGrid {
    /// Channel blocks.
    pub(crate) x: u32,
    /// Tile columns.
    pub(crate) y: u32,
    /// Batch times tile rows.
    pub(crate) z: u32,
    /// Tile rows of one batch element.
    pub(crate) rows: u32,
}

/// The grid for an output of `batch` by `c_out` by `output_h` by `output_w`.
///
/// Returns `None` for an empty output, or a grid past the WebGPU minimum limit
/// on any dimension.
pub(crate) fn tile_grid(
    batch: usize,
    c_out: usize,
    output_h: usize,
    output_w: usize,
) -> Option<TileGrid> {
    if batch == 0 || c_out == 0 || output_h == 0 || output_w == 0 {
        return None;
    }
    let x = c_out.div_ceil(CHANNEL_BLOCK);
    let y = output_w.div_ceil(TILE_COLUMNS);
    let rows = output_h.div_ceil(TILE_ROWS);
    let z = batch.checked_mul(rows)?;
    if x > MAX_GROUPS_PER_DIM || y > MAX_GROUPS_PER_DIM || z > MAX_GROUPS_PER_DIM {
        return None;
    }
    Some(TileGrid {
        x: x as u32,
        y: y as u32,
        z: z as u32,
        rows: rows as u32,
    })
}

/// The kernel of one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Conv2dTileShader {
    /// Element strides on input and weight. Any layout and `C_in`.
    Scalar,
    /// `vec4` loads on input and weight. NHWC, `C_in` a multiple of 4.
    Vec4,
}

impl Conv2dTileShader {
    /// The fastest kernel that reads `layout` and `c_in`.
    pub(crate) fn select(layout: ConvLayout, c_in: usize) -> Self {
        if layout == ConvLayout::Nhwc && c_in.is_multiple_of(CHANNEL_BLOCK) {
            Self::Vec4
        } else {
            Self::Scalar
        }
    }
}

/// The buffers of one call, in binding order.
pub(crate) struct Conv2dTileBindings<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) weight: &'a Buffer,
    pub(crate) bias: &'a Buffer,
    pub(crate) prelu: &'a Buffer,
    pub(crate) residual: &'a Buffer,
    pub(crate) output: &'a Buffer,
    pub(crate) params: &'a Buffer,
}

/// Launch one tile kernel over `grid`.
pub(crate) fn launch_conv2d_tile(
    cache: &PipelineCache,
    queue: &Queue,
    shader: Conv2dTileShader,
    bindings: &Conv2dTileBindings<'_>,
    grid: TileGrid,
) {
    let (name, source) = match shader {
        Conv2dTileShader::Scalar => ("conv2d_tile_scalar", SCALAR_SHADER),
        Conv2dTileShader::Vec4 => ("conv2d_tile_vec4", VEC4_SHADER),
    };
    let module = cache.get_or_create_module(name, source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 5,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[
            bindings.input,
            bindings.weight,
            bindings.bias,
            bindings.prelu,
            bindings.residual,
            bindings.output,
            bindings.params,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(grid.x, grid.y, grid.z);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_every_output_element() {
        assert_eq!(
            tile_grid(1, 64, 1080, 1920),
            Some(TileGrid {
                x: 16,
                y: 120,
                z: 135,
                rows: 135
            })
        );
        assert_eq!(
            tile_grid(2, 5, 9, 17),
            Some(TileGrid {
                x: 2,
                y: 2,
                z: 4,
                rows: 2
            })
        );
    }

    #[test]
    fn empty_or_oversized_output_has_no_grid() {
        assert_eq!(tile_grid(0, 4, 8, 8), None);
        assert_eq!(tile_grid(1, 0, 8, 8), None);
        assert_eq!(tile_grid(1, 4, 0, 8), None);
        assert_eq!(tile_grid(1, 4, 8, 0), None);
        assert_eq!(tile_grid(65536, 4, 8, 8), None);
        assert_eq!(tile_grid(1, 4 * 65536, 8, 8), None);
    }

    #[test]
    fn vec4_kernel_takes_nhwc_multiples_of_four_only() {
        assert_eq!(
            Conv2dTileShader::select(ConvLayout::Nhwc, 64),
            Conv2dTileShader::Vec4
        );
        assert_eq!(
            Conv2dTileShader::select(ConvLayout::Nhwc, 3),
            Conv2dTileShader::Scalar
        );
        assert_eq!(
            Conv2dTileShader::select(ConvLayout::Nchw, 64),
            Conv2dTileShader::Scalar
        );
    }

    #[test]
    fn wgsl_constants_match_the_grid() {
        assert!(SCALAR_SHADER.contains("const TILE_COLUMNS: u32 = 16u;"));
        assert!(SCALAR_SHADER.contains("const TILE_ROWS: u32 = 8u;"));
        assert!(SCALAR_SHADER.contains("const BLOCK: u32 = 4u;"));
        assert_eq!(std::mem::size_of::<Conv2dTileParams>(), 32 * 4);
    }
}
