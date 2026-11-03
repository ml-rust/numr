//! Params struct for the `resize` pass kernel, laid out to match
//! `ResizeParams` in image_resize.wgsl byte for byte.

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ResizeParams {
    /// Destination element count.
    pub(crate) total: u32,
    /// Destination C extent.
    pub(crate) dim_c: u32,
    /// Destination H extent.
    pub(crate) dim_h: u32,
    /// Destination W extent.
    pub(crate) dim_w: u32,
    /// 1 to decode the destination index with C fastest (NHWC order).
    pub(crate) channels_last: u32,
    /// Interpolated canonical axis: 2 for H, 3 for W.
    pub(crate) axis: u32,
    /// `ResizeMode::kernel_mode` of the pass.
    pub(crate) mode: u32,
    /// Source extent along `axis`.
    pub(crate) in_size: u32,
    /// Destination extent along `axis`.
    pub(crate) out_size: u32,
    /// Word index of the source view's first element in its buffer.
    pub(crate) src_base: u32,
    /// Word index of the destination view's first element in its buffer.
    pub(crate) dst_base: u32,
    /// `f32(in_size) / f32(out_size)`, correctly rounded.
    pub(crate) scale: f32,
    /// `1 / scale` when `scale >= 1`, else 1, correctly rounded.
    pub(crate) invscale: f32,
    /// Source strides as `[N, C, H, W]`, two's complement.
    pub(crate) src_strides: [u32; 4],
    /// Destination strides as `[N, C, H, W]`, two's complement.
    pub(crate) dst_strides: [u32; 4],
    /// Pads the struct to a multiple of 16 bytes.
    pub(crate) _pad: [u32; 3],
}
