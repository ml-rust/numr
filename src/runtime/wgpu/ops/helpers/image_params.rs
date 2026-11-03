//! Params struct for the image gather kernels, laid out to match
//! `ImageParams` in image.wgsl byte for byte.

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ImageParams {
    /// Output element count.
    pub(crate) total: u32,
    /// Output C extent.
    pub(crate) dim_c: u32,
    /// Output H extent.
    pub(crate) dim_h: u32,
    /// Output W extent.
    pub(crate) dim_w: u32,
    /// 1 to decode the output index with C fastest (NHWC order), 0 for W fastest.
    pub(crate) channels_last: u32,
    /// Shuffle or unshuffle factor. Unused by the nearest kernel.
    pub(crate) factor: u32,
    /// Input H extent. Used by the nearest kernel only.
    pub(crate) in_h: u32,
    /// Input W extent. Used by the nearest kernel only.
    pub(crate) in_w: u32,
    /// Nearest mode along H: 0 copy, 1 halve, 2 scale.
    pub(crate) mode_h: u32,
    /// Nearest mode along W: 0 copy, 1 halve, 2 scale.
    pub(crate) mode_w: u32,
    /// Nearest f32 scale along H.
    pub(crate) scale_h: f32,
    /// Nearest f32 scale along W.
    pub(crate) scale_w: f32,
    /// Word index of the source view's first element in its buffer.
    pub(crate) src_base: u32,
    /// Word index of the destination view's first element in its buffer.
    pub(crate) dst_base: u32,
    /// Source strides as `[N, C, H, W]`, two's complement.
    pub(crate) src_strides: [u32; 4],
    /// Destination strides as `[N, C, H, W]`, two's complement.
    pub(crate) dst_strides: [u32; 4],
    /// Pads the struct to a multiple of 16 bytes.
    pub(crate) _pad: [u32; 2],
}
