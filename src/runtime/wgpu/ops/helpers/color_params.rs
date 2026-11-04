//! Params structs of the colour kernels, laid out to match `YuvParams` in
//! color_yuv.wgsl, `Rgb24Params` in color_rgb24.wgsl and `PadLumaParams` in
//! color_pad.wgsl byte for byte.
//!
//! F32 constants travel as their bit patterns, because the kernels compute
//! on bits.

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct YuvParams {
    /// Invocation count.
    pub(crate) total: u32,
    /// Luma rows `H`.
    pub(crate) height: u32,
    /// Luma columns `W`.
    pub(crate) width: u32,
    /// Encode column groups per chroma row. Unused by decode.
    pub(crate) groups: u32,
    /// 0 NV12, 1 I420, 2 P010.
    pub(crate) format: u32,
    /// 1 for `Center` chroma siting, 0 for `Left`.
    pub(crate) center: u32,
    /// Word of luma row 0 in its buffer.
    pub(crate) y_base: u32,
    /// Luma row stride in words, two's complement.
    pub(crate) y_rs: u32,
    /// Word of Cb row 0, or of interleaved chroma row 0.
    pub(crate) u_base: u32,
    /// Cb or interleaved chroma row stride in words, two's complement.
    pub(crate) u_rs: u32,
    /// Word of Cr row 0, or of interleaved chroma row 0.
    pub(crate) v_base: u32,
    /// Cr or interleaved chroma row stride in words, two's complement.
    pub(crate) v_rs: u32,
    /// Word of RGB element `[0, 0, 0]`.
    pub(crate) rgb_base: u32,
    /// RGB strides as `[C, H, W]` in words, two's complement.
    pub(crate) rgb_strides: [u32; 3],
    /// Luma code of black.
    pub(crate) y_off: i32,
    /// Chroma code of zero.
    pub(crate) c_off: i32,
    /// `ColorCoeffs` f32 fields from `c_div16` to `c_hi`, as bits.
    pub(crate) coeffs: [u32; 14],
    /// `y_off` and `c_off` as f32 bits.
    pub(crate) offsets: [u32; 2],
    /// Pads the struct to a multiple of 16 bytes.
    pub(crate) _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Rgb24Params {
    /// Invocation count.
    pub(crate) total: u32,
    /// Pixels per row `W`.
    pub(crate) width: u32,
    /// Packed words per row written by the pack kernel: `ceil(3·W / 4)`.
    pub(crate) row_words: u32,
    /// 1 when the packed words are new, so bytes past a row become 0.
    pub(crate) fresh: u32,
    /// Word of packed row 0 in its buffer.
    pub(crate) p_base: u32,
    /// Packed row stride in words, two's complement.
    pub(crate) p_rs: u32,
    /// Word of float element `[0, 0, 0]`.
    pub(crate) f_base: u32,
    /// Float strides as `[C, H, W]` in words, two's complement.
    pub(crate) f_strides: [u32; 3],
    /// Pads the struct to a multiple of 16 bytes.
    pub(crate) _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PadLumaParams {
    /// Invocation count: one per output word.
    pub(crate) total: u32,
    /// Output words per row, which is also the output row stride.
    pub(crate) row_words: u32,
    /// Word of source row 0 in its buffer.
    pub(crate) src_base: u32,
    /// Source row stride in words, two's complement.
    pub(crate) src_rs: u32,
    /// Source rows `H`.
    pub(crate) src_rows: u32,
    /// Source columns `W`.
    pub(crate) src_width: u32,
    /// Output columns.
    pub(crate) columns: u32,
    /// Bytes per sample: 1 or 2.
    pub(crate) sample_bytes: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_match_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<YuvParams>(), 36 * 4);
        assert_eq!(std::mem::size_of::<Rgb24Params>(), 12 * 4);
        assert_eq!(std::mem::size_of::<PadLumaParams>(), 8 * 4);
    }
}
