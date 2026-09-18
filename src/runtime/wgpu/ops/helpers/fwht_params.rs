//! Params struct for the Walsh-Hadamard kernels, laid out to match
//! `FwhtParams` in fwht_local_f32.wgsl and fwht_global_f32.wgsl byte for byte.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FwhtParams {
    /// Independent rows: `numel / last_dim`.
    pub(crate) rows: u32,
    /// Last-dim width, a multiple of `block_size`.
    pub(crate) last_dim: u32,
    /// Transform width, a power of two.
    pub(crate) block_size: u32,
    /// Workgroup tile width of the local pass, a power of two >= 256.
    pub(crate) chunk: u32,
    /// Butterfly stride of one global-pass dispatch; unused by the local pass.
    pub(crate) h: u32,
    /// 1 when the `signs` binding holds a real row, 0 when it is the dummy.
    pub(crate) has_signs: u32,
    /// `1 / sqrt(block_size)`, folded into the local pass load.
    pub(crate) scale: f32,
}
