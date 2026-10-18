//! Params structs for the built-in RNG kernels, and the host-side seed source.
//!
//! `seed` and `seed_hi` together carry a full u64 seed, since WGSL uniforms
//! cannot hold a 64-bit integer.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) seed_hi: u32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandnParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) seed_hi: u32,
    pub(crate) _pad: u32,
}

/// Randint params for signed integer types (`I32`)
/// The `low` field is i32 to properly handle negative bounds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandintParamsI32 {
    pub(crate) numel: u32,
    pub(crate) low: i32, // Signed low bound
    pub(crate) range: u32,
    pub(crate) seed: u32,
}

/// Randint params for unsigned integer types (`U32`)
/// The `low` field is u32 for unsigned bounds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RandintParamsU32 {
    pub(crate) numel: u32,
    pub(crate) low: u32, // Unsigned low bound
    pub(crate) range: u32,
    pub(crate) seed: u32,
}

/// Return the u32 seed for one unseeded WebGPU RNG kernel that takes a single word.
///
/// The value folds [`crate::random::next_call_seed`] to 32 bits.
/// Both halves of the u64 reach the result, so the fold is deterministic.
pub(crate) fn generate_wgpu_seed() -> u32 {
    fold_seed(crate::random::next_call_seed())
}

/// Return `(seed, seed_hi)` for one unseeded kernel that takes both words.
pub(crate) fn generate_wgpu_seed_words() -> (u32, u32) {
    let seed = crate::random::next_call_seed();
    (seed as u32, (seed >> 32) as u32)
}

/// Fold a u64 seed to u32 by XOR of its two halves.
fn fold_seed(seed: u64) -> u32 {
    (seed ^ (seed >> 32)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_follow_the_seeded_stream() {
        let words = crate::random::with_seed(21, generate_wgpu_seed_words);
        let folded = crate::random::with_seed(21, generate_wgpu_seed);
        let raw = crate::random::with_seed(21, crate::random::next_call_seed);
        assert_eq!(words, (raw as u32, (raw >> 32) as u32));
        assert_eq!(folded, fold_seed(raw));
        assert_eq!(fold_seed(0x0000_0001_0000_0003), 2);
    }
}
