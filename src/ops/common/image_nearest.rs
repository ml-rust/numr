//! Source-index rule of `upsample_nearest2d`, shared by every backend.
//!
//! The rule follows PyTorch's CPU `nearest_idx` in `ATen/native/UpSample.h`:
//!
//! - output size equal to the input size: `src = dst`
//! - output size twice the input size: `src = dst >> 1`
//! - otherwise: `src = min(floor(f32(dst) * scale), in - 1)` in f32
//!
//! `scale` is `f32(1 / s)` for a scale factor `s` (the division runs in f64),
//! and `f32(in) / f32(out)` for an explicit size. Kernels receive the mode and
//! the f32 scale, so every backend runs the same single f32 multiply.

/// Source-index mode of one spatial axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NearestAxis {
    /// Output size equals input size: `src = dst`.
    Identity,
    /// Output size is twice the input size: `src = dst >> 1`.
    Half,
    /// `src = min(floor(f32(dst) * scale), in - 1)`.
    Scaled(f32),
}

impl NearestAxis {
    /// Pick the mode of an axis from its sizes and the optional scale factor.
    ///
    /// `in_size` and `out_size` are at least 1. `scale_factor`, when given,
    /// is finite and positive.
    pub fn new(in_size: usize, out_size: usize, scale_factor: Option<f64>) -> Self {
        if out_size == in_size {
            return Self::Identity;
        }
        if Some(out_size) == in_size.checked_mul(2) {
            return Self::Half;
        }
        let scale = match scale_factor {
            Some(s) => (1.0 / s) as f32,
            None => in_size as f32 / out_size as f32,
        };
        Self::Scaled(scale)
    }

    /// Source index of output index `dst` on an axis of `in_size` elements.
    #[inline]
    pub fn source_index(self, dst: usize, in_size: usize) -> usize {
        match self {
            Self::Identity => dst,
            Self::Half => dst >> 1,
            Self::Scaled(scale) => {
                // `as usize` saturates, and the product is never negative.
                let src = (dst as f32 * scale).floor() as usize;
                src.min(in_size - 1)
            }
        }
    }
}

/// Kernel arguments. Only the GPU backends pass the rule to a kernel.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
impl NearestAxis {
    /// Kernel mode word for [`NearestAxis::Identity`].
    pub const MODE_IDENTITY: u32 = 0;
    /// Kernel mode word for [`NearestAxis::Half`].
    pub const MODE_HALF: u32 = 1;
    /// Kernel mode word for [`NearestAxis::Scaled`].
    pub const MODE_SCALED: u32 = 2;

    /// The `(mode, scale)` pair a kernel receives. `scale` is 0 unless the
    /// mode is [`Self::MODE_SCALED`].
    pub fn kernel_args(self) -> (u32, f32) {
        match self {
            Self::Identity => (Self::MODE_IDENTITY, 0.0),
            Self::Half => (Self::MODE_HALF, 0.0),
            Self::Scaled(scale) => (Self::MODE_SCALED, scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(in_size: usize, out_size: usize, scale_factor: Option<f64>) -> Vec<usize> {
        let axis = NearestAxis::new(in_size, out_size, scale_factor);
        (0..out_size)
            .map(|dst| axis.source_index(dst, in_size))
            .collect()
    }

    #[test]
    fn size_mode_matches_pytorch_tables() {
        assert_eq!(table(3, 7, None), vec![0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(table(7, 3, None), vec![0, 2, 4]);
        assert_eq!(table(5, 8, None), vec![0, 0, 1, 1, 2, 3, 3, 4]);
        assert_eq!(table(5, 3, None), vec![0, 1, 3]);
    }

    #[test]
    fn scale_mode_matches_pytorch_tables() {
        assert_eq!(table(5, 8, Some(1.7)), vec![0, 0, 1, 1, 2, 2, 3, 4]);
        assert_eq!(table(4, 6, Some(1.5)), vec![0, 0, 1, 2, 2, 3]);
        assert_eq!(table(5, 3, Some(0.75)), vec![0, 1, 2]);
        assert_eq!(table(3, 7, Some(2.5)), vec![0, 0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn twice_the_input_size_shifts_even_with_another_scale_factor() {
        assert_eq!(NearestAxis::new(3, 6, Some(2.3)), NearestAxis::Half);
        assert_eq!(table(3, 6, Some(2.3)), vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn equal_sizes_copy_even_with_another_scale_factor() {
        assert_eq!(NearestAxis::new(3, 3, Some(1.2)), NearestAxis::Identity);
        assert_eq!(table(3, 3, Some(1.2)), vec![0, 1, 2]);
    }

    #[test]
    fn integer_factor_repeats_each_index() {
        assert_eq!(table(2, 6, Some(3.0)), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(table(2, 6, None), vec![0, 0, 0, 1, 1, 1]);
    }

    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    #[test]
    fn kernel_args_carry_the_scale_only_when_scaled() {
        assert_eq!(NearestAxis::Identity.kernel_args(), (0, 0.0));
        assert_eq!(NearestAxis::Half.kernel_args(), (1, 0.0));
        assert_eq!(NearestAxis::Scaled(0.5).kernel_args(), (2, 0.5));
    }
}
