//! Config of [`ImageOps::resize`](super::ImageOps::resize).

/// Interpolation filter of [`ImageOps::resize`](super::ImageOps::resize).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResizeFilter {
    /// Linear interpolation per axis, `mode="bilinear"` in PyTorch.
    ///
    /// Without antialias it reads 2 taps per axis. With antialias it uses the
    /// triangle filter `1 - |x|`.
    Bilinear,
    /// Cubic convolution per axis, `mode="bicubic"` in PyTorch.
    ///
    /// Without antialias it reads 4 taps per axis with `a = -0.75` and edge
    /// indices clamped to the border pixel. With antialias it uses the Keys
    /// kernel with `a = -0.5`.
    Bicubic,
}

/// Output geometry and filter of [`ImageOps::resize`](super::ImageOps::resize).
///
/// `align_corners` is always false, as in `F.interpolate` with `size=` given.
/// Build it with [`ResizeConfig::new`] and [`ResizeConfig::with_antialias`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ResizeConfig {
    /// Output size `(height, width)`. Each must be at least 1.
    pub size: (usize, usize),
    /// Interpolation filter.
    pub filter: ResizeFilter,
    /// Use the antialias filters, as `antialias=True` does.
    ///
    /// On a downscaled axis the window widens by the downscale factor. For
    /// `Bicubic` the cubic constant is `-0.5` on every axis, not `-0.75`.
    pub antialias: bool,
}

impl ResizeConfig {
    /// Resize to `size = (height, width)` with `filter` and no antialias.
    pub fn new(size: (usize, usize), filter: ResizeFilter) -> Self {
        Self {
            size,
            filter,
            antialias: false,
        }
    }

    /// Set [`antialias`](Self::antialias).
    pub fn with_antialias(mut self, antialias: bool) -> Self {
        self.antialias = antialias;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults_to_no_antialias() {
        let c = ResizeConfig::new((3, 5), ResizeFilter::Bicubic);
        assert_eq!(c.size, (3, 5));
        assert_eq!(c.filter, ResizeFilter::Bicubic);
        assert!(!c.antialias);
        assert!(c.with_antialias(true).antialias);
    }
}
