//! Packed 32-bit word form of a plane or an RGB24 frame for
//! [`ColorOps`](super::ColorOps).

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// One plane, or one RGB24 frame, stored as packed little-endian 32-bit words.
///
/// `words` is a U32 view `[rows, n]`. Row `r` starts at word `[r, 0]`. Its
/// samples follow back to back from byte 0 of that word, low byte first:
///
/// | Data | Bytes per column | Byte of column `c` |
/// | --- | --- | --- |
/// | U8 plane: NV12, I420 | 1 | `c` |
/// | U16 plane: P010 | 2 | `2c`, `2c + 1` |
/// | RGB24 | 3 | `3c` (R), `3c + 1` (G), `3c + 2` (B) |
///
/// `width` is the column count of the unpacked data: the `W` of `[H, W]` for
/// a plane or of `[H, W, 3]` for RGB24. An interleaved NV12 or P010 chroma
/// plane has `width = W`, two columns per chroma sample pair.
///
/// The row stride of `words` is the pitch in words, and can be any value.
/// Words along a row are contiguous: the column stride is 1. `n` covers the
/// row: `n >= ceil(width · bytes / 4)`. Bytes of a row past its last
/// column keep their values on every write, including the tail of the last
/// partial word.
///
/// On a little-endian host the words are the same bytes as the unpacked
/// U8 or U16 plane of the same pitch, so CPU and CUDA read and write them in
/// place. WebGPU kernels address 32-bit words only, so WebGPU takes planes
/// and RGB24 in this form only.
#[non_exhaustive]
pub struct PackedPlane<'a, R: Runtime> {
    /// U32 words `[rows, n]`, column stride 1.
    pub words: &'a Tensor<R>,
    /// Columns of the unpacked data per row.
    pub width: usize,
}

impl<R: Runtime> Clone for PackedPlane<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for PackedPlane<'_, R> {}

impl<R: Runtime> std::fmt::Debug for PackedPlane<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedPlane")
            .field("shape", &self.words.shape())
            .field("width", &self.width)
            .finish()
    }
}

impl<'a, R: Runtime> PackedPlane<'a, R> {
    /// Packed view of `width` columns per row over the U32 view `words`.
    pub fn new(words: &'a Tensor<R>, width: usize) -> Self {
        Self { words, width }
    }

    /// Words a row of `row_bytes` bytes covers: `ceil(row_bytes / 4)`.
    pub fn row_words(row_bytes: usize) -> usize {
        row_bytes.div_ceil(4)
    }

    /// Shape `[rows, pitch_bytes / 4]` of a U32 surface with a row pitch of
    /// `pitch_bytes`.
    ///
    /// # Errors
    ///
    /// `Error::InvalidArgument` for argument `pitch_bytes` when the pitch is
    /// not a whole number of 32-bit words.
    pub fn words_shape(rows: usize, pitch_bytes: usize) -> Result<[usize; 2]> {
        if !pitch_bytes.is_multiple_of(4) {
            return Err(Error::InvalidArgument {
                arg: "pitch_bytes",
                reason: format!(
                    "row pitch {pitch_bytes} bytes is not a whole number of 32-bit words; \
                     align the pitch to 4 bytes, for example {}",
                    pitch_bytes.div_ceil(4) * 4
                ),
            });
        }
        Ok([rows, pitch_bytes / 4])
    }
}
