//! Extensible data type trait for tensor element types.

use std::fmt;
use std::hash::Hash;

use super::DType;

/// Trait for data types that can be stored in tensors.
///
/// numr's [`DType`] implements this. Downstream libraries (e.g. boostr) can
/// define their own dtype enums with quantized variants that also implement
/// this trait. The [`Runtime`](crate::runtime::Runtime) trait has an associated
/// `DType` type bounded by `DataType`, enabling each runtime to specify its
/// own dtype enum.
pub trait DataType:
    Copy + Clone + fmt::Debug + PartialEq + Eq + Hash + Send + Sync + 'static
{
    /// Size of one element in bytes.
    ///
    /// For block-quantized types, returns 1 as placeholder — use
    /// [`block_bytes`](Self::block_bytes) / [`block_size`](Self::block_size) for exact sizing.
    fn size_in_bytes(self) -> usize;

    /// Short display name (e.g., "f32", "q4_0").
    fn short_name(self) -> &'static str;

    /// Whether this is a floating point type.
    fn is_float(self) -> bool;

    /// Whether this is an integer type.
    fn is_int(self) -> bool;

    /// Whether this is a quantized/block type.
    fn is_quantized(self) -> bool {
        false
    }

    /// Block size for quantized types (elements per block), 1 for scalar types.
    fn block_size(self) -> usize {
        1
    }

    /// Bytes per block for quantized types, `size_in_bytes()` for scalar types.
    fn block_bytes(self) -> usize {
        self.size_in_bytes()
    }

    /// Total storage bytes for `numel` elements.
    fn storage_bytes(self, numel: usize) -> usize {
        if self.is_quantized() {
            let bs = self.block_size();
            let bb = self.block_bytes();
            ((numel + bs - 1) / bs) * bb
        } else {
            numel * self.size_in_bytes()
        }
    }

    /// Try to convert to numr's standard [`DType`].
    ///
    /// Returns `None` for custom/quantized types that have no numr equivalent.
    fn as_standard(&self) -> Option<DType>;

    /// Fill a buffer with `count` elements set to `value`, returning raw bytes.
    ///
    /// This enables generic constructors (zeros, ones, full_scalar) to work
    /// with any DType, not just numr's built-in DType. The default impl
    /// delegates to `as_standard()` and uses numr's fill logic.
    ///
    /// Downstream libraries with custom dtypes (e.g. quantized types) should
    /// override this if they need fill support.
    fn fill_bytes(self, value: f64, count: usize) -> Option<Vec<u8>> {
        self.as_standard()
            .map(|std_dtype| std_dtype.fill_bytes_impl(value, count))
    }
}

/// Implement `DataType` for numr's built-in `DType`.
impl DataType for DType {
    #[inline]
    fn size_in_bytes(self) -> usize {
        DType::size_in_bytes(self)
    }

    #[inline]
    fn short_name(self) -> &'static str {
        DType::short_name(self)
    }

    #[inline]
    fn is_float(self) -> bool {
        DType::is_float(self)
    }

    #[inline]
    fn is_int(self) -> bool {
        DType::is_int(self)
    }

    #[inline]
    fn as_standard(&self) -> Option<DType> {
        Some(*self)
    }
}
