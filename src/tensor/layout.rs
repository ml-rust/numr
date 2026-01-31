//! Layout: shape, strides, and offset for tensor memory layout

use smallvec::SmallVec;
use std::fmt;

/// Stack allocation threshold for dimensions
/// Most tensors have 4 or fewer dimensions, so we stack-allocate up to 4
const STACK_DIMS: usize = 4;

/// Shape type: dimensions of a tensor
pub type Shape = SmallVec<[usize; STACK_DIMS]>;

/// Strides type: element offsets between consecutive elements along each dimension
/// Signed to support negative strides (e.g., for flip operations)
/// NOTE: Strides are in ELEMENTS, not bytes (per TDD §3.0.1)
pub type Strides = SmallVec<[isize; STACK_DIMS]>;

/// Layout describes the memory layout of a tensor
///
/// A tensor's elements are stored in a contiguous buffer, but not necessarily
/// in row-major order. The layout specifies how to compute the memory address
/// of any element given its indices.
///
/// Address of element at indices [i0, i1, ..., in]:
///   offset + i0 * strides[0] + i1 * strides[1] + ... + in * strides[n]
#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    /// Shape: size along each dimension
    shape: Shape,
    /// Strides: offset (in elements) between consecutive elements along each dimension
    strides: Strides,
    /// Offset: starting element index in the underlying storage
    offset: usize,
}

impl Layout {
    /// Create a new contiguous (row-major/C-order) layout from a shape
    ///
    /// # Example
    /// ```
    /// use numr::tensor::Layout;
    /// let layout = Layout::contiguous(&[2, 3, 4]);
    /// assert_eq!(layout.shape(), &[2, 3, 4]);
    /// assert_eq!(layout.strides(), &[12, 4, 1]);
    /// ```
    pub fn contiguous(shape: &[usize]) -> Self {
        let shape: Shape = shape.iter().copied().collect();
        let strides = Self::compute_contiguous_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create a layout with explicit shape, strides, and offset
    pub fn new(shape: Shape, strides: Strides, offset: usize) -> Self {
        debug_assert_eq!(shape.len(), strides.len());
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Create a scalar (0-dimensional) layout
    pub fn scalar() -> Self {
        Self {
            shape: SmallVec::new(),
            strides: SmallVec::new(),
            offset: 0,
        }
    }

    /// Compute contiguous strides for a given shape (row-major order)
    fn compute_contiguous_strides(shape: &[usize]) -> Strides {
        if shape.is_empty() {
            return SmallVec::new();
        }

        let mut strides: Strides = SmallVec::with_capacity(shape.len());
        let mut stride = 1isize;

        // Compute strides from last dimension to first
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim as isize;
        }

        strides.reverse();
        strides
    }

    /// Get the shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get the offset
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of dimensions (rank)
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements
    #[inline]
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the tensor is a scalar (0 dimensions)
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Check if memory is contiguous (row-major order)
    pub fn is_contiguous(&self) -> bool {
        if self.is_scalar() {
            return true;
        }

        let expected = Self::compute_contiguous_strides(&self.shape);
        self.strides == expected && self.offset == 0
    }

    /// Get size along a specific dimension
    ///
    /// Supports negative indexing: -1 is the last dimension
    pub fn dim(&self, d: isize) -> Option<usize> {
        let idx = self.normalize_dim(d)?;
        Some(self.shape[idx])
    }

    /// Get stride along a specific dimension
    pub fn stride(&self, d: isize) -> Option<isize> {
        let idx = self.normalize_dim(d)?;
        Some(self.strides[idx])
    }

    /// Normalize a dimension index (handle negative indices)
    pub fn normalize_dim(&self, d: isize) -> Option<usize> {
        let ndim = self.ndim() as isize;
        let idx = if d < 0 { ndim + d } else { d };
        if idx >= 0 && idx < ndim {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Compute the linear index (element offset) for given indices
    pub fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }

        // Check bounds
        for (idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if *idx >= dim {
                return None;
            }
        }

        let mut linear = self.offset as isize;
        for (&idx, &stride) in indices.iter().zip(self.strides.iter()) {
            linear += idx as isize * stride;
        }

        Some(linear as usize)
    }

    /// Create a transposed layout (swap two dimensions)
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Option<Self> {
        let d0 = self.normalize_dim(dim0)?;
        let d1 = self.normalize_dim(dim1)?;

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(d0, d1);
        new_strides.swap(d0, d1);

        Some(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Create a reshaped layout (if contiguous)
    ///
    /// Returns None if the tensor is not contiguous or shapes don't match
    pub fn reshape(&self, new_shape: &[usize]) -> Option<Self> {
        // Must be contiguous to reshape without copying
        if !self.is_contiguous() {
            return None;
        }

        // Element count must match
        let new_count: usize = new_shape.iter().product();
        if new_count != self.elem_count() {
            return None;
        }

        Some(Self::contiguous(new_shape))
    }

    /// Create a squeezed layout (remove dimensions of size 1)
    pub fn squeeze(&self, dim: Option<isize>) -> Self {
        match dim {
            Some(d) => {
                if let Some(idx) = self.normalize_dim(d).filter(|&i| self.shape[i] == 1) {
                    let mut new_shape = self.shape.clone();
                    let mut new_strides = self.strides.clone();
                    new_shape.remove(idx);
                    new_strides.remove(idx);
                    return Self::new(new_shape, new_strides, self.offset);
                }
                self.clone()
            }
            None => {
                let mut new_shape = Shape::new();
                let mut new_strides = Strides::new();
                for (&s, &st) in self.shape.iter().zip(self.strides.iter()) {
                    if s != 1 {
                        new_shape.push(s);
                        new_strides.push(st);
                    }
                }
                Self::new(new_shape, new_strides, self.offset)
            }
        }
    }

    /// Create an unsqueezed layout (add dimension of size 1)
    pub fn unsqueeze(&self, dim: isize) -> Option<Self> {
        let ndim = self.ndim();
        let idx = if dim < 0 {
            (ndim as isize + dim + 1) as usize
        } else {
            dim as usize
        };

        if idx > ndim {
            return None;
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        // Stride for the new dimension: product of strides after this position
        let new_stride = if idx < ndim {
            new_strides[idx] * new_shape[idx] as isize
        } else {
            // Last dimension or scalar case: stride = 1
            1
        };

        new_shape.insert(idx, 1);
        new_strides.insert(idx, new_stride);

        Some(Self::new(new_shape, new_strides, self.offset))
    }

    /// Create a permuted layout by reordering dimensions
    ///
    /// # Arguments
    /// * `dims` - New order of dimensions (permutation of 0..ndim)
    ///
    /// # Returns
    /// None if dims is invalid (wrong length, duplicates, or out-of-range values)
    ///
    /// # Example
    /// ```
    /// use numr::tensor::Layout;
    /// let layout = Layout::contiguous(&[2, 3, 4]);
    /// let permuted = layout.permute(&[2, 0, 1]).unwrap();
    /// assert_eq!(permuted.shape(), &[4, 2, 3]);
    /// assert_eq!(permuted.strides(), &[1, 12, 4]);
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Option<Self> {
        let ndim = self.ndim();

        // dims must have same length as number of dimensions
        if dims.len() != ndim {
            return None;
        }

        // Validate dims is a valid permutation: each index 0..ndim appears exactly once
        let mut seen = vec![false; ndim];
        for &d in dims {
            if d >= ndim || seen[d] {
                return None;
            }
            seen[d] = true;
        }

        // Create new shape and strides by reordering
        let mut new_shape = Shape::with_capacity(ndim);
        let mut new_strides = Strides::with_capacity(ndim);

        for &d in dims {
            new_shape.push(self.shape[d]);
            new_strides.push(self.strides[d]);
        }

        Some(Self::new(new_shape, new_strides, self.offset))
    }

    /// Create a narrowed layout (slice along a dimension)
    ///
    /// # Arguments
    /// * `dim` - Dimension to narrow
    /// * `start` - Starting index
    /// * `length` - Number of elements to keep
    ///
    /// # Returns
    /// None if parameters are out of bounds
    ///
    /// # Example
    /// ```
    /// use numr::tensor::Layout;
    /// let layout = Layout::contiguous(&[4, 5, 6]);
    /// let narrowed = layout.narrow(1, 1, 3).unwrap();
    /// assert_eq!(narrowed.shape(), &[4, 3, 6]);
    /// assert_eq!(narrowed.offset(), 6); // Skip first row of dim 1
    /// ```
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Option<Self> {
        if dim >= self.ndim() {
            return None;
        }

        let dim_size = self.shape[dim];
        if start >= dim_size || start + length > dim_size {
            return None;
        }

        if length == 0 {
            return None;
        }

        // New shape: same as original but with dim narrowed
        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;

        // Strides remain the same
        let new_strides = self.strides.clone();

        // Offset increases by start * stride[dim]
        let new_offset = self.offset as isize + start as isize * self.strides[dim];
        if new_offset < 0 {
            return None;
        }

        Some(Self::new(new_shape, new_strides, new_offset as usize))
    }

    /// Create a flipped layout along a dimension (zero-copy via negative stride)
    ///
    /// Reverses the order of elements along the specified dimension by:
    /// 1. Negating the stride for that dimension
    /// 2. Adjusting the offset to point to the last element along that dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension to flip (supports negative indexing)
    ///
    /// # Returns
    /// None if dimension is out of bounds
    ///
    /// # Example
    /// ```
    /// use numr::tensor::Layout;
    /// let layout = Layout::contiguous(&[2, 3]); // strides [3, 1]
    /// let flipped = layout.flip(-1).unwrap();   // flip last dim
    /// assert_eq!(flipped.strides(), &[3, -1]);  // negative stride
    /// assert_eq!(flipped.offset(), 2);          // points to last element of row
    /// ```
    pub fn flip(&self, dim: isize) -> Option<Self> {
        let idx = self.normalize_dim(dim)?;

        // If dimension size is 0 or 1, flip is a no-op
        if self.shape[idx] <= 1 {
            return Some(self.clone());
        }

        let mut new_strides = self.strides.clone();
        let old_stride = self.strides[idx];

        // Negate the stride
        new_strides[idx] = -old_stride;

        // Adjust offset to point to the "last" element along this dimension
        // New offset = old_offset + (dim_size - 1) * old_stride
        // Use checked arithmetic to prevent overflow
        let dim_size = self.shape[idx] as isize;
        let stride_factor = (dim_size - 1).checked_mul(old_stride)?;
        let new_offset = (self.offset as isize).checked_add(stride_factor)?;

        // Sanity check: offset should be non-negative
        if new_offset < 0 {
            return None;
        }

        Some(Self::new(
            self.shape.clone(),
            new_strides,
            new_offset as usize,
        ))
    }

    /// Create a flipped layout along multiple dimensions (zero-copy)
    ///
    /// Equivalent to calling `flip` multiple times, but more efficient.
    ///
    /// # Arguments
    /// * `dims` - Dimensions to flip (supports negative indexing)
    ///
    /// # Returns
    /// None if any dimension is out of bounds
    pub fn flip_dims(&self, dims: &[isize]) -> Option<Self> {
        let mut result = self.clone();
        for &dim in dims {
            result = result.flip(dim)?;
        }
        Some(result)
    }

    /// Create a broadcast layout to a target shape
    ///
    /// Returns None if shapes are not broadcastable
    pub fn broadcast_to(&self, target: &[usize]) -> Option<Self> {
        if target.len() < self.ndim() {
            return None;
        }

        let mut new_shape = Shape::new();
        let mut new_strides = Strides::new();

        // Pad with leading 1s
        let pad = target.len() - self.ndim();
        for &t in &target[..pad] {
            new_shape.push(t);
            new_strides.push(0); // Stride 0 for broadcast dimensions
        }

        // Check compatibility and compute strides
        for ((&s, &st), &t) in self
            .shape
            .iter()
            .zip(self.strides.iter())
            .zip(&target[pad..])
        {
            if s == t {
                new_shape.push(t);
                new_strides.push(st);
            } else if s == 1 {
                new_shape.push(t);
                new_strides.push(0); // Broadcast: stride 0
            } else {
                return None; // Incompatible shapes
            }
        }

        Some(Self::new(new_shape, new_strides, self.offset))
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layout {{ shape: {:?}, strides: {:?}, offset: {} }}",
            self.shape.as_slice(),
            self.strides.as_slice(),
            self.offset
        )
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.shape.as_slice())
    }
}

// Note: broadcast_shape is implemented in crate::ops::arithmetic and is the canonical version.
// Use crate::ops::broadcast_shape for broadcasting logic.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_layout() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        assert_eq!(layout.shape(), &[2, 3, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert_eq!(layout.elem_count(), 24);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_scalar_layout() {
        let layout = Layout::scalar();
        assert!(layout.is_scalar());
        assert_eq!(layout.elem_count(), 1);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_transpose() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        let transposed = layout.transpose(-1, -2).unwrap();
        assert_eq!(transposed.shape(), &[2, 4, 3]);
        assert_eq!(transposed.strides(), &[12, 1, 4]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_reshape() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        let reshaped = layout.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn test_squeeze() {
        let layout = Layout::contiguous(&[1, 3, 1, 4]);
        let squeezed = layout.squeeze(None);
        assert_eq!(squeezed.shape(), &[3, 4]);
    }

    #[test]
    fn test_unsqueeze() {
        let layout = Layout::contiguous(&[3, 4]);
        let unsqueezed = layout.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3, 4]);
    }

    // Note: broadcast_shape tests are in ops/arithmetic.rs

    #[test]
    fn test_index() {
        let layout = Layout::contiguous(&[2, 3]);
        assert_eq!(layout.index(&[0, 0]), Some(0));
        assert_eq!(layout.index(&[0, 2]), Some(2));
        assert_eq!(layout.index(&[1, 0]), Some(3));
        assert_eq!(layout.index(&[1, 2]), Some(5));
        assert_eq!(layout.index(&[2, 0]), None); // Out of bounds
    }

    #[test]
    fn test_permute() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        // Original strides: [12, 4, 1]

        // Permute to [4, 2, 3] -> dims [2, 0, 1]
        let permuted = layout.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(permuted.strides(), &[1, 12, 4]);
        assert!(!permuted.is_contiguous());

        // Identity permutation should preserve layout
        let identity = layout.permute(&[0, 1, 2]).unwrap();
        assert_eq!(identity.shape(), &[2, 3, 4]);
        assert_eq!(identity.strides(), &[12, 4, 1]);
        assert!(identity.is_contiguous());

        // Invalid permutation: wrong length
        assert!(layout.permute(&[0, 1]).is_none());

        // Invalid permutation: duplicate
        assert!(layout.permute(&[0, 0, 1]).is_none());

        // Invalid permutation: out of range
        assert!(layout.permute(&[0, 1, 5]).is_none());
    }

    #[test]
    fn test_narrow() {
        let layout = Layout::contiguous(&[4, 5, 6]);
        // Original strides: [30, 6, 1]

        // Narrow dim 1: take elements 1..4 (3 elements)
        let narrowed = layout.narrow(1, 1, 3).unwrap();
        assert_eq!(narrowed.shape(), &[4, 3, 6]);
        assert_eq!(narrowed.strides(), &[30, 6, 1]);
        assert_eq!(narrowed.offset(), 6); // 1 * stride[1] = 1 * 6 = 6

        // Narrow dim 0: take elements 2..4 (2 elements)
        let narrowed2 = layout.narrow(0, 2, 2).unwrap();
        assert_eq!(narrowed2.shape(), &[2, 5, 6]);
        assert_eq!(narrowed2.offset(), 60); // 2 * stride[0] = 2 * 30 = 60

        // Narrow last dim
        let narrowed3 = layout.narrow(2, 0, 3).unwrap();
        assert_eq!(narrowed3.shape(), &[4, 5, 3]);
        assert_eq!(narrowed3.offset(), 0);

        // Invalid: dim out of range
        assert!(layout.narrow(3, 0, 1).is_none());

        // Invalid: start out of range
        assert!(layout.narrow(0, 5, 1).is_none());

        // Invalid: start + length out of range
        assert!(layout.narrow(0, 3, 3).is_none());

        // Invalid: length = 0
        assert!(layout.narrow(0, 0, 0).is_none());
    }

    #[test]
    fn test_flip() {
        let layout = Layout::contiguous(&[2, 3]);
        // Original: strides [3, 1], offset 0

        // Flip last dim
        let flipped = layout.flip(-1).unwrap();
        assert_eq!(flipped.shape(), &[2, 3]);
        assert_eq!(flipped.strides(), &[3, -1]); // Negated stride
        assert_eq!(flipped.offset(), 2); // Point to last element of row (0 + 2*1)

        // Flip first dim
        let flipped2 = layout.flip(0).unwrap();
        assert_eq!(flipped2.shape(), &[2, 3]);
        assert_eq!(flipped2.strides(), &[-3, 1]); // Negated stride for dim 0
        assert_eq!(flipped2.offset(), 3); // Point to last row (0 + 1*3)

        // Flip dimension with size 1 - should be a no-op
        let layout1d = Layout::contiguous(&[1, 5]);
        let flipped1 = layout1d.flip(0).unwrap();
        assert_eq!(flipped1.strides(), &[5, 1]); // Unchanged since size is 1
        assert_eq!(flipped1.offset(), 0);

        // Invalid dimension
        assert!(layout.flip(5).is_none());
        assert!(layout.flip(-5).is_none());
    }

    #[test]
    fn test_flip_dims() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        // Original: strides [12, 4, 1], offset 0

        // Flip multiple dims
        let flipped = layout.flip_dims(&[0, 2]).unwrap();
        assert_eq!(flipped.strides(), &[-12, 4, -1]);
        // Offset: dim 0 adds (2-1)*12 = 12, dim 2 adds (4-1)*1 = 3 → total = 15
        assert_eq!(flipped.offset(), 15);

        // Empty dims - no change
        let flipped_empty = layout.flip_dims(&[]).unwrap();
        assert_eq!(flipped_empty.strides(), layout.strides());
        assert_eq!(flipped_empty.offset(), layout.offset());
    }

    #[test]
    fn test_flip_index() {
        // Test that flipped layout indexes correctly
        let layout = Layout::contiguous(&[2, 3]);
        // Memory: [0, 1, 2, 3, 4, 5] representing [[0,1,2], [3,4,5]]

        // Flip last dim: should index as [[2,1,0], [5,4,3]]
        let flipped = layout.flip(-1).unwrap();
        assert_eq!(flipped.index(&[0, 0]), Some(2)); // Was index 2, now at [0,0]
        assert_eq!(flipped.index(&[0, 1]), Some(1)); // Was index 1, now at [0,1]
        assert_eq!(flipped.index(&[0, 2]), Some(0)); // Was index 0, now at [0,2]
        assert_eq!(flipped.index(&[1, 0]), Some(5)); // Was index 5
        assert_eq!(flipped.index(&[1, 2]), Some(3)); // Was index 3
    }
}
