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
                if let Some(idx) = self.normalize_dim(d) {
                    if self.shape[idx] == 1 {
                        let mut new_shape = self.shape.clone();
                        let mut new_strides = self.strides.clone();
                        new_shape.remove(idx);
                        new_strides.remove(idx);
                        return Self::new(new_shape, new_strides, self.offset);
                    }
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

/// Compute the broadcast shape of two shapes
///
/// Currently used primarily in validation. May be exported or used by layout operations in future.
#[allow(dead_code)]
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Shape> {
    let max_ndim = a.len().max(b.len());
    let mut result = Shape::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None; // Incompatible shapes
        }
    }

    result.reverse();
    Some(result)
}

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

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(
            broadcast_shapes(&[3, 1], &[1, 4]),
            Some(SmallVec::from_slice(&[3, 4]))
        );
        assert_eq!(
            broadcast_shapes(&[2, 3, 4], &[4]),
            Some(SmallVec::from_slice(&[2, 3, 4]))
        );
        assert_eq!(broadcast_shapes(&[3], &[4]), None);
    }

    #[test]
    fn test_index() {
        let layout = Layout::contiguous(&[2, 3]);
        assert_eq!(layout.index(&[0, 0]), Some(0));
        assert_eq!(layout.index(&[0, 2]), Some(2));
        assert_eq!(layout.index(&[1, 0]), Some(3));
        assert_eq!(layout.index(&[1, 2]), Some(5));
        assert_eq!(layout.index(&[2, 0]), None); // Out of bounds
    }
}
