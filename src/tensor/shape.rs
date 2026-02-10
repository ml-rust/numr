//! Shape type: dimensions of a tensor

use smallvec::SmallVec;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

/// Stack allocation threshold for dimensions
/// Most tensors have 4 or fewer dimensions, so we stack-allocate up to 4
pub(crate) const STACK_DIMS: usize = 4;

/// Shape type: dimensions of a tensor
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Shape(SmallVec<[usize; STACK_DIMS]>);

impl Shape {
    /// Create an empty shape.
    pub fn new() -> Self {
        Self(SmallVec::new())
    }

    /// Create an empty shape with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(SmallVec::with_capacity(capacity))
    }

    /// Push a dimension.
    pub fn push(&mut self, dim: usize) {
        self.0.push(dim);
    }

    /// Remove dimension at index.
    pub fn remove(&mut self, index: usize) -> usize {
        self.0.remove(index)
    }

    /// Insert a dimension at index.
    pub fn insert(&mut self, index: usize, value: usize) {
        self.0.insert(index, value);
    }

    /// Swap two dimensions.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.0.swap(a, b);
    }

    /// View shape as a slice.
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    /// Number of dimensions in this shape.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Number of dimensions in this shape.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Whether this shape has zero dimensions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        self.0.as_slice()
    }
}

impl From<SmallVec<[usize; STACK_DIMS]>> for Shape {
    fn from(value: SmallVec<[usize; STACK_DIMS]>) -> Self {
        Self(value)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self(value.iter().copied().collect())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
