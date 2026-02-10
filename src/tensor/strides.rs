//! Strides type: element offsets for tensor memory layout

use super::shape::STACK_DIMS;
use smallvec::SmallVec;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

/// Strides type: element offsets between consecutive elements along each dimension
/// Signed to support negative strides (e.g., for flip operations)
/// NOTE: Strides are in ELEMENTS, not bytes
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Strides(SmallVec<[isize; STACK_DIMS]>);

impl Strides {
    /// Create empty strides.
    pub fn new() -> Self {
        Self(SmallVec::new())
    }

    /// Create empty strides with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(SmallVec::with_capacity(capacity))
    }

    /// Push a stride value.
    pub fn push(&mut self, stride: isize) {
        self.0.push(stride);
    }

    /// Remove stride at index.
    pub fn remove(&mut self, index: usize) -> isize {
        self.0.remove(index)
    }

    /// Insert stride at index.
    pub fn insert(&mut self, index: usize, value: isize) {
        self.0.insert(index, value);
    }

    /// Swap two strides.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.0.swap(a, b);
    }

    /// Reverse stride order.
    pub fn reverse(&mut self) {
        self.0.reverse();
    }

    /// View strides as a slice.
    pub fn as_slice(&self) -> &[isize] {
        self.0.as_slice()
    }

    /// Number of stride entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this stride vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Deref for Strides {
    type Target = [isize];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for Strides {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

impl fmt::Debug for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl AsRef<[isize]> for Strides {
    fn as_ref(&self) -> &[isize] {
        self.0.as_slice()
    }
}

impl From<SmallVec<[isize; STACK_DIMS]>> for Strides {
    fn from(value: SmallVec<[isize; STACK_DIMS]>) -> Self {
        Self(value)
    }
}

impl From<Vec<isize>> for Strides {
    fn from(value: Vec<isize>) -> Self {
        Self(value.into_iter().collect())
    }
}

impl From<&[isize]> for Strides {
    fn from(value: &[isize]) -> Self {
        Self(value.iter().copied().collect())
    }
}

impl<const N: usize> From<[isize; N]> for Strides {
    fn from(value: [isize; N]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl FromIterator<isize> for Strides {
    fn from_iter<T: IntoIterator<Item = isize>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
