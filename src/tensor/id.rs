//! Tensor ID generation for autograd graph tracking

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for unique tensor IDs
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a tensor
///
/// Used by the autograd system to track tensors in the computation graph.
/// IDs are guaranteed to be unique within a process lifetime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    /// Create a new unique tensor ID
    #[inline]
    pub fn new() -> Self {
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Create from raw value (for testing/serialization only)
    #[inline]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_ids() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        let id3 = TensorId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_id_incrementing() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        assert!(id2.raw() > id1.raw());
    }
}
