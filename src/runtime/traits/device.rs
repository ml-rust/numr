//! Trait for device identification

/// Trait for device identification
pub trait Device: Clone + Send + Sync + 'static {
    /// Unique identifier for this device
    fn id(&self) -> usize;

    /// Check if two devices are the same
    fn is_same(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Human-readable name
    fn name(&self) -> String {
        format!("Device({})", self.id())
    }
}
