//! CPU device implementation

use crate::runtime::Device;

/// CPU device (there's only one: the host CPU)
#[derive(Clone, Debug, Default)]
pub struct CpuDevice {
    id: usize,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new() -> Self {
        Self { id: 0 }
    }
}

impl Device for CpuDevice {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> String {
        "cpu".to_string()
    }
}
