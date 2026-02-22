//! Communicator groups for multi-dimensional parallelism.
//!
//! Splits a world communicator into sub-communicators for Tensor Parallelism
//! (TP), Pipeline Parallelism (PP), Data Parallelism (DP), and Expert
//! Parallelism (EP). Uses the `Communicator::split()` method to create
//! sub-groups.

use std::collections::HashMap;
use std::sync::Arc;

use super::Communicator;
use crate::error::{Error, Result};

/// Dimension of parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParallelDim {
    /// Data parallelism: replicate model, shard data.
    Data,
    /// Tensor parallelism: shard weight matrices within a layer.
    Tensor,
    /// Pipeline parallelism: shard layers across stages.
    Pipeline,
    /// Expert parallelism: distribute MoE experts across devices.
    Expert,
}

/// A group of sub-communicators for multi-dimensional parallelism.
///
/// Created by splitting a world communicator along TP, PP, and DP dimensions.
/// The layout is `[DP, PP, TP]` (TP innermost), meaning consecutive ranks
/// form a TP group.
///
/// # Example
///
/// ```ignore
/// // 8 GPUs: TP=2, PP=2, DP=2
/// let group = CommunicatorGroup::new(world_comm, 2, 2, 2)?;
/// let tp_comm = group.tp(); // 2 ranks per group
/// let pp_comm = group.pp(); // 2 ranks per group
/// let dp_comm = group.dp(); // 2 ranks per group
/// ```
pub struct CommunicatorGroup {
    world: Arc<dyn Communicator>,
    dims: HashMap<ParallelDim, Arc<dyn Communicator>>,
}

impl CommunicatorGroup {
    /// Create communicator groups from a world communicator.
    ///
    /// Layout: `[DP, PP, TP]` (TP innermost).
    /// - Ranks `[0..tp_size)` form the first TP group
    /// - Ranks with the same `rank % tp_size` and same PP stage form a DP group
    /// - etc.
    ///
    /// Requires `tp_size * pp_size * dp_size == world_size`.
    pub fn new(
        world: Arc<dyn Communicator>,
        tp_size: usize,
        pp_size: usize,
        dp_size: usize,
    ) -> Result<Self> {
        let ws = world.world_size();
        if tp_size * pp_size * dp_size != ws {
            return Err(Error::Backend(format!(
                "CommunicatorGroup: tp({tp_size}) * pp({pp_size}) * dp({dp_size}) = {} != world_size({ws})",
                tp_size * pp_size * dp_size,
            )));
        }

        let rank = world.rank();
        let mut dims = HashMap::new();

        // Layout: [DP, PP, TP] — TP innermost
        // rank = dp_idx * (pp_size * tp_size) + pp_idx * tp_size + tp_idx
        let tp_idx = rank % tp_size;
        let pp_idx = (rank / tp_size) % pp_size;
        let dp_idx = rank / (tp_size * pp_size);

        // TP group: same dp_idx, same pp_idx → color = dp_idx * pp_size + pp_idx
        if tp_size > 1 {
            let tp_color = (dp_idx * pp_size + pp_idx) as u32;
            if let Some(comm) = world.split(tp_color, tp_idx as u32)? {
                dims.insert(ParallelDim::Tensor, Arc::from(comm));
            }
        }

        // PP group: same dp_idx, same tp_idx → color = dp_idx * tp_size + tp_idx
        // Use offset to avoid color collision with TP
        if pp_size > 1 {
            let color_offset = dp_size * pp_size;
            let pp_color = (color_offset + dp_idx * tp_size + tp_idx) as u32;
            if let Some(comm) = world.split(pp_color, pp_idx as u32)? {
                dims.insert(ParallelDim::Pipeline, Arc::from(comm));
            }
        }

        // DP group: same pp_idx, same tp_idx → color = pp_idx * tp_size + tp_idx
        // Use offset to avoid collision with TP and PP
        if dp_size > 1 {
            let color_offset = dp_size * pp_size + dp_size * tp_size;
            let dp_color = (color_offset + pp_idx * tp_size + tp_idx) as u32;
            if let Some(comm) = world.split(dp_color, dp_idx as u32)? {
                dims.insert(ParallelDim::Data, Arc::from(comm));
            }
        }

        Ok(Self { world, dims })
    }

    /// The world communicator (all ranks).
    pub fn world(&self) -> &Arc<dyn Communicator> {
        &self.world
    }

    /// Tensor parallelism communicator. `None` if `tp_size == 1`.
    pub fn tp(&self) -> Option<&Arc<dyn Communicator>> {
        self.dims.get(&ParallelDim::Tensor)
    }

    /// Pipeline parallelism communicator. `None` if `pp_size == 1`.
    pub fn pp(&self) -> Option<&Arc<dyn Communicator>> {
        self.dims.get(&ParallelDim::Pipeline)
    }

    /// Data parallelism communicator. `None` if `dp_size == 1`.
    pub fn dp(&self) -> Option<&Arc<dyn Communicator>> {
        self.dims.get(&ParallelDim::Data)
    }

    /// Get communicator for an arbitrary parallelism dimension.
    pub fn get(&self, dim: ParallelDim) -> Option<&Arc<dyn Communicator>> {
        self.dims.get(&dim)
    }

    /// Add an expert parallelism communicator after construction.
    ///
    /// EP is orthogonal to the TP/PP/DP layout and may use a custom split.
    pub fn set_expert(&mut self, comm: Arc<dyn Communicator>) {
        self.dims.insert(ParallelDim::Expert, comm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::communicator::NoOpCommunicator;

    #[test]
    fn test_parallel_dim_eq() {
        assert_eq!(ParallelDim::Data, ParallelDim::Data);
        assert_ne!(ParallelDim::Data, ParallelDim::Tensor);
    }

    #[test]
    fn test_single_rank_group() {
        let world = Arc::new(NoOpCommunicator) as Arc<dyn Communicator>;
        let group = CommunicatorGroup::new(world, 1, 1, 1).unwrap();
        assert_eq!(group.world().world_size(), 1);
        // All dims are size 1, so no sub-communicators created
        assert!(group.tp().is_none());
        assert!(group.pp().is_none());
        assert!(group.dp().is_none());
    }

    #[test]
    fn test_invalid_dimensions() {
        let world = Arc::new(NoOpCommunicator) as Arc<dyn Communicator>;
        // 2*2*2=8 != 1
        let result = CommunicatorGroup::new(world, 2, 2, 2);
        assert!(result.is_err());
    }
}
