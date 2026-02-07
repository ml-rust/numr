//! Types for Algebraic Multigrid (AMG) preconditioner

use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// V-cycle type for AMG
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AmgCycleType {
    /// V-cycle: down-smooth → restrict → coarse-solve → prolongate → up-smooth
    #[default]
    V,
}

/// Configuration options for AMG (Algebraic Multigrid) preconditioner
#[derive(Debug, Clone)]
pub struct AmgOptions {
    /// Maximum number of multigrid levels (default: 25)
    pub max_levels: usize,
    /// Strength-of-connection threshold (default: 0.25)
    ///
    /// Connection i→j is "strong" if |a_ij| >= theta * max_k(|a_ik|)
    pub strength_threshold: f64,
    /// Number of smoother sweeps per level (default: 2)
    pub smoother_sweeps: usize,
    /// Smoother relaxation weight (default: 2/3)
    pub smoother_omega: f64,
    /// Cycle type (default: V-cycle)
    pub cycle_type: AmgCycleType,
    /// Minimum coarse level size before using direct solve (default: 10)
    pub coarse_size: usize,
}

impl Default for AmgOptions {
    fn default() -> Self {
        Self {
            max_levels: 25,
            strength_threshold: 0.25,
            smoother_sweeps: 2,
            smoother_omega: 2.0 / 3.0,
            cycle_type: AmgCycleType::V,
            coarse_size: 10,
        }
    }
}

/// AMG multigrid hierarchy (precomputed during setup)
#[derive(Debug)]
pub struct AmgHierarchy<R: Runtime> {
    /// Operator at each level: A_0 (original), A_1 (first coarse), ...
    pub operators: Vec<CsrData<R>>,
    /// Interpolation (prolongation) operators P_l: coarse → fine
    pub prolongations: Vec<CsrData<R>>,
    /// Restriction operators R_l: fine → coarse (R = P^T typically)
    pub restrictions: Vec<CsrData<R>>,
    /// Inverse diagonal of each operator (for Jacobi smoothing)
    pub diag_inv: Vec<Tensor<R>>,
    /// Options used to build this hierarchy
    pub options: AmgOptions,
    /// Number of levels
    pub num_levels: usize,
}
