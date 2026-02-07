//! Types for iterative solvers

mod amg;
mod common;
mod eigen;
mod solvers;
mod svd;

pub use amg::{AmgCycleType, AmgHierarchy, AmgOptions};
pub use common::{
    AdaptivePreconditionerOptions, ConvergenceReason, GmresDiagnostics, PreconditionerType,
    StagnationParams,
};
pub use eigen::{SparseEigComplexResult, SparseEigOptions, SparseEigResult, WhichEigenvalues};
pub use solvers::{
    AdaptiveGmresResult, BiCgStabOptions, BiCgStabResult, CgOptions, CgResult, CgsOptions,
    CgsResult, GmresOptions, GmresResult, JacobiOptions, JacobiResult, LgmresOptions, LgmresResult,
    MinresOptions, MinresResult, QmrOptions, QmrResult, SorOptions, SorResult,
};
pub use svd::{SparseSvdResult, SvdsOptions, WhichSingularValues};
