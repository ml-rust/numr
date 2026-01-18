//! Sparse operations implementation for CUDA runtime
//!
//! This module implements the SparseOps trait for CudaRuntime, providing
//! GPU-accelerated sparse matrix operations using CUDA kernels.
//!
//! ## Module Structure
//!
//! - `spmv` - Sparse matrix-vector and matrix-matrix multiplication
//! - `merge` - Element-wise operations (add/sub/mul/div) for CSR/CSC/COO formats
//! - `conversions` - Format conversions (COO↔CSR, COO↔CSC, CSR↔CSC, transpose)

/// Macro to dispatch sparse merge operations based on dtype
/// Eliminates duplication across CSR/CSC/COO merge wrappers.
macro_rules! sparse_dtype_dispatch {
    ($merge_fn:ident, $dtype:expr, $op_name:expr, ( $($args:expr),* $(,)? )) => {{
        use crate::dtype::DType;
        match $dtype {
            DType::F32 => unsafe { $merge_fn::<f32>($($args),*) },
            DType::F64 => unsafe { $merge_fn::<f64>($($args),*) },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe { $merge_fn::<half::f16>($($args),*) },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe { $merge_fn::<half::bf16>($($args),*) },
            _ => Err(crate::error::Error::Internal(format!(
                "Unsupported dtype for CUDA sparse {}: {:?}",
                $op_name, $dtype
            ))),
        }
    }};
}

// Submodules
mod conversions;
mod cusparse_helpers;
mod esc_spgemm;
mod high_level_ops;
mod merge;
mod spmv;
