//! Sparse tensor support for numr
//!
//! This module provides sparse tensor formats and operations for efficient
//! storage and computation on tensors with mostly zero elements.
//!
//! # Sparse Formats
//!
//! Three standard sparse formats are supported:
//!
//! - **COO** (Coordinate): Stores (row, col, value) triplets. Best for construction
//!   and format conversion. O(nnz) storage.
//!
//! - **CSR** (Compressed Sparse Row): Row-major compressed format. Best for
//!   row slicing and matrix-vector multiplication (SpMV). O(nnz + nrows) storage.
//!
//! - **CSC** (Compressed Sparse Column): Column-major compressed format. Best for
//!   column slicing and some linear algebra operations. O(nnz + ncols) storage.
//!
//! # Usage
//!
//! ```ignore
//! use numr::sparse::{SparseTensor, SparseFormat};
//! use numr::runtime::CpuRuntime;
//!
//! // Create COO tensor from triplets
//! let rows = vec![0, 1, 2];
//! let cols = vec![1, 0, 2];
//! let values = vec![1.0f32, 2.0, 3.0];
//! let shape = [3, 3];
//!
//! let coo = SparseTensor::<CpuRuntime>::from_coo(
//!     &rows, &cols, &values, &shape, &device
//! )?;
//!
//! // Convert to CSR for efficient SpMV
//! let csr = coo.to_csr()?;
//!
//! // Sparse matrix-vector multiplication
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
//! let y = sparse_ops.spmv(&csr, &x)?;  // y = A * x
//! ```
//!
//! # When to Use Sparse
//!
//! Sparse formats are beneficial when:
//! - Sparsity > 90% (less than 10% non-zeros)
//! - Memory is constrained
//! - Operations preserve sparsity (SpMV, element-wise on non-zeros)
//!
//! Dense is usually faster when:
//! - Sparsity < 50%
//! - Operations produce dense results
//! - Using GPU with high memory bandwidth

mod coo;
mod csc;
mod csr;
mod format;
mod ops;
mod tensor;

pub use coo::CooData;
pub use csc::CscData;
pub use csr::CsrData;
pub use format::{SparseFormat, SparseStorage};
pub use ops::SparseOps;
pub use tensor::SparseTensor;
