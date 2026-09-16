//! Sparse matrix merge algorithms
//!
//! This module implements parameterized merge patterns for sparse matrix
//! element-wise operations, eliminating code duplication across CSR, CSC, and COO formats.
//!
//! # Design Pattern
//!
//! Instead of having separate implementations for add, subtract, multiply, and
//! divide, we have:
//! - ONE generic merge function per format (CSR, CSC, COO)
//! - Strategy enum to control which positions to keep (union vs intersection)
//! - Semantics enum to control empty matrix handling
//! - Operation closures to define element-wise computation
//!
//! # Benefits
//!
//! 1. **Code Reuse**: Single merge implementation handles all operations
//!    - Before: ~900 lines (6 specialized functions)
//!    - After: ~300 lines (2 parameterized functions)
//!
//! 2. **Consistency**: All operations use identical merge logic, reducing bugs
//!
//! 3. **Extensibility**: New operations (e.g., max, min) require only:
//!    - Adding semantics to enum
//!    - Writing operation closure
//!    - No new merge code needed
//!
//! 4. **Testability**: Testing merge logic once covers all operations
//!
//! # Architecture
//!
//! ```text
//! User-facing trait method (add_csr, sub_csr, mul_csr, div_csr)
//!       │
//!       ├─> Calls merge_csr_impl with:
//!       │   - MergeStrategy (Union or Intersection)
//!       │   - OperationSemantics (Add, Subtract, Multiply, Divide)
//!       │   - Operation closure: |a, b| a + b
//!       │   - Transform closures: |a| a, |b| -b
//!       │
//!       └─> merge_csr_impl:
//!           1. Check empty matrices (using semantics)
//!           2. Merge based on strategy (union/intersection)
//!           3. Apply operations via closures
//!           4. Filter near-zeros
//! ```
//!
//! # Example Usage
//!
//! ```ignore
//! // Addition: union merge, identity transforms
//! fn add_csr<T>(...) -> Result<...> {
//!     merge_csr_impl(
//!         ...,
//!         MergeStrategy::Union,
//!         OperationSemantics::Add,
//!         |a, b| a + b,  // Both exist
//!         |a| a,         // A-only: keep
//!         |b| b,         // B-only: keep
//!     )
//! }
//!
//! // Subtraction: union merge, negate B
//! fn sub_csr<T>(...) -> Result<...> {
//!     merge_csr_impl(
//!         ...,
//!         MergeStrategy::Union,
//!         OperationSemantics::Subtract,
//!         |a, b| a - b,  // Both exist
//!         |a| a,         // A-only: keep
//!         |b| -b,        // B-only: negate
//!     )
//! }
//!
//! // Multiplication: intersection merge (0 * x = 0)
//! fn mul_csr<T>(...) -> Result<...> {
//!     merge_csr_impl(
//!         ...,
//!         MergeStrategy::Intersection,
//!         OperationSemantics::Multiply,
//!         |a, b| a * b,  // Both exist
//!         |a| a,         // Unused (intersection only)
//!         |b| b,         // Unused (intersection only)
//!     )
//! }
//! ```
//!
//! # Empty Matrix Handling
//!
//! OperationSemantics controls what happens when one or both matrices are empty:
//!
//! | Operation | A empty, B not | A not, B empty | Both empty |
//! |-----------|----------------|----------------|------------|
//! | Add       | Return B       | Return A       | Empty      |
//! | Subtract  | Return -B      | Return A       | Empty      |
//! | Multiply  | Empty (0*x=0)  | Empty (x*0=0)  | Empty      |
//! | Divide    | Empty (0/x=0)  | Error (x/0)    | Empty      |
//!
//! This logic is centralized in `common::handle_empty_compressed` and avoids
//! duplicating empty-check code across operations.

mod common;
mod coo;
mod csc;
mod csr;

// Re-export zero_tolerance from shared utilities module
// See runtime::sparse_utils::zero_tolerance for full documentation
pub(crate) use crate::runtime::common::sparse_utils::zero_tolerance;

pub(crate) use common::{MergeStrategy, OperationSemantics};
pub(crate) use coo::{intersect_coo_impl, merge_coo_impl};
pub(crate) use csc::merge_csc_impl;
pub(crate) use csr::merge_csr_impl;
