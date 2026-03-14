//! Trait definitions for sparse QR factorization
//!
//! Sparse QR uses free functions per backend (sparse_qr_cpu, sparse_qr_cuda, etc.)
//! rather than a trait-based dispatch pattern, because the CPU implementation
//! operates on extracted f64 data while GPU backends will need native kernels.
