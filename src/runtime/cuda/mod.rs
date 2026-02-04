//! CUDA runtime implementation
//!
//! This module provides GPU acceleration via NVIDIA CUDA using cudarc.
//!
//! # Features
//!
//! - `CudaDevice` - Represents a CUDA GPU device
//! - `CudaClient` - Manages GPU stream and context, launches kernels
//! - `CudaRuntime` - Implements the generic Runtime trait
//! - `TensorOps` - CUDA-accelerated tensor operations using cuBLAS
//!
//! # Panics
//!
//! The following operations may panic on CUDA errors (allocation failures are
//! typically unrecoverable in GPU contexts):
//!
//! - `Runtime::allocate` - Panics if CUDA memory allocation fails
//! - `Runtime::copy_to_device` - Panics if host-to-device copy fails
//! - `Runtime::copy_from_device` - Panics if device-to-host copy fails
//! - `Runtime::copy_within_device` - Panics if device-to-device copy fails
//!
//! These panics follow CUDA best practices where allocation failures indicate
//! an unrecoverable out-of-memory condition.

mod cache;
mod client;
mod device;
mod fft;
mod kernels;
mod linalg;
mod ops;
mod polynomial;
mod runtime;
#[cfg(feature = "sparse")]
mod sparse;
mod special;

pub use crate::tensor::Tensor;
pub use client::{CudaAllocator, CudaClient, CudaRawHandle};
pub use device::{CudaDevice, CudaError};
pub use runtime::{CudaRuntime, cuda_device, cuda_device_id};
