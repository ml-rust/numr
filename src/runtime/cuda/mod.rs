//! CUDA runtime implementation
//!
//! This module provides GPU acceleration via NVIDIA CUDA using cudarc.
//!
//! # Features
//!
//! - `CudaDevice` - Represents a CUDA GPU device
//! - `CudaClient` - Manages GPU stream and context, launches kernels
//! - `CudaRuntime` - Implements the generic Runtime trait
//! - `TensorOps` - CUDA tensor operations on numr's own kernels
//!
//! # Threads
//!
//! A `CudaClient` works from any thread. numr makes the client's CUDA
//! context current on the calling thread before each direct driver call, so
//! a caller never binds the context itself.
//!
//! # Errors
//!
//! Allocation and the host/device copies return driver errors. Two calls
//! panic instead:
//!
//! - `Runtime::default_client` and the first allocation on a device, when
//!   the device's client cannot be created.
//! - `Tensor::to_vec`, on a failed copy. `Tensor::try_to_vec` returns the
//!   error.

mod allocator;
mod arena;
mod cache;
pub mod capture;
mod client;
#[cfg(feature = "nccl")]
mod communicator;
mod context;
mod device;
mod env_config;
mod fft;
mod fft_bluestein;
mod graph;
pub mod kernels;
mod linalg;
mod ops;
mod polynomial;
mod runtime;
mod sobol_cache;
#[cfg(feature = "sparse")]
mod sparse;
mod special;
pub mod tune;

pub use crate::tensor::Tensor;
pub use allocator::CudaAllocator;
pub use capture::{GuardedLaunchBuilder, GuardedStream};
pub use client::{CudaClient, CudaRawHandle};
#[cfg(feature = "nccl")]
pub use communicator::NcclCommunicator;
pub use device::{CudaDevice, CudaError};
pub use graph::CudaGraph;
pub use runtime::{CudaRuntime, cuda_device, cuda_device_id, is_cuda_available};
pub use tune::{time_launches, tuned};
