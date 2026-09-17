//! Advanced PRNG kernels for CPU
//!
//! Implements counter-based and state-of-the-art PRNGs for reproducible parallel generation.
//!
//! # Algorithms
//!
//! - `philox`: Philox4x32-10 (Salmon et al. 2011)
//! - `threefry`: ThreeFry4x64-20 (Salmon et al. 2011)
//! - `pcg64`: PCG64 (O'Neill 2014)
//! - `xoshiro256`: Xoshiro256++ (Blackman & Vigna 2018)

mod common;
mod pcg64;
mod philox;
mod threefry;
mod xoshiro256;

pub use pcg64::{pcg64_randn_kernel, pcg64_uniform_kernel};
pub use philox::{philox_randn_kernel, philox_uniform_kernel};
pub use threefry::{threefry_randn_kernel, threefry_uniform_kernel};
pub use xoshiro256::{xoshiro256_randn_kernel, xoshiro256_uniform_kernel};

pub(crate) use common::box_muller;
