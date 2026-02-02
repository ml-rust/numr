//! Advanced PRNG WGSL kernel launchers
//!
//! Split into separate modules per algorithm for maintainability:
//! - `philox`: Philox4x32-10 (JAX/TensorFlow default)
//! - `threefry`: ThreeFry4x32-20 (cryptographic quality)
//! - `pcg64`: PCG64 (NumPy default, emulated 64-bit)
//! - `xoshiro256`: Xoshiro256++ (Rust rand default, emulated 64-bit)
//!
//! # WebGPU Limitations
//!
//! WGSL doesn't support native u64, so all PRNGs use emulated 64-bit.
//! Only F32 dtype is supported.

mod pcg64;
mod philox;
mod threefry;
mod xoshiro256;

pub use pcg64::{launch_pcg64_randn, launch_pcg64_uniform};
pub use philox::{launch_philox_randn, launch_philox_uniform};
pub use threefry::{launch_threefry_randn, launch_threefry_uniform};
pub use xoshiro256::{launch_xoshiro256_randn, launch_xoshiro256_uniform};

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Check that dtype is F32 (only supported type on WebGPU)
fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}
