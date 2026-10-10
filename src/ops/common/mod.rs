//! Common utilities and validation logic shared across operation backends.

pub mod complex_validation;
pub mod fwht_validation;
pub mod group_norm_validation;
pub mod quasirandom;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub mod scatter_axes;
pub mod scatter_reduce_validation;

pub use complex_validation::{validate_complex_real_inputs, validate_make_complex_inputs};
#[cfg(feature = "wgpu")]
pub use complex_validation::{
    validate_complex_real_inputs_f32_only, validate_make_complex_inputs_f32_only,
};
pub use fwht_validation::validate_fwht_args;
pub use group_norm_validation::group_norm_channels_per_group;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub use scatter_axes::collapse_scatter_axes;
pub use scatter_reduce_validation::validate_scatter_extents;
