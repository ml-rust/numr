//! Common utilities and validation logic shared across operation backends.

pub mod complex_validation;
pub mod fwht_validation;
pub mod group_norm_validation;
pub mod image_nearest;
pub mod image_nearest_plan;
pub mod image_resize_plan;
pub mod image_validation;
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
pub use image_nearest::NearestAxis;
pub use image_nearest_plan::plan_upsample_nearest2d;
pub use image_resize_plan::{
    ResizeMode, ResizePlan, ResizeSchedule, ResizeStep, plan_resize, validate_resize_out,
};
pub use image_validation::{
    ImageMap, ImagePlan, canonical_strides, plan_pixel_shuffle, plan_pixel_unshuffle,
    validate_image_out,
};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub use scatter_axes::collapse_scatter_axes;
pub use scatter_reduce_validation::validate_scatter_extents;
