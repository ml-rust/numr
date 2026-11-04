//! Common utilities and validation logic shared across operation backends.

pub mod color_calls;
pub mod color_packed;
pub mod color_packed_calls;
pub mod color_plan;
pub mod color_validation;
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

pub use color_calls::{
    plan_decode, plan_decode_into, plan_encode, plan_pack, plan_pack_into, plan_unpack,
    plan_unpack_into,
};
pub use color_packed::packed_rgb24_strides;
pub use color_packed_calls::{
    PackedLumaPad, packed_rgb24_shape, plan_pack_packed_into, plan_pad_packed_luma,
    plan_unpack_packed, plan_unpack_packed_into,
};
pub use color_plan::{ColorCoeffs, PlaneView, YuvFrame, rgb_strides, rgb24_strides};
pub use color_validation::rgb_shape;
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
