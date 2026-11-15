//! Interval and expansion arithmetic: argument checks, lane planning, and the composed reference.

#[cfg(any(feature = "wgpu", test))]
pub mod composed;
#[cfg(any(feature = "wgpu", test))]
mod composed_division;
#[cfg(any(feature = "wgpu", test))]
mod composed_expansion;
#[cfg(any(feature = "wgpu", test))]
mod composed_outward;
#[cfg(any(feature = "wgpu", test))]
mod composed_product;
#[cfg(any(feature = "wgpu", test))]
mod composed_tighten;
pub mod envelope;
pub mod expansion_check;
#[cfg(any(feature = "wgpu", test))]
pub mod expansion_impl;
pub mod interval_check;
#[cfg(any(feature = "wgpu", test))]
pub mod interval_impl;
pub mod lanes;
#[cfg(test)]
pub mod parity_expansion;
#[cfg(test)]
pub mod parity_interval;
#[cfg(test)]
pub mod parity_values;
pub mod validate;

pub use expansion_check::{ExpansionBinary, check_expansions, require_basic, require_fused};
#[cfg(feature = "wgpu")]
pub use expansion_impl::{
    expansion_binary_impl, expansion_flag_impl, expansion_from_interval_impl,
    expansion_interval_impl, expansion_normalize_impl, expansion_point_impl,
};
pub use interval_check::{IntervalBinary, binary_shape, check_binary, interval_status_impl};
#[cfg(feature = "wgpu")]
pub use interval_impl::{
    interval_binary_impl, interval_capture_impl, interval_flag_bounds_impl,
    interval_flag_nonfinite_impl, interval_flag_nonzero_impl, interval_sqrt_nonnegative_impl,
};
pub use lanes::{LaneOperand, lane_count, lane_operand};
