//! Common utilities and validation logic shared across operation backends.

pub mod complex_validation;
pub mod quasirandom;

pub use complex_validation::{
    validate_complex_real_inputs, validate_complex_real_inputs_f32_only,
    validate_make_complex_inputs, validate_make_complex_inputs_f32_only,
};
