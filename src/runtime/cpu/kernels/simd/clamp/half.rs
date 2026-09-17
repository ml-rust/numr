//! f16/bf16 wrappers for clamp operation (block-convert-compute via f32)

use super::dispatch::clamp_f32;

half_clamp!(clamp, clamp_f32);
