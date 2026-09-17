//! f16/bf16 comparison wrappers (block-convert-compute via f32)

use super::dispatch::compare_f32;
use crate::ops::CompareOp;

half_binary_op!(compare, compare_f32, CompareOp);
