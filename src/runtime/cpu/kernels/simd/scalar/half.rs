//! f16/bf16 wrappers for tensor-scalar operations, generated via the
//! crate's half-precision macros (convert to f32, run the f32 kernel,
//! convert back).

#[cfg(feature = "f16")]
use super::dispatch::{rsub_scalar_f32, scalar_f32};
#[cfg(feature = "f16")]
use crate::ops::BinaryOp;

half_scalar_op!(scalar, scalar_f32, BinaryOp);
half_unary_scalar!(rsub_scalar, rsub_scalar_f32);
