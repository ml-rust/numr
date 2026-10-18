//! Element-wise reduction of rank buffers in fixed rank order.
//!
//! Each output element is `b0 op b1 op ... op b(W-1)`, folded left to right from rank 0.
//! The order never changes, so the result is bit-identical on every rank and on every run.

use crate::dtype::DType;
use crate::runtime::communicator::ReduceOp;

/// One element as the reduction sees it.
trait Lane: Copy {
    /// Bytes of one stored element.
    const BYTES: usize;
    /// Decode one element from exactly `BYTES` native-endian bytes.
    fn load(chunk: &[u8]) -> Self;
    /// Append the stored form of this element to `out`.
    fn store(self, out: &mut Vec<u8>);
    /// Combine the running value with the next rank's value.
    fn combine(self, next: Self, op: ReduceOp) -> Self;
}

/// Integers wrap on overflow, as two's-complement hardware reductions do.
macro_rules! int_lane {
    ($($t:ty),*) => {$(
        impl Lane for $t {
            const BYTES: usize = std::mem::size_of::<$t>();
            fn load(chunk: &[u8]) -> Self {
                <$t>::from_ne_bytes(chunk.try_into().unwrap_or_default())
            }
            fn store(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_ne_bytes());
            }
            fn combine(self, next: Self, op: ReduceOp) -> Self {
                match op {
                    ReduceOp::Sum => self.wrapping_add(next),
                    ReduceOp::Prod => self.wrapping_mul(next),
                    ReduceOp::Max => Ord::max(self, next),
                    ReduceOp::Min => Ord::min(self, next),
                }
            }
        }
    )*};
}

/// Floats use IEEE `max` and `min`: a NaN loses to any number.
macro_rules! float_lane {
    ($($t:ty),*) => {$(
        impl Lane for $t {
            const BYTES: usize = std::mem::size_of::<$t>();
            fn load(chunk: &[u8]) -> Self {
                <$t>::from_ne_bytes(chunk.try_into().unwrap_or_default())
            }
            fn store(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_ne_bytes());
            }
            fn combine(self, next: Self, op: ReduceOp) -> Self {
                match op {
                    ReduceOp::Sum => self + next,
                    ReduceOp::Prod => self * next,
                    ReduceOp::Max => self.max(next),
                    ReduceOp::Min => self.min(next),
                }
            }
        }
    )*};
}

int_lane!(i8, i16, i32, i64, u8, u16, u32, u64);
float_lane!(f32, f64);

/// Half-precision elements widen to an `f32` accumulator and round once on store.
#[cfg(feature = "f16")]
macro_rules! half_lane {
    ($name:ident, $t:ty) => {
        #[derive(Clone, Copy)]
        struct $name(f32);

        impl Lane for $name {
            const BYTES: usize = 2;
            fn load(chunk: &[u8]) -> Self {
                Self(<$t>::from_ne_bytes(chunk.try_into().unwrap_or_default()).to_f32())
            }
            fn store(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&<$t>::from_f32(self.0).to_ne_bytes());
            }
            fn combine(self, next: Self, op: ReduceOp) -> Self {
                Self(self.0.combine(next.0, op))
            }
        }
    };
}

#[cfg(feature = "f16")]
half_lane!(Half, half::f16);
#[cfg(feature = "f16")]
half_lane!(Brain, half::bf16);

/// Fold `inputs[0] op inputs[1] op ...` element by element, in slice order.
fn fold<L: Lane>(inputs: &[Vec<u8>], op: ReduceOp) -> Vec<u8> {
    let Some((first, rest)) = inputs.split_first() else {
        return Vec::new();
    };
    let mut acc: Vec<L> = first.chunks_exact(L::BYTES).map(L::load).collect();
    for buf in rest {
        for (value, chunk) in acc.iter_mut().zip(buf.chunks_exact(L::BYTES)) {
            *value = value.combine(L::load(chunk), op);
        }
    }
    let mut out = Vec::with_capacity(first.len());
    for value in acc {
        value.store(&mut out);
    }
    out
}

/// Reduce equal-length rank buffers in rank order.
///
/// Returns `None` when `dtype` has no reduction here.
/// F16 and BF16 need the `f16` feature.
pub(super) fn reduce(inputs: &[Vec<u8>], dtype: DType, op: ReduceOp) -> Option<Vec<u8>> {
    let out = match dtype {
        DType::F64 => fold::<f64>(inputs, op),
        DType::F32 => fold::<f32>(inputs, op),
        #[cfg(feature = "f16")]
        DType::F16 => fold::<Half>(inputs, op),
        #[cfg(feature = "f16")]
        DType::BF16 => fold::<Brain>(inputs, op),
        DType::I64 => fold::<i64>(inputs, op),
        DType::I32 => fold::<i32>(inputs, op),
        DType::I16 => fold::<i16>(inputs, op),
        DType::I8 => fold::<i8>(inputs, op),
        DType::U64 => fold::<u64>(inputs, op),
        DType::U32 => fold::<u32>(inputs, op),
        DType::U16 => fold::<u16>(inputs, op),
        DType::U8 => fold::<u8>(inputs, op),
        _ => return None,
    };
    Some(out)
}

/// True when [`reduce`] accepts `dtype`.
pub(super) fn supports(dtype: DType) -> bool {
    reduce(&[], dtype, ReduceOp::Sum).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes<T: bytemuck::Pod>(values: &[T]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    #[test]
    fn f32_sum_follows_rank_order() {
        let inputs = [bytes(&[1e8f32]), bytes(&[1.0f32]), bytes(&[-1e8f32])];
        let out = reduce(&inputs, DType::F32, ReduceOp::Sum).unwrap();
        // (1e8 + 1) rounds to 1e8, so rank order gives 0. Any other order gives 1.
        assert_eq!(out, bytes(&[0.0f32]));
    }

    #[test]
    fn integers_wrap_instead_of_overflowing() {
        let inputs = [bytes(&[i32::MAX]), bytes(&[1i32])];
        let out = reduce(&inputs, DType::I32, ReduceOp::Sum).unwrap();
        assert_eq!(out, bytes(&[i32::MIN]));
        let inputs = [bytes(&[200u8]), bytes(&[2u8])];
        assert_eq!(
            reduce(&inputs, DType::U8, ReduceOp::Prod).unwrap(),
            vec![144u8]
        );
    }

    #[test]
    fn unsupported_dtypes_are_rejected() {
        assert!(!supports(DType::Bool));
        assert!(!supports(DType::Complex64));
        assert!(supports(DType::U64));
        assert_eq!(supports(DType::F16), cfg!(feature = "f16"));
    }

    #[cfg(feature = "f16")]
    #[test]
    fn half_widens_and_rounds_once() {
        use half::{bf16, f16};
        let h = |v: f32| f16::from_f32(v).to_ne_bytes().to_vec();
        // f16 spacing at 2048 is 2. Pairwise rounding loses both ones. Rounding once keeps them.
        let inputs = [h(2048.0), h(1.0), h(1.0)];
        assert_eq!(
            reduce(&inputs, DType::F16, ReduceOp::Sum).unwrap(),
            h(2050.0)
        );

        let b = |v: f32| bf16::from_f32(v).to_ne_bytes().to_vec();
        // bf16 spacing at 256 is 2.
        let inputs = [b(256.0), b(1.0), b(1.0)];
        assert_eq!(
            reduce(&inputs, DType::BF16, ReduceOp::Sum).unwrap(),
            b(258.0)
        );
    }
}
