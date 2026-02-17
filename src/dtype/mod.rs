//! Data type system for numr tensors.

pub mod complex;
mod data_type;
mod dtype_enum;
mod dtype_set;
mod element;
pub mod fp8;
mod half_util;
mod precision;
mod promotion;

pub use complex::{Complex64, Complex128};
pub use data_type::DataType;
pub use dtype_enum::DType;
pub use dtype_set::DTypeSet;
pub use element::Element;
pub use fp8::{FP8E4M3, FP8E5M2};
pub use half_util::half_from_f32_util;
pub use precision::ComputePrecision;
pub use promotion::promote;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F64.size_in_bytes(), 8);
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::I8.size_in_bytes(), 1);
        assert_eq!(DType::Bool.size_in_bytes(), 1);
        assert_eq!(DType::FP8E4M3.size_in_bytes(), 1);
        assert_eq!(DType::FP8E5M2.size_in_bytes(), 1);
    }

    #[test]
    fn test_dtype_categories() {
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());
        assert!(DType::I32.is_signed_int());
        assert!(DType::U32.is_unsigned_int());
        assert!(!DType::U32.is_signed());
        assert!(DType::FP8E4M3.is_float());
        assert!(DType::FP8E5M2.is_float());
        assert!(DType::FP8E4M3.is_signed());
        assert!(DType::FP8E5M2.is_signed());
    }

    #[test]
    fn test_dtype_set() {
        assert!(DTypeSet::FLOATS.contains(DType::F32));
        assert!(!DTypeSet::FLOATS.contains(DType::I32));
        assert!(DTypeSet::INTS.contains(DType::I32));
        assert!(DTypeSet::NUMERIC.contains(DType::F32));
        assert!(DTypeSet::NUMERIC.contains(DType::I32));
        assert!(DTypeSet::FLOATS.contains(DType::FP8E4M3));
        assert!(DTypeSet::FLOATS.contains(DType::FP8E5M2));
    }

    #[test]
    fn test_fp8_dtype_values() {
        assert_eq!(DType::FP8E4M3.min_value(), -448.0);
        assert_eq!(DType::FP8E4M3.max_value(), 448.0);
        assert_eq!(DType::FP8E5M2.min_value(), -57344.0);
        assert_eq!(DType::FP8E5M2.max_value(), 57344.0);
    }

    #[test]
    fn test_fp8_short_names() {
        assert_eq!(DType::FP8E4M3.short_name(), "fp8e4m3");
        assert_eq!(DType::FP8E5M2.short_name(), "fp8e5m2");
    }

    #[test]
    fn test_compute_precision_default() {
        assert_eq!(ComputePrecision::default(), ComputePrecision::BF16);
    }
}
