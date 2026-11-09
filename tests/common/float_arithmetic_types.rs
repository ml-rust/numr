//! Complete metadata-only dtype matrix without device allocation or computation.

use numr::dtype::DType;

pub const DTYPES: [DType; 17] = [
    DType::F64,
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::FP8E4M3,
    DType::FP8E5M2,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool,
    DType::Complex64,
    DType::Complex128,
];
