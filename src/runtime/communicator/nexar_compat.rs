//! Shared conversion helpers between numr and nexar types.

use super::ReduceOp;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Maps a numr `DType` to a nexar `DataType`.
///
/// Returns `Err` for types nexar doesn't support (Complex, Bool, FP8, I16, U16).
pub fn to_nexar_dtype(dtype: DType) -> Result<nexar::DataType> {
    match dtype {
        DType::F32 => Ok(nexar::DataType::F32),
        DType::F64 => Ok(nexar::DataType::F64),
        DType::F16 => Ok(nexar::DataType::F16),
        DType::BF16 => Ok(nexar::DataType::BF16),
        DType::I8 => Ok(nexar::DataType::I8),
        DType::I32 => Ok(nexar::DataType::I32),
        DType::I64 => Ok(nexar::DataType::I64),
        DType::U8 => Ok(nexar::DataType::U8),
        DType::U32 => Ok(nexar::DataType::U32),
        DType::U64 => Ok(nexar::DataType::U64),
        _ => Err(Error::Backend(format!(
            "nexar: unsupported dtype {dtype:?} for collective operation"
        ))),
    }
}

/// Maps a numr `ReduceOp` to a nexar `ReduceOp`.
pub fn to_nexar_op(op: ReduceOp) -> nexar::ReduceOp {
    match op {
        ReduceOp::Sum => nexar::ReduceOp::Sum,
        ReduceOp::Prod => nexar::ReduceOp::Prod,
        ReduceOp::Min => nexar::ReduceOp::Min,
        ReduceOp::Max => nexar::ReduceOp::Max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(to_nexar_dtype(DType::F32).unwrap(), nexar::DataType::F32);
        assert_eq!(to_nexar_dtype(DType::F64).unwrap(), nexar::DataType::F64);
        assert_eq!(to_nexar_dtype(DType::F16).unwrap(), nexar::DataType::F16);
        assert_eq!(to_nexar_dtype(DType::BF16).unwrap(), nexar::DataType::BF16);
        assert_eq!(to_nexar_dtype(DType::I8).unwrap(), nexar::DataType::I8);
        assert_eq!(to_nexar_dtype(DType::I32).unwrap(), nexar::DataType::I32);
        assert_eq!(to_nexar_dtype(DType::I64).unwrap(), nexar::DataType::I64);
        assert_eq!(to_nexar_dtype(DType::U8).unwrap(), nexar::DataType::U8);
        assert_eq!(to_nexar_dtype(DType::U32).unwrap(), nexar::DataType::U32);
        assert_eq!(to_nexar_dtype(DType::U64).unwrap(), nexar::DataType::U64);
    }

    #[test]
    fn test_dtype_mapping_unsupported() {
        assert!(to_nexar_dtype(DType::Bool).is_err());
        assert!(to_nexar_dtype(DType::Complex64).is_err());
        assert!(to_nexar_dtype(DType::Complex128).is_err());
    }

    #[test]
    fn test_reduce_op_mapping() {
        assert_eq!(to_nexar_op(ReduceOp::Sum), nexar::ReduceOp::Sum);
        assert_eq!(to_nexar_op(ReduceOp::Prod), nexar::ReduceOp::Prod);
        assert_eq!(to_nexar_op(ReduceOp::Min), nexar::ReduceOp::Min);
        assert_eq!(to_nexar_op(ReduceOp::Max), nexar::ReduceOp::Max);
    }
}
