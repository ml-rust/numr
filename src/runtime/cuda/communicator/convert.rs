//! Maps numr types and NCCL results to each other.

use cudarc::nccl::{result::NcclError, sys as nccl_sys};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::communicator::ReduceOp;

/// Map numr `DType` to NCCL data type.
pub(super) fn dtype_to_nccl(dtype: DType) -> Result<nccl_sys::ncclDataType_t> {
    match dtype {
        DType::F32 => Ok(nccl_sys::ncclDataType_t::ncclFloat32),
        DType::F64 => Ok(nccl_sys::ncclDataType_t::ncclFloat64),
        DType::F16 => Ok(nccl_sys::ncclDataType_t::ncclFloat16),
        DType::BF16 => Ok(nccl_sys::ncclDataType_t::ncclBfloat16),
        DType::FP8E4M3 => Ok(nccl_sys::ncclDataType_t::ncclFloat8e4m3),
        DType::FP8E5M2 => Ok(nccl_sys::ncclDataType_t::ncclFloat8e5m2),
        DType::I32 => Ok(nccl_sys::ncclDataType_t::ncclInt32),
        DType::I64 => Ok(nccl_sys::ncclDataType_t::ncclInt64),
        DType::I8 => Ok(nccl_sys::ncclDataType_t::ncclInt8),
        DType::U32 => Ok(nccl_sys::ncclDataType_t::ncclUint32),
        DType::U8 => Ok(nccl_sys::ncclDataType_t::ncclUint8),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "nccl_communication",
        }),
    }
}

/// Map numr `ReduceOp` to NCCL reduction operation.
pub(super) fn reduce_op_to_nccl(op: ReduceOp) -> nccl_sys::ncclRedOp_t {
    match op {
        ReduceOp::Sum => nccl_sys::ncclRedOp_t::ncclSum,
        ReduceOp::Prod => nccl_sys::ncclRedOp_t::ncclProd,
        ReduceOp::Min => nccl_sys::ncclRedOp_t::ncclMin,
        ReduceOp::Max => nccl_sys::ncclRedOp_t::ncclMax,
    }
}

/// Describe an NCCL error from `call`, naming the NCCL result code.
pub(super) fn nccl_err(call: &str, e: NcclError) -> String {
    format!("{call} failed with {:?}", e.0)
}

/// Describe an error from making the CUDA context current on this thread.
pub(super) fn context_err(e: impl std::fmt::Debug) -> String {
    format!("failed to make the CUDA context current on this thread: {e:?}")
}

/// Convert a rank count or rank index to the `int` NCCL takes.
pub(super) fn to_nccl_int(arg: &'static str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        arg,
        reason: format!("{value} exceeds the NCCL limit of {}", i32::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_to_nccl_mapping() {
        assert!(dtype_to_nccl(DType::F32).is_ok());
        assert!(dtype_to_nccl(DType::F64).is_ok());
        assert!(dtype_to_nccl(DType::F16).is_ok());
        assert!(dtype_to_nccl(DType::BF16).is_ok());
        assert!(dtype_to_nccl(DType::I32).is_ok());
        assert!(dtype_to_nccl(DType::I64).is_ok());
        assert!(dtype_to_nccl(DType::U32).is_ok());
        assert!(dtype_to_nccl(DType::U8).is_ok());
        assert!(dtype_to_nccl(DType::Bool).is_err());
    }

    #[test]
    fn test_reduce_op_mapping() {
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Sum),
            nccl_sys::ncclRedOp_t::ncclSum
        );
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Prod),
            nccl_sys::ncclRedOp_t::ncclProd
        );
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Min),
            nccl_sys::ncclRedOp_t::ncclMin
        );
        assert_eq!(
            reduce_op_to_nccl(ReduceOp::Max),
            nccl_sys::ncclRedOp_t::ncclMax
        );
    }

    #[test]
    fn test_nccl_err_names_call_and_code() {
        let msg = nccl_err(
            "ncclAllReduce",
            NcclError(nccl_sys::ncclResult_t::ncclRemoteError),
        );
        assert!(msg.contains("ncclAllReduce"), "{msg}");
        assert!(msg.contains("ncclRemoteError"), "{msg}");
    }

    #[test]
    fn test_to_nccl_int_bounds() {
        assert_eq!(to_nccl_int("rank", 3).unwrap(), 3);
        assert!(to_nccl_int("rank", i32::MAX as usize + 1).is_err());
    }
}
