//! Kernel names of the dtype-instantiated sparse CUDA kernels.
//!
//! Each `.cu` file instantiates a kernel once per value dtype and names it
//! `{base}_{suffix}`. The suffix comes from the numr `DType`, never from
//! cudarc's C type name (`float`, `double`, ...).

use crate::dtype::DType;
use crate::error::{Error, Result};

/// `{base}_{suffix}` for a kernel instantiated for F32, F64, F16 and BF16.
pub fn float_kernel_name(base: &str, dtype: DType, op: &'static str) -> Result<String> {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };
    Ok(format!("{base}_{suffix}"))
}

/// `{base}_{suffix}` for a merge compute kernel: the four float dtypes and
/// the signed integers I8, I16, I32 and I64.
pub fn merge_kernel_name(base: &str, dtype: DType, op: &'static str) -> Result<String> {
    let suffix = match dtype {
        DType::I8 => "i8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::I64 => "i64",
        _ => return float_kernel_name(base, dtype, op),
    };
    Ok(format!("{base}_{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_numr_dtype() {
        assert_eq!(
            float_kernel_name("csr_spmv", DType::F32, "t").unwrap(),
            "csr_spmv_f32"
        );
        assert_eq!(
            float_kernel_name("csr_spmv", DType::BF16, "t").unwrap(),
            "csr_spmv_bf16"
        );
        assert!(float_kernel_name("k", DType::I32, "t").is_err());
        assert_eq!(
            merge_kernel_name("csr_sub_compute", DType::I32, "t").unwrap(),
            "csr_sub_compute_i32"
        );
        assert_eq!(
            merge_kernel_name("csc_add_compute", DType::F16, "t").unwrap(),
            "csc_add_compute_f16"
        );
        assert!(merge_kernel_name("k", DType::U32, "t").is_err());
    }
}
