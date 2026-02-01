//! CPU implementation of complex number operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ComplexOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

/// ComplexOps implementation for CPU runtime.
impl ComplexOps<CpuRuntime> for CpuClient {
    fn conj(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        unsafe {
            match dtype {
                DType::Complex64 => {
                    kernels::conj_complex64(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                DType::Complex128 => {
                    kernels::conj_complex128(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                _ => unreachable!("conj called on non-complex dtype"),
            }
        }

        Ok(out)
    }

    fn real(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);

        // Determine output dtype (F32 for Complex64, F64 for Complex128)
        let out_dtype = dtype
            .complex_component_dtype()
            .ok_or_else(|| Error::Internal("Expected complex dtype".to_string()))?;

        let out = Tensor::<CpuRuntime>::empty(shape, out_dtype, &self.device);

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        unsafe {
            match dtype {
                DType::Complex64 => {
                    kernels::real_complex64(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                DType::Complex128 => {
                    kernels::real_complex128(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                _ => unreachable!("real called on non-complex dtype"),
            }
        }

        Ok(out)
    }

    fn imag(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let numel = a.numel();

        // For real types, return zeros
        if !dtype.is_complex() {
            return Ok(Tensor::<CpuRuntime>::zeros(shape, dtype, &self.device));
        }

        let a_contig = ensure_contiguous(a);

        // Determine output dtype (F32 for Complex64, F64 for Complex128)
        let out_dtype = dtype
            .complex_component_dtype()
            .ok_or_else(|| Error::Internal("Expected complex dtype".to_string()))?;

        let out = Tensor::<CpuRuntime>::empty(shape, out_dtype, &self.device);

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        unsafe {
            match dtype {
                DType::Complex64 => {
                    kernels::imag_complex64(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                DType::Complex128 => {
                    kernels::imag_complex128(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                _ => unreachable!("imag called on non-complex dtype"),
            }
        }

        Ok(out)
    }

    fn angle(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let numel = a.numel();

        let a_contig = ensure_contiguous(a);

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);

            // Handle empty tensors
            if numel == 0 {
                return Ok(out);
            }

            let a_ptr = a_contig.storage().ptr();
            let out_ptr = out.storage().ptr();

            unsafe {
                match dtype {
                    DType::F32 => {
                        kernels::angle_real_f32(a_ptr as *const _, out_ptr as *mut _, numel);
                    }
                    DType::F64 => {
                        kernels::angle_real_f64(a_ptr as *const _, out_ptr as *mut _, numel);
                    }
                    _ => {
                        // For integer types, angle doesn't make mathematical sense
                        // Return zeros
                        return Ok(Tensor::<CpuRuntime>::zeros(shape, dtype, &self.device));
                    }
                }
            }
            return Ok(out);
        }

        // Determine output dtype (F32 for Complex64, F64 for Complex128)
        let out_dtype = dtype
            .complex_component_dtype()
            .ok_or_else(|| Error::Internal("Expected complex dtype".to_string()))?;

        let out = Tensor::<CpuRuntime>::empty(shape, out_dtype, &self.device);

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        unsafe {
            match dtype {
                DType::Complex64 => {
                    kernels::angle_complex64(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                DType::Complex128 => {
                    kernels::angle_complex128(a_ptr as *const _, out_ptr as *mut _, numel);
                }
                _ => unreachable!("angle called on non-complex dtype"),
            }
        }

        Ok(out)
    }
}
