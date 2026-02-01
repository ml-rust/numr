//! Complex number operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ComplexOps;
use crate::runtime::cuda::kernels::{
    launch_angle, launch_angle_real, launch_conj, launch_fill_with_f64, launch_imag, launch_real,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl ComplexOps<CudaRuntime> for CudaClient {
    fn conj(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_conj(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn real(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // Determine output dtype: Complex64 → F32, Complex128 → F64
        let out_dtype = match dtype {
            DType::Complex64 => DType::F32,
            DType::Complex128 => DType::F64,
            _ => return Err(Error::UnsupportedDType { dtype, op: "real" }),
        };

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device);

        unsafe {
            launch_real(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn imag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // Determine output dtype
        let out_dtype = if dtype.is_complex() {
            match dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                _ => return Err(Error::UnsupportedDType { dtype, op: "imag" }),
            }
        } else {
            // For real types, return zeros with same dtype
            dtype
        };

        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device);

        // For real types, fill with zeros
        if !dtype.is_complex() {
            unsafe {
                launch_fill_with_f64(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    out_dtype,
                    0.0,
                    out.storage().ptr(),
                    out.numel(),
                )?;
            }
            return Ok(out);
        }

        // For complex types, extract imaginary part
        let a_contig = ensure_contiguous(a);

        unsafe {
            launch_imag(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn angle(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();

        // Determine output dtype
        let out_dtype = if dtype.is_complex() {
            match dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                _ => return Err(Error::UnsupportedDType { dtype, op: "angle" }),
            }
        } else {
            // For real types, return zeros with same dtype
            dtype
        };

        let out = Tensor::<CudaRuntime>::empty(a.shape(), out_dtype, &self.device);
        let a_contig = ensure_contiguous(a);

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            match dtype {
                DType::F32 | DType::F64 => unsafe {
                    launch_angle_real(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        dtype,
                        a_contig.storage().ptr(),
                        out.storage().ptr(),
                        a.numel(),
                    )?;
                },
                _ => {
                    // For integer types, return zeros (π as integer doesn't make mathematical sense)
                    unsafe {
                        launch_fill_with_f64(
                            &self.context,
                            &self.stream,
                            self.device.index,
                            out_dtype,
                            0.0,
                            out.storage().ptr(),
                            out.numel(),
                        )?;
                    }
                }
            }
            return Ok(out);
        }

        // For complex types, compute phase angle
        unsafe {
            launch_angle(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }
}
